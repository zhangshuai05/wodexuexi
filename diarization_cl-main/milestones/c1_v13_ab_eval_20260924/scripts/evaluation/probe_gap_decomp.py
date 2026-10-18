# -*- coding: utf-8 -*-
"""v13 归因分解：v12 真路径那 67 个点，分别长在窗集/嵌入/选K 哪一段。

背景：`der_v12_dev.py` 的 `winOR` 腿（RTTM 真值门 + 参考标签 + 2.0s 边界）
只给 4.51%，而 v12 真路径给 71.53%。两者用的窗集不同（缓存 562 vs 真实 422），
所以差距不能直接归因。本脚本把**窗集固定成 v12 的 VAD 门控窗**，只换身份来源：

    0  RTTM 门 + 参考标签   —— 复现 der_v12_dev.winOR（对尺子）
    A  VAD 门 + 参考标签    —— 完美身份下界：换门损失多少
    B  VAD 门 + AHC@真值K   —— 嵌入/聚类损失多少
    C  v12.diarize()        —— 选 K 与边界再损失多少

主指标是**段落级说话人正确率**（按参考段时长加权，段 >= min_seconds），
因为目标是"每句话的说话人有没有对"，不是 DER。DER 只作对照。

VAD 门控窗由缓存按几何过滤得到（`build_cache` 用同一个 `arange` 步进），
所以臂 0/A/B **零 CAM++ 推理**，只有臂 C 跑真实路径。

    python scripts/evaluation/probe_gap_decomp.py --meeting L_R003S01C02 --seconds 600
    python scripts/evaluation/probe_gap_decomp.py --split dev --seconds 600
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/`。**不要插仓库根**——那里有一份同名 `claude_a2.py`
# （DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`，
# 连带毁掉 test_service 的导入。回归测试见 tests/test_evaluator_path_hygiene.py
sys.path.insert(0, str(ROOT / "streaming_asr"))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import der, read_rttm  # noqa: E402
from v12_aishell4 import meeting_group, split_meetings  # noqa: E402

SKIP_SECONDS = 10.0


def rttm_speech(segs, skip=SKIP_SECONDS, seconds=None):
    """缓存建库时的门：RTTM 语音并集，去掉前 10 秒（照抄 build_cache.rttm_speech）。"""
    merged = []
    for start, end in sorted((max(skip, s["start"]), s["end"]) for s in segs
                             if s["end"] > skip):
        if seconds and start >= seconds:
            continue
        end = min(end, seconds) if seconds else end
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap(start, end, intervals):
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)


def cached_windows(cache_dir, meeting, window, hop):
    data = np.load(Path(cache_dir) / meeting / f"w{window:g}_h{hop:g}.npz",
                   allow_pickle=False)
    return data["emb"], np.asarray(data["times"], dtype=float)


def gate_windows(emb, times, intervals, window, min_vad=0.5):
    """窗里与给定 VAD 门等价的那些（按几何过滤，零推理）。"""
    if not intervals:
        return emb[:0], times[:0]
    keep = np.array([overlap(t, t + window, intervals) >= min_vad * window
                     for t in times])
    return emb[keep], times[keep]


def window_reference(reference, times, window):
    """逐窗参考多数说话人（按帧计票，与 build_cache 同法）。"""
    duration = max(s["end"] for s in reference) + window
    n = int(duration / 0.01) + 1
    active = [[] for _ in range(n)]
    for segment in reference:
        a = max(0, int(round(segment["start"] / 0.01)))
        b = min(n, int(round(segment["end"] / 0.01)))
        for i in range(a, b):
            active[i].append(segment["speaker"])
    labels = []
    for start in times:
        counts = Counter()
        for i in range(int(round(start / 0.01)),
                       min(n, int(round((start + window) / 0.01)))):
            counts.update(active[i])
        labels.append(counts.most_common(1)[0][0] if counts else "")
    return np.asarray(labels)


def segment_accuracy(reference, starts, labels, durations, min_seconds=0.5):
    """段落级说话人正确率：每段取覆盖它的预测说话人（按重叠时长多数），对不对。

    与 `der.py` 无关，是项目的**主指标**——目标场景是轮流发言，
    每句话归对人即可，不管边界差 0.3 秒。

    两条腿的预测形态不同，都用这一支：
      * 窗标签腿：`durations` = 窗长常数（0.6 / 2.0 秒）
      * v12 真实腿：`durations` = 每个段的真实时长
    只统计**参考段的时间范围之内**的重叠——v12 的段覆盖整条时间轴（含静音），
    不设这个边界就会把静音帧也算成它在说话。
    """
    keep = [s for s in reference if s["end"] - s["start"] >= min_seconds]
    if len(set(labels)) < 2:
        # 退化输入：只有 1 个预测说话人时，"正确率"恒等于 0，
        # 那不是度量结果而是输入退化的回声。报 None，让表里显示 "-"。
        return {"accuracy": None, "n_segments": len(keep),
                "n_pred_speakers": len(set(labels)),
                "note": "only one predicted speaker; accuracy undefined"}
    total_weight = sum(s["end"] - s["start"] for s in keep) or 1.0
    hits, wrong = 0.0, Counter()
    for segment in keep:
        counts = Counter()
        for t, name, duration in zip(starts, labels, durations):
            if not name:
                continue
            cover = (min(t + duration, segment["end"])
                     - max(t, segment["start"]))
            if cover > 0:
                counts[name] += cover
        seconds = segment["end"] - segment["start"]
        if not counts:
            wrong["<无窗>"] += 1
            continue
        best = counts.most_common(1)[0][0]
        if best == segment["speaker"]:
            hits += seconds
        else:
            wrong[f"{segment['speaker']}->{best}"] += 1
    return {"accuracy": round(hits / total_weight, 4),
            "n_segments": len(keep),
            "mean_seconds": round(total_weight / max(1, len(keep)), 2),
            "wrong_top": dict(wrong.most_common(6))}


def map_by_frames(reference, starts, labels, durations, frame=0.01):
    """把任意预测名按**帧共现**映射到参考名。预测名与参考名不同源时必须先过这一步：
    `v12_0` 与 `001-M` 直接比字符串恒为不等，正确率会假报 0。"""
    end = max(s["end"] for s in reference)
    n = int(end / frame) + 1
    ref = np.full(n, "", dtype=object)
    for s in reference:
        a = max(0, int(round(s["start"] / frame)))
        b = min(n, int(round(s["end"] / frame)))
        for i in range(a, b):
            ref[i] = s["speaker"]
    counts = Counter()
    for start, name, duration in zip(starts, labels, durations):
        a = max(0, int(round(start / frame)))
        b = min(n, int(round((start + duration) / frame)))
        for i in range(a, b):
            if ref[i]:
                counts[(str(name), str(ref[i]))] += 1
    mapping, used = {}, set()
    for (pred, name), _ in counts.most_common():
        if pred not in mapping and name not in used:
            mapping[pred] = name
            used.add(name)
    return np.asarray([mapping.get(str(x), "") for x in labels])


def frame_accuracy(reference, starts, labels, durations, frame=0.01):
    """同口径的帧级说话人正确率：预测帧被映射到参考名后与参考逐帧比对。

    与 `segment_accuracy` 的区别：这里是**逐帧**记账，长段权重大、短段权重小；
    `segment_accuracy` 是**逐段**记账，每段等权（不管段长）。两条腿口径不同时
    不能相减——`B` 真值K 与 `C` v12 谁更好，必须用同一支尺子量。

    只统计 `[0, 参考末尾]` 内、参考有语音的帧，不设 collar（这是内容指标，
    不是 md-eval 的边界指标）。
    """
    end = max(s["end"] for s in reference)
    n = int(end / frame) + 1
    ref = np.full(n, "", dtype=object)
    for s in reference:
        a = max(0, int(round(s["start"] / frame)))
        b = min(n, int(round(s["end"] / frame)))
        for i in range(a, b):
            ref[i] = s["speaker"]
    pred = np.full(n, "", dtype=object)
    for start, name, duration in zip(starts, labels, durations):
        if not name:
            continue
        a = max(0, int(round(start / frame)))
        b = min(n, int(round((start + duration) / frame)))
        if b > a:
            pred[a:b] = name
    speech = ref != ""
    total = int(speech.sum())
    if not total:
        return {"accuracy": None, "frames": 0}
    return {"accuracy": round(float((pred[speech] == ref[speech]).mean()), 4),
            "frames": total,
            "n_pred_speakers": len(set(labels))}


def cluster_labels(emb, k, metric="cosine"):
    from scipy.cluster.hierarchy import fcluster, linkage
    tree = linkage(emb, method="average", metric=metric)
    return fcluster(tree, t=k, criterion="maxclust") - 1


def map_to_names(predicted, truth):
    """贪心按共现窗数把簇号映射到参考名（并人时认领最大的那位）。"""
    pairs = Counter((int(p), str(t)) for p, t in zip(predicted, truth) if t)
    mapping, used = {}, set()
    for (p, t), _ in pairs.most_common():
        if p not in mapping and t not in used:
            mapping[p] = t
            used.add(t)
    return np.asarray([mapping.get(int(p), "") for p in predicted])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--meeting")
    ap.add_argument("--split", choices=("dev", "holdout"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=v12.WIN_S)
    ap.add_argument("--hop", type=float, default=v12.HOP_S)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--min-seconds", type=float, default=0.5)
    ap.add_argument("--skip-v12", action="store_true")
    ap.add_argument("--gate", choices=("rttm", "vad"), default="rttm",
                    help="门控源。rttm=参考语音并集（不跑 VAD，快）；vad=项目真实 FSMN VAD。"
                         "A/B 腿的口径必须与要对齐的那条腿一致，否则窗集不同、结论不可比。")
    ap.add_argument("--backend", choices=("none", "global-lda"), default="none",
                    help="global-lda：用**其它会议**的纯窗拟合 LDA（留一会议），"
                         "再套到目标会议的窗上。可部署、不挑人。")
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(args.root)
    meetings = [args.meeting] if args.meeting else split_meetings(root, args.split)

    pool_cache = {}
    if args.backend != "none":
        for meeting in meetings:
            data = np.load(Path(args.cache) / meeting /
                           f"w{args.window:g}_h{args.hop:g}.npz", allow_pickle=False)
            pool_cache[meeting] = (data["emb"],
                                   np.asarray(data["times"], dtype=float),
                                   data["speakers"].astype(str),
                                   np.asarray(data["votes"], dtype=float))

    models = None
    if not args.skip_v12:
        from web_server import get_models
        models = get_models(lambda m: None, need_stream=False, need_offline=False)

    rows = []
    for meeting in meetings:
        wav = root / "wav" / f"{meeting}.flac"
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        seconds = args.seconds if args.seconds > 0 else max(s["end"] for s in reference)
        reference = [{"start": s["start"], "end": min(s["end"], seconds),
                      "speaker": s["speaker"]} for s in reference if s["start"] < seconds]
        by_speaker = Counter()
        for s in reference:
            by_speaker[s["speaker"]] += s["end"] - s["start"]
        grand = sum(by_speaker.values()) or 1.0
        effective = sorted(n for n, v in by_speaker.items() if v / grand >= args.share)

        emb_all, times_all = cached_windows(args.cache, meeting, args.window, args.hop)

        # 门的选择必须显式：以前这里是 `if models is not None`，于是 `--skip-v12`
        # 会**静默**退回 RTTM 门，A/B 两腿与 v12 真实路径就不是同一个窗集了
        # （L_R003S01C02：RTTM 门 562 窗 k=2 [438,124] 合格；VAD 门 418 窗 [421,1] 不合格）。
        # 现在：--gate 说了算；--skip-v12 时默认仍用 RTTM 门，但会打印告警。
        if args.gate == "vad":
            if models is None:
                raise SystemExit("--gate vad 需要加载模型（不要加 --skip-v12）")
            vad_gate = [(a, min(b, seconds))
                        for a, b in v11._vad_segs(models["vad"], str(wav))
                        if a < seconds]
        else:
            vad_gate = rttm_speech(reference, seconds=seconds)
        emb_rttm, times_rttm = gate_windows(emb_all, times_all,
                                            rttm_speech(reference, seconds=seconds),
                                            args.window, v11.MIN_VAD_FRACTION)
        emb_v12, times_v12 = gate_windows(emb_all, times_all, vad_gate,
                                          args.window, v11.MIN_VAD_FRACTION)
        cluster_metric = "cosine"
        if args.backend == "global-lda":
            from probe_pair_oracle import lda_fit, pure_windows
            # 类是 **(身份组, 说话人)**：跨组的同名标签是不同的人（置换检验 p=0.25），
            # 按裸标签分组会把不同人并成一类，LDA 学到的是噪声。
            Xs, ys = [], []
            for other in meetings:
                if other == meeting:
                    continue
                pe, _, ps = pure_windows(*pool_cache[other])
                if len(pe):
                    prefix = meeting_group(other) + ":"
                    Xs.append(pe)
                    ys.append(np.array([prefix + x for x in ps]))
            fit = lda_fit(np.concatenate(Xs), np.concatenate(ys)) if Xs else None
            if fit is not None:
                emb_v12 = fit(emb_v12)
                cluster_metric = fit.metric
                row_fit_windows = int(sum(len(x) for x in Xs))
            else:
                row_fit_windows = 0
        truth_rttm = window_reference(reference, times_rttm, args.window)
        truth_v12 = window_reference(reference, times_v12, args.window)

        row = {"meeting": meeting, "seconds": seconds, "backend": args.backend,
               "gate": args.gate,
               "effective": effective, "true_k": len(effective),
               "n_windows_cache": int(len(emb_all)),
               "n_windows_rttm": int(len(emb_rttm)),
               "n_windows_v12": int(len(emb_v12))}

        # 臂 0：RTTM 门 + 参考标签（= der_v12_dev.winOR 那条腿）
        win = np.full(len(times_v12), args.window)
        row["A0_rttm_gate"] = segment_accuracy(
            reference, times_rttm, truth_rttm, np.full(len(times_rttm), args.window),
            args.min_seconds)
        # 臂 A：v12 的 VAD 门 + 参考标签（完美身份）
        row["A_vad_gate"] = segment_accuracy(
            reference, times_v12, truth_v12, win, args.min_seconds)
        row["A_vad_gate_frame"] = frame_accuracy(reference, times_v12, truth_v12, win)

        # 臂 B：AHC@真值K，裸余弦
        if len(emb_v12) >= 2 and len(effective) >= 2:
            k = min(len(effective), len(emb_v12))
            clusters = cluster_labels(emb_v12, k, metric=cluster_metric)
            named = map_to_names(clusters, truth_v12)
            row["B_trueK"] = segment_accuracy(reference, times_v12, named, win,
                                              args.min_seconds)
            row["B_trueK"]["k"] = k
            # 帧级并列：同一次聚类，簇->人的映射改用**帧共现**重做
            row["B_trueK_frame"] = frame_accuracy(
                reference, times_v12,
                map_by_frames(reference, times_v12, clusters, win), win)
            row["B_trueK_frame"]["k"] = k
            row["B_trueK"]["backend"] = args.backend
            row["B_trueK_frame"]["backend"] = args.backend
            if args.backend != "none":
                row["B_trueK"]["fit_windows"] = locals().get("row_fit_windows", 0)
                row["B_trueK_frame"]["fit_windows"] = locals().get("row_fit_windows", 0)

        # 臂 C：v12 真路径（真实段边界，逐段取中点落谁的段）
        if models is not None:
            result = v12.diarize(str(wav), models, max_seconds=args.seconds)
            pred = [{"start": float(s["start"]), "end": float(s["end"]),
                     "speaker": f"v12_{int(s['spk'])}"} for s in result["segments"]]
            starts = [s["start"] for s in pred]
            names = [s["speaker"] for s in pred]
            spans = [s["end"] - s["start"] for s in pred]
            # v12 的输出是 `v12_<int>`，与参考名不同源：两种口径都要先映射。
            row["C_v12"] = segment_accuracy(
                reference, starts, map_by_frames(reference, starts, names, spans),
                spans, args.min_seconds)
            row["C_v12_frame"] = frame_accuracy(
                reference, starts, map_by_frames(reference, starts, names, spans),
                spans)
            meta = {"k": int(result["k"]),
                    "level": ("refined" if result["refined"] else
                              "rescued" if result["rescued"] else
                              "2s-only" if not result["insufficient"] else "insufficient"),
                    "der": der(reference, pred, collar=0.25)["der"],
                    "n_segments": len(pred)}
            for arm in ("C_v12", "C_v12_frame"):
                if arm in row:
                    row[arm].update(meta)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(f"\n{'会议':<16}{'有效K':>5}{'窗数':>6}{'A段':>7}{'A帧':>7}{'B段':>7}"
          f"{'B帧':>7}{'C段':>7}{'C帧':>7}{'选K':>5}{'DER':>8}")
    for r in rows:
        def cell(arm):
            d = r.get(arm)
            if not d or d.get("accuracy") is None:
                return "      -"
            return f"{d['accuracy'] * 100:7.1f}%"
        c = r.get("C_v12_frame")
        der_cell = f"{c['der'] * 100:6.1f}%" if c and "der" in c else "     -"
        print(f"{r['meeting']:<16}{r['true_k']:>5}{r['n_windows_v12']:>6}"
              f"{cell('A_vad_gate'):>7}{cell('A_vad_gate_frame'):>7}"
              f"{cell('B_trueK'):>7}{cell('B_trueK_frame'):>7}"
              f"{cell('C_v12'):>7}{cell('C_v12_frame'):>7}"
              f"{str(c['k'] if c else '-'):>5}{der_cell:>8}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-gap-decomp-v1", "meetings": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
