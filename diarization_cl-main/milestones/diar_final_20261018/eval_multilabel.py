# -*- coding: utf-8 -*-
"""多标签口径评测：给「OSD+匈牙利后处理」（B 方案）一把对的尺子。

**为什么写这个**（2026-10-15）：
B 方案「真重叠时标出两人」唯一的实现是 run_3dspeaker_pipeline.post_process
（逐 OSD 窗匈牙利匹配，class 4/5/6 激活两个全局说话人）。此前用单标签句级
Turn Accuracy 评判它，得出"2/3 场有害"——但单标签指标天然惩罚"一句话给了
两个标签"，等于用错尺子下结论。本脚本补上多标签口径：

真值侧
    每句的真值说话人集合 = 参考 RTTM 里与该句 span 重叠
    ≥ max(min_abs, min_rel × 句长) 的所有说话人（不再假设一句只有一人）。

预测侧
    每句的预测标签集合 = 该句帧 span 内占比 ≥ second_thresh 的簇（∪ 主标签），
    簇 → 真值说话人的映射用**全会议帧级共现时长**的匈牙利双射（不逐句贪心）。

指标（全部按句平均）
    acc_single  主标签句级准确率（单标签对照，可跟旧 harm 表的趋势对，但映射方法不同）
    coverage    预测非空的句子比例
    det2        真值重叠句（|S_gt|≥2）中被预测出 ≥2 个标签的比例（多标签检出率）
    second_ok   预测出 ≥2 标签的句子里，"多出来的标签"确实属于真值集合的比例
    pair_exact  真值重叠句中，预测集合与真值集合**完全相等**的比例（最严的尺子）
    false2      真值单人句被标出 ≥2 人的比例（多标签误报率）

用法：
    PYTHONIOENCODING=utf-8 python eval_multilabel.py \
        --meetings R8007_M8010 R8008_M8013 R8009_M8020
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav
from probe_sentence_attribution import (read_textgrid,  # noqa: F401
                                       sentences_from_textgrid)
from run_3dspeaker_pipeline import (class_chunks, chunk_windows, embed_chunks,
                                    osd_probs, post_process,
                                    spectral_cluster)

_, read_rttm = der_tools()


def gt_speaker_sets(turns, ref, min_abs, min_rel):
    """每句的真值说话人集合（不再假设一句一人）。"""
    out = []
    for turn in turns:
        dur = turn["end"] - turn["start"]
        thr = max(min_abs, min_rel * dur)
        members = set()
        for seg in ref:
            ov = min(turn["end"], seg["end"]) - max(turn["start"], seg["start"])
            if ov >= thr:
                members.add(seg["speaker"])
        if not members:               # 极短句：退回 ≥0.05s 或文本层说话人
            for seg in ref:
                ov = min(turn["end"], seg["end"]) - max(turn["start"], seg["start"])
                if ov >= 0.05:
                    members.add(seg["speaker"])
        if not members:
            members = {turn["speaker"]}
        out.append(members)
    return out


def global_cluster_mapping(binary, ref, fs, n_spk_min=1):
    """簇 → 真值说话人：全会议帧级共现时长的匈牙利双射（比逐句贪心稳）。"""
    from scipy.optimize import linear_sum_assignment

    n, k = binary.shape
    speakers = sorted({seg["speaker"] for seg in ref})
    spk_frames = np.zeros((len(speakers), n), dtype=bool)
    for seg in ref:
        i0 = max(0, int(seg["start"] / fs))
        i1 = min(n, int(seg["end"] / fs))
        if i1 > i0:
            spk_frames[speakers.index(seg["speaker"]), i0:i1] = True
    cost = np.zeros((k, len(speakers)), dtype=np.float64)
    for c in range(k):
        col = binary[:, c] > 0.5
        if col.any():
            cost[c] = (spk_frames & col[None, :]).sum(axis=1)
    rows, cols = linear_sum_assignment(-cost)
    mapping = {}
    for r, c in zip(rows, cols):
        if cost[r, c] > 0:
            mapping[r] = speakers[c]
    return mapping


def sentence_pred_sets(binary, turns, fs, thresh, mapping):
    """每句的预测说话人集合（经全局映射；映射不到的簇计入 unmapped_extra）。"""
    n, k = binary.shape
    out = []
    for turn in turns:
        i0 = max(0, int(turn["start"] / fs))
        i1 = min(n, int(turn["end"] / fs))
        if i1 <= i0:
            out.append({"pred": set(), "primary": None})
            continue
        share = binary[i0:i1].mean(axis=0)
        if share.max() <= 0:
            # 全零 = 本句没有任何激活簇。旧代码仍取 argmax(全零)=0 当 primary，
            # 只要簇 0 有映射就会白送正确句（acc 虚高、coverage 不涨）。
            out.append({"pred": set(), "primary": None})
            continue
        clusters = [c for c in range(k) if share[c] >= thresh]
        if not clusters:
            clusters = [int(np.argmax(share))]
        pred = set()
        for c in clusters:
            if c in mapping:
                pred.add(mapping[c])
        primary = mapping.get(int(np.argmax(share)))
        out.append({"pred": pred, "primary": primary})
    return out


def vad_sentences(vad, wav, max_end_silence_ms=450):
    """生产口径的句子：fsmn-vad 切出的段就是 utt_id（450ms 尾部静音断句）。

    ⚠️ 生产里 VAD 是 `web_server.get_models` 用 `AutoModel(model=VAD_LOCAL, **options)`
    加载的，**没传 max_end_silence_time**，走的是模型默认值；而 450ms 是本方案提议的
    敏捷断句。两者是否等价**不靠猜** —— 本函数把两种参数的段数都返回，由调用方打印。

    另外：真实生产里 utt_id 是 VAD 段再过流式 ASR 得到的（见 .zhang/streaming_asr_engine.py），
    这里用 VAD 段直接当句子，是**同口径的近似**（少了 ASR 二次切分），差异要记在结论里。
    """
    import claude_a2_v11 as v11
    default_segs = v11._vad_segs(vad, str(wav))
    segs = default_segs
    result = None
    try:
        result = vad.generate(input=str(wav), max_end_silence_time=max_end_silence_ms,
                              dynamic_silence=False)
    except TypeError as exc:                    # 只吞"不认识该 kwarg"，真失败要炸出来
        print("    [VAD] %dms 覆盖失败（%s: %s），先只丢 dynamic_silence 重试"
              % (max_end_silence_ms, type(exc).__name__, str(exc)[:80]), flush=True)
        try:
            result = vad.generate(input=str(wav),
                                  max_end_silence_time=max_end_silence_ms)
        except TypeError as exc2:
            print("    [VAD] 仍不接受 max_end_silence_time（%s: %s），退回默认参数"
                  % (type(exc2).__name__, str(exc2)[:80]), flush=True)
    if result is not None:
        rows = result if isinstance(result, list) else [result]
        segs = [(float(x[0]) / 1000.0, float(x[1]) / 1000.0)
                for row in rows for x in row.get("value", [])]
    return ([{"start": a, "end": b, "speaker": None} for a, b in segs],
            len(default_segs), len(segs))


def attach_truth_spk(sents, ref):
    """给 VAD 句子补"真值主说话人"（段内重叠时长最高者），供 acc_single 使用。"""
    for s in sents:
        tally = defaultdict(float)
        for seg in ref:
            ov = min(s["end"], seg["end"]) - max(s["start"], seg["start"])
            if ov > 0:
                tally[seg["speaker"]] += ov
        s["speaker"] = max(tally, key=lambda x: tally[x]) if tally else "?"
    return sents


def chunk_binary(labels, kept, n_frames, k, fs):
    """chunk 标签铺成帧级 0/1 矩阵，好让 post 关的臂也用同一套匈牙利映射。"""
    binary = np.zeros((n_frames, k), dtype=np.float32)
    for (a, b), lab in zip(kept, labels.tolist()):
        i0, i1 = int(a / fs), min(n_frames, int(b / fs))
        if i1 > i0:
            binary[i0:i1, lab] = 1.0
    return binary


def single_label_preds(labels, kept, turns, mapping):
    """post 关的单标签臂：chunk 标签按重叠时长投票 -> 每句一个标签。"""
    out = []
    for turn in turns:
        tally = defaultdict(float)
        for (a, b), lab in zip(kept, labels.tolist()):
            ov = min(b, turn["end"]) - max(a, turn["start"])
            if ov > 0:
                tally[lab] += ov
        if not tally:
            out.append({"pred": set(), "primary": None})
            continue
        spk = mapping.get(max(tally, key=lambda x: tally[x]))
        out.append({"pred": {spk} if spk else set(), "primary": spk})
    return out


def score_arm(turns, gts, preds):
    """一组预测的六项指标（算法与骨架原版逐项一致，只是抽出来复用）。"""
    n = len(turns)
    acc = cov = 0
    n_gt2 = det2 = pair_exact = 0
    n_pred2 = second_ok = 0
    n_gt1 = false2 = 0
    for turn, g, p in zip(turns, gts, preds):
        P, prim = p["pred"], p["primary"]
        if P:
            cov += 1
        if prim == turn["speaker"]:
            acc += 1
        if len(g) >= 2:
            n_gt2 += 1
            if len(P) >= 2:
                det2 += 1
            if P and P == g:
                pair_exact += 1
        else:
            n_gt1 += 1
            if len(P) >= 2:
                false2 += 1
        if len(P) >= 2:
            n_pred2 += 1
            # 只把"主标签之外"的标签当第二人候选。旧代码在 prim∉P 时用
            # set(list(P)[1:])，依赖 set 迭代序，同一输入跨进程可能不同。
            extras = set(P) - {prim} if prim in P else set(P)
            if extras and extras <= g:
                second_ok += 1
    return {"acc_single": acc / max(n, 1), "coverage": cov / max(n, 1),
            "n_gt_overlap": n_gt2, "n_gt_single": n_gt1,
            "det2": det2 / max(n_gt2, 1),
            "second_ok": second_ok / max(n_pred2, 1), "n_pred2": n_pred2,
            "pair_exact": pair_exact / max(n_gt2, 1),
            "false2": false2 / max(n_gt1, 1)}


def evaluate_meeting(rows_out, meeting, args, models, osd_sess):
    t0 = time.time()
    wav = resolve_wav(meeting)
    audio, rate = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    ref = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    tg_turns = sentences_from_textgrid(read_textgrid(ALI_TG_LAZY(meeting)))
    vad_turns, n_vad_default, n_vad_450 = vad_sentences(
        models["vad"], wav, max_end_silence_ms=args.vad_silence)
    attach_truth_spk(vad_turns, ref)
    n_true_speakers = len({s["speaker"] for s in ref})

    # ★ 重活只跑一次：OSD 与嵌入跟"句子从哪来"无关，所以一次推理喂两条句子臂
    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins, _ = class_chunks(probs, fs)
    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    labels_auto, k_auto, _ = spectral_cluster(E, None)
    k_forced = min(n_true_speakers, max(len(E) - 1, 1))
    labels_forced, _, _ = spectral_cluster(E, k_forced)

    print("  [%s] 句子：TextGrid %d 句 / VAD %d 段（VAD 默认参数 %d 段，%dms %d 段）"
          % (meeting, len(tg_turns), len(vad_turns), n_vad_default,
             args.vad_silence, n_vad_450), flush=True)

    row = {"meeting": meeting, "true_k": n_true_speakers, "k_auto": int(k_auto),
           "n_chunks": len(E), "n_sentences_tg": len(tg_turns),
           "n_sentences_vad": len(vad_turns),
           "n_vad_default": n_vad_default, "n_vad_450": n_vad_450,
           "vad_silence": args.vad_silence,
           "second_thresh": args.second_thresh,
           "min_abs": args.min_abs, "min_rel": args.min_rel}

    binary_post = post_process(labels_auto, kept, probs, k_auto, fs)
    binary_nopost = chunk_binary(labels_auto, kept, len(probs), k_auto, fs)
    for src, turns in (("tg", tg_turns), ("vad", vad_turns)):
        gts = gt_speaker_sets(turns, ref, args.min_abs, args.min_rel)
        for mode, binary in (("post", binary_post), ("nopost", binary_nopost)):
            mapping = global_cluster_mapping(binary, ref, fs)
            preds = (sentence_pred_sets(binary, turns, fs, args.second_thresh, mapping)
                     if mode == "post"
                     else single_label_preds(labels_auto, kept, turns, mapping))
            m = score_arm(turns, gts, preds)
            m["n_mapped_clusters"] = len(mapping)
            m["k_used"] = int(k_auto)
            row["%s_%s" % (src, mode)] = m

    # 诊断臂：真值 K + post —— 回答"若人数已知，多标签能到多少"（不是可达收益）
    binary_fk = post_process(labels_forced, kept, probs, k_forced, fs)
    mapping_fk = global_cluster_mapping(binary_fk, ref, fs)
    row["tg_post_truek"] = score_arm(
        tg_turns, gt_speaker_sets(tg_turns, ref, args.min_abs, args.min_rel),
        sentence_pred_sets(binary_fk, tg_turns, fs, args.second_thresh, mapping_fk))
    row["tg_post_truek"]["k_used"] = int(k_forced)

    row["elapsed_s"] = round(time.time() - t0, 1)
    rows_out.append(row)
    print("    臂            单标签  检出2标  第二人对  集合全对  误报2标  覆盖")
    for key in ("tg_post", "tg_nopost", "vad_post", "vad_nopost", "tg_post_truek"):
        m = row[key]
        print("    %-13s %5.1f%%  %5.1f%%   %5.1f%%    %5.1f%%    %5.1f%%   %5.1f%%"
              % (key, m["acc_single"] * 100, m["det2"] * 100, m["second_ok"] * 100,
                 m["pair_exact"] * 100, m["false2"] * 100, m["coverage"] * 100),
              flush=True)


def ALI_TG_LAZY(meeting):
    from common import ALI_TG
    return ALI_TG / f"{meeting}.TextGrid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*",
                    default=["R8007_M8010", "R8008_M8013", "R8009_M8020"])
    ap.add_argument("--out", default="multilabel_eval.json")
    ap.add_argument("--vad-silence", type=int, default=450,
                    help="VAD 句尾静音阈值(ms)。默认 450 = HANDOFF_1015 §4 的口径")
    ap.add_argument("--second-thresh", type=float, default=0.25,
                    help="第二标签的帧占比门槛（与 turns_multilabel 的 0.25 对齐）")
    ap.add_argument("--min-abs", type=float, default=0.20,
                    help="真值说话人计入句子集合的最小重叠秒数")
    ap.add_argument("--min-rel", type=float, default=0.20,
                    help="真值说话人计入句子集合的最小重叠占句长比例")
    ap.add_argument("--judge-pair-exact", type=float, default=0.60,
                    help="判据①：真值重叠句上集合全对率需 >= 此值")
    ap.add_argument("--judge-false2", type=float, default=0.10,
                    help="判据②：真值单人句被标 >=2 人的比例需 <= 此值")
    args = ap.parse_args()

    import onnxruntime as ort
    from run_der import load_models

    models = load_models()
    osd_path = Path(r"D:\yinp\pyannote_seg3\model.onnx")
    if not osd_path.exists():
        raise SystemExit(f"找不到 OSD 模型 {osd_path}")
    osd_sess = ort.InferenceSession(str(osd_path), providers=["CPUExecutionProvider"])

    rows = []
    print("%-14s %-28s %-28s %-28s %-22s"
          % ("meeting", "单标签acc/检出2标", "第二人正确/集合全对", "单人句误报/覆盖", "K(真/自选)"),
          flush=True)
    for meeting in args.meetings:
        evaluate_meeting(rows, meeting, args, models, osd_sess)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"second_thresh": args.second_thresh,
                                   "min_abs": args.min_abs, "min_rel": args.min_rel,
                                   "note": "多标签口径：真值集合来自参考 RTTM 与句子 span 的重叠；"
                                           "簇→人映射=全会议帧级共现匈牙利；arm auto=谱聚类自选K，true=真值K"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        import statistics as st
        ARMS = ("tg_post", "tg_nopost", "vad_post", "vad_nopost", "tg_post_truek")
        METRICS = ("acc_single", "det2", "second_ok", "pair_exact", "false2", "coverage")
        print("\n=== 汇总（%d 场）===" % len(rows))
        print("  %-15s %8s %9s %9s %9s %9s %8s"
              % ("arm", "单标签", "检出2标", "第二人对", "集合全对", "误报2标", "覆盖"))
        for key in ARMS:
            vals = [st.mean([r[key][m] for r in rows]) * 100 for m in METRICS]
            print("  %-15s %7.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%% %7.1f%%"
                  % tuple([key] + vals))

        # 判据：跑之前约定、写在代码里，避免事后解释
        pe = st.mean([r["tg_post"]["pair_exact"] for r in rows])
        f2 = st.mean([r["tg_post"]["false2"] for r in rows])
        d2 = st.mean([r["tg_post"]["det2"] for r in rows])
        verdict = (pe >= args.judge_pair_exact) and (f2 <= args.judge_false2)
        print("\n=== 判据（事前约定）===")
        print("  人工句+多标签臂：集合全对 %.1f%%（需 ≥ %.0f%%）| 单人句误报 %.1f%%（需 ≤ %.0f%%）| 检出2标 %.1f%%"
              % (pe * 100, args.judge_pair_exact * 100,
                 f2 * 100, args.judge_false2 * 100, d2 * 100))
        print("  结论：%s"
              % ("多标签默认开（后处理保留）" if verdict else
                 "B 方案换实现路线 —— 直接消费 powerset 的 class 4/5/6，不经匈牙利匹配"))
        print("  口径提醒：VAD 臂用 VAD 段直接当句子，真实生产是 VAD 段再过流式 ASR，"
              "少了二次切分；比 VAD 默认段数均值（%d）与 %dms 段数均值（%d）可知断句粒度影响"
              % (int(st.mean([r["n_vad_default"] for r in rows])),
                 args.vad_silence,
                 int(st.mean([r["n_vad_450"] for r in rows]))))
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
