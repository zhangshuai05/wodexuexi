# -*- coding: utf-8 -*-
"""v13 真因判别：K 门是不是主因？—— 绕开选 K，在真值 K 上看聚类长什么样。

**这个脚本不能单独回答「改判据还是换后端」。** 纯度只说明「知道 K 时几何可分」，
不说明**存在一个不读真值的量**能选出这个 K。所以本文件同时打两条：

  * `cluster_purity` / `n_distinct_home_clusters` —— 知道 K 时的聚类质量；
  * `criterion_curve` / `criterion_picks` —— **现行免真值判据**（`build_unlocked`，
    与生产候选表同一份实现）在整条 k 网格上选了哪个、真 K 是不是它的极值。

判据选不中真 K ⇒ 问题不在门槛，在**判据没信号**，改门槛一分不赚。

**对照 k 得看 `n_clusters_actual`。** `fcluster(maxclust)` 只保证「最多 k 个」：
真的没生成出 k 个簇时，"少掉的簇"是**层不到位**，不是"簇很脏"。
所以空簇单列（`n_empty_clusters` / `reached_target_k`），**不进纯度统计**。
（v1 把空簇记成 `purity=0.0` 并留在 `rows` 里，`worst_cluster_purity` 会被拉成 0
——一个 0 就能把「簇数不够」读成「簇很脏」。本文件已修为 v2。）

`probe_v12_k.py` 在 L_R003S01C02 上读到 k=2 的簇是 [421, 1]、k=4 是 [70, 230, 121, 1]。
一个 1 窗的孤立簇在 `_silhouette` 里把整个 K 打成 −1.0（任一簇 <2 就返回 −1.0），
所以"放宽 MIN_CLUSTER_SHARE"这条改动在这个形态上**一分不赚**——尺寸门和分数门同时挡着，
而且 `_select_candidate` 最后还有一道 `best["score"] > 0.0`。

真正要问的是下一个问题：**绕开所有门，在真值 K 上聚类，出来的簇是不是干净？**

* 干净 → 罪在选 K 判据，改判据（v13 的决策层）。
* 脏（大簇吞人、小簇碎成渣）→ 罪在特征空间本身，改判据没用，得换后端。

这个脚本**不经过任何门、任何惩罚**，直接拿真实路径的窗嵌入做一件最朴素的事：
AHC@真K，然后量每个簇相对参考的纯度、每个参考说话人有没有自己的窗。

判据（防止"全塞进一个簇"伪装成高分）：
  * `cluster_purity` 要**逐簇**看，不看均值——一个 90% 的簇 + 四个 5% 的碎簇，
    均值可能还行，但那是失败。
  * `speakers_with_home` 是"每个有效说话人是否拥有一个以它为主的簇"，这才是
    K 选对了能不能兑现的成分。缺一个，那个人的时长就全记在别人头上。

    python scripts/evaluation/probe_truek_purity.py --split dev --seconds 600
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # `unlocked_selector` 在这

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from unlocked_selector import CAP, MIN_WIN, build_unlocked, pick_unlocked  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

FRAME = 0.01


def window_reference(reference, times, window):
    """逐窗参考多数说话人 + 该窗的参考纯度（多数占窗内语音的比例）。

    纯度低的窗是跨边界窗——"窗本身脏"的直接读数，与聚类无关。
    按**帧**计票：跨边界窗里说得久的自然占多数。
    """
    duration = max(s["end"] for s in reference) + window
    n = int(duration / FRAME) + 1
    active = [[] for _ in range(n)]
    for segment in reference:
        a = max(0, int(round(segment["start"] / FRAME)))
        b = min(n, int(round(segment["end"] / FRAME)))
        for i in range(a, b):
            active[i].append(segment["speaker"])

    labels, purity = [], []
    for start in times:
        counts = Counter()
        for i in range(int(round(start / FRAME)),
                       min(n, int(round((start + window) / FRAME)))):
            counts.update(active[i])
        if not counts:
            labels.append("")
            purity.append(0.0)
            continue
        name, hits = counts.most_common(1)[0]
        labels.append(name)
        purity.append(hits / sum(counts.values()))
    return labels, purity


def cluster_rows(vectors, reference_labels, k, times):
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(vectors, method="average", metric="cosine")
    predicted = fcluster(tree, t=k, criterion="maxclust") - 1
    truth = np.asarray(reference_labels)

    rows, empty = [], 0
    for cluster in range(k):
        mask = predicted == cluster
        if not mask.any():
            # `fcluster(maxclust)` 只保证**最多** k 个簇，可能少于 k。空簇不是
            # "很脏的簇"、也不是碎簇，是**根本没生成出来**：单列计数，绝不进纯度。
            # v1 把它记成 `purity=0.0` 并留在 `rows` 里，于是
            # `worst_cluster_purity` 会把「簇数不够」误报成「簇很脏」——
            # 而这正是「改判据 vs 换后端」的判别量，错不起。
            empty += 1
            continue
        counts = Counter(name for name in truth[mask].tolist() if name)
        name, hits = counts.most_common(1)[0] if counts else ("", 0)
        rows.append({
            "cluster": cluster, "windows": int(mask.sum()),
            "majority": name, "purity": round(hits / int(mask.sum()), 4),
            "speakers": dict(counts.most_common(5)),
            "span_sec": [round(float(times[mask].min()), 1),
                         round(float(times[mask].max()), 1)],
        })
    # 每个参考说话人"家"在哪：它作为多数派的窗占它全部窗的比例。
    homes = {}
    for name in sorted(set(truth.tolist()) - {""}):
        mask = truth == name
        if not mask.any():
            continue
        cluster = int(Counter(predicted[mask].tolist()).most_common(1)[0][0])
        homes[name] = {
            "windows": int(mask.sum()),
            "home_cluster": cluster,
            "share_in_home": round(float((predicted[mask] == cluster).mean()), 4),
        }
    return predicted, rows, homes, empty


def criterion_curve(E, truth_k):
    """同一份 embedding 上，**现行免真值判据**在 k 网格上选了哪个。

    这才是「改判据 vs 换后端」的真判别量：纯度只说明「知道 K 时几何可分」，
    **不说明存在一个不读真值的量能选出这个 K**。这一步不新增判据，
    只把生产候选表的同一份实现（`build_unlocked`）在整条 k 网格上的读数打出来，
    看真 K 是不是极值。真 K 不是任何判据的极值 ⇒ 问题不在门槛，在**判据没信号**。
    """
    cands = build_unlocked(E, CAP, MIN_WIN)
    curve = [{"k": c["k"], "score": round(float(c["score"]), 4),
              "objective": round(float(c["objective"]), 4),
              "eligible": bool(c["eligible"]), "sizes": list(c["sizes"])}
             for c in cands]
    elig = [c for c in cands if c["eligible"] and c["k"] >= 2]
    # 空候选（窗太少）时 `pick_unlocked` 返回 None，旧实现此时给 k=1 —— 别把库函数变成会抛。
    obj = pick_unlocked(cands, mode="objective") or {"k": 1}
    big = pick_unlocked(cands, mode="largest_eligible") or {"k": 1}
    picks = {
        # 两个与生产可比的判据直接调同一份实现：抄一遍 tie-break / 无合格候选回退 k=1 的
        # 规则，会在 pick_unlocked 改动后静默漂移，而结论会被当生产口径引用。
        "argmax_objective": obj["k"],
        "largest_eligible": big["k"],
        # `argmax_score`（轮廓极值）是本探针的第三个判据，生产没有对应实现。
        "argmax_score": max(elig, key=lambda c: (c["score"], -c["k"]))["k"] if elig else 1,
    }
    return curve, picks, {name: (k == truth_k) for name, k in picks.items()}


def selfcheck():
    """空簇不进纯度：构造「fcluster 返回少于 k」的退化输入，断言统计不变味。

    8 个完全相同的窗 → 距离全 0 → 层次树切不出 3 个簇。
    v1 在这里会报 `worst_cluster_purity = 0.0`（两个幻影簇造成的假脏）。
    """
    vectors = np.ones((8, 3))
    times = np.arange(8, dtype=float)
    _, rows, _, empty = cluster_rows(vectors, ["A", "B"] * 4, 3, times)
    assert empty == 2, "退化输入应当出 2 个空簇，实得 %d" % empty
    assert len(rows) == 1, "非空簇应当只有 1 个，实得 %d" % len(rows)
    worst = min(row["purity"] for row in rows)
    assert worst > 0.0, "空簇混进了纯度统计（worst=%r）" % worst
    print("selfcheck ✓  目标 k=3 → 实得簇 1 个、空簇 %d 个（单列）；"
          "最差簇纯度 %.2f，未被空簇拉成 0" % (empty, worst))
    # 判据曲线在退化输入上不得抛异常（k=1 单簇时 eligible 为空，会走 else 分支）
    curve, picks, hits = criterion_curve(np.ones((8, 3)), 1)
    assert curve and picks["argmax_objective"] >= 1, (curve, picks)
    print("selfcheck ✓  判据曲线退化输入不抛异常（k 网格 %d 项，选 %r）"
          % (len(curve), picks["argmax_objective"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--meeting")
    ap.add_argument("--split", choices=("dev", "holdout", "pilot"))
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--share", type=float, default=0.05,
                    help="有效说话人门槛：占参考语音时长的比例")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output")
    ap.add_argument("--selfcheck", action="store_true",
                    help="只跑统计口径的空跑（不加载模型）")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    root = Path(args.root)
    if args.meeting:
        meetings = [args.meeting]
    elif args.split:
        meetings = split_meetings(root, args.split)
    else:
        raise SystemExit("要给 --meeting 或 --split")

    from web_server import get_models

    models = get_models(lambda message: None, need_stream=False, need_offline=False)

    report = []
    for meeting in meetings:
        wav = root / "wav" / f"{meeting}.flac"
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        seconds = args.seconds if args.seconds > 0 else max(s["end"] for s in reference)
        reference = [s for s in reference if s["start"] < seconds]

        # 与 v12.diarize() 内部逐字同参地取窗，不经过选 K。
        info = sf.info(wav)
        stop = -1 if seconds <= 0 else min(int(seconds * info.samplerate), info.frames)
        audio, rate = sf.read(wav, dtype="float32", stop=stop)
        if audio.ndim > 1:
            audio = audio[:, 0]
        vad_segments = v11._vad_segs(models["vad"], str(wav))
        vad_segments = [(start, min(end, seconds)) for start, end in vad_segments
                        if start < seconds]
        with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
            E, times, errors = v11._extract_windows(audio, rate, vad_segments,
                                                    models["spk"])
        times = np.asarray(times)

        result = v12.diarize(str(wav), models, max_seconds=args.seconds)

        labels, purity = window_reference(reference, times, v12.WIN_S)
        seconds_by_speaker = {}
        for segment in reference:
            name = segment["speaker"]
            seconds_by_speaker[name] = (seconds_by_speaker.get(name, 0.0)
                                        + segment["end"] - segment["start"])
        grand = sum(seconds_by_speaker.values()) or 1.0
        effective = sorted(name for name, value in seconds_by_speaker.items()
                           if value / grand >= args.share)
        in_pool = {name for name in labels if name}

        item = {
            "meeting": meeting, "seconds": seconds,
            "n_windows": int(len(E)), "embedding_errors": int(errors),
            "window_seconds": v12.WIN_S,
            "reference_speakers_all": len(seconds_by_speaker),
            "effective": effective, "true_k": len(effective),
            "effective_without_window": [n for n in effective if n not in in_pool],
            "speakers_visible": len([n for n in effective if n in in_pool]),
            "window_purity_mean": round(float(np.mean(purity)), 4) if len(purity) else None,
            "window_purity_lt_50pct": int(sum(1 for value in purity if value < 0.5)),
            "v12_selected_k": int(result["k"]),
            "v12_refined": bool(result["refined"]),
            "v12_rescued": bool(result["rescued"]),
            "ahc_true_k": None, "cluster_purity": [], "speaker_homes": {},
            "min_cluster_windows": None,
            "n_clusters_actual": 0, "n_empty_clusters": 0,
            "reached_target_k": False, "n_distinct_home_clusters": 0,
            "criterion_curve": [], "criterion_picks": {}, "criterion_hits_truth_k": {},
        }

        if len(E) >= 2 and len(effective) >= 2:
            k = min(len(effective), len(E))
            _, rows, homes, empty = cluster_rows(E, labels, k, times)
            item["ahc_true_k"] = k
            item["cluster_purity"] = rows
            item["speaker_homes"] = homes
            # 空簇只在这两个数里出现，不参与下面的纯度与最小规模。
            item["n_clusters_actual"] = len(rows)
            item["n_empty_clusters"] = empty
            item["reached_target_k"] = (len(rows) == k)
            item["min_cluster_windows"] = min((row["windows"] for row in rows),
                                              default=0)
            # 纯度的诚实汇总：按窗加权，但另报**最差簇**——一个簇吞掉大半、
            # 其余碎成渣时均值会骗人，最差簇不会。（空簇不算最差簇。）
            item["worst_cluster_purity"] = round(
                min((row["purity"] for row in rows), default=0.0), 4)
            item["speakers_with_own_cluster"] = sum(
                1 for name in effective
                if homes.get(name, {}).get("share_in_home", 0.0) >= 0.5)
            # ⚠「有家」看不见合并：两人并进同一簇时两边 `share_in_home` 都≈1.0、
            # 都得满分。家簇**去重**后的个数才是「真分出了几个不同的家」。
            # dev 实测：12 场里 6 场报「有家=真K」，去重后家簇数显著更少。
            item["n_distinct_home_clusters"] = len({
                homes[name]["home_cluster"] for name in effective
                if homes.get(name, {}).get("share_in_home", 0.0) >= 0.5})
            curve, picks, hits = criterion_curve(E, len(effective))
            item["criterion_curve"] = curve
            item["criterion_picks"] = picks
            item["criterion_hits_truth_k"] = hits

        report.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-truek-purity-v2", "meetings": report},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")

    print(f"\n{'会议':<16}{'真K':>4}{'可见':>5}{'簇数':>5}{'空簇':>5}{'选K':>5}"
          f"{'窗数':>6}{'窗纯度':>8}{'最差簇':>8}{'有家':>6}{'家簇':>5}  缺窗说话人")
    for item in report:
        homes = item.get("speakers_with_own_cluster")
        worst = item.get("worst_cluster_purity")
        print(f"{item['meeting']:<16}{item['true_k']:>4}"
              f"{item.get('speakers_visible', 0):>5}"
              f"{item.get('n_clusters_actual', 0):>5}{item.get('n_empty_clusters', 0):>5}"
              f"{item['v12_selected_k']:>5}{item['n_windows']:>6}"
              f"{(item['window_purity_mean'] or 0) * 100:>7.1f}%"
              f"{(worst or 0) * 100:>7.1f}%"
              f"{str(homes) + '/' + str(item['true_k']):>6}"
              f"{item.get('n_distinct_home_clusters', 0):>5}  "
              f"{','.join(item['effective_without_window']) or '-'}")
        picks = item.get("criterion_picks") or {}
        if picks:
            print(f"{'':16}  真K={item['true_k']} 免真值判据选中: "
                  + "  ".join("%s=%d%s" % (
                      label, picks[name],
                      "✓" if item["criterion_hits_truth_k"].get(name) else "✗")
                      for name, label in (("argmax_objective", "目标函数"),
                                          ("argmax_score", "轮廓"),
                                          ("largest_eligible", "最大合格"))))

    if report:
        ok = [item for item in report if item.get("ahc_true_k")]
        if ok:
            n_ok = sum(1 for i in ok if i.get("reached_target_k"))
            print(f"\n真K 下：平均最差簇纯度 "
                  f"{np.mean([i['worst_cluster_purity'] for i in ok]) * 100:.1f}%"
                  f"，平均有家 {np.mean([i['speakers_with_own_cluster'] for i in ok]):.2f}"
                  f"/{np.mean([i['true_k'] for i in ok]):.2f} 人"
                  f"，去重后家簇 {np.mean([i['n_distinct_home_clusters'] for i in ok]):.2f}")
            print(f"达标 K（真生成出目标簇数）{n_ok}/{len(ok)} 场；"
                  f"v12 选 K 命中 {sum(1 for i in ok if i['v12_selected_k'] == i['true_k'])}"
                  f"/{len(ok)} 场")
            for name, label in (("argmax_objective", "argmax 目标函数"),
                                ("argmax_score", "argmax 轮廓"),
                                ("largest_eligible", "最大合格 k")):
                print(f"  免真值判据能选出真K：{label} "
                      f"{sum(1 for i in ok if i['criterion_hits_truth_k'].get(name))}"
                      f"/{len(ok)} 场")


if __name__ == "__main__":
    main()
