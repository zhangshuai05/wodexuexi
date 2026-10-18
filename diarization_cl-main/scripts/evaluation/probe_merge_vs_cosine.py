# -*- coding: utf-8 -*-
"""尺寸感知的并人诊断：大簇被近均分才算"真并人"，碎簇不算。

上一轮用 size-blind 的"最脏簇"去比"最像对"，把 3 窗的碎簇和 434 窗的真并人
并列，那个比较本身是错的（一致率也就没有意义）。这里改成：

  * 真并人 = 真值K 聚类后，>=100 窗 且 主说话人占比 < 80% 的簇
  * "最像对" 分三种：全池最像 / 大簇之间最像 / 真并人内部那两个

要回答的是：**"最像对"是不是"真并人"的代理**。如果不是，那么用单条
"max-pair 余弦 >= 0.5"当闸门就是错的，之前那套 r=0.755 的叙事也要重写。

    python scripts/evaluation/probe_merge_vs_cosine.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, gate_windows, rttm_speech, window_reference)

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
LARGE = 100        # 大簇门槛（窗数）
MERGED = 0.80      # 主说话人占比低于此值 = 近均分（真并人）
SECONDS = 600.0


def l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def analyse(meeting, split):
    reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                 if s["start"] < SECONDS]
    reference = [{"start": s["start"], "end": min(s["end"], SECONDS),
                  "speaker": s["speaker"]} for s in reference]
    emb_all, times_all = cached_windows(str(CACHE), meeting, 2.0, 1.0)
    emb, times = gate_windows(emb_all, times_all,
                              rttm_speech(reference, seconds=SECONDS), 2.0, 0.5)
    truth = window_reference(reference, times, 2.0)

    duration = Counter()
    for s in reference:
        duration[s["speaker"]] += s["end"] - s["start"]
    grand = sum(duration.values()) or 1.0
    effective = sorted(n for n, v in duration.items() if v / grand >= 0.05)

    centroids, names, sizes = [], [], []
    for name in sorted(set(truth.tolist()) - {""}):
        index = truth == name
        if index.sum() >= 8:
            centroids.append(l2norm(emb[index].mean(0)))
            names.append(name)
            sizes.append(int(index.sum()))
    if len(centroids) < 2:
        return None

    matrix = np.stack(centroids)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)

    i, j = np.unravel_index(int(similarity.argmax()), similarity.shape)
    cos_all = float(similarity[i, j])
    pair_all = tuple(sorted((names[int(i)], names[int(j)])))

    big = [x for x, size in enumerate(sizes) if size >= LARGE]
    cos_large, pair_large = None, None
    if len(big) >= 2:
        block = similarity[np.ix_(big, big)]
        a, b = np.unravel_index(int(block.argmax()), block.shape)
        cos_large = float(block[a, b])
        pair_large = tuple(sorted((names[big[int(a)]], names[big[int(b)]])))

    k = min(len(effective), len(emb))
    clusters = cluster_labels(emb, k)
    merges = []
    for cluster in sorted(set(clusters.tolist())):
        mask = clusters == cluster
        total = int(mask.sum())
        if total < LARGE:
            continue
        counts = Counter(t for t in truth[mask].tolist() if t)
        if not counts:
            continue
        ratio = counts.most_common(1)[0][1] / total
        if ratio < MERGED:
            merges.append((tuple(sorted(t for t, _ in counts.most_common(2))), total, ratio))
    merges.sort(key=lambda item: -item[1])
    merged_pair = merges[0][0] if merges else None

    cos_inside = None
    if merged_pair:
        index = [names.index(x) for x in merged_pair if x in names]
        if len(index) == 2:
            cos_inside = float(similarity[index[0], index[1]])

    return {"meeting": meeting, "split": split, "k": k,
            "n_windows": int(len(emb)), "n_speakers": len(names),
            "merged_pair": merged_pair, "n_merges": len(merges),
            "merged_windows": merges[0][1] if merges else None,
            "merged_ratio": round(merges[0][2], 3) if merges else None,
            "pair_all": pair_all, "cos_all": round(cos_all, 3),
            "pair_large": pair_large,
            "cos_large": round(cos_large, 3) if cos_large is not None else None,
            "cos_inside": round(cos_inside, 3) if cos_inside is not None else None,
            "agree_all": (pair_all == merged_pair) if merged_pair else None,
            "agree_large": (pair_large == merged_pair) if merged_pair else None}


def main():
    rows = []
    for split in ("dev", "holdout"):
        for meeting in split_meetings(ROOT_DATA, split):
            row = analyse(meeting, split)
            if row:
                rows.append(row)
                label = (f"{row['merged_pair'][0][:6]}/{row['merged_pair'][1][:6]}"
                         f" {row['merged_windows']}窗" if row["merged_pair"] else "无")
                flag = lambda v: f"{v:>9.3f}" if v is not None else "        -"
                agree = "-" if row["agree_all"] is None else ("是" if row["agree_all"] else "否")
                print(f"{row['meeting']:<16}{row['split']:<8}K={row['k']:<3}"
                      f"{label:>20}{row['cos_all']:>9.3f}{flag(row['cos_large'])}"
                      f"{flag(row['cos_inside'])}{agree:>5}", flush=True)

    print("\n===== 汇总 =====")
    for split in ("dev", "holdout"):
        part = [r for r in rows if r["split"] == split]
        with_merge = [r for r in part if r["merged_pair"]]
        print(f"  {split:<8} 有真并人 {len(with_merge)}/{len(part)}")
        for key, field in (("全池最像对", "agree_all"), ("大簇间最像对", "agree_large")):
            hit = sum(1 for r in with_merge if r[field])
            print(f"           {key:<12} 命中真并人对 {hit}/{len(with_merge)}")

    with_merge = [r for r in rows if r["merged_pair"]]
    if with_merge:
        inside = [r["cos_inside"] for r in with_merge if r["cos_inside"] is not None]
        allc = [r["cos_all"] for r in with_merge if r["cos_inside"] is not None]
        if inside:
            lower = sum(1 for a, b in zip(inside, allc) if a < b)
            print(f"\n  并人内那对余弦 中位 {np.median(inside):.3f}  vs  "
                  f"全池最像 中位 {np.median(allc):.3f}")
            print(f"  并人内更小（= 全池最像抓错了对）: {lower}/{len(inside)} 场")

    print("\n  门 cos>=0.5 预测「有真并人」的混淆表：")
    for split in ("dev", "holdout"):
        part = [r for r in rows if r["split"] == split]
        tp = sum(1 for r in part if r["cos_all"] >= 0.5 and r["merged_pair"])
        fp = sum(1 for r in part if r["cos_all"] >= 0.5 and not r["merged_pair"])
        fn = sum(1 for r in part if r["cos_all"] < 0.5 and r["merged_pair"])
        tn = sum(1 for r in part if r["cos_all"] < 0.5 and not r["merged_pair"])
        print(f"    {split:<8} TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"（召回 {tp}/{tp + fn}，误报 {fp}/{fp + tn}）")

    output = ROOT / "artifacts" / "v13_merge_vs_cosine.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps({"schema": "v13-merge-vs-cosine-v1", "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
