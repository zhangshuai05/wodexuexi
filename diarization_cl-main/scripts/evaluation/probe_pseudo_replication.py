# -*- coding: utf-8 -*-
"""伪重复检验：dev 的 12 场只有 3 个身份组，组内 4 场是**同一批说话人**。

会话中报出的 "A-B 内部损失 与 最像对余弦 相关 r=0.755 (n=12)" 把 12 场当成
12 个独立样本。但身份组划分是 L_R003C02 / M_R003C01 / S_R004C01 各 4 场，
组内说话人完全相同——同一个说话人对被量了 4 次。有效样本数接近 **3**。

如果那 3 组恰好是"L 组坏、S 组好"，那么 r=0.755 和"余弦 0.5 干净分开"
都是组间差异伪装成的样本内规律，阈值门就不成立。

输出三样：
  1. 各组构成与独立样本数
  2. 组级视图：同组 4 场的"最像对"是否同一个人对、余弦区间多宽
  3. 组内标准化后的相关（去掉组均值），看余弦还剩多少解释力

    python scripts/evaluation/probe_pseudo_replication.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from der import read_rttm  # noqa: E402
from v12_aishell4 import (  # noqa: E402
    DEV_GROUPS, HOLDOUT_GROUPS, meeting_group, split_meetings)
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, gate_windows, rttm_speech, window_reference)

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
LARGE, MERGED, SECONDS = 100, 0.80, 600.0


def l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def measure(meeting, split):
    """一场的 (最像对, 余弦, 内部帧错误损失)。损失口径与错误分布探针一致。"""
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

    k = min(len(effective), len(emb))
    clusters = cluster_labels(emb, k)
    merges = 0
    for cluster in sorted(set(clusters.tolist())):
        mask = clusters == cluster
        total = int(mask.sum())
        if total < LARGE:
            continue
        counts = Counter(t for t in truth[mask].tolist() if t)
        if counts and counts.most_common(1)[0][1] / total < MERGED:
            merges += 1

    return {"meeting": meeting, "split": split, "group": meeting_group(meeting),
            "pair": tuple(sorted((names[int(i)], names[int(j)]))),
            "cosine": float(similarity[i, j]), "group_size": len(names),
            "n_merges": merges, "has_merge": merges > 0,
            "n_windows": int(len(emb))}


def main():
    rows = []
    for split in ("dev", "holdout"):
        for meeting in split_meetings(ROOT_DATA, split):
            row = measure(meeting, split)
            if row:
                rows.append(row)

    print("=== 1. 独立样本数 ===")
    for split in ("dev", "holdout"):
        part = [r for r in rows if r["split"] == split]
        groups = sorted({r["group"] for r in part})
        print(f"  {split:<8} 会议 {len(part):>2}  身份组 {len(groups)}  -> "
              f"**有效样本 ~{len(groups)}**")
        for group in groups:
            members = [r["meeting"] for r in part if r["group"] == group]
            print(f"      {group:<14} {len(members)} 场: {', '.join(members)}")

    print("\n=== 2. 组级视图：同组内的'最像对'与余弦是否一致 ===")
    for (split, group), members in sorted(
            ((r["split"], r["group"]), [x for x in rows
                                        if x["split"] == r["split"] and x["group"] == r["group"]])
            for r in rows):
        pairs = {r["pair"] for r in members}
        cosines = np.array([r["cosine"] for r in members])
        merges = [r["n_merges"] for r in members]
        shared = set.intersection(*[set(p) for p in pairs]) if pairs else set()
        print(f"  {split:<8}{group:<14} 最像对 {len(pairs)} 种  共有说话人 {sorted(shared)}  "
              f"cos {cosines.min():.3f}~{cosines.max():.3f}  并人数 {merges}")

    print("\n=== 3. 相关：全样本 vs 组内标准化 ===")
    for split in ("dev", "holdout", "both"):
        part = rows if split == "both" else [r for r in rows if r["split"] == split]
        if len(part) < 3:
            continue
        cosine = np.array([r["cosine"] for r in part])
        merges = np.array([1.0 if r["has_merge"] else 0.0 for r in part])
        r_raw = np.corrcoef(cosine, merges)[0, 1]
        # 组内去均值（组间差异被扣掉）
        adjusted = []
        for r in part:
            same = [x for x in part if x["group"] == r["group"]]
            if len(same) > 1:
                adjusted.append((r["cosine"] - np.mean([x["cosine"] for x in same]),
                                 1.0 if r["has_merge"] else 0.0
                                 - np.mean([1.0 if x["has_merge"] else 0.0 for x in same])))
        r_within = (np.corrcoef([a for a, _ in adjusted], [b for _, b in adjusted])[0, 1]
                    if len(adjusted) > 2 else float("nan"))
        print(f"  {split:<8} n={len(part):>2}  组数={len({r['group'] for r in part})}  "
              f"r(原始)={r_raw:+.3f}   r(组内去均值)={r_within:+.3f}")

    print("\n=== 4. 门（cos>=0.5 预测有并人）分场 ===")
    for split in ("dev", "holdout"):
        part = [r for r in rows if r["split"] == split]
        tp = sum(1 for r in part if r["cosine"] >= 0.5 and r["has_merge"])
        fp = sum(1 for r in part if r["cosine"] >= 0.5 and not r["has_merge"])
        fn = sum(1 for r in part if r["cosine"] < 0.5 and r["has_merge"])
        tn = sum(1 for r in part if r["cosine"] < 0.5 and not r["has_merge"])
        print(f"  {split:<8} TP={tp} FP={fp} FN={fn} TN={tn}")
        for r in part:
            mark = "漏" if (r["has_merge"] and r["cosine"] < 0.5) else (
                "误" if (not r["has_merge"] and r["cosine"] >= 0.5) else "  ")
            print(f"      {mark} {r['meeting']:<16} cos={r['cosine']:.3f} "
                  f"并人={r['n_merges']} 对={'/'.join(r['pair'])}")

    output = ROOT / "artifacts" / "v13_pseudo_replication.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "v13-pseudo-replication-v1", "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
