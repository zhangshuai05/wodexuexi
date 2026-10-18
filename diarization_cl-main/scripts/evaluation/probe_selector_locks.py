# -*- coding: utf-8 -*-
"""v11 选 K 的三道锁逐一拆开，量每一道各值多少。

`_candidate_clusters` + `_select_candidate` 里有三道**串联**的锁，任一道拦下
就把整个 K>=2 候选集废掉，最终回退到 `candidates[0]` = k=1：

  锁 1 尺寸门   min(sizes) >= MIN_CLUSTER_WINDOWS(4)
  锁 2 份额门   min(sizes) / n >= MIN_CLUSTER_SHARE(0.015)
  锁 3 轮廓系数 任一簇 <2 窗 -> `_silhouette` 直接返回 -1.0
               -> `_select_candidate` 末尾的 `best["score"] > 0.0` 再拦一次

一个只含 1 个窗的孤立簇可以单独触发**锁 1 与锁 3**（它同时让尺寸门不合格、
让轮廓系数返回 -1.0），这就是"一个孤窗废掉整张候选表"的机制。

上一版只拆锁 1 的尺寸门、没动 `_silhouette`，所以一分不赚——**拆错了锁**。
这一版逐级拆，量出每道的真实价值。

零推理：全部走 `w2_h1.npz` 缓存（RTTM 门窗口集）。

    python scripts/evaluation/probe_selector_locks.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` 与本地目录。**不要插仓库根**——那里有一份同名
# `claude_a2.py`（DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`，
# 连带毁掉 test_service 的导入。回归测试见 tests/test_evaluator_path_hygiene.py
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402

from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0


def silhouette_core(E, labels, max_samples=512):
    """同 `v11._silhouette`，但**只保留 >=2 窗的簇**（孤窗不参与）。

    关键修复：丢掉孤窗后簇号不再连续，必须**重新编号** 0..m-1，
    否则 `groups[own]` 会越界（ids 沿用了原始标签）。
    """
    kept_indices = [np.flatnonzero(labels == label)
                    for label in sorted(set(labels.tolist()))]
    kept_indices = [group for group in kept_indices if len(group) >= 2]
    if len(kept_indices) < 2:
        return -1.0
    keep = np.concatenate(kept_indices)
    # 显式重编号：原始标签 -> 连续下标
    present = sorted({int(labels[i]) for i in keep})
    remap = {label: position for position, label in enumerate(present)}
    vectors = E[keep]
    relabelled = np.asarray([remap[int(labels[i])] for i in keep])
    groups = [np.flatnonzero(relabelled == position)
              for position in range(len(present))]
    sample = np.linspace(0, len(vectors) - 1, min(len(vectors), max_samples), dtype=int)
    values = []
    for index in sample:
        own = relabelled[index]
        distances = 1.0 - (vectors[index] @ vectors.T)
        same = groups[own]
        same = same[same != index]
        intra = float(distances[same].mean()) if len(same) else 0.0
        other = min(float(distances[group].mean())
                    for position, group in enumerate(groups) if position != own)
        values.append((other - intra) / max(other, intra, 1e-9))
    return float(np.mean(values))


def build(E, cap, unlock):
    """unlock: 0=现状 1=+放开上限 2=+拆尺寸/份额门 3=+轮廓系数容忍孤窗"""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n < 2:
        return []
    tree = linkage(E, method="average", metric="cosine")
    candidates = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
                   "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == label).sum()) for label in range(k)]
        if unlock >= 2:
            big = [size for size in sizes if size >= 2]
            eligible = (len(big) >= 2 and min(big) >= v11.MIN_CLUSTER_WINDOWS
                        and min(big) / n >= v11.MIN_CLUSTER_SHARE)
        else:
            eligible = (min(sizes) >= v11.MIN_CLUSTER_WINDOWS
                        and min(sizes) / n >= v11.MIN_CLUSTER_SHARE)
        scorer = silhouette_core if unlock >= 3 else v11._silhouette
        score = scorer(E, labels) if eligible else -1.0
        candidates.append({"k": k, "labels": labels, "score": score,
                           "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                           "sizes": sizes, "eligible": eligible})
    return candidates


def select(candidates, unlock):
    eligible = [c for c in candidates if c["eligible"] and c["k"] >= 2]
    if not eligible:
        return candidates[0]
    best = max(eligible, key=lambda c: (c["objective"], -c["k"]))
    if unlock >= 3:
        # 拆掉锁 3 的第二次拦截：仍有合格候选时不再无条件回退 k=1
        return best
    return best if best["score"] > 0.0 else candidates[0]


RULES = [("现状", 4, 0), ("放开上限", 10, 1), ("拆尺寸+份额门", 10, 2),
         ("再拆轮廓系数", 10, 3)]


def main():
    accuracy = {name: [] for name, _, _ in RULES}
    print(f"{'会议':<16}{'有效K':>6} | " + "".join(f"{name:>16}" for name, _, _ in RULES))
    for meeting in split_meetings(ROOT_DATA, "dev"):
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < SECONDS]
        reference = [{"start": s["start"], "end": min(s["end"], SECONDS),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=SECONDS), WIN, 0.5)
        duration = Counter()
        for s in reference:
            duration[s["speaker"]] += s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sorted(k for k, v in duration.items() if v / grand >= 0.05)
        if len(emb) < 2:
            continue
        win = np.full(len(times), WIN)
        cells = []
        for name, cap, unlock in RULES:
            best = select(build(emb, cap, unlock), unlock)
            named = map_by_frames(reference, times, best["labels"], win)
            value = frame_accuracy(reference, times, named, win)["accuracy"]
            accuracy[name].append(value)
            cells.append(f"k={best['k']}({value * 100:4.1f}%)")
        print(f"{meeting:<16}{len(effective):>6} | " + "".join(f"{c:>16}" for c in cells))

    print()
    base = np.array(accuracy["现状"])
    for name, _, _ in RULES:
        values = np.array(accuracy[name])
        delta = values - base
        print(f"  {name:<14} 帧级均值 {values.mean() * 100:5.1f}%  "
              f"最差 {values.min() * 100:5.1f}%  净 {delta.mean() * 100:+5.1f} 点  "
              f"赢{int((delta > 0.01).sum())}/亏{int((delta < -0.01).sum())}")


if __name__ == "__main__":
    main()
