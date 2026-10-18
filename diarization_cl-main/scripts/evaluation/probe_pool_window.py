# -*- coding: utf-8 -*-
"""把 95.8% ↔ 50.9% 的差归因：窗集纯度、取样密度，还是窗数？

这两个数此前不可比——它们同时改了三个变量。这里用 `build_cache.py` 落的窗池
（每一个 VAD 门控窗 + 窗内多数说话人 + `votes` = 多数说话人占窗内 RTTM 语音的比例）
一次只动一个，不需要任何 embedding 推理：

    python scripts/evaluation/probe_pool_window.py --meeting M_R003S04C01

`votes` 是"这个窗有多干净"的直接读数：=1.0 表示窗内只有一位参考说话人。
所以纯度轴由 votes 阈值扫出来，密度轴在每一档里等距抽到同一个每说话人配额。
K 固定为有效说话人数，判定函数用库里的 `_candidate_clusters` + `_select_candidate`。

注意：缓存是**整场**的（10s..场末），不是 600s 截断；读数与缓存的 22.79% 口径同源。
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
from v12_aishell4 import read_rttm  # noqa: E402


def load_pool(cache_dir, meeting, window):
    path = Path(cache_dir) / meeting / f"w{window:g}_h{window / 2:g}.npz"
    if not path.exists():
        raise SystemExit(f"没有窗池缓存：{path}\n先跑 _v13_spike/build_cache.py")
    data = np.load(path, allow_pickle=False)
    return data["emb"], data["times"], data["speakers"].astype(str), data["votes"]


def thin(indices, quota):
    """等距抽到配额（基准口径的位置分布）。"""
    if len(indices) <= quota:
        return list(indices)
    return [int(indices[i]) for i in
            np.linspace(0, len(indices) - 1, quota, dtype=int)]


def majority_hit(predicted, labels):
    """每簇取多数票当命中 —— 与库里 oracle 口径一致。"""
    hit = np.zeros(len(labels), dtype=bool)
    for cluster in set(predicted.tolist()):
        mask = predicted == cluster
        if mask.any():
            top = Counter(labels[mask].tolist()).most_common(1)[0][0]
            hit[mask] = labels[mask] == top
    return hit


def score(vectors, labels, k):
    if k > v11.MAX_SPEAKERS:
        # `_candidate_clusters` caps at MAX_SPEAKERS, so forced_k beyond it finds
        # no candidate and `_select_candidate` silently falls through to auto-K.
        # That once produced a fake 100% (accidental gender split at k=2).
        raise SystemExit(f"要求 K={k} 超过 MAX_SPEAKERS={v11.MAX_SPEAKERS}，"
                         "forced_k 会静默退化；要么调大 cap，要么换会议")
    candidates, _ = v11._candidate_clusters(vectors, max_speakers=k)
    if not any(item["k"] == k for item in candidates):
        raise SystemExit(f"层级里没有 k={k}（只有 {[c['k'] for c in candidates]}）")
    chosen = v11._select_candidate(candidates, forced_k=k)
    labels = np.asarray(labels)
    hit = majority_hit(chosen["labels"], labels)
    # 每说话人内部的参考时长不可得，用该说话人在**本档池内**的窗数当权重代理。
    counts = Counter(labels.tolist())
    weight = np.array([1.0 / counts[name] for name in labels])
    weight = weight / weight.sum()
    return hit.mean(), float(hit @ weight), chosen["k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--meeting", default="M_R003S04C01")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--share", type=float, default=0.05,
                    help="有效说话人门槛：占参考语音时长的比例")
    ap.add_argument("--quota", type=int, default=8, help="evenly 每说话人配额")
    ap.add_argument("--votes", type=float, nargs="+",
                    default=[0.5, 0.75, 0.9, 1.0],
                    help="纯度档：多数说话人占窗内语音的最低比例")
    args = ap.parse_args()

    root = Path(args.root)
    reference = read_rttm(root / "TextGrid" / f"{args.meeting}.rttm")
    seconds = {}
    for item in reference:
        seconds[item["speaker"]] = seconds.get(item["speaker"], 0.0) + item["end"] - item["start"]
    grand = sum(seconds.values())
    speakers = {name for name, value in seconds.items() if value / grand >= args.share}
    print(f"{args.meeting}  参考 {len(seconds)} 人，有效(>={args.share:.0%}) {len(speakers)} 人")
    print("有效人数占比：" + "  ".join(
        f"{n}={seconds[n] / grand:.1%}" for n in sorted(speakers, key=lambda n: -seconds[n])))

    emb, times, names, votes = load_pool(args.cache, args.meeting, args.window)
    print(f"\n窗池 {len(times)} 窗（{args.window}s 窗 / {args.window / 2}s 步，"
          f"整场 {times[0]:.0f}..{times[-1] + args.window:.0f}s）")

    print(f"\n{'votes阈':<9}{'位置':<12}{'窗数':>5}{'K':>3}{'逐窗acc':>10}{'按时长acc':>11}",
          flush=True)
    for threshold in sorted(args.votes):
        keep = [index for index in range(len(times))
                if names[index] in speakers and votes[index] >= threshold]
        by_speaker = {}
        for index in keep:
            by_speaker.setdefault(names[index], []).append(index)
        for placement in ("evenly", "continuous"):
            if placement == "evenly":
                order = [index for name in sorted(by_speaker)
                         for index in thin(by_speaker[name], args.quota)]
            else:
                order = keep
            labels = [names[index] for index in order]
            if len(set(labels)) < 2:
                print(f"{threshold:<9.2f}{placement:<12}{len(order):>5}  —— 说话人不足，跳过")
                continue
            plain, weighted, k = score(emb[order], labels, len(speakers))
            print(f"{threshold:<9.2f}{placement:<12}{len(order):>5}{k:>3}"
                  f"{plain * 100:>9.1f}%{weighted * 100:>10.1f}%", flush=True)


if __name__ == "__main__":
    main()
