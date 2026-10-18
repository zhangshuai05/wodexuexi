# -*- coding: utf-8 -*-
"""拆开"再拆轮廓系数"那一步：+6.8 点到底是谁挣的。

`probe_selector_locks.py` 的 `unlock >= 3` 同时改了三件事：

  A 放宽合格判定（尺寸门只对 >=2 窗的簇生效，份额门同样）
  B 轮廓系数只对 >=2 窗的簇计算（孤窗不参与）—— 拆锁 3 的第一道拦截
  C 去掉 `_select_candidate` 末尾的 `best["score"] > 0.0` 回退 —— 拆锁 3 的第二道

三者捆在一起报出 +6.8 帧级点 / -6.6 DER 点。但三件事的代价不同：
A 会放进更多候选（含碎簇风险），C 会让"分数为负"的候选也胜出。要提改动就必须知道
是哪一件在起作用，否则可能引入一个只是"偏爱高 k"的规则。

本脚本固定 A、B、C 的开关组合，逐组报 选k / 帧级 / DER / 碎簇数。

    python scripts/evaluation/probe_selector_locks_split3.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_selector_locks import silhouette_core  # noqa: E402
from probe_selector_locks_der import predicted_rttm  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0

# (放宽合格判定, 孤窗容忍轮廓系数, 去掉 score>0 回退)
VARIANTS = [
    ("现状", False, False, False),
    ("只拆B(轮廓)", False, True, False),
    ("只拆C(分数门)", False, False, True),
    ("拆B+C", False, True, True),
    ("拆A+B+C(=上版)", True, True, True),
    ("拆A+B(留分数门)", True, True, False),
]


def build(E, cap, relax_eligible, core_silhouette):
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
        if relax_eligible:
            big = [size for size in sizes if size >= 2]
            eligible = (len(big) >= 2 and min(big) >= v11.MIN_CLUSTER_WINDOWS
                        and min(big) / n >= v11.MIN_CLUSTER_SHARE)
        else:
            eligible = (min(sizes) >= v11.MIN_CLUSTER_WINDOWS
                        and min(sizes) / n >= v11.MIN_CLUSTER_SHARE)
        scorer = silhouette_core if core_silhouette else v11._silhouette
        score = scorer(E, labels) if eligible else -1.0
        candidates.append({"k": k, "labels": labels, "score": score,
                           "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                           "sizes": sizes, "eligible": eligible})
    return candidates


def select(candidates, drop_score_gate):
    eligible = [c for c in candidates if c["eligible"] and c["k"] >= 2]
    if not eligible:
        return candidates[0]
    best = max(eligible, key=lambda c: (c["objective"], -c["k"]))
    if drop_score_gate:
        return best
    return best if best["score"] > 0.0 else candidates[0]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    args = ap.parse_args()
    rows = []
    print(f"{'会议':<16}{'有效K':>6} | " + "".join(f"{n:>18}" for n, _, _, _ in VARIANTS))
    for meeting in split_meetings(ROOT_DATA, args.split):
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
        for name, relax, core, drop in VARIANTS:
            chosen = select(build(emb, 10 if relax else 4, relax, core), drop)
            named = map_by_frames(reference, times, chosen["labels"], win)
            accuracy = frame_accuracy(reference, times, named, win)["accuracy"]
            value = der(reference, predicted_rttm(times, chosen["labels"], WIN * 0.25),
                        collar=0.25)["der"]
            tiny = sum(1 for size in chosen["sizes"] if size == 1)
            rows.append({"meeting": meeting, "rule": name, "k": chosen["k"],
                         "accuracy": accuracy, "der": value, "sizes": chosen["sizes"],
                         "eff_k": len(effective), "tiny": tiny})
            cells.append(f"k={chosen['k']}({accuracy * 100:4.1f}/{value * 100:4.1f})")
        print(f"{meeting:<16}{len(effective):>6} | " + "".join(f"{c:>18}" for c in cells))

    print(f"\n{'变体':<18}{'帧级均值':>9}{'DER均值':>9}{'最差帧级':>9}{'碎簇场':>7}{'高估k':>7}")
    for name, _, _, _ in VARIANTS:
        part = [r for r in rows if r["rule"] == name]
        print(f"{name:<18}{np.mean([r['accuracy'] for r in part]) * 100:>8.1f}%"
              f"{np.mean([r['der'] for r in part]) * 100:>8.1f}%"
              f"{np.min([r['accuracy'] for r in part]) * 100:>8.1f}%"
              f"{sum(1 for r in part if r['tiny'] >= 2):>7}"
              f"{sum(1 for r in part if r['k'] > r['eff_k']):>7}")

    output = ROOT / "artifacts" / f"v13_selector_locks_split3_{args.split}.json"
    output.write_text(json.dumps({"schema": "v13-selector-locks-split3-v1", "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
