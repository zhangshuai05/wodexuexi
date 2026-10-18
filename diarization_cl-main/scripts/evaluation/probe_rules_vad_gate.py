# -*- coding: utf-8 -*-
"""真实 VAD 门下的规则扫描——本会话所有离线扫描都做错了门，这是重做。

此前 `probe_selector_locks*` 全部在 **RTTM 门窗集**上跑（参考语音并集，
`_v13_spike/cache/w2_h1.npz` 的原始窗），而 `v12.diarize()` 用的是 **FSMN VAD**
门（20 场实测只保留 81% 的窗）。两者不但是不同大小，候选表的形态也不同：

    L_R003S01C02   RTTM 门 [375,124,63]      VAD 门 [296,121,1]
    L_R003S02C02   RTTM 门 [514,24,7]        VAD 门 [273,3,2]

所以"上限该取多少、min_win 该取多少"必须在 **VAD 门**上定。
VAD 区间已缓存于 `cache/vad/<meeting>.json`（20 场全），本脚本零推理。

    python scripts/evaluation/probe_rules_vad_gate.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames, rttm_speech,
    window_reference)
from probe_selector_locks import silhouette_core  # noqa: E402
from probe_selector_locks_der import predicted_rttm  # noqa: E402

ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
VAD_CACHE = CACHE / "vad"
WIN, HOP, SECONDS = 2.0, 1.0, 600.0
MIN_VAD = v11.MIN_VAD_FRACTION


def build(E, cap, min_win, min_share, sliver_out, core):
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n < 2:
        return []
    tree = linkage(E, method="average", metric="cosine")
    out = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
            "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == l).sum()) for l in range(k)]
        pool = [s for s in sizes if s >= 2] if sliver_out else sizes
        eligible = (len(pool) >= 2 and min(pool) >= min_win and min(pool) / n >= min_share)
        if core and eligible:
            score = silhouette_core(E, labels)
        elif eligible:
            score = v11._silhouette(E, labels)
        else:
            score = -1.0
        out.append({"k": k, "labels": labels, "score": score,
                    "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                    "sizes": sizes, "eligible": eligible})
    return out


def pick(cands, mode):
    eligible = [c for c in cands if c["eligible"] and c["k"] >= 2]
    if mode == "largest":
        return max(cands, key=lambda c: c["k"])
    if not eligible:
        return cands[0]
    if mode == "largest_eligible":
        return max(eligible, key=lambda c: c["k"])
    return max(eligible, key=lambda c: (c["objective"], -c["k"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    data = {}
    for split in ("dev", "holdout"):
        for meeting in split_meetings(ROOT_DATA, split):
            vad_file = VAD_CACHE / f"{meeting}.json"
            if not vad_file.exists():
                continue
            reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                         if s["start"] < args.seconds]
            reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                          "speaker": s["speaker"]} for s in reference]
            if not reference:
                continue
            emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
            intervals = [(a, b) for a, b in
                         json.loads(vad_file.read_text(encoding="utf-8"))["intervals"]]
            emb, times = gate_windows(emb_all, times_all, intervals, WIN, MIN_VAD)
            if len(emb) >= 2:
                data[(split, meeting)] = (emb, times, reference)
    print(f"  VAD 门下纳入 {len(data)} 场（dev {sum(1 for k in data if k[0]=='dev')} / "
          f"holdout {sum(1 for k in data if k[0]=='holdout')}）\n", flush=True)

    def run(label, **kw):
        mode = kw.pop("mode", "objective")
        per = {}
        for (split, meeting), (emb, times, reference) in data.items():
            cands = build(emb, kw["cap"], kw["min_win"], kw["min_share"],
                          kw["sliver_out"], kw["core"])
            chosen = pick(cands, mode)
            win = np.full(len(times), WIN)
            value = der(reference, predicted_rttm(times, chosen["labels"], WIN * 0.25),
                        collar=0.25)["der"]
            acc = frame_accuracy(reference, times,
                                 map_by_frames(reference, times, chosen["labels"], win),
                                 win)["accuracy"]
            per[(split, meeting)] = (value, acc, chosen["k"])
        return per

    rows = []
    print(f"{'规则':<34}{'dev均':>8}{'hold均':>9}{'全20均':>9}{'中位':>8}{'最差':>8}"
          f"{'帧级均':>8}{'平均k':>7}")
    configs = [
        ("现状 v11（无门，cap4）", dict(cap=4, min_win=4, min_share=0.015,
                                        sliver_out=False, core=False), "objective"),
        ("cap8 + sliver_out", dict(cap=8, min_win=2, min_share=0.0,
                                   sliver_out=True, core=True), "objective"),
        ("cap10 + sliver_out", dict(cap=10, min_win=2, min_share=0.0,
                                    sliver_out=True, core=True), "objective"),
        ("cap8，取最大**合格**k", dict(cap=8, min_win=2, min_share=0.0,
                                       sliver_out=True, core=True), "largest_eligible"),
        ("cap8，取最大 k（无门）", dict(cap=8, min_win=2, min_share=0.0,
                                        sliver_out=True, core=True), "largest"),
        ("cap10，取最大 k（无门）", dict(cap=10, min_win=2, min_share=0.0,
                                         sliver_out=True, core=True), "largest"),
    ]
    for label, kw, mode in configs:
        per = run(label, mode=mode, **kw)
        dv = np.array([v for (s, _), (v, _, _) in per.items() if s == "dev"])
        hv = np.array([v for (s, _), (v, _, _) in per.items() if s == "holdout"])
        av = np.concatenate([dv, hv])
        ac = np.array([a for _, (_, a, _) in per.items()])
        ks = np.array([k for _, (_, _, k) in per.items()])
        rows.append({"rule": label, "mode": mode, **kw,
                     "dev": float(dv.mean()), "holdout": float(hv.mean()),
                     "all": float(av.mean()), "median": float(np.median(av)),
                     "worst": float(av.max()), "frame": float(ac.mean()),
                     "mean_k": float(ks.mean()),
                     "per_meeting": {f"{s}/{m}": {"der": v, "frame": a, "k": k}
                                     for (s, m), (v, a, k) in per.items()}})
        print(f"{label:<34}{dv.mean()*100:>7.1f}%{hv.mean()*100:>8.1f}%"
              f"{av.mean()*100:>8.1f}%{np.median(av)*100:>7.1f}%{av.max()*100:>7.1f}%"
              f"{ac.mean()*100:>7.1f}%{ks.mean():>7.1f}", flush=True)

    print("\n  对比（本会话此前在 RTTM 门上得到的数，不可直接比）：")
    print("    RTTM 门 现状 24.4% / cap10+sliver_out 13.6% / 取最大k@cap8 10.4%")

    output = ROOT / "artifacts" / "v13_rules_vad_gate.json"
    output.write_text(json.dumps({"schema": "v13-rules-vad-gate-v1", "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
