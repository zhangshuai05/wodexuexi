# -*- coding: utf-8 -*-
"""说话人时长平衡审计：塌陷是不是"一个人说了很久"造成的？

**起因**：`R8001_M8004` 与 `R8007_M8010` 两场预测塌陷成 1 个说话人（共 1785 句全归一人）。
用户给了一条领域判断：「有时候确实是只有一个人说话（一个人说了很久）」。

本脚本只用真值 RTTM（不跑模型、秒级），算每场每个说话人的**语音时长并集**
（并集而非求和：两人同时说话时不能重复计），再看两件事：

  1. **时长平衡**：最长说话人占比、最短说话人占比、最短/最长比
  2. **会不会被现有簇门当噪声**：`claude_a2_v11` 的门是"最小簇 >= 1.5% 窗数"，
     近似等于"该说话人占比 >= 1.5%"。占比更低的真人在现有判据下**必然被当离群簇**。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_speaker_time_balance.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, RESULTS, der_tools  # noqa: E402

_, read_rttm = der_tools()
COLLAPSED = {"R8001_M8004", "R8007_M8010"}      # 上一轮审计实测塌陷的两场
CLUSTER_GATE = 0.015                            # claude_a2_v11.MIN_CLUSTER_SHARE


def union_length(intervals):
    """区间并集总长（秒）。"""
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total, cur_s, cur_e = 0.0, ivs[0][0], ivs[0][1]
    for start, end in ivs[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    return total + (cur_e - cur_s)


def overlap_ratio(segments, step=0.01):
    """重叠语音占语音总时长的比例（同时 >=2 人发声的帧 / 有语音的帧）。"""
    import numpy as np
    if not segments:
        return 0.0
    end = max(s["end"] for s in segments)
    n = int(end / step) + 2
    cnt = np.zeros(n, dtype=np.int16)
    for seg in segments:
        i0, i1 = int(seg["start"] / step), int(seg["end"] / step)
        if i1 > i0:
            cnt[i0:i1] += 1
    active = int((cnt > 0).sum())
    return float((cnt > 1).sum()) / active if active else 0.0


def main():
    rows = []
    print("%-14s %5s %9s %9s %9s %8s %9s %s"
          % ("meeting", "人数", "语音总时长", "最长占比", "最短占比", "最短/最长",
             "重叠率", "会被 1.5% 门当噪声的人"),
          flush=True)
    for path in sorted(ALI_RTTM.glob("*.rttm")):
        meeting = path.stem
        segs = read_rttm(path)
        per = defaultdict(list)
        for seg in segs:
            per[seg["speaker"]].append((seg["start"], seg["end"]))
        durations = {spk: union_length(ivs) for spk, ivs in per.items()}
        total = union_length([(s["start"], s["end"]) for s in segs])
        shares = {spk: d / total for spk, d in durations.items()}
        longest = max(shares.values())
        shortest = min(shares.values())
        ov = overlap_ratio(segs)
        tiny = sorted(spk for spk, r in shares.items() if r < CLUSTER_GATE)
        rows.append({"meeting": meeting, "n_speakers": len(shares),
                     "total_seconds": round(total, 1),
                     "longest_share": longest, "shortest_share": shortest,
                     "ratio": shortest / longest if longest else None,
                     "overlap_ratio": ov,
                     "shares": {k: round(v, 4) for k, v in shares.items()},
                     "below_gate": tiny, "collapsed": meeting in COLLAPSED})
        print("%-14s %5d %8.0fs %8.1f%% %8.1f%% %9.2f %8.1f%%  %s"
              % (meeting, len(shares), total, longest * 100, shortest * 100,
                 (shortest / longest if longest else 0), ov * 100,
                 ("%d 人: %s" % (len(tiny), ",".join(s.split("_")[-1] for s in tiny)))
                 if tiny else "无"),
              flush=True)

    print("\n塌陷场 vs 正常场（关键对比）：")
    import statistics as st
    for label, group in (("塌陷", [r for r in rows if r["collapsed"]]),
                         ("正常", [r for r in rows if not r["collapsed"]])):
        if not group:
            continue
        print("  %-4s（%d 场）: 重叠率 %.1f%%  最长占比 %.1f%%  最短占比 %.1f%%  最短/最长 %.2f  门下真人 %d/%d"
              % (label, len(group),
                 st.mean([r["overlap_ratio"] for r in group]) * 100,
                 st.mean([r["longest_share"] for r in group]) * 100,
                 st.mean([r["shortest_share"] for r in group]) * 100,
                 st.mean([r["ratio"] for r in group]),
                 sum(1 for r in group if r["below_gate"]), len(group)))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "speaker_time_balance.json"
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval", "cluster_gate": CLUSTER_GATE,
                   "note": "时长用区间并集（同时说话不重复计）；门下真人 = 占比 < 1.5%"},
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 %s" % out)


if __name__ == "__main__":
    main()
