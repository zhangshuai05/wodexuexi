# -*- coding: utf-8 -*-
"""审计"句"的结构：span 分布、活跃人数、以及重叠率算得对不对。

**起因**：`truth_overlap_ratio` 在 R8007_M8010 上给出"句重叠率中位 1.00、均值 0.89"，
但同一场按帧算的总重叠率只有 57.4% —— 两者矛盾，说明"句"这个单位的结构有问题
（最可能是 TextGrid 的 interval 是**轮次级**长段，不是句级）。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_turn_structure.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr",
           REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, der_tools  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)

_, read_rttm = der_tools()
MEETINGS = ["R8007_M8010", "R8008_M8013", "R8009_M8020"]


def active_count(segments, step=0.01):
    """每 10ms 的活跃说话人数。"""
    end = max(s["end"] for s in segments)
    n = int(end / step) + 2
    cnt = np.zeros(n, dtype=np.int16)
    for seg in segments:
        i0, i1 = int(seg["start"] / step), int(seg["end"] / step)
        if i1 > i0:
            cnt[i0:i1] += 1
    return cnt


for meeting in MEETINGS:
    turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
    segs = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    spans = np.array([t["end"] - t["start"] for t in turns])
    cnt = active_count(segs)
    active = cnt > 0
    overlap_frame = (cnt > 1).sum() / max(active.sum(), 1)
    print("=== %s ===" % meeting)
    print("  TextGrid 句子数 %d   RTTM 段数 %d" % (len(turns), len(segs)))
    print("  句 span: 中位 %.2fs  均值 %.2fs  min %.2f  max %.2f  >10s 的占 %.0f%%"
          % (np.median(spans), spans.mean(), spans.min(), spans.max(),
             100.0 * (spans > 10).mean()))
    print("  按帧算的总重叠率: %.1f%%" % (100 * overlap_frame))

    # 用 RTTM 段本身当"句"再算一次，对比
    seg_spans = np.array([s["end"] - s["start"] for s in segs])
    print("  RTTM 段 span: 中位 %.2fs  均值 %.2fs" % (np.median(seg_spans), seg_spans.mean()))

    # 每个 tier 贡献多少 interval
    tiers = read_textgrid(ALI_TG / f"{meeting}.TextGrid")
    print("  每 tier 的 interval 数: %s"
          % {k: len(v) for k, v in list(tiers.items())[:5]})

    # 抽样：取一个句子，看它区间内"别人"覆盖了多少
    for turn in sorted(turns, key=lambda x: -(x["end"] - x["start"]))[:2]:
        others = [s for s in segs if s["speaker"] != turn["speaker"]]
        step = 0.02
        t = turn["start"]
        cov = 0.0
        while t < turn["end"]:
            if any(o["start"] <= t < o["end"] for o in others):
                cov += step
            t += step
        print("    最长句 %s %.1f-%.1f (%.1fs): 别人覆盖 %.0f%%"
              % (turn["speaker"], turn["start"], turn["end"],
                 turn["end"] - turn["start"], 100 * cov / max(turn["end"] - turn["start"], 1e-9)))
    print()
