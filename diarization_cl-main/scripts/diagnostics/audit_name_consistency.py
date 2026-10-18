# -*- coding: utf-8 -*-
"""检查 TextGrid tier 名与 RTTM speaker 名是否同源。

**为什么要查**：算"句内重叠率"时要排除"别人"的语音段——
`[s for s in segments if s["speaker"] != turn["speaker"]]`。
若两边的名字不同源（比如 tier 叫 `N_SPK8054` 而 RTTM 叫 `spk-8054`），
这个比较会**恒真**，"别人"就包含了该说话人自己，重叠率被算到接近 100%，
锚点判定随之全灭（实测症状：R8007_M8010 的锚点只剩 27/1181）。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_name_consistency.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr",
           REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, der_tools  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)

_, read_rttm = der_tools()


def main():
    bad = 0
    for path in sorted(ALI_TG.glob("*.TextGrid")):
        meeting = path.stem
        rttm = ALI_RTTM / f"{meeting}.rttm"
        if not rttm.exists():
            continue
        turns = sentences_from_textgrid(read_textgrid(path))
        segs = read_rttm(rttm)
        tier_names = {t["speaker"] for t in turns}
        rttm_names = {s["speaker"] for s in segs}
        only_tier = sorted(tier_names - rttm_names)
        only_rttm = sorted(rttm_names - tier_names)
        flag = "OK" if not (only_tier or only_rttm) else "**不同源**"
        if only_tier or only_rttm:
            bad += 1
        print("%-14s tier=%-2d rttm=%-2d 交集=%-2d  %s"
              % (meeting, len(tier_names), len(rttm_names),
                 len(tier_names & rttm_names), flag))
        if only_tier or only_rttm:
            print("    仅 tier: %s" % only_tier[:4])
            print("    仅 rttm: %s" % only_rttm[:4])
    print("\n小结：%d 场名字不同源（>0 会让句内重叠率被算废）" % bad)


if __name__ == "__main__":
    main()
