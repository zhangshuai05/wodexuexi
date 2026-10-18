# -*- coding: utf-8 -*-
"""把所有臂的预测统一用主办方 `md-eval-22.pl` 评一遍。

臂的预测文件名形如 `{meeting}.{vad}_{cand}.full.rttm`（`run_arms.py` 产出），
外加历史臂 `{meeting}.v12.full.rttm`（无猴补丁的生产现状）与
`{meeting}.baseline.full.rttm`（穷举 K 基线）。

同时给出 der.py 口径与官方口径两列 —— 两个都要报，只说一个就是在藏口径。

    python score_all_arms.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from common import ALI_RTTM, PREDICTIONS, RESULTS, der_tools
from md_eval_official import OFFICIAL, build_combined, parse_all, run_mdeval

der, read_rttm = der_tools()
COLLAR = 0.25


def discover_arms():
    """从 predictions/ 里发现所有臂（去掉 meeting 前缀与 .full.rttm 后缀）。"""
    arms = set()
    for path in PREDICTIONS.glob("*.full.rttm"):
        stem = path.name[: -len(".full.rttm")]
        for meeting in sorted((p.stem for p in ALI_RTTM.glob("*.rttm")),
                              key=len, reverse=True):
            if stem == meeting:
                arms.add("v12")
                break
            if stem.startswith(meeting + "."):
                arms.add(stem[len(meeting) + 1:])
                break
    return sorted(arms)


def arm_predictions(arm, meetings):
    return {m: PREDICTIONS / f"{m}.{arm}.full.rttm" for m in meetings}


def der_py_style(predictions, meetings):
    values = []
    for meeting in meetings:
        reference = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        values.append(der(reference, read_rttm(predictions[meeting]), collar=COLLAR)["der"])
    return float(np.mean(values))


def main():
    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    arms = discover_arms()
    if not arms:
        raise SystemExit("predictions/ 里没有 *.full.rttm，先跑 run_arms.py")

    import md_eval_official as M

    rows = {}
    for arm in arms:
        predictions = arm_predictions(arm, meetings)
        missing = [m for m, p in predictions.items() if not p.exists()]
        if missing:
            print(f"  [skip] {arm}: 缺 {len(missing)} 场")
            continue
        ref_lines, sys_lines, uem_lines = build_combined(meetings, arm)
        (M.WORK / "ref.rttm").write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
        (M.WORK / "sys.rttm").write_text("\n".join(sys_lines) + "\n", encoding="utf-8")
        (M.WORK / "all.uem").write_text("\n".join(uem_lines) + "\n", encoding="utf-8")
        official, counts = parse_all(run_mdeval(COLLAR))
        rows[arm] = {
            "der_py_style": der_py_style(predictions, meetings),
            "der_official": official,
            "counts": counts,
        }
        scored = counts.get("SCORED SPEAKER TIME", 1.0)
        rows[arm]["rates"] = {
            "missed": counts.get("MISSED SPEAKER TIME", 0.0) / scored,
            "false_alarm": counts.get("FALARM SPEAKER TIME", 0.0) / scored,
            "speaker_error": counts.get("SPEAKER ERROR TIME", 0.0) / scored,
        }

    print(f"\n{'=' * 96}")
    print("全臂对比（AliMeeting Eval 8 场整场，ch0 单通道，collar 0.25）")
    print(f"{'=' * 96}")
    print(f"{'臂':28s} {'der.py':>8s} {'官方口径':>9s} {'MISSED':>8s} {'FALARM':>8s} "
          f"{'SPKRERR':>8s}  {'对官基线差':>10s}")
    base = OFFICIAL["baseline_eval_far_collar0.25"]
    for arm, row in sorted(rows.items(), key=lambda x: x[1]["der_official"]):
        r = row["rates"]
        print(f"{arm:28s} {row['der_py_style'] * 100:7.2f}% "
              f"{row['der_official']:8.2f}% {r['missed'] * 100:7.2f}% "
              f"{r['false_alarm'] * 100:7.2f}% {r['speaker_error'] * 100:7.2f}%  "
              f"{row['der_official'] - base:+9.2f}")
    print(f"\n官基线（ch0 单通道 + oracle VAD）= {base:.2f}%")

    out = RESULTS / "arms_official_scored.json"
    out.write_text(json.dumps({"collar": COLLAR, "official_baseline": base,
                               "arms": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
