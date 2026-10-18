# -*- coding: utf-8 -*-
"""按参考人数分档，与官方基线的 Table 3 逐档对位。

官方基线（arXiv:2110.07393 Table 3，Eval-Ali-far，collar 0.25）：
    2 人 4.84%   3 人 10.90%   4 人 23.18%   合计 15.24%

只看总分 15.24% vs 33.09% 会丢掉关键信息：差距是均匀摊在每一档上，
还是集中在某一档？这决定接下来该修什么。

复用 `md_eval_official.py` 的拼接/调用/解析，保证与总分同一把尺子。

    python compare_by_speaker_count.py
"""

import json
import sys

from common import ALI_RTTM, PROJECT, RESULTS, der_tools
from md_eval_official import (MDEVAL, WORK, build_combined, parse_all,
                              run_mdeval)

_, read_rttm = der_tools()

# 官方基线 Table 3（Eval-Ali-far，collar 0.25）
OFFICIAL_BY_COUNT = {2: 4.84, 3: 10.90, 4: 23.18, "all": 15.24}
COLLAR = 0.25


def speaker_count(meeting):
    return len({s["speaker"] for s in read_rttm(ALI_RTTM / f"{meeting}.rttm")})


def score_group(meetings, arm, collar=COLLAR):
    ref_lines, sys_lines, uem_lines = build_combined(meetings, arm)
    (WORK / "ref.rttm").write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
    (WORK / "sys.rttm").write_text("\n".join(sys_lines) + "\n", encoding="utf-8")
    (WORK / "all.uem").write_text("\n".join(uem_lines) + "\n", encoding="utf-8")
    value, counts = parse_all(run_mdeval(collar))
    return value, counts


def main():
    if not MDEVAL.exists():
        raise SystemExit(f"缺 {MDEVAL}")
    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    groups = {}
    for meeting in meetings:
        groups.setdefault(speaker_count(meeting), []).append(meeting)

    print("按参考人数分档（官方 md-eval-22.pl，collar 0.25，ch0 单通道）")
    print(f"{'人数':>4s} {'场数':>4s} {'官基线':>8s} {'我们v12':>9s} {'差距':>9s} "
          f"{'倍数':>6s}   场次")
    report = {"official_by_count": OFFICIAL_BY_COUNT, "groups": {}}
    all_meetings = []
    for count in sorted(groups):
        subset = groups[count]
        all_meetings += subset
        ours, counts = score_group(subset, "v12")
        base = OFFICIAL_BY_COUNT[count]
        report["groups"][count] = {
            "meetings": subset, "ours_der": ours, "official_der": base,
            "counts": counts}
        print(f"{count:>4d} {len(subset):>4d} {base:7.2f}% {ours:8.2f}% "
              f"{ours - base:+8.2f} {ours / base:5.2f}x   {', '.join(subset)}")

    ours_all, counts_all = score_group(all_meetings, "v12")
    base_all = OFFICIAL_BY_COUNT["all"]
    report["all"] = {"ours_der": ours_all, "official_der": base_all, "counts": counts_all}
    print(f"{'合计':>4s} {len(all_meetings):>4d} {base_all:7.2f}% {ours_all:8.2f}% "
          f"{ours_all - base_all:+8.2f} {ours_all / base_all:5.2f}x")

    print("\n差距贡献分解（把总差距按档摊开看，谁是大头）")
    total_gap = ours_all - base_all
    print(f"  总差距 {total_gap:+.2f} pts（={ours_all:.2f}% - {base_all:.2f}%）")
    for count in sorted(groups):
        group = report["groups"][count]
        # 该档对总差距的贡献：本档差距 × 本档在总体里的权重。
        # 用参考说话人时间做权重最贴近 md-eval 的计分方式，而不是简单按场次平均。
        weight = group["counts"].get("SCORED SPEAKER TIME", 0.0)
        share = weight / counts_all.get("SCORED SPEAKER TIME", 1.0)
        contribution = (group["ours_der"] - group["official_der"]) * share
        print(f"  {count} 人档: 本档差 {group['ours_der'] - group['official_der']:+6.2f} pts"
              f" × 权重 {share * 100:4.1f}% = 对总差距贡献 {contribution:+6.2f} pts"
              f"（{(group['ours_der'] - group['official_der']) * share / total_gap * 100:4.0f}%）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "compare_by_speaker_count.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
