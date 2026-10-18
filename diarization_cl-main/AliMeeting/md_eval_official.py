# -*- coding: utf-8 -*-
"""用官方 `md-eval-22.pl` 评我们的预测 —— 不重写，直接跑主办方那个脚本。

为什么必须这样：M2MeT 官基线用的就是 AliMeeting 仓库里 `speaker/dscore/` 的
`md-eval-22.pl`（`dscore/metrics.py` 里 `MDEVAL_BIN` 指向它）。任何"我们自己复刻一个
md-eval"都会引入新的口径风险——而这次要回答的问题恰恰就是"差多少"，
口径必须零误差。

关键差异（读脚本源码得到，非推测）：

    md-eval-22.pl 第 1967-1994 行
      SCORED_SPEECH  += nref ? dur : 0        # 并集口径
      SCORED_SPEAKER += dur * nref            # <- 按活跃参考说话人数计权重
      MISSED_SPEAKER += dur * max(nref-nsys,0)
      FALARM_SPEAKER += dur * max(nsys-nref,0)
      SPEAKER_ERROR  += dur * (min(nref,nsys) - nmap)
      DER = (MISSED + FALARM + SPEAKER_ERROR) / SCORED_SPEAKER

    第 2414 行：printf " OVERALL SPEAKER DIARIZATION ERROR = %.2f percent of
                        scored speaker time"

    也就是说 **md-eval 的分母按"活跃说话人数"计权重**，重叠帧算多次。
    主仓 `streaming_asr/der.py` 的 docstring 写的是"md-eval 用并集口径、
    按 len(R) 计权重的那版不可比"——**与源码相反**。所以 der.py 的 35.39%
    与官方的 15.24% 不在同一把尺子上，必须用本脚本重测。

调用（与 dscore 完全一致）：
    md-eval-22.pl -af -r ref.rttm -s sys.rttm -c <collar> -u <uem>
`-f`  = 用最优说话人映射（论文 §3.1 说的 Hungarian）
UEM  = 按 dscore `uem.gen_uem`：每文件取 [min(onset), max(offset)] of ref∪sys

    python md_eval_official.py
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import ALI_RTTM, PREDICTIONS, PROJECT, RESULTS, der_tools, resolve_wav

_, read_rttm = der_tools()

MDEVAL = PROJECT / "md-eval-22.pl"
WORK = PROJECT / "results" / "_mdeval_work"
COLLARS = (0.25, 0.0)
ARMS = ("v12", "baseline")
# `-1` = 忽略参考重叠区（md-eval 的 ignore_overlaps）。官方**没有**开这个开关，
# 所以它不是可引用的成绩，只用来回答"我们那 14.52% 的 MISSED 里有多少来自重叠"。
IGNORE_OVERLAPS = True

# 官方论文数字（arXiv:2202.03647 Table 1 / arXiv:2110.07393 Table 3）
OFFICIAL = {
    "baseline_eval_far_collar0.25": 15.24,
    "baseline_eval_far_collar0.0": 24.52,
    "royalflush_fusion_eval_collar0.25": 6.30,
    "winner_A41_test_collar0.25": 2.98,
}


def to_rttm_line(file_id, seg, speaker):
    return (f"SPEAKER {file_id} 1 {seg['start']:.4f} "
            f"{seg['end'] - seg['start']:.4f} <NA> <NA> {speaker} <NA> <NA>")


def build_combined(meetings, arm):
    """把 8 场拼成一份 ref / sys RTTM，并生成 UEM。

    拼成一份而不是逐场跑，是为了拿 md-eval 的 `ALL` 行——它跨文件汇总计数
    （micro），与论文报的 15.24% 可比。逐场再平均（macro）是另一个数，两个都出。
    """
    ref_lines, sys_lines, uem_lines = [], [], []
    for meeting in meetings:
        reference = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        predicted = read_rttm(PREDICTIONS / f"{meeting}.{arm}.full.rttm")
        ref_lines += [to_rttm_line(meeting, s, s["speaker"]) for s in reference]
        sys_lines += [to_rttm_line(meeting, s, s["speaker"]) for s in predicted]
        # dscore uem.gen_uem：onset = min(ref onset, sys onset)，offset = max(ref offset, sys offset)
        onsets = [s["start"] for s in reference] + [s["start"] for s in predicted]
        offsets = [s["end"] for s in reference] + [s["end"] for s in predicted]
        uem_lines.append(f"{meeting} 1 {min(onsets):.4f} {max(offsets):.4f}")
    return ref_lines, sys_lines, uem_lines


def run_mdeval(collar, ignore_overlaps=False):
    cmd = ["perl", str(MDEVAL), "-af",
           "-r", "ref.rttm", "-s", "sys.rttm",
           "-c", str(collar), "-u", "all.uem"]
    if ignore_overlaps:
        cmd.append("-1")
    proc = subprocess.run(cmd, cwd=str(WORK), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 and "DIARIZATION ERROR" not in out:
        raise RuntimeError(f"md-eval 失败 rc={proc.returncode}\n{out[:800]}")
    return out


def parse_all(output):
    """取 `ALL` 条件那一行的 DER（跨文件汇总）。"""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if "OVERALL SPEAKER DIARIZATION ERROR" in line and "`(ALL)" in line:
            value = float(re.search(r"=\s*([\d.]+)", line).group(1))
            counts = {}
            for back in range(index, max(0, index - 8), -1):
                match = re.match(r"\s*([A-Z][A-Z ]+?)\s*=\s*([\d.]+)", lines[back])
                if match:
                    counts[match.group(1).strip()] = float(match.group(2))
            return value, counts
    # 只有单个文件时没有 ALL 行，退回到唯一的那个 DER
    match = re.search(r"OVERALL SPEAKER DIARIZATION ERROR = ([\d.]+)", output)
    if match:
        return float(match.group(1)), {}
    raise RuntimeError(f"没解析到 DER：\n{output[:800]}")


def main():
    if not MDEVAL.exists():
        raise SystemExit(f"缺 {MDEVAL}")
    if shutil.which("perl") is None:
        raise SystemExit("没找到 perl；md-eval-22.pl 需要 Perl 才能跑")

    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    missing = [m for m in meetings
               for arm in ARMS if not (PREDICTIONS / f"{m}.{arm}.full.rttm").exists()]
    if missing:
        raise SystemExit(f"预测缺失：{sorted(set(missing))}，先跑 run_der.py / run_baseline.py")

    WORK.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in ARMS:
        ref_lines, sys_lines, uem_lines = build_combined(meetings, arm)
        (WORK / "ref.rttm").write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
        (WORK / "sys.rttm").write_text("\n".join(sys_lines) + "\n", encoding="utf-8")
        (WORK / "all.uem").write_text("\n".join(uem_lines) + "\n", encoding="utf-8")
        for collar in COLLARS:
            der_value, counts = parse_all(run_mdeval(collar))
            results[f"{arm}|md-eval|collar{collar}"] = {"der": der_value, "counts": counts}
            print(f"  {arm:9s} collar={collar:<5} md-eval ALL DER = {der_value:6.2f}%"
                  f"   (SCORED SPEAKER = {counts.get('SCORED SPEAKER TIME', float('nan')):.1f}s)")
        if IGNORE_OVERLAPS:
            der_value, counts = parse_all(run_mdeval(0.25, ignore_overlaps=True))
            results[f"{arm}|md-eval|collar0.25|ignore_overlaps"] = {
                "der": der_value, "counts": counts}
            print(f"  {arm:9s} collar=0.25 -1 忽略重叠  DER = {der_value:6.2f}%"
                  f"   (诊断用，不等同官方成绩)")

    print("\n" + "=" * 78)
    print("同一批预测，两把尺子（整场 8 场，跨文件汇总）")
    print("=" * 78)
    print(f"{'尺子':42s} {'collar 0.25':>12s} {'collar 0':>10s}")
    for arm in ARMS:
        mv = results[f"{arm}|md-eval|collar0.25"]["der"]
        mv0 = results[f"{arm}|md-eval|collar0.0"]["der"]
        label = f"官方 md-eval-22.pl（{arm}）"
        print(f"{label:42s} {mv:11.2f}% {mv0:9.2f}%")
    print(f"{'主仓 der.py（v12，并集分母）':42s} {'35.39':>11s}% {'45.66':>9s}%")
    print(f"{'（上行为 der.py 口径，同一批预测）':42s}")

    print("\n" + "=" * 78)
    print("重叠天花板诊断：同一批预测，开 `-1` 忽略参考重叠区")
    print("=" * 78)
    for arm in ARMS:
        full = results[f"{arm}|md-eval|collar0.25"]
        ign = results[f"{arm}|md-eval|collar0.25|ignore_overlaps"]
        f_miss = full["counts"].get("MISSED SPEAKER TIME", 0)
        i_miss = ign["counts"].get("MISSED SPEAKER TIME", 0)
        print(f"  {arm:9s} 含重叠 {full['der']:6.2f}%  ->  忽略重叠 {ign['der']:6.2f}%"
              f"   差 {full['der'] - ign['der']:+5.2f} pts")
        print(f"               MISSED 绝对量 {f_miss:7.1f}s -> {i_miss:7.1f}s"
              f"   减少 {f_miss - i_miss:7.1f}s"
              f"（占含重叠时 MISSED 的 {(f_miss - i_miss) / f_miss * 100 if f_miss else 0:.0f}%）")
    print("  注：官方**未**开 -1，所以上面的含重叠数字才是可对标的；"
          "-1 只是用来回答\"MISSED 有多少来自重叠\"。")

    print("\n" + "=" * 78)
    print("对官方成绩（论文 arXiv:2202.03647 / 2110.07393）")
    print("=" * 78)
    ours = results["v12|md-eval|collar0.25"]["der"]
    base = OFFICIAL["baseline_eval_far_collar0.25"]
    print(f"  官基线 Eval-Ali-far  = {base:.2f}%   (ch0 单通道 + oracle VAD + collar 0.25)")
    print(f"  我们 v12              = {ours:.2f}%   (ch0 单通道 + 自研 FSMN-VAD + collar 0.25)")
    print(f"  -> 差距               = {ours - base:+.2f} pts   倍数 {ours / base:.2f}x")
    print()
    print(f"  Royalflush 融合 Eval  = {OFFICIAL['royalflush_fusion_eval_collar0.25']:.2f}%"
          f"   (多通道)  -> 差 {ours - OFFICIAL['royalflush_fusion_eval_collar0.25']:+.2f} pts"
          f"   倍数 {ours / OFFICIAL['royalflush_fusion_eval_collar0.25']:.2f}x")
    print(f"  冠军 A41 Test         = {OFFICIAL['winner_A41_test_collar0.25']:.2f}%"
          f"   (多通道+TS-VAD) -> 差 {ours - OFFICIAL['winner_A41_test_collar0.25']:+.2f} pts"
          f"   倍数 {ours / OFFICIAL['winner_A41_test_collar0.25']:.2f}x")

    print("\n注意：官方数字在 collar 0 下是 "
          f"{OFFICIAL['baseline_eval_far_collar0.0']:.2f}%，我们是 "
          f"{results['v12|md-eval|collar0.0']['der']:.2f}% —— 两把尺子的 collar 敏感度这一栏",
          "也一起比了。")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "der_official_mdeval.json"
    out.write_text(json.dumps({"ours": results, "official": OFFICIAL},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
