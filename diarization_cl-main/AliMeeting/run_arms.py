# -*- coding: utf-8 -*-
"""多臂实验：VAD 变量 × 候选准入变量，跑**同一条 v12 生产链路**。

关键手法：不复制 `claude_a2_v12.diarize()` 的任何代码，而是**猴补丁**它依赖的两个点：

    claude_a2_v11._vad_segs(vad, wav)          <- 换 VAD（含 oracle）
    claude_a2_v11._candidate_clusters(E, ...)  <- 换候选准入

`diarize()` 内部是在调用时做模块属性查找的，所以补丁生效；这样两级细化、
rescue 分支、`_segment_windows` 边界公式、sliver 过滤全部保持原样，
**变量只有那两处**。

    python run_arms.py --arms current/baseline tuned/baseline tuned/pruned
    python run_arms.py --arms oracle/baseline oracle/pruned --threshold -0.5

⚠️ `oracle/*` 用参考语音当 VAD，是**诊断上界**，不是可部署配置，不得当成绩引用。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import common  # noqa: F401
import claude_a2_v11 as v11
import claude_a2_v12 as v12
from cluster_fix import install as install_pruned, uninstall as uninstall_pruned
from common import ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools, resolve_wav
from run_der import load_models, overlap_attribution, truncate_reference, write_rttm
from vad_bench import (CURRENT_THRESHOLD, load_silero, load_vad, make_variant,
                       silero_segments, vad_segments)

der, read_rttm = der_tools()

SILERO_THRESHOLD = 0.3


def meeting_of(wav):
    """`R8001_M8004_MS801.wav` -> `R8001_M8004`，与 resolve_wav 的前缀规则互逆。"""
    stem = Path(wav).stem
    candidates = [p.stem for p in ALI_RTTM.glob("*.rttm") if stem.startswith(p.stem + "_")]
    if len(candidates) != 1:
        raise SystemExit(f"{stem} 无法唯一映射回会议 id：{candidates}")
    return candidates[0]


def reference_segments(meeting):
    merged = []
    for seg in sorted(read_rttm(ALI_RTTM / f"{meeting}.rttm"), key=lambda s: s["start"]):
        if merged and seg["start"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], seg["end"])
        else:
            merged.append([seg["start"], seg["end"]])
    return [(s, e) for s, e in merged]


class Arm:
    """装好一组补丁，跑完还原。"""

    def __init__(self, vad, candidates, threshold):
        self.vad, self.candidates, self.threshold = vad, candidates, threshold
        self._restore = []

    def __enter__(self):
        if self.vad == "oracle":
            table = {resolve_wav(p.stem).stem: reference_segments(p.stem)
                     for p in ALI_RTTM.glob("*.rttm")}
            v11._vad_segs = lambda vad, wav: table[Path(wav).stem]
            self._restore.append(("vad", None))
        elif self.vad == "tuned":
            model = load_vad(make_variant(self.threshold))
            v11._vad_segs = lambda vad, wav: vad_segments(model, wav)
            self._restore.append(("vad", None, model))
        elif self.vad == "silero":
            model = load_silero()
            v11._vad_segs = lambda vad, wav: silero_segments(model, wav, SILERO_THRESHOLD)
            self._restore.append(("vad", None, model))
        elif self.vad == "current":
            pass
        else:
            raise SystemExit(f"未知 vad 臂：{self.vad}")

        if self.candidates == "pruned":
            self._restore.append(("cand", install_pruned()))

        self._saved_vad = getattr(v11, "_vad_segs")
        return self

    def __exit__(self, *exc):
        for item in self._restore:
            if item[0] == "cand":
                uninstall_pruned(item[1])
        if self.vad != "current":
            # 还原原函数：从模块里删掉猴补丁，重新绑定原始实现
            v11._vad_segs = _ORIGINAL_VAD_SEGS
        return False


_ORIGINAL_VAD_SEGS = v11._vad_segs


def run_arm(models, arm_name, threshold, meetings):
    vad, candidates = arm_name.split("/")
    rows = []
    print(f"\n=== 臂 {arm_name}（VAD={vad}，候选={candidates}）===", flush=True)
    print(f"{'meeting':16s} {'真值K':>5s} {'档':>12s} {'k':>3s} {'DER':>7s} "
          f"{'miss':>6s} {'FA':>6s} {'conf':>6s} {'耗时':>6s}", flush=True)

    arm = Arm(vad, candidates, threshold)
    arm.__enter__()
    try:
        for meeting in meetings:
            try:
                wav = resolve_wav(meeting)
                reference = read_rttm(ALI_RTTM / f"{meeting}.rttm")
                true_k = len({s["speaker"] for s in reference})
                started = time.time()
                result = v12.diarize(str(wav), models, max_seconds=0.0)
                elapsed = time.time() - started

                kept = [s for s in result["segments"]
                        if s["n_windows"] > v12.SLIVER_WINDOWS]
                predicted = [{"start": float(s["start"]), "end": float(s["end"]),
                              "speaker": f"a_{int(s['spk'])}"} for s in kept]
                level = ("refined" if result["refined"] else
                         "rescued" if result["rescued"] else
                         "insufficient" if result["insufficient"] else "2s-only")
                PREDICTIONS.mkdir(parents=True, exist_ok=True)
                write_rttm(PREDICTIONS / f"{meeting}.{vad}_{candidates}.full.rttm",
                           predicted, meeting)

                row = der(reference, predicted, collar=COLLAR)
                row.update(overlap_attribution(reference, predicted, COLLAR))
                row.update({"meeting": meeting, "true_k": true_k, "k": result["k"],
                            "level": level, "n_windows": result["n_windows"],
                            "elapsed_s": round(elapsed, 1)})
                rows.append(row)
                print(f"{meeting:16s} {true_k:>5d} {level:>12s} {result['k']:>3d} "
                      f"{row['der'] * 100:6.2f}% {row['miss_rate'] * 100:5.2f}% "
                      f"{row['false_alarm_rate'] * 100:5.2f}% "
                      f"{row['confusion_rate'] * 100:5.2f}% {elapsed:5.0f}s", flush=True)
            except Exception as exc:
                print(f"{meeting:16s}  FAILED {type(exc).__name__}: {exc}", flush=True)
                rows.append({"meeting": meeting,
                             "error": f"{type(exc).__name__}: {exc}"})
    finally:
        arm.__exit__()

    ok = [r for r in rows if "error" not in r]
    if ok:
        mean = lambda key: float(np.mean([r[key] for r in ok]))
        hits = sum(1 for r in ok if r["k"] == r["true_k"])
        print(f"  平均 DER = {mean('der') * 100:6.2f}%   "
              f"K 命中真值 {hits}/{len(ok)}   "
              f"耗时 {sum(r['elapsed_s'] for r in ok) / 60:.1f} min", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*",
                        default=["current/baseline", "tuned/baseline",
                                 "tuned/pruned", "oracle/baseline", "oracle/pruned"])
    parser.add_argument("--threshold", type=float, default=-0.5)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--out", default="arms_alimeeting.json")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    models = load_models()

    out = {}
    for arm_name in args.arms:
        rows = run_arm(models, arm_name, args.threshold, meetings)
        out[arm_name] = rows
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(json.dumps(
            {"threshold": args.threshold, "arms": out}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print("\n" + "=" * 78)
    print("汇总（der.py 口径，整场 8 场）")
    print("=" * 78)
    print(f"{'臂':26s} {'DER':>8s} {'K命中':>8s} {'conf重叠':>9s} {'conf单讲':>9s}")
    for arm_name, rows in out.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            continue
        mean = lambda k: float(np.mean([r[k] for r in ok]))
        hits = sum(1 for r in ok if r["k"] == r["true_k"])
        print(f"{arm_name:26s} {mean('der') * 100:7.2f}% {hits:>4d}/{len(ok):<3d} "
              f"{mean('confusion_overlap_rate') * 100:8.2f}% "
              f"{mean('confusion_single_rate') * 100:8.2f}%")
    print(f"\n已写出 {RESULTS / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
