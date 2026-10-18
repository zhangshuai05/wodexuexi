# -*- coding: utf-8 -*-
"""在 AliMeeting Eval 远场 ch0 上跑主仓生产路径，算 DER。

口径（与主仓 `scripts/evaluation/der_v12_dev.py` 对齐，逐条写死）：

  * 音频：`Eval_Ali_ch0/*.wav`，16 kHz 单声道，由 `prepare_ch0.py` 从远场 8 通道取 ch0。
  * 参考：`reference_rttm/*.rttm`，由 `textgrid_to_rttm.py` 生成（口径 20/20 已验证）。
  * 预测：`claude_a2_v12.diarize()` 的**真实产出**，不是探针。
  * 过滤：剔除 `n_windows <= SLIVER_WINDOWS` 的孤立短段（`run_claude_lane` 的规则）。
  * 时长：`--seconds 0` 跑**整场**（默认）。主仓 AISHELL-4 口径是截到 600 秒，
    本子项目刻意跑全场，否则"长会议自动人数"这个缺口测不到。
  * 评分：主仓 `der.der()`，md-eval union 分母，collar=0.25。

`--seconds` 只跑前若干秒时必须显式声明——主仓 README 记着"DER 的两条口径差 69 个点
（参考是否截断到分析窗口）"，截断过的数字不能当整场数字用。

额外产出 **重叠归因**：把 miss / FA / confusion 按"参考帧是否为重叠帧"拆开。
这才是来 AliMeeting 的主要目的——把"欠分割"和"重叠导致的必然错"分开。

    python run_der.py --limit 1 --seconds 300      # 冒烟
    python run_der.py                              # 8 场整场（CPU 上很慢）
"""

import argparse
import json
import sys
import time

import numpy as np

from common import (ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools,
                    resolve_wav)

der, read_rttm = der_tools()


def load_models():
    """主仓服务器用的同一套模型（VAD + CAM++），不加载 ASR。

    `web_server` 在 `streaming_asr/` 下，靠 `common.py` 已插入的 ASR_DIR 找到。
    """
    from web_server import get_models
    return get_models(lambda message: None, need_stream=False, need_offline=False)


def predict(model_bundle, wav_path, seconds):
    import claude_a2_v12 as v12
    result = v12.diarize(str(wav_path), model_bundle, max_seconds=seconds)
    kept = [s for s in result["segments"] if s["n_windows"] > v12.SLIVER_WINDOWS]
    dropped = len(result["segments"]) - len(kept)
    segments = [{"start": float(s["start"]), "end": float(s["end"]),
                 "speaker": f"v12_{int(s['spk'])}"} for s in kept]
    level = ("refined" if result["refined"] else
             "rescued" if result["rescued"] else
             "insufficient" if result["insufficient"] else "2s-only")
    return segments, {"k": result["k"], "level": level, "dropped": dropped,
                      "n_windows": result["n_windows"]}


def prediction_path(meeting, arm, seconds):
    """预测 RTTM 的落盘路径。

    文件名**必须带臂名与时长口径**：之前整场腿与 600 秒腿都写 `{meeting}.rttm`，
    两条腿并行跑时互相覆盖，落盘结果变成 6 场 600 秒 + 2 场整场的混合物
    （JSON 里的 DER 是内存中算的所以没出错，但事后重评会发现对不上）。
    """
    suffix = "full" if seconds <= 0 else f"{seconds:g}s"
    return PREDICTIONS / f"{meeting}.{arm}.{suffix}.rttm"


def write_rttm(path, segments, uri):
    lines = [f"SPEAKER {uri} 1 {s['start']:.4f} {s['end'] - s['start']:.4f} "
             f"<NA> <NA> {s['speaker']} <NA> <NA>" for s in segments]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def overlap_attribution(reference, predicted, collar=COLLAR):
    """把误差按"参考帧是否重叠"拆开。

    复用 der.py 的 `_frames` / `_collar_mask`，保证与 `der()` 同一套帧栅格与
    collar 屏蔽；映射沿用同一套 greedy-by-overlap 规则。不这样做就会变成
    两个口径，数字对不上。
    """
    from der import _collar_mask, _frames

    duration = max([s["end"] for s in reference] + [s["end"] for s in predicted] + [0.0])
    n = int(duration / 0.01) + 1
    ref, hyp = _frames(reference, duration), _frames(predicted, duration)
    mask = _collar_mask(reference, collar, n)

    overlap = {}
    for i in range(n):
        if not mask[i]:
            for h in hyp[i]:
                for r in ref[i]:
                    overlap[(h, r)] = overlap.get((h, r), 0) + 1
    mapping, used = {}, set()
    for (h, r), _ in sorted(overlap.items(), key=lambda item: -item[1]):
        if h not in mapping and r not in used:
            mapping[h] = r
            used.add(r)

    buckets = {(k, o): 0 for k in ("missed", "false_alarm", "confusion")
               for o in ("overlap", "single")}
    for i in range(n):
        if mask[i]:
            continue
        R, H = ref[i], hyp[i]
        if not R:
            buckets[("false_alarm", "single")] += len(H)
            continue
        key = "overlap" if len(R) >= 2 else "single"
        mapped = {mapping[h] for h in H if h in mapping and mapping[h] in R}
        if R <= mapped:
            continue
        buckets[("missed" if not H else "confusion", key)] += 1

    divisor = max(1, sum(1 for i in range(n) if not mask[i] and ref[i]))
    out = {}
    for kind in ("missed", "false_alarm", "confusion"):
        for kind_overlap in ("overlap", "single"):
            frames = buckets[(kind, kind_overlap)]
            out[f"{kind}_{kind_overlap}"] = frames
            out[f"{kind}_{kind_overlap}_rate"] = round(frames / divisor, 4)
    out["total_frames"] = divisor
    return out


def truncate_reference(reference, seconds):
    """把参考截到分析窗口。

    不截的后果是真实存在的：主仓 README 记着「DER 的两条口径差 69 个点
    （参考是否截断到分析窗口）」。`--seconds 300` 而参考是整场（1574 秒）时，
    后面 1274 秒全非重叠的参考语音会整体落进 miss——冒烟时实测 DER 81.04%，
    其中 miss 78.74%，全是这个截断假象。

    与主仓 `der_v12_dev.py` 同口径：整段落在窗口内才保留，跨界的按 `seconds` 切齐。
    """
    if seconds <= 0:
        return reference
    return [{"start": s["start"], "end": min(s["end"], seconds),
             "speaker": s["speaker"]}
            for s in reference if s["start"] < seconds]


def run_meeting(model_bundle, meeting, seconds, collar):
    wav = resolve_wav(meeting)
    reference = truncate_reference(read_rttm(ALI_RTTM / f"{meeting}.rttm"), seconds)
    started = time.time()
    predicted, meta = predict(model_bundle, wav, seconds)
    elapsed = time.time() - started
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    write_rttm(prediction_path(meeting, "v12", seconds), predicted, meeting)

    result = der(reference, predicted, collar=collar)
    result.update(overlap_attribution(reference, predicted, collar))
    result.update(meta)
    result["elapsed_s"] = round(elapsed, 1)
    result["audio_hours"] = round(max(s["end"] for s in reference) / 3600, 3)
    result["scored_seconds"] = round(max(s["end"] for s in reference), 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 = 整场（默认）")
    parser.add_argument("--collar", type=float, default=COLLAR)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--out", default="der_alimeeting.json")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if not meetings:
        raise SystemExit(f"参考 RTTM 缺失，先跑：python textgrid_to_rttm.py --corpus alimeeting")
    if args.limit:
        meetings = meetings[:args.limit]

    models = load_models()
    rows = []
    print(f"{'meeting':16s} {'时长':>6s} {'k':>3s} {'档':>12s} "
          f"{'DER':>7s} {'miss':>6s} {'FA':>6s} {'conf':>6s} {'耗时':>7s}", flush=True)
    for meeting in meetings:
        try:
            row = run_meeting(models, meeting, args.seconds, args.collar)
        except Exception as exc:  # 失败场次显式记录，不静默跳过
            print(f"{meeting:16s}  FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"meeting": meeting, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(f"{meeting:16s} {row['audio_hours'] * 60:5.0f}m {row['k']:>3d} "
              f"{row['level']:>12s} {row['der'] * 100:6.2f}% "
              f"{row['miss_rate'] * 100:5.2f}% {row['false_alarm_rate'] * 100:5.2f}% "
              f"{row['confusion_rate'] * 100:5.2f}% {row['elapsed_s']:6.0f}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    if ok:
        mean = lambda key: float(np.mean([r[key] for r in ok]))
        print(f"\n平均（{len(ok)}/{len(rows)} 场，整场，collar={args.collar}，"
              f"md-eval union 分母）")
        print(f"  DER  = {mean('der') * 100:6.2f}%")
        print(f"  miss = {mean('miss_rate') * 100:6.2f}%   "
              f"FA = {mean('false_alarm_rate') * 100:6.2f}%   "
              f"confusion = {mean('confusion_rate') * 100:6.2f}%")
        print(f"\n重叠归因（占参考语音帧的百分点）")
        print(f"  confusion 落在参考重叠帧 = {mean('confusion_overlap_rate') * 100:6.2f}%")
        print(f"  confusion 落在参考单讲帧 = {mean('confusion_single_rate') * 100:6.2f}%")
        print(f"  missed    落在参考重叠帧 = {mean('missed_overlap_rate') * 100:6.2f}%")
        print(f"  missed    落在参考单讲帧 = {mean('missed_single_rate') * 100:6.2f}%")
        print(f"  总 CPU 耗时 {sum(r['elapsed_s'] for r in ok) / 3600:.2f} 小时")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "seconds": args.seconds or "full", "collar": args.collar,
                   "denominator": "md-eval union", "sliver_filter": "n_windows > 2"},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
