# -*- coding: utf-8 -*-
"""单变量对照：**只有选 K 策略不同**，其余与 v12 完全一致。

怎么得到这个单变量：

    baseline  穷举 k∈[2,10] 取轮廓系数最大，不做两级细化
    v12       候选 K 带簇尺寸门槛与小 K 惩罚（`_select_candidate`），
              再走 0.6 秒短窗细化边界

两腿共用：同一批 ch0 音频、同一套 VAD、同一个 CAM++、同 2.0 秒窗 / 1.0 秒步长、
同一个 `_segment_windows` 边界公式（`hop_s = v12.HOP_S`）、同一个 sliver 过滤、
同一把 `der.der()`。

⚠️ **这个 arm 不是主仓那条「22.79% 基线」。** 主仓 `der_v12_dev.py` 的探针腿用
`hop_s = window * 0.25 = 0.5` 且**不**过滤碎段；实测（见下）那会让每个窗自成一段
（窗间距 1.0 > 0.5×1.5），产出大量 1 窗碎段。这里是刻意对齐 v12 的分段规则，
换来"只有 K 不同"，代价是不能跟 22.79% 直接比。

    hop_s=0.5 + 1.0 秒窗间距 -> 每窗自成一段，n_windows 全为 1 -> 被 sliver 过滤清空
    hop_s=1.0 + 1.0 秒窗间距 -> 相邻同标签窗连通（正确行为）

    python run_baseline.py --limit 1 --seconds 300
    python run_baseline.py --out baseline_alimeeting_full.json
"""

import argparse
import json
import time

import numpy as np

from common import (ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools,
                    resolve_wav)
from run_der import (load_models, overlap_attribution, prediction_path,
                     truncate_reference, write_rttm)

der, read_rttm = der_tools()

# 探针口径的硬编码值（照抄 der_v12_dev.py 的选 K 部分）
PROBE_MAX_K = 10


def predict_baseline(model_bundle, wav_path):
    """2.0 秒窗 + AHC + 穷举轮廓系数选 K + 窗中点边界，无细化。"""
    import soundfile as sf

    import claude_a2_v11 as v11
    import claude_a2_v12 as v12
    from scipy.cluster.hierarchy import fcluster, linkage

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = v11._vad_segs(model_bundle["vad"], str(wav_path))

    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E, times, _errors = v11._extract_windows(audio, rate, vad_segments,
                                                 model_bundle["spk"])
    if len(E) < 2:
        return [], {"k": 0, "level": "insufficient", "dropped": 0,
                    "n_windows": int(len(E))}

    tree = linkage(E, method="average", metric="cosine")
    best_k, best_score = None, None
    for k in range(2, min(PROBE_MAX_K, len(E)) + 1):
        labels = fcluster(tree, t=k, criterion="maxclust") - 1
        score = v11._silhouette(E, labels)
        if best_score is None or score > best_score:
            best_k, best_score = k, score
    labels = fcluster(tree, t=best_k, criterion="maxclust") - 1
    segments = v12._segment_windows(times, labels, np.zeros(len(times)),
                                    hop_s=v12.HOP_S)
    kept = [s for s in segments if s["n_windows"] > v12.SLIVER_WINDOWS]
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "speaker": f"base_{int(s['spk'])}"} for s in kept], {
        "k": int(best_k), "level": "baseline-exhaustive-k",
        "dropped": len(segments) - len(kept), "n_windows": int(len(E)),
        "silhouette": round(float(best_score), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--collar", type=float, default=COLLAR)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--out", default="baseline_alimeeting.json")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if args.limit:
        meetings = meetings[:args.limit]

    models = load_models()
    rows = []
    print(f"{'meeting':16s} {'k':>3s} {'DER':>7s} {'miss':>6s} {'FA':>6s} "
          f"{'conf':>6s} {'耗时':>6s}", flush=True)
    for meeting in meetings:
        try:
            wav = resolve_wav(meeting)
            reference = truncate_reference(
                read_rttm(ALI_RTTM / f"{meeting}.rttm"), args.seconds)
            started = time.time()
            predicted, meta = predict_baseline(models, wav)
            elapsed = time.time() - started
            PREDICTIONS.mkdir(parents=True, exist_ok=True)
            write_rttm(prediction_path(meeting, "baseline", args.seconds),
                       predicted, meeting)
            row = der(reference, predicted, collar=args.collar)
            row.update(overlap_attribution(reference, predicted, args.collar))
            row.update(meta)
            row["meeting"] = meeting
            row["elapsed_s"] = round(elapsed, 1)
        except Exception as exc:
            print(f"{meeting:16s}  FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"meeting": meeting, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(f"{meeting:16s} {row['k']:>3d} {row['der'] * 100:6.2f}% "
              f"{row['miss_rate'] * 100:5.2f}% {row['false_alarm_rate'] * 100:5.2f}% "
              f"{row['confusion_rate'] * 100:5.2f}% {row['elapsed_s']:5.0f}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    if ok:
        mean = lambda key: float(np.mean([r[key] for r in ok]))
        print(f"\n平均（{len(ok)}/{len(rows)} 场，整场，collar={args.collar}）"
              f"  基线 DER = {mean('der') * 100:.2f}%")
        print(f"  miss = {mean('miss_rate') * 100:5.2f}%  "
              f"FA = {mean('false_alarm_rate') * 100:5.2f}%  "
              f"confusion = {mean('confusion_rate') * 100:5.2f}%")
        print(f"  confusion 落在参考重叠帧 = "
              f"{mean('confusion_overlap_rate') * 100:5.2f}%")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "arm": "single-variable: exhaustive-k AHC, no refinement",
                   "note": "不是主仓 22.79% 那条探针腿（分段规则不同，见模块 docstring）",
                   "seconds": args.seconds or "full", "collar": args.collar},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
