# -*- coding: utf-8 -*-
"""pyannote 外部基线（与同一批 ch0 音频、同一把 `der.der()` 对照）。

## 为什么这个脚本现在是「备好但跑不了」

`pyannote/speaker-diarization-3.1` 及其依赖的 `pyannote/segmentation-3.0`、
`pyannote/embedding` 在 HuggingFace 上都是 **gated=auto**：必须登录账号、
在模型页接受条款、再配 token，匿名下载一律 401。这一步只能你本人做。

三步：

    1) 注册/登录 https://huggingface.co
    2) 打开并接受这三个模型页的条款（都在页面顶部 "Agree and access repository"）：
         https://huggingface.co/pyannote/speaker-diarization-3.1
         https://huggingface.co/pyannote/segmentation-3.0
         https://huggingface.co/pyannote/embedding
    3) 生成 token 并落到本地：
         https://huggingface.co/settings/tokens
         huggingface-cli login        # 或设环境变量 HF_TOKEN=hf_xxx

## 环境隔离要求（重要）

pyannote.audio 4.0.7 要求 `torch>=2.8.0` + `torchaudio` + `torchcodec`。
本机 `funasr` 环境跑着主链路，**不要往里装**——主仓 README 写着 funasr 是魔改版，
装坏了整个生产链路就没了。请另建环境：

    conda create -n pyannote python=3.11 -y
    conda activate pyannote
    pip install "pyannote.audio>=4.0" soundfile torch torchaudio

然后：

    python run_pyannote.py --limit 1 --seconds 300    # 冒烟
    python run_pyannote.py --out pyannote_alimeeting_full.json

⚠️ pyannote 与我们的 collar/DER 口径不同（pyannote 官方用的是自己的
`DiarizationErrorRate`，Hungarian 标签映射 + 可选 collar）。本脚本**只用它出 RTTM**，
评分统一走主仓 `der.der()`，这样两边的数字才是同一把尺子量出来的。
"""

import argparse
import json
import os
import time

import numpy as np

from common import (ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools,
                    resolve_wav)
from run_der import (overlap_attribution, prediction_path,
                     truncate_reference, write_rttm)

der, read_rttm = der_tools()

MODEL_ID = "pyannote/speaker-diarization-3.1"


def load_pipeline():
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        return Pipeline.from_pretrained(MODEL_ID, use_auth_token=token)
    except Exception as exc:
        raise SystemExit(
            f"\n下载 {MODEL_ID} 失败：{type(exc).__name__}: {exc}\n"
            f"这是 gated 模型的预期失败。请按本文件顶部注释完成三步授权，"
            f"并确认已 `huggingface-cli login` 或设好 HF_TOKEN。\n") from exc


def predict(pipeline, wav_path, speaker_count_hint=None):
    """返回 RTTM 段。不限制人数，除非显式给了 hint——pyannote 支持自动人数，
    与我们的 auto-K 是同一条腿，给它 hint 就不是对照了。"""
    annotation = pipeline(str(wav_path))
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({"start": float(turn.start), "end": float(turn.end),
                         "speaker": f"pyannote_{speaker}"})
    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--collar", type=float, default=COLLAR)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--out", default="pyannote_alimeeting.json")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if args.limit:
        meetings = meetings[:args.limit]

    pipeline = load_pipeline()
    rows = []
    print(f"{'meeting':16s} {'真值K':>5s} {'predK':>5s} {'DER':>7s} {'miss':>6s} "
          f"{'FA':>6s} {'conf':>6s} {'耗时':>6s}", flush=True)
    for meeting in meetings:
        try:
            wav = resolve_wav(meeting)
            reference = truncate_reference(read_rttm(ALI_RTTM / f"{meeting}.rttm"),
                                           args.seconds)
            true_k = len({s["speaker"] for s in reference})
            started = time.time()
            predicted = predict(pipeline, wav)
            elapsed = time.time() - started
            PREDICTIONS.mkdir(parents=True, exist_ok=True)
            write_rttm(prediction_path(meeting, "pyannote", args.seconds),
                       predicted, meeting)
            row = der(reference, predicted, collar=args.collar)
            row.update(overlap_attribution(reference, predicted, args.collar))
            row["meeting"] = meeting
            row["true_k"] = true_k
            row["elapsed_s"] = round(elapsed, 1)
        except Exception as exc:
            print(f"{meeting:16s}  FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"meeting": meeting, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(row)
        print(f"{meeting:16s} {true_k:>5d} {row['n_pred_speakers']:>5d} "
              f"{row['der'] * 100:6.2f}% {row['miss_rate'] * 100:5.2f}% "
              f"{row['false_alarm_rate'] * 100:5.2f}% "
              f"{row['confusion_rate'] * 100:5.2f}% {row['elapsed_s']:5.0f}s", flush=True)

    ok = [r for r in rows if "error" not in r]
    if ok:
        mean = lambda key: float(np.mean([r[key] for r in ok]))
        print(f"\n平均（{len(ok)}/{len(rows)} 场）pyannote DER = {mean('der') * 100:.2f}%")
        print(f"  miss = {mean('miss_rate') * 100:5.2f}%  FA = {mean('false_alarm_rate') * 100:5.2f}%"
              f"  confusion = {mean('confusion_rate') * 100:5.2f}%")
        hits = sum(1 for r in ok if r["n_pred_speakers"] == r["true_k"])
        print(f"  人数命中真值: {hits}/{len(ok)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "model": MODEL_ID, "seconds": args.seconds or "full",
                   "collar": args.collar, "denominator": "md-eval union",
                   "note": "pyannote 只用来出 RTTM；评分统一走主仓 der.der()"},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
