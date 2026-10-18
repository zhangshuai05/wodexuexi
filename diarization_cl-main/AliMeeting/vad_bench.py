# -*- coding: utf-8 -*-
"""VAD 体检：现有 FSMN-VAD 到底损失多少 DER？阈值扫描 + oracle 上界。

为什么要先测再换：主仓现在用的 VAD 是 `speech_noise_thres=0.7`（烧在 config.yaml 里，
不是运行时参数）。直接把 VAD 换掉是赌——得先知道它现在损失多少，
以及换它是"有空间"还是"没空间"。

本脚本**不跑分离器**，只算 VAD 自身在 DER 口径下贡献多少误差：

    在 md-eval 的记账下（collar 0.25，只计参考有语音的帧）
      VAD 假警报 = 帧上 VAD 说语音、但参考一个说话人都没有
      VAD 漏检   = 帧上 VAD 说静音、但参考有 n_ref 个说话人  -> 计 n_ref（md-eval 按人数计权重）
      分母        = Σ n_ref（等价于 md-eval 的 SCORED_SPEAKER）

这给出的是**上界**：分离器不可能比 VAD 给出更多的语音。真实 DER 一定 ≥ 这个值。

    python vad_bench.py                 # 扫全部阈值
    python vad_bench.py --thresholds 0.7   # 只测现状
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from common import (AISHELL_ROOT, ALI_RTTM, FRAME, PROJECT, RESULTS, der_tools,
                    resolve_wav)

_, read_rttm = der_tools()

# 两语料的路径。交叉验证用：只在 AliMeeting 上选阈值会把阈值选成
# "AliMeeting 有语音时长占比高"这个特性的附属物，不是 VAD 本身好。
CORPORA = {
    "alimeeting": {"rttm": ALI_RTTM, "wavext": ".wav"},
    "aishell4": {"rttm": AISHELL_ROOT / "TextGrid", "wavext": ".flac"},
}
CORPUS_PATHS = CORPORA["alimeeting"]

# 主仓当前用的本地副本（web_server.py 的 VAD_LOCAL）
VAD_SOURCE = Path(r"C:\Users\EDY\Desktop\new\iic--speech_fsmn_vad_zh-cn-16k-common-pytorch\snapshots\master")
VAD_VARIANTS = Path(r"D:\yinp\AliMeeting\vad_variants")
CURRENT_THRESHOLD = 0.7
SWEEP = (0.7, 0.6, 0.5, 0.3, 0.0, -0.3)


def make_variant(threshold):
    """把本地副本复制一份并把 speech_noise_thres 改成指定值。

    `speech_noise_thres` 是构造参数（`fsmn_vad_streaming/model.py:100`），
    funasr 的 `inference(**kwargs)` 只允许覆盖 `max_end_silence_time`
    （`model.py:858`），所以只能改 config.yaml。这也正是主仓 README 说的
    \"调过的本地副本\"做法。
    """
    target = VAD_VARIANTS / f"vad_thres_{threshold:+.1f}".replace("+", "p").replace("-", "m")
    if target.exists():
        return target
    shutil.copytree(VAD_SOURCE, target)
    config = target / "config.yaml"
    text = config.read_text(encoding="utf-8")
    replaced = None
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("speech_noise_thres:"):
            replaced = line
            indent = line[:len(line) - len(line.lstrip())]
            line = f"{indent}speech_noise_thres: {threshold}"
        lines.append(line)
    if replaced is None:
        raise SystemExit(f"{config} 里没有 speech_noise_thres，改法不适用了")
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_vad(path):
    from funasr import AutoModel
    return AutoModel(model=str(path), disable_update=True, disable_pbar=True)


# silero-vad：与 FSMN-VAD 完全不同的模型（CRNN + ONNX），用来回答"换个 VAD 行不行"。
# 它的 threshold 是"语音概率阈值"（默认 0.5），语义与 FSMN 的 score 差阈值不同，
# 所以不能互相对照数值，只能各自扫各自的。
SILERO_THRESHOLDS = (0.3, 0.5, 0.7)


def load_silero():
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=True)


def silero_segments(model, wav, threshold):
    """silero 自己的 `read_audio` 走 torchaudio，而本机 torchaudio 2.11 需要 torchcodec 才能
    读文件（未装）。所以自己用 soundfile 读，再交给 silero——结果等价，且不再引进依赖。
    """
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps

    audio, rate = sf.read(str(wav), dtype="float32", always_2d=True)
    if rate != 16000:
        raise ValueError(f"silero 要求 16 kHz，实际 {rate}")
    tensor = torch.from_numpy(audio[:, 0].copy())
    stamps = get_speech_timestamps(tensor, model, threshold=threshold,
                                   sampling_rate=16000, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def vad_segments(model, wav):
    result = model.generate(input=str(wav), dynamic_silence=False)
    rows = result if isinstance(result, list) else [result]
    return [(float(item[0]) / 1000.0, float(item[1]) / 1000.0)
            for row in rows for item in row.get("value", [])]


def _resolve(corpus, meeting):
    if corpus == "aishell4":
        return AISHELL_ROOT / "wav" / f"{meeting}.flac"
    return resolve_wav(meeting)


def reference_frames(rttm_dir, meeting, duration):
    """每帧的参考说话人数（原值，不做 collar）。"""
    n = int(duration / FRAME) + 1
    counts = np.zeros(n, dtype=np.int16)
    for seg in read_rttm(rttm_dir / f"{meeting}.rttm"):
        a = max(0, int(round(seg["start"] / FRAME)))
        b = min(n, int(round(seg["end"] / FRAME)))
        if b > a:
            counts[a:b] += 1
    return counts


def _collar_from_reference(rttm_dir, meeting, n, collar):
    mask = np.zeros(n, dtype=bool)
    span = int(round(collar / FRAME))
    for seg in read_rttm(rttm_dir / f"{meeting}.rttm"):
        for boundary in (seg["start"], seg["end"]):
            centre = int(round(boundary / FRAME))
            mask[max(0, centre - span):min(n, centre + span + 1)] = True
    return mask


def window_yield(segments, duration, win_s=2.0, hop_s=1.0, min_fraction=0.5):
    """VAD 分段能撑起多少个 2.0 秒窗（主仓 `_extract_windows` 的门）。

    这是比帧级 F1 更贴近链路的指标：帧级 F1 可以很高，但分段碎成 1.5 秒一堆的话，
    大量 2.0 秒窗会因为 VAD 覆盖不足 50% 而被丢掉，嵌入就稀疏了。
    返回 (通过的窗数, 名义窗数)。
    """
    merged = []
    for start, end in sorted(segments):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    last = max(0.0, duration - win_s)
    starts = np.arange(0.0, last + 1e-9, hop_s)
    kept = 0
    index = 0
    for start in starts:
        end = start + win_s
        # 单调扫一遍合并后的区间，避免每个窗都 O(len(segments))
        while index < len(merged) and merged[index][1] <= start:
            index += 1
        cursor, covered = index, 0.0
        while cursor < len(merged) and merged[cursor][0] < end:
            covered += max(0.0, min(end, merged[cursor][1]) - max(start, merged[cursor][0]))
            cursor += 1
        if covered >= min_fraction * win_s:
            kept += 1
    return kept, len(starts)


def evaluate(corpus, rttm_dir, meeting, segments, collar=0.25):
    import soundfile as sf
    duration = sf.info(_resolve(corpus, meeting)).duration
    counts = reference_frames(rttm_dir, meeting, duration)
    n = len(counts)
    mask = _collar_from_reference(rttm_dir, meeting, n, collar)

    vad = np.zeros(n, dtype=bool)
    for start, end in segments:
        a = max(0, int(round(start / FRAME)))
        b = min(n, int(round(end / FRAME)))
        if b > a:
            vad[a:b] = True

    scored = ~mask
    ref_speech = counts >= 1
    denom = int(counts[scored].sum())
    false_alarm = int((vad & ~ref_speech & scored).sum())
    missed = int(counts[scored & ~vad].sum())

    tp = int((vad & ref_speech & scored).sum())
    fp = false_alarm
    fn = int((~vad & ref_speech & scored).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    kept, nominal = window_yield(segments, duration)
    ref_kept, _ = window_yield(
        [(s["start"], s["end"]) for s in read_rttm(rttm_dir / f"{meeting}.rttm")], duration)
    lengths = np.array([e - s for s, e in segments]) if segments else np.zeros(1)
    return {
        "denominator": denom,
        "false_alarm_frames": false_alarm,
        "missed_speaker_frames": missed,
        "fa_rate": round(false_alarm / denom, 4),
        "miss_rate": round(missed / denom, 4),
        "vad_der_floor": round((false_alarm + missed) / denom, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_segments": len(segments),
        "median_segment_s": round(float(np.median(lengths)), 2),
        "speech_seconds": round(sum(e - s for s, e in segments), 2),
        "window_kept": kept,
        "window_nominal": nominal,
        "window_yield": round(kept / max(1, nominal), 4),
        "window_kept_reference": ref_kept,
    }


def main():
    global CORPUS_PATHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=tuple(CORPORA), default="alimeeting")
    parser.add_argument("--silero", action="store_true",
                        help="额外跑 silero-vad（另一个模型，非阈值扫描）")
    parser.add_argument("--no-fsmn", action="store_true", help="只跑 silero")
    parser.add_argument("--thresholds", nargs="*", type=float, default=list(SWEEP))
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--collar", type=float, default=0.25)
    args = parser.parse_args()

    corpus = args.corpus
    rttm_dir = CORPORA[corpus]["rttm"]
    meetings = args.meetings or sorted(p.stem for p in rttm_dir.glob("*.rttm"))
    VAD_VARIANTS.mkdir(parents=True, exist_ok=True)
    print(f"语料: {corpus}（{len(meetings)} 场，collar={args.collar}）")

    arms = {}
    for threshold in ([] if args.no_fsmn else args.thresholds):
        path = make_variant(threshold)
        model = load_vad(path)
        started = time.time()
        per_meeting = {}
        for meeting in meetings:
            wav = _resolve(corpus, meeting)
            try:
                per_meeting[meeting] = evaluate(
                    corpus, rttm_dir, meeting, vad_segments(model, wav), args.collar)
            except Exception as exc:
                per_meeting[meeting] = {"error": f"{type(exc).__name__}: {exc}"}
        ok = [v for v in per_meeting.values() if "error" not in v]
        summary = {k: float(np.mean([v[k] for v in ok]))
                   for k in ("fa_rate", "miss_rate", "vad_der_floor",
                             "precision", "recall", "f1", "speech_seconds",
                             "n_segments", "median_segment_s", "window_yield",
                             "window_kept", "window_kept_reference")}
        arms[f"thres_{threshold}"] = {"per_meeting": per_meeting, "summary": summary,
                                      "elapsed_s": round(time.time() - started, 1)}
        tag = "  <- 主仓现行" if threshold == CURRENT_THRESHOLD else ""
        print(f"  阈值 {threshold:+.1f}  F1={summary['f1'] * 100:5.2f}%  "
              f"P={summary['precision'] * 100:5.2f}% R={summary['recall'] * 100:5.2f}%  "
              f"FA={summary['fa_rate'] * 100:5.2f}% MISS={summary['miss_rate'] * 100:5.2f}%  "
              f"DER下限={summary['vad_der_floor'] * 100:5.2f}%  "
              f"段{summary['n_segments']:5.0f} 中位{summary['median_segment_s']:4.2f}s  "
              f"窗得{summary['window_yield'] * 100:5.1f}%  "
              f"({arms[f'thres_{threshold}']['elapsed_s']}s){tag}", flush=True)
        del model

    if args.silero:
        s_model = load_silero()
        for threshold in SILERO_THRESHOLDS:
            started = time.time()
            per_meeting = {}
            for meeting in meetings:
                wav = _resolve(corpus, meeting)
                try:
                    per_meeting[meeting] = evaluate(
                        corpus, rttm_dir, meeting,
                        silero_segments(s_model, wav, threshold), args.collar)
                except Exception as exc:
                    per_meeting[meeting] = {"error": f"{type(exc).__name__}: {exc}"}
            ok = [v for v in per_meeting.values() if "error" not in v]
            summary = {k: float(np.mean([v[k] for v in ok]))
                       for k in ("fa_rate", "miss_rate", "vad_der_floor",
                                 "precision", "recall", "f1", "speech_seconds",
                                 "n_segments", "median_segment_s", "window_yield",
                                 "window_kept", "window_kept_reference")}
            arms[f"silero_{threshold}"] = {"per_meeting": per_meeting,
                                           "summary": summary,
                                           "elapsed_s": round(time.time() - started, 1)}
            print(f"  silero {threshold:.1f}  F1={summary['f1'] * 100:5.2f}%  "
                  f"P={summary['precision'] * 100:5.2f}% R={summary['recall'] * 100:5.2f}%  "
                  f"FA={summary['fa_rate'] * 100:5.2f}% MISS={summary['miss_rate'] * 100:5.2f}%  "
                  f"DER下限={summary['vad_der_floor'] * 100:5.2f}%  "
                  f"段{summary['n_segments']:5.0f} 中位{summary['median_segment_s']:4.2f}s  "
                  f"窗得{summary['window_yield'] * 100:5.1f}%  "
                  f"({arms[f'silero_{threshold}']['elapsed_s']}s)", flush=True)
        del s_model

    # oracle 上界：直接用参考语音并集当 VAD
    oracle = {}
    for meeting in meetings:
        ref = read_rttm(rttm_dir / f"{meeting}.rttm")
        # 参考段并集（重叠合并）
        merged = []
        for seg in sorted(ref, key=lambda s: s["start"]):
            if merged and seg["start"] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], seg["end"])
            else:
                merged.append([seg["start"], seg["end"]])
        oracle[meeting] = evaluate(corpus, rttm_dir, meeting,
                                   [(s, e) for s, e in merged], args.collar)
    oracle_summary = {k: float(np.mean([v[k] for v in oracle.values()]))
                      for k in ("fa_rate", "miss_rate", "vad_der_floor",
                                "precision", "recall", "f1",
                                "n_segments", "median_segment_s", "window_yield",
                                "window_kept", "window_kept_reference")}
    arms["oracle_reference"] = {"per_meeting": oracle, "summary": oracle_summary}
    print(f"  oracle（参考语音并集）  F1={oracle_summary['f1'] * 100:5.2f}%  "
          f"FA={oracle_summary['fa_rate'] * 100:5.2f}% "
          f"MISS={oracle_summary['miss_rate'] * 100:5.2f}%  "
          f"DER下限={oracle_summary['vad_der_floor'] * 100:5.2f}%  "
          f"段{oracle_summary['n_segments']:5.0f} 中位{oracle_summary['median_segment_s']:4.2f}s  "
          f"窗得{oracle_summary['window_yield'] * 100:5.1f}%")

    candidates = {k: v for k, v in arms.items()
                  if k.startswith("thres_") or k.startswith("silero_")}
    best = min(candidates, key=lambda k: arms[k]["summary"]["vad_der_floor"])
    current = arms.get(f"thres_{CURRENT_THRESHOLD}")
    if current:
        print(f"\n现行 {CURRENT_THRESHOLD} 的 VAD 下限 = "
              f"{current['summary']['vad_der_floor'] * 100:.2f} pts")
    print(f"最优 {best} 的 VAD 下限 = {arms[best]['summary']['vad_der_floor'] * 100:.2f} pts")
    if current:
        gain = (current['summary']['vad_der_floor']
                - arms[best]['summary']['vad_der_floor']) * 100
        print(f"-> 换/调 VAD 最多能省 {gain:+.2f} pts")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"vad_bench_{corpus}.json"
    out.write_text(json.dumps({"corpus": corpus, "collar": args.collar,
                               "current_threshold": CURRENT_THRESHOLD,
                               "arms": arms}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
