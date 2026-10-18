# -*- coding: utf-8 -*-
"""v10 E1.1: causal Paraformer draft + frozen v7 offline reference rewrite."""

import os
import threading
import time

import soundfile as sf

from v10_asr_probe import stream_replay


METHOD = "claude-a2-v10-e1-two-pass-reference"
# ponytail: serial draft inference; use a session worker when live streaming is implemented.
_inference_lock = threading.Lock()


def run_claude_lane(wav, models, lane_key, progress=None):
    if lane_key != "v10":
        raise ValueError(f"unknown v10 lane {lane_key}")
    missing = [key for key in ("stream_asr", "asr") if key not in models]
    if missing:
        raise RuntimeError(f"v10 model is not loaded: {', '.join(missing)}")

    audio, sample_rate = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        raise ValueError("v10 requires 16 kHz mono audio")
    if sample_rate != 16000:
        raise ValueError(f"v10 requires 16 kHz audio, got {sample_rate} Hz")

    started = time.perf_counter()
    if progress:
        progress("v10：连续回放 Paraformer 草稿（当前非实时上屏）…")
    with _inference_lock:
        draft = stream_replay(
            models["stream_asr"], audio, sample_rate, realtime=False
        )

    if progress:
        progress("v10：整文件到达，运行冻结 v7 转写对照…")
    from claude_a2_v7_v8 import run_claude_lane as run_v7

    endpoint = run_v7(wav, models, "c1", progress=progress)
    sentences = []
    for sentence in endpoint.get("sentence_info", []):
        final = dict(sentence)
        final.update({
            "asr_state": "offline_reference",
            "asr_source": "offline-asr-reference",
            "speaker_source": "v7_frozen_offline",
        })
        sentences.append(final)

    duration = len(audio) / sample_rate
    diag = dict(endpoint.get("diag", {}))
    diag.update({
        "stage": "e1_two_pass_reference",
        "causal_draft": True,
        "online_pipeline": False,
        "reference_method": endpoint["method"],
        "draft_text": draft["text"],
        "draft_chunks": len(draft["events"]),
        "draft_inference_seconds": draft["inference_seconds"],
        "draft_rtf": round(draft["inference_seconds"] / duration, 4)
        if duration else None,
        "draft_token_timestamps": False,
        "final_asr": "offline-asr-reference",
        "speaker_source": "v7_frozen_offline",
        "rescue_decision": "Paraformer 连续草稿 + v7 离线转写对照",
    })
    return {
        "method": METHOD,
        "file": os.path.basename(wav),
        "elapsed": round(time.perf_counter() - started, 1),
        "diag": diag,
        "verdict": "experimental",
        "verdict_reasons": [
            "两遍离线对照；在线端点、Speaker Tracker 和增量 LLM 尚未实现"
        ],
        "sentence_info": sentences,
    }
