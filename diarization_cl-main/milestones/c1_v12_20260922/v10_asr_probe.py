# -*- coding: utf-8 -*-
"""E1 probe: streaming Paraformer draft, Nano endpoint text, fa-zh alignment."""

import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import soundfile as sf


STREAM_MODEL = "paraformer-zh-streaming"
ALIGN_MODEL = "fa-zh"
DEFAULT_CHUNK_SIZE = (0, 10, 5)  # 600 ms input, 300 ms lookahead


def _text(result):
    return str(result[0].get("text", "") if result else "")


def stream_replay(model, audio, sample_rate=16000, chunk_size=DEFAULT_CHUNK_SIZE,
                  realtime=True, clock=time.perf_counter, sleep=time.sleep):
    """Feed only arrived samples and return an auditable chunk event log."""
    chunk_samples = int(chunk_size[1] * 0.06 * sample_rate)
    if chunk_samples <= 0:
        raise ValueError("chunk_size[1] must be positive")
    if sample_rate != 16000 or len(audio) == 0:
        raise ValueError("nonempty 16 kHz audio is required")

    cache = {}
    events = []
    pieces = []
    infer_seconds = 0.0
    started = clock()
    first_partial_ms = None

    for start in range(0, len(audio), chunk_samples):
        end = min(start + chunk_samples, len(audio))
        if realtime:
            wait = end / sample_rate - (clock() - started)
            if wait > 0:
                sleep(wait)

        call_started = clock()
        result = model.generate(
            input=audio[start:end],
            cache=cache,
            is_final=end == len(audio),
            chunk_size=list(chunk_size),
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        call_ended = clock()
        infer_seconds += call_ended - call_started
        text = _text(result)
        if text:
            pieces.append(text)
            if first_partial_ms is None:
                first_partial_ms = round((call_ended - started) * 1000, 1)
        events.append({
            "start_sample": start,
            "end_sample": end,
            "audio_end_ms": round(end * 1000 / sample_rate, 1),
            "is_final": end == len(audio),
            "text": text,
            "inference_ms": round((call_ended - call_started) * 1000, 1),
            "emitted_ms": round((call_ended - started) * 1000, 1),
            "queue_wait_ms": round(max(0, call_started - started - end / sample_rate) * 1000, 1) if realtime else None,
        })

    duration = len(audio) / sample_rate
    return {
        "text": "".join(pieces),
        "events": events,
        "first_partial_ms": first_partial_ms,
        "first_partial_audio_end_ms": next((e["audio_end_ms"] for e in events if e["text"]), None),
        "timing_note": "first_partial_ms is measured from replay start, not speech onset; fast mode is not latency",
        "token_timestamps_available": False,
        "inference_seconds": round(infer_seconds, 3),
        "inference_rtf": round(infer_seconds / duration, 4) if duration else None,
        "wall_seconds": round(clock() - started, 3),
    }


def load_audio(path):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        raise ValueError("E1 probe requires mono audio; convert explicitly before replay")
    if sample_rate != 16000:
        raise ValueError(f"E1 probe requires 16 kHz audio, got {sample_rate} Hz")
    return np.asarray(audio, dtype=np.float32), sample_rate


def load_model(name, device):
    from funasr import AutoModel
    return AutoModel(
        model=name,
        model_revision="v2.0.4",
        device=device,
        disable_update=True,
        disable_pbar=True,
    )


def release_model():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def endpoint_nano(path, device):
    from funasr import AutoModel
    from funasr_backend import FUNASR_CACHE, wrap_funasr
    model = AutoModel(
        model=FUNASR_CACHE,
        device=device,
        disable_update=True,
        disable_pbar=True,
    )
    started = time.perf_counter()
    text = _text(wrap_funasr(model).generate(input=path))
    elapsed = time.perf_counter() - started
    return model, {"text": text, "inference_seconds": round(elapsed, 3)}


def align_text(model, path, text):
    if not text:
        return {"text": "", "timestamp": [], "inference_seconds": 0.0}
    started = time.perf_counter()
    result = model.generate(
        input=(path, text),
        data_type=("sound", "text"),
    )
    elapsed = time.perf_counter() - started
    aligned = dict(result[0]) if result else {"text": text, "timestamp": []}
    aligned["inference_seconds"] = round(elapsed, 3)
    return aligned


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_evidence(output, evidence):
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as file:
            json.dump(evidence, file, ensure_ascii=False, indent=2)


def run(path, device="cuda:0", realtime=True, endpoint=False, align=False, output=None):
    audio, sample_rate = load_audio(path)
    evidence = {
        "audio": os.path.abspath(path),
        "audio_sha256": file_sha256(path),
        "sample_rate": sample_rate,
        "samples": len(audio),
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "realtime_replay": realtime,
        "chunk_size": list(DEFAULT_CHUNK_SIZE),
        "device": device,
        "denoised": False,
        "scope": "ASR capability probe only; whole-file EOF endpoint, no VAD/LLM/speaker tracking",
        "models": {"stream": STREAM_MODEL, "revision": "v2.0.4"},
    }

    print("Stage: Paraformer-Streaming", flush=True)
    loaded = time.perf_counter()
    model = load_model(STREAM_MODEL, device)
    evidence["stream_model_load_seconds"] = round(time.perf_counter() - loaded, 3)
    evidence["stream"] = stream_replay(model, audio, sample_rate, realtime=realtime)
    save_evidence(output, evidence)
    del model
    release_model()

    if endpoint or align:
        print("Stage: Nano EOF rewrite (not an online VAD endpoint)", flush=True)
        loaded = time.perf_counter()
        nano_model, evidence["endpoint_nano"] = endpoint_nano(path, device)
        evidence["nano_model_load_and_infer_seconds"] = round(time.perf_counter() - loaded, 3)
        save_evidence(output, evidence)
        del nano_model
        release_model()

    if align:
        print("Stage: fa-zh alignment", flush=True)
        loaded = time.perf_counter()
        model = load_model(ALIGN_MODEL, device)
        evidence["align_model_load_seconds"] = round(time.perf_counter() - loaded, 3)
        evidence["alignment"] = align_text(model, path, evidence["endpoint_nano"]["text"])
        save_evidence(output, evidence)
        del model
        release_model()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="16 kHz mono WAV used for causal file replay")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast", action="store_true", help="do not wait for 1x audio arrival")
    parser.add_argument("--endpoint-nano", action="store_true",
                        help="also run the slower endpoint rewrite")
    parser.add_argument("--align", action="store_true",
                        help="also run endpoint Nano and fa-zh alignment")
    parser.add_argument("--output", help="JSON output path")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(__file__), "output", "v10_asr_probe.json")
    result = run(
        args.audio,
        device=args.device,
        realtime=not args.fast,
        endpoint=args.endpoint_nano,
        align=args.align,
        output=output,
    )
    summary = {
        "output": os.path.abspath(output),
        "stream_text": result["stream"]["text"],
        "first_partial_ms": result["stream"]["first_partial_ms"],
        "inference_rtf": result["stream"]["inference_rtf"],
    }
    if "endpoint_nano" in result:
        summary["nano_text"] = result["endpoint_nano"]["text"]
    if "alignment" in result:
        summary["timestamps"] = len(result["alignment"].get("timestamp", []))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
