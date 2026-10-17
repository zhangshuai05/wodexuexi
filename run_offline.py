# -*- coding: utf-8 -*-
"""v0 端到端：WAV -> 流式三态 -> 异步润色 -> 单臂回填 -> session 目录。

用法：
    python run_offline.py --wav D:/Deepseek/yinp/smoke_clip_180s.wav --fast --no-llm
    python run_offline.py --wav <16k mono wav>            # 实时节奏
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import bootstrap
from turns import StreamTurnTracker


SAMPLE_RATE = 16000
MAX_END_SILENCE_MS = 1000  # 2026-10-17 拍板：1000ms（450 会把 ASR 切成碎片）

# 回填进度是“按已用时间推算”的假进度：上游 backfill_speakers 没有回调，且只读不改。
PROGRESS_INTERVAL_S = 5.0
PROGRESS_MIN_ESTIMATE_S = 60.0


def _notify(callback: Optional[Callable[[Dict], None]], ev: Dict) -> None:
    """回调隔离：HTTP 客户端断线/序列化失败不得杀死流水线。"""
    if callback is None:
        return
    try:
        callback(ev)
    except Exception:
        pass


def _estimate_backfill_seconds(processed_ms: int) -> float:
    """按已处理音频时长估算回填耗时：max(60s, 音频秒数 * 0.5 + 60s)。"""
    duration_s = max(0.0, processed_ms / 1000.0)
    return max(PROGRESS_MIN_ESTIMATE_S, duration_s * 0.5 + PROGRESS_MIN_ESTIMATE_S)


def _validate_wav(wav_path) -> None:
    import soundfile as sf

    info = sf.info(str(wav_path))
    if info.samplerate != SAMPLE_RATE:
        raise SystemExit(f"需要 16kHz，实际 {info.samplerate}Hz: {wav_path}")
    if info.channels != 1:
        raise SystemExit(f"需要单声道，实际 {info.channels} 声道: {wav_path}")


def iter_audio_chunks(
    wav_path, chunk_ms: int = 600, max_seconds: float = 0.0
) -> Iterator[Tuple[bytes, bool]]:
    """按 chunk_ms 切 16k 单声道 int16；最后一块的 is_last 为 True。"""
    import soundfile as sf

    if chunk_ms <= 0:
        raise SystemExit(f"chunk_ms 必须为正数: {chunk_ms}")
    _validate_wav(wav_path)

    frames = int(max_seconds * SAMPLE_RATE) if max_seconds > 0 else -1
    audio, _ = sf.read(str(wav_path), dtype="int16", always_2d=True, frames=frames)
    audio = audio[:, 0]

    step = max(1, int(SAMPLE_RATE * chunk_ms / 1000))
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        yield chunk.tobytes(), (start + step) >= len(audio)


def _default_session(wav_path, models):
    from streaming_asr_engine import StreamingSession
    return StreamingSession(str(wav_path),
                            asr_model=models["stream_asr"],
                            vad_model=models["vad"],
                            max_end_silence_time=MAX_END_SILENCE_MS)


def run(
    wav,
    session_dir,
    *,
    chunk_ms: int = 600,
    fast: bool = False,
    no_llm: bool = False,
    no_diar: bool = False,
    max_seconds: float = 0.0,
    models: Optional[Dict] = None,
    session_factory: Optional[Callable] = None,
    polisher_factory: Optional[Callable] = None,
    backfill_fn: Optional[Callable] = None,
    on_event: Optional[Callable[[Dict], None]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    _validate_wav(wav)
    bootstrap.install()
    from session_store import SessionStore

    store = SessionStore(Path(session_dir))

    # 复用已有 session 目录时，派生/追加文件必须清空，避免继承旧修正/标签或重复行
    for stale in ("polished.json", "diarization.json", "utterances.jsonl", "events.ndjson"):
        (store.dir / stale).unlink(missing_ok=True)

    if models is None:
        from web_server import get_models
        models = get_models(need_stream=True, need_offline=False)

    session = (session_factory or _default_session)(store.audio_path, models)

    # 每场会议一个 turn 归属器：只在对外通知时补充字段，落盘事件保持单句原形
    tracker = StreamTurnTracker()

    def emit(ev: Dict) -> None:
        _notify(on_event, tracker.assign(ev))

    polished_buffer: Dict[str, Dict] = {}

    def on_polish(ev: Dict) -> None:
        store.append_event(ev)
        polished_buffer[ev["utt_id"]] = {
            "polished_text": ev["polished_text"],
            "corrected": ev["corrected"],
            "bypassed": ev["bypassed"],
            "error": ev["error"],
        }
        # 每句完成即落盘：崩了也不丢已润色的部分
        store.write_polished(polished_buffer)
        store.build_session_json()
        # 落盘之后再通知外部（HTTP 客户端断线由 _notify 吞掉）
        emit(ev)

    polisher = None
    if not no_llm:
        # 延迟导入：上游 llm_correct 损坏时，--no-llm 仍必须可用
        from stream_llm_polish import Polisher
        polisher = (polisher_factory or Polisher)(on_polish)

    def handle(ev: Dict) -> None:
        if ev.get("t") == "asr_final":
            store.append_utterance(ev)
            store.append_event(ev)
            # 先发 asr_final 再 submit：FakePolisher/异常路径可能同步回调 llm_confirmed，
            # 事件顺序必须保持 asr_final -> llm_confirmed。
            emit(ev)
            if polisher is not None:
                polisher.submit(ev)
        else:
            store.append_event(ev)
            emit(ev)

    clock = time.monotonic()
    fed_seconds = 0.0
    fed_any = False
    stopped_early = False
    try:
        for chunk, is_last in iter_audio_chunks(wav, chunk_ms, max_seconds):
            # 用户点“结束会议”：停止喂流，走与自然结束相同的收尾与回填路径
            if should_stop is not None and should_stop():
                stopped_early = True
                break
            fed_any = True
            for ev in session.feed_pcm_chunk(chunk, is_final=is_last):
                handle(ev)
            if not fast:
                fed_seconds += len(chunk) / 2 / SAMPLE_RATE
                delay = fed_seconds - (time.monotonic() - clock)
                if delay > 0:
                    time.sleep(delay)
        # 0 采样 WAV 或提前结束：显式补一次收尾，确保 session 关闭录音文件
        if not fed_any or stopped_early:
            for ev in session.feed_pcm_chunk(b"", is_final=True):
                handle(ev)
    except BaseException:
        # 送流中途失败也尽量留下可恢复的 transcript；收尾失败不得掩盖原始异常
        try:
            for ev in session.feed_pcm_chunk(b"", is_final=True):
                handle(ev)
        except Exception:
            pass
        store.write_transcript(session.completed_utterances, duration_ms=session.processed_ms)
        store.build_session_json()
        if polisher is not None:
            # 异常路径不等待整条润色队列，`on_polish` 已逐句落盘
            results = polisher.close(timeout=5)
            store.write_polished(results)
            store.build_session_json()
        raise

    # 正常路径：先冻结 transcript，再让回填与润色并发
    store.write_transcript(session.completed_utterances, duration_ms=session.processed_ms)
    store.build_session_json()

    try:
        if not no_diar:
            from diar_backfill import run_backfill

            # 回填进度只能按时间估算：上游没有进度回调。心跳线程发估算值，
            # 结束前必须 stop + join，保证“完成”是最后一条 progress。
            total = len(session.completed_utterances)
            estimate_s = _estimate_backfill_seconds(session.processed_ms)
            started_at = time.monotonic()
            stop = threading.Event()

            def emit_progress(elapsed: float) -> None:
                done = 0 if total <= 0 else min(
                    total - 1, max(0, int(total * elapsed / estimate_s))
                )
                _notify(on_progress, {
                    "t": "progress",
                    "stage": "diarization",
                    "done": done,
                    "total": total,
                    "estimated": True,
                    "msg": (f"说话人回填进行中（按时间估算，非真实进度）："
                            f"已用 {elapsed:.0f}s / 预计 {estimate_s:.0f}s"),
                })

            def heartbeat() -> None:
                while not stop.wait(PROGRESS_INTERVAL_S):
                    emit_progress(time.monotonic() - started_at)

            beat = None
            if on_progress is not None and total > 0:
                beat = threading.Thread(target=heartbeat, name="backfill-progress", daemon=True)
                beat.start()
                # 先给客户端一条 0 进度，避免开头 5 秒没有任何反馈
                emit_progress(0.0)

            try:
                run_backfill(store.dir, models=models, backfill=backfill_fn)
            finally:
                stop.set()
                if beat is not None:
                    beat.join()
            _notify(on_progress, {
                "t": "progress",
                "stage": "diarization",
                "done": total,
                "total": total,
                "msg": "说话人回填完成",
            })
    finally:
        if polisher is not None:
            # 收敛不变量：回填 -> close -> write_polished -> build_session_json 是最终合并顺序，
            # 跨实例（润色回调线程 vs 回填路径）的 merge 竞争依赖该顺序才无害；不要重排。
            results = polisher.close(timeout=None)
            store.write_polished(results)
            store.build_session_json()

    return store.read_json("session.json") or {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v0 端到端 CLI 转写")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--session-root", default="output/sessions")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--chunk-ms", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--fast", action="store_true", help="全速推流，不按墙钟节流")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-diar", action="store_true")
    args = parser.parse_args(argv)

    bootstrap.install()
    drift = bootstrap.upstream_fingerprint()
    if drift:
        print("[warn] 上游依赖相对基线已变更，本机结果可能不可复现：", file=sys.stderr, flush=True)
        for _name, _status in drift.items():
            print(f"  {_name}: {_status}", file=sys.stderr, flush=True)
    from session_store import new_session_dir

    session_dir = (Path(args.session_dir) if args.session_dir
                   else new_session_dir(Path(args.session_root)))

    started = time.time()
    obj = run(
        Path(args.wav), session_dir,
        chunk_ms=args.chunk_ms, fast=args.fast,
        no_llm=args.no_llm, no_diar=args.no_diar,
        max_seconds=args.max_seconds,
    )
    elapsed = time.time() - started

    utts = obj.get("utterances", [])
    speakers = sorted({u["spk_label"] for u in utts if u.get("spk_label")})
    summary = {
        "session_dir": str(session_dir),
        "status": obj.get("status"),
        "utterances": len(utts),
        "speakers": speakers,
        "duration_ms": obj.get("duration_ms"),
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
