# -*- coding: utf-8 -*-
"""端到端离线闭环：流式转写 → T+0 冻结稿 → 回填说话人 → 润色收尾 → session.json

    python .zhang/offline_e2e.py                       # 默认 smoke_clip_180s.wav
    python .zhang/offline_e2e.py --wav X.wav --session-dir output/sessions/s_x

⚠️ **这不是产品路径。** 产品路径是 `Meeting_main/run_offline.py`（有 turns.py、
回填心跳、ffmpeg 转码、更丰富的 mapping schema）。本文件是离线侧的**独立自检器**：
改完 `offline_diarizer.py` 后一条命令验端到端。之所以从 `run_offline.py` 改名，
是因为与 Meeting_main 的同名模块会互相遮蔽（见 HANDOFF_OFFLINE_DIAR_AGENT.md §1.2）。

阶段（与产品契约一致）：
    1. 流式：600ms 分片喂入，双写 session_audio.wav，事件写 events.ndjson，
       asr_final 写 utterances.jsonl 并压入异步润色队列（主线程零等待）。
    2. T+0：音频一结束立刻写 transcript.json（status="transcript"）。
    3. 回填 + 润色收尾：backfill_speakers 用同一份 models；润色队列在后台继续排空。
    4. 合成 session.json（status="diarized"）。
"""
import argparse
import json
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "streaming_asr")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import soundfile as sf  # noqa: E402

from streaming_asr_engine import StreamingSession  # noqa: E402
from streaming_llm_polish import StreamingPolisher  # noqa: E402
from offline_diarizer import backfill_speakers  # noqa: E402

DEFAULT_WAV = r"D:\Deepseek\yinp\smoke_clip_180s.wav"


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=DEFAULT_WAV)
    ap.add_argument("--session-dir",
                    default=os.path.join(_ROOT, "output", "sessions", "s_smoke"))
    ap.add_argument("--chunk-ms", type=int, default=600)
    ap.add_argument("--max-end-silence", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.session_dir, exist_ok=True)
    d = args.session_dir
    wav_out = os.path.join(d, "session_audio.wav")
    t_start = time.time()

    from web_server import get_models
    models = get_models(need_stream=True, need_offline=False)

    session = StreamingSession(wav_out,
                               asr_model=models["stream_asr"],
                               vad_model=models["vad"],
                               max_end_silence_time=args.max_end_silence)
    print("会话录音 → %s" % wav_out, flush=True)

    polished = {}
    # on_confirmed 跑在润色 worker 线程上，与主线程共用 events_f；TextIOWrapper 非线程安全
    events_lock = threading.Lock()

    def _emit_event(ev):
        with events_lock:
            events_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            events_f.flush()

    def on_confirmed(ev):
        polished[ev["utt_id"]] = ev
        _emit_event(ev)
        print("  🖊  %s  %s" % (ev["utt_id"], ev["polished_text"]), flush=True)

    events_f = open(os.path.join(d, "events.ndjson"), "w", encoding="utf-8")
    utts_f = open(os.path.join(d, "utterances.jsonl"), "w", encoding="utf-8")
    polisher = StreamingPolisher(on_confirmed)

    # ---------- 阶段 1：流式 ----------
    audio, sr = sf.read(args.wav, dtype="int16", always_2d=True)
    audio = audio[:, 0]
    if sr != session.SAMPLE_RATE:
        raise SystemExit(f"需要 {session.SAMPLE_RATE}Hz，实际 {sr}Hz：{args.wav}")
    if len(audio) == 0:
        raise SystemExit(f"输入 WAV 为空：{args.wav}")
    step = int(sr * args.chunk_ms / 1000)
    n_partial = 0
    for i in range(0, len(audio), step):
        is_end = (i + step) >= len(audio)
        for ev in session.feed_pcm_chunk(audio[i:i + step].tobytes(), is_final=is_end):
            _emit_event(ev)
            if ev["t"] == "asr_partial":
                n_partial += 1
            else:
                utts_f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                utts_f.flush()
                polisher.submit(ev)          # 主线程零等待
                print("  ✅ %s [%6d → %6d ms] %s" % (
                    ev["utt_id"], ev["start_ms"], ev["end_ms"], ev["raw_text"]), flush=True)

    n_utt = len(session.completed_utterances)
    print("\n流式结束：%d 句锁定 / %d 次草稿更新（%.1fs）" % (
        n_utt, n_partial, time.time() - t_start), flush=True)

    # ---------- 阶段 2：T+0 冻结稿 ----------
    _write_json(os.path.join(d, "transcript.json"), {
        "status": "transcript",
        "audio": wav_out,
        "n_utterances": n_utt,
        "utterances": session.completed_utterances,
    })
    print("T+0 transcript.json 已写出", flush=True)

    # ---------- 阶段 3：回填（润色队列在后台继续排空）----------
    segments, sub_turns = {}, {}
    try:
        result = backfill_speakers(wav_out, session.completed_utterances,
                                   models=models, mode="auto", with_segments=True)
        mapping = result["mapping"]
        segments = result["segments"]
        sub_turns = result["sub_turns"]
    except Exception as exc:
        # 回填是实验性环节，失败不该丢掉 T+0 的转写与润色产物
        print("[diarizer] 回填失败，退回单说话人：%s: %s"
              % (type(exc).__name__, exc), flush=True)
        mapping = {u["utt_id"]: "Speaker_1" for u in session.completed_utterances}
    _write_json(os.path.join(d, "diarization.json"),
                {"status": "diarized", "mapping": mapping,
                 "segments": segments, "sub_turns": sub_turns})
    n_sub = sum(len(v) for v in sub_turns.values())
    print("说话人回填完成：%d 句 → %d 人（%d 句含句内切分，%d 个稳定 sub_turn）"
          % (len(mapping), len(set(mapping.values())), len(sub_turns), n_sub),
          flush=True)

    polisher.close()                          # 等队列排空
    _write_json(os.path.join(d, "polished.json"),
                {"status": "polished", "items": list(polished.values())})
    utts_f.close()
    events_f.close()

    # ---------- 阶段 4：合成 ----------
    merged = []
    for u in session.completed_utterances:
        label = mapping.get(u["utt_id"], "Speaker_?")
        try:
            spk = int(label.rsplit("_", 1)[-1]) - 1
        except ValueError:            # "Speaker_?" 也要能落到 -1，不能崩
            spk = -1
        merged.append({
            "utt_id": u["utt_id"],
            "start_ms": u["start_ms"],
            "end_ms": u["end_ms"],
            "raw_text": u["raw_text"],
            "polished_text": polished.get(u["utt_id"], {}).get("polished_text", u["raw_text"]),
            "spk": spk,
            "spk_label": label,
            "segments": segments.get(u["utt_id"], []),
            "sub_turns": sub_turns.get(u["utt_id"], []),
        })
    _write_json(os.path.join(d, "session.json"), {
        "status": "diarized",
        "audio": wav_out,
        "n_utterances": len(merged),
        "n_speakers": len({m["spk_label"] for m in merged}),
        "utterances": merged,
    })

    print("\n" + "=" * 66)
    for m in merged:
        print("[%s] %7.1fs  %s" % (m["spk_label"], m["start_ms"] / 1000.0,
                                   m["polished_text"]))
    print("=" * 66)
    print("总耗时 %.1fs → %s" % (time.time() - t_start,
                              os.path.join(d, "session.json")), flush=True)


if __name__ == "__main__":
    main()
