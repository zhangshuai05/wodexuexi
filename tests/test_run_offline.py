# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import bootstrap

bootstrap.install()

import run_offline  # noqa: E402


class FakeSession:
    """按脚本吐出事件，模拟 StreamingSession 的接口。"""

    def __init__(self, wav_path):
        self.wav_path = Path(wav_path)
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        self.wav_path.write_bytes(b"RIFF")
        self.completed_utterances = []
        self.processed_ms = 0

    def feed_pcm_chunk(self, pcm_bytes, is_final=False):
        events = [{"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 0,
                   "current_ms": 100, "text": "你"}]
        if is_final:
            utt = {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 200, "raw_text": "你好"}
            self.completed_utterances.append(utt)
            self.processed_ms = 200
            events.append({"t": "asr_final", **utt})
        return events


class FakePolisher:
    def __init__(self, on_result):
        self.on_result = on_result
        self.submitted = []
        self.closed = False

    def submit(self, ev):
        self.submitted.append(ev)
        self.on_result({
            "t": "llm_confirmed", "utt_id": ev["utt_id"],
            "start_ms": ev["start_ms"], "end_ms": ev["end_ms"],
            "raw_text": ev["raw_text"], "polished_text": "你好。",
            "corrected": True, "bypassed": False, "error": None,
        })

    def close(self, timeout=None):
        self.closed = True
        return {"utt_0001": {"polished_text": "你好。", "corrected": True,
                             "bypassed": False, "error": None}}

    def results(self):
        return {}


def _write_wav(path: Path, seconds: float = 0.4):
    samples = int(16000 * seconds)
    sf.write(str(path), np.zeros(samples, dtype=np.int16), 16000, subtype="PCM_16")


def test_run_writes_all_contract_files(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_1"

    obj = run_offline.run(
        wav, session_dir,
        chunk_ms=200, fast=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        backfill_fn=lambda wav_path, utts, models, mode: {"utt_0001": "Speaker_1"},
        models={"vad": None, "spk": None},
    )

    files = {p.name for p in session_dir.iterdir()}
    assert {"utterances.jsonl", "events.ndjson", "transcript.json",
            "polished.json", "diarization.json", "session.json"} <= files

    assert obj["status"] == "diarized"
    utt = obj["utterances"][0]
    assert utt["raw_text"] == "你好"
    assert utt["polished_text"] == "你好。"
    assert utt["spk"] == 0
    assert utt["spk_label"] == "speaker_0"

    transcript = json.loads((session_dir / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["status"] == "transcript"
    assert transcript["duration_ms"] == 200


def test_no_diar_and_no_llm(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_2"

    obj = run_offline.run(
        wav, session_dir,
        chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={},
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"][0]["spk"] is None
    assert not (session_dir / "diarization.json").exists()


def test_events_ndjson_contains_all_events(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_3"

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        models={},
    )

    events = [json.loads(line) for line in
              (session_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()]
    kinds = [e["t"] for e in events]
    assert kinds.count("asr_partial") >= 2
    assert kinds.count("asr_final") == 1
    assert kinds.count("llm_confirmed") == 1


def test_iter_audio_chunks_rejects_non_16k(tmp_path):
    wav = tmp_path / "bad.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.int16), 8000, subtype="PCM_16")
    with pytest.raises(SystemExit):
        list(run_offline.iter_audio_chunks(wav, 600))


def test_iter_audio_chunks_respects_max_seconds(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav, seconds=2.0)
    chunks = list(run_offline.iter_audio_chunks(wav, 600, max_seconds=0.6))
    assert sum(len(c) for c, _ in chunks) == int(16000 * 0.6) * 2
    assert chunks[-1][1] is True


def test_zero_sample_wav_still_finalizes(tmp_path):
    wav = tmp_path / "empty.wav"
    _write_wav(wav, seconds=0.0)
    session_dir = tmp_path / "s_empty"

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={},
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"], "0 采样 WAV 也必须走一次 is_final 收尾"
