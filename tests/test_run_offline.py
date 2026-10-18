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
        chunk_ms=200, fast=True, no_final_polish=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        backfill_fn=lambda wav_path, utts, models, mode: {"utt_0001": "Speaker_1"},
        models={"vad": None, "spk": None}, no_final_asr=True,
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
        models={}, no_final_asr=True,
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"][0]["spk"] is None
    assert not (session_dir / "diarization.json").exists()


def test_events_ndjson_contains_all_events(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_3"

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_diar=True, no_final_polish=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        models={}, no_final_asr=True,
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
        models={}, no_final_asr=True,
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"], "0 采样 WAV 也必须走一次 is_final 收尾"


def test_default_session_pins_end_silence(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeStreamingSession:
        def __init__(self, wav_path, asr_model=None, vad_model=None,
                     max_end_silence_time=None):
            captured["max_end_silence_time"] = max_end_silence_time

    fake = types.ModuleType("streaming_asr_engine")
    fake.StreamingSession = FakeStreamingSession
    monkeypatch.setitem(sys.modules, "streaming_asr_engine", fake)

    run_offline._default_session("x.wav", {"stream_asr": None, "vad": None})

    assert captured["max_end_silence_time"] == 1000, "断句阈值必须显式 pin，不能继承上游默认值"


def test_multichannel_wav_uses_channel_0(tmp_path):
    wav = tmp_path / "stereo.wav"
    data = np.zeros((16000, 2), dtype=np.int16)
    data[:, 1] = 1234
    sf.write(str(wav), data, 16000, subtype="PCM_16")

    chunks = list(run_offline.iter_audio_chunks(wav, 600))
    fed = np.frombuffer(b"".join(chunk for chunk, _ in chunks), dtype=np.int16)

    assert fed.size == 16000
    assert not fed.any(), "必须只取 ch0，不能把右声道混进来"


class _FlakySession(FakeSession):
    """第二次 feed 抛异常，用于验证送流中断时的兜底落盘。"""

    def __init__(self, wav_path, fail_on=2):
        super().__init__(wav_path)
        self.calls = 0
        self.fail_on = fail_on

    def feed_pcm_chunk(self, pcm_bytes, is_final=False):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("boom")
        return super().feed_pcm_chunk(pcm_bytes, is_final=is_final)


def test_exception_mid_feed_still_writes_transcript(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav, seconds=2.0)
    session_dir = tmp_path / "s_boom"

    with pytest.raises(RuntimeError):
        run_offline.run(
            wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
            session_factory=lambda p, m: _FlakySession(p),
            models={}, no_final_asr=True,
        )

    assert (session_dir / "transcript.json").exists()


def test_rerun_clears_stale_derived_artifacts(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_stale"
    session_dir.mkdir()
    (session_dir / "polished.json").write_text(
        '{"utt_old": {"polished_text": "旧"}}', encoding="utf-8")
    (session_dir / "diarization.json").write_text(
        '{"utt_old": {"spk": 0, "spk_label": "speaker_0"}}', encoding="utf-8")
    (session_dir / "utterances.jsonl").write_text(
        '{"utt_id": "utt_old", "start_ms": 0, "end_ms": 10, "raw_text": "旧"}\n',
        encoding="utf-8")
    (session_dir / "events.ndjson").write_text('{"t": "old"}\n', encoding="utf-8")
    (session_dir / "turns_polished.json").write_text(
        '{"turn_old": "旧精修"}', encoding="utf-8")

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, no_final_asr=True,
    )

    assert not (session_dir / "diarization.json").exists()
    assert not (session_dir / "polished.json").exists()
    assert not (session_dir / "turns_polished.json").exists()
    utt_text = (session_dir / "utterances.jsonl").read_text(encoding="utf-8")
    event_text = (session_dir / "events.ndjson").read_text(encoding="utf-8")
    assert "utt_old" not in utt_text
    assert '"t": "old"' not in event_text
    assert len(utt_text.splitlines()) == 1, "旧行不得残留，新行不得重复"


def test_chunk_ms_must_be_positive(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)

    with pytest.raises(SystemExit):
        list(run_offline.iter_audio_chunks(wav, 0))


def test_on_event_receives_pipeline_events(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_hooks"
    seen = []

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_final_polish=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        backfill_fn=lambda wav_path, utts, models, mode: {"utt_0001": "Speaker_1"},
        models={}, no_final_asr=True,
        on_event=seen.append,
    )

    kinds = [ev["t"] for ev in seen]
    assert "asr_partial" in kinds
    assert "asr_final" in kinds
    assert "llm_confirmed" in kinds
    assert kinds.index("asr_final") < kinds.index("llm_confirmed"), (
        "同句 llm_confirmed 必须晚于 asr_final"
    )


def test_events_have_turn_fields(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_turn_fields"
    seen = []

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_final_polish=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        backfill_fn=lambda wav_path, utts, models, mode: {"utt_0001": "Speaker_1"},
        models={}, no_final_asr=True,
        on_event=seen.append,
    )

    assert seen
    assert all(ev.get("turn_id") for ev in seen), "每条对外事件都必须带 turn_id"
    for ev in seen:
        if ev["t"] in ("asr_partial", "asr_final"):
            assert "turn_text" in ev
    confirmed = next(ev for ev in seen if ev["t"] == "llm_confirmed")
    assert confirmed["turn_id"] == "turn_0001"
    assert confirmed["turn_polished_text"] == "你好。"


def test_backfill_failure_still_closes_polisher(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_bf_fail"
    created = []

    def factory(on_result):
        polisher = FakePolisher(on_result)
        created.append(polisher)
        return polisher

    def boom(wav_path, utts, models, mode):
        raise RuntimeError("backfill down")

    with pytest.raises(RuntimeError):
        run_offline.run(
            wav, session_dir, chunk_ms=200, fast=True,
            session_factory=lambda p, m: FakeSession(p),
            polisher_factory=factory,
            backfill_fn=boom,
            models={}, no_final_asr=True,
        )

    assert len(created) == 1
    assert created[0].closed is True, "回填异常时也必须关闭润色器"


def test_should_stop_still_freezes_and_writes_transcript(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav, seconds=1.0)
    session_dir = tmp_path / "s_stop"

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, no_final_asr=True,
        should_stop=lambda: True,
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"], "提前结束也必须冻结已锁定句子"


def test_should_stop_mid_feed_keeps_partial_transcript(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav, seconds=1.0)
    session_dir = tmp_path / "s_stop_mid"
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, no_final_asr=True,
        should_stop=stop_after_two,
    )

    assert obj["utterances"], "中途取消也必须冻结已锁定句子"
    assert obj["duration_ms"] < 1000, "中途取消后不应喂完整段音频"


def test_offline_second_pass_sets_final_text(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_final_asr"

    def fake_offline(audio_path, utterances, on_progress=None, should_abort=None):
        assert Path(audio_path).name == "session_audio.wav"
        return {u["utt_id"]: {"text": "最终稿。", "start_ms": u["start_ms"],
                              "end_ms": u["end_ms"]} for u in utterances}

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, offline_asr_fn=fake_offline,
    )

    assert obj["utterances"][0]["final_text"] == "最终稿。"
    assert obj["utterances"][0]["raw_text"] == "你好", "直播稿必须保留"
    assert obj["turns"][0]["text"] == "最终稿。"
    assert (session_dir / "final_text.json").exists()


def test_offline_second_pass_failure_keeps_live_text(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_final_asr_fail"

    def boom(audio_path, utterances, on_progress=None, should_abort=None):
        raise RuntimeError("model missing")

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, offline_asr_fn=boom,
    )

    assert obj["utterances"][0]["final_text"] is None
    assert obj["turns"][0]["text"] == "你好"
    assert not (session_dir / "final_text.json").exists()


def test_no_final_asr_skips_offline_pass(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_final_asr_skip"
    calls = []

    def fake_offline(audio_path, utterances, on_progress=None, should_abort=None):
        calls.append(utterances)
        return {}

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        no_final_asr=True,
        session_factory=lambda p, m: FakeSession(p),
        models={}, offline_asr_fn=fake_offline,
    )

    assert calls == []
    assert obj["utterances"][0]["final_text"] is None
    assert not (session_dir / "final_text.json").exists()


def test_final_polish_overrides_turn_text(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_final"
    seen_turns = []

    def fake_final(turns, on_progress=None, should_abort=None):
        seen_turns.extend(turns)
        first = turns[0]
        return {first["turn_id"]: {"text": "你好呀。",
                                   "start_ms": first["start_ms"],
                                   "end_ms": first["end_ms"]}}

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        final_polish_fn=fake_final,
        models={}, no_final_asr=True,
    )

    assert seen_turns[0]["text"] == "你好。", "精修输入必须是直播润色稿"
    assert obj["turns"][0]["text"] == "你好呀。"
    assert obj["utterances"][0]["raw_text"] == "你好", "raw 必须保留以便回溯"
    assert (session_dir / "turns_polished.json").exists()


def test_no_final_polish_skips_second_pass(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_final_skip"
    calls = []

    def fake_final(turns, on_progress=None, should_abort=None):
        calls.append(turns)
        return {}

    obj = run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_diar=True, no_final_polish=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        final_polish_fn=fake_final,
        models={}, no_final_asr=True,
    )

    assert calls == []
    assert obj["turns"][0]["text"] == "你好。"
    assert not (session_dir / "turns_polished.json").exists()
