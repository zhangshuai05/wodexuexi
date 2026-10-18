# -*- coding: utf-8 -*-
import json
from pathlib import Path

import bootstrap

bootstrap.install()

import pytest

from diar_backfill import load_utterances, run_backfill, to_contract  # noqa: E402
from session_store import SessionStore  # noqa: E402


def test_to_contract_converts_one_based_label_to_zero_based_spk():
    out = to_contract({"utt_0001": "Speaker_3"})
    assert out == {"utt_0001": {"spk": 2, "spk_label": "speaker_2"}}


def test_to_contract_rejects_speaker_zero():
    with pytest.raises(ValueError):
        to_contract({"u": "Speaker_0"})


def test_to_contract_rejects_malformed_label():
    with pytest.raises(ValueError):
        to_contract({"u": "spk1"})


def test_to_contract_handles_two_speakers():
    out = to_contract({"a": "Speaker_1", "b": "Speaker_2"})
    assert out["a"] == {"spk": 0, "spk_label": "speaker_0"}
    assert out["b"] == {"spk": 1, "spk_label": "speaker_1"}


def test_load_utterances_reads_jsonl_and_keeps_last(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "旧"})
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "新"})
    store.append_utterance({"utt_id": "u2", "start_ms": 10, "end_ms": 20, "raw_text": "二"})

    utts = load_utterances(store.dir)

    assert [u["utt_id"] for u in utts] == ["u1", "u2"]
    assert utts[0]["raw_text"] == "新"
    assert set(utts[0]) == {"utt_id", "start_ms", "end_ms", "raw_text"}


def test_load_utterances_skips_corrupt_lines(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "好"})
    with (store.dir / "utterances.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"utt_id": "u2", "start_ms": 10, "end_ms"')  # 截断的损坏行

    utts = load_utterances(store.dir)

    assert [u["utt_id"] for u in utts] == ["u1"]


def test_load_utterances_falls_back_on_empty_jsonl(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    (store.dir / "utterances.jsonl").write_text("", encoding="utf-8")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "好"}],
        duration_ms=10,
    )

    utts = load_utterances(store.dir)

    assert [u["utt_id"] for u in utts] == ["u1"]
    assert utts[0]["raw_text"] == "好"


def test_run_backfill_writes_contract_and_merges(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"}],
        duration_ms=100,
    )
    store.audio_path.write_bytes(b"RIFF")

    called = {}

    def fake_backfill(wav_path, utterances, models, mode):
        called["args"] = (Path(wav_path).name, [u["utt_id"] for u in utterances], models, mode)
        return {"u1": "Speaker_2"}

    out = run_backfill(store.dir, models={"vad": "x", "spk": "y"},
                       backfill=fake_backfill)

    assert called["args"] == ("session_audio.wav", ["u1"], {"vad": "x", "spk": "y"}, "auto")
    assert out == {"u1": {"spk": 1, "spk_label": "speaker_1"}}
    assert json.loads((store.dir / "diarization.json").read_text(encoding="utf-8")) == out
    session = store.build_session_json()
    assert session["status"] == "diarized"
    assert session["utterances"][0]["spk"] == 1


def test_run_backfill_is_idempotent_unless_forced(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"}],
        duration_ms=100,
    )
    store.audio_path.write_bytes(b"RIFF")
    store.write_diarization({"u1": {"spk": 0, "spk_label": "speaker_0"}})

    def should_not_run(*args, **kwargs):
        raise AssertionError("已有 diarization.json 时不应重跑")

    out = run_backfill(store.dir, backfill=should_not_run)
    assert out == {"u1": {"spk": 0, "spk_label": "speaker_0"}}


def test_run_backfill_requires_audio(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([], duration_ms=0)

    with pytest.raises(SystemExit):
        run_backfill(store.dir, backfill=lambda *a: {})


def test_run_backfill_empty_utterances_writes_empty_mapping(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([], duration_ms=0)
    store.audio_path.write_bytes(b"RIFF")

    out = run_backfill(store.dir, backfill=lambda *a: {})
    assert out == {}


def test_rerun_rebuilds_session_json(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"}],
        duration_ms=100,
    )
    store.audio_path.write_bytes(b"RIFF")
    store.write_diarization({"u1": {"spk": 0, "spk_label": "speaker_0"}})
    store.build_session_json()
    (store.dir / "session.json").unlink()

    out = run_backfill(store.dir)

    assert out == {"u1": {"spk": 0, "spk_label": "speaker_0"}}
    assert (store.dir / "session.json").exists(), "幂等路径也必须重建 session.json"
    session = json.loads((store.dir / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "diarized"


def test_crash_recovery_reconstructs_transcript(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"})
    store.audio_path.write_bytes(b"RIFF")

    run_backfill(store.dir, backfill=lambda *a: {"u1": "Speaker_1"})

    assert (store.dir / "transcript.json").exists(), "崩溃恢复必须重建 transcript.json"
    session = json.loads((store.dir / "session.json").read_text(encoding="utf-8"))
    assert len(session["utterances"]) == 1
    assert session["status"] == "diarized"


def test_load_utterances_handles_truncated_utf8(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "好"})
    truncated = '{"utt_id": "u2", "start_ms": 10, "end_ms": 20, "raw_text": "你'.encode("utf-8")[:-1]
    with (store.dir / "utterances.jsonl").open("ab") as handle:
        handle.write(truncated)  # 末尾是被截断的多字节字符，无换行

    utts = load_utterances(store.dir)

    assert [u["utt_id"] for u in utts] == ["u1"]


def test_to_contract_accepts_rich_result_with_sub_turns():
    raw = {
        "mapping": {"u1": "Speaker_2"},
        "segments": {},
        "sub_turns": {"u1": [
            {"start_ms": 100, "end_ms": 1100, "speaker": "Speaker_3"},
        ]},
    }

    out = to_contract(raw)

    assert out["u1"]["spk"] == 1
    assert out["u1"]["spk_label"] == "speaker_1"
    assert out["u1"]["sub_turns"] == [
        {"start_ms": 100, "end_ms": 1100, "spk": 2, "spk_label": "speaker_2"}]


def test_run_backfill_writes_sub_turns_and_expands_session(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 2000, "raw_text": "甲乙"}],
        duration_ms=2000,
    )
    store.audio_path.write_bytes(b"RIFF")

    def fake(wav_path, utterances, models, mode):
        return {"mapping": {"u1": "Speaker_1"},
                "sub_turns": {"u1": [{"start_ms": 500, "end_ms": 1500,
                                      "speaker": "Speaker_2"}]}}

    out = run_backfill(store.dir, backfill=fake)

    assert out["u1"]["sub_turns"][0]["spk_label"] == "speaker_1"
    session = store.build_session_json()
    assert [u["utt_id"] for u in session["utterances"]] == [
        "u1#1", "u1#2", "u1#3"]
    assert [u["spk_label"] for u in session["utterances"]] == [
        "speaker_0", "speaker_1", "speaker_0"]
