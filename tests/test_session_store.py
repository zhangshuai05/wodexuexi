# -*- coding: utf-8 -*-
import json
from datetime import datetime

from session_store import SessionStore, new_session_dir


def _utterance(uid="utt_0001", start=0, end=1000, text="你好"):
    return {"utt_id": uid, "start_ms": start, "end_ms": end, "raw_text": text}


def test_transcript_only_falls_back_to_raw(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)

    obj = store.build_session_json()

    assert obj["status"] == "transcript"
    assert obj["utterances"][0]["polished_text"] == "你好"
    assert obj["utterances"][0]["corrected"] is False
    assert obj["utterances"][0]["bypassed"] is False
    assert obj["utterances"][0]["spk"] is None
    assert obj["utterances"][0]["spk_label"] is None
    assert obj["utterances"][0]["overlap_spk"] is None


def test_polished_is_applied(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.write_polished({
        "utt_0001": {"polished_text": "你好。", "corrected": True,
                     "bypassed": False, "error": None},
    })

    obj = store.build_session_json()

    assert obj["utterances"][0]["polished_text"] == "你好。"
    assert obj["utterances"][0]["corrected"] is True
    assert obj["status"] == "transcript", "润色不改变 status"


def test_diarization_sets_status_and_fields(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.write_diarization({
        "utt_0001": {"spk": 0, "spk_label": "speaker_0"},
    })

    obj = store.build_session_json()

    assert obj["status"] == "diarized"
    assert obj["utterances"][0]["spk"] == 0
    assert obj["utterances"][0]["spk_label"] == "speaker_0"


def test_build_is_idempotent(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)

    first = store.build_session_json()
    second = store.build_session_json()

    assert first == second


def test_append_only_files(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance(_utterance())
    store.append_utterance(_utterance("utt_0002", 1000, 2000, "再见"))
    store.append_event({"t": "asr_partial", "utt_id": "utt_0001", "text": "你"})

    lines = (store.dir / "utterances.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["utt_id"] == "utt_0002"
    assert len((store.dir / "events.ndjson").read_text(encoding="utf-8").splitlines()) == 1


def test_no_tmp_files_left_behind(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.build_session_json()

    assert list(store.dir.glob("*.tmp")) == []


def test_new_session_dir_collision(tmp_path):
    when = datetime(2026, 10, 17, 14, 32, 0)
    first = new_session_dir(tmp_path, when=when)
    second = new_session_dir(tmp_path, when=when)

    assert first.name == "s_20261017_143200"
    assert second.name == "s_20261017_143200_1"
