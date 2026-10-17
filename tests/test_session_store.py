# -*- coding: utf-8 -*-
import json
import threading
from datetime import datetime

import pytest

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


def test_empty_diarization_does_not_flip_status(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.write_diarization({})

    obj = store.build_session_json()

    assert obj["status"] == "transcript", "空 mapping 不代表已回填"


def test_append_utterance_rejects_missing_keys(tmp_path):
    store = SessionStore(tmp_path / "s_1")

    with pytest.raises(ValueError):
        store.append_utterance({"utt_id": "utt_0001", "start_ms": 0, "raw_text": "你好"})


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


def test_concurrent_builds_from_two_instances_do_not_raise(tmp_path):
    store_a = SessionStore(tmp_path / "s_1")
    store_a.write_transcript([_utterance()], duration_ms=1000)
    store_b = SessionStore(tmp_path / "s_1")

    errors = []

    def _run(store):
        try:
            for _ in range(20):
                store.build_session_json()
        except Exception as exc:  # 收集异常，线程内不能直接断言
            errors.append(exc)

    threads = [
        threading.Thread(target=_run, args=(store_a,)),
        threading.Thread(target=_run, args=(store_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []


def test_concurrent_appends_do_not_corrupt_lines(tmp_path):
    import json
    import threading

    store = SessionStore(tmp_path / "s_1")

    def worker(n):
        for i in range(50):
            store.append_event({"t": "x", "n": n, "i": i})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (store.dir / "events.ndjson").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 100
    for line in lines:
        json.loads(line)  # 每行都必须是完整 JSON


def test_merge_lock_is_shared_across_instances(tmp_path):
    store_a = SessionStore(tmp_path / "s_1")
    store_b = SessionStore(tmp_path / "s_2")

    assert store_a._merge_lock is store_b._merge_lock


def test_new_session_dir_atomic_claim(tmp_path):
    when = datetime(2026, 10, 17, 14, 32, 0)
    results = []
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait()
        results.append(new_session_dir(tmp_path, when=when))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] != results[1], "并发认领必须得到不同目录"
    assert all(path.is_dir() for path in results)
