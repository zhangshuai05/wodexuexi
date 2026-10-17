# -*- coding: utf-8 -*-
"""stream_service HTTP 契约测试：假流水线 + 临时端口，不加载任何真实模型。"""
import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import bootstrap

bootstrap.install()

import stream_service  # noqa: E402
from session_store import SessionStore  # noqa: E402


UTT = {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 800, "raw_text": "你好"}
POLISHED = {"polished_text": "你好。", "corrected": True, "bypassed": False, "error": None}
DIARIZATION = {"utt_0001": {"spk": 0, "spk_label": "speaker_0"}}


def _wav_bytes(seconds: float = 0.2) -> bytes:
    """16 kHz 单声道 WAV，让服务跳过 ffmpeg 转码。"""
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16000 * seconds), dtype=np.int16), 16000,
             subtype="PCM_16", format="WAV")
    return buffer.getvalue()


def _fake_pipeline(captured: dict):
    def fake(wav, session_dir, **kwargs):
        captured["wav"] = Path(wav)
        captured["wav_parent_existed"] = Path(wav).parent.is_dir()
        captured["session_dir"] = Path(session_dir)
        kwargs["on_event"]({"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 0,
                            "current_ms": 300, "text": "你"})
        kwargs["on_event"]({"t": "asr_final", **UTT})
        kwargs["on_event"]({"t": "llm_confirmed", **UTT, **POLISHED})
        kwargs["on_progress"]({"t": "progress", "stage": "diarization", "done": 0,
                               "total": 1, "estimated": True, "msg": "估算中"})

        store = SessionStore(Path(session_dir))
        store.write_transcript([UTT], duration_ms=800)
        store.write_polished({"utt_0001": POLISHED})
        store.write_diarization(DIARIZATION)
        return store.build_session_json()
    return fake


@pytest.fixture
def live_server(tmp_path, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(stream_service, "_PIPELINE", _fake_pipeline(captured))
    monkeypatch.setattr(stream_service, "_get_models", lambda: {"fake": True})
    monkeypatch.setattr(stream_service, "SESSION_ROOT", tmp_path / "sessions")

    server = stream_service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read()


def _post(url: str, body: bytes, headers=None):
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.headers, response.read()


def test_get_index_serves_html(live_server):
    base, _ = live_server
    status, body = _get(base + "/")
    assert status == 200
    assert "<title>会议实时记录" in body.decode("utf-8")


def test_health(live_server):
    base, _ = live_server
    status, body = _get(base + "/api/health")
    obj = json.loads(body.decode("utf-8"))
    assert status == 200
    assert obj["ok"] is True
    assert "busy" in obj


def test_stream_meeting_emits_contract_events(live_server):
    base, _ = live_server
    status, headers, body = _post(
        base + "/api/stream_meeting?fast=1", _wav_bytes(),
        {"Content-Type": "application/octet-stream"},
    )
    assert status == 200
    assert "application/x-ndjson" in headers.get("Content-Type", "")

    events = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    kinds = [ev["t"] for ev in events]
    for expected in ("asr_partial", "asr_final", "llm_confirmed", "progress",
                     "speaker_diarized", "finish"):
        assert expected in kinds, f"缺少事件 {expected}: {kinds}"

    diarized = next(ev for ev in events if ev["t"] == "speaker_diarized")
    assert diarized["n_clusters"] == 1
    assert diarized["mapping"] == DIARIZATION
    assert diarized["batch"] == {"done": 1, "total": 1}
    assert events[-1]["t"] == "finish"


def test_filename_is_sanitized(live_server):
    base, captured = live_server
    _post(base + "/api/stream_meeting?name=../../evil.wav", _wav_bytes(),
          {"Content-Type": "application/octet-stream"})

    wav = captured["wav"]
    assert wav.name == "evil.wav", "只保留 basename"
    assert ".." not in wav.parts
    assert wav.parent.name.startswith("meeting_upload_"), "文件必须落在临时上传目录"
    assert captured["wav_parent_existed"] is True
    assert not (Path.cwd() / "evil.wav").exists(), "客户端路径不得逃逸到工作目录"


def test_busy_returns_409(live_server):
    base, _ = live_server
    assert stream_service._SESSION_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(urllib.error.HTTPError) as info:
            _post(base + "/api/stream_meeting?fast=1", _wav_bytes(),
                  {"Content-Type": "application/octet-stream"})
        assert info.value.code == 409
        obj = json.loads(info.value.read().decode("utf-8"))
        info.value.close()
        assert obj["error"] == "已有会议在运行"
    finally:
        stream_service._SESSION_LOCK.release()


def test_unsupported_route_returns_404(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as info:
        _get(base + "/api/nope")
    assert info.value.code == 404
    info.value.close()
