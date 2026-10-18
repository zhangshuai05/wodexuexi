from types import SimpleNamespace
import sys

import numpy as np
import soundfile as sf

import claude_a2_v10 as v10


def test_v10_uses_stream_only_as_draft_and_v7_as_endpoint(tmp_path, monkeypatch):
    wav = tmp_path / "sample.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(wav, audio, 16000)
    seen = []

    def fake_stream(model, samples, sample_rate, realtime):
        seen.append((len(samples), sample_rate, realtime))
        return {
            "text": "stream draft",
            "events": [{"text": "stream draft"}],
            "inference_seconds": 0.1,
        }

    def fake_v7(path, models, lane, progress=None):
        assert lane == "c1"
        return {
            "method": "claude-a2-v7-c1",
            "diag": {"n_speakers": 2},
            "verdict": "reliable",
            "verdict_reasons": [],
            "sentence_info": [
                {"start": 0.0, "end": 1.0, "spk": 1, "text": "endpoint final"}
            ],
        }

    monkeypatch.setattr(v10, "stream_replay", fake_stream)
    monkeypatch.setitem(
        sys.modules,
        "claude_a2_v7_v8",
        SimpleNamespace(run_claude_lane=fake_v7),
    )
    result = v10.run_claude_lane(
        str(wav), {"stream_asr": object(), "asr": object()}, "v10"
    )

    assert seen == [(16000, 16000, False)]
    assert result["sentence_info"][0]["text"] == "endpoint final"
    assert result["sentence_info"][0]["asr_state"] == "offline_reference"
    assert result["diag"]["draft_text"] == "stream draft"
    assert result["diag"]["final_asr"] == "offline-asr-reference"
