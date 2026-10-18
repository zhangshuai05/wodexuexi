"""Regression: the VAD copy must stay the tuned local one (speech_noise_thres=0.7).

With the shared modelscope cache copy (0.6) this audio is treated as one
continuous speech segment and the lane output collapses (5 turns -> 3).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import web_server


def test_vad_uses_tuned_local_copy():
    cfg = os.path.join(web_server.VAD_LOCAL, "config.yaml")
    assert os.path.isfile(cfg), web_server.VAD_LOCAL
    assert "modelscope\\models" not in web_server.VAD_LOCAL
    with open(cfg, encoding="utf-8") as f:
        assert "speech_noise_thres: 0.7" in f.read()


def test_local_vad_copy_lives_next_to_project():
    assert os.path.isdir(web_server.VAD_LOCAL)