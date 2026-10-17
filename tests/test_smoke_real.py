# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

import bootstrap

bootstrap.install()

import run_offline  # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")


@pytest.mark.slow
@pytest.mark.skipif(not SMOKE.exists(), reason="缺少 smoke_clip_180s.wav")
def test_real_smoke_transcript_only(tmp_path):
    """只看转写链路；--no-diar 让这个测试在 1 分钟内可完成。"""
    obj = run_offline.run(
        SMOKE, tmp_path / "s_smoke",
        chunk_ms=600, fast=True, no_llm=True, no_diar=True,
        max_seconds=30.0,
    )
    assert obj["status"] == "transcript"
    assert obj["utterances"], "30 秒内应至少锁定一句"
    for utt in obj["utterances"]:
        assert utt["end_ms"] > utt["start_ms"] >= 0
        assert utt["raw_text"].strip()


@pytest.mark.slow
@pytest.mark.skipif(not SMOKE.exists(), reason="缺少 smoke_clip_180s.wav")
def test_real_smoke_full(tmp_path):
    """完整链路含回填。耗时约 1-5 分钟。"""
    obj = run_offline.run(
        SMOKE, tmp_path / "s_full",
        chunk_ms=600, fast=True, no_llm=True,
    )
    assert obj["status"] == "diarized"
    assert obj["utterances"], "完整链路必须产出至少一句，否则回填断言是真空通过"
    assert all(u["spk_label"] == f"speaker_{u['spk']}" for u in obj["utterances"])
    assert all(u["spk"] is not None for u in obj["utterances"])
    assert all(u["spk_label"].startswith("speaker_") for u in obj["utterances"])
