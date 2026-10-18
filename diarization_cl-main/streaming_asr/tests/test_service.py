from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import claude_a2
from claude_a2_v7_v8 import _absorb_slivers, _rescue_gate, _stabilize_labels
from web_server import cached_denoise, select_denoise_mode, select_lanes


def test_visible_lanes_include_v10_two_pass_v11_and_v12_meeting():
    assert list(claude_a2.visible_versions()) == ["v7", "v8", "v9", "v10", "v11", "v12"]
    assert "两遍对照实验" in claude_a2.visible_versions()["v10"]["label"]
    assert "自动人数" in claude_a2.visible_versions()["v11"]["label"]
    assert "2秒窗" in claude_a2.visible_versions()["v12"]["label"]


def test_lane_selection_preserves_display_order_and_rejects_unknown():
    assert [key for key, _ in select_lanes({"lanes": ["v10,v7"]})] == ["v7", "v10"]
    with pytest.raises(ValueError, match="不支持的方法"):
        select_lanes({"lanes": ["cur"]})


def test_denoise_mode_defaults_to_auto_and_rejects_unknown():
    assert select_denoise_mode({}) == "auto"
    assert select_denoise_mode({"denoise": ["on"]}) == "on"
    assert select_denoise_mode({"denoise": ["off"]}) == "off"
    with pytest.raises(ValueError, match="denoise"):
        select_denoise_mode({"denoise": ["maybe"]})


def test_denoise_cache_reuses_the_first_result(tmp_path):
    source = tmp_path / "input.wav"
    source.write_bytes(b"normalized-audio")
    calls = []

    def fake_denoise(src, dst):
        calls.append(src)
        Path(dst).write_bytes(Path(src).read_bytes() + b"-frozen")

    first, hit1, source_hash1 = cached_denoise(str(source), fake_denoise, str(tmp_path / "cache"))
    second, hit2, source_hash2 = cached_denoise(str(source), fake_denoise, str(tmp_path / "cache"))
    assert (hit1, hit2) == (False, True)
    assert first == second and source_hash1 == source_hash2
    assert Path(first).read_bytes() == b"normalized-audio-frozen"
    assert len(calls) == 1


def test_dispatch_uses_each_frozen_implementation(monkeypatch):
    calls = []

    def fake_v8(*args, **kwargs):
        calls.append(("v7_v8", args[2]))
        return {"method": args[2]}

    def fake_v9(*args, **kwargs):
        calls.append(("v9", args[2]))
        return {"method": args[2]}

    def fake_v10(*args, **kwargs):
        calls.append(("v10", args[2]))
        return {"method": args[2]}

    def fake_v11(*args, **kwargs):
        calls.append(("v11", args[2]))
        return {"method": args[2]}

    monkeypatch.setitem(sys.modules, "claude_a2_v7_v8", SimpleNamespace(run_claude_lane=fake_v8))
    monkeypatch.setitem(sys.modules, "claude_a2_v9", SimpleNamespace(run_claude_lane=fake_v9))
    monkeypatch.setitem(sys.modules, "claude_a2_v10", SimpleNamespace(run_claude_lane=fake_v10))
    monkeypatch.setitem(sys.modules, "claude_a2_v11", SimpleNamespace(run_claude_lane=fake_v11))
    claude_a2.run_claude_lane("x.wav", {}, "v7")
    claude_a2.run_claude_lane("x.wav", {}, "v8")
    claude_a2.run_claude_lane("x.wav", {}, "v9")
    claude_a2.run_claude_lane("x.wav", {}, "v10")
    claude_a2.run_claude_lane("x.wav", {}, "v11")
    assert calls == [
        ("v7_v8", "c1"), ("v7_v8", "v8"), ("v9", "v9"), ("v10", "v10"),
        ("v11", "v11")
    ]


def test_multiwindow_short_reply_is_not_absorbed():
    import numpy as np

    labels = np.array([0, 0, 1, 1, 0, 0])
    margins = np.array([-0.5, -0.5, 0.1, 0.1, -0.5, -0.5])
    times = np.arange(len(labels)) * 0.15
    assert _stabilize_labels(labels, margins).tolist() == labels.tolist()
    absorbed, _ = _absorb_slivers(labels, margins, times, win_s=0.6)
    assert absorbed.tolist() == labels.tolist()


def test_v7_rescue_gate_rejects_dense_switches_against_undercut_window():
    candidate = [(i, i + 0.5, i % 2) for i in range(5)]
    dense, window, passed = _rescue_gate(
        candidate, [0, 0, 1, 1], [0.0, 0.6, 1.2, 1.8], 1.2
    )
    assert (dense, window, passed) == (4, 1, False)


def test_frontend_is_copied_comparison_ui_without_removed_services():
    html = Path("diar_web.html").read_text(encoding="utf-8")
    assert 'id="drop"' in html
    assert 'id="player"' in html
    assert 'id="cards"' in html
    assert "fetch('/api/lanes')" in html
    assert "'/api/compare?name='" in html
    assert 'id="denoiseSeg"' in html
    assert "&denoise=" in html
    assert "/api/llm" in html
    assert "LLM 校正" in html
    assert "fmtTs(s.start)" in html and "fmtTs(s.end)" in html
    assert "d.diag.rescue_decision" in html
    for removed in ("DSH", "snapshot_v", "reassign"):
        assert removed not in html
