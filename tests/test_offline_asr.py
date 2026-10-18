# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import bootstrap

bootstrap.install()

from offline_asr import (  # noqa: E402
    _agc,
    _group_utterances,
    _split_ranges,
    _split_text,
    transcribe_utterances,
)


def _wav(path, seconds=3.0):
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype=np.int16),
             16000, subtype="PCM_16")


def test_split_ranges_caps_segment_length():
    assert _split_ranges(0, 5000, 8000) == [(0, 5000)]

    ranges = _split_ranges(0, 17000, 8000)

    assert len(ranges) == 3
    assert all(b - a <= 8000 for a, b in ranges)
    assert ranges[0][0] == 0 and ranges[-1][1] == 17000


def test_long_utterance_is_split_and_joined(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav)
    calls = []

    def fake(path):
        calls.append(path)
        return "字"

    out = transcribe_utterances(
        wav, [{"utt_id": "u1", "start_ms": 0, "end_ms": 3000}],
        asr_call=fake, max_segment_ms=1000,
    )

    assert len(calls) == 3
    assert out["u1"] == {"text": "字，字，字", "start_ms": 0, "end_ms": 3000}


def test_abort_stops_without_calls(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav)
    calls = []

    out = transcribe_utterances(
        wav, [{"utt_id": "u1", "start_ms": 0, "end_ms": 1000}],
        asr_call=lambda path: calls.append(path) or "字",
        should_abort=lambda: True,
    )

    assert calls == []
    assert out == {}


def test_segment_failure_is_skipped(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav)

    def boom(path):
        raise RuntimeError("model down")

    out = transcribe_utterances(
        wav, [{"utt_id": "u1", "start_ms": 0, "end_ms": 1000}], asr_call=boom,
    )

    assert out == {}


def test_degenerate_segment_is_skipped(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav)
    calls = []

    out = transcribe_utterances(
        wav, [{"utt_id": "u1", "start_ms": 0, "end_ms": 0}],
        asr_call=lambda path: calls.append(path) or "字",
    )

    assert calls == []
    assert out == {}


def test_progress_reports_totals(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav)
    seen = []

    transcribe_utterances(
        wav, [{"utt_id": "u1", "start_ms": 0, "end_ms": 1000}],
        asr_call=lambda path: "字", on_progress=seen.append,
    )

    assert seen[-1]["stage"] == "final_asr"
    assert seen[-1]["done"] == 1
    assert seen[-1]["total"] == 1


def test_agc_boosts_weak_clip_to_target_rms():
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(16000) * 0.01).astype(np.float32)
    stats = {}

    out = _agc(clip, stats)

    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert 0.045 <= rms <= 0.055
    assert float(np.max(np.abs(out))) <= 0.951
    assert stats["agc_clips"] == 1


def test_agc_leaves_normal_level_clip_unchanged():
    rng = np.random.default_rng(1)
    clip = (rng.standard_normal(16000) * 0.2).astype(np.float32)

    out = _agc(clip.copy(), {})

    assert np.array_equal(out, clip)


def test_group_utterances_adjacent_and_same_speaker():
    def u(uid, start, end, spk=None):
        return {"utt_id": uid, "start_ms": start, "end_ms": end, "spk_label": spk}

    groups = _group_utterances([
        u("a", 0, 1000), u("b", 1500, 2500), u("c", 5000, 6500)])
    assert [len(g) for g in groups] == [2, 1]

    groups = _group_utterances([
        u("a", 0, 1000, "s0"), u("b", 3000, 4000, "s0")])
    assert [len(g) for g in groups] == [2]

    groups = _group_utterances([
        u("a", 0, 1000, "s0"), u("b", 3000, 4000, "s1")])
    assert [len(g) for g in groups] == [1, 1]


def test_group_utterances_respects_max_segment():
    def u(uid, start):
        return {"utt_id": uid, "start_ms": start, "end_ms": start + 5000}

    groups = _group_utterances([
        u("a", 0), u("b", 5100), u("c", 10200), u("d", 15300)])

    assert [len(g) for g in groups] == [3, 1]
    for group in groups:
        assert sum(x["end_ms"] - x["start_ms"] for x in group) <= 15000


def test_split_text_preserves_total_and_order():
    parts = _split_text("abcdef", [1, 1, 1])

    assert parts == ["ab", "cd", "ef"]
    assert "".join(parts) == "abcdef"


def test_grouped_decode_assigns_every_utt(tmp_path):
    wav = tmp_path / "a.wav"
    _wav(wav, seconds=6)
    calls = []

    def fake(path):
        calls.append(path)
        return "一二三四"

    out = transcribe_utterances(
        wav,
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 2000},
         {"utt_id": "u2", "start_ms": 2100, "end_ms": 4000}],
        asr_call=fake,
    )

    assert set(out) == {"u1", "u2"}
    assert "".join(out[uid]["text"] for uid in ("u1", "u2")) == "一二三四"
    assert len(calls) == 1


def test_group_utterances_never_joins_different_speakers():
    def u(uid, start, end, spk):
        return {"utt_id": uid, "start_ms": start, "end_ms": end, "spk_label": spk}

    # sub_turns 拆出的子段首尾相接；不同说话人即使 gap=0 也必须各自成组
    groups = _group_utterances([
        u("a", 0, 1000, "s0"),
        u("b", 1000, 2000, "s1"),
        u("c", 2000, 3000, "s0"),
    ])

    assert [len(g) for g in groups] == [1, 1, 1]
