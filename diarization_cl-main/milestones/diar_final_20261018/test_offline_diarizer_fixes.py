# -*- coding: utf-8 -*-
"""2026-10-18 解冻修复的纯函数单元测试（无模型推理）。

覆盖：
  - 密集窗身份复核：3 个真簇 -> K=3；2 个簇 -> K=2；一人被拆成 3 团 -> 拒绝；
  - 弱音提拉：只提拉、不衰减、静音不动；
  - 句内 sub_turns：第二人 >=1.0s 才输出，短插话只留在 segments；
  - 全零 share 不再被当成"预测了簇 0"（§10.2 缺陷 1/2）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / ".zhang", ROOT / "AliMeeting", ROOT / "streaming_asr",
           ROOT / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from offline_diarizer import _normalize_audio, _refined_turn_segments  # noqa: E402
from run_3dspeaker_pipeline import (  # noqa: E402
    dense_identity_estimate, turns_multilabel)
from eval_multilabel import sentence_pred_sets  # noqa: E402


def _unit(vec):
    return vec / (np.linalg.norm(vec) + 1e-12)


def _blobs(k, per=200, dim=192, spread=0.25, seed=0):
    rng = np.random.RandomState(seed)
    centers = [_unit(rng.randn(dim)) for _ in range(k)]
    rows = [[_unit(c + spread * rng.randn(dim)) for _ in range(per)]
            for c in centers]
    return np.vstack(rows).astype(np.float32)


def test_dense_identity_recovers_three_speakers():
    est = dense_identity_estimate(_blobs(3, seed=0))
    assert est is not None and est["k"] == 3
    assert est["max_centroid_cos"] < 0.65


def test_dense_identity_does_not_split_one_speaker():
    # 一个人被拆成三个很近的团：K=3 质心余弦 >0.65，必须拒绝，停在 K=2
    rng = np.random.RandomState(1)
    base = _unit(rng.randn(192))
    pert = _unit(rng.randn(192)) * 0.15
    centers = [base, _unit(base + pert), _unit(base - pert)]
    rows = [[_unit(c + 0.05 * rng.randn(192)) for _ in range(200)]
            for c in centers]
    est = dense_identity_estimate(np.vstack(rows).astype(np.float32))
    assert est is not None and est["k"] == 2


def test_dense_identity_two_speakers_stay_two():
    est = dense_identity_estimate(_blobs(2, seed=2))
    assert est is not None and est["k"] == 2


def test_normalize_lifts_weak_audio_only():
    weak = np.array([0.0, 0.1, -0.2], dtype=np.float32)
    out, path = _normalize_audio(weak, 16000, 0.85)
    assert path is not None
    assert np.isclose(np.max(np.abs(out)), 0.85, atol=1e-6)
    Path(path).unlink()

    loud = np.array([0.0, 0.9, -1.0], dtype=np.float32)
    out2, path2 = _normalize_audio(loud, 16000, 0.85)
    assert path2 is None and out2 is loud

    silent = np.zeros(100, dtype=np.float32)
    out3, path3 = _normalize_audio(silent, 16000, 0.85)
    assert path3 is None and out3 is silent


def _insert_case(insert_end):
    """0.6s 窗 / 0.15s 步长；2.2s 起插入第二人，到 insert_end 结束。"""
    hop = 0.15
    times = np.arange(0.0, 5.4, hop)
    labels = np.array([1 if 2.2 <= t < insert_end else 0 for t in times])
    wins = [(float(t), float(t) + 0.6) for t in times]
    turns = [{"utt_id": "u1", "start": 0.0, "end": 6.0}]
    return _refined_turn_segments(turns, wins, labels, [0], hop,
                                  min_subturn_s=1.0, min_segment_s=0.3)


def test_sub_turn_requires_one_second():
    segs, subs = _insert_case(3.4)          # 第二人 1.2s
    assert segs["u1"] and subs["u1"]
    segs2, subs2 = _insert_case(2.8)        # 第二人 0.6s：只进 segments
    assert segs2["u1"]
    assert "u1" not in subs2
    assert any(not piece["is_main"] for piece in segs2["u1"])


def test_all_zero_share_is_not_a_prediction():
    binary = np.zeros((100, 2), dtype=np.float32)
    turns = [{"start": 0.0, "end": 5.0, "speaker": "A"},
             {"start": 5.0, "end": 10.0, "speaker": "B"}]
    result = turns_multilabel(binary, turns, 0.1)
    assert result["acc"] == 0.0 and result["coverage"] == 0.0
    preds = sentence_pred_sets(binary, turns, 0.1, 0.25, {0: "A", 1: "B"})
    assert all(p["primary"] is None and not p["pred"] for p in preds)
