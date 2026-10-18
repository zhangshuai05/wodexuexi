import numpy as np

from claude_a2_v11 import (_block_candidates, _candidate_clusters, _largest_window_gaps,
                           _select_candidate, _split_for_asr)


def test_v11_selects_three_well_supported_clusters():
    rng = np.random.default_rng(7)
    bases = np.eye(3, dtype=np.float32)
    vectors = np.concatenate([base + rng.normal(0, 0.01, (12, 3)) for base in bases])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert _select_candidate(_candidate_clusters(vectors)[0])["k"] == 3


def test_v11_forced_three_mode_uses_the_three_cluster_candidate():
    rng = np.random.default_rng(8)
    vectors = np.concatenate([base + rng.normal(0, 0.01, (8, 3))
                              for base in np.eye(3, dtype=np.float32)])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert _select_candidate(_candidate_clusters(vectors)[0], forced_k=3)["k"] == 3


def test_v11_forced_three_keeps_an_under_supported_diagnostic_candidate():
    vectors = np.vstack([np.eye(3, dtype=np.float32)[0]] * 10 +
                        [np.eye(3, dtype=np.float32)[1]] * 10 +
                        [np.eye(3, dtype=np.float32)[2]])
    selected = _select_candidate(_candidate_clusters(vectors)[0], forced_k=3)
    assert selected["k"] == 3
    assert selected["eligible"] is False


def test_v11_gap_diagnostics_preserve_neighbor_labels():
    gaps = _largest_window_gaps(np.array([0.0, 0.15, 2.0]), np.array([0, 0, 2]))
    assert gaps[0] == {"after": 0.75, "before": 2.0, "gap": 1.25,
                       "left_label": 0, "right_label": 2}


def test_v11_block_diagnostics_split_only_on_long_vad_gaps():
    vectors = np.vstack([np.eye(2, dtype=np.float32)[0]] * 6 +
                        [np.eye(2, dtype=np.float32)[1]] * 6)
    times = np.r_[np.arange(6) * .15, 3.0 + np.arange(6) * .15]
    blocks = _block_candidates(vectors, times, gap_seconds=.75)
    assert [(block["n_windows"], block["selected_k"]) for block in blocks] == [(6, 1), (6, 1)]


def test_v11_asr_chunk_cap_preserves_speaker_and_full_range():
    segment = {"start": 0.0, "end": 20.0, "spk": 2}
    pieces = _split_for_asr(segment, [], max_seconds=8.0)
    assert [(piece["start"], piece["end"], piece["spk"]) for piece in pieces] == [
        (0.0, 8.0, 2), (8.0, 16.0, 2), (16.0, 20.0, 2)
    ]
    assert all(piece["asr_split"] for piece in pieces)
