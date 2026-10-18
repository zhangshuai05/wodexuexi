import numpy as np

from v12_aishell4 import (DEV_GROUPS, HOLDOUT_GROUPS, cluster_metrics, eer, meeting_group,
                          pair_scores, pure_windows, split_meetings, threshold_metrics,
                          thresholds_from_report)


def test_v12_group_split_has_no_identity_group_leakage():
    assert DEV_GROUPS.isdisjoint(HOLDOUT_GROUPS)
    root = r"D:\yinp\AISHELL-4\data\test"
    dev, holdout = split_meetings(root, "dev"), split_meetings(root, "holdout")
    assert len(dev) == 12 and len(holdout) == 8
    assert {meeting_group(item) for item in dev} == DEV_GROUPS
    assert {meeting_group(item) for item in holdout} == HOLDOUT_GROUPS


def test_v12_pure_windows_exclude_overlap_and_first_ten_seconds():
    segments = [
        {"start": 4.0, "end": 14.0, "speaker": "a"},
        {"start": 12.0, "end": 13.0, "speaker": "b"},
        {"start": 15.0, "end": 18.0, "speaker": "b"},
    ]
    windows = pure_windows(segments, 1.0, per_speaker=20)
    assert windows
    assert all(start >= 10.0 for _, start, _ in windows)
    assert all(not (start < 13.0 and end > 12.0) for _, start, end in windows)


def test_v12_eer_is_zero_for_separated_scores():
    result = eer(np.array([0.8, 0.9]), np.array([0.1, 0.2]))
    assert result["eer"] == 0.0


def test_v12_pair_scores_use_each_pair_once():
    vectors = np.eye(4, dtype=np.float32)
    same, different = pair_scores(vectors, ["a", "a", "b", "b"])
    assert len(same) == 2
    assert len(different) == 4


def test_v12_oracle_clustering_recovers_separated_speakers():
    vectors = np.vstack([[np.eye(3, dtype=np.float32)[index]] * 4
                         for index in range(3)])
    vectors = vectors.reshape(12, 3)
    report = cluster_metrics(vectors, ["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    assert report["true_k"] == report["auto_k"] == 3
    assert report["oracle_k_accuracy"] == 1.0


def test_v12_oracle_accuracy_survives_true_k_above_max_k():
    """The oracle branch only fires at k == true_k, so the sweep must reach it."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(24, 8)).astype(np.float32)
    report = cluster_metrics(vectors, [f"s{index}" for index in range(12) for _ in range(2)],
                             max_k=5)
    assert report["true_k"] == 12
    assert max(item["k"] for item in report["candidates"]) == 12
    assert report["oracle_k_accuracy"] is not None


def test_v12_cluster_metrics_needs_two_vectors():
    assert cluster_metrics(np.zeros((1, 4), dtype=np.float32), ["a"]) is None


def test_v12_auto_k_ignores_singleton_clusters():
    """silhouette returns -1 for a degenerate k, which must never win the sweep."""
    vectors = np.vstack([[np.eye(2, dtype=np.float32)[index]] * 4 for index in range(2)])
    vectors = vectors.reshape(8, 2)
    report = cluster_metrics(vectors, ["a"] * 4 + ["b"] * 4)
    assert report["auto_k"] == 2
    assert all(item["silhouette"] != -1.0 for item in report["candidates"]
               if item["k"] == report["auto_k"])


def test_v12_pair_rate_keeps_a_common_denominator():
    vectors = np.eye(4, dtype=np.float32)
    full_same, full_different = pair_scores(vectors, ["a", "a", "b", "b"])
    same, different = pair_scores(vectors, ["a", "a", "b", "b"], pair_rate=0.5)
    assert len(same) == 1 and len(different) == 2
    assert len(same) / len(full_same) == len(different) / len(full_different)


def test_v12_pair_rate_is_deterministic():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, 6)).astype(np.float32)
    labels = ["a" if index % 2 else "b" for index in range(40)]
    first = pair_scores(vectors, labels, pair_rate=0.3)
    second = pair_scores(vectors, labels, pair_rate=0.3)
    assert np.array_equal(np.sort(first[0]), np.sort(second[0]))
    assert np.array_equal(np.sort(first[1]), np.sort(second[1]))


def test_v12_threshold_metrics_scores_a_frozen_threshold():
    same, different = np.array([0.8, 0.9]), np.array([0.1, 0.2])
    assert threshold_metrics(same, different, 0.5)["eer"] == 0.0
    frozen = threshold_metrics(same, different, 0.95)
    assert frozen["frr"] == 1.0 and frozen["far"] == 0.0
    assert threshold_metrics(np.array([]), np.array([]), 0.5) is None


def test_v12_locked_thresholds_come_from_the_dev_report():
    report = {"results": [
        {"model": "campplus", "window_seconds": 2.0,
         "metric": {"eer": 0.09, "threshold": 0.32}},
        {"model": "eres2netv2", "window_seconds": 2.0, "metric": None},
    ]}
    locked = thresholds_from_report(report)
    assert locked["campplus@2.0"]["threshold"] == 0.32
    assert "eres2netv2@2.0" not in locked
