# -*- coding: utf-8 -*-
"""v8 外生证据层 + ASR 契约的单元测试（不加载模型）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 同一份测试既跑在仓库 tests/ 下，也跑在 c1_v8 归档根目录下，两种情况都要
# 能找到 claude_a2_v8 / claude_a2_asr_contract。
for _p in (os.path.dirname(HERE), HERE):
    if os.path.exists(os.path.join(_p, "claude_a2_v8.py")):
        sys.path.insert(0, _p)
        break

# claude_a2_v8.py / claude_a2_asr_contract.py 按 .gitignore:23 的约定
# （"本线只上传 DSH 算法，Claude 侧由其自行管理"）不进本仓库。
# 在只有本仓库的检出里显式 skip，而不是让 pytest 报一个费解的 ImportError；
# 完整算法与测试见归档 milestones/c1_v8_20260914/。
if not os.path.exists(os.path.join(os.path.dirname(HERE), "claude_a2_v8.py")) \
        and not os.path.exists(os.path.join(HERE, "claude_a2_v8.py")):
    import pytest
    pytest.skip("claude_a2_v8.py 未随仓库分发（见里程碑 c1_v8_20260914）",
                allow_module_level=True)

import claude_a2_asr_contract as C  # noqa: E402
import claude_a2_v8 as V8  # noqa: E402


# ---------- pause_len / boundary_support ----------

def test_pause_len_counts_uncovered():
    vsegs = [(0.0, 1.0), (2.0, 3.0)]
    assert abs(V8.pause_len(1.0, 2.0, vsegs) - 1.0) < 1e-9
    assert abs(V8.pause_len(0.0, 3.0, vsegs) - 1.0) < 1e-9
    assert abs(V8.pause_len(0.2, 0.8, vsegs) - 0.0) < 1e-9


def test_boundary_support_marks_a_real_pause_as_direct():
    segs = [(0.0, 1.0, 0), (1.5, 2.5, 1)]
    vsegs = [(0.0, 1.0), (1.5, 2.5)]
    sup = V8.boundary_support(segs, vsegs, min_pause=0.15, min_run=0.75)
    assert len(sup) == 1
    assert sup[0]["direct"] is True
    assert abs(sup[0]["pause"] - 0.5) < 1e-9


def test_boundary_support_rejects_a_cut_inside_speech():
    """切在连续有声区里（模拟把一个人拆成两个）→ 没有静音支撑。"""
    segs = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
    vsegs = [(0.0, 2.0)]
    sup = V8.boundary_support(segs, vsegs, min_pause=0.15, min_run=0.75)
    assert sup[0]["direct"] is False
    assert abs(sup[0]["pause"]) < 1e-9


def test_boundary_support_rejects_sliver_even_with_pause():
    """有静音但两侧段太短 → 不算强证据（v7 的碎段正是这种）。"""
    segs = [(0.0, 0.2, 0), (0.4, 0.6, 1)]
    vsegs = [(0.0, 0.2), (0.4, 0.6)]
    sup = V8.boundary_support(segs, vsegs, min_pause=0.15, min_run=0.75)
    assert sup[0]["direct"] is False
    assert sup[0]["pause"] >= 0.15  # 静音确实在


# ---------- run_stats：这是替代 v7 n_sw_win 基线的量 ----------

def test_run_stats_penalises_unsupported_switches_not_switch_count():
    """关键性质：欠切方案不会因切换少而占优，过切方案会被无支撑切换抓出来。"""
    vsegs = [(0.0, 1.0), (1.5, 2.5), (3.0, 4.0)]
    under = [(0.0, 4.0, 0)]                                   # 全揉成一段
    good = [(0.0, 1.0, 0), (1.5, 2.5, 1), (3.0, 4.0, 0)]      # 切在静音上
    over = [(0.0, 0.5, 0), (0.5, 0.9, 1), (0.9, 1.0, 0),
            (1.5, 2.5, 1), (3.0, 3.4, 0), (3.4, 4.0, 1)]      # 切在有声区里
    su, sg, so = (V8.run_stats(s, vsegs) for s in (under, good, over))
    assert su["n_sw"] == 0
    assert sg["n_sw"] == 2 and sg["n_unsupported"] == 0
    assert so["n_sw"] == 5 and so["n_unsupported"] >= 3
    # v7 的判据（切换数）会认为 under 最好；外生判据认为 good 最好。
    assert sg["unsupported_ratio"] < so["unsupported_ratio"]


def test_run_stats_counts_short_runs():
    segs = [(0.0, 1.0, 0), (1.5, 1.7, 1), (2.0, 3.0, 0)]
    vsegs = [(0.0, 1.0), (1.5, 1.7), (2.0, 3.0)]
    st = V8.run_stats(segs, vsegs, short_dur=0.4)
    assert st["n_short"] == 1
    assert st["n_seg"] == 3


# ---------- fit_multi_k / count_gate ----------

def _two_cluster_reps(sep=0.9, n=4):
    a = np.zeros(8)
    a[0] = 1.0
    b = np.zeros(8)
    b[0] = np.sqrt(max(0.0, 1 - sep * sep))
    b[1] = sep
    rng = np.random.default_rng(0)
    return np.stack([a + rng.normal(0, 0.02, 8) for _ in range(n)]
                    + [b + rng.normal(0, 0.02, 8) for _ in range(n)])


def _reps_unimodal(n=8):
    rng = np.random.default_rng(1)
    r = np.zeros(8)
    r[0] = 1.0
    return np.stack([r + rng.normal(0, 0.02, 8) for _ in range(n)])


def test_fit_multi_k_finds_the_two_cluster_gap():
    tree = V8.fit_multi_k(_two_cluster_reps(), kmax=4)
    g2 = next(t for t in tree if t["k"] == 2)
    assert g2["gap"] > 0


def test_count_gate_single_when_embeddings_are_unimodal():
    segs = [(0.0, 2.0, 0), (2.2, 4.0, 0), (4.2, 6.0, 0)]
    vsegs = [(0.0, 6.0)]
    tree = V8.fit_multi_k(_reps_unimodal(), kmax=4)
    n, why = V8.count_gate(3, tree, vsegs, segs, gap_single=0.02)
    assert n == 1, why


def test_count_gate_single_when_one_representative_segment():
    n, why = V8.count_gate(1, [], [(0.0, 5.0)], [(0.0, 5.0, 0)])
    assert n == 1
    assert "单一代表段" in why[0]


def test_fit_multi_k_reports_cluster_sizes_without_empty_bins():
    """回归：bincount 会补齐到 max(label)+1，标签不从 0 起时会产生空桶。
    真 3 人曾被读成 [3,3,3,0] 而误判"孤段"（2026-09-14 实测发现）。"""
    reps = _two_cluster_reps()
    tree = V8.fit_multi_k(reps, kmax=4)
    for t in tree:
        assert 0 not in t["sizes"], t
        assert sum(t["sizes"]) == len(reps), t


def test_count_gate_uncertain_when_three_clusters_look_better():
    """k=3 间隙大**且**三簇都像样 → 不得声称 2 人（无真实 3 人素材）。"""
    a = np.zeros(8); a[0] = 1.0
    b = np.zeros(8); b[0] = 0.0; b[1] = 1.0
    c = np.zeros(8); c[0] = 0.4; c[1] = 0.3; c[2] = 0.85
    rng = np.random.default_rng(2)
    reps = np.stack([x + rng.normal(0, 0.01, 8)
                     for x in (a, b, c) for _ in range(3)])
    tree = V8.fit_multi_k(reps, kmax=4)
    segs = [(i * 1.0, i * 1.0 + 0.9, 0) for i in range(len(reps))]
    vsegs = [(i * 1.0, i * 1.0 + 0.9) for i in range(len(reps))]
    n, why = V8.count_gate(len(reps), tree, vsegs, segs, gap_single=0.001)
    assert n == "uncertain", why


def test_count_gate_two_when_third_cluster_is_a_lone_segment():
    """回归（2026-09-14 真实数据教训）：k3/k2 大只说明第三簇离得远。

    首版按倍数判人数，把 6 个真值 2 人文件全误判 uncertain，其中包括
    全集合 DER 最好的文件。构造"两个紧密簇 + 一个游离孤段"（这正是真实
    2 人数据硬切 k=3 的样子：5/4/1），必须判 2 人且理由说明是孤段。
    """
    a = np.zeros(8); a[0] = 1.0
    b = np.zeros(8); b[1] = 1.0
    # 游离段：与 a、b 都远，单独成一簇 → k=3 的第三个簇只有 1 段
    stray = np.zeros(8); stray[2] = 0.5; stray[3] = 0.5; stray[4] = 0.5
    rng = np.random.default_rng(3)
    reps = np.stack([x + rng.normal(0, 0.005, 8)
                     for x in (a, b) for _ in range(5)] + [stray])
    tree = V8.fit_multi_k(reps, kmax=4)
    g3 = {t["k"]: t for t in tree}.get(3)
    if g3 is None or g3["gap"] < 0.10:
        # 构造没造出"大 k=3 间隙"就没测到该路径，直接跳过而不是假通过
        import pytest
        pytest.skip(f"构造未产生大 k=3 间隙: {g3}")
    segs = [(i * 2.0, i * 2.0 + 1.5, 0) for i in range(len(reps))]
    vsegs = [(i * 2.0, i * 2.0 + 1.5) for i in range(len(reps))]
    n, why = V8.count_gate(len(reps), tree, vsegs, segs, gap_single=0.001)
    assert n == 2, why
    assert any("孤段" in w for w in why), why


def test_count_gate_two_when_two_clean_clusters():
    reps = _two_cluster_reps(sep=0.95)
    tree = V8.fit_multi_k(reps, kmax=4)
    segs = []
    t = 0.0
    for i in range(len(reps)):
        segs.append((t, t + 1.0, 0))
        t += 1.5
    vsegs = [(s, e) for s, e, _ in segs]
    n, why = V8.count_gate(len(reps), tree, vsegs, segs, gap_single=0.001)
    assert n == 2, why


# ---------- recon_error ----------

def test_recon_error_low_between_neighbours_same_speaker():
    r = np.zeros(8); r[0] = 1.0
    reps = np.stack([r] * 3)
    errs = V8.recon_error(None, reps, [0.0, 1.0, 2.0], 1.2)
    assert errs[1] is not None and errs[1] < 0.05


def test_recon_error_high_when_middle_is_a_different_speaker():
    """a-b-a：中间夹第三人时中点无法预测它 → 残差抬升。"""
    a = np.zeros(8); a[0] = 1.0
    b = np.zeros(8); b[0] = 0.0; b[1] = 1.0
    reps = np.stack([a, b, a])
    errs = V8.recon_error(None, reps, [0.0, 1.0, 2.0], 1.2)
    same = V8.recon_error(None, np.stack([a, a, a]), [0.0, 1.0, 2.0], 1.2)
    assert errs[1] > same[1] + 0.5
