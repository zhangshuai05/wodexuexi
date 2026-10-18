# -*- coding: utf-8 -*-
"""两个 K 候选判据的单元测试（纯函数，无推理依赖）。

测试的边界：
  1. 阈值越大 → 合并越多 → 簇数越少（单调性，这是 A 的整个前提）
  2. 谱间隙在**构造出的明显分块图**上必须回到真值 K
  3. `W` 必须非负，否则归一化拉普拉斯会出现复数
  4. 退化输入（1 个点、全同点）不得崩、不得返回无意义的 K
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from k_candidates import affinity, cut_by_distance, spectral_gap_k  # noqa: E402


def _tree(emb):
    from scipy.cluster.hierarchy import linkage
    return linkage(emb, method="average", metric="cosine")


def _blobs(k, per=20, dim=8, spread=0.02, seed=0):
    """k 个明显分离的团，用来验证"该分几簇"。"""
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, dim) * 5.0
    return np.vstack([c + rng.randn(per, dim) * spread for c in centers])


# ───────────────────── A ─────────────────────

def test_threshold_monotone_in_cluster_count():
    """阈值越大，簇数**不增**（合并只会更激进）。"""
    emb = _blobs(4, seed=1)
    tree = _tree(emb)
    counts = [len(set(cut_by_distance(tree, t).tolist()))
              for t in (0.1, 0.3, 0.5, 0.7, 1.0)]
    assert counts == sorted(counts, reverse=True), f"簇数不单调：{counts}"


def test_cap_truncates_cluster_count():
    """给了 K 上限就必须受它约束。"""
    emb = _blobs(6, seed=2)
    tree = _tree(emb)
    labels = cut_by_distance(tree, 0.01, n_clusters_cap=3)   # 阈值极小 → 簇很多
    assert len(set(labels.tolist())) <= 3


def test_tiny_threshold_gives_many_clusters():
    """阈值趋近 0 → 几乎每个点自成一簇（A 的下界行为）。"""
    emb = _blobs(3, per=10, seed=3)
    tree = _tree(emb)
    assert len(set(cut_by_distance(tree, 0.0).tolist())) > 3


# ───────────────────── B ─────────────────────

def test_affinity_is_nonnegative_and_bounded():
    """`W` 必须在 [0,1]，否则归一化拉普拉斯会出现复数。"""
    emb = _blobs(3, seed=4)
    W = affinity(emb)
    assert W.min() >= 0.0, f"W 出现负值：{W.min()}"
    assert W.max() <= 1.0 + 1e-9
    assert np.allclose(W, W.T), "相似度图必须对称"


def test_affinity_maps_orthogonal_vectors_to_zero():
    """**正交向量的相似度必须是 0，不是 0.5。**

    回归：初版用 `W = (1 + cos)/2`，把正交（cos=0）映射成 **0.5**，
    于是"互相正交的 k 个团"之间仍有强连接，**图不是块对角**，
    谱上根本不会出现 k 个零特征值。实测该构造下
    λ₁..λ_{k-1} = k/(k+1)（0.667 / 0.75 / 0.8 …），最大间隙恒在第 0 位。
    这是**图构造的错**，不是谱间隙方法不适用。

    改成非负截断 `max(cos, 0)` 后，块对角结构成立，λ₀..λ_{k-1} ≈ 0。
    """
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    W = affinity(np.vstack([a, b]))
    assert abs(W[0, 1]) < 1e-12, f"正交对相似度应为 0，实得 {W[0,1]}"


def _orthogonal_groups(k, per=30, dim=64, jitter=0.0, seed=0):
    """k 个**互相正交**的向量团，余弦相似度结构可控（团内 1、团间 0）。

    为什么不用 `_blobs`：低维随机中心之间余弦相似度约 0.5（近正交是维度的性质），
    团间相似度并不接近 0，图上**没有分块结构**，谱间隙方法不适用。
    用正交单位基向量 + 可选抖动，才能造出"团间 = 0 / 团内 = 1"的块对角图。
    """
    rng = np.random.RandomState(seed)
    basis = np.eye(dim)[:k]
    rows = []
    for i in range(k):
        block = np.tile(basis[i], (per, 1))
        if jitter:
            block = block + rng.randn(per, dim) * jitter
        rows.append(block)
    return np.vstack(rows)


def test_spectral_gap_recovers_block_structure_including_k1():
    """块对角图 → K=团数。**包含 K=1**（不许跳过首间隙）。

    回归：初版把索引写成 `argmax + 2`，对任何输入都返回 2 或 1；
    随后若改成"跳过首个间隙"，虽能让 k≥2 通过，却会**排除合法的 K=1**——
    那是绕症状，不是修根因。
    """
    for k in (1, 2, 3, 4, 5):
        emb = _orthogonal_groups(k, per=30, dim=64, seed=10 + k)
        got, gaps = spectral_gap_k(emb, cap=10)
        assert got == k, (
            f"构造 {k} 个正交团，谱间隙给出 k={got}（gaps={np.round(gaps[:6], 4)}）"
        )


def test_spectral_gap_survives_mild_jitter():
    """加一点团内抖动仍应给出正确 K（不只在完美正交上碰巧对）。"""
    emb = _orthogonal_groups(3, per=30, dim=64, jitter=0.05, seed=99)
    got, _ = spectral_gap_k(emb, cap=10)
    assert got == 3, f"带抖动时谱间隙给出 k={got}"


def test_spectral_gap_respects_cap():
    """K 不得超过 cap。"""
    emb = _orthogonal_groups(5, per=25, dim=64, seed=7)
    k, _ = spectral_gap_k(emb, cap=3)
    assert k <= 3, f"k={k} 超过 cap=3"


def test_all_identical_vectors_give_k1_not_k2():
    """全同向量 → K=1。**这条查的是答案，不是"没越界"**。

    回归：初版在全同向量上返回 K=2（索引错），而旧测试只断言 `1 <= k <= 5`
    ——那种断言抓不到错答案。
    """
    k_same, _ = spectral_gap_k(np.ones((10, 4)), cap=5)
    assert k_same == 1, f"全同向量应给 K=1，实得 {k_same}"


def test_degenerate_inputs_do_not_crash():
    """1 个点、2 个点：不得崩。"""
    assert spectral_gap_k(np.zeros((1, 4)), cap=5)[0] >= 1
    assert spectral_gap_k(np.zeros((2, 4)), cap=5)[0] >= 1


def test_cut_by_distance_empty_tree():
    """空树（0 个合并）不得崩。"""
    assert cut_by_distance(np.zeros((0, 4)), 0.5).shape == (1,)
