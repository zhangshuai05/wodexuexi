# -*- coding: utf-8 -*-
"""两个简单候选 K 判据（纯函数，无推理依赖）：AHC 距离阈值 / 归一化谱间隙。

## 本轮定位（见 artifacts/v13_ab_dev_spec_20260924.md）

- **A — AHC 距离阈值切树**：在聚类树高度上切一刀。阈值 `t` **只在 dev 上选**。
- **B — 归一化谱间隙**：由相似度图的谱结构产生 K 候选，**本轮只估计 K**。

两者都**不**替换 AHC 标签生成本身——标签仍由 `average`/`cosine` 的 AHC 给出，
A/B 只决定"切到几个簇"。这样与真实路径的细化策略共用同一套标签。

## 与 `sentence_k_rule` 的区别

`sentence_k_rule.pick_best_k` 是**拿归属率选 k**（需要真值，属评价侧）。
本文件的两个候选**只用嵌入与图的几何**，不读真值——这是"可部署"方向所要求的。

        from k_candidates import cut_by_distance, spectral_gap_k
"""
import numpy as np


# ───────────────────── A: AHC 距离阈值切树 ─────────────────────

def cut_by_distance(tree, threshold, n_clusters_cap=None):
    """按 AHC 树高度阈值切树，返回簇标签。

    `tree` 是 `scipy.cluster.hierarchy.linkage` 的输出（(n-1, 4)）。
    阈值就是**合并代价上限**：只保留高度 <= threshold 的合并。
    `scipy` 的 `fcluster(criterion="distance")` 正是这个语义，直接用它，
    不自己遍历树（rung 5：已装的依赖能做就别重写）。

    `n_clusters_cap` 若给定，簇数超过它时**截断到 cap**（用 maxclust 重切），
    以保证 A 与 B 在同一个 K 上限下比较。
    """
    from scipy.cluster.hierarchy import fcluster

    if len(tree) == 0:
        return np.zeros(1, dtype=int)
    labels = fcluster(tree, t=threshold, criterion="distance")
    labels = labels - labels.min()
    if n_clusters_cap is not None and labels.max() + 1 > n_clusters_cap:
        labels = fcluster(tree, t=n_clusters_cap, criterion="maxclust") - 1
    return labels


# ───────────────────── B: 归一化谱间隙 ─────────────────────

def affinity(emb):
    """余弦相似度图的**非负截断**构造：`W = max(cos, 0)`。

    ⚠️ **不要写成 `(1 + cos) / 2`**（本函数初稿的错）。那个变换把
    正交向量对（cos=0）映射成 **0.5** —— 团之间仍有强连接，
    **图不是块对角**，于是"k 个团 → k 个零特征值"这个谱间隙赖以成立的前提
    直接不成立。实测该构造下理想块数据的谱是
    `0, k/(k+1), …, k/(k+1), 1, …`，最大间隙**恒在第 0 位**，
    与真实 k 无关——不是谱间隙方法失效，是图造错了。

    非负截断保留"正交 = 无连接"，块对角结构成立（`test_affinity_maps_orthogonal_vectors_to_zero`）。
    它在真实嵌入上是**更保守**的选择（负余弦被当 0 而不是弱连接），
    本轮作为简单基线，**不宣称它在真实会议上更好**。

    **本函数固定为余弦距离**（初版留了个从未使用的 `metric` 形参，
    会让调用方以为能切换度量——已删除）。若将来要换度量，请连同
    `spectral_gap_k` 的调用方与测试一起改。
    """
    x = np.asarray(emb, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    cos = (x / norm) @ (x / norm).T
    return np.maximum(cos, 0.0)


def spectral_gap_k(emb, cap=12):
    """归一化拉普拉斯的最大特征值间隙 → K。

    步骤（规格冻结：图构造见 `affinity`）：
      1. `W = affinity(emb)`（余弦非负截断）
      2. `L_sym = I - D^{-1/2} W D^{-1/2}`
      3. 取最小的 `cap+1` 个特征值 `λ_0..λ_cap`（对称矩阵用 `eigvalsh`）
      4. **K = argmax_i (λ_{i+1} - λ_i) + 1**

    ⚠️ **`+1` 是标准零起始索引**：k 个连通分量对应前 k 个为 0 的特征值
    （λ_0 = 0 恒成立），跳变落在零起始的 `gaps[k-1]`，故 `K = argmax + 1`。
    本函数初稿写成 `+2`（对任何输入都返回 2 或 1），随后一度考虑
    "跳过首个间隙"来绕过——那是**错误的**，它会排除合法的 **K=1**。
    `test_spectral_gap_recovers_block_structure_including_k1` 与
    `test_all_identical_vectors_give_k1_not_k2` 锁住这两点。

    返回 (k, gaps)。`gaps` 供核查——最大间隙是否**显著**是判断依据，
    不能只看 argmax。本轮**不**引入相对间隙或随机图零模型（见修订规格）。
    """
    n = len(emb)
    if n < 3:
        return 1, np.zeros(0)
    cap = int(min(cap, n - 1))
    W = affinity(emb)
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    inv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(n) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
    # eigvalsh 保证返回值已按升序，无需再 sort（初版多了一次 O(n log n)）
    vals = np.linalg.eigvalsh(L)[:cap + 1]
    gaps = np.diff(vals)
    if len(gaps) == 0:
        return 1, gaps
    k = int(np.argmax(gaps)) + 1
    return max(1, min(k, cap)), gaps
