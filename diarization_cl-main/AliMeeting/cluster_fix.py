# -*- coding: utf-8 -*-
"""离群簇准入的替代实现：**先合并小簇再判合格**，而不是整候选全否。

## 改的是什么

主仓 `claude_a2_v11._candidate_clusters`：

    eligible = (min(sizes) >= MIN_CLUSTER_WINDOWS       # = 4
                and min(sizes) / n >= MIN_CLUSTER_SHARE)  # = 0.015
    score = _silhouette(E, labels) if eligible else -1.0

而 `_select_candidate` 在没有任何合格候选时返回 `candidates[0]`，**即 k=1**。

AHC 在任何 k 超过真值时必然切出 1–5 窗的离群簇，于是好候选被一票否决。
实测（`results/oracle_k_alimeeting_full.json`）：**64/64 个被否的 k≥2 候选，
否决原因都是"候选里含 size ≤ 5 的离群簇"**，没有一个是轮廓系数低。

最刺眼的例子（`R8007_M8011`，真值 4 人）：

    k=4  sizes=[518, 921, 410, 1]   -> 否决    只因末尾那个 1 窗簇
    k=2  sizes=[518, 1332]          -> 合格，被选中

## 改法

不调门限数值（调低只会放进更多噪声候选），而是改**语义**：
把不合格的小簇先并入最近邻簇（按质心余弦），再判合格。

判定与合并都用与主仓相同的常量，保证不是"偷偷放宽"：

    MIN_CLUSTER_WINDOWS = 4        # 主仓 claude_a2_v11.py:20
    MIN_CLUSTER_SHARE   = 0.015    # 主仓 claude_a2_v11.py:21

合并后有效簇数才是真正的"人数"，所以在候选里同时记 `k`（有效）与 `k_cut`（原切割数）。
多个 `k_cut` 可能合并到同一个有效 k，按 labels 去重，保留轮廓系数最高者。

    python -c "import cluster_fix; cluster_fix.selfcheck()"
"""

import numpy as np

import common  # noqa: F401  —— 先把主仓路径插进 sys.path
import claude_a2_v11 as v11

MIN_CLUSTER_WINDOWS = v11.MIN_CLUSTER_WINDOWS   # 4
MIN_CLUSTER_SHARE = v11.MIN_CLUSTER_SHARE       # 0.015


def _centroid(E, labels, label):
    rows = E[labels == label]
    centroid = rows.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 0 else centroid


def prune_small_clusters(E, labels, verbose=False):
    """把不满足准入门的小簇并入最近邻簇，返回 (新 labels, 合并次数)。

    反复处理"当前最小的簇"，直到全部满足门或只剩一个簇。
    并入选的是质心余弦最大的另一个簇——这与 `_candidate_clusters` 用余弦距离
    做层次聚类的度量一致，不是另起一套。
    """
    labels = np.asarray(labels).copy()
    n = len(labels)
    merges = 0
    while True:
        unique = np.unique(labels)
        if len(unique) <= 1:
            break
        sizes = {int(label): int((labels == label).sum()) for label in unique}
        smallest = min(sizes, key=lambda label: sizes[label])
        if (sizes[smallest] >= MIN_CLUSTER_WINDOWS
                and sizes[smallest] / n >= MIN_CLUSTER_SHARE):
            break
        victim = _centroid(E, labels, smallest)
        best_label, best_similarity = None, -np.inf
        for label in unique:
            if label == smallest:
                continue
            similarity = float(victim @ _centroid(E, labels, label))
            if similarity > best_similarity:
                best_label, best_similarity = int(label), similarity
        labels[labels == smallest] = best_label
        merges += 1
        if verbose:
            print(f"    合并簇 {smallest}(size={sizes[smallest]}) -> {best_label} "
                  f"(相似度 {best_similarity:.3f})")
    return v11._relabel_by_first_seen(labels), merges


def candidate_clusters_pruned(E, max_speakers=v11.MAX_SPEAKERS):
    """与 `v11._candidate_clusters` 同签名同返回结构，只多了"先合并再判合格"。"""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n == 0:
        return [], np.empty((0,), dtype=int)
    if n == 1:
        return [{"k": 1, "k_cut": 1, "labels": np.zeros(1, dtype=int), "score": 0.0,
                 "objective": 0.0, "sizes": [1], "eligible": True, "merges": 0}], \
               np.zeros(1, dtype=int)

    tree = linkage(E, method="average", metric="cosine")
    candidates = [{"k": 1, "k_cut": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
                   "objective": 0.0, "sizes": [n], "eligible": True, "merges": 0}]
    seen = {tuple(np.zeros(n, dtype=int))}

    # 上界放宽到 max_speakers + 余量：合并会把 k 拉低，要切到更高才能得到
    # 有效 k = max_speakers 的候选。
    for k_cut in range(2, min(max_speakers, n) + 1 + MIN_CLUSTER_WINDOWS):
        raw = v11._relabel_by_first_seen(fcluster(tree, t=k_cut, criterion="maxclust"))
        pruned, merges = prune_small_clusters(E, raw)
        key = tuple(pruned)
        if key in seen:
            continue
        seen.add(key)
        effective = int(pruned.max()) + 1
        sizes = [int((pruned == label).sum()) for label in range(effective)]
        eligible = (min(sizes) >= MIN_CLUSTER_WINDOWS
                    and min(sizes) / n >= MIN_CLUSTER_SHARE)
        score = v11._silhouette(E, pruned) if eligible else -1.0
        candidates.append({
            "k": effective, "k_cut": k_cut, "labels": pruned, "score": score,
            "objective": score - v11.MULTI_K_PENALTY * (effective - 2),
            "sizes": sizes, "eligible": eligible, "merges": merges,
        })
    return candidates, tree


def install():
    """把主仓的 `_candidate_clusters` 换成这个版本（只在本进程内生效）。"""
    original = v11._candidate_clusters
    v11._candidate_clusters = candidate_clusters_pruned
    return original


def uninstall(original):
    v11._candidate_clusters = original


def selfcheck():
    """合成用例：4 个真簇 + 3 个 1 窗离群簇，看新旧两版各自选到几。"""
    rng = np.random.default_rng(7)
    centroids = rng.normal(size=(4, 32))
    blocks = [c + rng.normal(scale=0.25, size=(60, 32)) for c in centroids]
    outliers = rng.normal(scale=4.0, size=(3, 32))   # 远离所有簇
    E = np.vstack(blocks + [outliers])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)

    old, _ = v11._candidate_clusters(E, max_speakers=10)
    new, _ = candidate_clusters_pruned(E, max_speakers=10)
    old_pick = v11._select_candidate(old)
    new_pick = v11._select_candidate(new)
    print(f"合成用例：4 个真簇(60 窗) + 3 个离群窗 = 243 窗")
    print(f"  旧实现（整候选全否）选中 k={old_pick['k']}"
          f"  eligible={old_pick['eligible']}  sizes={old_pick['sizes']}")
    print(f"  新实现（先合并再判）选中 k={new_pick['k']}"
          f"  eligible={new_pick['eligible']}  sizes={new_pick['sizes']}"
          f"  merges={new_pick.get('merges')}")
    ok = new_pick["k"] == 4 and old_pick["k"] != 4
    print(f"  -> 合成用例{'通过' if ok else '未通过'}"
          f"（期望：旧版选不中 4，新版选中 4）")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if selfcheck() else 1)
