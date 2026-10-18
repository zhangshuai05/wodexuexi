# -*- coding: utf-8 -*-
"""解锁的选 K 规则（唯一实现，供探针与端到端测试共用）。

规则只有两条，都是"把碎簇排除出决策"而不是"放宽准入"：

  1. 孤窗（<2 窗的簇）**不参与合格判定、也不参与轮廓系数评分**。
  2. 除孤窗外的簇，不设额外尺寸门（`min_win` 默认 2 = 仅排除孤窗）。

为什么不是"放宽尺寸门"：候选表显示坏场次在 k>=3 之后被一个 **2 窗**小簇挡住
（`[355,218,2]`、`[373,124,63,2]`、`[293,280,2,1]`、`[229,219,62,14,5]`）。
`v11._silhouette` 只要任一簇 <2 成员就返回 −1.0，`v11._candidate_clusters` 的
`min(sizes) >= 4` 门又先一步拦掉整张表 —— 两道都源自同一个 2 窗小簇。

**注意（未解决）**：`min_win=2` 是本会话在**全部 20 场**上选出来的，
dev 与 holdout 因此都不再是留出集。引用该数字时必须标注为 in-sample tuned。

    from unlocked_selector import build_unlocked, pick_unlocked, unlocked
"""
import contextlib
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` 与本地目录。**不要插仓库根**——那里有一份同名
# `claude_a2.py`（DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`，
# 连带毁掉 test_service 的导入。回归测试见 tests/test_evaluator_path_hygiene.py
for _p in (str(_ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402

from probe_selector_locks import silhouette_core  # noqa: E402

CAP = 10          # K 搜索上限；v11 常量是 4，而 dev 最大有效人数是 5
MIN_WIN = 2       # 除孤窗外的簇的最小规模（2 = 仅排除孤窗）


def build_unlocked(E, cap=CAP, min_win=MIN_WIN):
    """构造候选表：孤窗排除在判定与评分之外。返回与 v11 同构的列表。"""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n < 2:
        return []
    tree = linkage(E, method="average", metric="cosine")
    candidates = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
                   "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == label).sum()) for label in range(k)]
        core = [size for size in sizes if size >= 2]
        eligible = len(core) >= 2 and min(core) >= min_win
        score = silhouette_core(E, labels) if eligible else -1.0
        candidates.append({"k": k, "labels": labels, "score": score,
                           "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                           "sizes": sizes, "eligible": eligible})
    return candidates


def pick_unlocked(candidates, mode="objective"):
    """按 `mode` 选一个候选。

    `objective`（默认）—— 取目标函数最大的**合格**候选；无合格候选时退回 k=1。
    `largest_eligible`  —— 取 k 最大的合格候选。
    `largest`            —— 取 k 最大的候选（不看合格判定）。

    VAD 门下的扫描（`probe_rules_vad_gate.py`）显示 `largest_eligible` 优于
    `objective`（10.6% vs 12.9%），且 cap=8 时合格判定从不收紧，
    两种 largest 完全等价。默认仍保留 objective，改动需显式指定。
    """
    if not candidates:
        return None
    if mode == "largest":
        return max(candidates, key=lambda c: c["k"])
    eligible = [c for c in candidates if c["eligible"] and c["k"] >= 2]
    if mode == "largest_eligible":
        return max(eligible, key=lambda c: c["k"]) if eligible else candidates[0]
    if not eligible:
        return candidates[0]
    return max(eligible, key=lambda c: (c["objective"], -c["k"]))


@contextlib.contextmanager
def unlocked(min_win=MIN_WIN, cap=CAP, mode="objective"):
    """调用期间替换 v11 的候选构造与选择函数，退出时还原。"""
    original_candidates = v11._candidate_clusters
    original_select = v11._select_candidate

    def candidates(E, max_speakers=None):
        return build_unlocked(E, cap, min_win), None

    def select(cands, forced_k=None):
        if forced_k is not None:
            forced = next((c for c in cands if c["k"] == forced_k), None)
            if forced is not None:
                return forced
        return pick_unlocked(cands, mode=mode)

    v11._candidate_clusters = candidates
    v11._select_candidate = select
    try:
        yield
    finally:
        v11._candidate_clusters = original_candidates
        v11._select_candidate = original_select
