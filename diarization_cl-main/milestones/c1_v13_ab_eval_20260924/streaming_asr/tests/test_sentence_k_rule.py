# -*- coding: utf-8 -*-
"""句子级归属率的**选 k** 单元测试（纯函数，不 import 任何推理模块）。

背景：`probe_sentence_vs_realpath.py` 实测代理 argmax 系统性偏高
（9/12 场偏高，代理均值 5.7 vs 真实路径 3.2）。机制是代理（2.0s 粗窗、
无两级细化）对碎簇零成本——而 §4.1f 已实测细化在高 k 上有害。

修法复用仓库里已有的东西，不新造抽象：
  * `v11.MULTI_K_PENALTY`（= 0.015）已经是仓库的 k 惩罚常数
  * 候选表里已有 `sizes`，碎簇数量可直接看出
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from sentence_k_rule import pick_best_k  # noqa: E402


def test_tiny_clusters_lose_at_equal_accuracy():
    """同样 90% 归属率：k=6（簇健康）应赢过 k=9（带 3 个碎簇）。"""
    acc = {6: 0.90, 9: 0.90}
    sizes = {6: [200, 150, 120, 100, 80, 60],
             9: [200, 150, 120, 100, 80, 60, 3, 2, 2]}
    assert pick_best_k(acc, sizes) == 6


def test_real_advantage_still_wins():
    """k=6 归属率明显更高时，即使它带碎簇，仍应选 k=6。

    这条防"惩罚过头把改善场次也压回低 k"——那会让 3/3 命中变 0/3。
    """
    acc = {3: 0.35, 6: 0.97}
    sizes = {3: [300, 200, 100],
             6: [200, 150, 120, 60, 5, 2]}
    assert pick_best_k(acc, sizes) == 6


def test_smaller_k_wins_when_accuracy_ties_and_clusters_equal():
    """簇都健康且归属率相同 → 取小 k（平局时更简单者胜，不靠运气）。"""
    acc = {2: 0.80, 4: 0.80}
    sizes = {2: [300, 250], 4: [200, 150, 120, 80]}
    assert pick_best_k(acc, sizes) == 2


def test_single_candidate_returns_it():
    """只有一个候选时直接返回，不崩。"""
    assert pick_best_k({5: 0.7}, {5: [100, 90, 80, 70, 60]}) == 5


def test_empty_returns_none():
    """没有候选 → None（调用方负责处理，不抛异常）。"""
    assert pick_best_k({}, {}) is None


def test_absent_sizes_treated_as_healthy():
    """`sizes` 缺某个 k 时不得崩，按"无碎簇"处理。"""
    acc = {2: 0.90, 5: 0.90}
    assert pick_best_k(acc, {2: [300, 250]}) == 2
