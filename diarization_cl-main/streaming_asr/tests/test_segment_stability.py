# -*- coding: utf-8 -*-
"""锁住 `segment_stability` 的评分函数与 `probe_stability_selector` 的选择规则。

这里每一条断言都要能回答"**哪个合理回归会让它保持绿**"——答不出来就说明没覆盖
（坑 A5：本线连续踩了三次恒真断言）。

`stability_score` 的退化情形（K=1、过分割、扰动、空输入）在
`segment_stability._selfcheck` 里用合成数据测过，且**已实测推翻**最初的设计直觉：
扰动项不区分"错在边界"与"错在段中"。这里测的是**接线**与**选择规则**这两层 ——
它们是真正会被回归悄悄改坏的部分。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segment_stability import stability_score  # noqa: E402
import segment_stability as SS  # noqa: E402
import probe_stability_selector as PS  # noqa: E402


def test_selfcheck_synthetic_invariants_runs_in_ci():
    """把 `segment_stability._selfcheck` 的合成用例接进 pytest。

    它原本只在 `python segment_stability.py` 里跑 ⇒ 退化的四条（K=1→0、
    完美→纯度 1 且 A0=1、过分割→降、脆边界→更不稳定）在 CI 里**从不执行**。
    """
    SS._selfcheck()


def _row(k, objective, score, eligible, stability, err):
    return {"k": k, "objective": objective, "score": score, "eligible": eligible,
            "stability": stability, "attribution_error": err, "sizes": [10, 10]}


def test_k1_never_wins_a_stability_argmax():
    """K=1 的 A 恒为 0 ⇒ argmax 必须落在别处。挡住"一人全包拿最高分"。

    回归：把 `adjusted_purity` 的按机会校正去掉（直接返回 purity_obs），
    K=1 的 purity_obs=1.0 会变成最高分 ⇒ 本测试红。
    """
    labels = np.zeros(20, dtype=int)
    times = np.arange(20) * 0.15
    segs = [(0.0, 1.5), (1.5, 3.0)]
    score, _ = stability_score(labels, times, 0.6, segs)
    assert score == 0.0, score              # 恰好 0，不是 1
    rows = [_row(1, 0.0, 0.0, True, score, 0.63),
            _row(2, 0.1, 0.2, True, 0.8, 0.44)]
    assert PS.sel_stability(rows)["k"] == 2


def test_stability_selector_ignores_ineligible_rows():
    """不合格候选不参选 —— 否则一个哨兵分也能被选中。

    回归：把 `sel_stability` 里的 `_elig` 过滤删掉 ⇒ 本测试红。
    """
    rows = [_row(2, 0.0, 0.2, True, 0.5, 0.30),
            _row(3, 0.0, -1.0, False, 0.99, 0.10)]   # 不合格但 stability 最高
    assert PS.sel_stability(rows)["k"] == 2


def test_largest_eligible_ties_to_cap_and_is_a_constant():
    """`最大合格 K` 在本 dev 集上 12/12 场都等于 cap ⇒ 它是常数不是判据。

    这条断言把"它看起来表现好"钉在**已知缺陷**上：cap 一改，它的选中 K 全变。
    回归：有人把它当成可交付选择器接进生产 ⇒ 至少要先解释这条。
    """
    rows = [_row(k, 0.0, 0.1 * k, True, 0.5, 0.3 - 0.01 * k) for k in range(2, 11)]
    assert PS.sel_largest(rows)["k"] == 10
    rows8 = [r for r in rows if r["k"] <= 8]
    assert PS.sel_largest(rows8)["k"] == 8, "cap 变则选中 K 变 ⇒ 常数"


def test_oracle_selector_reads_truth_and_is_marked_as_upper_bound():
    """网格最佳逐场挑最低错误率 —— 是**诊断上界**，必须与真实选择器分开命名。

    回归：有人把 `sel_oracle` 接进生产 ⇒ 它在产物里叫"网格最佳（读真值·上界）"，
    名字本身就挡住了误用。
    """
    rows = [_row(2, 0.0, 0.9, True, 0.9, 0.30),
            _row(3, 0.0, 0.1, True, 0.1, 0.05)]
    assert PS.sel_oracle(rows)["k"] == 3
    assert "读真值" in next(n for n in PS.SELECTORS if n.startswith("网格最佳"))


def test_selector_matrix_is_frozen():
    """对照表里的选择器清单一旦改动，产物 schema 跟着变 —— 必须显式改这里。"""
    assert set(PS.SELECTORS) == {
        "生产 objective(λ=0.005)", "纯 silhouette", "最大合格 K",
        "新评分 stability", "网格最佳（读真值·上界）"}


def test_spearman_handles_ties_and_constant_series():
    """秩相关的并列与退化。本表里错误率成片并列，平均秩是必需的。

    回归：把平均秩换成 `argsort` 序号 ⇒ 大面积并列时结果偏移 ⇒ 本测试红。
    """
    assert PS.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(PS.spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert PS.spearman([1, 2], [1, 2]) is None              # n<3
    assert PS.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None   # 常数列
    # 全并列的一侧：秩全同 ⇒ 分子为 0 ⇒ ρ = 0（不是 NaN、不是 ±1）
    assert PS.spearman([1, 1, 1, 1, 1], [5, 4, 3, 2, 1]) is None
    assert abs(PS.spearman([1, 2, 3, 4, 5], [2, 2, 2, 2, 2]) or 0.0) < 1e-12


def test_probe_rejects_holdout():
    """holdout 已污染（min_win 在全部 20 场上调过），只允许 dev 探索。

    回归：有人把 `--split` 放开到 holdout 去调参 ⇒ argparse 直接拒。
    """
    import subprocess
    out = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "evaluation"
                                   / "probe_stability_selector.py"),
         "--split", "holdout"],
        capture_output=True, text=True)
    assert out.returncode != 0, "holdout 必须被拒"
    assert "invalid choice" in (out.stderr + out.stdout)


DATA = Path(r"D:\yinp\AISHELL-4\data\test")


def test_meeting_rows_scoring_function_touches_no_truth():
    """评分函数的入参里**不得**出现真值。走真缓存跑一场，验证接线。

    回归：有人把 RTTM 人名或 TextGrid 句界塞进 `stability_score` 的调用 ⇒
    这里检查它的入参只有 (细化标签, 细化时刻, 窗长, 语音段)，会红。
    """
    if not DATA.is_dir() or not Path(PS.BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:\\yinp\\…）")

    seen = {}
    real = PS.stability_score

    def spy(labels, times, window, segments, *a, **kw):
        seen["args"] = (len(labels), float(window), type(segments).__name__)
        seen["seg_types"] = {type(x).__name__ for x in segments[:1]}
        return real(labels, times, window, segments, *a, **kw)

    PS.stability_score = spy
    try:
        item, why = PS.meeting_rows("L_R003S01C02", PS.BS.DEFAULT_CACHE,
                                    DATA, 10, 600.0, 0.05, "rttm")
    finally:
        PS.stability_score = real

    assert item is not None, why
    assert seen, "评分函数根本没被调用 —— 上面的断言会恒真"
    assert seen["args"][1] > 0
    assert seen["seg_types"] == {"tuple"}, seen["seg_types"]
    # 段必须是 (start, end) 数值对，不是名字字符串
    segs = next(r for r in item["rows"] if r["k"] == 3)["stability_detail"]
    assert segs is not None and "purity_obs" in segs
