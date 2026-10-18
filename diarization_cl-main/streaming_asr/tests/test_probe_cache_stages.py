# -*- coding: utf-8 -*-
"""`build_probe_cache` 的三个门（主窗/细化窗/救援窗）必须各不相同。

## 为什么需要这个测试

主窗与救援窗**共用** `v11._extract_windows`，只靠 `_v11_window_params` 换常量区分；
细化窗走 `v12._refine_windows`，有**自己的**门限 `REFINE_MIN_VAD`。
而细化窗与救援窗的**窗长、步长完全相同**（0.6 / 0.15），只有门限差 0.3 vs 0.5。

混用不会报错——只会让救援路径的窗集悄悄变成细化窗集，而下游照常出数。
本测试锁住三点：门限取值、阶段清单、以及"门限不同 ⇒ 窗集可不同"。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "streaming_asr"))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
import build_probe_cache as bpc  # noqa: E402


def test_stages_are_exactly_three_and_distinct():
    assert set(bpc.STAGES) == {"main", "refine", "rescue"}


def test_stage_params_match_production_constants():
    """阶段参数必须**引用**生产常量，不是抄下来的字面量。"""
    assert bpc.STAGES["main"] == {"window": v12.WIN_S, "hop": v12.HOP_S,
                                  "gate": v11.MIN_VAD_FRACTION}
    assert bpc.STAGES["refine"] == {"window": v12.REFINE_WIN_S,
                                    "hop": v12.REFINE_HOP_S,
                                    "gate": v12.REFINE_MIN_VAD}
    assert bpc.STAGES["rescue"] == {"window": v12.REFINE_WIN_S,
                                    "hop": v12.REFINE_HOP_S,
                                    "gate": v11.MIN_VAD_FRACTION}


def test_refine_and_rescue_differ_only_in_gate():
    """细化窗与救援窗同长同步，**只有门限不同**——这是最危险的一对。"""
    r, s = bpc.STAGES["refine"], bpc.STAGES["rescue"]
    assert (r["window"], r["hop"]) == (s["window"], s["hop"])
    assert r["gate"] != s["gate"], "两阶段门限相同则不该分开存，需复核生产常量"
    assert r["gate"] < s["gate"], "细化窗门限应更松（0.3 < 0.5）"


@pytest.mark.parametrize("gate,selected", [(0.3, True), (0.5, False)])
def test_gate_threshold_actually_changes_window_set(gate, selected):
    """门限确实会改变窗集：只覆盖 0.24s 的 0.6s 窗在 0.3 门下入选、0.5 门下落选。

    0.24 落在两门之间（0.3×0.6=0.18 ≤ 0.24 < 0.5×0.6=0.30），所以这一个窗
    正是"细化窗与救援窗会给出不同窗集"的最小实例。

    用**纯几何**验证（`v11._overlap`），不加载模型——门限语义是纯函数。
    """
    covered = v11._overlap(0.0, 0.6, [(0.0, 0.24)])
    assert covered == pytest.approx(0.24)
    assert (covered >= gate * 0.6) is selected


def test_extract_dispatches_by_window_constant(monkeypatch):
    """`extract('main')` 与 `extract('rescue')` 必须在**不同 WIN_S** 下调用 v11。"""
    seen = []

    def spy(audio, rate, vad_segments, spk_model):
        seen.append(round(v11.WIN_S, 4))
        return np.zeros((1, 4), dtype=np.float32), np.array([0.0]), 0

    monkeypatch.setattr(v11, "_extract_windows", spy)
    audio = np.zeros(16000, dtype=np.float32)
    bpc.extract("main", audio, 16000, [(0.0, 1.0)], {"spk": None})
    bpc.extract("rescue", audio, 16000, [(0.0, 1.0)], {"spk": None})
    assert seen == [round(v12.WIN_S, 4), round(v12.REFINE_WIN_S, 4)], (
        f"分派到 _extract_windows 的 WIN_S 序列是 {seen}，"
        "应分别为主窗与救援窗的窗长；相同则两条路径共用同一份窗集"
    )


def test_rescue_is_not_dispatched_to_refine(monkeypatch):
    """`extract('rescue')` 不得走 `v12._refine_windows`。

    细化窗的门限更松，若救援被派到细化上，救援窗集会**静默变大**——
    这正是本次会话反复踩的"派发错但无人报错"模式。
    """
    def boom(*args, **kwargs):
        raise AssertionError("rescue 不应调用 v12._refine_windows")

    monkeypatch.setattr(v12, "_refine_windows", boom)
    # 主窗与救援窗都不该碰它；只有 refine 该碰（此处置成 boom 后必须被调用来证伪）
    monkeypatch.setattr(v11, "_extract_windows",
                        lambda *a, **k: (np.zeros((1, 4), dtype=np.float32),
                                         np.array([0.0]), 0))
    audio = np.zeros(16000, dtype=np.float32)
    bpc.extract("main", audio, 16000, [(0.0, 1.0)], {"spk": None})
    bpc.extract("rescue", audio, 16000, [(0.0, 1.0)], {"spk": None})
    with pytest.raises(AssertionError):
        bpc.extract("refine", audio, 16000, [(0.0, 1.0)], {"spk": None})


def test_no_production_literals_duplicated(monkeypatch):
    """生产改了常量，STAGES 必须跟着变（本模块不存字面量副本）。

    ⚠️ 还原必须写成 `monkeypatch.undo()` **之后**再 reload。`monkeypatch` 的 teardown
    发生在测试函数**退出时**，所以函数体内的 `reload` 都仍在 patch 生效期间跑 ——
    初版末行只写了一句 `importlib.reload(bpc)` 当"还原"，实际是把 `STAGES` 用
    `REFINE_MIN_VAD=0.123` 又重建了一遍，跑完本测试后 gate 就留在 0.123
    （下一条测试是它的回归）。
    """
    monkeypatch.setattr(v12, "REFINE_MIN_VAD", 0.123)
    import importlib
    importlib.reload(bpc)
    assert bpc.STAGES["refine"]["gate"] == 0.123, "STAGES 未跟随生产常量（存了字面量副本？）"
    monkeypatch.undo()          # 先撤 patch，再 reload，才是真的还原
    importlib.reload(bpc)


def test_refine_gate_restored_after_monkeypatch():
    """紧邻上一条：跑完之后 `STAGES` 必须回到**生产值**，不得残留 0.123。

    上一条初版以 `importlib.reload(bpc)`（在 patch 生效时）收尾，把 0.123 留在了
    模块里；本测试是它的回归。依赖 pytest 按定义顺序收集。
    """
    assert v12.REFINE_MIN_VAD == 0.3, "REFINE_MIN_VAD 未还原"
    assert bpc.STAGES["refine"]["gate"] == v12.REFINE_MIN_VAD, (
        f"上一条测试把 STAGES 污染了：gate={bpc.STAGES['refine']['gate']} "
        f"!= 生产值 {v12.REFINE_MIN_VAD}")
