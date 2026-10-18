# -*- coding: utf-8 -*-
"""web_server /api/compare 的参数校验与泳道标签（Stage A′ A6）。

固定两件事：
  1. 开关矩阵：snap=v0 + reassign=on 必须被拒（v0 快照无 reassign 形参、无 reassign_core.py）；
     非法取值被拒；缺省 = denoise auto / reassign off / snap v0。
  2. 标签写事实：当前泳道带 reassign 状态；snap=v1 的标签明说「与当前代码逐字节相同」，
     不得出现「v1 vs v2」这类把同码副本说成对照的说法。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import web_server as ws  # noqa: E402


def _q(**kw):
    return {k: [v] for k, v in kw.items()}


def test_defaults():
    p, err = ws.parse_compare_params(_q())
    assert err is None
    assert (p["denoise"], p["reassign"], p["snap"]) == ("auto", "off", "v0")


def test_v0_with_reassign_on_rejected():
    _, err = ws.parse_compare_params(_q(snap="v0", reassign="on"))
    assert err and "v0" in err


def test_v1_with_reassign_on_ok():
    p, err = ws.parse_compare_params(_q(snap="v1", reassign="on"))
    assert err is None and p["reassign"] == "on" and p["snap"] == "v1"


def test_bad_values_rejected():
    for kw in ({"snap": "v9"}, {"reassign": "yes"}, {"denoise": "maybe"}):
        _, err = ws.parse_compare_params(_q(**kw))
        assert err, kw


def test_lane_labels_are_factual():
    lanes = ws.lane_specs("v1", "v0", "off", {})
    lab = dict((k, l) for k, l, _ in lanes)
    assert "重分配off" in lab["cur"]
    assert lab["snap"] == "DSH v0（历史最优）"

    lanes = ws.lane_specs("v1", "v1", "on", {})
    lab = dict((k, l) for k, l, _ in lanes)
    assert "重分配on" in lab["cur"]
    assert "逐字节相同" in lab["snap"]
    assert "v2" not in lab["snap"]


def test_lane_specs_include_claude_lanes():
    """claude 泳道在前（前端只显示这两条），DSH 两版在后、仅供显式调用。"""
    lanes = ws.lane_specs("v1", "v0", "off", {"c1": {"label": "Claude c1"}})
    assert [k for k, _, _ in lanes][:1] == ["c1"]
    assert [k for k, _, _ in lanes][-2:] == ["cur", "snap"]
    assert lanes[0] == ("c1", "Claude c1", "claude")


def test_visible_versions_exposes_only_v7_and_v8():
    """前端体验版只下发 v7(c1) 与 v8；c2 仍可显式调用但不进界面。"""
    import claude_a2
    vis = claude_a2.visible_versions()
    assert list(vis) == ["c1", "v8"]
    assert vis["c1"]["method"] == "claude-a2-v7-c1"
    assert vis["v8"]["method"] == "claude-a2-v8-c3"
    assert "c2" not in vis
    assert "c2" in claude_a2.VERSIONS  # 显式调用仍可用
