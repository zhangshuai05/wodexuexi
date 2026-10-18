# -*- coding: utf-8 -*-
"""v8.1：把 v7 的两处"判据与结论脱节"钉死。

① c1/v7 的 verdict 原本是字面量 "reliable"，无论分离成什么样都报可靠。
   §37.3 实测：`n_clusters=2` + `verdict=reliable` 可以和"实际只输出了一个标签"
   同时成立，二者之间没有任何一致性检查。
② 带种子细化（cluster_level=dense_refine）此前无过切门：种子质心来自段级划分，
   而段级划分本身可能已经把两个人归成一簇，细化推出来仍是错的，且无处可比。

纯函数单测，不加载任何模型。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import claude_a2_v7_v8 as M  # noqa: E402


def _segs(*spec):
    """spec: (start, end, spk)"""
    return [{"start": float(a), "end": float(b), "spk": int(s)} for a, b, s in spec]


def _diag(**kw):
    M._A2D.clear()
    M._A2D.update(kw)


# ---------- ① verdict 不再是常量 ----------

def test_single_label_output_is_uncertain():
    """§37.3 的形态：算法按 2 人输出，实际只有 1 个标签撑起全部语音。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 0), (12.0, 13.0, 1))
    _diag()
    verdict, reasons = M._c1_verdict(["ok", "ok", "ok"], segs, 2)
    assert verdict == "uncertain"
    assert any("撑起有效语音" in r for r in reasons)


def test_pure_single_label_output_is_caught():
    """真·单标签输出：n_clusters 也退化成 1。判据①的分母必须是 assumed_k，
    否则最该报的形态反而溜过去（这是 v8.1 自己踩过的坑）。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 0), (12.0, 18.0, 0))
    _diag()
    verdict, reasons = M._c1_verdict(["ok"] * 3, segs, n_clusters=1, assumed_k=2)
    assert verdict == "uncertain", "单标签输出不得报 reliable"
    assert any("算法按 2 个说话人输出" in r for r in reasons)
    assert any("实际输出 1 种标签" in r for r in reasons)


def test_empty_output_is_uncertain():
    """空输出不能报 reliable（计划 §47）。"""
    _diag()
    verdict, reasons = M._c1_verdict([], [], 0)
    assert verdict == "uncertain"
    assert any("无语音段" in r for r in reasons)


def test_share_uses_vad_speech_not_segment_span():
    """占比按与 VAD 相交的有效语音算，不按段跨度。一个含 6s 静音的 7s 段
    不该被当成"撑起语音的第二个人"（计划 §44）。"""
    vsegs = [(0.0, 1.0), (12.0, 18.0)]
    segs = _segs((0.0, 7.0, 0), (7.0, 18.0, 1))
    _diag()
    # 段跨度口径：label0 = 7s / 18s = 39% → 会被当成够重 → reliable
    # VAD 口径：label0 = 1s / 7s = 14%……仍够重，故构造更极端的
    segs = _segs((0.0, 7.0, 0), (7.0, 19.0, 1))
    vsegs = [(0.0, 0.4), (13.0, 19.0)]
    verdict, reasons = M._c1_verdict(["ok", "ok"], segs, n_clusters=2,
                                     assumed_k=2, vsegs=vsegs)
    # VAD 口径 label0 = 0.4s/6.4s = 6.2% < 10% → 只算 1 个够重标签 → uncertain
    assert verdict == "uncertain", reasons
    assert any("仅 6.2% 有效语音" in r for r in reasons)


def test_no_vsegs_falls_back_to_span():
    _diag()
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    assert M._c1_verdict(["ok", "ok"], segs, 2)[0] == "reliable"


def test_low_share_is_a_warning_not_a_deletion_rule():
    """低占比只说明"支持不足"，不能证明这个人不存在（计划 §45）。
    判据必须只影响 verdict，不得改动 segs 本身。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 0), (12.0, 12.5, 1))
    before = [dict(s) for s in segs]
    _diag()
    M._c1_verdict(["ok"] * 3, segs, 2)
    assert segs == before, "判据不得删除或改写少数发言者的段"


def test_collapsed_wording_does_not_conclude_rescue_failed():
    """collapsed 为真只说明段级那条路塌了，成功救援时它同样为真（计划 §47）。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(collapsed=True)
    _, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    joined = "；".join(reasons)
    assert "救援未恢复第二人" not in joined
    assert "本旗标本身不下此结论" in joined


def test_gate_wording_does_not_claim_fallback_is_worse():
    """门拒绝 ≠ 回退更差（0657 上试出过拒绝一次有效细化，计划 §47/§49）。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(rescue_gate={"dense_switches": 13, "window_switches": 2, "passed": False})
    _, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    joined = "；".join(reasons)
    assert "它自己判定更差" not in joined
    assert "未经本判据比较" in joined


def test_two_label_output_stays_reliable():
    """健康双人文件不得因为这条降级（23 语料零回归的根据）。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1), (12.0, 18.0, 0))
    _diag(rescue_gate=None)
    verdict, reasons = M._c1_verdict(["ok", "ok", "ok"], segs, 2)
    assert verdict == "reliable", reasons


def test_isolated_long_fragment_does_not_count_as_a_label():
    """0657 实测形态：一个 1.0s 孤立条目（占 6.6% 语音、不落在任何真值轮上）
    曾被算成"第二个人"，于是 `n_clusters=2` 看起来自洽。它不该算。"""
    segs = _segs((0.0, 7.57, 1), (7.57, 14.0, 1), (14.0, 14.9, 1),
                 (31.27, 32.27, 0))
    _diag()
    verdict, reasons = M._c1_verdict(["ok"] * 4, segs, 2)
    assert verdict == "uncertain", reasons
    assert any("标签 0 仅 6" in r for r in reasons)


def test_collapse_flagged_makes_it_uncertain():
    """塌缩判据为真 = 算法自己认定两人被揉成一团，此时不得再报可靠。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(collapsed=True)
    verdict, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    assert verdict == "uncertain"
    assert any("段级路径判出塌缩" in r for r in reasons)


def test_gate_failed_makes_it_uncertain():
    """门拒绝救援：算法自己放弃了一条路，不得对外报可靠。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(rescue_gate={"dense_switches": 13, "window_switches": 2, "passed": False})
    verdict, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    assert verdict == "uncertain"
    assert any("超出过切预算被弃用" in r for r in reasons)


def test_gate_passed_does_not_downgrade():
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(rescue_gate={"dense_switches": 2, "window_switches": 2, "passed": True})
    verdict, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    assert verdict == "reliable", reasons


def test_new_refine_gate_keys_are_read():
    """塌缩分支的救援门写的是 dense_switches/window_switches，读的时候不能读成
    None —— 否则这条否决永远进不了结论（0657 那类文件就会继续报 reliable）。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(rescue_gate={"dense_switches": 13, "window_switches": 2, "passed": False})
    verdict, reasons = M._c1_verdict(["ok", "ok"], segs, 2)
    assert verdict == "uncertain"
    assert any("候选切换 13" in r and "基线 2" in r for r in reasons), reasons


def test_seeded_refine_has_no_switch_budget_gate():
    """带种子细化**不应**有基于切换数的门：细化的职责就是找换人点，它的切换数
    天然比段级多，用"超过基线 1.5 倍"判它过切，判的是任务目标本身。

    行为测试（计划 §49：用行为证明，不靠源码字符串断言）：给一个切换数远高于
    段级基线的候选，`_rescue_gate` 会判它超出预算（false）——这正是当初那版门
    会用来否决有效细化的判据。所以实现里不能拿它做判据，两个计数只记账。
    """
    cand = [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0), (3.0, 4.0, 1), (4.0, 5.0, 0)]
    seq = [0, 0, 1, 1, 1]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_sw_cand, n_sw_seg, ok = M._rescue_gate(cand, seq, times, 1.2)
    assert n_sw_cand == 4 and n_sw_seg == 1
    assert ok is False, "高切换数候选会被预算判负——所以它不能用来否决细化"
    # 两个计数确实进了 diag（备查），但 refine_gate 这个键不存在
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "claude_a2_v7_v8.py"), encoding="utf-8").read()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert '"refine_switch_counts"' in code
    assert '_A2D["refine_gate"]' not in code
    assert "refine_ok" not in code


def test_short_segments_alone_do_not_downgrade():
    """短段未解码是覆盖缺口，不是幻觉——否则任何短答文件都不可用。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(glob_ratio=0.1)
    verdict, _ = M._c1_verdict(["ok", "too_short", "too_short"], segs, 2)
    assert verdict == "reliable"


def test_suspect_text_downgrades():
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(glob_ratio=0.1)
    verdict, reasons = M._c1_verdict(["ok", "repeated"], segs, 2)
    assert verdict == "uncertain"
    assert any("文本可疑" in r for r in reasons)


def test_reasons_listed_even_when_reliable():
    """verdict_reasons 是给人看的仪表，可靠时也要带上下文。"""
    segs = _segs((0.0, 6.0, 0), (6.0, 12.0, 1))
    _diag(glob_ratio=0.1)
    verdict, reasons = M._c1_verdict(["ok", "too_short"], segs, 2)
    assert verdict == "reliable"
    assert any("未解码" in r for r in reasons)


# ---------- ② 带种子细化的过切门 ----------

def test_gate_rejects_refine_that_oversegments():
    """cand 在 5 个元素上切 4 次，段级基线只切 1 次 → 判过切。"""
    cand = [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0), (3.0, 4.0, 1), (4.0, 5.0, 0)]
    seq = [0, 0, 1, 1, 1]
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    n_sw_cand, n_sw_base, ok = M._rescue_gate(cand, seq, times, 1.2)
    assert n_sw_cand == 4
    assert n_sw_base == 1
    assert ok is False


def test_gate_allows_refine_close_to_baseline():
    cand = [(0.0, 1.0, 0), (1.0, 2.0, 0), (2.0, 3.0, 1)]
    seq = [0, 0, 1]
    times = [0.0, 1.0, 2.0]
    n_sw_cand, n_sw_base, ok = M._rescue_gate(cand, seq, times, 1.2)
    assert n_sw_cand == n_sw_base == 1
    assert ok is True


def test_gate_budget_floor_lets_a_2_switch_candidate_through():
    """基线切 0 次时上限仍为 2（max(1.5*1, 0+2)），不是 0——否则轻微细化全被拒。"""
    cand = [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0)]
    seq = [0, 0, 0]
    times = [0.0, 1.0, 2.0]
    n_sw_cand, n_sw_base, ok = M._rescue_gate(cand, seq, times, 1.2)
    assert (n_sw_cand, n_sw_base, ok) == (2, 0, True)
