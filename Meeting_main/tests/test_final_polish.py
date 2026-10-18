# -*- coding: utf-8 -*-
import json

import bootstrap

bootstrap.install()

from final_polish import accept_final, polish_turns  # noqa: E402


def _turn(turn_id, text, start_ms=0, end_ms=1000):
    return {"turn_id": turn_id, "text": text, "start_ms": start_ms, "end_ms": end_ms}


def test_accept_final_allows_punctuation_and_obvious_fix():
    assert accept_final("今天我们把思露整理一下", "今天我们把思路整理一下。") is True


def test_accept_final_rejects_growth_and_shrink():
    assert accept_final("你好", "你好你好你好") is False
    assert accept_final("我们今天把思路整理一下", "今天") is False


def test_accept_final_allows_digit_change():
    assert accept_final("我们三点开会", "我们四点开会") is True


def test_accept_final_relaxed_length_bounds():
    raw = "我们今天开会讨论"  # 8 字
    assert accept_final(raw, "我们今天开会讨论一下") is True   # 10/8 = 1.25
    assert accept_final(raw, "今天开会讨论") is True          # 6/8 = 0.75
    assert accept_final(raw, "我们今天开会讨论讨论讨论讨论") is False  # 16/8 = 2.0


def test_accept_final_strict_preserves_length():
    assert accept_final("我们三点开会", "我们四点开会", strict=True) is True
    assert accept_final("我们今天开会", "今天开会", strict=True) is False


def test_accept_final_rejects_empty():
    assert accept_final("你好", "   ") is False


def test_polish_turns_keeps_only_accepted_texts():
    def fake_call(prompt):
        raw = prompt.rsplit("待校对整段：", 1)[1]
        if "第二段" in raw:
            return json.dumps({"polished": "完全重写到底完全不同的一句话很长很长很长"},
                              ensure_ascii=False)
        if "第三段" in raw:
            raise RuntimeError("timeout")
        # 6 字 raw 最多长到 8 字（MAX_GROWTH=1.35），7 字加标点会被接受
        return json.dumps({"polished": "第一段改好了。"}, ensure_ascii=False)

    out = polish_turns([
        _turn("t1", "第一段改好了"),
        _turn("t2", "第二段原样"),
        _turn("t3", "第三段原样"),
    ], call=fake_call)

    assert out == {"t1": {"text": "第一段改好了。", "start_ms": 0, "end_ms": 1000}}


def test_polish_turns_uses_previous_turn_as_context():
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"polished": "改好了。"}'

    polish_turns([_turn("t1", "第一段内容"), _turn("t2", "第二段内容")], call=fake_call)

    assert "（无）" in prompts[0]
    assert "第一段内容" in prompts[1]


def test_polish_turns_stops_on_abort():
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return '{"polished": "改好了。"}'

    out = polish_turns([_turn("t1", "第一段内容"), _turn("t2", "第二段内容")],
                       call=fake_call, should_abort=lambda: True)

    assert calls == []
    assert out == {}


def test_polish_turns_reports_progress():
    seen = []
    polish_turns([_turn("t1", "改好了")],
                 call=lambda prompt: '{"polished": "改好了。"}',
                 on_progress=seen.append)

    assert seen == [{
        "t": "progress", "stage": "final_polish",
        "done": 1, "total": 1, "msg": "会后精修中…",
    }]
