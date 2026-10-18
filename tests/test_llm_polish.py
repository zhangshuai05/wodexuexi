# -*- coding: utf-8 -*-
import json
import threading
import time

import bootstrap

bootstrap.install()

from stream_llm_polish import Polisher, accept_rewrite, is_short  # noqa: E402


def _event(uid="utt_0001", raw="大家早上好今天我们把思露整理一下"):
    return {"t": "asr_final", "utt_id": uid, "start_ms": 0, "end_ms": 2000, "raw_text": raw}


def test_is_short_ignores_punctuation():
    assert is_short("好的") is True
    assert is_short("嗯。") is True
    assert is_short("三点钟") is False


def test_accept_rewrite_allows_punctuation_and_homophone():
    raw = "大家早上好今天我们把思露整理一下"
    polished = "大家早上好，今天我们把思路整理一下。"
    assert accept_rewrite(raw, polished) is True


def test_accept_rewrite_rejects_growth():
    assert accept_rewrite("你好", "你好你好你好你好你好") is False


def test_accept_rewrite_rejects_shrink():
    assert accept_rewrite("我们今天把思路整理一下", "今天") is False


def test_accept_rewrite_allows_digit_change():
    # 2026-10-18 用户拍板移除数字集合守卫：同音误识/格式转换不能被误杀
    assert accept_rewrite("我们3点开会", "我们4点开会") is True
    assert accept_rewrite("我们三点开会", "我们四点开会") is True


def test_strict_rewrite_preserves_non_punctuation_length():
    assert accept_rewrite("我们3点开会", "我们4点开会", strict=True) is True
    assert accept_rewrite("你好", "你好啊", strict=True) is False
    assert accept_rewrite("你好", "你好。", strict=True) is True


def test_accept_rewrite_rejects_empty():
    assert accept_rewrite("你好", "   ") is False


def test_short_sentence_is_bypassed_without_calling_llm():
    calls = []
    results = []
    polisher = Polisher(results.append, call=lambda prompt: calls.append(prompt) or '{"polished":"x"}')

    polisher.submit(_event(raw="好的"))
    out = polisher.close(timeout=5)

    assert calls == []
    assert out["utt_0001"]["bypassed"] is True
    assert out["utt_0001"]["polished_text"] == "好的。"
    assert results[0]["t"] == "llm_confirmed"


def test_normal_sentence_calls_llm_and_reports_corrected():
    def fake_call(prompt):
        assert "上文语境" in prompt
        return json.dumps({"polished": "大家早上好，今天我们把思路整理一下。"}, ensure_ascii=False)

    results = []
    polisher = Polisher(results.append, call=fake_call)
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["corrected"] is True
    assert out["utt_0001"]["bypassed"] is False
    assert out["utt_0001"]["error"] is None
    assert results[0]["polished_text"] == "大家早上好，今天我们把思路整理一下。"


def test_llm_exception_fails_soft():
    def boom(prompt):
        raise RuntimeError("network down")

    results = []
    polisher = Polisher(results.append, call=boom)
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["corrected"] is False
    assert "network down" in out["utt_0001"]["error"]


def test_unparseable_output_fails_soft():
    results = []
    polisher = Polisher(results.append, call=lambda prompt: "这不是 JSON")
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["error"] == "LLM 输出无法解析为 JSON"


def test_null_polished_fails_soft():
    results = []
    polisher = Polisher(results.append, call=lambda prompt: '{"polished": null}')
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["error"] == "LLM 输出无法解析为 JSON"


def test_guard_rejection_keeps_raw():
    results = []
    polisher = Polisher(results.append, call=lambda prompt: '{"polished":"完全不同的另一句话很长很长很长很长很长很长很长很长很长"}')
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["corrected"] is False
    assert out["utt_0001"]["error"] == "守卫驳回改写"


def test_uses_turn_context_in_prompt():
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"polished": "改好了。"}'

    polisher = Polisher(lambda ev: None, call=fake_call)
    polisher.submit({
        "t": "asr_final", "utt_id": "u1", "start_ms": 0, "end_ms": 1000,
        "raw_text": "改好了", "turn_confirmed_text": "这是当前段的完整上文",
    })
    polisher.close(timeout=5)

    assert "这是当前段的完整上文" in prompts[0]


def test_target_sentence_is_removed_from_turn_context():
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"polished": "改好了。"}'

    polisher = Polisher(lambda ev: None, call=fake_call)
    polisher.submit({
        "t": "asr_final", "utt_id": "u1", "start_ms": 0, "end_ms": 3000,
        "raw_text": "改好了", "turn_confirmed_text": "上文内容，改好了",
    })
    polisher.close(timeout=5)

    assert "上文内容" in prompts[0]
    assert prompts[0].count("改好了") == 1, "当前句只能出现在待纠错区，不能再作为上文"


def test_failed_or_rejected_result_is_not_used_as_context():
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return '{"polished":"完全不同的另一句话很长很长很长很长很长很长很长很长很长"}'
        return '{"polished":"第二句改好了。"}'

    polisher = Polisher(lambda ev: None, call=fake_call)
    polisher.submit(_event(uid="u1", raw="第一句内容"))
    polisher.submit(_event(uid="u2", raw="第二句内容"))
    polisher.close(timeout=5)

    assert "完全不同的另一句话" not in prompts[1], "被驳回的候选不得进入上文"


def test_confident_context_is_preferred_over_turn_context():
    prompts = []

    def fake_call(prompt):
        prompts.append(prompt)
        return '{"polished":"第一句改好了。"}'

    polisher = Polisher(lambda ev: None, call=fake_call)
    polisher.submit(_event(uid="u1", raw="第一句内容"))
    polisher.submit({
        "t": "asr_final", "utt_id": "u2", "start_ms": 2000, "end_ms": 4000,
        "raw_text": "第二句内容", "turn_confirmed_text": "脏的原始上文，第二句内容",
    })
    polisher.close(timeout=5)

    assert "第一句改好了" in prompts[1]
    assert "脏的原始上文" not in prompts[1], "有高置信度上文时不得退回原始 turn 文本"


def test_submit_is_non_blocking_and_preserves_order():
    gate = threading.Event()

    def slow_call(prompt):
        gate.wait(2)
        return json.dumps({"polished": "改好了。"}, ensure_ascii=False)

    results = []
    polisher = Polisher(results.append, call=slow_call)

    started = time.monotonic()
    for i in range(3):
        polisher.submit(_event(uid=f"utt_{i:04d}"))
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, "submit 必须立即返回"

    gate.set()
    out = polisher.close(timeout=10)
    assert set(out) == {"utt_0000", "utt_0001", "utt_0002"}
    assert [ev["utt_id"] for ev in results] == ["utt_0000", "utt_0001", "utt_0002"]


def test_callback_exception_does_not_kill_worker():
    seen = []

    def boom_once(ev):
        seen.append(ev["utt_id"])
        if len(seen) == 1:
            raise RuntimeError("disk full")

    polisher = Polisher(boom_once, call=lambda prompt: '{"polished":"改好了。"}')
    # raw 必须是 3 个有效字："改好了。" 才能通过缩减守卫（4 >= 3 * 0.7）并被润色器接受
    polisher.submit(_event(uid="utt_0001", raw="改好了"))
    polisher.submit(_event(uid="utt_0002"))
    out = polisher.close(timeout=5)

    assert set(out) == {"utt_0001", "utt_0002"}
    assert seen == ["utt_0001", "utt_0002"], "第二个事件仍必须被回调"
    assert out["utt_0001"]["polished_text"] == "改好了。"
    assert "callback" in out["utt_0001"]["error"]


def test_malformed_raw_text_does_not_kill_worker():
    polisher = Polisher(lambda ev: None, call=lambda prompt: '{"polished":"好。"}')
    polisher.submit({"t": "asr_final", "utt_id": "u_bad", "start_ms": 0, "end_ms": 1, "raw_text": None})
    polisher.submit(_event(uid="utt_0002"))
    out = polisher.close(timeout=5)

    assert "u_bad" in out
    assert "utt_0002" in out
