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


def test_accept_rewrite_rejects_digit_change():
    assert accept_rewrite("我们3点开会", "我们4点开会") is False


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


def test_guard_rejection_keeps_raw():
    results = []
    polisher = Polisher(results.append, call=lambda prompt: '{"polished":"完全不同的另一句话很长很长很长很长很长很长很长很长很长"}')
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["corrected"] is False
    assert out["utt_0001"]["error"] == "守卫驳回改写"


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
