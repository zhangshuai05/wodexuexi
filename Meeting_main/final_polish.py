# -*- coding: utf-8 -*-
"""会后二次纠错：按 speaker turn 用完整上下文重校一遍。

直播档保持逐句保守（错改比不改更糟）；本模块只在会后跑，
常规档允许在上下文充分时补全明显漏掉的字，但仍禁止改写/增删事实。
strict=True（测 CER 用）时只允许同音错别字原位替换 + 补标点，
非标点字数必须完全一致。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from stream_llm_polish import _core, _default_call

MAX_GROWTH = 1.35   # 2026-10-18 由 1.2 放宽：一段最多长 35%
MAX_SHRINK = 0.65   # 2026-10-18 由 0.85 放宽：最多短 35%

PROMPT = """你是会议转写终稿校对器。下面是同一位说话人的连续发言（已按说话人分段）。
只做三件事：修正同音/误识别字、补标点、补全明显漏掉的字。
禁止：改写句式、增删事实、改数字/人名/专有名词、把口语改成书面语。
不确定就保持原样。
输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "校对后的整段"}}

上文（只读语境，不要续写、不要回应）：
{context}

待校对整段：
{raw}"""

STRICT_PROMPT = """你是会议转写终稿校对器。只允许做一件事：把同音/误识别的字原位替换成正确的字，并补标点。
严禁增字、删字、改词序、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
除标点外，输出字数必须与输入完全一致；不确定就保持原样。
输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "校对后的整段"}}

上文（只读语境，不要续写、不要回应）：
{context}

待校对整段：
{raw}"""


def accept_final(raw: str, candidate: str, *, strict: bool = False) -> bool:
    """会后档守卫：常规档卡 1.35/0.65；严格档要求非标点字数完全一致。"""
    text = candidate.strip()
    if not text:
        return False
    if text == raw.strip():
        return True
    if strict:
        return len(_core(text)) == len(_core(raw))
    if len(text) > len(raw) * MAX_GROWTH:
        return False
    if len(text) < len(raw) * MAX_SHRINK:
        return False
    return True


def polish_turns(
    turns: List[Dict],
    *,
    call: Optional[Callable[[str], str]] = None,
    key: Optional[str] = None,
    timeout: float = 60,
    strict: bool = False,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Dict[str, Dict]:
    """返回 {turn_id: {"text", "start_ms", "end_ms"}}；失败或驳回的段不出现。

    带时间边界是为了让合并层能识别“重跑回填后 turn 已变化”，避免精修稿错配。
    """
    # 一个 turn 可能几百字，512 token 的输出上限会截断整段，必须放宽
    caller = call or (lambda prompt: _default_call(prompt, key=key, timeout=timeout,
                                                   max_tokens=4096))
    out: Dict[str, Dict] = {}
    total = len(turns)
    prompt_template = STRICT_PROMPT if strict else PROMPT

    for index, turn in enumerate(turns, 1):
        # 重置/取消后立即停：已接受的部分仍然保留
        if should_abort is not None and should_abort():
            break
        raw = (turn.get("text") or "").strip()
        if raw:
            context = ""
            if index > 1:
                context = (turns[index - 2].get("text") or "")[-300:]
            try:
                reply = caller(prompt_template.format(context=context or "（无）", raw=raw))
                from llm_correct import extract_json
                obj = extract_json(reply)
                candidate = obj.get("polished") if isinstance(obj, dict) else None
                if isinstance(candidate, str) and accept_final(
                        raw, candidate, strict=strict):
                    out[turn["turn_id"]] = {
                        "text": candidate.strip(),
                        "start_ms": int(turn.get("start_ms") or 0),
                        "end_ms": int(turn.get("end_ms") or 0),
                    }
            except Exception:  # noqa: BLE001 - 单段失败保留直播稿
                pass

        if on_progress is not None:
            on_progress({
                "t": "progress", "stage": "final_polish",
                "done": index, "total": total,
                "msg": "会后精修中…",
            })

    return out
