# -*- coding: utf-8 -*-
"""句尾异步 DeepSeek 润色：单消费者 FIFO，保守纠错，失败降级。

契约见 STREAMING_PLAN.md §6.2：
    允许：改同音/误识别字、补标点
    禁止：改词序、增删内容、改专有名词、删口语重复、去语气词、改写成书面语
    守卫：长度增幅 > 50% / 缩减 > 30% => 丢弃改写，保留 raw_text
          2026-10-18 移除“数字集合必须一致”守卫：ASR 同音误识（三/四）与
          数字格式转换会被误杀；strict=True 时改为“非标点字数必须完全一致”。
"""
from __future__ import annotations

import json
import queue
import threading
import unicodedata
import urllib.request
from typing import Callable, Dict, List, Optional

from llm_correct import _get_key, extract_json

MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"

SHORT_CIRCUIT_CHARS = 2
MAX_GROWTH = 1.5
MAX_SHRINK = 0.7
CONTEXT_SENTENCES = 3
MAX_CONTEXT_CHARS = 800   # 直播上下文按字符预算截断，随 turn 增长而不是固定 3 句
PUNCTUATION = "，。！？、,.!? 　"

PROMPT_TMPL = """你是 ASR 转写纠错器。只做两件事：改同音/误识别的字，补标点。
严格禁止：改词序、增删内容、改专有名词、删口语重复、去语气词、改写成书面语。
不确定的地方保持原样——宁可少改，不可改错。

上文语境（仅用于判断同音字，不要续写、不要回应）：
{context}

待纠错句子：
{raw}

输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "纠错后的整句"}}"""

# 测 CER 时用严格档：只允许原位替换同音字 + 补标点，非标点字数必须完全一致。
STRICT_PROMPT_TMPL = """你是 ASR 转写纠错器。只允许做一件事：把同音/误识别的字原位替换成正确的字，并补标点。
严禁增字、删字、改词序、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
除标点外，输出字数必须与输入完全一致；不确定就保持原样。

上文语境（仅用于判断同音字，不要续写、不要回应）：
{context}

待纠错句子：
{raw}

输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "纠错后的整句"}}"""


def is_short(raw: str) -> bool:
    """去掉标点与空白后 <= 2 个字符，视为短句。"""
    core = "".join(ch for ch in raw if ch not in PUNCTUATION)
    return len(core) <= SHORT_CIRCUIT_CHARS


def _core(text: str) -> str:
    """去掉所有 Unicode 标点与空白后的正文，用于长度/严格档比较。"""
    return "".join(
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def accept_rewrite(raw: str, polished: str, *, strict: bool = False) -> bool:
    """守卫：严格档要求非标点字数完全一致；常规档只卡长度增幅/缩减。"""
    candidate = polished.strip()
    if not candidate:
        return False
    if candidate == raw.strip():
        return True
    if strict:
        return len(_core(candidate)) == len(_core(raw))
    if len(candidate) > len(raw) * MAX_GROWTH:
        return False
    if len(candidate) < len(raw) * MAX_SHRINK:
        return False
    return True


def _default_call(prompt: str, key: Optional[str] = None, timeout: float = 60,
                  max_tokens: int = 512) -> str:
    resolved = _get_key(key)
    if not resolved:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（进程环境或 Windows 用户级环境变量任一处配置即可）")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {resolved}",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


class Polisher:
    """单消费者 FIFO。submit 非阻塞；结果按提交顺序回调 on_result。"""

    def __init__(
        self,
        on_result: Callable[[Dict], None],
        *,
        call: Optional[Callable[[str], str]] = None,
        key: Optional[str] = None,
        timeout: float = 60,
        strict: bool = False,
    ) -> None:
        self._on_result = on_result
        self._timeout = timeout
        self._strict = strict
        if call is not None:
            self._call: Callable[[str], str] = call
        else:
            def _bound(prompt: str) -> str:
                return _default_call(prompt, key=key, timeout=timeout)
            self._call = _bound

        self._queue: "queue.Queue[Optional[Dict]]" = queue.Queue()
        self._results: Dict[str, Dict] = {}
        self._context: List[str] = []
        self._lock = threading.Lock()
        self._aborted = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="llm-polish", daemon=True)
        self._thread.start()

    # ---- 对外 ----

    def submit(self, ev: Dict) -> None:
        """非阻塞。ev 至少含 utt_id / raw_text / start_ms / end_ms。"""
        if self._aborted.is_set():
            return
        self._queue.put({
            "utt_id": ev["utt_id"],
            "raw_text": ev.get("raw_text") or "",
            "start_ms": ev.get("start_ms"),
            "end_ms": ev.get("end_ms"),
            # 当前 turn 已确认的全部文本；比固定 3 句窗口更能判断同音字
            "context": ev.get("turn_confirmed_text") or "",
        })

    def results(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._results)

    def close(self, timeout: Optional[float] = None) -> Dict[str, Dict]:
        self._queue.put(None)
        self._thread.join(timeout)
        return self.results()

    def abort(self) -> Dict[str, Dict]:
        """丢弃尚未开始的请求，不等待正在进行的单次 HTTP 调用。"""
        self._aborted.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()
        self._queue.put(None)
        return self.results()

    # ---- 内部 ----

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                result = self._polish_one(item["raw_text"], item.get("context"))
                if self._aborted.is_set():
                    return
                with self._lock:
                    self._results[item["utt_id"]] = result
                # 脏上文传染：只有“无错误、无守卫驳回”的结果才进入后续语境。
                # 失败/被驳回的候选往往是模型幻觉，再喂回去会污染下一句。
                if not result.get("error") and not result.get("rejected_text"):
                    self._context.append(result["polished_text"])
                    del self._context[:-CONTEXT_SENTENCES]
                try:
                    self._on_result({
                        "t": "llm_confirmed",
                        "utt_id": item["utt_id"],
                        "start_ms": item["start_ms"],
                        "end_ms": item["end_ms"],
                        "raw_text": item["raw_text"],
                        "polished_text": result["polished_text"],
                        "corrected": result["corrected"],
                        "bypassed": result["bypassed"],
                        "error": result["error"],
                        "rejected_text": result.get("rejected_text"),
                    })
                except Exception as exc:  # noqa: BLE001 - 回调异常（如落盘失败）不得杀死消费者
                    with self._lock:
                        self._results[item["utt_id"]] = {
                            **result,
                            "error": ((result.get("error") or "") + f" | callback: {type(exc).__name__}: {exc}").strip(" |"),
                        }
            except Exception as exc:  # noqa: BLE001 - 单条任务异常不得杀死消费者
                with self._lock:
                    self._results[item["utt_id"]] = {
                        "polished_text": item.get("raw_text") or "",
                        "corrected": False, "bypassed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            finally:
                self._queue.task_done()

    def _polish_one(self, raw: str, turn_context: Optional[str] = None) -> Dict:
        base = {
            "polished_text": raw,
            "corrected": False,
            "bypassed": False,
            "error": None,
        }
        if is_short(raw):
            text = raw.strip()
            if text and text[-1] not in "，。！？、,.!?":
                text += "。"
            return {**base, "polished_text": text, "bypassed": True}

        try:
            # 只取高置信度句子作上文：self._context 里只保存过“无错误、无驳回”的结果。
            # 没有高置信度上文时才退回当前 turn 的原始文本（首句/重启场景）。
            context_text = "\n".join(self._context[-CONTEXT_SENTENCES:])
            if not context_text:
                context_text = (turn_context or "").strip()
                # asr_final 的 turn 文本末尾就是当前句；留在“上文”里会让模型照抄原句
                if raw and context_text.endswith(raw):
                    context_text = context_text[:-len(raw)].rstrip("，。！？、；：,.!? ")
            context = (context_text or "（无）")[-MAX_CONTEXT_CHARS:]
            template = STRICT_PROMPT_TMPL if self._strict else PROMPT_TMPL
            reply = self._call(template.format(context=context, raw=raw))
            obj = extract_json(reply)
            polished = obj.get("polished") if isinstance(obj, dict) else None
            if not isinstance(polished, str):
                return {**base, "error": "LLM 输出无法解析为 JSON"}
            candidate = polished.strip()
            if not accept_rewrite(raw, candidate, strict=self._strict):
                return {**base, "error": "守卫驳回改写", "rejected_text": candidate}
            return {
                **base,
                "polished_text": candidate,
                "corrected": candidate != raw.strip(),
            }
        except Exception as exc:  # noqa: BLE001 - 失败必须降级而不是卡死队列
            return {**base, "error": f"{type(exc).__name__}: {exc}"}
