# -*- coding: utf-8 -*-
"""句尾异步 DeepSeek 润色：单消费者 FIFO，保守纠错，失败降级。

契约见 STREAMING_PLAN.md §6.2：
    允许：改同音/误识别字、补标点
    禁止：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语
    守卫：长度增幅 > 50% / 缩减 > 30% / 数字集合变化 => 丢弃改写，保留 raw_text
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.request
from typing import Callable, Dict, List, Optional

from llm_correct import _get_key, extract_json

MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"

SHORT_CIRCUIT_CHARS = 2
MAX_GROWTH = 1.5
MAX_SHRINK = 0.7
CONTEXT_SENTENCES = 3
PUNCTUATION = "，。！？、,.!? 　"
CJK_NUMERALS = "零〇一二三四五六七八九十百千万亿两"

PROMPT_TMPL = """你是 ASR 转写纠错器。只做两件事：改同音/误识别的字，补标点。
严格禁止：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
不确定的地方保持原样——宁可少改，不可改错。

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


def _digits(text: str) -> List[str]:
    return sorted(ch for ch in text if ch.isdigit() or ch in CJK_NUMERALS)


def accept_rewrite(raw: str, polished: str) -> bool:
    """三道守卫。任一道不过就丢弃改写。"""
    candidate = polished.strip()
    if not candidate:
        return False
    if candidate == raw.strip():
        return True
    if len(candidate) > len(raw) * MAX_GROWTH:
        return False
    if len(candidate) < len(raw) * MAX_SHRINK:
        return False
    if _digits(candidate) != _digits(raw):
        return False
    return True


def _default_call(prompt: str, key: Optional[str] = None, timeout: float = 60) -> str:
    resolved = _get_key(key)
    if not resolved:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（进程环境或 Windows 用户级环境变量任一处配置即可）")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
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
    ) -> None:
        self._on_result = on_result
        self._timeout = timeout
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
        self._thread = threading.Thread(target=self._worker, name="llm-polish", daemon=True)
        self._thread.start()

    # ---- 对外 ----

    def submit(self, ev: Dict) -> None:
        """非阻塞。ev 至少含 utt_id / raw_text / start_ms / end_ms。"""
        self._queue.put({
            "utt_id": ev["utt_id"],
            "raw_text": ev.get("raw_text") or "",
            "start_ms": ev.get("start_ms"),
            "end_ms": ev.get("end_ms"),
        })

    def results(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._results)

    def close(self, timeout: Optional[float] = None) -> Dict[str, Dict]:
        self._queue.put(None)
        self._thread.join(timeout)
        return self.results()

    # ---- 内部 ----

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                result = self._polish_one(item["raw_text"])
                with self._lock:
                    self._results[item["utt_id"]] = result
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

    def _polish_one(self, raw: str) -> Dict:
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
            context = "\n".join(self._context) if self._context else "（无）"
            reply = self._call(PROMPT_TMPL.format(context=context, raw=raw))
            obj = extract_json(reply)
            polished = obj.get("polished") if isinstance(obj, dict) else None
            if not isinstance(polished, str):
                return {**base, "error": "LLM 输出无法解析为 JSON"}
            candidate = polished.strip()
            if not accept_rewrite(raw, candidate):
                return {**base, "error": "守卫驳回改写"}
            return {
                **base,
                "polished_text": candidate,
                "corrected": candidate != raw.strip(),
            }
        except Exception as exc:  # noqa: BLE001 - 失败必须降级而不是卡死队列
            return {**base, "error": f"{type(exc).__name__}: {exc}"}
