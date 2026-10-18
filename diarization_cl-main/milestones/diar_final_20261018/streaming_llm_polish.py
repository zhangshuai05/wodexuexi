# -*- coding: utf-8 -*-
"""句尾异步润色：单线程 FIFO，把 asr_final 变成 llm_confirmed。

为什么必须 FIFO 而不能并发：第 k 句润色后的正体要作为第 k+1 句的只读语境，
并发会让上下文乱序。

复用 `llm_correct` 的 `_get_key` / `call_llm`（同一个 DEEPSEEK_API_KEY 解析路径：
进程环境变量 → Windows 用户级注册表兜底）。
"""
import os
import queue
import re
import sys
import threading
from typing import Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASR_DIR = os.path.join(os.path.dirname(_HERE), "streaming_asr")
if _ASR_DIR not in sys.path:
    sys.path.insert(0, _ASR_DIR)

from llm_correct import _get_key, call_llm  # noqa: E402

CONTEXT_N = 3        # 带上前 N 句已固化的正体作为只读语境
MIN_CHARS = 2        # 去标点后有效字数 <= 此值直接短路，不调 API
END_PUNCT = "。！？，、；："

PROMPT = """你是会议转写校对器。只做两件事：改同音/近音错字、补标点。
禁止扩写、改写、增删信息、改变原意。读不通也照原样保留，不要脑补。
前文（只读语境，不要输出）：
{context}
待校对（只输出校对后的这一句，不要解释，不要引号）：
{raw}"""


def _effective_len(text: str) -> int:
    """去掉标点与空白后的字数。"""
    return len(re.sub(r"[\W_]+", "", text, flags=re.UNICODE))


def polish_text(raw: str, context: List[str], timeout: float = 30.0,
                key: Optional[str] = None, context_n: int = CONTEXT_N) -> str:
    """单句润色。<=2 字短路，不发网络请求。"""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if _effective_len(raw) <= MIN_CHARS:
        return raw if raw[-1] in END_PUNCT else raw + "。"

    # context_n=0 不能写成 [-0:]（那是整个列表）
    ctx_list = context[-context_n:] if context_n > 0 else []
    ctx = "\n".join("- " + c for c in ctx_list) or "（无）"
    out = call_llm(PROMPT.format(context=ctx, raw=raw),
                   key=key, timeout=timeout).strip()
    out = out.strip('"“”\n ').strip()
    # 模型偶尔返回空 —— 宁可退回原文，不可丢句
    return out or raw


class StreamingPolisher:
    """FIFO 单线程润色器。submit() 立即返回，主音频线程零等待。"""

    def __init__(self, on_confirmed: Callable[[Dict], None],
                 context_n: int = CONTEXT_N, timeout: float = 30.0,
                 key: Optional[str] = None):
        self._on_confirmed = on_confirmed
        self._context_n = context_n
        self._timeout = timeout
        self._key = key if key is not None else _get_key()
        self._queue = queue.Queue()
        self._stop = object()
        self._context: List[str] = []
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, utt: Dict) -> None:
        """utt = asr_final 载荷：{utt_id, start_ms, end_ms, raw_text}"""
        self._queue.put(utt)

    def close(self) -> None:
        """等队列排空后停线程；返回时所有 llm_confirmed 都已回调。"""
        self._queue.put(self._stop)
        self._worker.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._stop:
                self._queue.task_done()
                return
            try:
                text = polish_text(item["raw_text"], self._context,
                                   self._timeout, self._key, self._context_n)
                ev = {"t": "llm_confirmed", "utt_id": item["utt_id"],
                      "start_ms": item["start_ms"], "end_ms": item["end_ms"],
                      "raw_text": item["raw_text"], "polished_text": text}
            except Exception as exc:      # 网络/解析失败不中断整场，退回原文
                ev = {"t": "llm_confirmed", "utt_id": item["utt_id"],
                      "start_ms": item["start_ms"], "end_ms": item["end_ms"],
                      "raw_text": item["raw_text"],
                      "polished_text": item["raw_text"],
                      "error": ("%s: %s" % (type(exc).__name__, exc))[:200]}
            self._context.append(ev["polished_text"])
            # context_n<=0 表示"不给上下文"：[-0:] 是整个列表，会无限增长。
            self._context = self._context[-self._context_n:] if self._context_n > 0 else []
            try:
                self._on_confirmed(ev)
            except Exception as cb_exc:   # 回调抛错不能杀死 worker，否则后续全部静默丢弃
                print("[polisher] on_confirmed 失败：%s" % cb_exc, file=sys.stderr)
            self._queue.task_done()
