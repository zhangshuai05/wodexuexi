# -*- coding: utf-8 -*-
"""说话人 turn 合并层：utterances 保细粒度，turns 供显示/导出。

设计规则：utterances = 声纹/审计粒度，turns = 显示/导出粒度。
本模块只做纯文本/时间合并，不读写 session 目录，也不依赖 session_store（避免循环导入）。
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

STREAM_TURN_GAP_MS = 2000    # 流式显示：与上一句间隔超过它才换行
FINAL_TURN_SPLIT_MS = 8000   # 最终合并：同一说话人也要在长静音处断开

# 中文标点与汉字一样不插空格；其余字符（英文、数字等）之间补一个空格
_CJK_PUNCT = "，。！？、；："
_END_PUNCT = "，。！？、；：,.!?…“”‘’（）【】"


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否属于中日韩统一表意文字或中文标点。"""
    return "一" <= ch <= "鿿" or ch in _CJK_PUNCT


def join_texts(parts: List[str]) -> str:
    """拼接多个文本片段。

    边界已有标点则直接相连；否则中文侧补逗号（长 turn 不连成一整串），
    纯英文之间补空格。这是机械补标点，不改变原句段的措辞。
    """
    joined = ""
    for part in parts:
        text = (part or "").strip()
        if not text:
            continue
        if joined and joined[-1] not in _END_PUNCT and text[0] not in _END_PUNCT:
            joined += "，" if (_is_cjk(joined[-1]) or _is_cjk(text[0])) else " "
        joined += text
    return joined


def build_turns(
    utterances: List[Dict],
    *,
    split_gap_ms: int = FINAL_TURN_SPLIT_MS,
) -> List[Dict]:
    """把 utterances 合并为 turns。

    有 spk_label 时：相邻且同一说话人、且间隔 < split_gap_ms 才合并。
    全部 spk_label 为 None（--no-diar）时：只按流式的 STREAM_TURN_GAP_MS 合并，
    避免整场退化成一个大 turn。
    """
    if not utterances:
        return []

    has_labels = any(utt.get("spk_label") is not None for utt in utterances)
    gap_ms = split_gap_ms if has_labels else STREAM_TURN_GAP_MS

    drafts: List[Dict] = []  # 合并中的草稿，最后统一生成 turn_id
    for utt in utterances:
        start_ms = int(utt["start_ms"])
        end_ms = int(utt["end_ms"])
        text = utt.get("polished_text") or utt.get("raw_text") or ""
        spk_label = utt.get("spk_label")

        if drafts:
            draft = drafts[-1]
            same_spk = (not has_labels) or (spk_label == draft["spk_label"])
            if same_spk and start_ms - draft["end_ms"] < gap_ms:
                draft["end_ms"] = max(draft["end_ms"], end_ms)
                draft["parts"].append(text)
                draft["utterance_ids"].append(utt["utt_id"])
                draft["group"].append(utt)
                continue

        drafts.append({
            "spk": utt.get("spk"),
            "spk_label": spk_label,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "parts": [text],
            "utterance_ids": [utt["utt_id"]],
            "group": [utt],
        })

    turns: List[Dict] = []
    for index, draft in enumerate(drafts, start=1):
        group = draft["group"]
        turns.append({
            "turn_id": f"turn_{index:04d}",
            "spk": draft["spk"],
            "spk_label": draft["spk_label"],
            "start_ms": draft["start_ms"],
            "end_ms": draft["end_ms"],
            "text": join_texts(draft["parts"]),
            "utterance_ids": draft["utterance_ids"],
            "corrected": any(bool(u.get("corrected")) for u in group),
            "bypassed": bool(group) and all(bool(u.get("bypassed")) for u in group),
        })
    return turns


class StreamTurnTracker:
    """流式事件 -> turn 的在线归属器。

    run_offline 每个 run() 创建一个实例；assign() 就地补充 turn_id 与累计文本字段，
    不改动事件里原有的单句字段（utt_id / raw_text / polished_text 等）。
    """

    def __init__(self, gap_ms: int = STREAM_TURN_GAP_MS) -> None:
        self._lock = threading.Lock()
        self.gap_ms = gap_ms
        self.utt_turn: Dict[str, str] = {}                 # utt_id -> turn_id
        self.turn_texts: Dict[str, Dict[str, str]] = {}    # turn_id -> {utt_id: 文本}，插入序
        self.last_end_ms: Optional[int] = None
        self.count = 0
        self._current_turn: Optional[str] = None

    def _new_turn(self) -> str:
        """开启新 turn，编号 1-based、四位零填充。"""
        self.count += 1
        turn_id = f"turn_{self.count:04d}"
        self.turn_texts[turn_id] = {}
        self._current_turn = turn_id
        return turn_id

    def _turn_for(self, ev: Dict) -> str:
        """为新 utt 分配合适的 turn：首句或与上一句间隔达到阈值时换 turn。"""
        utt_id = ev.get("utt_id")
        known = self.utt_turn.get(utt_id)
        if known is not None:
            return known

        start_ms = int(ev.get("start_ms") or 0)
        if self._current_turn is None:
            turn_id = self._new_turn()
        elif self.last_end_ms is None:
            # 上一句还没 asr_final（没有结束时间），只能并入当前 turn
            turn_id = self._current_turn
        elif start_ms - self.last_end_ms >= self.gap_ms:
            turn_id = self._new_turn()
        else:
            turn_id = self._current_turn

        self.utt_turn[utt_id] = turn_id
        return turn_id

    def _texts(self, turn_id: str) -> List[str]:
        return list(self.turn_texts[turn_id].values())

    def assign(self, ev: Dict) -> Dict:
        """就地补充并返回事件；未知事件类型原样返回，保证流水线不被拖垮。"""
        # 喂入线程（run_offline）与润色线程（on_polish）会并发调用；
        # 整个归属/累计文本更新必须是原子的，否则遍历 turn_texts 时插入会抛
        # RuntimeError: dictionary changed size during iteration。
        with self._lock:
            kind = ev.get("t")
            utt_id = ev.get("utt_id")

            if kind == "asr_partial" and utt_id is not None:
                turn_id = self._turn_for(ev)
                self.turn_texts[turn_id][utt_id] = ev.get("text") or ""
                ev["turn_id"] = turn_id
                ev["turn_text"] = join_texts(self._texts(turn_id))
                return ev

            if kind == "asr_final" and utt_id is not None:
                turn_id = self._turn_for(ev)
                self.turn_texts[turn_id][utt_id] = ev.get("raw_text") or ""
                ev["turn_id"] = turn_id
                ev["turn_text"] = join_texts(self._texts(turn_id))
                end_ms = ev.get("end_ms")
                if end_ms is not None:
                    self.last_end_ms = max(self.last_end_ms or 0, int(end_ms))
                return ev

            if kind == "llm_confirmed" and utt_id is not None:
                # 不回填新 utt 的归属；未知 utt 退到当前 turn，绝不崩溃
                turn_id = self.utt_turn.get(utt_id)
                if turn_id is None:
                    turn_id = self._current_turn
                if turn_id is None:
                    turn_id = self._new_turn()
                self.turn_texts[turn_id][utt_id] = ev.get("polished_text") or ""
                ev["turn_id"] = turn_id
                ev["turn_polished_text"] = join_texts(self._texts(turn_id))
                return ev

            return ev
