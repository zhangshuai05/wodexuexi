# -*- coding: utf-8 -*-
"""说话人 turn 合并层：utterances 保细粒度，turns 供显示/导出。

设计规则：utterances = 声纹/审计粒度，turns = 显示/导出粒度。
本模块只做纯文本/时间合并，不读写 session 目录，也不依赖 session_store（避免循环导入）。
"""
from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

STREAM_TURN_GAP_MS = 5000    # 流式显示：只有长停顿才换段，避免说话人未知时切得太碎
FINAL_TURN_SPLIT_MS = 8000   # 最终合并：同一说话人也要在长静音处断开
MIN_SUB_SEGMENT_MS = 100     # sub_turns 拆分后短于此的碎片并入相邻段
PAUSE_COMMA_GAP_MS = 450     # 有明显停顿时才在连续中文句段间补逗号

# 中文标点与汉字一样不插空格；其余字符（英文、数字等）之间补一个空格
_CJK_PUNCT = "，。！？、；：“”‘’（）《》【】…"
_END_PUNCT = "，。！？、；：,.!?…'\" ()【】”’》"
_CJK_RANGE = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]"
_RE_CJK_SPACE_BOTH = re.compile(f"({_CJK_RANGE})\\s+({_CJK_RANGE})")
_RE_CJK_SPACE_LEFT = re.compile(f"({_CJK_RANGE})\\s+([,.:;!?%])")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否属于中日韩统一表意文字或中文标点。"""
    return "一" <= ch <= "鿿" or ch in _CJK_PUNCT


def clean_cjk_spaces(text: Optional[str]) -> str:
    """删除 SEACO 在汉字和中文标点间插入的分词空格，保留英文词间空格。"""
    if not text:
        return ""
    value = str(text).strip()
    if not value:
        return ""
    # 两轮即可合并连续分词空格，同时保留英文/数字单词间的空格。
    value = _RE_CJK_SPACE_BOTH.sub(r"\1\2", value)
    value = _RE_CJK_SPACE_BOTH.sub(r"\1\2", value)
    value = _RE_CJK_SPACE_LEFT.sub(r"\1\2", value)
    return _RE_MULTI_SPACE.sub(" ", value).strip()


def join_texts(parts: List[str], *, insert_comma: bool = True) -> str:
    """拼接多个文本片段。

    边界已有标点则直接相连；否则中文侧补逗号（长 turn 不连成一整串），
    纯英文之间补空格。这是机械补标点，不改变原句段的措辞。
    """
    joined = ""
    for part in parts:
        text = clean_cjk_spaces(part)
        if not text:
            continue
        if joined and joined[-1] not in _END_PUNCT and text[0] not in _END_PUNCT:
            if _is_cjk(joined[-1]) or _is_cjk(text[0]):
                if insert_comma:
                    joined += "，"
            else:
                joined += " "
        joined += text
    return joined


def _join_turn_utterances(group: List[Dict]) -> str:
    """按离线拼接标记和实际停顿连接 utterance，避免在被切开的词中补逗号。"""
    joined = ""
    prev_end_ms: Optional[int] = None
    for utt in group:
        final_text = utt.get("final_text")
        raw_piece = (final_text if final_text is not None else
                     utt.get("polished_text") or utt.get("raw_text") or "")
        text = clean_cjk_spaces(raw_piece)
        if not text:
            continue

        start_ms = int(utt.get("start_ms") or 0)
        end_ms = int(utt.get("end_ms") or 0)
        gap_ms = start_ms - prev_end_ms if prev_end_ms is not None else 0
        if joined and joined[-1] not in _END_PUNCT and text[0] not in _END_PUNCT:
            if _is_cjk(joined[-1]) or _is_cjk(text[0]):
                if (not utt.get("grouped_cont")
                        and gap_ms >= PAUSE_COMMA_GAP_MS):
                    joined += "，"
            else:
                joined += " "
        joined += text
        prev_end_ms = max(prev_end_ms or 0, end_ms)
    return joined


def _split_text_by_weights(text, weights: List[int]):
    """按时间权重把一段文本切给多个子段；None 保持 None。"""
    if text is None:
        return [None] * len(weights)
    count = len(weights)
    if count <= 1:
        return [text]
    total = sum(weights) or count
    bounds = [0]
    acc = 0
    for weight in weights[:-1]:
        acc += max(1, weight)
        bounds.append(int(round(len(text) * acc / total)))
    bounds.append(len(text))
    for index in range(1, len(bounds)):
        bounds[index] = max(bounds[index], bounds[index - 1])
    return [text[bounds[i]:bounds[i + 1]] for i in range(count)]


def split_utterance_by_sub_turns(
    utt: Dict,
    sub_turns,
    *,
    min_segment_ms: int = MIN_SUB_SEGMENT_MS,
) -> List[Dict]:
    """把一句按上游 `sub_turns` 拆成独立的单人子段。

    `sub_turns` 来自 `backfill_speakers(..., with_segments=True)`，只含
    “非主说话人且持续 >=1.0s”的片段。其余时间归主说话人。
    文本按子段时长比例切分；离线二遍会逐子段重转写并覆盖 final_text。
    没有有效 sub_turns 时原样返回 `[utt]`。
    """
    try:
        start, end = int(utt["start_ms"]), int(utt["end_ms"])
    except (KeyError, TypeError, ValueError):
        return [utt]
    if not sub_turns or end <= start:
        return [utt]

    main_spk, main_label = utt.get("spk"), utt.get("spk_label")
    marks = []
    for item in sub_turns:
        if not isinstance(item, dict):
            continue
        try:
            a, b = int(item["start_ms"]), int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        a, b = max(start, a), min(end, b)
        if b - a < min_segment_ms:
            continue
        spk = item.get("spk")
        label = item.get("spk_label") or item.get("speaker")
        if spk is None and isinstance(label, str) and label.startswith("speaker_"):
            try:
                spk = int(label.split("_", 1)[1])
            except ValueError:
                spk = None
        if not label and spk is not None:
            label = f"speaker_{spk}"
        marks.append((a, b, spk, label))
    if not marks:
        return [utt]
    marks.sort(key=lambda item: (item[0], item[1]))

    # 先按时间切出「主说话人 - 子说话人 - 主说话人…」分区
    segments = []
    cursor = start
    for a, b, spk, label in marks:
        if a - cursor >= min_segment_ms:
            segments.append((cursor, a, main_spk, main_label, False))
        elif a > cursor and segments:
            # 太短的主说话人空隙：并入前一个子段，避免制造碎片
            prev = segments[-1]
            segments[-1] = (prev[0], a, prev[2], prev[3], prev[4])
        seg_start = max(a, cursor)
        if b - seg_start >= min_segment_ms:
            segments.append((seg_start, b, spk, label, True))
        cursor = max(cursor, b)
    if end - cursor >= min_segment_ms:
        segments.append((cursor, end, main_spk, main_label, False))
    elif cursor < end and segments:
        prev = segments[-1]
        segments[-1] = (prev[0], end, prev[2], prev[3], prev[4])

    # 去重叠、保证单调
    cleaned = []
    for seg in segments:
        if cleaned and seg[0] < cleaned[-1][1]:
            seg = (cleaned[-1][1], seg[1], seg[2], seg[3], seg[4])
        if seg[1] - seg[0] >= min_segment_ms:
            cleaned.append(seg)
    # 相邻同标签段合并（微小空隙被并入后可能留下两段同说话人碎片）
    merged_segments = []
    for seg in cleaned:
        prev = merged_segments[-1] if merged_segments else None
        if (prev is not None and prev[3] == seg[3] and prev[4] == seg[4]
                and seg[0] <= prev[1] + 1):
            merged_segments[-1] = (prev[0], max(prev[1], seg[1]),
                                   prev[2], prev[3], prev[4])
        else:
            merged_segments.append(seg)
    cleaned = merged_segments
    if len(cleaned) <= 1 or not any(seg[4] for seg in cleaned):
        return [utt]

    weights = [seg[1] - seg[0] for seg in cleaned]
    raw_parts = _split_text_by_weights(utt.get("raw_text") or "", weights)
    polished_parts = _split_text_by_weights(utt.get("polished_text"), weights)
    final_parts = _split_text_by_weights(utt.get("final_text"), weights)
    parent_id = utt.get("parent_utt_id") or utt["utt_id"]
    out = []
    for index, (seg, raw, polished, final) in enumerate(
            zip(cleaned, raw_parts, polished_parts, final_parts), start=1):
        a, b, spk, label, is_sub = seg
        out.append({
            "utt_id": f"{utt['utt_id']}#{index}",
            "parent_utt_id": parent_id,
            "start_ms": a,
            "end_ms": b,
            "raw_text": raw,
            "polished_text": polished,
            "final_text": final,
            "spk": spk,
            "spk_label": label,
            "overlap_spk": utt.get("overlap_spk"),
            "corrected": bool(utt.get("corrected", False)),
            "bypassed": bool(utt.get("bypassed", False)),
            "error": utt.get("error"),
            "is_sub_turn": is_sub,
        })
    return out


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
        spk_label = utt.get("spk_label")

        if drafts:
            draft = drafts[-1]
            same_spk = (not has_labels) or (spk_label == draft["spk_label"])
            if same_spk and start_ms - draft["end_ms"] < gap_ms:
                draft["end_ms"] = max(draft["end_ms"], end_ms)
                draft["utterance_ids"].append(utt["utt_id"])
                draft["group"].append(utt)
                continue

        drafts.append({
            "spk": utt.get("spk"),
            "spk_label": spk_label,
            "start_ms": start_ms,
            "end_ms": end_ms,
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
            "text": _join_turn_utterances(group),
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
        self._partial_utt: Optional[str] = None   # 正在说、尚未 asr_final 的句子

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

    def _split_texts(self, turn_id: str):
        """把 turn 文本拆成 (已说完的正体部分, 正在说的斜体部分)。"""
        partial_utt = self._partial_utt if self.utt_turn.get(self._partial_utt) == turn_id else None
        confirmed: List[str] = []
        partial = ""
        for utt_id, text in self.turn_texts[turn_id].items():
            if utt_id == partial_utt:
                partial = text
            else:
                confirmed.append(text)
        return join_texts(confirmed), partial

    def _annotate(self, ev: Dict, turn_id: str) -> Dict:
        """补充 turn 字段：整段、已确认段、当前句段。"""
        confirmed, partial = self._split_texts(turn_id)
        ev["turn_id"] = turn_id
        ev["turn_confirmed_text"] = confirmed
        ev["turn_partial_text"] = partial
        ev["turn_text"] = join_texts([confirmed, partial]) if partial else confirmed
        return ev

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
                self._partial_utt = utt_id
                return self._annotate(ev, turn_id)

            if kind == "asr_final" and utt_id is not None:
                turn_id = self._turn_for(ev)
                self.turn_texts[turn_id][utt_id] = ev.get("raw_text") or ""
                if self._partial_utt == utt_id:
                    self._partial_utt = None
                end_ms = ev.get("end_ms")
                if end_ms is not None:
                    self.last_end_ms = max(self.last_end_ms or 0, int(end_ms))
                return self._annotate(ev, turn_id)

            if kind == "llm_confirmed" and utt_id is not None:
                # 不回填新 utt 的归属；未知 utt 退到当前 turn，绝不崩溃
                turn_id = self.utt_turn.get(utt_id)
                if turn_id is None:
                    turn_id = self._current_turn
                if turn_id is None:
                    turn_id = self._new_turn()
                self.turn_texts[turn_id][utt_id] = ev.get("polished_text") or ""
                ev["turn_polished_text"] = join_texts(self._texts(turn_id))
                return self._annotate(ev, turn_id)

            return ev
