# -*- coding: utf-8 -*-
"""离线质量度量：session 文本对 TextGrid 真值的 CER。

用法:
    python cer_eval.py output/sessions/s_xxx TextGrid/xxx.TextGrid
    python cer_eval.py <session> <TextGrid> --skip-start-ms 23000 --legacy --json

口径（2026-10-18 修正）:
- 主口径是**标准 Levenshtein**（替换=1，删除=1，插入=1），并用 rapidfuzz 的
  editops 精确拆出 Sub / Del / Ins。旧的 difflib 公式是 LCS indel 距离
  （len(ref)+len(hyp)-2*matched），把一次替换记成 2 次编辑，会系统性高估 CER；
  `--legacy` 仍打印旧口径，仅用于和历史数字对照，不得混用。
- 参考与预测都先做 TN：剥掉 `<sil>` / `<%>` 等尖括号标记，剥标点与空白，
  并把中文数字串与阿拉伯数字统一成同一种数字串（"零零六" 与 "006" 都归一为 "006"）。
- 额外输出 VAD 覆盖率：参考语音里有多少字符**完全没有**被 session 的 VAD 句覆盖，
  用来把 Del 拆成「VAD 漏检」与「ASR 内部删字」。
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover - 环境缺依赖时给出明确指引
    Levenshtein = None


_TIME_RE = re.compile(r"^\s*(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)\s*$")


def parse_time_ms(value: str) -> int:
    """把 mm:ss.s / hh:mm:ss.s / 纯毫秒解析成 int 毫秒。"""
    text = str(value).strip()
    if not text:
        raise ValueError("空时间")
    if re.fullmatch(r"\d+", text):
        return int(text)
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"无法解析时间: {value}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = float(match.group("s"))
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))


def format_time_ms(ms: int) -> str:
    """格式化为 mm:ss.s（超过 1 小时用 hh:mm:ss.s）。"""
    total = max(0, int(ms))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis // 100}"
    return f"{minutes:02d}:{seconds:02d}.{millis // 100}"


def parse_window(value: str) -> Tuple[int, int]:
    """解析 START-END，支持 mm:ss.s / hh:mm:ss.s / 纯毫秒。"""
    text = str(value).strip()
    if "-" not in text:
        raise ValueError("窗口格式应为 START-END")
    start, end = text.split("-", 1)
    start_ms, end_ms = parse_time_ms(start), parse_time_ms(end)
    if end_ms < start_ms:
        raise ValueError("窗口结束早于开始")
    return start_ms, end_ms


# ---------------------------------------------------------------------------
# TextGrid 读取
# ---------------------------------------------------------------------------
def read_textgrid(path) -> Dict[str, List[Tuple[float, float, str]]]:
    """返回 {tier: [(xmin, xmax, text)]}（不修改原始文本）。"""
    tiers: Dict[str, List[Tuple[float, float, str]]] = {}
    name = None
    cur = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("intervals ["):
            cur = None
        elif line.startswith("xmin = "):
            cur = {"xmin": float(line.split("=", 1)[1])}
        elif line.startswith("xmax = ") and cur is not None:
            cur["xmax"] = float(line.split("=", 1)[1])
        elif line.startswith("text = ") and cur is not None:
            cur["text"] = line.split("=", 1)[1].strip().strip('"')
            tiers.setdefault(name, []).append((cur["xmin"], cur["xmax"], cur["text"]))
            cur = None
    return tiers


# ---------------------------------------------------------------------------
# 文本归一化（TN）
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]*>")
_KEEP_RE = re.compile(r"[^一-鿿A-Za-z0-9]")

# 中文数字字符 -> 0-9
_CN_DIGITS = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "壹": 1, "幺": 1,
    "二": 2, "贰": 2, "两": 2, "兩": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
_CN_UNITS = {
    "十": 10, "拾": 10,
    "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
    "万": 10000, "萬": 10000,
    "亿": 100000000, "億": 100000000,
}
_NUM_CHARS = "".join(_CN_DIGITS) + "".join(_CN_UNITS)
_NUM_RUN_RE = re.compile(rf"[0-9{re.escape(_NUM_CHARS)}]+")


def _parse_cn_number(run: str) -> Optional[int]:
    """把中文/混合数字串解析成 int；失败返回 None。

    只用于 TN，不追求语言学完备：两侧用同一把尺子，宁可机械也不漏。
    """
    if not run:
        return None
    if not any(ch in _CN_UNITS for ch in run):
        # 纯数字串：逐位拼接（"零零六" -> "006"，"二零二六" -> "2026"）
        parts = []
        for ch in run:
            if ch.isdigit():
                parts.append(str(int(ch)))
            elif ch in _CN_DIGITS:
                parts.append(str(_CN_DIGITS[ch]))
            else:
                return None
        return int("".join(parts) or "0")

    total = 0
    section = 0
    number = 0
    saw = False
    for ch in run:
        if ch.isdigit():
            number = int(ch)
            saw = True
        elif ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
            saw = True
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            saw = True
            if unit == 10000:
                section += number
                total += section * 10000
                section = 0
                number = 0
            elif unit == 100000000:
                section += number
                total = (total + section) * 100000000
                section = 0
                number = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
                number = 0
        else:
            return None
    if not saw:
        return None
    return total + section + number


def _canon_number(run) -> str:
    """数字串规范化：阿拉伯数字保持；中文数字转阿拉伯。

    同时接受 re.Match（供 re.sub 直接调用）与 str。
    """
    if hasattr(run, "group"):
        run = run.group(0)
    if all(ch.isdigit() for ch in run):
        return run
    if not any(ch in _CN_UNITS for ch in run):
        return "".join(
            str(int(ch)) if ch.isdigit() else str(_CN_DIGITS[ch])
            for ch in run
        )
    value = _parse_cn_number(run)
    return run if value is None else str(value)


def normalize(text: str) -> str:
    """CER 前的统一归一化：去标记、数字统一、去标点空白、英文小写。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = _TAG_RE.sub("", text)          # <sil> / <%> / <unk> 等
    text = _NUM_RUN_RE.sub(_canon_number, text)
    text = text.lower()
    return _KEEP_RE.sub("", text)


# ---------------------------------------------------------------------------
# 编辑距离与拆解
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EditCounts:
    ref_len: int
    sub: int
    dele: int
    ins: int

    @property
    def edits(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def cer(self) -> float:
        if self.ref_len <= 0:
            return 0.0 if self.edits == 0 else 1.0
        return self.edits / self.ref_len

    @property
    def sub_rate(self) -> float:
        return self.sub / self.ref_len if self.ref_len else 0.0

    @property
    def del_rate(self) -> float:
        return self.dele / self.ref_len if self.ref_len else 0.0

    @property
    def ins_rate(self) -> float:
        return self.ins / self.ref_len if self.ref_len else 0.0


def _require_rapidfuzz() -> None:
    if Levenshtein is None:
        raise RuntimeError(
            "需要 rapidfuzz 才能拆 Sub/Del/Ins；"
            "请在 funasr 环境执行: python -m pip install rapidfuzz"
        )


def edit_counts(ref: str, hyp: str) -> EditCounts:
    """标准 Levenshtein 的精确 Sub/Del/Ins 拆解。"""
    _require_rapidfuzz()
    if not ref:
        return EditCounts(0, 0, 0, len(hyp))
    ops = Levenshtein.editops(ref, hyp)
    sub = sum(1 for op in ops if op.tag == "replace")
    dele = sum(1 for op in ops if op.tag == "delete")
    ins = sum(1 for op in ops if op.tag == "insert")
    return EditCounts(len(ref), sub, dele, ins)


def cer(ref: str, hyp: str) -> float:
    """标准 Levenshtein CER（编辑数 / 参考长度）。"""
    return edit_counts(ref, hyp).cer


def legacy_lcs_cer(ref: str, hyp: str) -> float:
    """旧口径：difflib 最长匹配块 + LCS indel 距离（仅作历史对照）。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    matcher = SequenceMatcher(None, ref, hyp, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    edits = len(ref) + len(hyp) - 2 * matched
    return edits / len(ref)


# ---------------------------------------------------------------------------
# 区间与覆盖率
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefInterval:
    start_ms: int
    end_ms: int
    text: str
    tier: str
    timed: bool = True


@dataclass(frozen=True)
class Coverage:
    ref_chars: int
    ref_ms: int
    uncovered_chars: float
    uncovered_ms: int
    covered_ratio: float


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def reference_intervals(
    textgrid_path, *, skip_start_ms: int = 0
) -> List[RefInterval]:
    out: List[RefInterval] = []
    for tier, rows in read_textgrid(textgrid_path).items():
        for start_s, end_s, raw_text in rows:
            start_ms = int(round(start_s * 1000))
            end_ms = int(round(end_s * 1000))
            if end_ms <= skip_start_ms:
                continue
            if start_ms < skip_start_ms:
                # 跨边界的区间整段丢弃：文字不能按时间比例切开，宁可少算
                continue
            text = normalize(raw_text)
            if not text:
                continue
            out.append(RefInterval(start_ms, end_ms, text, tier))
    out.sort(key=lambda item: (item.start_ms, item.end_ms, item.tier))
    return out


_REF_TS_RE = re.compile(
    r"^\s*\[(?P<start>[^\]]+?)\s*-\s*(?P<end>[^\]]+?)\]\s*(?P<rest>.*)$")
_REF_SPK_RE = re.compile(
    r"^(?P<spk>[A-Za-z][A-Za-z0-9_?-]{0,20})\s*[:：]\s*(?P<text>.*)$")


def read_ref_text(path) -> List[RefInterval]:
    """读取 UTF-8 文本参考。

    带时间戳的行：`[mm:ss.s-mm:ss.s] speaker: text`；speaker 可省略。
    不带时间戳的行：整行都是文本，timed=False（只能全片评测）。
    """
    out: List[RefInterval] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts_match = _REF_TS_RE.match(line)
        if ts_match:
            start_ms = parse_time_ms(ts_match.group("start"))
            end_ms = parse_time_ms(ts_match.group("end"))
            if end_ms < start_ms:
                raise ValueError(f"参考区间结束早于开始: {line}")
            rest = ts_match.group("rest")
            spk_match = _REF_SPK_RE.match(rest)
            if spk_match:
                tier = spk_match.group("spk")
                text = spk_match.group("text")
            else:
                tier = "ref"
                text = rest
            timed = True
        else:
            start_ms = end_ms = 0
            tier = "ref"
            text = line
            timed = False
        normalized = normalize(text)
        if normalized:
            out.append(RefInterval(start_ms, end_ms, normalized, tier, timed))
    if out and all(item.timed for item in out):
        out.sort(key=lambda item: (item.start_ms, item.end_ms, item.tier))
    return out


def coverage(ref_intervals: Sequence[RefInterval],
             utterance_intervals: Sequence[Tuple[int, int]],
             *, pad_ms: int = 0) -> Coverage:
    padded = [(s - pad_ms, e + pad_ms) for s, e in utterance_intervals]
    merged = _merge_intervals(padded)
    total_chars = sum(len(item.text) for item in ref_intervals)
    total_ms = 0
    uncovered_ms = 0
    uncovered_chars = 0.0
    for item in ref_intervals:
        duration = max(1, item.end_ms - item.start_ms)
        covered = sum(_overlap_ms(item.start_ms, item.end_ms, a, b) for a, b in merged)
        covered = min(covered, duration)
        total_ms += duration
        if covered <= 0:
            uncovered_ms += duration
            uncovered_chars += len(item.text)
        elif covered < duration:
            uncovered_ms += duration - covered
            uncovered_chars += len(item.text) * (duration - covered) / duration
    return Coverage(
        ref_chars=total_chars,
        ref_ms=total_ms,
        uncovered_chars=uncovered_chars,
        uncovered_ms=uncovered_ms,
        covered_ratio=(1.0 - uncovered_chars / total_chars) if total_chars else 0.0,
    )


def cross_speaker_overlap(ref_intervals: Sequence[RefInterval]) -> Tuple[int, int]:
    """返回 (存在跨 tier 重叠的参考字符数, 区间数)。

    单支麦 + 单流 ASR 对重叠语音天然吃亏；这部分字符是 Del/Sub 的高危区。
    """
    marked_chars = 0
    marked_count = 0
    items = list(ref_intervals)
    for i, item in enumerate(items):
        for j, other in enumerate(items):
            if i == j or item.tier == other.tier:
                continue
            if min(item.end_ms, other.end_ms) > max(item.start_ms, other.start_ms):
                marked_chars += len(item.text)
                marked_count += 1
                break
    return marked_chars, marked_count


def _new_bucket() -> Dict[str, int]:
    return {"ref_chars": 0, "sub": 0, "del": 0, "ins": 0}


def _finalize_bucket(bucket: Dict[str, int]) -> Dict[str, float]:
    ref_chars = bucket["ref_chars"]
    edits = bucket["sub"] + bucket["del"] + bucket["ins"]
    return {
        **bucket,
        "cer": edits / ref_chars if ref_chars else 0.0,
        "sub_rate": bucket["sub"] / ref_chars if ref_chars else 0.0,
        "del_rate": bucket["del"] / ref_chars if ref_chars else 0.0,
        "ins_rate": bucket["ins"] / ref_chars if ref_chars else 0.0,
    }


def attribute_edits(
    ref_intervals: Sequence[RefInterval],
    hyp: str,
    utterance_intervals: Sequence[Tuple[int, int]],
    *,
    pad_ms: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """把每个 Sub/Del/Ins 精确归因到参考区间。

    用 Levenshtein editops 的 src_pos 映射回参考字符位置，再按
    「VAD 覆盖状态 / 跨说话人重叠 / 说话人 tier」三个维度聚合。
    三个维度内部各自是完整划分，因此各自的总编辑数应等于 S+D+I。
    """
    _require_rapidfuzz()
    if not ref_intervals:
        return {"coverage": {}, "overlap": {}, "tier": {}, "total_edits": 0}

    ref = "".join(item.text for item in ref_intervals)
    starts: List[int] = []
    cursor = 0
    for item in ref_intervals:
        starts.append(cursor)
        cursor += len(item.text)

    padded = _merge_intervals(
        [(s - pad_ms, e + pad_ms) for s, e in utterance_intervals])
    statuses: List[str] = []
    for item in ref_intervals:
        duration = max(1, item.end_ms - item.start_ms)
        covered = sum(
            _overlap_ms(item.start_ms, item.end_ms, a, b) for a, b in padded)
        ratio = min(covered, duration) / duration
        if ratio <= 0:
            statuses.append("0%")
        elif ratio < 0.25:
            statuses.append("0-25%")
        elif ratio < 0.5:
            statuses.append("25-50%")
        elif ratio < 0.75:
            statuses.append("50-75%")
        elif ratio < 0.999:
            statuses.append("75-100%")
        else:
            statuses.append("100%")

    overlaps: List[bool] = []
    for i, item in enumerate(ref_intervals):
        overlaps.append(any(
            j != i and other.tier != item.tier
            and min(item.end_ms, other.end_ms) > max(item.start_ms, other.start_ms)
            for j, other in enumerate(ref_intervals)
        ))

    coverage_buckets = {key: _new_bucket()
                        for key in ("0%", "0-25%", "25-50%", "50-75%",
                                    "75-100%", "100%")}
    overlap_buckets = {key: _new_bucket()
                       for key in ("overlap", "non_overlap")}
    tier_buckets: Dict[str, Dict[str, int]] = {}

    for item, status, is_overlap in zip(ref_intervals, statuses, overlaps):
        for bucket in (
            coverage_buckets[status],
            overlap_buckets["overlap" if is_overlap else "non_overlap"],
            tier_buckets.setdefault(item.tier, _new_bucket()),
        ):
            bucket["ref_chars"] += len(item.text)

    for op in Levenshtein.editops(ref, hyp):
        index = bisect.bisect_right(starts, op.src_pos) - 1
        index = max(0, min(index, len(ref_intervals) - 1))
        item = ref_intervals[index]
        kind = {"replace": "sub", "delete": "del", "insert": "ins"}[op.tag]
        for bucket in (
            coverage_buckets[statuses[index]],
            overlap_buckets["overlap" if overlaps[index] else "non_overlap"],
            tier_buckets[item.tier],
        ):
            bucket[kind] += 1

    return {
        "coverage": {key: _finalize_bucket(value)
                     for key, value in coverage_buckets.items()},
        "overlap": {key: _finalize_bucket(value)
                    for key, value in overlap_buckets.items()},
        "tier": {key: _finalize_bucket(value)
                 for key, value in sorted(tier_buckets.items())},
        "total_edits": sum(
            value["sub"] + value["del"] + value["ins"]
            for value in coverage_buckets.values()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VersionMetrics:
    name: str
    hyp_len: int
    counts: EditCounts
    legacy_cer: Optional[float] = None

    def to_dict(self) -> Dict:
        payload = {
            "hyp_len": self.hyp_len,
            "ref_len": self.counts.ref_len,
            "cer": self.counts.cer,
            "sub_rate": self.counts.sub_rate,
            "del_rate": self.counts.del_rate,
            "ins_rate": self.counts.ins_rate,
            "sub": self.counts.sub,
            "del": self.counts.dele,
            "ins": self.counts.ins,
        }
        if self.legacy_cer is not None:
            payload["legacy_lcs_cer"] = self.legacy_cer
        return payload


def _text_versions(obj: Dict) -> Dict[str, str]:
    utterances = obj.get("utterances") or []
    raw = "".join(normalize(u.get("raw_text") or "") for u in utterances)
    polished = "".join(
        normalize(u.get("polished_text") or u.get("raw_text") or "")
        for u in utterances
    )
    # strict：只统计真正有 final_text 的句子（缺失的算空），用于诊断离线二遍覆盖率
    final_strict = "".join(normalize(u.get("final_text") or "") for u in utterances)
    # display：与 session.json.turns 完全一致（final -> polished -> raw 逐句回退）
    turns = "".join(normalize(t.get("text") or "") for t in obj.get("turns") or [])
    return {
        "raw": raw,
        "polished": polished,
        "final": final_strict,
        "turns": turns,
    }


def evaluate(
    session_dir,
    textgrid_path,
    *,
    skip_start_ms: int = 0,
    legacy: bool = False,
    pad_ms: int = 80,
) -> Dict:
    session_dir = Path(session_dir)
    obj = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    ref_intervals = reference_intervals(
        textgrid_path, skip_start_ms=skip_start_ms)
    ref = "".join(item.text for item in ref_intervals)

    utterances = obj.get("utterances") or []
    utt_intervals = [
        (int(u.get("start_ms") or 0), int(u.get("end_ms") or 0))
        for u in utterances
    ]

    texts = _text_versions(obj)
    versions: Dict[str, VersionMetrics] = {}
    for name, hyp in texts.items():
        versions[name] = VersionMetrics(
            name=name,
            hyp_len=len(hyp),
            counts=edit_counts(ref, hyp),
            legacy_cer=legacy_lcs_cer(ref, hyp) if legacy else None,
        )

    attribution = attribute_edits(
        ref_intervals, texts["turns"], utt_intervals, pad_ms=pad_ms)

    raw_cover = coverage(ref_intervals, utt_intervals, pad_ms=0)
    padded_cover = coverage(ref_intervals, utt_intervals, pad_ms=pad_ms)
    overlap_chars, overlap_intervals = cross_speaker_overlap(ref_intervals)

    tier_stats: Dict[str, Dict] = {}
    for item in ref_intervals:
        stat = tier_stats.setdefault(
            item.tier, {"chars": 0, "intervals": 0, "overlap_chars": 0})
        stat["chars"] += len(item.text)
        stat["intervals"] += 1
    for item in ref_intervals:
        if any(
            other.tier != item.tier
            and min(item.end_ms, other.end_ms) > max(item.start_ms, other.start_ms)
            for other in ref_intervals
        ):
            tier_stats[item.tier]["overlap_chars"] += len(item.text)

    return {
        "session": session_dir.name,
        "textgrid": str(textgrid_path),
        "skip_start_ms": skip_start_ms,
        "reference": {
            "chars": len(ref),
            "intervals": len(ref_intervals),
            "ms": raw_cover.ref_ms,
            # 口癖/语气词单独计量：ASR 普遍不转写这些，会被算成 Del
            "filler_chars": sum(1 for ch in ref if ch in "呃嗯额哦"),
        },
        "utterances": len(utterances),
        "turns": len(obj.get("turns") or []),
        "versions": {name: metrics.to_dict() for name, metrics in versions.items()},
        "attribution": attribution,
        "coverage": {
            "vad_uncovered_chars": round(raw_cover.uncovered_chars, 1),
            "vad_uncovered_ratio": (
                round(raw_cover.uncovered_chars / raw_cover.ref_chars, 4)
                if raw_cover.ref_chars else 0.0
            ),
            "vad_uncovered_ms": raw_cover.uncovered_ms,
            "padded_uncovered_chars": round(padded_cover.uncovered_chars, 1),
            "padded_uncovered_ratio": (
                round(padded_cover.uncovered_chars / padded_cover.ref_chars, 4)
                if padded_cover.ref_chars else 0.0
            ),
            "padded_uncovered_ms": padded_cover.uncovered_ms,
            "cross_speaker_overlap_chars": overlap_chars,
            "cross_speaker_overlap_ratio": (
                round(overlap_chars / len(ref), 4) if ref else 0.0
            ),
            "cross_speaker_overlap_intervals": overlap_intervals,
        },
        "tiers": tier_stats,
    }


def _print_report(result: Dict) -> None:
    ref = result["reference"]
    print(f"session={result['session']}  ref={ref['chars']}  "
          f"ref_intervals={ref['intervals']}  utts={result['utterances']}  "
          f"turns={result['turns']}")
    if result["skip_start_ms"]:
        print(f"  （已跳过前 {result['skip_start_ms'] / 1000:.1f}s）")
    if ref.get("filler_chars"):
        print(f"  参考口癖字符（呃/嗯/额/哦）: {ref['filler_chars']}"
              f" ({ref['filler_chars'] / ref['chars']:.2%})")

    print("  version    CER       Sub       Del       Ins     "
          "S       D       I    hyp")
    for name, item in result["versions"].items():
        print(f"  {name:9s} {item['cer']:8.2%} {item['sub_rate']:8.2%} "
              f"{item['del_rate']:8.2%} {item['ins_rate']:8.2%} "
              f"{item['sub']:6d}  {item['del']:6d}  {item['ins']:6d}  "
              f"{item['hyp_len']:6d}")
        if "legacy_lcs_cer" in item:
            print(f"  {'':9s} legacy difflib/LCS CER={item['legacy_lcs_cer']:.2%}"
                  f"（仅历史对照，不得与新口径混用）")

    attribution = result.get("attribution") or {}
    if attribution:
        print("  错误归因（turns 口径，按参考区间精确归因）：")
        for dimension, title in (
            ("coverage", "VAD 覆盖"),
            ("overlap", "跨说话人"),
            ("tier", "说话人"),
        ):
            for key, item in attribution.get(dimension, {}).items():
                print(f"    [{title}] {key:12s} ref={item['ref_chars']:5d} "
                      f"S={item['sub']:5d} D={item['del']:5d} I={item['ins']:5d} "
                      f"CER={item['cer']:7.2%} "
                      f"(Sub {item['sub_rate']:.2%} / Del {item['del_rate']:.2%} / "
                      f"Ins {item['ins_rate']:.2%})")

    cov = result["coverage"]
    print("  覆盖/重叠：")
    print(f"    VAD 原始未覆盖参考字符: {cov['vad_uncovered_chars']:.0f}"
          f" / {ref['chars']} ({cov['vad_uncovered_ratio']:.2%})，"
          f"时长 {cov['vad_uncovered_ms'] / 1000:.1f}s")
    print(f"    加 80ms pad 后未覆盖:   {cov['padded_uncovered_chars']:.0f}"
          f" ({cov['padded_uncovered_ratio']:.2%})，"
          f"时长 {cov['padded_uncovered_ms'] / 1000:.1f}s")
    print(f"    跨说话人重叠参考字符:   {cov['cross_speaker_overlap_chars']}"
          f" ({cov['cross_speaker_overlap_ratio']:.2%})，"
          f"{cov['cross_speaker_overlap_intervals']} 个区间")
    print("  分说话人：")
    for tier, stat in sorted(result["tiers"].items()):
        print(f"    {tier}: chars={stat['chars']} intervals={stat['intervals']} "
              f"overlap_chars={stat['overlap_chars']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CER（Levenshtein + Sub/Del/Ins）")
    parser.add_argument("session_dir")
    parser.add_argument("textgrid")
    parser.add_argument("--skip-start-ms", type=int, default=0,
                        help="跳过开头 N 毫秒（如 23000 排除报号段）")
    parser.add_argument("--legacy", action="store_true",
                        help="额外打印旧 difflib/LCS 口径，仅作历史对照")
    parser.add_argument("--pad-ms", type=int, default=80,
                        help="离线二遍两侧 pad（覆盖率诊断用，默认 80ms）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    result = evaluate(
        args.session_dir,
        args.textgrid,
        skip_start_ms=args.skip_start_ms,
        legacy=args.legacy,
        pad_ms=args.pad_ms,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
