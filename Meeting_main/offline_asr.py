# -*- coding: utf-8 -*-
"""会后离线二遍转写：按 VAD 句子重转写，产出最终稿。

与直播润色（stream_llm_polish）和会后 LLM 精修（final_polish）分离：
本模块只做 ASR，不改写；失败/空段由调用方回退直播 raw_text。

2026-10-18 两项修正（用户要求）:
1. RMS AGC：0.002 < rms < 0.04 的片段线性放大到 rms≈0.05，peak 限幅 0.95。
   远场弱音段（如 RMS 0.0081）在流式 VAD/ASR 上损失严重，离线二遍先把电平拉齐。
2. 长段拼接：相邻且间隔 <600ms（或有 spk_label 时同说话人且间隔 ≤2s）的短句
   先拼成 4s~15s 的长段，再送入离线 SEACO，恢复长上下文；解码文本按各句时长
   比例回填到 utt_id。CER 只关心拼接后的总文本，因此比例切分不改变 CER。
   注意：旧版上限是 8s（上游 v11 约定）；本版按用户要求放宽到 15s。

另一个限制必须说清楚：AGC 只能挽救“VAD 已切出、但电平过低”的片段，
**救不回 VAD 根本没切出来的语音**。后者只能靠降低 VAD 阈值重跑，
不在本模块职责内。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from turns import clean_cjk_spaces, join_texts

MAX_SEGMENT_MS = 15000   # 用户要求 4s~15s；旧值是 8000（上游 v11）
MIN_GROUP_MS = 4000      # 短句拼段的目标下限；做不到也不强拼
ADJACENT_GAP_MS = 600    # 相邻句间隔小于此值即可拼段
SPEAKER_JOIN_GAP_MS = 2000   # 有说话人标签时，同说话人可跨过的最大间隔
SHORT_GROUP_BRIDGE_GAP_MS = 1500  # 当前段不足 4s 时，允许再放宽一点
PAD_MS = 80              # 两侧各留 80ms，避免切掉字头
MIN_CLIP_MS = 200
LAST_CHUNK_MERGE_MS = 500  # 长段切分后，尾块小于此值则并入前块
_TRAILING_PUNCT = "，。！？、；："

AGC_MIN_RMS = 0.002
AGC_MAX_RMS = 0.04
AGC_TARGET_RMS = 0.05
AGC_PEAK_LIMIT = 0.95

_model = None
_model_lock = threading.Lock()


def _load_asr():
    """延迟加载离线 ASR（Fun-ASR-Nano 优先，缺失时由 get_models 回退 SEACO）。"""
    global _model
    with _model_lock:
        if _model is None:
            from web_server import get_models
            _model = get_models(need_offline=True)["asr"]
        return _model


def _split_ranges(start_ms: int, end_ms: int, max_ms: int) -> List[tuple]:
    total = max(0, end_ms - start_ms)
    if total <= max_ms:
        return [(start_ms, end_ms)]
    count = (total + max_ms - 1) // max_ms
    step = total / count
    return [(int(start_ms + i * step), int(start_ms + (i + 1) * step))
            for i in range(count)]


def _utt_duration(utt: Dict) -> int:
    return max(0, int(utt["end_ms"]) - int(utt["start_ms"]))


def _utt_weight(utt: Dict) -> int:
    """按原句有效字数分配拼接后的识别文本，避免单纯按时长切断词语。"""
    reference = clean_cjk_spaces(utt.get("polished_text") or utt.get("raw_text") or "")
    char_count = len(reference.rstrip(_TRAILING_PUNCT + ",.!? "))
    return char_count * 200 if char_count else max(1, _utt_duration(utt))


def _restore_trailing_punct(piece: str, utt: Dict) -> str:
    """恢复流式原句末尾有、离线识别结果缺失的中文句末标点。"""
    piece = clean_cjk_spaces(piece)
    if not piece or piece[-1] in _TRAILING_PUNCT + ",.!?":
        return piece
    reference = clean_cjk_spaces(utt.get("polished_text") or utt.get("raw_text") or "")
    if reference and reference[-1] in _TRAILING_PUNCT:
        return piece + reference[-1]
    return piece


def _agc(clip: np.ndarray, stats: Optional[Dict] = None) -> np.ndarray:
    """RMS 自动增益：只处理 0.002 < rms < 0.04 的弱音片段。"""
    if clip.size == 0:
        return clip
    mono = clip.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    if not (AGC_MIN_RMS < rms < AGC_MAX_RMS):
        return clip

    gain = AGC_TARGET_RMS / rms
    peak = float(np.max(np.abs(mono)))
    limited = False
    if peak > 0:
        peak_gain = AGC_PEAK_LIMIT / peak
        if gain > peak_gain:
            gain = peak_gain
            limited = True
    if stats is not None:
        stats["agc_clips"] = stats.get("agc_clips", 0) + 1
        if limited:
            stats["agc_peak_limited"] = stats.get("agc_peak_limited", 0) + 1
    return (mono * gain).astype(np.float32)


def _clip(audio: np.ndarray, sr: int, start_ms: int, end_ms: int) -> Optional[np.ndarray]:
    lo = max(0, int((start_ms - PAD_MS) * sr / 1000))
    hi = min(len(audio), int((end_ms + PAD_MS) * sr / 1000))
    if hi - lo < int(MIN_CLIP_MS * sr / 1000):
        return None
    return audio[lo:hi].copy()


def _group_utterances(utterances: Sequence[Dict]) -> List[List[Dict]]:
    """把短句贪心合并成 4s~15s 的组。

    规则：
    - 有 spk_label 且相同、间隔 ≤2s：可拼；
    - 间隔 <600ms（无标签时的主判据）：可拼；
    - 当前组不足 4s 时，允许把间隔放宽到 1.5s 再拼一句（努力凑够 4s）；
    - 组时长超过 15s 就断开；单句本身超过 15s 时由解码阶段再切。
    """
    groups: List[List[Dict]] = []
    current: List[Dict] = []
    for utt in utterances:
        if not current:
            current = [utt]
            continue
        prev = current[-1]
        gap = int(utt["start_ms"]) - int(prev["end_ms"])
        same_spk = (
            utt.get("spk_label") is not None
            and utt.get("spk_label") == prev.get("spk_label")
        )
        # 不同说话人绝不拼段：sub_turns 拆出的子段首尾相接，若只按 gap 判会
        # 把两个人的音频重新拼成一组，等于把句内换人又吞回去。
        no_labels = (utt.get("spk_label") is None
                     and prev.get("spk_label") is None)
        join = ((same_spk and gap <= SPEAKER_JOIN_GAP_MS)
                or (no_labels and gap < ADJACENT_GAP_MS))
        current_duration = sum(_utt_duration(item) for item in current)
        if (not join and current_duration < MIN_GROUP_MS
                and gap < SHORT_GROUP_BRIDGE_GAP_MS
                and (no_labels or same_spk)):
            join = True
        if join and current_duration + _utt_duration(utt) <= MAX_SEGMENT_MS:
            current.append(utt)
        else:
            groups.append(current)
            current = [utt]
    if current:
        groups.append(current)
    return groups


def _split_text(text: str, weights: Sequence[int]) -> List[str]:
    """按权重把整段文本切回各 utt。

    CER 只看拼接后的总文本，因此这里的切点不影响 CER；比例切分只是让
    session.json 的逐句 final_text 不至于全部堆在第一句。
    """
    count = len(weights)
    if count <= 1:
        return [text]
    total = sum(weights) or count
    bounds = [0]
    acc = 0
    for weight in weights[:-1]:
        acc += weight
        bounds.append(int(round(len(text) * acc / total)))
    bounds.append(len(text))
    # 保证单调不减
    for index in range(1, len(bounds)):
        bounds[index] = max(bounds[index], bounds[index - 1])
    return [text[bounds[i]:bounds[i + 1]] for i in range(count)]


def _decode_single(
    audio: np.ndarray,
    sr: int,
    utt: Dict,
    caller: Callable[[str], str],
    tmp_path: str,
    max_segment_ms: int,
    use_agc: bool,
    stats: Dict,
) -> str:
    import soundfile as sf

    parts: List[str] = []
    start_ms, end_ms = int(utt["start_ms"]), int(utt["end_ms"])
    for lo_ms, hi_ms in _split_ranges(start_ms, end_ms, max_segment_ms):
        clip = _clip(audio, sr, lo_ms, hi_ms)
        if clip is None:
            continue
        if use_agc:
            clip = _agc(clip, stats)
        sf.write(tmp_path, clip, sr, subtype="PCM_16")
        text = (caller(tmp_path) or "").strip()
        stats["segments"] = stats.get("segments", 0) + 1
        if text:
            parts.append(text)
    joined = join_texts(parts, insert_comma=False) if parts else ""
    return _restore_trailing_punct(joined, utt)


def _decode_group(
    audio: np.ndarray,
    sr: int,
    group: Sequence[Dict],
    caller: Callable[[str], str],
    tmp_path: str,
    max_segment_ms: int,
    use_agc: bool,
    stats: Dict,
) -> Dict[str, Dict]:
    """解码一个组，返回 {utt_id: {text,start_ms,end_ms,grouped_cont}}。"""
    import soundfile as sf

    if len(group) == 1:
        utt = group[0]
        text = _decode_single(audio, sr, utt, caller, tmp_path,
                              max_segment_ms, use_agc, stats)
        if not text:
            return {}
        return {utt["utt_id"]: {
            "text": text,
            "start_ms": int(utt["start_ms"]),
            "end_ms": int(utt["end_ms"]),
            "grouped_cont": False,
        }}

    members: List[tuple] = []
    for utt in group:
        clip = _clip(audio, sr, int(utt["start_ms"]), int(utt["end_ms"]))
        if clip is None:
            continue
        members.append((utt, _agc(clip, stats) if use_agc else clip))
    if not members:
        return {}
    joined = np.concatenate([clip for _, clip in members])

    max_samples = max(1, int(max_segment_ms * sr / 1000))
    chunks = [joined[i:i + max_samples] for i in range(0, len(joined), max_samples)]
    merge_samples = int(LAST_CHUNK_MERGE_MS * sr / 1000)
    if len(chunks) >= 2 and len(chunks[-1]) < merge_samples:
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()

    parts: List[str] = []
    for chunk in chunks:
        sf.write(tmp_path, chunk, sr, subtype="PCM_16")
        text = clean_cjk_spaces(caller(tmp_path) or "")
        stats["segments"] = stats.get("segments", 0) + 1
        if text:
            parts.append(text)
    text = join_texts(parts, insert_comma=False) if parts else ""
    if not text:
        return {}

    weights = [_utt_weight(utt) for utt, _ in members]
    pieces = _split_text(text, weights)
    out: Dict[str, Dict] = {}
    for index, ((utt, _), piece) in enumerate(zip(members, pieces)):
        if index == len(members) - 1:
            piece = _restore_trailing_punct(piece, utt)
        out[utt["utt_id"]] = {
            "text": piece,
            "start_ms": int(utt["start_ms"]),
            "end_ms": int(utt["end_ms"]),
            "grouped_cont": index > 0,
        }
    return out


def transcribe_utterances(
    audio_path,
    utterances: List[Dict],
    *,
    asr_call: Optional[Callable[[str], str]] = None,
    max_segment_ms: int = MAX_SEGMENT_MS,
    agc: bool = True,
    group_short: bool = True,
    hotwords: Optional[Union[str, List[str]]] = None,
    on_progress: Optional[Callable[[Dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    on_stats: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, Dict]:
    """返回 {utt_id: {"text", "start_ms", "end_ms", "grouped_cont"}}；失败/空段不出现。

    asr_call 注入时忽略 hotwords（注入函数自行决定行为）。
    """
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = data[:, 0]

    stats: Dict = {
        "utterances": len(utterances),
        "groups": 0,
        "segments": 0,
        "agc_clips": 0,
        "agc_peak_limited": 0,
        "failed_groups": 0,
        "empty_groups": 0,
        "hotword_fallbacks": 0,
    }

    hotword_state = {"enabled": hotwords is not None and asr_call is None}

    if asr_call is not None:
        caller = asr_call
    else:
        model = _load_asr()

        def caller(path: str) -> str:
            if hotword_state["enabled"]:
                try:
                    result = model.generate(input=path, hotword=hotwords)
                except Exception:  # noqa: BLE001 - 模型不支持/参数不兼容时退回无热词
                    hotword_state["enabled"] = False
                    stats["hotword_fallbacks"] += 1
                    result = model.generate(input=path)
            else:
                result = model.generate(input=path)
            return (result[0].get("text") or "") if result else ""

    out: Dict[str, Dict] = {}
    total = len(utterances)
    done = 0
    tmp = os.path.join(tempfile.gettempdir(), f"offline_asr_{os.getpid()}.wav")
    groups = _group_utterances(utterances) if group_short else [[u] for u in utterances]

    for group in groups:
        if should_abort is not None and should_abort():
            break
        stats["groups"] += 1
        try:
            part = _decode_group(audio, sr, group, caller, tmp,
                                 max_segment_ms, agc, stats)
        except Exception:  # noqa: BLE001 - 单组失败保留直播稿
            stats["failed_groups"] += 1
            part = {}
        if part:
            out.update(part)
        else:
            stats["empty_groups"] += 1

        done += len(group)
        if on_progress is not None:
            on_progress({
                "t": "progress", "stage": "final_asr",
                "done": done, "total": total,
                "msg": "最终稿离线转写中…",
            })

    if on_stats is not None:
        on_stats(stats)
    return out


# ---------------------------------------------------------------------------
# 独立重跑（A/B 用）：不重跑流式，只对已有 session_audio.wav 重做离线二遍
# ---------------------------------------------------------------------------
def _load_session_utterances(session_dir: Path) -> List[Dict]:
    session_json = session_dir / "session.json"
    transcript_json = session_dir / "transcript.json"
    if session_json.exists():
        obj = json.loads(session_json.read_text(encoding="utf-8"))
    else:
        obj = json.loads(transcript_json.read_text(encoding="utf-8"))
    return obj.get("utterances") or []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="对已有 session 目录重跑离线二遍 ASR（不重跑流式）")
    parser.add_argument("--session", required=True, help="session 目录")
    parser.add_argument("--audio", default=None,
                        help="默认 <session>/session_audio.wav")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 路径；默认写回 <session>/final_text.json")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="不重建 session.json")
    parser.add_argument("--no-agc", action="store_true")
    parser.add_argument("--no-group", action="store_true",
                        help="关闭长段拼接（每句独立解码）")
    parser.add_argument("--max-segment-ms", type=int, default=MAX_SEGMENT_MS)
    parser.add_argument("--hotwords", default=None,
                        help="会议热词，空格分隔；仅离线模型支持时生效")
    args = parser.parse_args(argv)

    import bootstrap
    bootstrap.install()

    session_dir = Path(args.session)
    audio_path = Path(args.audio) if args.audio else session_dir / "session_audio.wav"
    utterances = _load_session_utterances(session_dir)

    last = {"done": 0}

    def progress(ev: Dict) -> None:
        done, total = int(ev.get("done") or 0), int(ev.get("total") or 0)
        if done == total or done - last["done"] >= 50:
            last["done"] = done
            print(f"[final_asr] {done}/{total}", flush=True)

    result = transcribe_utterances(
        audio_path,
        utterances,
        max_segment_ms=args.max_segment_ms,
        agc=not args.no_agc,
        group_short=not args.no_group,
        hotwords=args.hotwords,
        on_progress=progress,
        on_stats=lambda stats: print(
            json.dumps(stats, ensure_ascii=False), flush=True),
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        from session_store import SessionStore
        store = SessionStore(session_dir)
        store.write_final_text(result)
        if not args.no_rebuild:
            store.build_session_json()

    print(json.dumps({
        "session": str(session_dir),
        "utterances": len(utterances),
        "final_entries": len(result),
        "output": args.output or str(session_dir / "final_text.json"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
