# -*- coding: utf-8 -*-
"""Session 目录契约：六文件读写 + 原子落盘 + 最终合并。

文件职责严格隔离（用户 2026-10-17 定稿）：
    session_audio.wav   流式边收边写（由 StreamingSession 负责，本模块不管）
    utterances.jsonl    每个 asr_final append 一行 {utt_id,start_ms,end_ms,raw_text}
    transcript.json     T+0 全量文本，status="transcript"
    polished.json       润色结果 {utt_id:{polished_text,corrected,bypassed,error}}
    diarization.json    回填结果 {utt_id:{spk,spk_label}}
    session.json        幂等合并后的最终态
    events.ndjson       所有流式事件（前端回放用）
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from turns import build_turns, split_utterance_by_sub_turns

SCHEMA_VERSION = 1

UTTERANCE_KEYS = ("utt_id", "start_ms", "end_ms", "raw_text")

# 进程内替换锁：不同 SessionStore 实例可能并发写同一目录（Windows 下 replace 需串行）
_REPLACE_LOCK = threading.Lock()

# 跨实例共享，因为 `run_backfill` 会为同一目录新建第二个 store
_MERGE_LOCK = threading.Lock()

# 进程内追加锁：主线程与润色线程会并发 append，Windows 下需串行化整次写入
_APPEND_LOCK = threading.Lock()


def new_session_dir(root: Path, when: Optional[datetime] = None) -> Path:
    """创建 output/sessions/s_YYYYMMDD_HHMMSS（重名时追加 _1、_2…）。"""
    when = when or datetime.now()
    base = "s_" + when.strftime("%Y%m%d_%H%M%S")
    root = Path(root)
    candidate = root / base
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = root / f"{base}_{suffix}"


def _atomic_write_json(path: Path, obj: Any) -> None:
    # 临时文件名带上进程号与线程号，避免多个 SessionStore 实例共享同一临时路径
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows 下并发 replace 同一目标会抛 WinError 5，进程内串行化替换
    with _REPLACE_LOCK:
        os.replace(tmp, path)


def _append_ndjson(path: Path, obj: Any) -> None:
    # 先序列化再持锁打开写入，避免两线程的写入交错破坏行
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)


def _validated_row(utt: Dict[str, Any]) -> Dict[str, Any]:
    """契约四键必须齐全，且 utt_id/start_ms/end_ms 不得为 None（raw_text 允许 ""）。"""
    missing = [key for key in UTTERANCE_KEYS if key not in utt]
    missing += [key for key in ("utt_id", "start_ms", "end_ms")
                if key in utt and utt[key] is None]
    if missing:
        raise ValueError(f"utterance 缺少字段: {missing}")
    return {key: utt[key] for key in UTTERANCE_KEYS}


class SessionStore:
    """一个 session 目录的读写门面。

    合并锁保证 polished 线程与回填路径并发调用 build_session_json() 时不会互相覆盖。
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        session_id: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or self.dir.name
        self.started_at = started_at or datetime.now().astimezone().isoformat(timespec="seconds")
        self.audio_path = self.dir / "session_audio.wav"
        self._merge_lock = _MERGE_LOCK

    # ---- 写入 ----

    def append_utterance(self, utt: Dict[str, Any]) -> None:
        _append_ndjson(self.dir / "utterances.jsonl", _validated_row(utt))

    def append_event(self, ev: Dict[str, Any]) -> None:
        _append_ndjson(self.dir / "events.ndjson", ev)

    def write_transcript(self, utterances: List[Dict[str, Any]], duration_ms: int) -> Path:
        obj = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "status": "transcript",
            "audio": str(self.audio_path),
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "utterances": [_validated_row(utt) for utt in utterances],
        }
        path = self.dir / "transcript.json"
        _atomic_write_json(path, obj)
        return path

    def write_polished(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "polished.json"
        _atomic_write_json(path, mapping)
        return path

    def write_diarization(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "diarization.json"
        _atomic_write_json(path, mapping)
        return path

    def write_turns_polished(self, mapping: Dict[str, str]) -> Path:
        path = self.dir / "turns_polished.json"
        _atomic_write_json(path, mapping)
        return path

    def write_final_text(self, mapping: Dict[str, Dict]) -> Path:
        path = self.dir / "final_text.json"
        _atomic_write_json(path, mapping)
        return path

    # ---- 读取 ----

    def read_json(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.dir / name
        if not path.exists():
            return None
        with _REPLACE_LOCK:
            return json.loads(path.read_text(encoding="utf-8"))

    # ---- 合并 ----

    def build_session_json(self) -> Dict[str, Any]:
        """读 transcript + 可选的 polished / diarization，幂等合成 session.json。"""
        with self._merge_lock:
            transcript = self.read_json("transcript.json") or {"utterances": []}
            polished = self.read_json("polished.json") or {}
            diarization = self.read_json("diarization.json")
            final_text = self.read_json("final_text.json") or {}

            merged = []
            for utt in transcript.get("utterances", []):
                uid = utt["utt_id"]
                polish = polished.get(uid, {})
                speaker = (diarization or {}).get(uid, {})
                if not isinstance(speaker, dict):
                    speaker = {}
                parent = {
                    **utt,
                    "polished_text": polish.get("polished_text", utt.get("raw_text", "")),
                    "final_text": (final_text.get(uid) or {}).get("text"),
                    "corrected": bool(polish.get("corrected", False)),
                    "bypassed": bool(polish.get("bypassed", False)),
                    "error": polish.get("error"),
                    "spk": speaker.get("spk"),
                    "spk_label": speaker.get("spk_label"),
                    "overlap_spk": None,
                }
                # 句内换人：上游 sub_turns 把长句拆成单人子段；离线二遍会逐子段
                # 重转写（final_text 用子段 id），turns 因此不会被整句投票吞掉。
                for part in split_utterance_by_sub_turns(
                        parent, speaker.get("sub_turns")):
                    sub_final = final_text.get(part["utt_id"])
                    if isinstance(sub_final, dict) and sub_final.get("text") is not None:
                        part["final_text"] = sub_final["text"]
                    merged.append(part)

            turns = build_turns(merged)
            turns_polished = self.read_json("turns_polished.json") or {}
            for turn in turns:
                entry = turns_polished.get(turn["turn_id"])
                if not isinstance(entry, dict):
                    continue
                # 独立重跑回填可能改变 turn 边界；对不上就保留直播稿，宁可少改
                if (entry.get("start_ms") != turn["start_ms"]
                        or entry.get("end_ms") != turn["end_ms"]):
                    continue
                polished_text = entry.get("text")
                if isinstance(polished_text, str) and polished_text:
                    turn["text"] = polished_text

            obj = {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "status": "diarized" if diarization else "transcript",
                "audio": str(self.audio_path),
                "started_at": transcript.get("started_at", self.started_at),
                "ended_at": transcript.get("ended_at"),
                "duration_ms": transcript.get("duration_ms"),
                "utterances": merged,
                # utterances 是声纹/审计粒度；turns 是显示/导出粒度
                "turns": turns,
            }
            _atomic_write_json(self.dir / "session.json", obj)
            return obj
