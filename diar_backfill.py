# -*- coding: utf-8 -*-
"""单臂回填：调 .zhang/offline_diarizer.backfill_speakers，转成 session 契约。

底层返回 {utt_id: "Speaker_{p+1}"}（p 是 0-based 簇标签）；
契约要   {utt_id: {"spk": p, "spk_label": f"speaker_{p}"}}。
所以 spk = int(label.split("_")[1]) - 1 —— 这个 off-by-one 只在这里换算一次。

独立 CLI（崩溃后重跑）：
    python diar_backfill.py --session output/sessions/s_xxx
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bootstrap

_LABEL_RE = re.compile(r"^Speaker_(\d+)$")


def to_contract(mapping: Dict[str, str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for utt_id, label in mapping.items():
        match = _LABEL_RE.match(str(label))
        if match is None or int(match.group(1)) < 1:
            raise ValueError(f"上游返回了无法识别的说话人标签: {label!r} (utt_id={utt_id})")
        index = int(match.group(1)) - 1
        out[utt_id] = {"spk": index, "spk_label": f"speaker_{index}"}
    return out


def _normalize_row(row: Dict) -> Dict:
    return {
        "utt_id": row["utt_id"],
        "start_ms": row["start_ms"],
        "end_ms": row["end_ms"],
        "raw_text": row.get("raw_text", ""),
    }


def load_utterances(session_dir: Path) -> List[Dict]:
    """从 utterances.jsonl 读句表，按 utt_id 去重（后写覆盖先写）。

    崩溃恢复场景：跳过损坏/不完整的行；若一行有效的都没有，回退到 transcript.json。
    """
    session_dir = Path(session_dir)
    path = session_dir / "utterances.jsonl"
    rows: Dict[str, Dict] = {}
    skipped = 0
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _normalize_row(json.loads(line))
                    rows[row["utt_id"]] = row
                except (json.JSONDecodeError, KeyError, TypeError):
                    skipped += 1
                    continue

    if skipped:
        print(f"[backfill] 跳过 {skipped} 行损坏的 utterances.jsonl", flush=True)

    if rows:
        return list(rows.values())

    transcript = session_dir / "transcript.json"
    if not transcript.exists():
        return []
    obj = json.loads(transcript.read_text(encoding="utf-8"))
    return [_normalize_row(row) for row in obj.get("utterances", [])]


def _real_backfill(wav_path, utterances, models, mode):
    from offline_diarizer import backfill_speakers
    return backfill_speakers(wav_path, utterances, models=models, mode=mode)


def run_backfill(
    session_dir,
    *,
    models: Optional[Dict] = None,
    mode: str = "auto",
    backfill: Optional[Callable] = None,
    force: bool = False,
) -> Dict[str, Dict]:
    bootstrap.install()
    from session_store import SessionStore

    store = SessionStore(Path(session_dir))
    existing = store.read_json("diarization.json")
    if existing is not None and not force:
        store.build_session_json()
        return existing

    if not store.audio_path.exists():
        raise SystemExit(f"缺少音频文件: {store.audio_path}")

    utterances = load_utterances(store.dir)
    if not utterances:
        store.write_diarization({})
        store.build_session_json()
        return {}

    fn = backfill or _real_backfill
    raw = fn(str(store.audio_path), utterances, models, mode)
    mapping = to_contract(raw)

    if store.read_json("transcript.json") is None and utterances:
        end_ms = max(int(u["end_ms"]) for u in utterances)
        store.write_transcript(utterances, duration_ms=end_ms)

    store.write_diarization(mapping)
    store.build_session_json()
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="单臂说话人回填（可独立重跑）")
    parser.add_argument("--session", required=True, help="session 目录，如 output/sessions/s_xxx")
    parser.add_argument("--mode", default="auto", choices=("auto", "class", "vad"))
    parser.add_argument("--force", action="store_true", help="忽略已有 diarization.json")
    args = parser.parse_args(argv)

    bootstrap.install()
    from session_store import SessionStore

    session_dir = Path(args.session)
    if not (session_dir / "session_audio.wav").exists():
        raise SystemExit(f"缺少音频文件: {session_dir / 'session_audio.wav'}")
    store = SessionStore(session_dir)
    if not args.force and store.read_json("diarization.json") is not None:
        print(f"[backfill] 已有 diarization.json，跳过（--force 强制重算）: {session_dir}",
              flush=True)
        return 0

    from web_server import get_models

    started = time.time()
    models = get_models(need_stream=False, need_offline=False)
    load_seconds = time.time() - started
    print(f"[backfill] 模型就绪 {load_seconds:.1f}s", flush=True)

    started = time.time()
    mapping = run_backfill(session_dir, models=models, mode=args.mode, force=args.force)
    elapsed = time.time() - started

    speakers = sorted({v["spk_label"] for v in mapping.values()})
    print(f"[backfill] {len(mapping)} 句 -> {len(speakers)} 人 {speakers} "
          f"({elapsed:.1f}s，不含模型加载)", flush=True)
    print(f"[backfill] 写入 {session_dir / 'diarization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
