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
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bootstrap


def to_contract(mapping: Dict[str, str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for utt_id, label in mapping.items():
        index = int(str(label).split("_")[1]) - 1
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

    jsonl 不存在时回退读 transcript.json（崩溃后重跑、或只有 T+0 全量文本的场景）。
    """
    session_dir = Path(session_dir)
    path = session_dir / "utterances.jsonl"
    if path.exists():
        rows: Dict[str, Dict] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[row["utt_id"]] = _normalize_row(row)
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
    from web_server import get_models

    started = time.time()
    models = get_models(need_stream=False, need_offline=False)
    load_seconds = time.time() - started
    print(f"[backfill] 模型就绪 {load_seconds:.1f}s", flush=True)

    started = time.time()
    mapping = run_backfill(Path(args.session), models=models, mode=args.mode, force=args.force)
    elapsed = time.time() - started

    speakers = sorted({v["spk_label"] for v in mapping.values()})
    print(f"[backfill] {len(mapping)} 句 -> {len(speakers)} 人 {speakers} "
          f"({elapsed:.1f}s，不含模型加载)", flush=True)
    print(f"[backfill] 写入 {Path(args.session) / 'diarization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
