# -*- coding: utf-8 -*-
"""v13 Step 0：在 v12 **真实产出路径**上取 K 的候选表与口径事实。

v12 的 K 判据整套继承自 v11，门限读 v11 的常量
（`MIN_CLUSTER_WINDOWS=4` / `MIN_CLUSTER_SHARE=0.015`）。

历史（2026-09-28 前）：`MAX_SPEAKERS` 曾是**死值**——`_candidate_clusters` 的
默认参数在 def 时绑成 `v11.MAX_SPEAKERS=4`，改模块变量无效，故 cap 实际恒为 4。
该 bug 已修（`diarize` 显式传参，见 `977fc6b` 与 `tests/test_k_cap_dispatch.py`）。
`MIN_CLUSTER_WINDOWS` / `MIN_CLUSTER_SHARE` 在 v12 里也定义过（3 / 0.02），
与 v11 的值不同且从未被读到；那两份死常量已删除。

这个探针只读不动：跑 `v12.diarize()`，把它内部真实用到的候选表打出来，
外加判定它走的是哪条出口（细化 / 救援 / 裸长窗）。

    python scripts/evaluation/probe_v12_k.py --meeting L_R003S01C02 --seconds 600
    python scripts/evaluation/probe_v12_k.py --split dev --seconds 600
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from v12_aishell4 import read_rttm, split_meetings  # noqa: E402


def effective_speakers(rttm_path, share=0.05):
    segments = read_rttm(rttm_path)
    seconds = {}
    for item in segments:
        seconds[item["speaker"]] = seconds.get(item["speaker"], 0.0) + item["end"] - item["start"]
    grand = sum(seconds.values())
    names = {name for name, value in seconds.items() if value / grand >= share}
    return names, seconds, grand


def candidate_table(candidates):
    rows = []
    for item in candidates:
        rows.append({
            "k": int(item["k"]),
            "sizes": [int(value) for value in item["sizes"]],
            "eligible": bool(item["eligible"]),
            "score": round(float(item["score"]), 4),
            "objective": round(float(item["objective"]), 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    parser.add_argument("--meeting")
    parser.add_argument("--split", choices=("dev", "holdout", "pilot"))
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root)
    if args.meeting:
        meetings = [args.meeting]
    elif args.split:
        meetings = split_meetings(root, args.split)
    else:
        raise SystemExit("要给 --meeting 或 --split")

    sys.path.insert(0, str(ROOT / "streaming_asr"))
    from web_server import get_models
    models = get_models()

    report = []
    for meeting in meetings:
        wav = root / "wav" / f"{meeting}.flac"
        rttm = root / "TextGrid" / f"{meeting}.rttm"
        effective, seconds, grand = effective_speakers(rttm)
        started = time.perf_counter()
        result = v12.diarize(str(wav), models, max_seconds=args.seconds)
        elapsed = time.perf_counter() - started

        candidates = result.get("candidates") or []
        # 池内实际有几个说话人：只读标签做不到，用有效集与"池里有没有它的窗"分开记
        item = {
            "meeting": meeting,
            "seconds": args.seconds,
            "elapsed": round(elapsed, 1),
            "insufficient": result["insufficient"],
            "n_windows": int(result["n_windows"]),
            "k_selected": int(result["k"]),
            "refined": bool(result["refined"]),
            "rescued": bool(result["rescued"]),
            "window_seconds": result.get("window_seconds"),
            "hop_seconds": result.get("hop_seconds"),
            "embedding_errors": int(result.get("embedding_errors", 0)),
            "reference_speakers_all": len(seconds),
            "reference_speakers_effective": len(effective),
            "effective_names": sorted(effective),
            "speaker_shares": {name: round(seconds[name] / grand, 4)
                               for name in sorted(seconds, key=lambda n: -seconds[n])},
            "n_segments": len(result["segments"]),
            "candidates": candidate_table(candidates),
        }
        report.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-step0-v1", "meetings": report},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
