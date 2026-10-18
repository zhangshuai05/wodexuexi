# -*- coding: utf-8 -*-
"""把"上限 4->10"从"放宽合格判定"里拆出来，看 +6.8 到底需要几个参数。

`probe_selector_locks_split3.py` 的 `relax` 开关同时管两件事：
  * 上限 4 -> 10（`MAX_SPEAKERS` 封顶）
  * 放宽合格判定（尺寸门/份额门只对 >=2 窗的簇生效）
两者从未分开测过。已知的三格：cap10 单独 +0.8、cap4+core +0.0、cap10+core +6.8，
看起来是**交互效应**，但没有"cap10 + core、不放宽合格判定"这一格，
就无法判断要提的改动是一条参数还是两条。

本脚本补全这张 2x2x2 表（RTTM 门窗口集，零推理）。复用 `split3` 的
`build`/`select`（它们的签名本来就带 cap 与 relax 两个独立参数）。

单独成文件的原因：`probe_selector_locks_vad.py` 会 import `split3.VARIANTS`，
改那个常量的元组长度会打断正在跑的批处理链。这里不动它。

    python scripts/evaluation/probe_selector_locks_cap.py --split dev
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_selector_locks_der import predicted_rttm  # noqa: E402
from probe_selector_locks_split3 import build, select  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0

# (名字, 上限, 放宽合格判定, 孤窗容忍轮廓系数)
# 不加 `drop`：拆分数门那一路已证明是空操作（split3 的两行永远相同）。
VARIANTS = [
    ("现状 cap4", 4, False, False),
    ("cap10", 10, False, False),
    ("cap4+core", 4, False, True),
    ("cap10+core", 10, False, True),
    ("cap4+放宽+core", 4, True, True),
    ("cap10+放宽+core", 10, True, True),
    ("cap10+放宽", 10, True, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    args = ap.parse_args()

    rows = []
    print(f"{'会议':<16}{'有效K':>6} | "
          + " | ".join(f"{name:>16}" for name, _, _, _ in VARIANTS))
    for meeting in split_meetings(ROOT_DATA, args.split):
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < SECONDS]
        reference = [{"start": s["start"], "end": min(s["end"], SECONDS),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=SECONDS), WIN, 0.5)
        duration = Counter()
        for s in reference:
            duration[s["speaker"]] += s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sorted(k for k, v in duration.items() if v / grand >= 0.05)
        if len(emb) < 2:
            continue
        win = np.full(len(times), WIN)
        cells = []
        for name, cap, relax, core in VARIANTS:
            chosen = select(build(emb, cap, relax, core), False)
            named = map_by_frames(reference, times, chosen["labels"], win)
            accuracy = frame_accuracy(reference, times, named, win)["accuracy"]
            value = der(reference, predicted_rttm(times, chosen["labels"], WIN * 0.25),
                        collar=0.25)["der"]
            rows.append({"meeting": meeting, "rule": name, "k": chosen["k"],
                         "accuracy": accuracy, "der": value,
                         "sizes": chosen["sizes"], "eff_k": len(effective)})
            cells.append(f"k={chosen['k']}({accuracy * 100:4.1f}/{value * 100:4.1f})")
        print(f"{meeting:<16}{len(effective):>6} | "
              + " | ".join(f"{c:>16}" for c in cells), flush=True)

    print(f"\n{'变体':<18}{'帧级均值':>9}{'DER均值':>9}{'最差帧级':>9}{'DER最差':>9}{'高估k':>7}")
    base = None
    for name, _, _, _ in VARIANTS:
        part = [r for r in rows if r["rule"] == name]
        if not part:
            continue
        accuracy = np.array([r["accuracy"] for r in part])
        value = np.array([r["der"] for r in part])
        base = accuracy.copy() if base is None else base
        print(f"{name:<18}{accuracy.mean() * 100:>8.1f}%{value.mean() * 100:>8.1f}%"
              f"{accuracy.min() * 100:>8.1f}%{value.max() * 100:>8.1f}%"
              f"{sum(1 for r in part if r['k'] > r['eff_k']):>7}")

    output = ROOT / "artifacts" / f"v13_selector_locks_cap_{args.split}.json"
    output.write_text(json.dumps({"schema": "v13-selector-locks-cap-v1",
                                  "split": args.split, "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
