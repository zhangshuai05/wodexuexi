# -*- coding: utf-8 -*-
"""选 K 解锁在**真实 VAD 门**下是否成立。

`probe_selector_locks_split3.py` 在 RTTM 门窗口集上给出：
  拆A+B(留分数门)  帧级 +6.8 点 / DER −6.6 点（dev），holdout +8.4 / −8.5
但门不同，窗集就不同——同一场 `L_R003S01C02` RTTM 门是 562 窗、k=2 的簇是
`[438,124]`（合格），VAD 门是 418 窗、`[421,1]`（一个孤窗废掉整张表）。
所以"拆锁"的收益必须在**真实门**上重测，否则是在错的窗口集上证明了结论。

不需要重新推理：缓存的窗嵌入按 VAD 区间几何过滤即可（与
`probe_gap_decomp.py --gate vad` 同一支），只有 VAD 模型要加载。

    python scripts/evaluation/probe_selector_locks_vad.py --split dev
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

import claude_a2_v11 as v11  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames, window_reference)
from probe_selector_locks_der import predicted_rttm  # noqa: E402
from probe_selector_locks_split3 import VARIANTS, build, select  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    from web_server import get_models

    models = get_models(lambda m: None, need_stream=False, need_offline=False)

    rows = []
    header = " | ".join(f"{name:>18}" for name, _, _, _ in VARIANTS)
    print(f"{'会议':<16}{'有效K':>6}{'窗':>5} | {header}")
    for meeting in split_meetings(ROOT_DATA, args.split):
        wav = ROOT_DATA / "wav" / f"{meeting}.flac"
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
        # 真实门：项目的 FSMN VAD
        intervals = [(a, min(b, args.seconds))
                     for a, b in v11._vad_segs(models["vad"], str(wav))
                     if a < args.seconds]
        emb, times = gate_windows(emb_all, times_all, intervals, WIN,
                                  v11.MIN_VAD_FRACTION)

        duration = Counter()
        for s in reference:
            duration[s["speaker"]] += s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sorted(k for k, v in duration.items() if v / grand >= 0.05)
        if len(emb) < 2:
            print(f"{meeting:<16}  窗不足，跳过")
            continue

        win = np.full(len(times), WIN)
        cells = []
        for name, relax, core, drop in VARIANTS:
            chosen = select(build(emb, 10 if relax else 4, relax, core), drop)
            named = map_by_frames(reference, times, chosen["labels"], win)
            accuracy = frame_accuracy(reference, times, named, win)["accuracy"]
            value = der(reference, predicted_rttm(times, chosen["labels"], WIN * 0.25),
                        collar=0.25)["der"]
            rows.append({"meeting": meeting, "rule": name, "k": chosen["k"],
                         "accuracy": accuracy, "der": value,
                         "sizes": chosen["sizes"], "eff_k": len(effective),
                         "n_windows": int(len(emb)), "gate": "vad"})
            cells.append(f"k={chosen['k']}({accuracy * 100:4.1f}/{value * 100:4.1f})")
        print(f"{meeting:<16}{len(effective):>6}{len(emb):>5} | "
              + " | ".join(f"{c:>18}" for c in cells), flush=True)

    print(f"\n{'变体':<18}{'帧级均值':>9}{'DER均值':>9}{'最差帧级':>9}{'高估k':>7}")
    for name, _, _, _ in VARIANTS:
        part = [r for r in rows if r["rule"] == name]
        if not part:
            continue
        print(f"{name:<18}{np.mean([r['accuracy'] for r in part]) * 100:>8.1f}%"
              f"{np.mean([r['der'] for r in part]) * 100:>8.1f}%"
              f"{np.min([r['accuracy'] for r in part]) * 100:>8.1f}%"
              f"{sum(1 for r in part if r['k'] > r['eff_k']):>7}")

    output = ROOT / "artifacts" / f"v13_selector_locks_vad_{args.split}.json"
    output.write_text(json.dumps({"schema": "v13-selector-locks-vad-v1",
                                  "split": args.split, "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
