# -*- coding: utf-8 -*-
"""[已被 probe_e2e_unlocked2.py 取代，保留仅作历史对照——不要引用本文件的数字]

**为什么取代**：本文件内联了一份选 K 规则（下方 `UNLOCK`，`min_win` 用的是继承值 4），
而 `probe_e2e_unlocked2.py` 复用 `unlocked_selector.py` 的**唯一实现**
（`CAP=10, MIN_WIN=2`）。两份内联副本会漂移：本文件在 VAD 门下是 18.8%，
`*2` 在 VAD 门下是 14.8%，差 4.0 点全部来自 `M_R003S05C01`
（`min_win=4` 被一个 3 窗小簇卡住 k=6）。

要跑真实路径请用：
    python -u -B scripts/evaluation/probe_e2e_unlocked2.py --split dev --min-win 2
    python -u -B scripts/evaluation/probe_e2e_unlocked2.py --split dev --mode largest_eligible

本文件原本的说明（历史）：

离线探针（`probe_selector_locks*`）只在 2.0 秒主窗上替换了选 K 规则，而生产路径
在选完 K 之后还要走两级细化（0.6s/0.15s 密窗重判边界）。收益能不能穿过细化、
有没有被细化吃掉，只有跑 `diarize()` 才知道。

不改动任何冻结代码：用上下文管理器在调用期间替换 `v11._candidate_clusters` 与
`v11._select_candidate`（v12 内部正是调这两个）。退出时还原。

同时补齐 holdout 的 miss/FA/conf 分解（批处理那轮只存了 der 标量）。

    python scripts/evaluation/probe_e2e_unlocked.py --split dev
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_selector_locks_split3 import build, select  # noqa: E402

ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
SECONDS = 600.0
UNLOCK = (10, True, True)   # cap10 + 放宽合格判定 + 孤窗容忍轮廓系数（探针里 +6.8/-6.6 那一组）


@contextlib.contextmanager
def unlocked_selector():
    """临时替换 v11 的候选构造与选择函数，退出时原样还原。"""
    original_candidates = v11._candidate_clusters
    original_select = v11._select_candidate

    def patched_candidates(E, max_speakers=None):
        # 上限由 UNLOCK[0] 决定，不再受 def 时绑定的 MAX_SPEAKERS=4 影响。
        return build(E, UNLOCK[0], UNLOCK[1], UNLOCK[2]), None

    def patched_select(candidates, forced_k=None):
        if forced_k is not None:
            forced = next((c for c in candidates if c["k"] == forced_k), None)
            if forced is not None:
                return forced
        return select(candidates, True)

    v11._candidate_clusters = patched_candidates
    v11._select_candidate = patched_select
    try:
        yield
    finally:
        v11._candidate_clusters = original_candidates
        v11._select_candidate = original_select


def score(reference, result):
    predicted = [{"start": float(s["start"]), "end": float(s["end"]),
                  "speaker": f"v_{int(s['spk'])}"} for s in result["segments"]]
    detail = der(reference, predicted, collar=0.25)
    level = ("refined" if result["refined"] else
             "rescued" if result["rescued"] else
             "2s-only" if not result["insufficient"] else "insufficient")
    return detail, level, len(result["segments"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    from web_server import get_models

    models = get_models(lambda m: None, need_stream=False, need_offline=False)

    rows = []
    print(f"{'会议':<16}{'真K':>5} | {'现状 k/DER/miss/FA/conf':>34} | "
          f"{'解锁 k/DER/miss/FA/conf':>34} | {'ΔDER':>7}")
    for meeting in split_meetings(ROOT_DATA, args.split):
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        duration = {}
        for s in reference:
            duration[s["speaker"]] = duration.get(s["speaker"], 0.0) + s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sum(1 for v in duration.values() if v / grand >= 0.05)

        wav = str(ROOT_DATA / "wav" / f"{meeting}.flac")
        base = v12.diarize(wav, models, max_seconds=args.seconds)
        with unlocked_selector():
            fixed = v12.diarize(wav, models, max_seconds=args.seconds)

        base_detail, base_level, base_segments = score(reference, base)
        fix_detail, fix_level, fix_segments = score(reference, fixed)
        delta = base_detail["der"] - fix_detail["der"]

        def cell(detail, k, level, segments):
            return (f"k={k:<2}{detail['der'] * 100:5.1f}% "
                    f"{detail['miss_rate'] * 100:4.1f}/{detail['false_alarm_rate'] * 100:4.1f}/"
                    f"{detail['confusion_rate'] * 100:4.1f}")

        rows.append({"meeting": meeting, "eff_k": effective,
                     "base": {**base_detail, "k": base["k"], "level": base_level,
                              "n_segments": base_segments},
                     "fixed": {**fix_detail, "k": fixed["k"], "level": fix_level,
                               "n_segments": fix_segments}})
        print(f"{meeting:<16}{effective:>5} | {cell(base_detail, base['k'], base_level, base_segments):>34} | "
              f"{cell(fix_detail, fixed['k'], fix_level, fix_segments):>34} | "
              f"{'-' if delta == 0 else f'{delta * 100:+6.1f}'}", flush=True)

    def mean(arm, key):
        values = [r[arm][key] for r in rows]
        return float(np.mean(values))

    print(f"\n{'':<16}{'':>5} | {'DER':>8}{'miss':>8}{'FA':>8}{'conf':>8} | "
          f"{'DER':>8}{'miss':>8}{'FA':>8}{'conf':>8}")
    for key in ("der", "miss_rate", "false_alarm_rate", "confusion_rate"):
        label = {"der": "均值"}.get(key, "")
        print(f"  {label or '':<14}  | {mean('base', key) * 100:7.1f}%{'':>4}"
              f" | {mean('fixed', key) * 100:7.1f}%")

    print(f"\n  DER 均值 {mean('base', 'der') * 100:.1f}% -> {mean('fixed', 'der') * 100:.1f}%  "
          f"（{args.split}，{len(rows)} 场，真实路径含两级细化）")
    deltas = np.array([r["base"]["der"] - r["fixed"]["der"] for r in rows])
    print(f"  逐场 ΔDER: 改善 {int((deltas > 0.001).sum())} 场 / 恶化 "
          f"{int((deltas < -0.001).sum())} 场 / 不变 {int((abs(deltas) <= 0.001).sum())} 场")
    print(f"  选 K 变化: " + ", ".join(
        f"{r['meeting']} {r['base']['k']}->{r['fixed']['k']}"
        for r in rows if r["base"]["k"] != r["fixed"]["k"]) or "  选 K 变化: 无")

    output = ROOT / "artifacts" / f"v13_e2e_unlocked_{args.split}.json"
    output.write_text(json.dumps({"schema": "v13-e2e-unlocked-v1", "split": args.split,
                                  "rows": rows}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
