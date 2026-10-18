# -*- coding: utf-8 -*-
"""缓存 20 场的真实 FSMN VAD 区间，并用它证伪/证实 §4.1b 的解释。

背景：端到端跑 `min_win=2` 时，已出的 7 行与 `min_win=4` **逐位相同**，
包括离线差距最大的 `L_R003S01C02`。我在发现记录 §4.1b 里写的解释是
"离线那 3.5 点来自 RTTM 门窗集里的 1–2 窗 sliver，而 VAD 门窗集更小
（418 vs 562），这些 sliver 大多不在其中"。

**这个解释当时是推的，不是量的。** 本脚本把它量出来。

只加载 VAD（不加载 CAM++，省内存、不与正在跑的 e2e 抢 GPU）。
输出：
  1. `cache/vad/<meeting>.json` —— 真实 VAD 区间（秒），供后续零推理探针复用
  2. VAD 门 vs RTTM 门的候选表对照（sliver 是否存在）

    python scripts/evaluation/cache_vad_intervals.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402

from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, gate_windows, overlap, rttm_speech)

ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
VAD_CACHE = CACHE / "vad"
# 与 web_server.py:29 一致：HERE 是 streaming_asr/，其 dirname 才是项目根
VAD_LOCAL = ROOT / "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch" / "snapshots" / "master"
WIN, SECONDS = 2.0, 600.0


def load_vad_only():
    """只加载 VAD 模型。不走 web_server.get_models（那会连 CAM++ 一起加载）。"""
    if not VAD_LOCAL.is_dir():
        raise SystemExit("VAD 模型目录不存在：%s" % VAD_LOCAL)
    from funasr import AutoModel
    return AutoModel(model=str(VAD_LOCAL), disable_update=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--force", action="store_true", help="忽略已有缓存重算")
    args = ap.parse_args()

    VAD_CACHE.mkdir(parents=True, exist_ok=True)
    meetings = sorted(set(split_meetings(ROOT_DATA, "dev")
                          + split_meetings(ROOT_DATA, "holdout")))
    print(f"  目标 {len(meetings)} 场，窗口 {WIN}s，VAD={VAD_LOCAL.name}", flush=True)

    vad = None
    for index, meeting in enumerate(meetings, 1):
        target = VAD_CACHE / f"{meeting}.json"
        if target.exists() and not args.force:
            print(f"  [{index}/{len(meetings)}] {meeting:<16} 已缓存，跳过", flush=True)
            continue
        if vad is None:
            print("  加载 VAD 模型…", flush=True)
            vad = load_vad_only()
        wav = str(ROOT_DATA / "wav" / f"{meeting}.flac")
        intervals = [(a, min(b, args.seconds))
                     for a, b in v11._vad_segs(vad, wav) if a < args.seconds]
        mean_len = (sum(b - a for a, b in intervals) / len(intervals)) if intervals else 0.0
        target.write_text(json.dumps({
            "meeting": meeting, "seconds": args.seconds,
            "n_intervals": len(intervals),
            "speech_seconds": round(sum(b - a for a, b in intervals), 3),
            "mean_interval": round(mean_len, 3),
            "intervals": [[round(a, 3), round(b, 3)] for a, b in intervals],
        }, ensure_ascii=False), encoding="utf-8")
        print(f"  [{index}/{len(meetings)}] {meeting:<16} {len(intervals):>4} 段  "
              f"语音 {sum(b - a for a, b in intervals):7.1f}s  均长 {mean_len:5.2f}s", flush=True)

    # ---- 用缓存的 VAD 区间证伪 §4.1b ----
    print("\n" + "=" * 74)
    print("§4.1b 检验：VAD 门下候选表里还有没有 1–2 窗的 sliver")
    print("=" * 74)
    print(f"{'会议':<16}{'RTTM窗':>7}{'VAD窗':>7}  "
          f"{'RTTM门 k=3 sizes':<22}{'VAD门 k=3 sizes':<22}")
    sliver_rttm = sliver_vad = 0
    for meeting in meetings:
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, WIN / 2)
        vad_data = json.loads((VAD_CACHE / f"{meeting}.json").read_text(encoding="utf-8"))
        vad_intervals = [(a, b) for a, b in vad_data["intervals"]]

        e_rttm, _ = gate_windows(emb_all, times_all,
                                 rttm_speech(reference, seconds=args.seconds),
                                 WIN, v11.MIN_VAD_FRACTION)
        e_vad, _ = gate_windows(emb_all, times_all, vad_intervals,
                                WIN, v11.MIN_VAD_FRACTION)
        if len(e_rttm) < 3 or len(e_vad) < 3:
            continue

        def sizes3(E):
            from scipy.cluster.hierarchy import fcluster, linkage
            if len(E) < 3:
                return []
            tree = linkage(E, method="average", metric="cosine")
            labels = v11._relabel_by_first_seen(fcluster(tree, t=3, criterion="maxclust"))
            return sorted([int((labels == l).sum()) for l in range(3)], reverse=True)

        s_rttm, s_vad = sizes3(e_rttm), sizes3(e_vad)
        has_r = bool(s_rttm) and min(s_rttm) < 4
        has_v = bool(s_vad) and min(s_vad) < 4
        sliver_rttm += has_r
        sliver_vad += has_v
        if has_r and not has_v:
            mark = "   <-- VAD 门吃掉了 sliver"
        elif has_v and not has_r:
            mark = "   <-- VAD 门反而引入了 sliver"
        else:
            mark = ""
        print(f"{meeting:<16}{len(e_rttm):>7}{len(e_vad):>7}  "
              f"{str(s_rttm):<22}{str(s_vad):<22}{mark}")

    print(f"\n  k=3 时存在 <4 窗小簇的场次：RTTM 门 {sliver_rttm}/20，VAD 门 {sliver_vad}/20")
    if sliver_vad < sliver_rttm:
        print("  -> §4.1b 的解释成立：VAD 门确实吃掉了一部分 sliver")
    else:
        print("  -> §4.1b 的解释**不成立**：sliver 在两种门下同样存在，")
        print("     那么 min_win=2 在真实路径上无效果必有别的机制，需重查")


if __name__ == "__main__":
    main()
