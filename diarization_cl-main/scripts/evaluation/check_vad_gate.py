# -*- coding: utf-8 -*-
"""用**已缓存**的 VAD 区间回答两个问题（零推理，不加载任何模型）。

`cache_vad_intervals.py` 落盘 `cache/vad/<meeting>.json`，本脚本只读已有文件，
因此可以在缓存任务还没跑完（20 场）时就用已完成的那些场次做事。

  问题一（§4.1b 证伪）：VAD 门下候选表里还有没有 <4 窗的 sliver？
      我此前把"min_win=2 在真实路径无效"解释成"VAD 门吃掉了 sliver"，那是推的。
      若 VAD 门下 sliver 照样存在，解释就不成立，得另找机制。

  问题二（覆盖）：VAD 丢掉的窗里，是静音还是语音？
      `L_R003S02C02` VAD 语音 287.4s vs 参考语音 524.8s，差 237s。
      若那 237s 里含大量参考语音，则 VAD 在丢**说话人**，
      这层缺陷位于本会话所有测量之下，报告的边界结论要跟着改。

    python scripts/evaluation/check_vad_gate.py
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
WIN, SECONDS = 2.0, 600.0
MIN_VAD = v11.MIN_VAD_FRACTION


def vad_covered(intervals, start, end):
    """[start, end) 内有多少秒被 VAD 判为语音。"""
    return overlap(start, end, intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    cached = sorted(p.stem for p in VAD_CACHE.glob("*.json"))
    if not cached:
        raise SystemExit("没有已缓存的 VAD 区间；先跑 cache_vad_intervals.py")
    print(f"  已缓存 {len(cached)} 场：{', '.join(cached)}\n")

    print("=" * 78)
    print("问题一（§4.1b）：VAD 门下还有没有 <4 窗的 sliver")
    print("=" * 78)
    print(f"{'会议':<16}{'RTTM窗':>7}{'VAD窗':>7}  "
          f"{'RTTM k=3 sizes':<20}{'VAD k=3 sizes':<20}  结论")
    rows_sliver = []
    for meeting in cached:
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, WIN / 2)
        vad_intervals = [(a, b) for a, b in
                         json.loads((VAD_CACHE / f"{meeting}.json").read_text(
                             encoding="utf-8"))["intervals"]]

        e_rttm, _ = gate_windows(emb_all, times_all,
                                 rttm_speech(reference, seconds=args.seconds), WIN, MIN_VAD)
        e_vad, _ = gate_windows(emb_all, times_all, vad_intervals, WIN, MIN_VAD)
        if len(e_rttm) < 3 or len(e_vad) < 3:
            continue

        def sizes3(E):
            from scipy.cluster.hierarchy import fcluster, linkage
            tree = linkage(E, method="average", metric="cosine")
            labels = v11._relabel_by_first_seen(fcluster(tree, t=3, criterion="maxclust"))
            return sorted([int((labels == l).sum()) for l in range(3)], reverse=True)

        s_rttm, s_vad = sizes3(e_rttm), sizes3(e_vad)
        has_r, has_v = min(s_rttm) < 4, min(s_vad) < 4
        verdict = ("sliver 消失" if has_r and not has_v else
                   "两门都有" if has_r and has_v else
                   "两门都无" if not has_r and not has_v else "VAD 门反而有")
        rows_sliver.append((meeting, has_r, has_v))
        print(f"{meeting:<16}{len(e_rttm):>7}{len(e_vad):>7}  "
              f"{str(s_rttm):<20}{str(s_vad):<20}  {verdict}")

    n_r = sum(1 for _, r, _ in rows_sliver if r)
    n_v = sum(1 for _, _, v in rows_sliver if v)
    print(f"\n  k=3 时存在 <4 窗小簇：RTTM 门 {n_r}/{len(rows_sliver)}，"
          f"VAD 门 {n_v}/{len(rows_sliver)}")
    print("  -> §4.1b 的解释" + ("成立" if n_v < n_r else "**不成立**（sliver 同样存在，需另找机制）"))

    print()
    print("=" * 78)
    print("问题二：VAD 丢掉的窗里是静音还是语音")
    print("=" * 78)
    print(f"{'会议':<16}{'参考语音':>9}{'VAD语音':>9}{'差':>8}"
          f"{'丢失时段里的参考语音':>22}{'丢失窗数':>9}")
    lost_speech_pct = []
    for meeting in cached:
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        speech = rttm_speech(reference, seconds=args.seconds)
        vad_intervals = [(a, b) for a, b in
                         json.loads((VAD_CACHE / f"{meeting}.json").read_text(
                             encoding="utf-8"))["intervals"]]
        ref_sec = sum(b - a for a, b in speech)
        vad_sec = sum(b - a for a, b in vad_intervals)

        # 参考语音里，有多少秒**没有被任何 VAD 区间覆盖**
        def uncovered(bounds, intervals):
            total = 0.0
            for a, b in intervals:
                cursor = a
                for s, e in bounds:
                    if e <= cursor or s >= b:
                        continue
                    total += max(0.0, min(b, e) - max(a, s, cursor))
                    cursor = max(cursor, min(b, e))
            return total

        gap_speech = uncovered(speech, vad_intervals)

        # 丢失的窗：RTTM 门留下、VAD 门不要的那些
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, WIN / 2)
        keep_r = np.array([vad_covered(speech, t, t + WIN) >= MIN_VAD * WIN for t in times_all])
        keep_v = np.array([vad_covered(vad_intervals, t, t + WIN) >= MIN_VAD * WIN
                           for t in times_all])
        lost = int((keep_r & ~keep_v).sum())
        pct = gap_speech / ref_sec * 100 if ref_sec else 0.0
        lost_speech_pct.append(pct)
        print(f"{meeting:<16}{ref_sec:>8.1f}s{vad_sec:>8.1f}s{ref_sec - vad_sec:>7.1f}s"
              f"{gap_speech:>20.1f}s{pct:>8.1f}%{lost:>9}")

    if lost_speech_pct:
        print(f"\n  丢失时段里的参考语音占比：中位 {np.median(lost_speech_pct):.1f}%  "
              f"最大 {max(lost_speech_pct):.1f}%")
        print("  判读：占比高 = VAD 在丢说话人（下游结论要跟着改）；")
        print("        占比低 = 丢的是静音/非语音，覆盖不是问题。")


if __name__ == "__main__":
    main()
