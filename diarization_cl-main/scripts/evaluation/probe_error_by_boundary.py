# -*- coding: utf-8 -*-
"""错误在时间轴上的分布：A/B 两腿的帧级错误是"边界量化"还是"真的认错人"。

上一轮我把 `A帧=85.0%` 归因成"2.0s 窗覆盖稀疏"，那是错的：hop=1.0/window=2.0
的窗格逐帧铺下来（后窗覆盖前窗）是**满覆盖**。所以 85% 要有别的解释，而唯一
有信息量的切法是**错误离参考边界有多远**：

  * 错误集中在参考段边界 ±`--near` 之内 -> 2.0s 粒度本身的量化代价，
    换嵌入救不了，该做的是细化边界（v12 的 0.6s 腿就是干这个的）。
  * 错误均匀铺在段内部             -> 嵌入真的把这个人认成了别人，
    换嵌入 / 加 PLDA 才是对症的药。

A 腿（参考标签）与 B 腿（AHC@真值K）用**同一个窗格、同一套铺法**，唯一差别是
标签来源，所以两腿在段内部的错误率之差 = **纯粹的聚类损失**，与边界无关。

另外两件事顺手量掉：
  * **覆盖率**：参考有语音的帧里，有多少帧根本没有窗覆盖（预测为空）。
    这直接证伪/证实"覆盖稀疏"那个说法。
  * **加 collar 后的值**：`der.py` 带 0.25s collar，本脚本的帧级尺子不带。
    报一列 collar 内/外的错误率，把 `winOR=4.51%` 与 `A帧=85%` 的缺口对上。

零推理：全部走 `w2_h1.npz` 缓存。

    python scripts/evaluation/probe_error_by_boundary.py --split dev --seconds 600
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

FRAME = 0.01


def frame_grid(reference):
    """参考逐帧标签 + 每帧到最近参考边界的距离（秒）。"""
    end = max(s["end"] for s in reference)
    n = int(end / FRAME) + 1
    ref = np.full(n, "", dtype=object)
    for s in reference:
        a = max(0, int(round(s["start"] / FRAME)))
        b = min(n, int(round(s["end"] / FRAME)))
        ref[a:b] = s["speaker"]
    distance = np.full(n, np.inf)
    for s in reference:
        for boundary in (s["start"], s["end"]):
            centre = int(round(boundary / FRAME))
            index = np.arange(n)
            distance = np.minimum(distance, np.abs(index - centre) * FRAME)
    return ref, distance


def tile(starts, labels, durations, n):
    """把每段标签铺到帧上（后写的覆盖先写的，与主探针同法）。"""
    pred = np.full(n, "", dtype=object)
    for start, name, duration in zip(starts, labels, durations):
        if not name:
            continue
        a = max(0, int(round(start / FRAME)))
        b = min(n, int(round((start + duration) / FRAME)))
        if b > a:
            pred[a:b] = name
    return pred


def profile(ref, pred, distance, near, collar):
    """按"离边界多远"和"是否在 collar 内"分组报错误率。"""
    speech = ref != ""
    total = int(speech.sum())
    if not total:
        return None
    error = speech & (pred != ref)
    covered = (pred != "") & speech

    def rate(mask):
        hits = int(mask.sum())
        return (round(float(error[mask].mean()), 4) if hits else None, hits)

    interior = speech & (distance > near)
    near_mask = speech & (distance <= near)
    inside_collar = speech & (distance <= collar)
    outside_collar = speech & (distance > collar)
    out = {"frames": total,
           "coverage": round(float(covered.sum()) / total, 4),
           "err_all": rate(speech)}
    for name, mask in (("interior", interior), ("near", near_mask),
                       ("collar_in", inside_collar), ("collar_out", outside_collar)):
        value, frames = rate(mask)
        out[name] = {"err": value, "frames": frames}
    if out["collar_out"]["frames"]:
        out["der_equiv"] = out["collar_out"]["err"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--meeting")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=v12.WIN_S)
    ap.add_argument("--hop", type=float, default=v12.HOP_S)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--near", type=float, default=0.5, help="近边界带宽度（秒）")
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(args.root)
    meetings = [args.meeting] if args.meeting else split_meetings(root, args.split)

    rows = []
    for meeting in meetings:
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        seconds = args.seconds if args.seconds > 0 else max(s["end"] for s in reference)
        reference = [{"start": s["start"], "end": min(s["end"], seconds),
                      "speaker": s["speaker"]} for s in reference if s["start"] < seconds]
        by_speaker = Counter()
        for s in reference:
            by_speaker[s["speaker"]] += s["end"] - s["start"]
        grand = sum(by_speaker.values()) or 1.0
        effective = sorted(n for n, v in by_speaker.items() if v / grand >= args.share)

        emb_all, times_all = cached_windows(args.cache, meeting, args.window, args.hop)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=seconds),
                                  args.window, v11.MIN_VAD_FRACTION)
        truth = window_reference(reference, times, args.window)
        win = np.full(len(times), args.window)
        ref, distance = frame_grid(reference)
        n = len(ref)

        row = {"meeting": meeting, "true_k": len(effective),
               "n_windows": int(len(emb)), "seconds": seconds}

        # 臂 A：参考标签（完美身份）
        row["A"] = profile(ref, tile(times, truth, win, n), distance,
                           args.near, args.collar)

        # 臂 B：AHC@真值K，簇号按帧共现映射到人
        if len(emb) >= 2 and len(effective) >= 2:
            k = min(len(effective), len(emb))
            clusters = cluster_labels(emb, k)
            named = map_by_frames(reference, times, clusters, win)
            row["B"] = profile(ref, tile(times, named, win, n), distance,
                               args.near, args.collar)
            row["B"]["k"] = k
            row["B"]["n_clusters"] = int(len(set(clusters.tolist())))

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def cell(arm, key, field="err"):
        value = ((row.get(arm) or {}).get(key) or {}).get(field)
        return f"{value * 100:6.1f}%" if value is not None else "     -"

    print(f"\n{'会议':<16}{'覆盖':>7}"
          f"{'A内部':>8}{'A近界':>8}{'B内部':>8}{'B近界':>8}{'A-B内部':>9}")
    for row in rows:
        cov = (row.get("A") or {}).get("coverage")
        a = (row.get("A") or {}).get("interior", {}).get("err")
        b = (row.get("B") or {}).get("interior", {}).get("err")
        delta = f"{(b - a) * 100:8.1f}" if (a is not None and b is not None) else "       -"
        print(f"{row['meeting']:<16}"
              f"{(f'{cov * 100:6.1f}%' if cov is not None else '     -'):>7}"
              f"{cell('A', 'interior'):>8}{cell('A', 'near'):>8}"
              f"{cell('B', 'interior'):>8}{cell('B', 'near'):>8}{delta:>9}")

    def mean_pair(arm, key):
        values = [((r.get(arm) or {}).get(key) or {}).get("err") for r in rows]
        values = [v for v in values if v is not None]
        return f"{np.mean(values) * 100:.1f}%" if values else "-"

    print(f"\n平均：A 内部 {mean_pair('A', 'interior')} / 近界 {mean_pair('A', 'near')}"
          f"   B 内部 {mean_pair('B', 'interior')} / 近界 {mean_pair('B', 'near')}")
    cover = [(r.get("A") or {}).get("coverage") for r in rows]
    cover = [c for c in cover if c is not None]
    if cover:
        print(f"参考语音帧的窗覆盖率：中位 {np.median(cover) * 100:.1f}%  "
              f"最低 {min(cover) * 100:.1f}%")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-error-by-boundary-v1", "meetings": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
