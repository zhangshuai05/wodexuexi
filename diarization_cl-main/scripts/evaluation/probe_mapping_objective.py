# -*- coding: utf-8 -*-
"""口径不一致的**量级**：匈牙利映射按时长 vs 按句计数，主指标差多少。

## 背景

`attribution` 里 `hungarian_map(p, truth, weights)` 传的 `weights` 是**句时长**，
于是匹配目标是"时长最优"；但主指标报的是 `per_sentence`（**按句平均**）。
两者的最优置换**可以不同** —— 一个长句的权重能压过好几个短句。

本脚本不改任何生产代码：`hungarian_map` 的 `weights` 参数就是接缝，
把 `attribution` 内部的权重换成 `[1]*n` 即得"按句计数映射"。

对每个 (场次, k) 同时报四个数：
  dur_map/dur  = 现行口径（时长映射，报时长加权）—— 表里已有的那个
  dur_map/sent = 现行映射下**按句平均**（= 0924 主指标 A）
  cnt_map/sent = 按句计数映射 + 按句平均（口径自洽的口径 B）
  cnt_map/dur  = 按句映射 + 时长加权（另一侧）
以及最优可能 `oracle_sent`（用真值直接算，任意置换的上界），
用来区分"映射选错"与"K 本身选错"。

    python -u -B scripts/evaluation/probe_mapping_objective.py --limit 3
    python -u -B scripts/evaluation/probe_mapping_objective.py          # 12 场全网格
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_sentence_attribution as P  # noqa: E402
import probe_sentence_vs_realpath as V  # noqa: E402

GRID_DEFAULT = "v13_sentence_attrib_dev_fullgrid_FIXED.json"


def score(sents, pred, by_count):
    """一次归属计算。`by_count=True` 时映射权重全取 1（按句计数）。

    只换映射，不改其余记账规则（弃权计入分母等）。
    """
    original = P.hungarian_map
    if by_count:
        P.hungarian_map = lambda p, t, w: original(p, t, [1.0] * len(w))
    try:
        return P.attribution(sents, pred)
    finally:
        P.hungarian_map = original


def score_oracle(sents, pred):
    """`per_sentence` 在**任意映射**下的上界：每个预测标签独立取它最常对应的真值。

    与 `cnt_map_sent` 的关键区别：匈牙利求的是**双射**最优。过分割时两个簇同指
    一人，双射只能把真值给其中一个、另一个必然判错；独立取最大则两个都能对上。
    所以这个值是"映射层面的天花板"，用它区分**映射选错**与 **K 本身选错**。
    平局按真值名取小，保证可复现。
    """
    original = P.hungarian_map

    def best_map(p_, t_, _w):
        counts = Counter()
        for x, y in zip(p_, t_):
            # 与 `hungarian_map` 同一条规则：显式判 None，别用真值判空（0 是 falsy）。
            if x is None or x == "" or not y:
                continue
            counts[(str(x), str(y))] += 1
        best = {}
        for (px, ty), c in counts.items():
            if px not in best or (c, ty) > (best[px][0], best[px][1]):
                best[px] = (c, ty)
        return {k: v[1] for k, v in best.items()}

    P.hungarian_map = best_map
    try:
        return P.attribution(sents, pred)
    finally:
        P.hungarian_map = original


def collect(grid_path, limit):
    grid = json.loads(grid_path.read_text(encoding="utf-8"))["reports"]
    meetings = sorted(grid) if not limit else sorted(grid)[:limit]
    rows = []
    for meeting in meetings:
        for k in sorted(grid[meeting], key=lambda x: int(x)):
            k = int(k)
            emb, times = V.P.cached_windows(str(P.CACHE), meeting, P.WIN, P.HOP)
            vad = json.loads((P.VAD_CACHE / f"{meeting}.json").read_text(encoding="utf-8"))
            gated, gtimes = V.gate_windows(emb, times, [(a, b) for a, b in vad["intervals"]],
                                           P.WIN, P.MIN_VAD)
            tiers = P.read_textgrid(P.DATA / "TextGrid" / f"{meeting}.TextGrid")
            sents = P.annotate_overlap(P.sentences_from_textgrid(tiers, P.SECONDS), tiers)
            if not sents:
                continue
            end = max(s["end"] for s in sents)
            labels = P.clusters_for(gated, k)
            segs = P.segment_map(gtimes, labels, np.full(len(gtimes), P.WIN), end)
            pred = P.sentence_dict_predictions(sents, segs)
            dur = score(sents, pred, by_count=False)
            cnt = score(sents, pred, by_count=True)
            orc = score_oracle(sents, pred)
            rows.append({
                "meeting": meeting, "k": k,
                "keep": grid[meeting][str(k)]["overall"]["per_sentence"],  # 存档值，做一致性校验
                "dur_map_dur": dur["overall"]["duration_weighted"],
                "dur_map_sent": dur["overall"]["per_sentence"],
                "cnt_map_sent": cnt["overall"]["per_sentence"],
                "cnt_map_dur": cnt["overall"]["duration_weighted"],
                "oracle_sent": orc["overall"]["per_sentence"],
            })
    return rows


def report(rows):
    if not rows:
        print("无可用行。")
        return
    # 一致性校验：重算的 dur_map_dur 应与存档一致，否则说明复现路径有漂移
    drift = [r for r in rows if r["keep"] is not None
             and abs(r["dur_map_sent"] - r["keep"]) > 1e-4]
    print(f"共 {len(rows)} 个 (场次, k) 单元；一致性校验：{len(rows)-len(drift)}/{len(rows)} 与存档吻合")
    if drift:
        print("  ⚠️ 以下单元重算与存档不符，先查漂移再读下面的数：")
        for r in drift[:5]:
            print(f"    {r['meeting']} k={r['k']}: 存档 {r['keep']} vs 重算 {r['dur_map_sent']}")

    # 映射分歧率：两个映射给出不同 per_sentence 的单元占比
    diff = [r for r in rows if abs(r["cnt_map_sent"] - r["dur_map_sent"]) > 1e-9]
    d = np.array([abs(r["cnt_map_sent"] - r["dur_map_sent"]) for r in rows])
    s = np.array([r["cnt_map_sent"] - r["dur_map_sent"] for r in rows])
    print(f"\n映射口径分歧：{len(diff)}/{len(rows)} 个单元不同（{100*len(diff)/len(rows):.1f}%）")
    print(f"  |差| 均值 {100*d.mean():.2f} pts，最大 {100*d.max():.2f} pts，"
          f"有符号均值 {100*s.mean():+.2f} pts")
    if s.std() > 0:
        print(f"  符号：按句映射更高 {int((s>0).sum())} 个 / 更低 {int((s<0).sum())} 个")

    # 按场次取最大 k（现行主指标口径）看结论会不会翻转
    print("\n每场取 per_sentence 最大的 k：")
    print(f"{'会议':<22}{'时长映射':>10}{'按句映射':>10}{'结论':>22}")
    flips = 0
    for meeting in sorted({r["meeting"] for r in rows}):
        rs = [r for r in rows if r["meeting"] == meeting]
        kd = max(rs, key=lambda r: r["dur_map_sent"])
        kc = max(rs, key=lambda r: r["cnt_map_sent"])
        same = kd["k"] == kc["k"]
        flips += (not same)
        if same:
            note = "一致"
        else:
            note = f"★ 翻转 k={kd['k']}→{kc['k']}"
        print(f"{meeting:<22}{kd['k']:>10}{kc['k']:>10}{note:>22}"
              f"   {100*kd['dur_map_sent']:.1f}% vs {100*kc['cnt_map_sent']:.1f}%")
    print(f"\n  ★ 选取的 k 发生翻转：{flips}/{len({r['meeting'] for r in rows})} 场")

    # oracle：任意映射下 per_sentence 的上界。用来区分"映射选错"与"K 本身选错"。
    gap = np.array([r["oracle_sent"] - r["dur_map_sent"] for r in rows])
    print("\n映射层面的天花板（oracle_sent - 现行 dur_map_sent）：")
    print(f"  均值 {100*gap.mean():+.2f} pts，最大 {100*gap.max():+.2f} pts，"
          f"完全无余量 {int((gap <= 1e-9).sum())}/{len(rows)} 个单元")
    print("  余量大 → 换映射还有空间；余量≈0 → 该单元的损失不在映射，在 K 或聚类本身")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 场（调试用）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows = collect(ROOT / "artifacts" / args.grid, args.limit)
    report(rows)
    if args.out:
        (ROOT / "artifacts" / args.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
