# -*- coding: utf-8 -*-
"""A/B 结果分析：按修订规格 §R8 出主表。

要点：
  * **A 的阈值在 dev 12 场上选一个全局值**（不是逐场用真值选）。
    选择规则：主指标（按句平均）的**均值**最高；平局取更小的阈值（更少簇）。
  * B 的 cap 固定，不调。
  * 逐场给出 k / 归属率 / 弃权 / 未匹配 / 回退原因。
  * 报告各阈值的**平台期**：若多个阈值给出同一 k，那是 cap 造成，不是判据。
  * **不作迁移结论**（只跑 dev）。

    python -u -B scripts/evaluation/analyze_ab_dev.py [--results artifacts/v13_ab_dev_results.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = "per_sentence"          # 修订规格 §R1：主指标按句平均
AUX = "duration_weighted"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_global_threshold(results, key=MAIN):
    """在全部场次上选一个全局阈值：主指标均值最高，平局取更小阈值。"""
    thresholds = sorted({float(c.split("@")[1])
                         for r in results for c in r["candidates"] if c.startswith("A@")})
    scores = {}
    for t in thresholds:
        vals = [r["candidates"].get(f"A@{t}", {}).get(key) for r in results]
        vals = [v for v in vals if v is not None]
        scores[t] = float(np.mean(vals)) if vals else None
    best = max((t for t in thresholds if scores[t] is not None),
               key=lambda t: (scores[t], -t))
    return best, scores


def show_platform_analysis(results):
    """各阈值的 (cap 前原始簇数 → cap 后 k) 分布，暴露 cap 造成的平台期。"""
    print("\n" + "=" * 100)
    print("A 的阈值行为：cap 前原始簇数 → cap 后 k（同一 k 聚集多个阈值 = cap 平台期）")
    print("=" * 100)
    thresholds = sorted({float(c.split("@")[1])
                         for r in results for c in r["candidates"] if c.startswith("A@")})
    print(f"{'阈值':>6}{'k=1场':>7}{'被cap场':>9}{'k 中位':>8}{'k 范围':>10}{'主指标均':>10}")
    for t in thresholds:
        ks, capped, vals = [], 0, []
        for r in results:
            c = r["candidates"].get(f"A@{t}")
            if not c:
                continue
            ks.append(c["k"])
            capped += 1 if c.get("capped") else 0
            if c.get(MAIN) is not None:
                vals.append(c[MAIN])
        if not ks:
            continue
        print(f"{t:>6.1f}{sum(1 for x in ks if x == 1):>7}{capped:>9}"
              f"{int(np.median(ks)):>8}{f'{min(ks)}-{max(ks)}':>10}"
              f"{100*np.mean(vals):>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="artifacts/v13_ab_dev_results.json")
    args = ap.parse_args()

    d = load(ROOT / args.results)
    results = [r for r in d["results"] if not r.get("insufficient")]
    failed = d.get("failed", [])
    print(f"规格: {d.get('spec')}   split: {d.get('split')}   成功 {len(results)} 场，"
          f"失败 {len(failed)} 场")
    if failed:
        print("**失败场次（显式记录，未静默跳过）**：")
        for f in failed:
            print(f"  - {f['meeting']}: {f['error']}")

    show_platform_analysis(results)

    best_t, scores = pick_global_threshold(results)
    print("\n" + "=" * 100)
    print(f"A 的全局阈值：**t* = {best_t}**（在 dev {len(results)} 场上选，主指标={MAIN}）")
    print("=" * 100)
    print(f"{'阈值':>6}{'主指标均':>10}{'时长加权均':>11}")
    for t in sorted(scores):
        vals = [r["candidates"][f"A@{t}"][AUX] for r in results
                if f"A@{t}" in r["candidates"] and r["candidates"][f"A@{t}"].get(AUX) is not None]
        print(f"{t:>6.1f}{100*scores[t]:>9.1f}%{100*np.mean(vals):>10.1f}%")

    # ── 逐场主表
    print("\n" + "=" * 100)
    print("逐场主表（主指标=按句平均；弃权/未匹配分开报）")
    print("=" * 100)
    hdr = (f"{'会议':<20}{'窗/密窗':>10}{'句':>5}"
           f"{'base k':>8}{'base主':>8}"
           f"{'A k':>6}{'A主':>7}{'A弃':>6}{'A未':>6}"
           f"{'B k':>6}{'B主':>7}{'B弃':>6}{'B未':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for r in results:
        c = r["candidates"]
        a = c.get(f"A@{best_t}", {})
        b = c["B"]
        base = c["base"]
        windows = f"{r['n_windows']}/{r['n_dense']}"
        line = (f"{r['meeting']:<20}{windows:>10}"
                f"{r['n_sentences']:>5}"
                f"{base['k']:>8}{100*base[MAIN]:>7.1f}%"
                f"{a.get('k', 0):>6}{100*a.get(MAIN, 0):>6.1f}%"
                f"{100*a.get('abstain', 0):>5.1f}%{100*a.get('unmapped', 0):>5.1f}%"
                f"{b['k']:>6}{100*b[MAIN]:>6.1f}%"
                f"{100*b.get('abstain', 0):>5.1f}%{100*b.get('unmapped', 0):>5.1f}%")
        print(line)
        rows.append((r["meeting"], base, a, b))

    # ── 汇总（**只作参考，不代替逐场**）
    print("\n" + "=" * 100)
    print("分候选汇总（**不代替逐场**；项目负责人明确要求不要只报均值）")
    for name, getter in (("base", lambda r: r["candidates"]["base"]),
                         (f"A@{best_t}", lambda r: r["candidates"].get(f"A@{best_t}", {})),
                         ("B", lambda r: r["candidates"]["B"])):
        vals = [getter(r).get(MAIN) for r in results if getter(r).get(MAIN) is not None]
        aux = [getter(r).get(AUX) for r in results if getter(r).get(AUX) is not None]
        if not vals:
            continue
        print(f"  {name:<10} 主指标均 {100*np.mean(vals):>5.1f}%  中位 {100*np.median(vals):>5.1f}%"
              f"  最差 {100*min(vals):>5.1f}%  |  时长加权均 {100*np.mean(aux):>5.1f}%"
              f"  ({len(vals)} 场)")

    print("\n" + "=" * 100)
    print("限制（写进报告时必须带上）")
    print("=" * 100)
    print(f"  1. 全部数字是 **dev（样本内）**；A 的阈值 t*={best_t} 在 dev {len(results)} 场上选的。")
    print("  2. **不作迁移结论**：未碰 holdout / train_L。")
    print("  3. **不作部署结论**：本指标读目标会议真值，定义上不能用于未知会议推理。")
    print("  4. 主指标=按句平均；总体含全部参考句，弃权算未答对（§R1）。")


if __name__ == "__main__":
    main()
