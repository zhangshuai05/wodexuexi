# -*- coding: utf-8 -*-
"""判别：A 的收益来自"阈值判据"还是仅来自"k 大"？

## 方法（零推理，只读已有结果）

把每个 (场次, 候选) 抽成 `(k, 主指标)` 一个点。全部 12 场 × 13 候选 ≈ 156 个点。
然后回答三个问题：

1. **同 k 上不同候选的成绩是否一致？**（把 k 当自变量，看成绩的组内离散度）
   若一致 → 成绩由 k 决定，阈值的具体位置无关 → A 只是 k 选择器。
2. **A 的成绩 ≈ 该场"取 k 最大"的成绩吗？**（A 是否退化成"取最大 k"变体）
3. **B 的 k=1 是不是"总选最小 k"？**（对照同 k 的固定成绩）

    python -u -B scripts/evaluation/discriminate_ab.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = "per_sentence"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    d = load(ROOT / "artifacts" / "v13_ab_dev_results.json")
    results = [r for r in d["results"] if not r.get("insufficient")]

    # ── 收集 (场次, 候选名, k, 主指标)
    by_k = defaultdict(list)          # k -> [(meeting, cand, score)]
    by_meeting = defaultdict(list)    # meeting -> [(cand, k, score)]
    for r in results:
        for cand, c in r["candidates"].items():
            s = c.get(MAIN)
            if s is None:
                continue
            by_k[c["k"]].append((r["meeting"], cand, s))
            by_meeting[r["meeting"]].append((cand, c["k"], s))

    print("=" * 100)
    print("问题 1：同一个 k 上，不同候选的成绩是否一致？（若一致 → 成绩由 k 决定）")
    print("=" * 100)
    print(f"{'k':>4}{'点数':>6}{'场次':>6}{'成绩均':>9}{'组内标准差':>11}{'组内极差':>10}"
          f"{'  ← 同 k 同场次内的极差中位':>22}")
    for k in sorted(by_k):
        pts = by_k[k]
        scores = [s for _, _, s in pts]
        # 同 k 同场次内的极差（同场次、同 k、不同候选 → 应完全相同，否则有别的因素）
        per_meet = defaultdict(list)
        for m, _, s in pts:
            per_meet[m].append(s)
        within = [max(v) - min(v) for v in per_meet.values() if len(v) > 1]
        n_same_k_same_meet = sum(1 for v in per_meet.values() if len(v) > 1)
        wm = f"{100*np.median(within):.2f}pts ({n_same_k_same_meet} 组)" if within else "—"
        print(f"{k:>4}{len(pts):>6}{len(set(m for m,_,_ in pts)):>6}"
              f"{100*np.mean(scores):>8.1f}%{100*np.std(scores):>10.1f}%"
              f"{100*(max(scores)-min(scores)):>9.1f}%{wm:>22}")

    print("\n" + "=" * 100)
    print("问题 2：A 的成绩 ≈ 该场「取 k 最大」的成绩吗？（是否退化成「取最大 k」变体）")
    print("=" * 100)
    print(f"{'会议':<20}{'A@0.6 k':>9}{'A@0.6主':>9}{'该场最大k':>10}{'最大k候选':>20}"
          f"{'最大k主':>9}{'差':>8}")
    a_better = a_equal = a_worse = 0
    for r in results:
        c = r["candidates"]
        a = c.get("A@0.6", {})
        # 该场全部候选中 k 最大的那个
        best = max(c.items(), key=lambda kv: (kv[1]["k"], kv[1].get(MAIN) or 0))
        da = a.get(MAIN)
        db = best[1].get(MAIN)
        if da is None or db is None:
            continue
        diff = da - db
        tag = "=" if abs(diff) < 1e-9 else ("A高" if diff > 0 else "A低")
        if tag == "=":
            a_equal += 1
        elif tag == "A高":
            a_better += 1
        else:
            a_worse += 1
        print(f"{r['meeting']:<20}{a.get('k',0):>9}{100*da:>8.1f}%"
              f"{best[1]['k']:>10}{best[0]:>20}{100*db:>8.1f}%"
              f"{100*diff:>7.1f}pt {tag}")
    print(f"\n  汇总：A 更高 {a_better} 场 / 相同 {a_equal} 场 / A 更低 {a_worse} 场")
    print("  → 若「相同」占多数，A@0.6 与「取最大 k」等价，不是独立判据。")

    print("\n" + "=" * 100)
    print("问题 3：B 的 k=1 是「总选最小 k」吗？（对照同 k 的同场成绩）")
    print("=" * 100)
    print(f"{'会议':<20}{'B k':>5}{'B主':>8}{'同场base k':>11}{'base主':>8}{'差':>8}")
    for r in results:
        c = r["candidates"]
        # 与问题 2 同款防护：B/base 缺条目或某个候选缺 MAIN 时显式跳过，
        # 不要让 KeyError 半路中断整张表（本文件已有 `candidates` 非空但缺 `base`
        # 的先例，见 `probe_ab_dev` 的 "base" not in c 分支）。
        if "B" not in c or "base" not in c:
            print(f"{r['meeting']:<20}{'—':>5}{'—':>8}{'—':>11}{'—':>8}{'—':>8}"
                  f"   缺 B 或 base 条目，跳过")
            continue
        b, base = c["B"], c["base"]
        if b.get(MAIN) is None or base.get(MAIN) is None:
            print(f"{r['meeting']:<20}{b.get('k', '?'):>5}{'—':>8}"
                  f"{base.get('k', '?'):>11}{'—':>8}{'—':>8}   缺主指标，跳过")
            continue
        print(f"{r['meeting']:<20}{b['k']:>5}{100*b[MAIN]:>7.1f}%"
              f"{base['k']:>11}{100*base[MAIN]:>7.1f}%"
              f"{100*(b[MAIN]-base[MAIN]):>7.1f}pt")
    b_k1 = sum(1 for r in results
               if r["candidates"].get("B", {}).get("k") == 1)
    print(f"\n  B 输出 k=1 的场次：{b_k1}/{len(results)}")
    print("  → 若 B 恒给 k=1，它就不是判据（与「取最大 k」同病：一个常数）。")


if __name__ == "__main__":
    main()
