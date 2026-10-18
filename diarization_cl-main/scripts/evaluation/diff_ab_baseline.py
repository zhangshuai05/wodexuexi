# -*- coding: utf-8 -*-
"""旧/新 A/B 产物对比：`base` 的 cap 修正（4 -> 10）改动了什么。

## 背景

`probe_ab_dev` 的 base 早先漏传 `max_speakers`，候选空间只有 1..4；
而 A（`n_clusters_cap=v12.MAX_SPEAKERS`）与 B（`B_CAP`）都是 10。
**base vs A/B 因此不是公平比较。** 修正后重跑，本脚本量化改动范围。

预期（由步骤 A 与缓存版逐场复算独立得出）：**仅 `S_R004S01C01` 发生变化**。
本脚本用真实重跑产物**验证**这个预期，而不是假定它成立。

## 比什么

  * `base.k` / `base.per_sentence` / `base.level`：改动直接作用的对象
  * A / B 的分支：**应当完全不变**（本次只改了 base 那一行）。
    若 A 或 B 也变了，说明改动波及了不该波及的地方 —— 必须报警。
  * `caps` 字段：新产物应记录三处 cap；旧产物没有。

    python -u -B scripts/evaluation/diff_ab_baseline.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OLD_DEFAULT = ("milestones/c1_v13_ab_eval_rev1_20260929/artifacts/"
               "v13_ab_dev_results_PREFIX_CAP_BUG.json")
NEW_DEFAULT = "artifacts/v13_ab_dev_results.json"

# 只比这些字段，避免把 elapsed_s 之类的噪声算成"差异"
FIELDS = ("k", "level", "per_sentence", "duration_weighted")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_pair(a, b, tol=5e-4):
    """归属率显示：只在真变了的时候显示 `旧 -> 新`，否则只显示一个数。

    早先这行把 `k` 的箭头和百分比的箭头拼在一个 f-string 里，
    产出 `4 -> 578.7%` 这种错位——两个字段的格式必须分开算。
    """
    if a is None and b is None:
        return "-"
    if a is None:
        return f"? -> {100*b:.1f}%"
    if b is None:
        return f"{100*a:.1f}% -> ?"
    if abs(a - b) > tol:
        return f"{100*a:.1f}% -> {100*b:.1f}%"
    return f"{100*a:.1f}%"


def branch_diff(old, new, name):
    """返回该分支的差异说明列表（空列表 = 无差异）。"""
    notes = []
    for field in FIELDS:
        a, b = old.get(field), new.get(field)
        if a is None and b is None:
            continue
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or abs(a - b) > 5e-4:
                notes.append(f"{name}.{field}: {a} -> {b}")
        elif a != b:
            notes.append(f"{name}.{field}: {a} -> {b}")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD_DEFAULT)
    ap.add_argument("--new", default=NEW_DEFAULT)
    args = ap.parse_args()

    old = {r["meeting"]: r for r in load(ROOT / args.old)["results"]}
    new = {r["meeting"]: r for r in load(ROOT / args.new)["results"]}

    missing = sorted(set(old) ^ set(new))
    if missing:
        print(f"⚠️ 两边场次不同：{missing}")

    print(f"{'会议':<16}{'base k':>10}   {'base 归属率':<18}说明")
    base_changed, ab_changed, unchanged = [], [], []
    for meeting in sorted(set(old) & set(new)):
        o, n = old[meeting], new[meeting]
        bo = (o.get("candidates") or {}).get("base") or {}
        bn = (n.get("candidates") or {}).get("base") or {}
        base_notes = branch_diff(bo, bn, "base")
        notes = list(base_notes)

        # A / B 分支**不应变化**——变了就是改动波及过广。
        # ⚠️ 两轮分支集合取**并集**：只在一边出现的分支此前被静默跳过，
        # 而"新产物多了一个分支"或"旧产物多了一个分支"本身就是需要排查的变化。
        oc, nc = (o.get("candidates") or {}), (n.get("candidates") or {})
        ab_notes = []
        for branch in sorted(set(oc) | set(nc)):
            if branch == "base":
                continue
            if branch not in oc:
                ab_notes.append(f"{branch}: 只在**新**产物里存在")
            elif branch not in nc:
                ab_notes.append(f"{branch}: 只在**旧**产物里存在")
            else:
                ab_notes += branch_diff(oc[branch], nc[branch], branch)
        notes += ab_notes

        # 两类分别入桶：此前只要 notes 含 base 差异就整条归入 base_changed，
        # 于是同一场里的 A/B 回归**不会出现在** ab_changed 的报告里。
        if base_notes:
            base_changed.append((meeting, base_notes))
        if ab_notes:
            ab_changed.append((meeting, ab_notes))

        pct_o = bo.get("per_sentence")
        pct_n = bn.get("per_sentence")
        show = (f"{bo.get('k')} -> {bn.get('k')}" if bo.get("k") != bn.get("k")
                else str(bo.get("k")))
        pcts = _pct_pair(pct_o, pct_n)
        # 两列之间留出显式分隔符：`show` 与 `pcts` 都是变长文本，
        # 仅靠空格宽度补位会让它们黏在一起（`4 -> 5` + `78.7% -> 90.7%`
        # 会被读成 `4 -> 578.7%`）。
        print(f"{meeting:<16}{show:>10}   {pcts:<18} "
              f"{'★ ' + '; '.join(notes) if notes else '不变'}")
        if not notes:
            unchanged.append(meeting)

    n = len(set(old) & set(new))
    print(f"\n{'='*90}")
    print(f"base 变化 {len(base_changed)}/{n}   "
          f"A/B 分支意外变化 {len(ab_changed)}/{n}   完全不变 {len(unchanged)}/{n}")

    if ab_changed:
        print("\n★ A/B 分支被波及（本次改动**只**应影响 base）——需排查：")
        for meeting, notes in ab_changed:
            print(f"  - {meeting}: {'; '.join(notes)}")
    if base_changed:
        print("\n★ base 变化（修正的**预期**效果）：")
        for meeting, notes in base_changed:
            print(f"  - {meeting}: {'; '.join(notes)}")

    # caps 字段：新产物应有三处 cap 记录，且三者相同。
    # 分母是**新产物的场次数**，不是上面的交集 n —— 只在 new 里出现的场次
    # 也会被计入 caps_seen，用 n 当分母会印出 "13/12" 这种越界分数。
    caps_seen = {}
    for meeting in sorted(new):
        caps = new[meeting].get("caps")
        if caps:
            caps_seen[meeting] = caps
    if caps_seen:
        distinct = {tuple(sorted(c.items())) for c in caps_seen.values()}
        print(f"\ncaps 记录：{len(caps_seen)}/{len(new)} 场有 | 取值组合 {len(distinct)} 种")
        for combo in distinct:
            print(f"  {dict(combo)}")
        if any(len(set(c.values())) > 1 for c in caps_seen.values()):
            print("  ★ 某场三个分支的 cap 仍不一致——公平比较的前提未满足")
    else:
        print("\n⚠️ 新产物没有 caps 字段（旧版本跑的？）")


if __name__ == "__main__":
    main()
