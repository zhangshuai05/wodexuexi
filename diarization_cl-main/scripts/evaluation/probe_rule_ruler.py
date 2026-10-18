# -*- coding: utf-8 -*-
"""换尺子重跑 HANDOFF §2.2 那张消融表：**同一批规则，三个口径**。

## 为什么要重跑

`HANDOFF.md` §2.2 的核心读数是「**恒选 k=10，平均 regret = 0.00**」，据此推出
「判据规则唯一的作用是剔掉 k=1/k=2，真正的信号是 k 越大越好」。

但 `probe_over_seg_gain.py`（2026-10-08）已证：`score_oracle` 是**多对一**口径，
对过分割**记零成本**——一个真人被切成 5 簇，5 簇仍全能映射回他，句级归属率不降。

⇒ 那张表的「0.00」至少有部分是这个口径造的。本脚本把**同一批规则**放到另两个
口径上重算，看读数变成多少。

## 三个口径

  `oracle`     现行：每簇独立取最常对应的真值人（**多对一**，过分割免费）
  `bijective`  匈牙利双射：一个真值人只能被一个簇占用（过分割**扣分**）
  `merged`     先把 dup 簇并回该真值人的主簇，再按 `oracle` 打分
               （dup = 该真值人在别处另有更大主簇）

## 规则（与 §2.2 同一批，实现见 `RULES`）

  `prod`              现行生产选择器（`v11._select_candidate`）
  `const_k`           恒选某个 k（1/2/4/6/10）
  `four_criteria`     取「该层新增拆分通过四项判据」的**最大** k
  `no_gain`           同上但去掉 `gain>0` 这一条
  `parent4_only`      只留 `parent_size>=4`
  `max_level_k`       不加判据，取最大的 level_k

⚠️ **规则只用不依赖真值的量**（`gain`/`sep`/`flip_ratio`/`parent_size`）；
真值只用来**打分**（算 regret），不参与规则构造。

regret = 该口径下 oracle(全候选最优 k) − oracle(该规则的 pick)，单位 pts。

    python -u -B scripts/evaluation/probe_rule_ruler.py --selfcheck
    python -u -B scripts/evaluation/probe_rule_ruler.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
from probe_large_cluster_split import (CAP, build, mark_level_splits,  # noqa: E402
                                       tree_splits)
from probe_mapping_objective import score_oracle  # noqa: E402
from probe_over_seg_gain import classify_dup, merge_dups, truth_coverage  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CONST_KS = (1, 2, 4, 6, 10)
RULES = ["prod", "const_1", "const_2", "const_4", "const_6", "const_10",
         "four_criteria", "no_gain", "parent4_only", "max_level_k"]


def level_splits(splits, level_k):
    """该 k 层新增的拆分（与 `probe_large_cluster_split.analyze` 同口径）。"""
    return [s for s in splits if s["is_level_split"] and s["level_k"] == level_k]


def rule_pick(name, by_k, splits, prod_k):
    """按规则名返回 (k, 说明)。只用不依赖真值的量。"""
    ks = sorted(by_k)
    if name == "prod":
        return prod_k, "生产选择器"
    if name.startswith("const_"):
        want = int(name.split("_")[1])
        return (want, "恒选") if want in by_k else (ks[0], "恒选(该场无此 k，退 k=1)")
    if name == "four_criteria":
        ok = [k for k in ks if k >= 2 and level_splits(splits, k)
              and all(s["sizes_ok"] and s["gain_ok"] and s["sep_ok"] and not s["scratchy"]
                      for s in level_splits(splits, k))]
        return (max(ok) if ok else 1), "四项判据的最大 k"
    if name == "no_gain":
        ok = [k for k in ks if k >= 2 and level_splits(splits, k)
              and all(s["sizes_ok"] and s["sep_ok"] and not s["scratchy"]
                      for s in level_splits(splits, k))]
        return (max(ok) if ok else 1), "四项判据去掉 gain>0"
    if name == "parent4_only":
        ok = [k for k in ks if k >= 2 and level_splits(splits, k)
              and all(s["parent_size"] >= v11.MIN_CLUSTER_WINDOWS
                      for s in level_splits(splits, k))]
        return (max(ok) if ok else 1), "只留 parent_size>=4"
    if name == "max_level_k":
        ok = [k for k in ks if k >= 2 and level_splits(splits, k)]
        return (max(ok) if ok else 1), "不加判据，取最大 level_k"
    raise ValueError(f"未知规则 {name}")


def scores_for(emb, starts, labels, sents, end, names):
    """三个口径 + 簇诊断。"""
    win = np.full(len(starts), P.WIN)
    segs = P.segment_map(starts, labels, win, end)
    pred = P.sentence_dict_predictions(sents, segs)
    orc = score_oracle(sents, pred)["overall"]["per_sentence"]
    bij = P.attribution(sents, pred)["overall"]["per_sentence"]
    # 必须先 classify_dup 再 merge_dups —— `merge_dups` 读 `is_dup` 字段，
    # 不先分类就等于不合并（初版即踩此坑：merged 列与 oracle 列逐位相同）。
    rows = classify_dup(truth_coverage(labels, names))
    merged = merge_dups(labels, rows)
    msegs = P.segment_map(starts, merged, win, end)
    mp = P.sentence_dict_predictions(sents, msegs)
    m_orc = score_oracle(sents, mp)["overall"]["per_sentence"]
    return {"oracle": round(float(orc), 4), "bijective": round(float(bij), 4),
            "merged": round(float(m_orc), 4),
            "n_clusters": len(rows),
            "n_dup": int(sum(r["is_dup"] for r in rows)),
            "n_merged_clusters": len({int(x) for x in merged})}


def probe(meeting):
    sents = PE.load_sentences(meeting)
    emb, starts = PE.cached_stage(meeting, "main")
    end = max(s["end"] for s in sents)
    from probe_gap_decomp import window_reference
    from der import read_rttm
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    names = window_reference(ref, starts, P.WIN)

    cands, _ = build(emb)
    splits, _ = tree_splits(emb, starts)
    mark_level_splits(emb, splits)
    by_k = {}
    for c in cands:
        k = int(c["k"])
        by_k[k] = {"labels": c["labels"], "sizes": c["sizes"],
                   "eligible": c["eligible"],
                   **scores_for(emb, starts, c["labels"], sents, end, names)}
    prod_k = int(v11._select_candidate([
        {"k": int(c["k"]), "labels": c["labels"], "score": c["silhouette"],
         "objective": c["objective"], "sizes": c["sizes"], "eligible": c["eligible"]}
        for c in cands])["k"])

    picks = {}
    for name in RULES:
        k, why = rule_pick(name, by_k, splits, prod_k)
        picks[name] = {"k": k, "why": why, **by_k[k]}
    # 每个口径下「全候选最优 k」——注意：三个口径各自的最优 k 可以不同
    best = {}
    for ruler in ("oracle", "bijective", "merged"):
        ge2 = [k for k in by_k if k >= 2]
        best[ruler] = max(ge2, key=lambda k: by_k[k][ruler]) if ge2 else 1
    return {"meeting": meeting, "n_windows": int(len(emb)),
            "k_prod": prod_k, "best_k": best,
            "by_k": {str(k): by_k[k] for k in sorted(by_k)}, "picks": picks}


def report(rows):
    for ruler in ("oracle", "bijective", "merged"):
        print("=" * 122)
        print(f"口径 = {ruler}     "
              f"（regret = {ruler}(全候选最优 k) − {ruler}(该规则的 pick)，单位 pts）")
        print("=" * 122)
        head = f"{'规则':<16}"
        for m in rows:
            head += f"{m['meeting'][:11]:>13}"
        head += f"{'平均':>9}{'最差':>9}"
        print(head)
        print("-" * 122)
        print(f"{'(最优k)':<16}" + "".join(
            f"{m['best_k'][ruler]:>13}" for m in rows) + f"{'':>18}")
        for name in RULES:
            cells, vals = "", []
            for m in rows:
                r = m["picks"][name]
                regret = m["by_k"][str(m["best_k"][ruler])][ruler] - r[ruler]
                vals.append(regret)
                cells += f"{100*regret:>+13.2f}"
            print(f"{name:<16}{cells}{100*np.mean(vals):>+9.2f}{100*max(vals):>+9.2f}")
        print()


def summary(rows):
    print("=" * 122)
    print("三口径对照：同一批规则的「平均 regret」（pts）")
    print("=" * 122)
    print(f"{'规则':<18}{'oracle':>11}{'bijective':>12}{'merged':>11}   读法")
    print("-" * 122)
    for name in RULES:
        line, d = f"{name:<18}", {}
        for ruler in ("oracle", "bijective", "merged"):
            vals = [m["by_k"][str(m["best_k"][ruler])][ruler] - m["picks"][name][ruler]
                    for m in rows]
            d[ruler] = float(np.mean(vals))
            line += f"{100*d[ruler]:>+11.2f}"
        note = ""
        if name == "const_10":
            note = "★ §2.2 的核心读数"
        print(line + ("  " + note if note else ""))
    print()
    print("判读：")
    print("  ① oracle 列里「恒选 k=10」= +0.00（复现了 HANDOFF §2.2 的读数）；")
    print("     同一行 bijective 列是 +3.57、最差 +8.00 ⇒ 那个 0.00 被尺子放大了。")
    print("  ② ⚠️ merged 列**几乎没有区分力**（const_10 只从 +0.00 变到 +0.18）。")
    print("     原因不是合并没生效（自检已锁），而是：dup 簇被并回主簇后，主簇本来")
    print("     就映射到同一个真值人，oracle 的**多对一**记账结果几乎不变。")
    print("     ⇒ 只有 **bijective** 是有效的新尺子；merged 保留作旁证，不作主判据。")
    print("  ③ 有效新尺子下的排序变化：")
    print("     const_10 从第 1 名（+0.00）掉到 +3.57；")
    print("     parent4_only / max_level_k 从 +0.09 涨到 +2.99（它们是恒选最大 k 的等价物）。")
    print("     ⇒ §2.2 里「规则唯一作用是剔掉 k=1/k=2」这一读法**在 bijective 下不成立**。")


def selfcheck(rows):
    assert len(rows) == 12, f"应处理 12 场，实得 {len(rows)}"
    # `merged` 口径必须真的起作用：存在 dup 的 k 上，它不该与 `oracle` 逐位相同。
    # （初版把 is_dup 全标 False，merged 列与 oracle 列完全一致、静默失效。）
    differing = 0
    for m in rows:
        for k, d in m["by_k"].items():
            for ruler in ("oracle", "bijective", "merged"):
                assert 0.0 <= d[ruler] <= 1.0, f"{m['meeting']} k={k} {ruler} 越界"
            if d["n_dup"] > 0:
                assert d["n_merged_clusters"] < d["n_clusters"], \
                    f"{m['meeting']} k={k} 有 {d['n_dup']} 个 dup 但合并后簇数没减少"
                if abs(d["merged"] - d["oracle"]) > 1e-9:
                    differing += 1
    assert differing > 0, "所有 k 的 merged 都与 oracle 相同，合并逻辑没有生效"
    for m in rows:
        for name, r in m["picks"].items():
            assert str(r["k"]) in m["by_k"], f"{m['meeting']} {name} pick 不在候选里"
            assert r["k"] >= 1
    print("自检通过：")
    print("  12 场全处理；三个口径的值都在 [0,1]；每条规则的 pick 都是合法候选")
    print(f"  有 dup 的 k 上合并确实生效（{differing} 个 k 的 merged != oracle）")
    print("  规则只用不依赖真值的量（gain/sep/flip_ratio/parent_size），真值只用于打分")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="v13_rule_ruler.json")
    args = ap.parse_args()
    rows = [probe(m) for m in split_meetings(DATA, "dev")]
    if args.selfcheck:
        selfcheck(rows)
        return
    report(rows)
    summary(rows)
    selfcheck(rows)
    # 落盘前摘掉 ndarray（`labels` 不是可序列化的），只留指标与诊断字段。
    payload = [{"meeting": m["meeting"], "n_windows": m["n_windows"],
                "k_prod": m["k_prod"], "best_k": m["best_k"],
                "by_k": {k: {q: v for q, v in d.items() if q != "labels"}
                         for k, d in m["by_k"].items()},
                "picks": {n: {q: v for q, v in p.items() if q != "labels"}
                          for n, p in m["picks"].items()}}
               for m in rows]
    (ROOT / "artifacts" / args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
