# -*- coding: utf-8 -*-
"""三道锁拆解 + 候选结构统计 + 选择目标统计（dev 12 场，**零模型成本**）。

## 为什么必须拆三道锁

`L_R003S01C02` 实测：门限放宽到 (窗门=1, 份额门=0.0) 时，k=2..10 **全部合格**，
但 `_select_candidate` 仍然回退 k=1。原因是三道锁**串联**，只拆第一道不解决问题：

  A. **准入锁** `min(sizes) >= MIN_CLUSTER_WINDOWS` 且 `min(sizes)/n >= MIN_CLUSTER_SHARE`
  B. **轮廓锁** `_silhouette` 只要任一簇 <2 窗就返回 `-1.0`（孤窗直接毒化该 k 的 score）
  C. **回退锁** `_select_candidate` 末行 `best["score"] > 0.0`，不满足则回退 `candidates[0]`(k=1)

所以本脚本把 A/B/C 逐级拆开（**不改生产**，局部 monkeypatch + finally 还原）：

  unlock 0 = 现状（A+B+C 全在）
  unlock 1 = 去 A（准入恒通过，B/C 仍在）
  unlock 2 = 去 A+B（轮廓改为**忽略孤窗**的 core 版，C 仍在）
  unlock 3 = 去 A+B+C（再拆掉 score>0 回退）

`silhouette_core` 复用 `probe_selector_locks.py`，不另写。

## 用法

    python -u -B scripts/evaluation/probe_lock_diag.py --selfcheck
    python -u -B scripts/evaluation/probe_lock_diag.py               # 三部分全量 12 场
    python -u -B scripts/evaluation/probe_lock_diag.py --detail L_R003S01C02
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import probe_path_equiv as E  # noqa: E402
from der import read_rttm  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_mapping_objective import score_oracle  # noqa: E402
from probe_selector_locks import silhouette_core  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

CAP = 10
KS = list(range(1, 11))
DATA = Path(r"D:\yinp\AISHELL-4\data\test")
W, S = v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE


def build(E, cap=CAP, unlock=0, gates=(W, S)):
    """构造候选表。`unlock` 见模块 docstring。**调用时**读门限常量。"""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n < 2:
        return []
    tree = linkage(E, method="average", metric="cosine")
    out = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
            "objective": 0.0, "sizes": [n], "eligible": True}]
    win_gate, share_gate = gates
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == label).sum()) for label in range(k)]
        if unlock >= 1:
            eligible = True                      # 去 A
        else:
            eligible = (min(sizes) >= win_gate and min(sizes) / n >= share_gate)
        if unlock >= 2:
            score = silhouette_core(E, labels)   # 去 B：忽略孤窗的轮廓
        else:
            score = v11._silhouette(E, labels) if eligible else -1.0
        out.append({"k": k, "labels": labels, "score": float(score),
                    "objective": float(score) - v11.MULTI_K_PENALTY * (k - 2),
                    "sizes": sizes, "eligible": bool(eligible)})
    return out


def select(cands, unlock=0):
    """复刻 `_select_candidate`，按 `unlock` 决定 C 是否生效。"""
    eligible = [c for c in cands if c["eligible"] and c["k"] >= 2]
    if not eligible:
        return cands[0], "无合格候选->k=1"
    best = max(eligible, key=lambda c: (c["objective"], -c["k"]))
    if unlock >= 3:
        return best, "去A+B+C"
    if best["score"] > 0.0:
        return best, "A+B+C"
    return cands[0], "C回退(score<=0)->k=1"


def oracle_of(meeting, sents, times, labels, cache, key):
    if key not in cache:
        segs = P.segment_map(times, labels, np.full(len(times), P.WIN),
                             max(s["end"] for s in sents))
        pred = P.sentence_dict_predictions(sents, segs)
        cache[key] = score_oracle(sents, pred)["overall"]["per_sentence"]
    return cache[key]


def analyze_meeting(meeting):
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    end = max(s["end"] for s in sents)
    ocache = {}

    base_cands = build(emb, unlock=0)
    per_k = []
    for c in base_cands:
        k = c["k"]
        per_k.append({
            "k": k, "min_size": min(c["sizes"]),
            "min_share": min(c["sizes"]) / len(emb),
            "has_orphan": any(s == 1 for s in c["sizes"]),
            "has_tiny2": any(s <= 2 for s in c["sizes"]),
            "eligible": c["eligible"], "silhouette": round(c["score"], 4),
            "objective": round(c["objective"], 4),
            "score_gate_blocked": (c["k"] >= 2 and c["eligible"] and c["score"] <= 0.0),
            "oracle": oracle_of(meeting, sents, times, c["labels"], ocache, k),
        })

    cur, why = select(base_cands, unlock=0)
    elig_ks = [c["k"] for c in base_cands if c["eligible"] and c["k"] >= 2]
    cands_ge2 = [c for c in base_cands if c["k"] >= 2]
    all_best = max(cands_ge2, key=lambda c: per_k[c["k"] - 0]["oracle"] if False else
                   oracle_of(meeting, sents, times, c["labels"], ocache, c["k"]))
    elig_best = (max(elig_ks, key=lambda k: ocache[k]) if elig_ks else None)

    # ── 第二部分：逐级拆锁
    locks = {}
    for unlock, tag in [(1, "去A"), (2, "去A+B"), (3, "去A+B+C")]:
        cands = build(emb, unlock=unlock)
        got, why2 = select(cands, unlock=unlock)
        locks[tag] = {
            "selected_k": int(got["k"]),
            "silhouette": round(float(got["score"]), 4),
            "objective": round(float(got["objective"]), 4),
            "reason": why2,
            "oracle": oracle_of(meeting, sents, times, got["labels"], ocache, got["k"]),
        }

    # ── 第三部分：几种选择规则
    rules = {}
    add = lambda name, k, note="": rules.__setitem__(name, {
        "k": int(k),
        "oracle": ocache[int(k)],
        "regret": (ocache[all_best["k"]] - ocache[int(k)]) if int(k) in ocache else None,
        "note": note,
    })
    add("现行(silhouette/objective)", cur["k"], why)
    if elig_best is not None:
        add("合格集内oracle最优", elig_best)
    add("全候选oracle最优", all_best["k"])
    if elig_ks:
        add("取最大合格k", max(elig_ks))
    core = build(emb, unlock=2)
    got_core, _w = select(core, unlock=2)
    add("轮廓忽略孤窗(去A+B)", got_core["k"], _w)

    return {
        "meeting": meeting, "n_windows": len(emb), "n_sentences": len(sents),
        "n_truth_speakers": len({s["speaker"] for s in ref}),
        "current_k": int(cur["k"]), "current_reason": why,
        "eligible_ks": elig_ks,
        "all_oracle_best": int(all_best["k"]),
        "elig_oracle_best": elig_best,
        "per_k": per_k, "locks": locks, "rules": rules,
    }


# ───────────────────────── 报告 ─────────────────────────

def part1(rows):
    print("=" * 126)
    print("第一部分：候选结构统计（逐场逐 k）")
    print("=" * 126)
    hdr = (f"{'会议':<17}{'k':>3}{'min窗':>6}{'min份额':>9}{'孤窗':>5}{'<=2窗':>6}"
           f"{'合格':>5}{'轮廓':>9}{'objective':>10}{'C拦截':>7}{'句oracle':>9}"
           f"{'现行':>6}{'最优k':>6}")
    print(hdr)
    print("-" * 126)
    for m in rows:
        for r in m["per_k"]:
            mark = ""
            if r["k"] == m["current_k"]:
                mark = " *现行"
            elif r["k"] == m["all_oracle_best"]:
                mark = " ooracle最优"
            print(f"{m['meeting']:<17}{r['k']:>3}{r['min_size']:>6}{r['min_share']:>9.4f}"
                  f"{('是' if r['has_orphan'] else ''):>5}{('是' if r['has_tiny2'] else ''):>6}"
                  f"{('是' if r['eligible'] else '否'):>5}{r['silhouette']:>9.4f}"
                  f"{r['objective']:>10.4f}{('是' if r['score_gate_blocked'] else ''):>7}"
                  f"{r['oracle']*100:>8.2f}%{mark}")
        print("-" * 126)


def part2(rows):
    print("\n" + "=" * 126)
    print("第二部分：三道锁逐级拆解")
    print("=" * 126)
    print(f"{'会议':<17}{'现行k':>6}{'合格集':>12} | {'去A: 选中k':>11}{'是否仍回退':>11}"
          f"{'去A+B: k':>10}{'稳定?':>7}{'去A+B+C: k':>12}{'句oracle':>10}")
    print("-" * 126)
    still = 0
    for m in rows:
        a, ab, abc = m["locks"]["去A"], m["locks"]["去A+B"], m["locks"]["去A+B+C"]
        still += (a["selected_k"] == 1)
        e = ",".join(map(str, m["eligible_ks"])) or "空"
        print(f"{m['meeting']:<17}{m['current_k']:>6}{e:>12} | "
              f"{a['selected_k']:>11}{('是' if a['selected_k'] == 1 else '否'):>11}"
              f"{ab['selected_k']:>10}{('是' if ab['selected_k'] >= 2 else '否'):>7}"
              f"{abc['selected_k']:>12}{abc['oracle']*100:>9.2f}%")
    print("-" * 126)
    print(f"只去 A 后**仍然回退 k=1** 的场次：{still}/{len(rows)}"
          f"   ← 这些场次光放宽门限拿不到任何东西")


def part3(rows):
    print("\n" + "=" * 126)
    print("第三部分：选择目标统计（dev 内诊断，**不是**泛化效果）")
    print("=" * 126)
    names = ["现行(silhouette/objective)", "合格集内oracle最优", "全候选oracle最优",
             "取最大合格k", "轮廓忽略孤窗(去A+B)"]
    print(f"{'会议':<17}{'合格集':>10}{'全优k':>7} | " +
          "".join(f"{n[:12]:>14}" for n in names))
    print("-" * 126)
    stats = {n: {"hit": 0, "tie": 0, "regret": []} for n in names}
    for m in rows:
        e = ",".join(map(str, m["eligible_ks"])) or "空"
        cells = []
        best_val = max(r["oracle"] for r in m["per_k"] if r["k"] >= 1)
        best_ks = {r["k"] for r in m["per_k"] if abs(r["oracle"] - best_val) < 1e-12}
        for n in names:
            r = m["rules"].get(n)
            if r is None:
                cells.append("--")
                continue
            hit = r["k"] in best_ks
            stats[n]["hit"] += hit
            if r["oracle"] == best_val:
                stats[n]["tie"] += 1
            if r["regret"] is not None:
                stats[n]["regret"].append(r["regret"])
            cells.append(f"k{r['k']}({r['oracle']*100:.1f}%)" + (" 命中" if hit else " 未中"))
        print(f"{m['meeting']:<17}{e:>10}{m['all_oracle_best']:>7} | " +
              "".join(f"{c:>14}" for c in cells))
    print("-" * 126)
    n = len(rows)
    print(f"{'规则':<26}{'top-1命中(严格)':>16}{'并列容忍':>10}{'平均regret(pt)':>16}")
    for nm in names:
        s = stats[nm]
        reg = np.mean(s["regret"]) * 100 if s["regret"] else float("nan")
        print(f"{nm:<26}{f'{s["hit"]}/{n}':>16}"
              f"{f'{s["tie"]}/{n}':>10}{reg:>16.2f}")
    print("\n严格命中 = 选中 k 恰在全候选 oracle 最优的 k 集合内（并列全算命中）")
    print("平均 regret = oracle(全候选最优) - oracle(该规则的 pick)，单位 pts")


def detail(meeting):
    """逐簇真值构成：每个簇对应哪些真值人、大簇是否混合、孤窗是什么。"""
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    names = window_reference(ref, times, P.WIN)
    cands = build(emb, unlock=0)
    print(f"\n{'='*100}\n逐簇真值构成：{meeting}（n={len(emb)} 窗，"
          f"真值说话人 {sorted({s['speaker'] for s in ref})}）\n{'='*100}")
    for c in cands:
        if c["k"] < 2:
            continue
        if c["k"] > 6:
            continue
        print(f"\n-- k={c['k']}  合格={c['eligible']}  轮廓={c['score']:.4f}  "
              f"尺寸={sorted(c['sizes'], reverse=True)}")
        for label in sorted(set(int(x) for x in c["labels"])):
            mask = c["labels"] == label
            got = [str(x) for x in names[mask] if str(x) != ""]
            total = int(mask.sum())
            if not got:
                print(f"     簇{label}: {total} 窗，无有效真值覆盖")
                continue
            comp = Counter(got)
            top = comp.most_common(3)
            desc = "  ".join(f"{k}:{v}({100*v/len(got):.0f}%)" for k, v in top)
            mix = "*混合" if len(comp) > 1 else ""
            orphan = "  (孤窗)" if total == 1 else ""
            print(f"     簇{label}: {total:>4} 窗  主={desc} {mix}{orphan}")


def selfcheck():
    """锁死本轮纠正：门限放宽不足以救 L_R003S01C02。"""
    emb, _t = E.cached_stage("L_R003S01C02", "main")
    cur, _w = select(build(emb, unlock=0), 0)
    assert cur["k"] == 1, f"现状应为 k=1，实得 {cur['k']}"
    wide = build(emb, unlock=0, gates=(1, 0.0))
    elig = [c["k"] for c in wide if c["eligible"] and c["k"] >= 2]
    assert elig == list(range(2, 11)), f"放宽后应全合格，实得 {elig}"
    got, why = select(wide, 0)
    assert got["k"] == 1, f"仅放宽门限应仍回退 k=1，实得 {got['k']}（{why}）"
    never = build(emb, unlock=1)
    got2, why2 = select(never, 1)
    assert got2["k"] == 1, f"仅去 A 应仍回退，实得 {got2['k']}"
    ab = build(emb, unlock=2)
    got3, _ = select(ab, 2)
    assert got3["k"] >= 2, f"去 A+B 后应得到稳定候选，实得 {got3['k']}"

    emb2, _ = E.cached_stage("M_R003S05C01", "main")
    c2 = build(emb2, unlock=0)
    e2 = [c["k"] for c in c2 if c["eligible"] and c["k"] >= 2]
    assert e2 == [2, 3], f"M_R003S05C01 合格集应为 [2,3]，实得 {e2}"
    s2, _ = select(c2, 0)
    assert s2["k"] == 2, f"M_R003S05C01 现行应选 2，实得 {s2['k']}"

    assert (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE) == (W, S), "门限常量被改动"
    print("自检通过：")
    print("  L_R003S01C02 现状 k=1；放宽门限(1,0.0) 后 2..10 全合格但**仍回退 k=1**")
    print("  仅去 A 仍回退；去 A+B 才得到稳定候选  ← 三道锁串联，已被锁住")
    print("  M_R003S05C01 合格集 [2,3]，现行选 2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--detail", default="")
    ap.add_argument("--out", default="v13_lock_diag.json")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.detail:
        detail(args.detail)
        return
    rows = [analyze_meeting(m) for m in split_meetings(DATA, "dev")]
    sent = Counter()
    for m in rows:
        sent[m["meeting"]] = 0
    print(f"dev 12 场（按 split_meetings 顺序）：{len(rows)} 场\n")
    part1(rows)
    part2(rows)
    part3(rows)
    (ROOT / "artifacts" / args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
