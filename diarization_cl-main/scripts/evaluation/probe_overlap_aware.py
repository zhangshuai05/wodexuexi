# -*- coding: utf-8 -*-
"""重叠感知候选诊断：高 K 的收益到底来自重叠句还是干净句（**零模型成本**）。

## 要回答的问题

11/12 场的全候选 oracle 最优 k 不在当前 eligible 集内。上一轮查出小簇 64% 落在
**重叠语音窗**。于是必须分清：**高 K 的提升是"接住了重叠区"，还是"真的分对了人"？**

对 dev 12 场、k=2..10 的每个候选，同时报 overall / clean / overlap 三个口径的
oracle 与 mapped，并标出四个"最优 k"：
  * 当前生产选中的 k
  * eligible 集内 oracle 最优 k
  * overall oracle 最优 k
  * **clean-only oracle 最优 k**（只看非重叠句）
并检查 overall 最优与 clean-only 最优是否一致。

## 为什么 clean 口径是关键

若高 K 只在 overlap 句上提升，说明收益来自"重叠区被多簇切碎后偶然命中"，
属于单通道重叠边界问题，不是候选有效性缺陷。
若高 K 在 **clean** 句上也提升，则候选有效性仍值得研究。

## ⚠️ 口径纪律

* `k_all_best` 是**按整体 oracle 选出来的**，因此"它在 clean 上也好"含有**选择偏差**：
  它不是无偏样本。本脚本对每条 k 都报 clean 值，让读者自己看趋势，
  而不是只报被 oracle 挑中的那一个 k。
* 高 K 的收益是否依赖 1/2 窗簇，由"该 k 是否含小簇"直接标注。

    python -u -B scripts/evaluation/probe_overlap_aware.py --selfcheck
    python -u -B scripts/evaluation/probe_overlap_aware.py
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
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CAP = 10


def silhouette_of(emb, labels):
    return float(v11._silhouette(emb, labels))


def cluster_stats(emb, labels, names):
    """整体纯度 + 每个真值人的主簇尺寸（用于 speaker home）。"""
    pur, home = [], {}
    for lab in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == lab)
        got = [str(names[i]) for i in idx if str(names[i]) != ""]
        if not got:
            pur.append(None)
            continue
        top, n = Counter(got).most_common(1)[0]
        pur.append(n / len(got))
        home[top] = max(home.get(top, 0), len(idx))
    return (float(np.mean([p for p in pur if p is not None])) if pur else None), home


def split_scores(sents, pred):
    """三种口径各自的 per_sentence。`score_oracle` 与 `attribution` 都对同一份 sents 分栏。"""
    o = score_oracle(sents, pred)
    m = P.attribution(sents, pred)
    pick = lambda d, f: (d[f]["per_sentence"] if d[f]["n"] else None)
    return {
        "overall_oracle": pick(o, "overall"), "overall_mapped": pick(m, "overall"),
        "clean_oracle": pick(o, "clean"), "clean_mapped": pick(m, "clean"),
        "overlap_oracle": pick(o, "overlap"), "overlap_mapped": pick(m, "overlap"),
        "n_clean": m["clean"]["n"], "n_overlap": m["overlap"]["n"],
        "n_overall": m["overall"]["n"],
        "abstain": m["overall"]["abstain"], "unmapped": m["overall"]["unmapped"],
    }


def diag(meeting):
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    end = max(s["end"] for s in sents)
    names = window_reference(ref, times, P.WIN)

    from scipy.cluster.hierarchy import fcluster, linkage
    tree = linkage(emb, method="average", metric="cosine")
    rows = []
    for k in range(2, min(CAP, len(emb)) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = sorted([int((labels == l).sum()) for l in range(k)], reverse=True)
        eligible = (min(sizes) >= v11.MIN_CLUSTER_WINDOWS
                    and min(sizes) / len(emb) >= v11.MIN_CLUSTER_SHARE)
        sil = silhouette_of(emb, labels)
        segs = P.segment_map(times, labels, np.full(len(times), P.WIN), end)
        pred = P.sentence_dict_predictions(sents, segs)
        pur, home = cluster_stats(emb, labels, names)
        rows.append({
            "k": k, "eligible": bool(eligible), "sizes": sizes, "min_size": min(sizes),
            "silhouette": round(sil, 4),
            "objective": round(sil - v11.MULTI_K_PENALTY * (k - 2), 4),
            "n_homes": len(home), "purity": (round(pur, 4) if pur is not None else None),
            "has_small": any(s <= 2 for s in sizes),
            **split_scores(sents, pred),
        })

    orc = {r["k"]: r["overall_oracle"] for r in rows}
    cln = {r["k"]: r["clean_oracle"] for r in rows}
    elig = [r["k"] for r in rows if r["eligible"]]
    cands = [{"k": 1, "labels": np.zeros(len(emb), dtype=int), "score": 0.0,
              "objective": 0.0, "sizes": [len(emb)], "eligible": True}] + [
        {"k": r["k"], "labels": None, "score": r["silhouette"],
         "objective": r["objective"], "sizes": r["sizes"],
         "eligible": r["eligible"]} for r in rows]
    k_prod = int(v11._select_candidate(cands)["k"])
    k_all = max(orc, key=lambda k: orc[k])
    k_clean = max((k for k in cln if cln[k] is not None), key=lambda k: cln[k])
    k_elig = max(elig, key=lambda k: orc[k]) if elig else None

    return {
        "meeting": meeting, "n_windows": int(len(emb)), "n_sentences": len(sents),
        "n_truth_speakers": len({s["speaker"] for s in ref}),
        "k_prod": k_prod, "k_all_best": k_all, "k_clean_best": k_clean,
        "k_elig_best": k_elig,
        "all_equals_clean": k_all == k_clean,
        "rows": rows,
    }


def report(rows):
    for m in rows:
        print(f"\n{'='*124}\n{m['meeting']}  主窗 {m['n_windows']}  句子 {m['n_sentences']}  "
              f"真值人 {m['n_truth_speakers']}  |  生产k={m['k_prod']}  "
              f"eligible最优k={m['k_elig_best']}  overall最优k={m['k_all_best']}  "
              f"clean最优k={m['k_clean_best']}  "
              f"{'一致' if m['all_equals_clean'] else '★不一致'}\n{'='*124}")
        print(f"{'k':>3}{'合格':>5}{'min':>5}{'小簇':>5}{'轮廓':>8}{'objective':>10}"
              f"{'纯度':>7}{'主场':>5} | {'整体o':>7}{'整体m':>7}"
              f"{'清洁o':>7}{'清洁m':>7}{'重叠o':>7}{'重叠m':>7} | {'弃权':>6}{'未匹配':>7}")
        print("-" * 124)
        for r in m["rows"]:
            star = ""
            if r["k"] == m["k_prod"]:
                star += "★生产"
            if r["k"] == m["k_all_best"]:
                star += "☆整体"
            if r["k"] == m["k_clean_best"] and r["k"] != m["k_all_best"]:
                star += "◇清洁"
            f = lambda v: (f"{100*v:6.2f}" if v is not None else "     -")
            pur = f"{100*r['purity']:5.1f}%" if r["purity"] is not None else "    -"
            print(f"{r['k']:>3}{('是' if r['eligible'] else '否'):>5}{r['min_size']:>5}"
                  f"{('是' if r['has_small'] else ''):>5}{r['silhouette']:>8.4f}"
                  f"{r['objective']:>10.4f}{pur:>7}"
                  f"{r['n_homes']:>5} | {f(r['overall_oracle'])}{f(r['overall_mapped'])}"
                  f"{f(r['clean_oracle'])}{f(r['clean_mapped'])}"
                  f"{f(r['overlap_oracle'])}{f(r['overlap_mapped'])} | "
                  f"{f(r['abstain'])}{f(r['unmapped'])}  {star}")


def summary(rows):
    print("\n" + "=" * 124)
    print("汇总结论")
    print("=" * 124)
    same = sum(1 for m in rows if m["all_equals_clean"])
    print(f"overall 最优 k 与 clean-only 最优 k **一致**的场次：{same}/12")

    print(f"\n{'会议':<18}{'生产k':>6}{'整体o':>8}{'清洁o':>8}{'整体最优k':>10}"
          f"{'clean最优k':>11}{'整体o@全优k':>12}{'清洁o@全优k':>12}"
          f"{'清洁o@生产k':>12}{'全优k含小簇':>12}")
    print("-" * 124)
    for m in rows:
        d = {x["k"]: x for x in m["rows"]}
        r = d.get(m["k_all_best"])
        prod = d.get(m["k_prod"])
        f = lambda v: (f"{100*v:7.2f}" if v is not None else "      -")
        # k=1 时 rows 里没有该行（k=2..10），故 clean 值无定义，显式标出
        po = f(prod["overall_oracle"]) if prod else "     k=1"
        pc = f(prod["clean_oracle"]) if prod else "     k=1"
        print(f"{m['meeting']:<18}{m['k_prod']:>6}{po:>8}{pc:>8}"
              f"{m['k_all_best']:>10}{m['k_clean_best']:>11}"
              f"{f(r['overall_oracle']):>12}{f(r['clean_oracle']):>12}"
              f"{pc:>12}{('是' if r['has_small'] else '否'):>12}")

    print(f"\n高 K 的 clean 增益（全优 k 的 clean - 生产 k 的 clean，仅两者不同的场次）：")
    gains = []
    for m in rows:
        d = {x["k"]: x for x in m["rows"]}
        if m["k_all_best"] == m["k_prod"] or m["k_prod"] not in d:
            print(f"  {m['meeting']:<18} k 未变（{m['k_prod']}），无增益可算")
            continue
        if d[m["k_all_best"]]["clean_oracle"] is None or d[m["k_prod"]]["clean_oracle"] is None:
            print(f"  {m['meeting']:<18} clean 句数为 0，无定义")
            continue
        g = d[m["k_all_best"]]["clean_oracle"] - d[m["k_prod"]]["clean_oracle"]
        o = d[m["k_all_best"]]["overall_oracle"] - d[m["k_prod"]]["overall_oracle"]
        gains.append((m["meeting"], g, o))
        print(f"  {m['meeting']:<18} 整体 {100*o:+6.2f} pts   clean {100*g:+6.2f} pts"
              f"   {'clean 也改善' if g > 1e-9 else ('clean 不变' if abs(g) <= 1e-9 else '★clean 变差')}")
    if gains:
        n_clean_up = sum(1 for _, g, _ in gains if g > 1e-9)
        print(f"\n  其中 clean 也改善 {n_clean_up}/{len(gains)} 场，"
              f"clean 不变或变差 {len(gains)-n_clean_up}/{len(gains)} 场")


def selfcheck(rows):
    assert len(rows) == 12, f"应处理 12 场，实得 {len(rows)}"
    for m in rows:
        assert len(m["rows"]) == 9, f"{m['meeting']} 应有 k=2..10 共 9 行，实得 {len(m['rows'])}"
        ks = [r["k"] for r in m["rows"]]
        assert ks == list(range(2, 11)), f"{m['meeting']} k 序列异常：{ks}"
        for r in m["rows"]:
            if r["n_overlap"]:
                assert r["n_clean"] + r["n_overlap"] == r["n_overall"], \
                    f"{m['meeting']} k={r['k']} clean+overlap != overall"
            else:
                assert r["n_clean"] == r["n_overall"], \
                    f"{m['meeting']} k={r['k']} 无重叠句时 clean 应等于 overall"
    # 原始 labels 未被修改：重算一次候选并与记录比对规模
    m0 = rows[0]
    emb, times = E.cached_stage(m0["meeting"], "main")
    from scipy.cluster.hierarchy import fcluster, linkage
    tree = linkage(emb, method="average", metric="cosine")
    for r in m0["rows"]:
        lab = v11._relabel_by_first_seen(fcluster(tree, t=r["k"], criterion="maxclust"))
        sizes = sorted([int((lab == l).sum()) for l in range(r["k"])], reverse=True)
        assert sizes == r["sizes"], f"{m0['meeting']} k={r['k']} 规模漂移"
    print("自检通过：")
    print("  12 场 × k=2..10 全在；clean+overlap 句数 == overall；原始 labels 未被修改")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="v13_overlap_aware_candidate_diag.json")
    args = ap.parse_args()
    rows = [diag(m) for m in split_meetings(DATA, "dev")]
    if args.selfcheck:
        selfcheck(rows)
        return
    report(rows)
    summary(rows)
    selfcheck(rows)
    (ROOT / "artifacts" / args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
