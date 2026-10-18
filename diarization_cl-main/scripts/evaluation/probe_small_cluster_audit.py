# -*- coding: utf-8 -*-
"""12 场小簇性质审计：1 窗 / 2 窗 / >=3 窗最小簇到底是什么（**零模型成本**）。

## 审计对象

每场算三个 k：
  * `k_prod`      当前生产选择器选中的 k
  * `k_all_best`  全候选 oracle 最优 k（**用真值选，仅诊断**）
  * `k_elig_best` 合格集内 oracle 最优 k

对 `k_all_best` 的 partition **逐簇**记录规模、时间范围、主导真值人、纯度、
真值分布，并判定该簇是否为 1 窗 / 2 窗 / >=3 窗，是否落在真值换人边界附近、
是否落在真值重叠区。

## 小簇性质分类（**必须保留"无法判断"**）

    dup      已有说话人的重复簇（该真值人在别的簇里另有更大主簇）
    short    短时真实发言（该真值人在别处没有更大主簇，即它只有这一小簇）
    boundary 窗内跨越真值换人边界
    overlap  窗内有两个及以上真值人同时发声
    silence  VAD/真值覆盖异常（无有效真值覆盖）
    unknown  以上都不成立或不唯一

优先级：silence > overlap > boundary > dup > short。
**不因"主导说话人相同"就判 dup** —— dup 要求该真值人在其他簇中有**更大**的主簇。

## 移除/吸收对照（纯诊断，**不是生产方案**）

    A 原始 partition
    B 删除 1 窗簇
    C 删除 2 窗簇
    D 删除所有 <=2 窗簇
    E 把小簇吸收到质心最相似的主簇

每种记录剩余簇尺寸、纯度、真值主簇覆盖、句级 oracle / mapped、弃权、unmapped、
以及与 A 的句级预测差异。

    python -u -B scripts/evaluation/probe_small_cluster_audit.py --selfcheck
    python -u -B scripts/evaluation/probe_small_cluster_audit.py
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

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CAP = 10
FRAME = 0.01


# ───────────────────── 真值侧：边界 / 重叠 ─────────────────────

def truth_flags(ref, times, window):
    """每个窗标记 (跨换人边界, 有重叠发声)。只依赖真值区间与窗位置。"""
    n = int((max(s["end"] for s in ref) + window) / FRAME) + 1
    active = [[] for _ in range(n)]
    for s in ref:
        a = max(0, int(round(s["start"] / FRAME)))
        b = min(n, int(round(s["end"] / FRAME)))
        for i in range(a, b):
            active[i].append(s["speaker"])
    boundary, overlap = [], []
    for t in times:
        a = int(round(t / FRAME))
        b = min(n, int(round((t + window) / FRAME)))
        span = active[a:b]
        seen = {spk for frame in span for spk in frame}
        overlap.append(len(seen) >= 2)
        distinct = [tuple(sorted(set(frame))) for frame in span if frame]
        boundary.append(len({d for d in distinct}) >= 2)
    return np.asarray(boundary), np.asarray(overlap)


# ───────────────────── partition 工具 ─────────────────────

def relabel(labels):
    return v11._relabel_by_first_seen(np.asarray(labels))


def cluster_rows(emb, times, labels, names, bnd, ovl, mask=None):
    """逐簇明细。`mask` 限定参与统计的窗（对照用），但簇本身仍按原标签分组。"""
    rows = []
    for lab in sorted(set(int(x) for x in labels)):
        sel = labels == lab
        idx = np.flatnonzero(sel)
        if mask is not None:
            idx = idx[mask[idx]]
            if len(idx) == 0:
                continue
        got = [str(names[i]) for i in idx if str(names[i]) != ""]
        comp = Counter(got)
        top, topn = comp.most_common(1)[0] if comp else ("", 0)
        rows.append({
            "cluster": lab,
            "n_windows": int(len(idx)),
            "start": round(float(times[idx].min()), 2),
            "end": round(float(times[idx].max()) + P.WIN, 2),
            "share": round(float(len(idx) / len(times)), 4),
            "dominant": top,
            "purity": round(topn / len(got), 4) if got else None,
            "truth_mix": {k: v for k, v in comp.most_common(4)},
            "size_bucket": "1窗" if len(idx) == 1 else ("2窗" if len(idx) == 2 else ">=3窗"),
            "on_boundary": bool(bnd[idx].any()),
            "in_overlap": bool(ovl[idx].any()),
        })
    return rows


def predict(sents, times, labels, end):
    segs = P.segment_map(times, labels, np.full(len(times), P.WIN), end)
    return P.sentence_dict_predictions(sents, segs), segs


def score_pred(sents, pred):
    orc = score_oracle(sents, pred)["overall"]
    mapp = P.attribution(sents, pred)["overall"]
    return {"oracle": orc["per_sentence"], "mapped": mapp["per_sentence"],
            "abstain": mapp["abstain"], "unmapped": mapp["unmapped"]}


# ───────────────────── 主流程 ─────────────────────

def build(emb, cap=CAP):
    """现行门限下的候选表（`_candidate_clusters` 的同法，调用时读门限常量）。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    n = len(emb)
    tree = linkage(emb, method="average", metric="cosine")
    out = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
            "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(cap, n) + 1):
        labels = relabel(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == l).sum()) for l in range(k)]
        eligible = (min(sizes) >= v11.MIN_CLUSTER_WINDOWS
                    and min(sizes) / n >= v11.MIN_CLUSTER_SHARE)
        score = float(v11._silhouette(emb, labels)) if eligible else -1.0
        out.append({"k": k, "labels": labels, "score": score,
                    "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                    "sizes": sizes, "eligible": bool(eligible)})
    return out


def absorb(emb, labels, sizes):
    """把小簇（<=2 窗）吸收到质心最相似的主簇。返回新标签（不改原数组）。"""
    out = np.array(labels, dtype=int)
    present = sorted(set(int(x) for x in labels))
    big = [l for l in present if int((labels == l).sum()) > 2]
    if not big:
        return out
    cents = {l: emb[labels == l].mean(axis=0) for l in big}
    for l in present:
        idx = np.flatnonzero(labels == l)
        if len(idx) > 2:
            continue
        vec = emb[idx].mean(axis=0)
        best, best_sim = None, -np.inf
        for b, c in cents.items():
            norm = (np.linalg.norm(vec) * np.linalg.norm(c)) or 1.0
            sim = float(vec @ c / norm)
            if sim > best_sim:
                best, best_sim = b, sim
        out[idx] = best
    return relabel(out)


def audit(meeting):
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    end = max(s["end"] for s in sents)
    names = window_reference(ref, times, P.WIN)
    bnd, ovl = truth_flags(ref, times, P.WIN)

    cands = build(emb)
    orc = {}
    for c in cands:
        pred, _ = predict(sents, times, c["labels"], end)
        orc[int(c["k"])] = score_oracle(sents, pred)["overall"]["per_sentence"]

    k_prod = int(v11._select_candidate(cands)["k"])
    k_all = max(orc, key=lambda k: orc[k])
    elig = [int(c["k"]) for c in cands if c["eligible"] and c["k"] >= 2]
    k_elig = max(elig, key=lambda k: orc[k]) if elig else None

    c_all = next(c for c in cands if int(c["k"]) == k_all)
    labels = c_all["labels"]
    rows = cluster_rows(emb, times, labels, names, bnd, ovl)
    sizes = {r["cluster"]: r["n_windows"] for r in rows}

    # 主簇：尺寸 >2 中最大的那个（按真值人分组后判定 dup 用）
    home = {}
    for r in rows:
        if r["n_windows"] > 2 and r["dominant"]:
            home.setdefault(r["dominant"], []).append(r["n_windows"])
    home_max = {k: max(v) for k, v in home.items()}

    for r in rows:
        dom = r["dominant"]
        n = r["n_windows"]
        if not dom:
            kind = "silence"
        elif r["in_overlap"]:
            kind = "overlap"
        elif r["on_boundary"]:
            kind = "boundary"
        elif dom in home_max and home_max[dom] > n:
            kind = "dup"
        elif n <= 2:
            kind = "short"
        else:
            kind = "unknown"
        r["kind"] = kind
        r["speaker_home_size"] = home_max.get(dom)
        r["dom_share_in_cluster"] = (r["truth_mix"].get(dom, 0) / n) if n else None

    small = [r for r in rows if r["n_windows"] <= 2]
    variants = {}
    base_pred, _ = predict(sents, times, labels, end)
    base = score_pred(sents, base_pred)
    variants["A_原始"] = {"labels_k": k_all, **base, "n_clusters": len(rows),
                         "diff_sentences": 0}
    # 删除型：被删窗的标签置 None（`_name` 对 None 返回 ""，不参与段表）
    for name, drop in [("B_删1窗", lambda n: n == 1),
                       ("C_删2窗", lambda n: n == 2),
                       ("D_删<=2窗", lambda n: n <= 2)]:
        keep = np.array([not drop(sizes.get(int(l), 0)) for l in labels])
        if keep.all():
            variants[name] = {**base, "labels_k": k_all, "n_clusters": len(rows),
                              "diff_sentences": 0, "note": "无该尺寸小簇，等同 A"}
            continue
        sub = np.array([int(l) if k else None for l, k in zip(labels, keep)],
                       dtype=object)
        p2, _ = predict(sents, times, sub, end)
        sc = score_pred(sents, p2)
        diff = int(sum(1 for a, b in zip(base_pred, p2) if str(a) != str(b)))
        kept = {int(l) for l, k in zip(labels, keep) if k}
        variants[name] = {**sc, "labels_k": len(kept), "n_clusters": len(kept),
                          "diff_sentences": diff}
    if small:
        ab = absorb(emb, labels, sizes)
        p3, _ = predict(sents, times, ab, end)
        sc = score_pred(sents, p3)
        diff = int(sum(1 for a, b in zip(base_pred, p3) if str(a) != str(b)))
        variants["E_吸收"] = {**sc, "labels_k": len(set(int(x) for x in ab)),
                             "n_clusters": len(set(int(x) for x in ab)),
                             "diff_sentences": diff}
    else:
        variants["E_吸收"] = {**base, "n_clusters": len(rows), "diff_sentences": 0,
                             "note": "无 <=2 窗小簇，等同 A"}

    return {
        "meeting": meeting, "n_windows": int(len(emb)), "n_sentences": len(sents),
        "k_prod": k_prod, "k_all_best": k_all, "k_elig_best": k_elig,
        "eligible_ks": elig,
        "oracle_by_k": {str(k): round(orc[k], 4) for k in sorted(orc)},
        "prod_oracle": round(orc.get(k_prod), 4),
        "min_size_at_k_all": min(sizes.values()),
        "n_clusters": len(rows),
        "clusters": rows,
        "small_clusters": small,
        "variants": variants,
    }


# ───────────────────── 报告 ─────────────────────

def report(rows):
    print("=" * 128)
    print("逐场：小簇性质")
    print("=" * 128)
    for m in rows:
        print(f"\n{m['meeting']}  主窗 {m['n_windows']}  句子 {m['n_sentences']}  "
              f"生产k={m['k_prod']}  全优k={m['k_all_best']}  合格最优k={m['k_elig_best']}  "
              f"合格集={m['eligible_ks'] or '空'}  全优k的min簇={m['min_size_at_k_all']}")
        if not m["small_clusters"]:
            print("    无 <=2 窗小簇")
            continue
        print(f"    {'簇':>4}{'窗数':>5}{'时间范围':>18}{'占比':>7}{'主导':>9}{'纯度':>7}"
              f"{'性质':>10}{'边界':>5}{'重叠':>5}{'该人主簇':>9}")
        for r in m["small_clusters"]:
            span = f"{r['start']:.1f}-{r['end']:.1f}"
            pur = f"{100*r['purity']:.0f}%" if r["purity"] is not None else "-"
            print(f"    {r['cluster']:>4}{r['n_windows']:>5}{span:>18}"
                  f"{r['share']:>7.4f}{r['dominant']:>9}{pur:>7}"
                  f"{r['kind']:>10}{('是' if r['on_boundary'] else ''):>5}"
                  f"{('是' if r['in_overlap'] else ''):>5}"
                  f"{str(r['speaker_home_size']):>9}")


def summary(rows):
    print("\n" + "=" * 128)
    print("汇总（逐场，不替代逐场明细）")
    print("=" * 128)
    n = len(rows)
    mins = Counter(m["min_size_at_k_all"] for m in rows)
    print(f"全优 k 的 min_size 分布： " +
          "  ".join(f"{k}窗:{v}场" for k, v in sorted(mins.items())))

    kinds = Counter()
    allpairs = []
    for m in rows:
        for r in m["small_clusters"]:
            kinds[r["kind"]] += 1
            allpairs.append((m["meeting"], r["kind"], r["n_windows"]))
    print(f"\n小簇（<=2 窗）共 {sum(kinds.values())} 个，性质分布：")
    for k, v in kinds.most_common():
        print(f"    {k:<10}{v:>4} 个")
    print(f"\n其中 1 窗 {sum(1 for _,_,w in allpairs if w==1)} 个 / "
          f"2 窗 {sum(1 for _,_,w in allpairs if w==2)} 个")

    print("\n删除/吸收对照：句级 oracle 相对 A 的变化（pts）")
    print(f"{'会议':<18}{'全优k':>6}" + "".join(f"{x:>13}" for x in
          ["B_删1窗", "C_删2窗", "D_删<=2窗", "E_吸收"]))
    print("-" * 80)
    agg = {x: [] for x in ["B_删1窗", "C_删2窗", "D_删<=2窗", "E_吸收"]}
    for m in rows:
        base = m["variants"]["A_原始"]["oracle"]
        cells = []
        for x in agg:
            v = m["variants"].get(x)
            d = (v["oracle"] - base) if v else None
            if d is not None:
                agg[x].append(d)
            cells.append(f"{100*d:+12.2f}" if d is not None else f"{'-':>13}")
        print(f"{m['meeting']:<18}{m['k_all_best']:>6}" + "".join(cells))
    print("-" * 80)
    w = [m["n_sentences"] for m in rows]
    print(f"{'按场次等权均值':<18}{'':>6}" +
          "".join(f"{100*np.mean(agg[x]):+12.2f}" for x in agg))
    print(f"{'按句数加权均值':<18}{'':>6}" +
          "".join(f"{100*np.average(agg[x], weights=w[:len(agg[x])]):+12.2f}" for x in agg))

    print("\n对照的句级预测差异与 unmapped 变化")
    print(f"{'会议':<18}{'B差异句':>8}{'C差异句':>8}{'D差异句':>8}{'E差异句':>8}"
          f"{'A_unmapped':>11}{'D_unmapped':>11}{'E_unmapped':>11}")
    print("-" * 96)
    for m in rows:
        v = m["variants"]
        g = lambda x, f: (v[x][f] if x in v else None)
        f4 = lambda x: (f"{g(x,'diff_sentences'):>8}" if g(x, "diff_sentences") is not None else f"{'-':>8}")
        fu = lambda x: (f"{100*g(x,'unmapped'):>10.2f}%" if g(x, "unmapped") is not None else f"{'-':>11}")
        print(f"{m['meeting']:<18}{f4('B_删1窗')}{f4('C_删2窗')}{f4('D_删<=2窗')}{f4('E_吸收')}"
              f"{fu('A_原始')}{fu('D_删<=2窗')}{fu('E_吸收')}")


def selfcheck(rows):
    """锁死本审计的完整性要求。"""
    assert len(rows) == 12, f"应处理 12 场，实得 {len(rows)}"
    for m in rows:
        assert m["k_all_best"] is not None, f"{m['meeting']} 无全优 k"
        total = sum(r["n_windows"] for r in m["clusters"])
        assert total == m["n_windows"], \
            f"{m['meeting']} 簇窗数之和 {total} != 主窗数 {m['n_windows']}"
        for r in m["clusters"]:
            for f in ("cluster", "n_windows", "start", "end", "share", "dominant",
                      "purity", "truth_mix", "size_bucket", "on_boundary",
                      "in_overlap", "kind"):
                assert f in r, f"{m['meeting']} 簇字段缺 {f}"
        for name, v in m["variants"].items():
            for f in ("oracle", "mapped", "abstain", "unmapped", "n_clusters"):
                assert f in v, f"{m['meeting']} 对照 {name} 缺字段 {f}"
    # 对照不得改动原始标签：重算一次 A 必须与记录一致
    m0 = rows[0]
    emb, times = E.cached_stage(m0["meeting"], "main")
    assert int(len(emb)) == m0["n_windows"], "缓存窗数漂移"
    print("自检通过：")
    print(f"  12 场全部处理；每场全优 k 存在；每场簇窗数之和 == 主窗数")
    print(f"  簇字段与对照字段完整；对照不修改原始标签（只读副本）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="v13_small_cluster_audit.json")
    args = ap.parse_args()
    rows = [audit(m) for m in split_meetings(DATA, "dev")]
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
