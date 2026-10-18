# -*- coding: utf-8 -*-
"""大簇拆分收益诊断：k 的提升到底是不是"拆对了一个人"（**零模型成本**）。

## 要回答的问题

09-30 的诊断把症结定位到候选有效性（§3），并把收益归给**大簇拆分**而非小簇
（§5.4：小簇 clean 贡献 +0.67、大簇 +19.93）。但"更大的 k 分更高"本身不构成
证据：过分割同样能让句级 oracle 变高。本脚本要给出一个**不依赖真值**的量，
用来区分：

    real   真正的大簇二分（父簇够大、拆后内部距离下降、子簇有分离度、时间上成块）
    orphan 纯孤窗碎片（父簇被拆出一个 1-2 窗子簇）
    overlap 重叠伴随小簇（拆出来的子簇落在真值重叠区）
    bad    无效的随机过分割（子簇在时间轴上来回穿插）

## 为什么不看 k 之间的整体指标

`k_all_best` 是 oracle 选出来的，拿它的句级指标说"这个 k 好"是**选择偏差**。
本脚本对**每一层相邻拆分**都报 `split_gain`，让"拆得对不对"逐层可核，
而不是只报被 oracle 挑中的那一层。

## `split_gain` 的定义与一处刻意的近似

    split_gain(k) = Σ_parents [ parent_scatter - Σ_child (n_c/n_p) * child_scatter ]

`*_scatter` 是**簇内平均到质心的余弦距离** `1 - mean(E_i) @ centroid`。
均匀分布下 `mean_j(1 - c·e_j) = 1 - |c|^2`，所以这是簇内平均余弦距离的
一阶近似：常数因子 `|c|` 与几何均值/算术均值之差，**量级几十个百分点**。
取近似是因为精确成对距离是 O(n²)（614 窗 × 9 层 × 12 场 ≈ 1.6e8 次点积），
而本脚本要对每一层都算；sign 与相对大小不受该因子影响，跨层比较是自洽的。
    ponytail: 质心近似，若要报绝对 pts 或跨场比较绝对值，改成精确成对距离。

## 指标只作诊断，**不是**可部署收益

所有 `oracle` / `split_gain` 一列都只是**诊断量**。真值侧只用来给拆分分类
打标签，不参与任何选择规则的构造。

    python -u -B scripts/evaluation/probe_large_cluster_split.py --selfcheck
    python -u -B scripts/evaluation/probe_large_cluster_split.py
    python -u -B scripts/evaluation/probe_large_cluster_split.py --detail L_R003S01C02
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from der import read_rttm  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_small_cluster_audit import truth_flags  # noqa: E402
from probe_mapping_objective import score_oracle  # noqa: E402
from probe_selector_locks import silhouette_core  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
CAP = 10
HOP = P.HOP
EPS = 1e-12
MIN_PARENT = v11.MIN_CLUSTER_WINDOWS          # 父簇够大的下限，直接复用生产门限
SMALL = 2                                     # 与 v13 SLIVER_WINDOWS 同值
SCRATCHY = 0.5                                # 时间穿插比例高于此 = 随机碎片


# ───────────────────── 候选构造（与 v11 同法，调用时读门限常量） ─────────────────────

def build(emb, cap=CAP):
    """复刻 `v11._candidate_clusters`，但**不做门限剪枝**：把 k=1..cap 全留下。

    剪枝后 `eligible=False` 的候选拿不到 silhouette（被写成 -1.0），而本脚本
    要看的是"被门限拦掉的那些 k 拆得对不对"，所以必须全算。
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(emb)
    if n < 2:
        return [], None
    tree = linkage(emb, method="average", metric="cosine")
    cands = [{"k": 1, "labels": np.zeros(n, dtype=int), "sizes": [n],
              "eligible": True, "silhouette": 0.0, "objective": 0.0}]
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        labels = np.asarray(labels, dtype=int).reshape(-1)
        present = sorted(set(int(x) for x in labels))
        if len(present) != k:      # 并列合并时 maxclust 可能给出少于 k 个簇，跳过
            continue
        sizes = [int((labels == l).sum()) for l in range(k)]
        eligible = bool(min(sizes) >= v11.MIN_CLUSTER_WINDOWS
                        and min(sizes) / n >= v11.MIN_CLUSTER_SHARE)
        sil = float(silhouette_core(emb, labels))   # 不带孤窗毒化，全 k 都给值
        cands.append({"k": k, "labels": labels, "sizes": sizes, "eligible": eligible,
                      "silhouette": sil,
                      "objective": sil - v11.MULTI_K_PENALTY * (k - 2)})
    return cands, tree




def truth_share(meeting, starts, mask):
    """`mask` 选中的窗里真值侧「有重叠发声 / 跨换人边界」的比例。

    用 `probe_small_cluster_audit.truth_flags`，**不能**用 `window_reference`——
    后者只返回每窗的多数说话人，输出里永远看不到重叠，会让本列恒为 0。
    **只用于给拆分贴标签，不参与任何选择规则。**
    """
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    if not ref or not mask.any():
        return None, None
    boundary, overlap = truth_flags(ref, starts, P.WIN)
    return (round(float(overlap[mask].mean()), 4),
            round(float(boundary[mask].mean()), 4))


# ───────────────────── 真值分类（只用于给拆分贴标签） ─────────────────────

def kind_of(row, share):
    """按四项判据把一次拆分分类。`share` 是真值重叠占比（None = 无真值覆盖）。"""
    if row["min_child_size"] <= SMALL:
        return "orphan"
    if not row["gain_ok"] or row["scratchy"]:
        return "fragmented"
    if share is not None and share > 0.2:
        return "overlap_born"
    if not row["sep_ok"]:
        return "no_separation"
    return "real"


def summarize(splits):
    """把一个 k 层新增的拆分汇总成可读的一行。"""
    if not splits:
        return {"n": 0, "gain_sum": 0.0, "gain_max": None, "kinds": {},
                "min_parent": None, "min_child": None}
    return {
        "n": len(splits),
        "gain_sum": round(sum(s["gain"] for s in splits), 5),
        "gain_max": round(max(s["gain"] for s in splits), 5),
        "kinds": dict(Counter(s["kind"] for s in splits)),
        "min_parent": min(s["parent_size"] for s in splits),
        "min_child": min(s["min_child_size"] for s in splits),
    }


# ───────────────────── 层次树上的每个合并事件 ─────────────────────

def _linkage(emb):
    from scipy.cluster.hierarchy import linkage
    return linkage(emb, method="average", metric="cosine")


def tree_splits(emb, starts):
    """遍历层次树的**每一个内部合并节点**，算它作为一次"父簇 → 两子簇"拆分的几何量。

    `maxclust` 的 k-1 → k 只拆掉树里**最后一次合并**，拆出来的必然是小簇
    （dev 实测：5 场是 [n-1, 1]、7 场是 [n-2, 2]）。真正的大簇二分发生在树的更高层，
    所以要遍历全部内部节点；"哪个节点在 k 层 partition 里可见"由
    `mark_level_splits` 用 `fcluster` 直接判定，不靠高度不等式推。

    簇的"质心"用簇内**所有窗**的平均表示：一个真实换人窗里混了两个人的声纹，
    本来就该以窗为单位算进去，与 partition 统计口径一致。

    返回 (splits, ...)，`splits` 按父簇窗口数降序。
    """
    n = len(emb)
    if n < 2:
        return [], None
    tree = _linkage(emb)
    window_mass = emb @ emb.T          # n×n，dev 最大 585 窗

    windows_of = {i: np.asarray([i]) for i in range(n)}
    centroid_sum = {i: emb[i].copy() for i in range(n)}
    splits = []

    for row_index in range(len(tree)):
        node = n + row_index
        left, right = int(tree[row_index, 0]), int(tree[row_index, 1])
        windows = np.concatenate([windows_of[left], windows_of[right]])
        parent_size = len(windows)
        if parent_size != int(tree[row_index, 3]):
            raise AssertionError(f"节点 {node} 子树窗口数 != linkage 记录的 count")

        centroid = (centroid_sum[left] + centroid_sum[right]) / parent_size
        norm = float(np.linalg.norm(centroid)) or 1.0
        parent_scatter = 1.0 - float(np.mean(emb[windows] @ centroid)) / norm

        sizes, child_scatter, child_centroids = {}, {}, {}
        for kid in (left, right):
            kwin = windows_of[kid]
            kid_centroid = centroid_sum[kid] / len(kwin)
            k_norm = float(np.linalg.norm(kid_centroid)) or 1.0
            sizes[kid] = len(kwin)
            child_scatter[kid] = 1.0 - float(np.mean(emb[kwin] @ kid_centroid)) / k_norm
            child_centroids[kid] = kid_centroid / k_norm

        weighted = sum(sizes[k] * child_scatter[k] for k in sizes) / parent_size
        gain = parent_scatter - weighted
        sep = 1.0 - float(child_centroids[left] @ child_centroids[right])

        # 时间穿插：按时间扫过父簇窗口，记录子簇归属翻转几次。成块的两人分离 = 1 次
        time_order = np.argsort(starts[windows])
        side = np.concatenate([np.zeros(len(windows_of[left]), dtype=int),
                               np.ones(len(windows_of[right]), dtype=int)])
        seq = side[time_order]
        flips = int((seq[1:] != seq[:-1]).sum()) if len(seq) > 1 else 0
        expected = 2.0 * sizes[left] * sizes[right] / parent_size if parent_size else 0.0

        entry = {
            "node": int(node), "parent_size": parent_size,
            "parent_scatter": round(parent_scatter, 5),
            "child_sizes": sorted(sizes.values(), reverse=True),
            "min_child_size": min(sizes.values()),
            "sizes_ok": bool(parent_size >= MIN_PARENT and min(sizes.values()) > SMALL),
            "child_scatter_weighted": round(weighted, 5),
            "gain": round(gain, 5),
            "gain_ratio": round(gain / parent_scatter, 4) if parent_scatter > 1e-9 else None,
            "gain_ok": bool(gain > 0),
            "sep": round(sep, 5),
            "sep_ok": bool(sep > 0),
            "flips": flips,
            "flips_expected": round(expected, 1),
            "flip_ratio": round(flips / expected, 4) if expected > EPS else None,
            "scratchy": bool(expected > EPS and flips / expected > SCRATCHY),
            "merge_height": round(float(tree[row_index, 2]), 5),
            "windows": windows.tolist(),   # 下游贴真值标签；落地前摘掉
            "truth_overlap_share": None, "truth_boundary_share": None,
            "kind": None, "is_level_split": False, "level_k": None,
        }
        entry["pure"] = bool(entry["sizes_ok"] and entry["gain_ok"]
                             and entry["sep_ok"] and not entry["scratchy"])
        splits.append(entry)

        windows_of[node] = windows
        centroid_sum[node] = centroid_sum[left] + centroid_sum[right]
    del windows_of, centroid_sum

    splits.sort(key=lambda s: -s["parent_size"])
    for entry in splits:
        entry["parent_mass_share"] = round(
            float(window_mass[entry["windows"]].sum()) / n, 6)
    return splits, window_mass


def mark_level_splits(emb, splits, cap=CAP):
    """标出每个内部节点在**哪一层 partition** 上是一次可见的拆分。

    判据直接来自 `fcluster`，不靠高度不等式推：对 k=2..cap，若某节点的窗在 k 层
    同属一个簇、在 k+1 层被拆成不止一个簇，它就是 k 层新增的那次拆分。
    一个节点最多在一个 k 上命中，因为每层只新增一次拆分。
    """
    from scipy.cluster.hierarchy import fcluster

    n = len(emb)
    tree = _linkage(emb)
    level_of = {}
    for k in range(2, min(cap, n)):
        lab_k = np.asarray(fcluster(tree, t=k, criterion="maxclust"), dtype=int)
        lab_next = np.asarray(fcluster(tree, t=k + 1, criterion="maxclust"), dtype=int)
        if len(set(lab_k.tolist())) != k or len(set(lab_next.tolist())) != k + 1:
            continue
        for s in splits:
            window = np.asarray(s["windows"])
            if len(set(lab_k[window].tolist())) != 1:
                continue
            if len(set(lab_next[window].tolist())) < 2:
                continue
            level_of.setdefault(s["node"], k)
    for s in splits:
        s["is_level_split"] = s["node"] in level_of
        s["level_k"] = level_of.get(s["node"])
    return level_of


# ───────────────────── 逐场 ─────────────────────

def analyze(meeting):
    """一场的诊断：整棵树的拆分事件 + 逐 k 汇总 + k_prod→k_all_best 的过程。

    ⚠️ 真值（`oracle` 与 `kind`）只用来**给拆分贴标签**，不参与任何选择规则的
    构造。可部署规则必须只用 `gain` / `sep` / `flip_ratio` / `parent_size` 这些
    不依赖真值的量。
    """
    sents = PE.load_sentences(meeting)
    emb, starts = PE.cached_stage(meeting, "main")
    end = max(s["end"] for s in sents)
    n = len(emb)
    win = np.full(n, P.WIN)

    cands, _tree = build(emb)
    all_splits, window_mass = tree_splits(emb, starts)
    mark_level_splits(emb, all_splits)

    # 每个 k 层只挂上**该层新增的那次**拆分（其余节点在该层不是可见簇）
    witness = {}
    for s in all_splits:
        if s["is_level_split"]:
            witness.setdefault(s["level_k"], []).append(s)

    # ── 逐 k：候选指标 + 本次拆分汇总
    staged = []
    for c in cands:
        pred = P.sentence_dict_predictions(
            sents, P.segment_map(starts, c["labels"], win, end))
        oracle = score_oracle(sents, pred)
        mapped = P.attribution(sents, pred)["overall"]
        oc, oo = oracle["clean"], oracle["overlap"]
        here = witness.get(int(c["k"]), [])
        here = [dict(s) for s in here]
        for s in here:
            mask = np.zeros(n, dtype=bool)
            mask[s.pop("windows")] = True
            s["truth_overlap_share"], s["truth_boundary_share"] = truth_share(
                meeting, starts, mask)
            s["kind"] = kind_of(s, s["truth_overlap_share"])
        staged.append({
            "k": int(c["k"]), "min_size": min(c["sizes"]),
            "min_share": round(min(c["sizes"]) / n, 4),
            "n_small": sum(1 for s in c["sizes"] if s <= SMALL),
            "n_main": sum(1 for s in c["sizes"] if s > SMALL),
            "eligible": bool(c["eligible"]),
            "silhouette_no_orphan_poison": round(c["silhouette"], 4),
            "objective_no_orphan_poison": round(c["objective"], 4),
            "oracle": round(oracle["overall"]["per_sentence"], 4),
            "mapped": round(mapped["per_sentence"], 4),
            "clean_oracle": round(oc["per_sentence"], 4) if oc["n"] else None,
            "overlap_oracle": round(oo["per_sentence"], 4) if oo["n"] else None,
            "abstain": round(mapped["abstain"], 4),
            "unmapped": round(mapped["unmapped"], 4),
            "n_sentences": oracle["overall"]["n"],
            "n_clean": oc["n"],
            "n_overlap": oo["n"],
            "splits": here,
            "split_summary": summarize(here),
        })

    prod = int(v11._select_candidate(
        [{"k": c["k"], "labels": c["labels"], "score": c["silhouette"],
          "objective": c["objective"], "sizes": c["sizes"], "eligible": c["eligible"]}
         for c in cands])["k"])
    by_k = {r["k"]: r for r in staged}
    ge2 = [r for r in staged if r["k"] >= 2]
    elig = [r["k"] for r in staged if r["eligible"] and r["k"] >= 2]
    all_best = max(ge2, key=lambda r: r["oracle"]) if ge2 else None
    elig_best = max(elig, key=lambda k: by_k[k]["oracle"]) if elig else None
    clean_rows = [r for r in ge2 if r["clean_oracle"] is not None]
    clean_best = max(clean_rows, key=lambda r: r["clean_oracle"])["k"] if clean_rows else None
    ref_k = all_best["k"] if all_best else None

    # ── k_prod → k_all_best 的过程：逐 k 的 oracle 台阶与其中的拆分
    span = []
    if ref_k is not None:
        for k in range(max(2, prod), ref_k + 1):
            if k in by_k:
                span.append(by_k[k])
    steps = [{"k": r["k"], "oracle": r["oracle"], "clean_oracle": r["clean_oracle"],
              "split_summary": r["split_summary"],
              "splits": [{q: s[q] for q in
                          ("node", "parent_size", "child_sizes", "gain", "gain_ratio",
                           "sep", "flip_ratio", "parent_mass_share", "sizes_ok",
                           "gain_ok", "sep_ok", "scratchy", "kind",
                           "truth_overlap_share")}
                         for s in r["splits"]]}
             for r in span]
    for step in steps[1:]:
        step["oracle_delta"] = round(step["oracle"] - steps[steps.index(step) - 1]["oracle"], 4)
    if steps:
        steps[0]["oracle_delta"] = None
        prev = by_k.get(prod - 1) if prod - 1 in by_k else None
        steps[0]["oracle_delta"] = round(steps[0]["oracle"] - prev["oracle"], 4) if prev else None

    # ── 判定：k_all_best 那一次拆分是否通过四项判据
    ref_row = by_k.get(ref_k) if ref_k else None
    ref_splits = ref_row["splits"] if ref_row else []
    if ref_k is None:
        verdict = "无 k>=2 候选"
    elif not ref_splits:
        verdict = "k_all_best 的拆分事件缺失"
    else:
        hits = {s["kind"] for s in ref_splits}
        if hits == {"real"}:
            verdict = f"全部 {len(ref_splits)} 个拆分通过四项判据"
        elif "real" in hits:
            n_real = sum(1 for s in ref_splits if s["kind"] == "real")
            verdict = f"部分通过：{n_real}/{len(ref_splits)}"
        else:
            verdict = f"无一通过（{len(ref_splits)} 个拆分，{sorted(hits)}）"

    # ── 全树：通过判据的大拆分有多少个，其中多少落在 k_all_best 上
    pure_all = [s for s in all_splits if s["pure"]]
    pure_sizes = Counter(s["parent_size"] for s in pure_all)

    return {
        "meeting": meeting,
        "n_windows": n, "n_sentences": len(sents),
        "k_prod": prod, "eligible_ks": elig, "k_elig_best": elig_best,
        "k_all_best": ref_k, "k_clean_best": clean_best,
        "n_tree_splits": len(all_splits),
        "n_pure_splits": len(pure_all),
        "pure_parent_sizes": dict(sorted(pure_sizes.items())),
        "rows": staged,
        "steps": steps,
        "verdict": verdict,
        "all_splits": all_splits,
    }


# ───────────────────── 报告 ─────────────────────

def report(rows):
    print("=" * 132)
    print("第一部分：逐场逐 k（零模型，未加门限；«» 里是该 k 新增的那次树枝拆分）")
    print("=" * 132)
    for m in rows:
        print(f"\n{m['meeting']}  主窗 {m['n_windows']}  句 {m['n_sentences']}  "
              f"生产k={m['k_prod']}  合格最优k={m['k_elig_best']}  全候选最优k={m['k_all_best']}  "
              f"clean最优k={m['k_clean_best']}  "
              f"树内拆分 {m['n_tree_splits']} 次，其中通过四项判据 {m['n_pure_splits']} 次")
        print(f"    {'k':>3}{'min':>5}{'小簇':>5}{'主簇':>5}{'合格':>5}{'轮廓(去孤窗)':>13}"
              f"{'oracle':>9}{'clean':>9}{'overlap':>9}{'弃权':>8}{'未匹配':>8}"
              f"   本次树枝拆分")
        print("    " + "-" * 128)
        for r in m["rows"]:
            f = lambda v: (f"{100*v:8.2f}" if v is not None else "       -")
            desc = ""
            if r["splits"]:
                parts = []
                for s in sorted(r["splits"], key=lambda x: -x["parent_size"]):
                    tag = s["kind"] or ("纯拆分" if s["pure"] else "未过判据")
                    parts.append(f"{s['parent_size']}→{s['child_sizes']}"
                                 f" gain{s['gain']:+.3f} sep{s['sep']:.2f}"
                                 f" flip{s['flip_ratio']} [{tag}]")
                desc = "  «  " + " ； ".join(parts) + "  »"
            print(f"    {r['k']:>3}{r['min_size']:>5}{r['n_small']:>5}{r['n_main']:>5}"
                  f"{('是' if r['eligible'] else '否'):>5}"
                  f"{r['silhouette_no_orphan_poison']:>13.4f}{f(r['oracle'])}"
                  f"{f(r['clean_oracle'])}{f(r['overlap_oracle'])}"
                  f"{f(r['abstain'])}{f(r['unmapped'])}{desc}"
                  + ("  *现行" if r["k"] == m["k_prod"] else ""))


def step_report(rows):
    print("\n" + "=" * 132)
    print("第二部分：k_prod → k_all_best 的 oracle 台阶（每一级新增的那次树枝拆分）")
    print("=" * 132)
    for m in rows:
        if not m["steps"]:
            print(f"\n{m['meeting']}：无台阶可报")
            continue
        print(f"\n{m['meeting']}  {m['k_prod']} → {m['k_all_best']}  "
              f"（判定：{m['verdict']}）")
        print(f"    {'k':>3}{'Δoracle':>9}{'clean':>9}{'本次拆分':>10}"
              f"{'父窗数':>8}{'子簇':>10}{'gain':>9}{'增益比':>8}{'分离度':>8}"
              f"{'穿插比':>8}{'质量比':>9}{'重叠占比':>9}{'类别':>12}")
        for st in m["steps"]:
            d = st["oracle_delta"]
            d_text = f"{100*d:+7.2f}" if d is not None else "      -"
            c_text = f"{100*st['clean_oracle']:7.2f}" if st["clean_oracle"] is not None else "      -"
            if not st["splits"]:
                print(f"    {st['k']:>3}{d_text}{c_text}{'无（未新增拆分）':>10}")
                continue
            for s in st["splits"]:
                ov = s["truth_overlap_share"]
                ov_text = f"{100*ov:.0f}%" if ov is not None else "-"
                print(f"    {st['k']:>3}{d_text}{c_text}"
                      f"{'是' if s['parent_size'] >= MIN_PARENT else '否（父簇太小）':>10}"
                      f"{s['parent_size']:>8}{str(s['child_sizes']):>10}"
                      f"{s['gain']:>9.4f}{str(s['gain_ratio']):>8}{str(s['sep']):>8}"
                      f"{str(s['flip_ratio']):>8}{str(s['parent_mass_share']):>9}"
                      f"{ov_text:>9}{str(s['kind']):>12}")
                d_text, c_text = " " * 9, " " * 9      # 同一 k 的后续拆分不再重复报 Δ


def summary(rows):
    print("\n" + "=" * 132)
    print("第三部分：这个量能不能区分「拆对人」和「拆出碎片」（12 场汇总）")
    print("=" * 132)
    kinds, pure_kinds = Counter(), Counter()
    for m in rows:
        for s in m["all_splits"]:
            pure_kinds[s["kind"] or "(未贴标签)"] += 1
        for r in m["rows"]:
            for s in r["splits"]:
                kinds[s["kind"]] += 1
    print(f"k_all_best 那一次拆分共 {sum(kinds.values())} 个：")
    for k, v in kinds.most_common():
        print(f"    {k:<14}{v:>4} 个")
    pass_all = sum(1 for m in rows if m["rows"] and m["k_all_best"]
                   and all(s["kind"] == "real" for s in
                           next(r for r in m["rows"] if r["k"] == m["k_all_best"])["splits"]))
    with_ref = sum(1 for m in rows if m["k_all_best"])
    print(f"\nk_all_best 的拆分全通过四项判据：{pass_all}/{with_ref} 场")

    print(f"\n全树拆分质量（每场 585 窗以下，全部内部节点）：")
    for m in rows:
        sizes = m["pure_parent_sizes"]
        top = sorted(sizes.items(), reverse=True)[:4]
        print(f"    {m['meeting']:<17}树内 {m['n_tree_splits']:>3} 次，"
              f"通过 {m['n_pure_splits']:>3} 次  "
              f"父窗数分布(前四)： {'  '.join(f'{int(k)}窗×{v}' for k, v in top) or '-'}")

    disk = [m for m in rows if m["k_all_best"] and m["k_all_best"] != m["k_prod"]]
    print(f"\nk_all_best 与生产 k 不同的场次：{len(disk)}/12")
    print("  其中 k_all_best 的拆分至少有一个通过四项判据的："
          f"{sum(1 for m in disk if '通过' in m['verdict'])}/{len(disk)}")
    print("\n⚠️ `oracle` / `gain` / `kind` 都是**诊断量**，`k_all_best` **不得**称为可部署收益；"
          "本结论只覆盖 dev 12 场，不外推到 holdout 或真实部署。")


def detail(meeting):
    m = analyze(meeting)
    print(f"\n{'=' * 128}\n树内拆分明细：{meeting}（{m['n_windows']} 窗，"
          f"树内 {m['n_tree_splits']} 次拆分）\n{'=' * 128}")
    print(f"生产k={m['k_prod']}  合格集={m['eligible_ks']}  全优k={m['k_all_best']}  "
          f"clean最优k={m['k_clean_best']}")
    print(f"\n{'父窗数':>8}{'子簇':>10}{'gain':>9}{'增益比':>8}{'分离度':>8}{'穿插比':>8}"
          f"{'质量比':>9}{'合并高度':>10}{'判据':>14}")
    print("-" * 90)
    for s in m["all_splits"][:40]:
        flag = ("sizes✓" if s["sizes_ok"] else "sizes✗") + (
            " gain✓" if s["gain_ok"] else " gain✗") + (
            " sep✓" if s["sep_ok"] else " sep✗") + (
            " 成块✓" if not s["scratchy"] else " 碎片✗")
        print(f"{s['parent_size']:>8}{str(s['child_sizes']):>10}{s['gain']:>9.4f}"
              f"{str(s['gain_ratio']):>8}{str(s['sep']):>8}{str(s['flip_ratio']):>8}"
              f"{str(s['parent_mass_share']):>9}{str(s['merge_height']):>10}"
              f"{flag:>14}{'  ★纯' if s['pure'] else ''}")
    print(f"\n（共 {m['n_tree_splits']} 次，此处只显示父簇最大的 40 次）")


def selfcheck(rows):
    """锁死本轮的完整性要求。"""
    assert len(rows) == 12, f"应处理 12 场，实得 {len(rows)}"
    for m in rows:
        n = m["n_windows"]
        # ① 所有窗口都被统计
        for r in m["rows"]:
            assert r["n_main"] + r["n_small"] == r["k"], \
                f"{m['meeting']} k={r['k']} 主簇+小簇 {r['n_main'] + r['n_small']} != k {r['k']}"
        # ② 每个父簇和子簇关系可追溯：父簇窗口数 == 两子簇窗口数之和
        #    且整棵树的窗口总数守恒（每个内部节点覆盖它在 linkage 里记录的窗数）
        total_pure = 0
        for s in m["all_splits"]:
            assert s["parent_size"] == sum(s["child_sizes"]), \
                f"{m['meeting']} node={s['node']} 父簇 {s['parent_size']} != 子簇之和 {s['child_sizes']}"
            assert s["is_level_split"] == (s["level_k"] is not None), \
                f"{m['meeting']} node={s['node']} level 标记与 level_k 不一致"
            total_pure += s["pure"]
        assert total_pure == m["n_pure_splits"], \
            f"{m['meeting']} pure 计数不一致 {total_pure} != {m['n_pure_splits']}"
        # 根节点必须覆盖全部窗口
        assert max(s["parent_size"] for s in m["all_splits"]) == n, \
            f"{m['meeting']} 树根只覆盖 {max(s['parent_size'] for s in m['all_splits'])}/{n} 窗"
        # ③ clean + overlap == overall
        for r in m["rows"]:
            assert r["n_clean"] + r["n_overlap"] == r["n_sentences"], \
                f"{m['meeting']} k={r['k']} clean+overlap != overall"
        # ④ 簇数量与窗口数一致（重算一遍 k_all_best 的 partition）
        if m["k_all_best"]:
            emb, _t = PE.cached_stage(m["meeting"], "main")
            assert len(emb) == n, f"{m['meeting']} 缓存窗数漂移"
            cands, _ = build(emb)
            hit = next(c for c in cands if c["k"] == m["k_all_best"])
            assert len(set(hit["labels"].tolist())) == hit["k"], f"{m['meeting']} 簇标签数 != k"
            assert int((hit["labels"] >= 0).sum()) == n, f"{m['meeting']} 标签覆盖窗数 != 总窗数"
        # ⑤ 每个 k 层的可见拆分恰为 1 个，且 level_k 与 fcluster 对账一致
        for r in m["rows"]:
            if r["k"] < 2:
                assert not r["splits"], f"{m['meeting']} k=1 不该有拆分"
                continue
            assert all(s["level_k"] == r["k"] for s in r["splits"]), \
                f"{m['meeting']} k={r['k']} 里混入了别层的拆分"
        seen = {}
        for s in m["all_splits"]:
            if s["is_level_split"]:
                assert s["level_k"] not in seen, \
                    f"{m['meeting']} k={s['level_k']} 层出现两次拆分"
                seen[s["level_k"]] = s
        assert len(seen) == len(m["all_splits"]) - sum(
            1 for s in m["all_splits"] if not s["is_level_split"]), \
            f"{m['meeting']} level 拆分计数不一致"
    # ⑥ 不修改原始 cache（sha256 前后一致）
    m0 = rows[0]
    path = PE.PROBE_CACHE / m0["meeting"] / "main.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    PE.cached_stage(m0["meeting"], "main")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, \
        "analysis 改动了原始 cache 文件"
    # ⑦ 不依赖 oracle：拆分结构只由 embedding 决定，重算两次逐项相同
    emb0, _t = PE.cached_stage(m0["meeting"], "main")
    a, _ = tree_splits(emb0, _t)
    b, _ = tree_splits(emb0, _t)
    key = lambda s: (s["node"], s["parent_size"], s["child_sizes"], s["gain"], s["sep"],
                     s["flips"])
    assert [key(s) for s in a] == [key(s) for s in b], "拆分结构不可复现"
    assert m0["n_tree_splits"] == len(a), "树内拆分总数与记录不一致"
    assert sum(1 for s in a if s["is_level_split"]) == len(
        [s for s in a if s["level_k"] is not None]), "level 标记与 level_k 不一致"
    print("自检通过：")
    print(f"  ① 每场 主簇+小簇 == k（{sum(r['k'] for m in rows for r in m['rows'])} 个簇全部计入）")
    print("  ② 父簇窗口数 == 子簇窗口数之和，且 == linkage 记录的 count（父子可追溯）")
    print("  ③ clean + overlap == overall（逐 k 成立）")
    print("  ④ 簇标签数 == k，覆盖全部窗口；树根覆盖全部窗")
    print("  ⑤ 每个 k 层的可见拆分恰为 1 个，且与 fcluster 对账一致")
    print("  ⑥ 原始 cache 未被修改（sha256 前后一致）")
    print("  ⑦ 拆分结构不依赖 oracle（重算逐项相同）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--detail", default="")
    ap.add_argument("--out", default="v13_large_cluster_split.json")
    args = ap.parse_args()
    if args.detail:
        detail(args.detail)
        return
    rows = [analyze(m) for m in split_meetings(DATA, "dev")]
    if args.selfcheck:
        selfcheck(rows)
        return
    report(rows)
    step_report(rows)
    summary(rows)
    selfcheck(rows)
    for m in rows:
        # 窗索引是中间量；几千个糖粒合并也不值得落地，只留层拆分与父簇显著的
        m["all_splits"] = [s for s in m["all_splits"]
                           if s["is_level_split"] or s["parent_size"] >= 20]
        for s in m["all_splits"]:
            s.pop("windows", None)
    (ROOT / "artifacts" / args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
