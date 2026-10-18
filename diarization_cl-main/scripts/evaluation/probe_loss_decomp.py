# -*- coding: utf-8 -*-
"""损失分层：映射 / K 选择 / 细化 / 聚类。**零模型成本**（只读缓存）。

## 三条腿的口径必须分开，不能混在一个差值里

上一版把「生产变长段」「2s 最优候选」「纯真值 2s 窗」放进同一个差值，
拿出来的数**不是任何一层的贡献**，是混合口径诊断量。本版按三个固定口径重做：

### 腿 1 —— 统一 2s 窗口径（`leg1`）
所有候选共用：同一产主窗、同一 2 秒窗、同一句集、同一 `segment_map`、
同一 oracle。分别报：
  * `k_all_best`  全部生成候选里 oracle 最好的 k（**不含门限** = 强制候选上界）
  * `k_elig_best` 仅 `eligible` 候选中 oracle 最好的 k（≈ 当前门限空间的理想余量）
  * `k_base`      `_select_candidate` 实际选中（现行基线）
  * `eligible_ks` / `touched_cap`（是否顶到 cap=10）
`L_K_elig = oracle(k_elig_best) − oracle(k_base)` 才是"门限内的理想选择余量"。

### 腿 2 —— 生产路径的 K（`leg2`）
对**主窗候选**逐个强制 K，走**生产同一套 `real_path_labels` 细化**，记：
是否真的选中该 K、是否 refined、最终句级 oracle。
`L_K_prod = oracle(生产选择器选的 k) − 现行生产结果` —— 负值说明现行已是最优。

### 腿 3 —— 固定 K 的细化对照（`leg3`）
同一个 K 下比两条：
  2s 主窗输出 → 段化       vs       2s 主窗输出 → 0.6s 细化 → 段化
差值才可称 **"固定 K 下细化路径的净变化"**。

另留 `e2e`：自动 2s base → 自动生产 v12，作为**端到端流程差异**，
**不得**称细化贡献（它同时含 K 不同 + 窗长不同 + 段化不同）。

## ⚠️ 精度与限制

* `attribution` / `score_oracle` 的 `per_sentence` 都**四舍五入到 4 位**。
  所以"映射损失为零"只能说**在当前报告精度下为零**；故本版另报未四舍五入的
  `raw_hits/raw_n`（从 `attribution` 的 `n_wrong`/`n_sentences_scored` 反推不出来，
  改用直接对比 `oracle_pct` 与 `mapped_pct` 的高精度重算，见 `_pct_precise`）。
* `L_clu` 仍是混合口径，单列并标注，**不作为结论**。

    python -u -B scripts/evaluation/probe_loss_decomp.py --limit 2
    python -u -B scripts/evaluation/probe_loss_decomp.py
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
import probe_path_equiv as E  # noqa: E402
import probe_ab_dev as AB  # noqa: E402
from der import read_rttm  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_mapping_objective import score_oracle  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402


def _labels_for(emb, k):
    """主窗簇标签（与 `clusters_for`/生产同法），带缓存以免重复建树。"""
    key = (id(emb), k)
    if key not in _LCACHE:
        if k <= 1 or len(emb) < 2:
            _LCACHE[key] = np.zeros(len(emb), dtype=int)
        else:
            tree = linkage(emb, method="average", metric="cosine")
            _LCACHE[key] = P.v11._relabel_by_first_seen(
                fcluster(tree, t=k, criterion="maxclust"))
    return _LCACHE[key]


_LCACHE = {}

GRID_DEFAULT = "v13_sentence_attrib_dev_fullgrid_FIXED.json"


def _sentences(meeting):
    return E.load_sentences(meeting)


def _pct_precise(sents, labels, times, win, oracle):
    """未四舍五入的按句正确率，用于验证"映射损失是否真的为 0"。

    `attribution`/`score_oracle` 都把结果 round 到 4 位；这里自己数命中数，
    避免"四舍五入后相等"被误读成"数学上相等"。
    """
    segs = P.segment_map(times, labels, np.full(len(labels), win), P.WIN and win)
    pred = P.sentence_dict_predictions(sents, segs)
    truth = [s["speaker"] for s in sents]
    original = P.hungarian_map
    if oracle:
        def best_map(p_, t_, _w):
            counts = Counter()
            for x, y in zip(p_, t_):
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
        weights = [s["end"] - s["start"] for s in sents]
        mapping = P.hungarian_map(list(pred), truth, weights)
        mapped = [mapping.get(str(x), "") for x in pred]
        hits = sum(1 for m, t in zip(mapped, truth) if m and m == t)
        return hits / len(sents), mapping, pred
    finally:
        P.hungarian_map = original


def _score(sents, labels, times, win, oracle):
    segs = P.segment_map(times, labels, np.full(len(labels), win), win + 0.0)
    pred = P.sentence_dict_predictions(sents, segs)
    got = score_oracle(sents, pred) if oracle else P.attribution(sents, pred)
    return got["overall"]["per_sentence"]


def _score_seg(sents, segments, oracle):
    """生产变长段腿。标签统一 str（`diarize` 的 `spk` 是 int）。"""
    segs = [(round(s["start"], 3), round(s["end"], 3), str(s["spk"]))
            for s in segments]
    pred = P.sentence_dict_predictions(sents, segs)
    got = score_oracle(sents, pred) if oracle else P.attribution(sents, pred)
    return got["overall"]["per_sentence"]


def collect(grid_path, limit):
    grid = json.loads(grid_path.read_text(encoding="utf-8"))["reports"]
    meetings = sorted(grid) if not limit else sorted(grid)[:limit]
    rows = []
    for meeting in meetings:
        sents = _sentences(meeting)
        if not sents:
            continue
        end = max(s["end"] for s in sents)

        main_emb, main_times = E.cached_stage(meeting, "main")
        rE, rtimes = E.cached_stage(meeting, "refine")

        # -- 腿 1：统一 2s 窗口径，全候选 + 门限内外分开
        cands, _ = P.v11._candidate_clusters(main_emb, max_speakers=P.v11.MAX_SPEAKERS)
        per_k = {}
        for c in cands:
            k = int(c["k"])
            per_k[k] = {
                "oracle": _score(sents, c["labels"], main_times, P.WIN, oracle=True),
                "mapped": _score(sents, c["labels"], main_times, P.WIN, oracle=False),
                "eligible": bool(c["eligible"]),
                "sizes": c["sizes"],
            }
        k_base = int(P.v11._select_candidate(cands)["k"])
        elig = [k for k, v in per_k.items() if v["eligible"] and k >= 2]
        k_elig_best = max(elig, key=lambda k: per_k[k]["oracle"]) if elig else None
        k_all_best = max(per_k, key=lambda k: per_k[k]["oracle"])

        # -- 腿 2：生产路径下**逐个强制 K**（走同一套 real_path_labels 细化）
        prod_k = {}
        for k in sorted(per_k):
            if k < 2:
                continue
            flabels, ftimes, level = AB.real_path_labels(
                _labels_for(main_emb, k), main_emb, main_times, rE, rtimes)
            win = AB._win(level)
            prod_k[k] = {
                "level": level,
                "oracle": _score(sents, flabels, ftimes, win, oracle=True),
                "mapped": _score(sents, flabels, ftimes, win, oracle=False),
            }

        # -- 腿 3：固定 K 的细化对照（取生产实际选中的 k）
        #     A: 2s 主窗输出直接段化     B: 经 0.6s 细化后段化
        l3 = None
        if k_base >= 2:
            l3 = {
                "k": k_base,
                "a_2s": _score(sents, _labels_for(main_emb, k_base), main_times,
                               P.WIN, oracle=True),
                "b_refined": prod_k[k_base]["oracle"],
                "level": prod_k[k_base]["level"],
            }
            l3["delta"] = l3["b_refined"] - l3["a_2s"]

        # -- 端到端：自动 2s base → 自动生产 v12（**不是**细化贡献）
        prod, _calls = E.run_diarize(meeting)
        e2e = {
            "k_prod": int(prod["k"]),
            "refined": bool(prod["refined"]),
            "rescued": bool(prod["rescued"]),
            "mapped": _score_seg(sents, prod["segments"], oracle=False),
            "oracle": _score_seg(sents, prod["segments"], oracle=True),
            "n_segments": len(prod["segments"]),
        }
        e2e["delta_vs_2s_base"] = e2e["oracle"] - per_k[k_base]["oracle"]

        # -- 精度验证：映射 vs oracle 的未四舍五入差（腿 1 的 base 划分）
        p_oracle, _m, _p = _pct_precise(sents, _labels_for(main_emb, k_base), main_times,
                                       P.WIN, oracle=True)
        p_mapped, mapping, pred = _pct_precise(sents, _labels_for(main_emb, k_base),
                                               main_times, P.WIN, oracle=False)
        truth = [s["speaker"] for s in sents]
        mapped_names = [mapping.get(str(x), "") for x in pred]
        # 簇冲突 = 多个**预测簇**指向同一真值人。这是 oracle 严格优于双射的唯一前提。
        # 必须按映射对计数（簇 → 名字），不能按句子数：1 个簇映射到 1 个真值人时，
        # 命中的句子数再多也只有 1 个簇，冲突恒为 0。
        pairs = Counter((str(x), m) for x, m in zip(pred, mapped_names)
                        if m and x is not None and x != "")
        by_target = Counter(t for _c, t in pairs)
        conflict = sum(c - 1 for c in by_target.values() if c > 1)
        n_pred_clusters = len({str(x) for x in pred if x is not None and x != ""})
        n_matched_clusters = len(pairs)

        # -- 聚类层（混合口径，仅诊断）：纯真值 2s 窗 vs 腿1 base
        ref = [s for s in read_rttm(P.DATA / "TextGrid" / f"{meeting}.rttm")
               if s["start"] < P.SECONDS]
        gt_labels = window_reference(ref, main_times, P.WIN)
        gt_oracle = (_score(sents, gt_labels, main_times, P.WIN, oracle=True)
                     if len(set(gt_labels.tolist())) >= 2 else None)

        rows.append({
            "meeting": meeting,
            "n_truth_speakers": len({s["speaker"] for s in ref}),
            "n_sentences": len(sents),
            "n_windows_2s": int(len(main_emb)),
            "n_pred_clusters_base": n_pred_clusters,
            "n_matched_clusters": n_matched_clusters,
            "conflict_clusters": conflict,
            "precise_oracle": p_oracle,
            "precise_mapped": p_mapped,
            "precise_gap": p_oracle - p_mapped,
            "k_base": k_base,
            "k_elig_best": k_elig_best,
            "k_all_best": k_all_best,
            "eligible_ks": elig,
            "touched_cap": max(per_k) >= P.v11.MAX_SPEAKERS,
            "per_k_1_oracle": {k: v["oracle"] for k, v in per_k.items()},
            "L_K_elig": ((per_k[k_elig_best]["oracle"] - per_k[k_base]["oracle"])
                         if k_elig_best is not None else None),
            "L_K_all": per_k[k_all_best]["oracle"] - per_k[k_base]["oracle"],
            "L_map_2s": per_k[k_base]["oracle"] - per_k[k_base]["mapped"],
            "leg1_base_oracle": per_k[k_base]["oracle"],
            "prod_k": prod_k,
            "L2_best_forced": max(v["oracle"] for v in prod_k.values()) if prod_k else None,
            "L2_prod_k_oracle": prod_k.get(k_base, {}).get("oracle"),
            "leg3": l3,
            "e2e": e2e,
            "gt_oracle": gt_oracle,
        })
    return rows


_LCACHE = {}


def _labels_for(emb, k):
    key = (id(emb), k)
    if key not in _LCACHE:
        from scipy.cluster.hierarchy import fcluster, linkage
        tree = linkage(emb, method="average", metric="cosine")
        _LCACHE[key] = P.v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
    return _LCACHE[key]


def _p(v, w=8, suffix="%"):
    """定宽打印：None → 占位。"""
    return (f"{100*v:{w}.2f}{suffix}" if v is not None else " " * (w + len(suffix) - 1) + "-")


def report(rows):
    if not rows:
        print("无可用行。")
        return
    print("=" * 118)
    print("腿 1：统一 2s 窗口径 —— 门限内 vs 门限外")
    print("=" * 118)
    print(f"{'会议':<18}{'真值':>5}{'K_base':>7}{'K_elig优':>9}{'K_all优':>9}"
          f"{'合格集':>16}{'顶cap':>6} | {'base':>8}{'elig优':>8}{'all优':>8}"
          f" | {'L_K_elig':>9}{'L_K_all':>9}")
    print("-" * 118)
    for r in rows:
        e = r["eligible_ks"]
        es = ",".join(str(x) for x in e) if e else "空集"
        ke = r["k_elig_best"]
        pk = r["per_k_1_oracle"]
        elig_orc = pk.get(ke) if ke is not None else None
        print(f"{r['meeting']:<18}{r['n_truth_speakers']:>5}{r['k_base']:>7}"
              f"{str(ke):>9}{r['k_all_best']:>9}{es:>16}"
              f"{('是' if r['touched_cap'] else '否'):>6} | "
              f"{_p(r['leg1_base_oracle'], 7)}{_p(elig_orc, 7)}"
              f"{_p(pk[r['k_all_best']], 7)} | "
              f"{_p(r['L_K_elig'], 9, '')}{_p(r['L_K_all'], 9, '')}")
    print("-" * 118)

    print("\n-- 腿 1 的 K 余量（pts）--")
    print(f"{'会议':<18}{'L_K_elig(门限内)':>18}{'L_K_all(强制上界)':>19}{'L_map(映射)':>13}")
    print("-" * 72)
    ok, oa, om = [], [], []
    for r in rows:
        if r["L_K_elig"] is not None:
            ok.append(r["L_K_elig"])
        oa.append(r["L_K_all"])
        om.append(r["L_map_2s"])
        fs = lambda v: f"{100*v:17.2f}"
        fe = (f"{100*r['L_K_elig']:17.2f}" if r["L_K_elig"] is not None else f"{'(无合格K)':>17}")
        print(f"{r['meeting']:<18}{fe}{fs(r['L_K_all'])}{fs(r['L_map_2s'])}")
    print("-" * 72)
    m = lambda a: (100 * float(np.mean(a)) if a else float("nan"))
    print(f"{'逐场均值':<18}{m(ok):17.2f}{m(oa):17.2f}{m(om):13.2f}")
    print(f"  注：L_K_elig 有 {len(ok)}/{len(rows)} 场可算（其余合格集为空，回退 k=1）")

    print("\n-- 腿 1 精度：映射是否真的为零（未四舍五入）--")
    print(f"{'会议':<18}{'oracle':>10}{'双射映射':>10}{'未round差(pts)':>15}"
          f"{'预测簇':>7}{'映射对':>7}{'真值人':>7}{'簇冲突':>7}")
    print("-" * 83)
    print("-" * 76)
    for r in rows:
        print(f"{r['meeting']:<18}{100*r['precise_oracle']:9.4f}%"
              f"{100*r['precise_mapped']:9.4f}%{100*r['precise_gap']:14.4f}"
              f"{r['n_pred_clusters_base']:>7}{r['n_matched_clusters']:>7}"
              f"{r['n_truth_speakers']:>7}{r['conflict_clusters']:>7}")

    print("\n" + "=" * 118)
    print("腿 2：生产路径下逐个强制 K（走 real_path_labels 细化）")
    print("=" * 118)
    print(f"{'会议':<18}{'K_base':>7}{'现行oracle':>11}{'强制最优':>10}{'K':>4}"
          f"{'L_K_prod':>10} | 逐 K oracle")
    print("-" * 118)
    for r in rows:
        pk = r["prod_k"]
        if not pk:
            print(f"{r['meeting']:<18}（无可强制 K）")
            continue
        best_k = max(pk, key=lambda k: pk[k]["oracle"])
        cur = pk.get(r["k_base"], {}).get("oracle")
        lk = (pk[best_k]["oracle"] - cur) if cur is not None else None
        detail = " ".join(f"{k}:{100*v['oracle']:.1f}" for k, v in sorted(pk.items()))
        print(f"{r['meeting']:<18}{r['k_base']:>7}"
              f"{(f'{100*cur:10.2f}%' if cur is not None else '        -- ')}"
              f"{100*pk[best_k]['oracle']:9.2f}%{best_k:>4}"
              f"{(f'{100*lk:9.2f}' if lk is not None else '       --')} | {detail}")

    print("\n" + "=" * 118)
    print("腿 3：固定 K 的细化净变化 + 端到端流程差异")
    print("=" * 118)
    print(f"{'会议':<18}{'K':>4} | {'2s输出':>10}{'0.6s细化':>10}{'细化净变化':>11}"
          f" | {'自动生产K':>9}{'生产oracle':>11}{'端到端差':>10}")
    print("-" * 118)
    l3a, e2a = [], []
    for r in rows:
        l3, e2 = r["leg3"], r["e2e"]
        if l3:
            l3a.append(l3["delta"])
            l3s = f"{'':>4}{l3['k']:>4} | {100*l3['a_2s']:9.2f}%{100*l3['b_refined']:9.2f}%"
            l3s += f"{100*l3['delta']:10.2f}"
        else:
            l3s = f"{'':>4}{'--':>4} | {'':>10}{'':>10}{'':>11}"
        e2a.append(e2["delta_vs_2s_base"])
        print(f"{r['meeting']:<18}{l3s} | {e2['k_prod']:>9}"
              f"{100*e2['oracle']:10.2f}%{100*e2['delta_vs_2s_base']:9.2f}")
    print("-" * 118)
    print(f"{'逐场均值':<18}{'':>4} | {'':>10}{'':>10}{m(l3a):10.2f}"
          f" | {'':>9}{'':>11}{m(e2a):9.2f}")
    print(f"  注：腿 3 只有 {len(l3a)}/{len(rows)} 场可算（k_base<2 时无细化）")
    print("  端到端差 = 自动2s(含门限选K) → 自动生产(变K+细化+变长段化)，**不是**细化贡献")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 场（调试用）")
    ap.add_argument("--out", default="v13_loss_decomp.json")
    args = ap.parse_args()
    rows = collect(ROOT / "artifacts" / args.grid, args.limit)
    report(rows)
    if args.out:
        (ROOT / "artifacts" / args.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
