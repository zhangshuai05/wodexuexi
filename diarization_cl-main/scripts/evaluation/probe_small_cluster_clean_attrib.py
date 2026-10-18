# -*- coding: utf-8 -*-
"""小簇是否直接承载 clean 句的正确归属（**零模型成本**）。

## 要区分四种可能

    A 小簇直接承载 clean 句的正确归属
    B 大簇拆分才是 clean 增益，小簇只是伴随产物
    C 小簇主要影响 overlap，clean 增益来自别处
    D 无法区分

## 做法

对每场每个 k（**k=2..10 全部保留明细**，不只 oracle 最优 k）：
  1. 逐簇记 窗数 / 纯度 / 主导真值人 / clean 句覆盖与命中 / overlap 句覆盖与命中；
  2. **删除单个簇**（其余不动）后重算 clean / overlap / overall，得到该簇的贡献；
  3. 比较现行 partition 与全优 partition 的 clean 命中差异。

关键量是 **每簇的 clean 贡献** = 单删该簇后的 clean 变化。
若小簇的 clean 贡献 ≈ 0 而大簇拆分带来增益 ⇒ B；若小簇 clean 贡献显著 ⇒ A。

## 三种加权口径同时报（按你要求）

  * 按句数（`per_sentence`）
  * 按句时长（`duration_weighted`）
  * clean / overlap / overall 三栏

## 限制（随结论一起抄）

* `k_all_best` 是**真值诊断量**，不是可部署规则；
* clean 增益**不称为可获得收益**；
* "整体最优与 clean 最优一致"**不作为收益来源的单独证明**。

    python -u -B scripts/evaluation/probe_small_cluster_clean_attrib.py --selfcheck
    python -u -B scripts/evaluation/probe_small_cluster_clean_attrib.py
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


def scores(sents, pred):
    """三栏 × 两口径。"""
    o = score_oracle(sents, pred)
    m = P.attribution(sents, pred)
    out = {}
    for name in ("overall", "clean", "overlap"):
        for src, tag in ((o, "oracle"), (m, "mapped")):
            cell = src[name]
            out[f"{name}_{tag}"] = cell["per_sentence"]
            out[f"{name}_{tag}_dur"] = cell["duration_weighted"]
        out[f"n_{name}"] = m[name]["n"]
    out["abstain"] = m["overall"]["abstain"]
    out["unmapped"] = m["overall"]["unmapped"]
    return out


def predict(sents, times, labels, end):
    segs = P.segment_map(times, labels, np.full(len(times), P.WIN), end)
    return P.sentence_dict_predictions(sents, segs)


def percl(meeting):
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    end = max(s["end"] for s in sents)
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    names = window_reference(ref, times, P.WIN)

    from scipy.cluster.hierarchy import fcluster, linkage
    tree = linkage(emb, method="average", metric="cosine")
    per_k = []
    for k in range(2, min(CAP, len(emb)) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        base_pred = predict(sents, times, labels, end)
        base = scores(sents, base_pred)
        clusters = []
        for lab in sorted(set(int(x) for x in labels)):
            idx = np.flatnonzero(labels == lab)
            got = [str(names[i]) for i in idx if str(names[i]) != ""]
            comp = Counter(got)
            top, n = comp.most_common(1)[0] if comp else ("", 0)
            # 单删该簇
            sub = np.array([l if l != lab else None for l in labels], dtype=object)
            pr = predict(sents, times, sub, end)
            sc = scores(sents, pr)
            clusters.append({
                "cluster": lab, "n_windows": int(len(idx)),
                "start": round(float(times[idx].min()), 2),
                "end": round(float(times[idx].max()) + P.WIN, 2),
                "dominant": top,
                "purity": round(n / len(got), 4) if got else None,
                "size_bucket": "小簇(<=2)" if len(idx) <= 2 else "大簇(>2)",
                "clean_drop": (base["clean_oracle"] - sc["clean_oracle"]
                               if base["clean_oracle"] is not None and sc["clean_oracle"] is not None
                               else None),
                "overlap_drop": (base["overlap_oracle"] - sc["overlap_oracle"]
                                 if base["overlap_oracle"] is not None and sc["overlap_oracle"] is not None
                                 else None),
                "overall_drop": base["overall_oracle"] - sc["overall_oracle"],
                "clean_drop_mapped": (base["clean_mapped"] - sc["clean_mapped"]
                                      if base["clean_mapped"] is not None and sc["clean_mapped"] is not None
                                      else None),
            })
        per_k.append({"k": k, "sizes": sorted(
            [int((labels == l).sum()) for l in range(k)], reverse=True),
            "min_size": int(min((labels == l).sum() for l in range(k))),
            "has_small": any(int((labels == l).sum()) <= 2 for l in range(k)),
            "silhouette": round(float(v11._silhouette(emb, labels)), 4),
            "clusters": clusters, **base})
    orc = {r["k"]: r["overall_oracle"] for r in per_k}
    cln = {r["k"]: r["clean_oracle"] for r in per_k}
    elig = [r["k"] for r in per_k if
            r["min_size"] >= v11.MIN_CLUSTER_WINDOWS
            and r["min_size"] / len(emb) >= v11.MIN_CLUSTER_SHARE]
    mk = lambda d: max((k for k in d if d[k] is not None), key=lambda k: d[k])
    cands = [{"k": 1, "labels": None, "score": 0.0, "objective": 0.0,
              "sizes": [len(emb)], "eligible": True}] + [
        {"k": r["k"], "labels": None, "score": r["silhouette"],
         "objective": r["silhouette"] - v11.MULTI_K_PENALTY * (r["k"] - 2),
         "sizes": r["sizes"], "eligible": r["k"] in elig} for r in per_k]
    return {
        "meeting": meeting, "n_windows": int(len(emb)), "n_sentences": len(sents),
        "k_prod": int(v11._select_candidate(cands)["k"]),
        "k_all_best": mk(orc), "k_clean_best": mk(cln),
        "k_elig_best": (max(elig, key=lambda k: orc[k]) if elig else None),
        "eligible_ks": elig, "per_k": per_k,
    }


def report(rows):
    for m in rows:
        print(f"\n{'='*130}")
        print(f"{m['meeting']}  主窗 {m['n_windows']}  生产k={m['k_prod']}  "
              f"overall最优k={m['k_all_best']}  clean最优k={m['k_clean_best']}  "
              f"eligible最优k={m['k_elig_best']}")
        print("=" * 130)
        for r in m["per_k"]:
            tag = []
            if r["k"] == m["k_all_best"]:
                tag.append("☆整体最优")
            if r["k"] == m["k_clean_best"]:
                tag.append("◇clean最优")
            if r["k"] == m["k_prod"]:
                tag.append("★生产")
            f = lambda v: (f"{100*v:6.2f}" if v is not None else "     -")
            print(f"  k={r['k']:<2} 尺寸={str(r['sizes']):<28} min={r['min_size']:<4}"
                  f"{'含小簇' if r['has_small'] else '      '} "
                  f"| overall o/m {f(r['overall_oracle'])}/{f(r['overall_mapped'])} "
                  f"clean o/m {f(r['clean_oracle'])}/{f(r['clean_mapped'])} "
                  f"ovlp o/m {f(r['overlap_oracle'])}/{f(r['overlap_mapped'])} "
                  f"(clean {r['n_clean']} 句)  {' '.join(tag)}")
            for c in r["clusters"]:
                if c["n_windows"] > 2 and r["k"] != m["k_all_best"]:
                    continue   # 大簇明细只在全优 k 打印，避免刷屏
                cd = f"{100*c['clean_drop']:+6.2f}" if c["clean_drop"] is not None else "     -"
                od = f"{100*c['overlap_drop']:+6.2f}" if c["overlap_drop"] is not None else "     -"
                pur = f"{100*c['purity']:.0f}%" if c["purity"] is not None else "-"
                print(f"      簇{c['cluster']:<3} {c['size_bucket']:<9} "
                      f"{c['n_windows']:>4}窗 纯度{pur:>5} "
                      f"主导={c['dominant']:<8} 删后 clean {cd}  ovlp {od}")


def summary(rows):
    print("\n" + "=" * 130)
    print("汇总：小簇 vs 大簇 的 clean 贡献（单删该簇后的 clean 变化，pts）")
    print("=" * 130)
    small_all, big_all = [], []
    print(f"{'会议':<18}{'k':>3}{'小簇数':>7}{'小簇clean贡献均值':>18}{'大簇数':>7}"
          f"{'大簇clean贡献均值':>18}{'判断':>12}")
    print("-" * 130)
    verdicts = Counter()
    for m in rows:
        for r in m["per_k"]:
            sm_ = [c for c in r["clusters"] if c["n_windows"] <= 2]
            bg_ = [c for c in r["clusters"] if c["n_windows"] > 2]
            sv = [c["clean_drop"] for c in sm_ if c["clean_drop"] is not None]
            bv = [c["clean_drop"] for c in bg_ if c["clean_drop"] is not None]
            if sv:
                small_all += sv
            big_all += bv
            if r["k"] != m["k_all_best"]:
                continue
            ms_ = float(np.mean(sv)) if sv else 0.0
            mb_ = float(np.mean(bv)) if bv else 0.0
            if not sv:
                v = "无小簇"
            elif ms_ > 1e-9 and ms_ >= mb_:
                v = "A 小簇承载"
            elif ms_ <= 1e-9:
                v = "B 小簇无贡献"
            else:
                v = "D 无法区分"
            verdicts[v] += 1
            print(f"{m['meeting']:<18}{r['k']:>3}{len(sm_):>7}{100*ms_:>18.2f}"
                  f"{len(bg_):>7}{100*mb_:>18.2f}{v:>12}")
    print("-" * 130)
    n = lambda a: (100 * float(np.mean(a)) if a else float("nan"))
    print(f"{'全部 k 的逐簇均值':<18}{'':>3}{len(small_all):>7}{n(small_all):>18.2f}"
          f"{len(big_all):>7}{n(big_all):>18.2f}")
    print(f"\n全优 k 上的判断分布：{dict(verdicts)}")


def selfcheck(rows):
    assert len(rows) == 12, f"应 12 场，实得 {len(rows)}"
    for m in rows:
        assert len(m["per_k"]) == 9, f"{m['meeting']} 应 k=2..10 共 9 行"
        for r in m["per_k"]:
            assert sum(c["n_windows"] for c in r["clusters"]) == m["n_windows"], \
                f"{m['meeting']} k={r['k']} 簇窗数之和 != 主窗数"
            assert abs(sum(c["n_windows"] for c in r["clusters"]
                           if c["size_bucket"].startswith("小"))
                       - sum(c["n_windows"] for c in r["clusters"] if c["n_windows"] <= 2)) == 0
    emb, _ = E.cached_stage(rows[0]["meeting"], "main")
    assert len(emb) == rows[0]["n_windows"]
    print("自检通过：12 场 × k=2..10；簇窗数之和 == 主窗数；小簇分桶自洽；缓存未改")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="v13_small_cluster_clean_attrib.json")
    args = ap.parse_args()
    rows = [percl(m) for m in split_meetings(DATA, "dev")]
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
