# -*- coding: utf-8 -*-
"""候选簇准入修法对照：**先合并小簇，再判合格**。

现状（`claude_a2_v11._candidate_clusters`）：

    eligible = (min(sizes) >= MIN_CLUSTER_WINDOWS      # 4
                and min(sizes) / n >= MIN_CLUSTER_SHARE)  # 0.015
    score = _silhouette(E, labels) if eligible else -1.0

`min(sizes)` 把账算到**整个候选**头上，而 `fcluster(criterion="maxclust", t=k)`
**强制切出 k 个簇**，k 一超真值就必然产生小簇 ⇒ 好候选被整候选否决 ⇒ 退回 k=1。

审计证据（`scripts/diagnostics/audit_candidate_eligibility.py`，AliMeeting Eval 8 场）：
    72 个候选里 **64 个被否（89%）**；其中 **61 个**「去掉小簇后仍有 >=2 个大簇」。
    `R8001_M8004`：k=7 切出 4 个健康大簇 [581,331,302,192] + 3 个小簇，
    整候选被否 ⇒ auto_k=1 ⇒ DER 69.66%。真值就是 4 人。

**本脚本只改准入那一步，聚类本体（average-linkage + cosine）一行不动**：

    1) `fcluster` 照旧切 k 个簇
    2) 把小于阈值 `max(4, ceil(0.015*n))` 的簇**合并到余弦最近的大簇**
    3) 用**合并后**的标签算 eligible / silhouette / 实际簇数
    4) 实际簇数 <2 的候选丢掉；不同 k 合并后落到同一簇数的，保留 score 最高的

三臂只有「准入方式」不同：`auto`（现状）/ `fixed`（修法）/ `truek`（真值 K，诊断上界）。

    $env:PYTHONIOENCODING="utf-8"
    python run_tiny_fix.py --limit 1            # 冒烟（单场约 90 秒 CPU）
    python run_tiny_fix.py --out tiny_fix_alimeeting_full.json
"""

import argparse
import json
import math
import sys
import time

import numpy as np

from common import (ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools,
                    resolve_wav)
from run_der import (load_models, overlap_attribution, truncate_reference,
                     write_rttm)
from run_oracle_k import extract, segments_from

der, read_rttm = der_tools()


def merge_tiny_clusters(E, labels, min_size):
    """把小于 `min_size` 的簇合并到余弦最近的大簇。

    合并而不是丢弃：所有窗都有标签，下游（细化 / 句级归属 / 段表）零改动。
    """
    labels = np.asarray(labels).copy()
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) < 2:
            break
        tiny = [u for u, c in zip(uniq, counts) if c < min_size]
        if not tiny:
            break
        # 每次只并掉最小的那个小簇，并完重算，避免一次连锁并空
        victim = min(tiny, key=lambda u: int((labels == u).sum()))
        rest = [u for u in uniq if u != victim]
        vic = E[labels == victim].mean(axis=0)
        vic = vic / (np.linalg.norm(vic) + 1e-9)
        cents = np.stack([E[labels == u].mean(axis=0) for u in rest])
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        target = rest[int(np.argmax(cents @ vic))]
        labels[labels == victim] = target
    # 合并后重新编号，保持「按首次出现」的既有约定
    order, out = [], []
    for label in labels.tolist():
        if label not in order:
            order.append(label)
        out.append(order.index(label))
    return np.asarray(out, dtype=int)


def candidates_fixed(E, max_speakers):
    """与 `v11._candidate_clusters` 同构，只在准入处插入小簇合并。"""
    import claude_a2_v11 as v11
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    min_size = max(v11.MIN_CLUSTER_WINDOWS, int(math.ceil(v11.MIN_CLUSTER_SHARE * n)))
    tree = linkage(E, method="average", metric="cosine")
    candidates = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
                   "objective": 0.0, "sizes": [n], "eligible": True}]
    seen = {}
    for k in range(2, min(max_speakers, n) + 1):
        raw = fcluster(tree, t=k, criterion="maxclust")
        merged = merge_tiny_clusters(E, raw, min_size)
        k_eff = int(merged.max()) + 1
        if k_eff < 2:
            continue
        sizes = [int((merged == label).sum()) for label in range(k_eff)]
        score = v11._silhouette(E, merged)
        item = {"k": k_eff, "k_raw": k, "labels": merged, "score": score,
                "objective": score - v11.MULTI_K_PENALTY * (k_eff - 2),
                "sizes": sizes, "eligible": True}
        if k_eff not in seen or item["score"] > seen[k_eff]["score"]:
            seen[k_eff] = item
    candidates.extend(seen[k] for k in sorted(seen))
    return candidates, tree, min_size


def _write(out, args, rows):
    """落盘（每场调一次：中途被打断不该丢掉已跑完的场次）。"""
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "collar": args.collar,
                   "fix": "小簇先合并到余弦最近的大簇，再判合格（聚类本体未动）",
                   "note": "true-K 是诊断上界，用了参考人数先验，不得当作可达收益"},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--collar", type=float, default=COLLAR)
    parser.add_argument("--out", default="tiny_fix_alimeeting_full.json")
    parser.add_argument("--resume", action="store_true",
                        help="读已有产物，跳过已跑完的场次（被打断后续跑用）")
    parser.add_argument("--write-rttm", action="store_true",
                        help="把修法段表写成 predictions/<场次>.fixed.full.rttm，"
                             "供官方 md-eval-22.pl 重评（对外引用的数字要用那把尺子）")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if args.limit:
        meetings = meetings[:args.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    done = []
    if args.resume and out.exists():
        done = [r for r in json.loads(out.read_text(encoding="utf-8")).get("rows", [])
                if "error" not in r]
        finished = {r["meeting"] for r in done}
        meetings = [m for m in meetings if m not in finished]
        print("续跑：已完成 %d 场，本轮剩 %d 场" % (len(finished), len(meetings)),
              flush=True)

    models = load_models()
    rows = []
    header = ("meeting", "真值", "现状k", "修法k", "现状DER", "修法DER", "真值KDER")
    print("%-14s %4s %6s %6s %9s %9s %9s" % header, flush=True)
    for meeting in meetings:
        try:
            wav = resolve_wav(meeting)
            reference = read_rttm(ALI_RTTM / f"{meeting}.rttm")
            true_k = len({s["speaker"] for s in reference})

            started = time.time()
            E, times = extract(models, wav)
            if len(E) < 2:
                raise RuntimeError(f"窗数不足: {len(E)}")
            import claude_a2_v11 as v11
            import claude_a2_v12 as v12
            with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
                old_cands, _ = v11._candidate_clusters(
                    E, max_speakers=v12.MAX_SPEAKERS)
            new_cands, _tree, min_size = candidates_fixed(E, v12.MAX_SPEAKERS)
            elapsed = time.time() - started

            old = v11._select_candidate(old_cands)
            new = v11._select_candidate(new_cands)
            forced = next((c for c in new_cands if c["k"] == true_k), None)

            seg_old, _ = segments_from(old["labels"], times, "auto")
            seg_new, _ = segments_from(new["labels"], times, "fixed")
            row = {
                "meeting": meeting, "true_k": true_k, "n_windows": int(len(E)),
                "min_size": min_size, "elapsed_s": round(elapsed, 1),
                "auto_k": int(old["k"]), "fixed_k": int(new["k"]),
                "auto_candidates": [{"k": c["k"], "score": round(float(c["score"]), 4),
                                     "eligible": bool(c["eligible"]), "sizes": c["sizes"]}
                                    for c in old_cands],
                "fixed_candidates": [{"k": c["k"], "k_raw": c.get("k_raw", c["k"]),
                                      "score": round(float(c["score"]), 4),
                                      "sizes": c["sizes"]} for c in new_cands],
                "der_auto": der(reference, seg_old, collar=args.collar),
                "der_fixed": der(reference, seg_new, collar=args.collar),
            }
            row["der_auto"].update(overlap_attribution(reference, seg_old, args.collar))
            row["der_fixed"].update(overlap_attribution(reference, seg_new, args.collar))
            if forced is not None:
                seg_true, _ = segments_from(forced["labels"], times, "truek")
                row["der_true_k"] = der(reference, seg_true, collar=args.collar)
            print("%-14s %4d %6d %6d %8.2f%% %8.2f%% %8s"
                  % (meeting, true_k, row["auto_k"], row["fixed_k"],
                     row["der_auto"]["der"] * 100, row["der_fixed"]["der"] * 100,
                     "-" if forced is None
                     else "%.2f%%" % (row["der_true_k"]["der"] * 100)), flush=True)
            if args.write_rttm:
                PREDICTIONS.mkdir(parents=True, exist_ok=True)
                write_rttm(PREDICTIONS / f"{meeting}.fixed.full.rttm",
                           seg_new, meeting)
            rows.append(row)
            _write(out, args, done + rows)
        except Exception as exc:
            print("%-14s  FAILED %s: %s" % (meeting, type(exc).__name__, exc), flush=True)
            rows.append({"meeting": meeting, "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in rows if "error" not in r]
    if ok:
        print("\n平均（%d/%d 场）" % (len(ok), len(rows)))
        for key, name in (("der_auto", "现状"), ("der_fixed", "修法")):
            print("  %s DER = %6.2f%%" % (name, float(np.mean(
                [r[key]["der"] for r in ok])) * 100))
        print("  -> 修法增益 = %+.2f pts" % ((float(np.mean([r["der_fixed"]["der"] for r in ok]))
                                            - float(np.mean([r["der_auto"]["der"] for r in ok]))) * 100))
        print("  auto K 命中真值: %d/%d ; 修法后: %d/%d"
              % (sum(1 for r in ok if r["auto_k"] == r["true_k"]), len(ok),
                 sum(1 for r in ok if r["fixed_k"] == r["true_k"]), len(ok)))
        print("\n按真值人数分组：")
        for size in sorted({r["true_k"] for r in ok}):
            group = [r for r in ok if r["true_k"] == size]
            print("  %d 人场（%d 场）: 现状 %6.2f%% / 修法 %6.2f%%  命中 %d/%d -> %d/%d"
                  % (size, len(group),
                     float(np.mean([r["der_auto"]["der"] for r in group])) * 100,
                     float(np.mean([r["der_fixed"]["der"] for r in group])) * 100,
                     sum(1 for r in group if r["auto_k"] == r["true_k"]), len(group),
                     sum(1 for r in group if r["fixed_k"] == r["true_k"]), len(group)))

    _write(out, args, done + rows)
    print("\n已写出 %s" % out)


def _selfcheck():
    """合并逻辑自检（不加载模型、秒级）。

    判别性：4 个 100 窗的大簇 + 1 个贴着簇 0 的 3 窗碎簇。
    阈值 10 时必须合并成 4 簇且碎簇落到最近的大簇；阈值 1 时必须保持 5 簇
    （反向对照，挡「无条件合并」这种写死）。
    """
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    blocks = [centers[i] + 0.01 * rng.normal(size=(100, 16)).astype(np.float32)
              for i in range(4)]
    tiny = centers[0] + 0.02 * rng.normal(size=(3, 16)).astype(np.float32)
    E = np.vstack(blocks + [tiny])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 3)

    merged = merge_tiny_clusters(E, labels, min_size=10)
    n_merged = len(set(merged.tolist()))
    assert n_merged == 4, "碎簇没被合并，仍是 %d 簇" % n_merged
    assert set(merged[-3:].tolist()) == {merged[0]}, "碎簇没并到余弦最近的大簇"

    keep = merge_tiny_clusters(E, labels, min_size=1)
    n_keep = len(set(keep.tolist()))
    assert n_keep == 5, "阈值 1 不该合并，却变成 %d 簇" % n_keep
    print("selfcheck OK：碎簇合并成 4 簇且落到最近大簇；阈值 1 时保持 5 簇")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
