# -*- coding: utf-8 -*-
"""门限准入诊断：候选是被**门限**拦住，还是合格但**选择目标**选错，还是候选本身碎。

## 三个诊断

* **A/B/C 区分**（核心）
  * **A 门限挡住潜在稳定候选** → 放宽后被放行的 k 有稳定簇（每簇 >=2 窗、
    窗级纯度高、每个有效真值说话人有自己的主簇）
  * **B 候选已合格但选择目标选错** → 该 k 在现行门限下 `eligible=True`
    却不是 `_select_candidate` 选中的那个，且它的句级 oracle 更高
  * **C 候选本身碎簇/embedding 结构** → 放行后新增簇多为 1–2 窗，
    或纯度低，或没有真值人占主簇

## 用法

    python -u -B scripts/evaluation/probe_gate_diag.py --selfcheck
    python -u -B scripts/evaluation/probe_gate_diag.py --sweep L_R003S01C02
    python -u -B scripts/evaluation/probe_gate_diag.py --target M_R003S05C01
    python -u -B scripts/evaluation/probe_gate_diag.py --control S_R004S01C01

**不改生产**：门限常量用局部 monkeypatch + `finally` 还原；`_candidate_clusters`
与 `_select_candidate` 都**调用时**读模块全局，所以改常量真的生效
（这正是本项目 bug 族里那条"def 时绑定"陷阱的反面。
⚠️ `max_speakers` 是唯一的 def 时绑定项，必须**显式传**）。
"""
import argparse
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import probe_path_equiv as E  # noqa: E402
from der import read_rttm  # noqa: E402
from probe_gap_decomp import rttm_speech  # noqa: E402

CAP = 10
WINDOWS_GRID = [1, 2, 3, 4, 6, 8]
SHARE_GRID = [0.0, 0.002, 0.005, 0.010, 0.015, 0.025, 0.040]
DEFAULT_WINDOWS, DEFAULT_SHARE = 4, 0.015


@contextmanager
def gate(windows, share):
    """临时换两道门限常量。`_candidate_clusters` 调用时读全局，故真的生效。"""
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = windows, share
    try:
        yield
    finally:
        v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = old


def truth_of(meeting):
    return [s for s in read_rttm(P.DATA / "TextGrid" / f"{meeting}.rttm")
            if s["start"] < P.SECONDS]


def effective_speakers(ref, min_share=0.05):
    """有效真值说话人：发言时长占比 >= 5%（与 `probe_selector_locks` 同口径）。"""
    dur = Counter()
    for s in ref:
        dur[s["speaker"]] += s["end"] - s["start"]
    grand = sum(dur.values()) or 1.0
    return sorted(k for k, v in dur.items() if v / grand >= min_share)


def cluster_purity(emb, times, labels, ref, window=P.WIN):
    """窗级纯度：每个簇内，窗的**多数真值说话人**占该簇的比例。

    返回 (整体纯度, 每簇纯度, 每簇主说话人)。与句级指标无关——这里刻意
    不走句子，避免与归属率同源。
    """
    from probe_gap_decomp import window_reference
    per_window = window_reference(ref, times, window)
    purities, homes = [], []
    for label in sorted(set(int(x) for x in labels)):
        mask = labels == label
        names = [str(n) for n in per_window[mask] if str(n) != ""]
        if not names:
            purities.append(None)
            homes.append(None)
            continue
        top, count = Counter(names).most_common(1)[0]
        purities.append(count / len(names))
        homes.append(top)
    weighted = [p for p in purities if p is not None]
    overall = float(np.mean(weighted)) if weighted else None
    return overall, purities, homes


def analyze(meeting, ks):
    sents = E.load_sentences(meeting)
    main_emb, main_times = E.cached_stage(meeting, "main")
    ref = truth_of(meeting)
    eff = effective_speakers(ref)
    end = max(s["end"] for s in sents)
    out = {"meeting": meeting, "n_windows": len(main_emb), "n_sentences": len(sents),
           "effective_speakers": eff, "n_effective": len(eff),
           "rows": [], "selected_k": None}
    with gate(DEFAULT_WINDOWS, DEFAULT_SHARE):
        cands, _ = v11._candidate_clusters(main_emb, max_speakers=CAP)
        out["selected_k"] = int(v11._select_candidate(cands)["k"])
    for k in ks:
        with gate(DEFAULT_WINDOWS, DEFAULT_SHARE):
            cands, _ = v11._candidate_clusters(main_emb, max_speakers=CAP)
        c = next((x for x in cands if int(x["k"]) == k), None)
        if c is None:
            continue
        labels = c["labels"]
        sizes = sorted(c["sizes"], reverse=True)
        purity, per_purity, homes = cluster_purity(main_emb, main_times, labels, ref)
        segs = P.segment_map(main_times, labels, np.full(len(main_times), P.WIN), end)
        pred = P.sentence_dict_predictions(sents, segs)
        from probe_mapping_objective import score_oracle
        orc = score_oracle(sents, pred)["overall"]
        mapp = P.attribution(sents, pred)["overall"]
        named_homes = [h for h in homes if h]
        out["rows"].append({
            "k": k, "eligible": bool(c["eligible"]), "sizes": sizes,
            "min_size": min(sizes), "min_share": min(sizes) / len(main_emb),
            "silhouette": round(float(c["score"]), 4),
            "objective": round(float(c["objective"]), 4),
            "purity": purity, "per_cluster_purity": per_purity,
            "homes": named_homes,
            "speakers_with_home": len(set(named_homes)),
            "tiny_clusters": sum(1 for s in sizes if s <= 2),
            "oracle": orc["per_sentence"], "mapped": mapp["per_sentence"],
            "abstain": mapp["abstain"], "unmapped": mapp["unmapped"],
        })
    return out


def fmt(v, w=7, pct=True):
    if v is None:
        return " " * w + ("    -" if pct else "-")
    return f"{100*v:{w}.2f}" if pct else f"{v:{w}}"


def print_analysis(a, title):
    print(f"\n{'='*112}\n{title}\n{'='*112}")
    print(f"场次 {a['meeting']}  主窗 {a['n_windows']}  句子 {a['n_sentences']}  "
          f"有效真值说话人 {a['n_effective']} {a['effective_speakers']}  "
          f"现行选中 k={a['selected_k']}")
    print(f"{'k':>3}{'合格':>6}{'min窗':>7}{'min份额':>9}{'轮廓':>9}{'objective':>11}"
          f"{'句oracle':>10}{'句mapped':>10}{'弃权':>7}{'未匹配':>8}"
          f"{'簇尺寸':>22}{'纯':>7}{'碎<=2':>6}{'有主场':>7}")
    print("-" * 112)
    for r in a["rows"]:
        star = "★" if r["k"] == a["selected_k"] else " "
        print(f"{star}{r['k']:>2}{('是' if r['eligible'] else '否'):>6}"
              f"{r['min_size']:>7}{r['min_share']:>9.4f}{r['silhouette']:>9.4f}"
              f"{r['objective']:>11.4f}{fmt(r['oracle'],8)}{fmt(r['mapped'],8)}"
              f"{fmt(r['abstain'],6)}{fmt(r['unmapped'],7)}"
              f"{str(r['sizes']):>22}{fmt(r['purity'],6)}"
              f"{r['tiny_clusters']:>6}{r['speakers_with_home']:>6}/{a['n_effective']}")


def sweep(meeting):
    """两门限**联合**扫描（任务一格式）：每格记放行的 k 集合与选中 k。"""
    print(f"\n{'='*112}\n两门限联合扫描 {meeting}"
          f"（行=MIN_CLUSTER_WINDOWS，列=MIN_CLUSTER_SHARE）\n{'='*112}")
    print("每格：放行的合格 k 集合 / 现行选择器选中的 k（k=1 表示全被拦）")
    head = "".join(f"{s:>14}" for s in SHARE_GRID)
    print(f"{'窗门\\份额门':<10}{head}")
    print("-" * (10 + 14 * len(SHARE_GRID)))
    for w in WINDOWS_GRID:
        cells = []
        for s in SHARE_GRID:
            with gate(w, s):
                emb, _t = E.cached_stage(meeting, "main")
                cands, _ = v11._candidate_clusters(emb, max_speakers=CAP)
                sel = int(v11._select_candidate(cands)["k"])
            elig = sorted(int(c["k"]) for c in cands if c["eligible"] and c["k"] >= 2)
            tag = ",".join(map(str, elig)) if elig else "空"
            cells.append(f"{tag}->{sel}")
        print(f"{w:<10}" + "".join(f"{c:>14}" for c in cells))
    print("\n注：'空->1' 表示该格下没有任何合格候选，选择器回退 k=1。")


def selfcheck():
    """最小自检：门限常量真的生效、且还原干净。"""
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    with gate(99, 0.5):
        assert (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE) == (99, 0.5), "门限未生效"
        emb, _t = E.cached_stage("M_R003S05C01", "main")
        cands, _ = v11._candidate_clusters(emb, max_speakers=CAP)
        assert all(not c["eligible"] for c in cands if c["k"] >= 2), \
            "窗门=99 时不该有合格候选"
    assert (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE) == old, "门限未还原"
    with gate(4, 0.015):
        emb, _t = E.cached_stage("M_R003S05C01", "main")
        cands, _ = v11._candidate_clusters(emb, max_speakers=CAP)
        elig = [int(c["k"]) for c in cands if c["eligible"] and c["k"] >= 2]
        assert elig == [2, 3], f"M_R003S05C01 默认门限下合格集应为 [2,3]，实得 {elig}"
    assert (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE) == old, "门限未还原"
    print("自检通过：门限生效且还原干净；M_R003S05C01 默认合格集 = [2, 3]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--control", default="")
    ap.add_argument("--ks", default="2,3,4,5,6")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    ks = [int(x) for x in args.ks.split(",")]
    if args.sweep:
        sweep(args.sweep)
        print_analysis(analyze(args.sweep, ks),
                       f"任务一：{args.sweep} 门限准入诊断（现行门限下的逐 k 明细）")
    if args.target:
        print_analysis(analyze(args.target, ks),
                       f"任务二：{args.target} 选择目标诊断（**不改门限**）")
    if args.control:
        print_analysis(analyze(args.control, ks),
                       f"控制场：{args.control}（base 稳定，查放宽后是否过分割）")
        sweep(args.control)
    if not any([args.sweep, args.target, args.control]):
        ap.print_help()


if __name__ == "__main__":
    main()
