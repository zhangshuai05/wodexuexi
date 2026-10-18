# -*- coding: utf-8 -*-
"""稳定性评分 vs silhouette：候选选择对照（**诊断，不接生产**）。

对每场每个 K 记录：silhouette / objective / eligible / 实际簇数 /
细化后句级四类 / **稳定性评分**，再比较几条选择规则的**选择损失**
（相对逐场 K 网格最佳候选 —— 读真值，仅作诊断上界）。

## 两种门分开报

* `--gate rttm` —— 语音段取自 RTTM 并集。与 `v13_candidate_score_table_*`
  同一口径，便于对齐已有结论。
* `--gate vad`  —— 语音段取自 **VAD**（探针缓存 manifest 里建库时那份）。
  这是**部署时真正拿得到**的东西。

⚠️ 两者不可互换（见坑 W4），产物里显式标 `gate`，不混着报。

## 候选评分函数读什么

只读：主窗聚类结果、0.6 秒细化结果、VAD/ASR 可获得的**语音段**、预测标签与边界。
不读：RTTM 人名、TextGrid 句界、真值人数、匈牙利映射、句级真值。
真值**只**出现在诊断字段（`attribution_error` 及选择损失）。

    python -u -B scripts/evaluation/probe_stability_selector.py --gate rttm --split dev
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
import probe_ab_dev as AB  # noqa: E402
import probe_backend_sweep as BS  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames, rttm_speech)
from probe_selector_locks import silhouette_core  # noqa: E402
from segment_stability import stability_score  # noqa: E402
from unlocked_selector import CAP, MIN_WIN  # noqa: E402

PROBE_CACHE = Path(BS.DEFAULT_CACHE) / "probe"


def _git_head():
    """当前提交号。**动态取** —— 硬编码会在下一次提交后静默过期，
    而产物里的 "code.head" 是给人复算用的，报错了比不报更坑。
    仓库外 / 无 git 时退回 "unknown"（不让它把整个扫描搞崩）。"""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(ROOT), capture_output=True, text=True,
                             timeout=10)
        head = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=str(ROOT), capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return (head or "unknown") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def vad_segments(meeting, seconds):
    """部署时可得：VAD 语音段（建缓存时那份，记在 probe manifest 里）。"""
    payload = json.loads((PROBE_CACHE / "manifest.json").read_text(encoding="utf-8"))
    return [(a, min(b, seconds)) for a, b in
            payload["meetings"][meeting]["vad_intervals"] if a < seconds]


def raw_candidates(emb, cap):
    """原始空间候选：与 `unlocked_selector.build_unlocked` 逐字同法（余弦 + average）。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    n = len(emb)
    if n < 2:
        return []
    tree = linkage(emb, method="average", metric="cosine")
    out = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
            "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(cap, n) + 1):
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == x).sum()) for x in range(k)]
        core = [s for s in sizes if s >= 2]
        eligible = len(core) >= 2 and min(core) >= MIN_WIN
        score = silhouette_core(emb, labels) if eligible else -1.0
        out.append({"k": k, "labels": labels, "score": score,
                    "objective": score - v11.MULTI_K_PENALTY * (k - 2),
                    "sizes": sizes, "eligible": eligible})
    return out


def meeting_rows(meeting, cache, root, cap, seconds, share, gate):
    rE, rtimes = PE.cached_stage(meeting, "refine")
    reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
    limit = seconds if seconds > 0 else max((s["end"] for s in reference), default=0.0)
    reference = [{"start": s["start"], "end": min(s["end"], limit),
                  "speaker": s["speaker"]} for s in reference if s["start"] < limit]
    by_speaker = {}
    for s in reference:
        by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0.0) + s["end"] - s["start"]
    grand = sum(by_speaker.values()) or 1.0
    effective = sorted(n for n, v in by_speaker.items() if v / grand >= share)

    emb_all, times_all = cached_windows(cache, meeting, v12.WIN_S, v12.HOP_S)
    speech = (rttm_speech(reference, seconds=limit) if gate == "rttm"
              else vad_segments(meeting, limit))
    emb, times = gate_windows(emb_all, times_all, speech, v12.WIN_S, v11.MIN_VAD_FRACTION)
    sents = AB.load_sentences(meeting)
    if len(emb) < 2 or not sents:
        return None, "窗数或句数不足"

    rows = []
    for cand in raw_candidates(emb, cap):
        k = int(cand["k"])
        fclusters, ftimes, level = BS.refine_labels(
            np.asarray(cand["labels"]), emb, times, rE, rtimes, True)
        fwin = v12.WIN_S if level == "none" else AB._win(level)
        # 评分函数只见这些：细化标签、细化时刻、窗长、语音段。无真值。
        stab, sdetail = stability_score(fclusters, ftimes, fwin, speech)
        sent = BS.sentence_metrics(sents, fclusters, ftimes, fwin)["clean"]
        rows.append({
            "k": k, "score": float(cand["score"]),
            "objective": float(cand["objective"]),
            "eligible": bool(cand["eligible"]),
            "sizes": [int(x) for x in cand["sizes"]],
            "n_clusters_refined": int(len(set(int(x) for x in fclusters))),
            "stability": stab,
            "stability_detail": sdetail,
            "level": level,
            "attribution_error": sent["attribution_error"],
            "abstain": sent["abstain"], "wrong": sent["wrong"],
            "unmapped": sent["unmapped"], "per_sentence": sent["per_sentence"],
        })
    return {"meeting": meeting, "true_k": len(effective),
            "n_windows": int(len(emb)), "n_segments": len(speech),
            "rows": rows}, None


# ───────────────────── 选择规则（全部只读无真值字段） ─────────────────────

def _elig(rows):
    return [r for r in rows if r["eligible"] and r["k"] >= 2]


def sel_objective(rows):
    e = _elig(rows)
    return max(e, key=lambda r: (r["objective"], -r["k"])) if e else rows[0]


def sel_silhouette(rows):
    e = _elig(rows)
    return max(e, key=lambda r: (r["score"], -r["k"])) if e else rows[0]


def sel_largest(rows):
    e = _elig(rows)
    return max(e, key=lambda r: r["k"]) if e else rows[0]


def sel_stability(rows):
    e = [r for r in _elig(rows) if r["stability"] is not None]
    return max(e, key=lambda r: (r["stability"], -r["k"])) if e else rows[0]


def sel_oracle(rows):
    """逐场用真值挑最优 —— **只作诊断上界**，不是选择器。"""
    e = _elig(rows) or rows
    return min(e, key=lambda r: r["attribution_error"])


SELECTORS = {
    "生产 objective(λ=0.005)": sel_objective,
    "纯 silhouette": sel_silhouette,
    "最大合格 K": sel_largest,
    "新评分 stability": sel_stability,
    "网格最佳（读真值·上界）": sel_oracle,
}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return None if den == 0 else float((ra * rb).sum() / den)


def report(meetings, gate, cap, split):
    print(f"\n=== 逐场选择（gate={gate} cap={cap}） ===")
    hdr = f"{'场次':16}" + "".join(f"{n:>22}" for n in SELECTORS)
    print(hdr)
    table = {}
    for item in meetings:
        rows = item["rows"]
        cells, chosen = [], {}
        for name, fn in SELECTORS.items():
            r = fn(rows)
            chosen[name] = r
            cells.append(f"  k={r['k']:>2} {r['attribution_error'] * 100:5.1f}%")
        table[item["meeting"]] = chosen
        print(f"{item['meeting']:16}" + "".join(cells))

    print(f"\n=== 汇总（{len(meetings)} 场，gate={gate}，cap={cap}） ===")
    oracle_mean = None
    print(f"  {'选择规则':22}{'平均错误率':>12}{'选择损失':>10}"
          f"{'赢/亏/平':>12}{'最差场':>26}")
    stats = {}
    for name, fn in SELECTORS.items():
        errs = [fn(it["rows"])["attribution_error"] for it in meetings]
        mean = float(np.mean(errs))
        if name.startswith("网格最佳"):
            oracle_mean = mean
        losses = [fn(it["rows"])["attribution_error"]
                  - sel_oracle(it["rows"])["attribution_error"] for it in meetings]
        wins = sum(1 for x in losses if x < -1e-9)
        bad = sum(1 for x in losses if x > 1e-9)
        worst = max(zip([it["meeting"] for it in meetings], losses),
                    key=lambda kv: kv[1])
        stats[name] = {"mean": mean, "loss": float(np.mean(losses)),
                       "wins": wins, "losses": bad, "ties": len(losses) - wins - bad,
                       "worst": [worst[0], float(worst[1])]}
        print(f"  {name:22}{mean * 100:11.2f}%{np.mean(losses) * 100:9.2f}pt"
              f"{f'{wins}/{bad}/{len(losses) - wins - bad}':>12}"
              f"{f'{worst[0]} {worst[1] * 100:+.1f}pt':>26}")
    print(f"\n  诊断上界（读真值）= {oracle_mean * 100:.2f}%")

    # 场内相关：才是选 K 真正依赖的东西
    print(f"\n=== 场内相关（gate={gate} cap={cap}，K≥2 合格） ===")
    for key, label in (("score", "silhouette"), ("objective", "objective"),
                       ("stability", "stability")):
        per = []
        for it in meetings:
            rs = _elig(it["rows"])
            rs = [r for r in rs if r[key] is not None]
            if len(rs) >= 3:
                per.append(spearman([r[key] for r in rs],
                                    [r["attribution_error"] for r in rs]))
        per = [x for x in per if x is not None]
        hits = sum(1 for it in meetings
                   if (lambda rs: (rs and abs(max(rs, key=lambda r: (r[key], -r['k']))
                                              ['attribution_error']
                                              - min(rs, key=lambda r: r['attribution_error'])
                                              ['attribution_error']) < 1e-9))(
                       [r for r in _elig(it["rows"]) if r[key] is not None]))
        print(f"  {label:12} 场内 ρ 中位 {np.median(per):+.3f} "
              f"（负 {(np.array(per) < 0).sum()}/{len(per)} 场）  "
              f"命中最低错误率 {hits}/{len(meetings)} 场")

    # CAP 依赖：选中 K 的分布 + 与 cap 相邻的表现
    print(f"\n=== CAP 依赖检查（cap={cap}） ===")
    for name, fn in SELECTORS.items():
        ks = [fn(it["rows"])["k"] for it in meetings]
        at_cap = sum(1 for k in ks if k == cap)
        print(f"  {name:22} 选中 K={sorted(ks)}  等于cap 的场次 {at_cap}/{len(ks)}")

    # 两场坏场
    print(f"\n=== 指定场次逐 K（gate={gate}） ===")
    for target in ("L_R003S01C02", "M_R003S04C01"):
        item = next((it for it in meetings if it["meeting"] == target), None)
        if not item:
            print(f"  {target}: 未跑")
            continue
        print(f"  {target}:")
        print(f"    {'K':>3}{'sil':>10}{'eligible':>10}{'stability':>11}"
              f"{'错误率':>9}{'弃权':>8}")
        for r in item["rows"]:
            st = f"{r['stability']:9.4f}" if r["stability"] is not None else "     None"
            print(f"    {r['k']:>3}{r['score']:10.4f}{str(r['eligible']):>10}{st}"
                  f"{r['attribution_error'] * 100:8.2f}%{r['abstain'] * 100:7.1f}%")
    return stats, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=BS.DEFAULT_CACHE)
    ap.add_argument("--split", choices=("dev",), default="dev",
                    help="只允许 dev —— holdout 已污染，双绿不构成独立验证")
    ap.add_argument("--gate", choices=("rttm", "vad"), default="rttm")
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

    root = Path(args.root)
    AB.DATA = root
    meetings = args.meetings or split_meetings(root, args.split)

    table, skipped = [], []
    for meeting in meetings:
        try:
            item, why = meeting_rows(meeting, args.cache, root, args.cap,
                                     args.seconds, args.share, args.gate)
        except FileNotFoundError as exc:
            skipped.append(f"{meeting}: {exc}")
            continue
        if item is None:
            skipped.append(f"{meeting}: {why}")
            continue
        table.append(item)

    if not table:
        raise SystemExit(f"无可用场次：{skipped}")
    stats, chosen = report(table, args.gate, args.cap, args.split)

    if args.output:
        Path(args.output).write_text(json.dumps({
            "schema": "v13-selector-stability-diagnosis-v1",
            "gate": args.gate, "split": args.split, "cap": args.cap,
            "code": {
                "driver": "scripts/evaluation/probe_stability_selector.py",
                "scoring": "scripts/evaluation/segment_stability.py",
                "reuses": "probe_backend_sweep.refine_labels/sentence_metrics",
                "head": _git_head(),
            },
            "truth_used": {
                "in_scoring_function": False,
                "in_diagnostics": ["attribution_error 及由其算出的选择损失/上界"],
                "note": "评分函数只吃 细化标签 / 细化时刻 / 窗长 / 语音段；"
                        "真值只在诊断字段。网格最佳是逐场用真值挑的 ⇒ 诊断上界。",
            },
            "stability_signal": {
                "formula": "A=(purity_obs-purity_null)/(1-purity_null)；"
                           "score = mean(A over {δ=0} ∪ 3 次边界抖动)；"
                           "instability = 该集合上 A 的极差",
                "purity": "按窗数加权的段内主导标签占比",
                "null": "同一划分下随机置换标签后的占比（保留簇/段大小）",
                "degeneracies": {
                    "K=1": "purity_obs=purity_null=1 ⇒ A≡0，不拿最高分",
                    "CAP": "A 上界为 1，多切簇不白拿分；实测见 cap 扫描",
                    "边界税": "抖动把边界窗推去邻段 ⇒ 正确而锐利的边界也掉分，"
                              "税随 K 增大，与纯度项方向相反（已实测，非实现 bug）",
                },
                "known_limitation": "扰动项**不**区分『错在边界』与『错在段中』——"
                                    "合成数据已实测推翻该设计直觉（见 selfcheck ④）。",
            },
            "selectors": stats, "chosen": chosen,
            "meetings": table, "skipped": skipped,
        }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"\n写入 {args.output}")
    if skipped:
        print(f"⚠ 跳过 {len(skipped)} 场: {skipped}")


if __name__ == "__main__":
    main()
