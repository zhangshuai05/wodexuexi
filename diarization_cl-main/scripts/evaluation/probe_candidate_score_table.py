# -*- coding: utf-8 -*-
"""候选评分表：逐场逐 K 的 score / objective / eligible / 簇数 / 句级四类。

## 为什么需要它

`v13_selector_score_diagnosis_4238cb9c.json` 只落了**汇总**与几条规则选中的结果，
没有逐 (场次, K) 的对齐表 —— 于是 ρ=−0.617 这个数**没有人能复算**：

* 样本里含不含 K=1？（`build_unlocked` 的 k=1 是 `score=0.0` 的退化候选，
  放进相关里就是一个"最低分"的点，会把负相关**做出来**）
* 含不含不合格候选？（不合格的 `score` 恒为 **-1.0 哨兵值**，不是轮廓系数，
  拿它算秩相关等于把哨兵当测量值）
* 是否只用 eligible 且 K≥2？

本脚本把表补齐，并把上面三种口径的相关**并排算出来** —— 结论随口径变就是口径问题，
不随口径变才是信号。

## 口径

与 `probe_backend_sweep.py` **同源**：直接 import 它的 `sentence_metrics` /
`refine_labels`，复用 `probe_gap_decomp` 的门控与 `build_unlocked` 的候选构造。
不另立一套，避免坑 A7（两条路径不同构）。

    python -u -B scripts/evaluation/probe_candidate_score_table.py --split dev
"""
import argparse
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
from unlocked_selector import CAP, MIN_WIN, build_unlocked  # noqa: E402


def meeting_rows(meeting, cache, root, cap, seconds, share, window, hop):
    """一场 × K=1..cap 的完整行。RTTM 门（与台子一致）。"""
    rE, rtimes = PE.cached_stage(meeting, "refine")
    reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
    seconds = seconds if seconds > 0 else max(
        (s["end"] for s in reference), default=0.0)
    reference = [{"start": s["start"], "end": min(s["end"], seconds),
                  "speaker": s["speaker"]} for s in reference if s["start"] < seconds]
    by_speaker = {}
    for s in reference:
        by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0.0) + s["end"] - s["start"]
    grand = sum(by_speaker.values()) or 1.0
    effective = sorted(n for n, v in by_speaker.items() if v / grand >= share)

    emb_all, times_all = cached_windows(cache, meeting, window, hop)
    emb, times = gate_windows(emb_all, times_all, rttm_speech(reference, seconds=seconds),
                              window, v11.MIN_VAD_FRACTION)
    sents = AB.load_sentences(meeting)
    if len(emb) < 2 or not sents:
        return None

    rows = []
    for cand in build_unlocked(emb, cap, MIN_WIN):
        k = int(cand["k"])
        if k == 1:
            # k=1 是退化候选：单簇，`silhouette_core` 直接 -1.0，而
            # `build_unlocked` 给它 `score=0.0, eligible=True`。两种写法都不是
            # 真轮廓系数 —— 标出来，别让它冒充测量值。
            fclusters, ftimes, level = AB.real_path_labels(
                np.asarray(cand["labels"]), emb, times, rE, rtimes)
            fwin = v12.WIN_S if level == "none" else AB._win(level)
        else:
            fclusters, ftimes, level = BS.refine_labels(
                np.asarray(cand["labels"]), emb, times, rE, rtimes, True)
            fwin = v12.WIN_S if level == "none" else AB._win(level)
        named = map_by_frames(reference, ftimes, fclusters, np.full(len(ftimes), fwin))
        sent = BS.sentence_metrics(sents, fclusters, ftimes, fwin)
        clean = sent["clean"]
        rows.append({
            "meeting": meeting, "k": k, "k_requested": k,
            "n_clusters_raw": int(len(set(int(x) for x in cand["labels"]))),
            "n_clusters_refined": int(len(set(int(x) for x in fclusters))),
            "sizes": [int(x) for x in cand["sizes"]],
            "eligible": bool(cand["eligible"]),
            "score": float(cand["score"]),
            "objective": float(cand["objective"]),
            "level": level,
            "frame_accuracy": float(frame_accuracy(
                reference, ftimes, named, np.full(len(ftimes), fwin))["accuracy"]),
            "per_sentence": clean["per_sentence"],
            "wrong": clean["wrong"],
            "unmapped": clean["unmapped"],
            "abstain": clean["abstain"],
            "attribution_error": clean["attribution_error"],
            "n_sentences": clean["n"],
        })
    return {"meeting": meeting, "true_k": len(effective), "n_windows": int(len(emb)),
            "rows": rows}


def spearman(a, b):
    """秩相关。并列取平均秩（本表里有大量并列 —— 见 `tie_ratio`）。"""
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
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return None if denom == 0 else float((ra * rb).sum() / denom)


def tie_ratio(values):
    """与前一个值相同的比例 —— 并列多少会直接决定秩相关的可信度。"""
    v = np.asarray(values, float)
    return float((v[1:] == v[:-1]).mean()) if len(v) > 1 else 0.0


def summarize(all_rows):
    """三种口径并排。差异 = 口径问题；一致 = 信号。"""
    out = {}
    defs = {
        "all_含K1_含不合格": lambda r: True,
        "K>=2_含不合格": lambda r: r["k"] >= 2,
        "K>=2_仅合格": lambda r: r["k"] >= 2 and r["eligible"],
    }
    for name, keep in defs.items():
        sel = [r for r in all_rows if keep(r)]
        if len(sel) < 3:
            out[name] = {"n": len(sel), "rho": None}
            continue
        scores = [r["score"] for r in sel]
        errs = [r["attribution_error"] for r in sel]
        out[name] = {
            "n": len(sel),
            "rho_score_vs_error": spearman(scores, errs),
            "rho_objective_vs_error": spearman([r["objective"] for r in sel], errs),
            "score_min": float(np.min(scores)), "score_max": float(np.max(scores)),
            "n_sentinel_-1": int(sum(1 for x in scores if x == -1.0)),
            "err_min": float(np.min(errs)), "err_max": float(np.max(errs)),
            "score_tie_ratio": tie_ratio(scores),
            "err_tie_ratio": tie_ratio(errs),
        }
    # 场内跨度（各场内部 max-min），两种量纲都报
    per_meeting = {}
    for m in sorted({r["meeting"] for r in all_rows}):
        rs = [r for r in all_rows if r["meeting"] == m and r["k"] >= 2]
        elig = [r for r in rs if r["eligible"]]
        if not rs:
            continue
        per_meeting[m] = {
            "score_span": float(max(r["score"] for r in rs) - min(r["score"] for r in rs)),
            "err_span_points": float(max(r["attribution_error"] for r in rs)
                                     - min(r["attribution_error"] for r in rs)),
            "err_span_eligible_points": (
                float(max(r["attribution_error"] for r in elig)
                      - min(r["attribution_error"] for r in elig)) if elig else None),
        }
    out["_span"] = {
        "score_span_min": min(v["score_span"] for v in per_meeting.values()),
        "score_span_max": max(v["score_span"] for v in per_meeting.values()),
        "err_span_min_points": min(v["err_span_points"] for v in per_meeting.values()),
        "err_span_max_points": max(v["err_span_points"] for v in per_meeting.values()),
        "err_span_eligible_min_points": min(
            v["err_span_eligible_points"] for v in per_meeting.values()
            if v["err_span_eligible_points"] is not None),
        "err_span_eligible_max_points": max(
            v["err_span_eligible_points"] for v in per_meeting.values()
            if v["err_span_eligible_points"] is not None),
    }
    out["_per_meeting_span"] = per_meeting
    out["_boundary_validation"] = {
        "all_err_min": min(r["attribution_error"] for r in all_rows),
        "all_err_max": max(r["attribution_error"] for r in all_rows),
        "n_rows": len(all_rows),
        "n_K1_or_ineligible": int(sum(
            1 for r in all_rows if r["k"] == 1 or not r["eligible"])),
    }
    return out


def selfcheck():
    """退化输入：秩相关的并列与单调性、空表不崩。"""
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert spearman([1, 2], [1, 2]) is None            # n<3 返回 None
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None  # 常数列 -> 分母 0
    assert tie_ratio([1, 1, 2, 3, 3]) == 0.5
    assert tie_ratio([5]) == 0.0
    print("selfcheck ✓ 秩相关单调/并列/退化三种情形通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=BS.DEFAULT_CACHE)
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--output", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

    # `BS.sentence_metrics` / `BS.refine_labels` 不读参数，只需 import 成功即可。
    root = Path(args.root)
    AB.DATA = root
    meetings = args.meetings or split_meetings(root, args.split)

    table, skipped = [], []
    for meeting in meetings:
        try:
            item = meeting_rows(meeting, args.cache, root, args.cap,
                                args.seconds, args.share, v12.WIN_S, v12.HOP_S)
        except FileNotFoundError as exc:
            skipped.append(f"{meeting}: {exc}")
            continue
        if item is None:
            skipped.append(f"{meeting}: 窗数或句数不足")
            continue
        table.append(item)
        print(f"[{len(table)}/{len(meetings)}] {meeting} n={item['n_windows']} "
              f"true_k={item['true_k']}", flush=True)

    if not table:
        raise SystemExit(f"无可用场次：{skipped}")

    all_rows = [r for item in table for r in item["rows"]]
    summary = summarize(all_rows)
    print("\n=== 三种口径的秩相关 ===")
    for name, s in summary.items():
        if name.startswith("_"):
            continue
        print(f"  {name:>22}  n={s['n']:>3}  ρ(score,err)={s['rho_score_vs_error']}  "
              f"ρ(obj,err)={s['rho_objective_vs_error']}  "
              f"哨兵-1 个数={s['n_sentinel_-1']}  评分并列率={s['score_tie_ratio']:.2f}  "
              f"错误率并列率={s['err_tie_ratio']:.2f}")
    print("\n=== 场内跨度 ===")
    print(json.dumps(summary["_span"], ensure_ascii=False, indent=1))

    if args.output:
        Path(args.output).write_text(json.dumps({
            "schema": "v13-candidate-score-table-v1",
            "gate": "rttm",
            "split": args.split,
            "cap": args.cap,
            "note": [
                "逐 (场次, K) 完整表 —— 补 `v13_selector_score_diagnosis_*` 缺的那张表，"
                "使 ρ 可复算。",
                "`score` 对 k=1 是 0.0、对不合格候选是 **-1.0 哨兵**，两者都不是轮廓系数；"
                "算相关前必须选口径，本表把三种口径并排报出来。",
                "口径与 `probe_backend_sweep.py` 同源（直接 import 它的 "
                "`sentence_metrics`/`refine_labels`）。gate=rttm，不可与生产 VAD 门直接比。",
            ],
            "summary": summary, "meetings": table, "skipped": skipped,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")
    if skipped:
        print(f"⚠ 跳过 {len(skipped)} 场: {skipped}")


if __name__ == "__main__":
    main()
