# -*- coding: utf-8 -*-
"""步骤 A：12 场上比对「修复后的生产 `diarize`」与「A/B 探针的 base」。

## 回答什么问题

`977fc6b` 把 `v12.diarize` 的候选 K 空间从 1..4 恢复到 1..10。问题是：
**这个修复会不会改变每场的 base 选择？** 若不变，此前在 v12 生产路径上出的
指标仍有效，A/B 的 base 也不必重算；若变了，旧基线作废。

判据要**逐项**比，不能只看"成绩接近"——成绩接近可以是两处错误互相抵消：
  * `k` 相同
  * 路径级别（生产 `refined`/`rescued` vs 探针 `level`）
  * 逐句预测**逐句**相同（决定归属率是否可能不同）
  * 归属率（`per_sentence` / `duration_weighted`）若前几项都同则必然同

## 已知差异（不许当成"不等价"来报）

探针 `real_path_labels` 在 `k < 2` 时**直接返回主窗标签**，不做救援；
而生产 `diarize` 在 `not refined and k==1` 时会重提 0.6s 窗救援。
所以 `diarize.rescued=True` 的场次，**探针结构上不可能给出同一结果**——
这类场次要单列，不能算作"修复引起的差异"。

    python -u -B scripts/evaluation/probe_step_a_baseline.py --limit 3
    python -u -B scripts/evaluation/probe_step_a_baseline.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_path_equiv as E  # noqa: E402
import probe_ab_dev as ab  # noqa: E402

AB_RESULTS = ROOT / "artifacts" / "v13_ab_dev_results.json"


def level_of(result):
    """生产 `diarize` 结果 → 探针的 `level` 口径（只有两个值）。

    `rescued` 也是 0.6s 窗，但探针**没有**救援分支，故单独标记，
    不与 `refined` 混为一谈。
    """
    if result["rescued"]:
        return "rescue(探针无此分支)"
    return "refined" if result["refined"] else "2s-only"


def run_one(meeting):
    """跑修复后的生产 `diarize`，返回与探针 base 可比的量。"""
    result, calls = E.run_diarize(meeting)
    sentences = E.load_sentences(meeting)
    scored = ab.score(sentences, result["labels"], result["times"],
                      result["window_seconds"])
    # 主窗数（探针 `n_windows` 的同义量）：两边必须相同，否则是窗集不同源，
    # 后面所有比较都无意义。从生产门缓存直接读，不经 `diarize` 的最终态。
    main_emb, _t = E.cached_stage(meeting, "main")
    return {
        "k": int(result["k"]),
        "level": level_of(result),
        "refined": bool(result["refined"]), "rescued": bool(result["rescued"]),
        "n_segments": len(result["segments"]),
        # ⚠️ 重命名：`diarize` 的 `n_windows` 是**最终**（细化后）窗数，
        # 而 A/B 探针的 `n_windows` 是**主窗**数——同名不同义。
        # 例如 S_R004S01C01：diarize 3409（细化）vs 探针 513（主窗）。
        "n_windows_final": int(result["n_windows"]),
        "n_windows_main": int(len(main_emb)),
        "window_seconds": result["window_seconds"],
        "candidate_ks": [int(c["k"]) for c in result["candidates"]],
        "eligible_ks": sorted(int(c["k"]) for c in result["candidates"]
                              if c["eligible"] and c["k"] >= 2),
        "calls": calls,
        **scored,
    }


def compare(meeting, diag, probe_base, probe_row):
    """逐项比对。返回 (结论标签, 差异说明列表)。

    `probe_base` 是 A/B 结果里的 `candidates["base"]`（**只有指标字段**）；
    主窗数在它的**兄弟层**（行上）`row["n_windows"]`，故单独传 `probe_row`。
    """
    notes = []
    if probe_base is None:
        return "无探针参照", ["A/B 产物里没有这场的 base"]

    # 硬判据：两边主窗数必须相同，否则窗集不同源，后面全部比较无意义。
    # （探针的 `n_windows` 是主窗数，与 `diarize` 的最终窗数同名不同义，
    #  故这里比的是 diag["n_windows_main"]。）
    want_main = probe_row.get("n_windows")
    if diag["n_windows_main"] != want_main:
        return "★ 窗集不同源", [
            f"主窗数不同：生产 {diag['n_windows_main']} vs 探针 {want_main}"
            "——两边不是同一个窗集，本场不可比"]

    if diag["k"] != probe_base["k"]:
        notes.append(f"k 不同：生产 {diag['k']} vs 探针 {probe_base['k']}")

    # 路径级别：探针没有 rescue，故 rescue 场次只报不比。
    # ⚠️ 已累积的 notes 必须随结论一起返回：只 return 一句硬编码说明会把
    # "k 不同"整个丢掉（本场既跑了救援、k 又不同时，那才是真差异）。
    if diag["level"].startswith("rescue"):
        return "探针无救援分支（已知差异）", notes + [
            "生产 rescued=True，探针在 k<2 时不救援——结构上不可比"]

    probe_level = probe_base.get("level")
    if diag["level"] != probe_level:
        notes.append(f"level 不同：生产 {diag['level']} vs 探针 {probe_level}")

    # 逐句预测不一定能比：A/B 产物没存 pred，只存了聚合量。
    # 退而比聚合量——但若 k 与 level 都同，逐句预测必然同（同一 labels+times）。
    for field in ("per_sentence", "duration_weighted"):
        got, want = diag.get(field), probe_base.get(field)
        if got is None or want is None:
            continue
        if abs(got - want) > 5e-4:
            notes.append(f"{field} 不同：{got} vs {want}（差 {100*(got-want):+.2f} pts）")

    if not notes:
        return "行为保持", []
    return "★ 有差异", notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 场（调试用）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ab_rows = {r["meeting"]: r for r in
               json.loads(AB_RESULTS.read_text(encoding="utf-8"))["results"]}
    meetings = sorted(ab_rows)
    if args.limit:
        meetings = meetings[:args.limit]

    print(f"{'会议':<16}{'生产 k':>6}{'探针 k':>6}{'生产级别':>12}{'探针级别':>10}"
          f"{'归属率':>9}{'结论':>16}")
    rows, changed, rescue_only, uncomparable, no_reference = [], [], [], [], []
    for meeting in meetings:
        diag = run_one(meeting)
        probe_base = (ab_rows[meeting].get("candidates") or {}).get("base")
        verdict, notes = compare(meeting, diag, probe_base, ab_rows[meeting])
        row = {"meeting": meeting, "diarize": diag,
               "probe_base": probe_base, "verdict": verdict, "notes": notes}
        rows.append(row)
        if verdict.startswith("★ 窗集"):
            uncomparable.append(row)
        elif verdict.startswith("★"):
            changed.append(row)
        elif verdict.startswith("探针无"):
            rescue_only.append(row)
        elif verdict.startswith("无探针参照"):
            # 没跑过比对 ≠ 行为保持：不能让它混进"行为保持"的分子。
            no_reference.append(row)
        pct = diag.get("per_sentence")
        print(f"{meeting:<16}{diag['k']:>6}{(probe_base or {}).get('k', '?'):>6}"
              f"{diag['level']:>12}{str((probe_base or {}).get('level')):>10}"
              f"{(f'{100*pct:.1f}%' if pct is not None else '-'):>9}{verdict:>16}")

    n = len(rows)
    print(f"\n{'='*90}")
    print(f"行为保持 {n - len(changed) - len(rescue_only) - len(uncomparable) - len(no_reference)}/{n}"
          f"   有差异 {len(changed)}/{n}"
          f"   窗集不同源 {len(uncomparable)}/{n}"
          f"   探针无救援分支 {len(rescue_only)}/{n}"
          f"   无探针参照 {len(no_reference)}/{n}")
    if uncomparable:
        print("\n★ 窗集不同源（两边不是同一个窗集，本场不可比）：")
        for r in uncomparable:
            print(f"  - {r['meeting']}: {'; '.join(r['notes'])}")
    if changed:
        print("\n★ 修复改变了这些场次的 base（旧基线在这些场作废）：")
        for r in changed:
            print(f"  - {r['meeting']}: {'; '.join(r['notes'])}")
    if rescue_only:
        print("\n以下场次生产走了救援、探针结构上没有该分支（**不算差异**）：")
        for r in rescue_only:
            print(f"  - {r['meeting']}: {'; '.join(r['notes'])}")
    if no_reference:
        print("\n⚠️ 以下场次在 A/B 产物里没有 base，**未做任何比对**"
              "（不是「行为保持」）：")
        for r in no_reference:
            print(f"  - {r['meeting']}: {'; '.join(r['notes'])}")

    if args.out:
        (ROOT / "artifacts" / args.out).write_text(
            json.dumps({"schema": "v13-step-a-baseline-v1", "rows": rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
