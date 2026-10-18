# -*- coding: utf-8 -*-
"""从 `v13_candidate_score_table_dev.json` 重算候选评分诊断，**替代**旧产物。

旧产物 `v13_selector_score_diagnosis_4238cb9c.json` 的两个问题：

1. **没有生成脚本**进仓库 —— 它落下的每个数字都无法复算。
2. 其中两处表述在数学上是错的（符号解释、跨度比较），一处统计口径不明（5/12 vs 7/2/3），
   一处数字查无出处（holdout 的 6.11% / 0.02–0.75 点）。

本脚本只做**读表重算**，不引入任何新实验，因此输出可被任何人从已提交的产物复核。

    python -u -B scripts/evaluation/build_selector_score_diagnosis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "v13_candidate_score_table_dev.json"
OUT = ROOT / "artifacts" / "v13_selector_score_diagnosis_v2.json"


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


def elig(rows):
    return [r for r in rows if r["eligible"] and r["k"] >= 2]


def main():
    table = json.loads(SRC.read_text(encoding="utf-8"))
    meetings = table["meetings"]
    rows = [r for it in meetings for r in it["rows"]]

    # ── 1. ρ 的三种口径（差异 = 口径问题；一致 = 信号） ──
    defs = {
        "含K1_含不合格": lambda r: True,
        "K>=2_含不合格": lambda r: r["k"] >= 2,
        "K>=2_仅合格": lambda r: r["k"] >= 2 and r["eligible"],
    }
    rho = {}
    for name, keep in defs.items():
        sel = [r for r in rows if keep(r)]
        rho[name] = {"n": len(sel),
                     "rho_score_vs_error": spearman([r["score"] for r in sel],
                                                    [r["attribution_error"] for r in sel]),
                     "rho_objective_vs_error": spearman([r["objective"] for r in sel],
                                                        [r["attribution_error"] for r in sel]),
                     "n_sentinel_score_-1": int(sum(1 for r in sel if r["score"] == -1.0))}

    # ── 2. 跨度：三种筛选并排，**并核对旧产物的两个端点出自哪个筛选** ──
    span_filters = {
        "全表_含K1_含不合格": lambda r: True,
        "K>=2_含不合格": lambda r: r["k"] >= 2,
        "K>=2_仅合格": lambda r: r["k"] >= 2 and r["eligible"],
    }
    spans, span_by_filter = [], {}
    for name, keep in span_filters.items():
        per = []
        for it in meetings:
            rs = [r for r in it["rows"] if keep(r)]
            per.append({
                "meeting": it["meeting"],
                "silhouette_span": float(max(r["score"] for r in rs)
                                         - min(r["score"] for r in rs)),
                "error_span_points": float(max(r["attribution_error"] for r in rs)
                                           - min(r["attribution_error"] for r in rs)) * 100,
            })
        span_by_filter[name] = {
            "silhouette_span_min": min(s["silhouette_span"] for s in per),
            "silhouette_span_max": max(s["silhouette_span"] for s in per),
            "error_span_min_points": min(s["error_span_points"] for s in per),
            "error_span_max_points": max(s["error_span_points"] for s in per),
            "per_meeting": per,
        }
    spans = span_by_filter["K>=2_仅合格"]["per_meeting"]
    span_summary = {
        "by_filter": span_by_filter,
        "old_artifact_claimed": {"silhouette": "0.049-0.279",
                                 "error_points": "1.3-54.0"},
        "which_filter_matches_old_artifact": (
            "**K≥2 且合格** —— 0.0486–0.2788 与 1.32–53.97。"
            "这不是推测：含 K=1 会让 silhouette 下界变成 0.0，含不合格会让下界变成 "
            "-1.0（哨兵值），两者都不会给出 0.049。"
        ),
        "reading": (
            "① **旧产物的两个跨度其实取自同一筛选**（上面已核）—— "
            "『一个来自全表、一个来自合格表』这句话本身是错的，已撤回。"
            "② 但**两个跨度量纲不同**（0–1 相似度 vs 百分点），比大小没有意义。"
            "③ 更要紧的是：**跨度小 ≠ 无分辨力** —— 判别力是**排序**性质，"
            "不是幅度性质；单调而幅度小的评分照样能选对 K。"
            "④ 『噪声驱动』需要**扰动稳定性**证据（重采样 / 加噪后 argmax 是否翻），"
            "本产物没有做 ⇒ 它是**未验证假设**，不是根因。"
        ),
    }

    # ── 3. 场内相关（选 K 真正依赖的）+ 命中计数 ──
    within, hits = [], {}
    for key in ("score", "objective"):
        per = []
        hit_strict = hit_tie = 0
        for it in meetings:
            e = elig(it["rows"])
            if len(e) < 3:
                continue
            per.append(spearman([r[key] for r in e], [r["attribution_error"] for r in e]))
            best = max(e, key=lambda r: (r[key], -r["k"]))
            low = min(r["attribution_error"] for r in e)
            ties = [r["k"] for r in e if abs(r["attribution_error"] - low) < 1e-9]
            hit_strict += int(best["k"] == min(ties))
            hit_tie += int(best["k"] in ties)
        per = [x for x in per if x is not None]
        hits[key] = {"hit_strict": hit_strict, "hit_tie": hit_tie,
                     "hit_strict_basis": f"argmax({key}) 的 k == 最小最优 k",
                     "hit_tie_basis": f"argmax({key}) 的 k ∈ 并列最优集合",
                     "n_meetings": len(meetings)}
        within.append({"key": key, "median_within_meeting_rho": float(np.median(per)),
                       "n_negative_meetings": int((np.array(per) < 0).sum()),
                       "n_meetings": len(per)})

    # ── 4. λ 惩罚：逐场，改善与变差都要计 ──
    def pick(rs, penalty):
        e = elig(rs)
        if not e:
            return rs[0]
        key = ((lambda r: (r["score"] - 0.005 * (r["k"] - 2), -r["k"])) if penalty
               else (lambda r: (r["score"], -r["k"])))
        return max(e, key=key)
    lam, better, worse, kchange = [], [], [], []
    for it in meetings:
        a, b = pick(it["rows"], True), pick(it["rows"], False)
        if a["k"] == b["k"]:
            continue
        kchange.append({"meeting": it["meeting"], "k_with_penalty": a["k"],
                        "k_without_penalty": b["k"],
                        "err_with": a["attribution_error"], "err_without": b["attribution_error"]})
        d = (a["attribution_error"] - b["attribution_error"]) * 100   # 正 = 去掉惩罚更好
        (better if d > 1e-9 else worse if d < -1e-9 else lam).append(
            {"meeting": it["meeting"], "delta_points": float(d)})
    lam_summary = {
        "n_meetings_changed": len(kchange),
        "changed": kchange,
        "remove_penalty_better": better,
        "remove_penalty_worse": worse,
        "k_changed_error_same": lam,
        "mean_with_penalty": float(np.mean([pick(it["rows"], True)["attribution_error"]
                                            for it in meetings])),
        "mean_without_penalty": float(np.mean([pick(it["rows"], False)["attribution_error"]
                                               for it in meetings])),
        "reading": ("逐场净出：改善场与变差场必须**一起**计。"
                    "只累加变差场会把一个净收益说成净损失。"),
    }

    # ── 5. 方向计数（严格口径 = 取最小最优 K；含并列口径另报） ──
    need_big = need_small = optimal = 0
    for it in meetings:
        e = elig(it["rows"])
        p = pick(it["rows"], True) if elig(it["rows"]) else it["rows"][0]
        low = min(r["attribution_error"] for r in e)
        ties = [r["k"] for r in e if abs(r["attribution_error"] - low) < 1e-9]
        if p["k"] < min(ties):
            need_big += 1
        elif p["k"] > max(ties):
            need_small += 1
        else:
            optimal += 1

    # ── 6. CAP / 常数检查 ──
    max_elig_k = sorted({max(r["k"] for r in elig(it["rows"])) for it in meetings})
    cap_check = {
        "cap": table["cap"],
        "distinct_max_eligible_K_values": max_elig_k,
        "n_meetings_whose_max_eligible_K_equals_cap":
            sum(1 for it in meetings if max(r["k"] for r in elig(it["rows"])) == table["cap"]),
        "n_meetings": len(meetings),
        "reading": ("若**每一场**的最大合格 K 都等于 cap，则『取最大合格 K』本身也是常数 —— "
                    "它看起来优于生产规则，测的是『cap 定得比生产选的大』，不是某个判据更聪明。"),
    }

    # ── 6b. ρ 溯源：−0.617 到底出自哪个筛选 ──
    search_filters = {
        "全表": lambda r: True,
        "K>=2": lambda r: r["k"] >= 2,
        "K>=2&合格": lambda r: r["k"] >= 2 and r["eligible"],
        "K>=3": lambda r: r["k"] >= 3,
        "K>=2&去掉M_R003S04C01": lambda r: r["k"] >= 2 and r["meeting"] != "M_R003S04C01",
        "K>=2&合格&去掉M_R003S04C01": lambda r: (r["k"] >= 2 and r["eligible"]
                                              and r["meeting"] != "M_R003S04C01"),
    }
    rho_search = {}
    for name, keep in search_filters.items():
        sel = [r for r in rows if keep(r)]
        rho_search[name] = {"n": len(sel),
                            "rho": spearman([r["score"] for r in sel],
                                            [r["attribution_error"] for r in sel])}
    rho_search["old_artifact_value"] = -0.617
    rho_search["reading"] = (
        "旧产物的 −0.617 **不能**由任何自然筛选得到：声明的『K≥2 且合格』下是 −0.611。"
        "−0.617 只在**额外剔除 `M_R003S04C01`**（n=99）时出现 —— 而该场恰好是最差两场之一。"
        "在报告里剔除一场最差会议会**改变要解释的对象**，且旧产物未声明该操作 ⇒ "
        "以 **−0.611（n=107，K≥2 且合格）** 为准。"
    )

    doc = {
        "schema": "v13-selector-score-diagnosis-v2",
        "supersedes": "v13_selector_score_diagnosis_4238cb9c.json",
        "gate": "rttm", "split": "dev",
        "source_table": "artifacts/v13_candidate_score_table_dev.json",
        "recompute": "python -u -B scripts/evaluation/build_selector_score_diagnosis.py",
        "corrections_to_previous": [
            {
                "field": "rho_note",
                "previous": "跨 (场次, k) 的秩相关为负：silhouette 越高，错误率越高。"
                            "这与「用轮廓系数选 K」的直觉相反。",
                "error": "**符号解释错误**。ρ<0 表示评分**越高**、错误率**越低** —— "
                         "即排序**方向与直觉一致**，不是相反。按原文读法（高分 ⇒ 高错）"
                         "只有 ρ>0 才成立。",
                "correct": "ρ<0 ⇒ silhouette 越高，归属错误率越低。方向正确；"
                           "问题在**幅度与口径**，不在符号。",
            },
            {
                "field": "scale_mismatch",
                "previous": "评分的场内跨度比错误率跨度小 1-2 个数量级 ⇒ 候选之间几乎无分辨力，"
                            "选 K 实际由惩罚项与噪声驱动。",
                "error": "**两处**：① 两个跨度单位不同（0–1 相似度 vs 百分点），比大小无意义；"
                         "② 『跨度小 ⇒ 无分辨力』是**未证推理** —— 判别力是排序性质，"
                         "单调且幅度小的评分照样能选对。"
                         "⚠️ 注意：本修正的**初稿**又错了一次（曾声称两个跨度取自不同筛选）。"
                         "复核发现旧产物的 0.049–0.279 与 1.3–54.0 **都**是 K≥2&合格的数"
                         "（含 K=1 下界会是 0.0，含不合格会是 -1.0）—— 已在产物里撤回。",
                "correct": "跨度在三种筛选下并排重算（见 `spans.by_filter`），确认旧值属 K≥2&合格；"
                           "『噪声驱动』需要扰动稳定性证据，本线没有做，故降级为**待验证假设**。",
            },
            {
                "field": "production_rule_accuracy / direction",
                "previous": "「12 场中 5 场选到句级最优 K」与「7 场需更大 / 2 场需更小 / 3 场已最优」"
                            "并列出现，各自的口径未写。",
                "error": "两者**比较对象不同**且未声明；`需要更小` 在本表上复算为 **0 场**，不是 2 场。",
                "correct": "严格口径（argmax 命中最小最优 K）与含并列口径的计数分别列出；"
                           "方向计数按『生产 k 落在并列最优集合之外』判定。",
            },
            {
                "field": "counterfactual_rules.holdout_8 / artifact_warning",
                "previous": "纯「取最大合格 K」已达 dev 5.95% / holdout **6.11%**，"
                            "容差规则相对它只多赚 **0.02-0.75 点**。",
                "error": "**查无出处**：本线产物里**没有** holdout 的 `largest_eligible` 数字"
                         "（诊断 JSON 只有 dev 的 `largest_eligible_k=0.0595`）；"
                         "而 JSON 自述的 holdout `sil_tol0.15_largest_k=0.0502` 与引用值 0.0611 "
                         "相减是 **1.09 点**，方向还与『多赚』相反 ⇒ 这句与其自引数字矛盾。"
                         "进一步核查 holdout 整腿：唯一一份 holdout clean 桶产物"
                         "（`v13_clean_bucket_holdout_lamscan_mw2_objective_sel8-*.json`）的 `mw2` 变体"
                         "均值是 **6.23%**，而旧 JSON 声称 **6.12%**；两者逐场对照有"
                         "**3 场对不上任何已提交产物**（S_R003S02C01 声称 0.74% vs 产物 2.21%；"
                         "S_R003S03C01 声称 2.70% vs 3.38%；S_R003S04C01 声称 6.41% vs mw2 5.13%）。"
                         "⇒ holdout 腿的三个数（6.12 / 5.02 / 2.72）均**不可复算**。",
                "correct": "0.02–0.75 点的范围、holdout 6.11%、以及 holdout 整条 counterfactual "
                           "**一并撤回**。能复算的只有 **dev 腿**：5.95%（最大合格 K）→ 5.20%"
                           "（容差规则，但它的选中 k 就是 CAP ⇒ 常数） = 0.75 点。",
            },
            {
                "field": "conclusions",
                "previous": "「根因是**评分尺度与目标尺度不匹配**」。",
                "error": "把**一个未验证的候选解释**写成了已确定的根因。",
                "correct": "改述为『当前证据支持、且尚未被独立验证的假设之一』。"
                           "已确认的只有**现象**：生产 objective 相对诊断上界有显著选择损失。",
            },
            {
                "field": "conclusions[1]",
                "previous": "错误方向**系统性偏向小 K**（7 场需要更大 K）。",
                "error": "漏了同样重要的另一半：**5 场已最优**，且 `需要更小 K` = 0 场 ⇒ "
                         "生产规则**从不选得过大**，偏小是单向的。",
                "correct": "方向、严格命中、含并列命中三个计数并列报出。",
            },
        ],
        "confirmed_observations": {
            "spearman_by_filter": rho,
            "spearman_reading": "三种口径都给出 -0.61 ~ -0.71。**负号 = 评分越高、错误率越低**。"
                                "口径之间只差一行（M_R003S04C01 k=2，唯一的不合格行，"
                                "其 score 是 -1.0 哨兵值而非轮廓系数）——"
                                "把哨兵当测量值会污染相关系数，故 K≥2 且合格的 -0.611 应为准。",
            "within_meeting_spearman": within,
            "within_meeting_reading": "场内 ρ 中位 −0.33（silhouette）比跨场 −0.61 弱得多。"
                                      "**选 K 只依赖场内排序**，因此场内数字才是相关的那个。",
            "selection_hits": hits,
            "direction_counts": {"needs_larger_K": need_big, "needs_smaller_K": need_small,
                                 "already_optimal": optimal, "basis": "严格口径（最小最优 K）"},
            "spans": span_summary, "per_meeting_spans": spans,
            "penalty_lambda_0.005": lam_summary,
            "cap_dependence": cap_check,
        },
        "hypotheses_not_root_causes": [
            "『silhouette 与句级质量尺度不匹配 ⇒ 选 K 被噪声驱动』—— 方向正确（ρ<0）"
            "但『无分辨力』未被扰动稳定性证据支持，且跨度不可比。**是假设，不是根因。**",
            "『需要一个与句级归属质量相关的目标函数』—— 依据是选择损失存在，"
            "但选择损失本身是**诊断上界差**（逐场读真值），不能直接推出『换目标函数就能兑现』。",
        ],
        "data_gaps": [
            "旧诊断产物没有生成脚本进仓库 ⇒ 其数字当时不可复算。本产物可由 "
            "`v13_candidate_score_table_dev.json` 重算。",
            "`v13_candidate_score_table_dev.json` 只覆盖 **RTTM 门**。VAD 门（部署时可得）"
            "需要另跑；两种门不可互换。",
            "holdout 整腿不可复算：仓库里**没有** holdout 的 `largest_eligible` 句级产物；"
            "旧 JSON 的 holdout 逐场值混了 `base`/`mw2` 两个变体，其中 3 场对不上任何产物"
            "（详见 corrections_to_previous 的第 4 条）。",
        ],
        "rho_provenance_search": rho_search,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {OUT}")

    print("\n=== ρ 三种口径 ===")
    for k, v in rho.items():
        print(f"  {k:16} n={v['n']:3} ρ=-{abs(v['rho_score_vs_error']):.3f} "
              f"哨兵行={v['n_sentinel_score_-1']}")
    print(f"\n场内 ρ 中位: " + ", ".join(
        f"{w['key']} {w['median_within_meeting_rho']:+.3f}" for w in within))
    print(f"命中: " + ", ".join(
        f"{k} 严格{v['hit_strict']}/12 含并列{v['hit_tie']}/12" for k, v in hits.items()))
    print(f"方向: 需更大 {need_big} / 需更小 {need_small} / 已最优 {optimal}")
    print(f"λ: 改 {len(kchange)} 场；去掉惩罚 更好 {len(better)} / 更差 {len(worse)}；"
          f"{lam_summary['mean_with_penalty'] * 100:.2f}% → "
          f"{lam_summary['mean_without_penalty'] * 100:.2f}%")
    print(f"CAP: 最大合格 K 等于 cap 的场次 "
          f"{cap_check['n_meetings_whose_max_eligible_K_equals_cap']}/{cap_check['n_meetings']}，"
          f"取值集合 {max_elig_k}")
    print("ρ 溯源: " + "; ".join(
        f"{k} n={v['n']} ρ={v['rho']:+.4f}" for k, v in rho_search.items()
        if isinstance(v, dict)))


if __name__ == "__main__":
    main()
