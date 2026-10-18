# -*- coding: utf-8 -*-
"""`unlocked_selector` 的**句级主指标**（真实路径口径，零推理）。

为什么需要它：unlocked 的收益此前**只用 DER 说过**（`probe_e2e_unlocked2`），
而项目主指标是句级归属率。两把尺子对"多出来的簇"记账方式不同（DER 记为 FA，
主指标走匈牙利映射），**不能互相换算**，必须实测。

为什么便宜：`probe_path_equiv` 已验证「缓存 embedding + `real_path_labels`」
与 `v12.diarize()` 在句级**逐位一致**（26 个对比 `n_sentence_diff` 全 0）。
走真模型是 **21 分钟/场**，读缓存是秒级。

同时跑 base（生产 v12 选择器）与 `min_win` 多档，回答两件事：
  * 主指标上 unlocked 到底赚不赚；
  * `min_win=2` 是**全部 20 场选出来的**（in-sample），相邻档位是否同样好。

    python scripts/evaluation/probe_unlocked_sentence.py --min-wins 2,3,4
    python scripts/evaluation/probe_unlocked_sentence.py --selfcheck   # 对齐真模型读数
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

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

import probe_ab_dev as AB  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

from unlocked_selector import CAP, build_unlocked, pick_unlocked  # noqa: E402

E2E = ROOT / "artifacts" / "v13_e2e_unlocked2_dev_mw2_objective.json"


def _fingerprint(meetings):
    """场次集合的 8 位指纹：同 tag 下选不同场次（哪怕数量相同）也不会撞名。"""
    import hashlib

    return hashlib.sha1(",".join(sorted(meetings)).encode("utf-8")).hexdigest()[:8]


def candidates_for(E, min_win, mode, penalty):
    """`None` 表示 base（生产选择器）；否则返回 unlocked 选中的候选。

    `penalty` 只作用在 unlocked 这条支线上：base 必须恒用生产值，
    否则基线会被扫描值改掉，前后两腿不可比。
    """
    if min_win is None:
        cands, _ = v11._candidate_clusters(E, max_speakers=v12.MAX_SPEAKERS)
        selected = v11._select_candidate(cands)
        return {"k": int(selected["k"]), "labels": selected["labels"]}
    # `build_unlocked` 读的是模块变量 `v11.MULTI_K_PENALTY`（调用时求值）。
    saved = v11.MULTI_K_PENALTY
    v11.MULTI_K_PENALTY = penalty
    try:
        return pick_unlocked(build_unlocked(E, CAP, min_win), mode=mode)
    finally:
        v11.MULTI_K_PENALTY = saved


def repick(cands, penalty, mode="objective"):
    """在**同一份候选表**上换 λ 重选 k。

    候选的 `score`（轮廓）与 λ 无关，只有 `objective = score − λ(k−2)` 那一项变，
    所以扫 λ 不必重建候选表（HANDOFF §4.5 的零计算法）。这里直接复用
    `pick_unlocked`，保证 tie-break / 回退规则与生产**同一份实现**。

    `mode != "objective"` 时 λ 不参与选择，重选是恒等操作——直接返回原表。
    """
    if mode != "objective":
        return None
    scored = [{**c, "objective": c["score"] - penalty * (c["k"] - 2)} for c in cands]
    return pick_unlocked(scored, mode="objective")


def run_meeting(meeting, min_wins, mode, penalty):
    E, times = PE.cached_stage(meeting, "main")
    rE, rtimes = PE.cached_stage(meeting, "refine")
    if len(E) < 2:
        return None
    sentences = AB.load_sentences(meeting)
    if not sentences:
        return None
    out = {}
    for mw in [None] + list(min_wins):
        chosen = candidates_for(E, mw, mode, penalty)
        labels = chosen["labels"]
        flabels, ftimes, level = AB.real_path_labels(np.asarray(labels), E, times, rE, rtimes)
        out["base" if mw is None else "mw%d" % mw] = {
            "k": int(chosen["k"]), "level": level,
            "n_clusters_after_refine": int(len(set(int(x) for x in flabels))),
            **AB.score(sentences, flabels, ftimes, AB._win(level))}
    return out


def run_meeting_scan(meeting, min_win, mode, penalties):
    """同一场在多个 λ 上的读数（clean/overlap 桶 × λ 网格）。

    **base 臂按每个 λ 重建**：坑 22 —— λ 落地时 base 自己也动了，把 base 钉死在
    某个值会让"收益"变成跨口径比（HANDOFF §9 禁引）。成本为零，因为 base 用的是
    生产选择器，与 λ 无关，下面按 k 备忘。

    `real_path_labels + score` 按 (臂, k) **备忘**：同一场 7 个 λ 通常只落 2–3 个不同 k。
    """
    E, times = PE.cached_stage(meeting, "main")
    rE, rtimes = PE.cached_stage(meeting, "refine")
    if len(E) < 2:
        return None
    sentences = AB.load_sentences(meeting)
    if not sentences:
        return None

    cache = {}

    def measure(arm, k, labels):
        key = (arm, k)
        if key not in cache:
            fl, ft, lvl = AB.real_path_labels(np.asarray(labels), E, times, rE, rtimes)
            cache[key] = {"k": int(k), "level": lvl,
                          "n_clusters_after_refine": int(len(set(int(x) for x in fl))),
                          **AB.score(sentences, fl, ft, AB._win(lvl))}
        return cache[key]

    # 候选表只建一次（`score` 与 λ 无关）。`build_unlocked` 现在读模块常量，
    # 但 `repick` 不再经过它，所以这里用**当前生产值**建表即可，取值不影响重选。
    cands = build_unlocked(E, CAP, min_win)
    base = candidates_for(E, None, mode, None)

    out = {}
    for lam in penalties:
        row = {"base": measure("base", base["k"], base["labels"])}
        picked = repick(cands, lam, mode)
        if picked is None:
            # λ 不参与选择（mode != objective）：退化为单点，显式标注而不是假造一列。
            row["mw%d" % min_win] = {
                "k": None, "level": None, "n_clusters_after_refine": None,
                "note": "λ 不参与 mode=%s 的选择，本列恒等" % mode}
        else:
            row["mw%d" % min_win] = measure("mw%d" % min_win, picked["k"], picked["labels"])
        out[lam] = row
    return out



def run_lambda_scan(args, tag, meetings, min_wins, default_penalty):
    """λ 网格 × clean/overlap 桶。

    回答的问题（HANDOFF 未答、卡在 clean 桶上的那一个）：
    train_L 的 clean 桶缺口里，**覆盖率（弃权）占多少、聚类几何（错答）占多少**。

    判据：**clean 桶的数值对 λ 不敏感** ⇒ 它与聚类几何无关，是覆盖率/噪音面。
    """
    penalties = [float(x) for x in args.lambda_scan.split(",") if x.strip()]
    if not penalties:
        raise SystemExit("--lambda-scan 解析后为空")
    mw = min_wins[0]
    if len(min_wins) > 1:
        print("  注意：扫描模式只跑 --min-wins 的第一档 %d，其余忽略" % mw)

    rows, skipped = {}, []
    for meeting in meetings:
        try:
            r = run_meeting_scan(meeting, mw, args.mode, penalties)
        except FileNotFoundError as exc:
            skipped.append("%s: %s" % (meeting, exc))
            continue
        if r is None:
            skipped.append("%s: 窗数或句子数不足" % meeting)
            continue
        rows[meeting] = r

    if not rows:
        raise SystemExit("无可用场次：%s" % skipped)

    bucket_keys = ("overall", "clean", "overlap")
    print("λ 扫描  cohort=%s  场次=%d  min_win=%d  mode=%s  （base 与 unlocked 同口径重建）"
          % (tag, len(rows), mw, args.mode))
    print()
    arm = "mw%d" % mw

    def mean_of(cells, bucket, field):
        """空桶的字段是 `None`（`attribution._empty()`）——必须滤掉，否则 np.mean 抛 TypeError。
        例：某场一句重叠都没有时 overlap 桶为空。全空时返回 None 而不是 NaN，让调用方显式打印。
        """
        vals = [c for c in cells
                if c.get("k") is not None and c[bucket][field] is not None]
        return 100 * float(np.mean([c[bucket][field] for c in vals])) if vals else None

    def fmt_pair(cells, bucket):
        p, a = mean_of(cells, bucket, "per_sentence"), mean_of(cells, bucket, "abstain")
        if p is None:
            return "n/a"
        return "%.1f/%.1f" % (p, a if a is not None else float("nan"))

    print("%-9s %-13s %-13s %-13s %-13s %s" % (
        "penalty", "overall 主/弃", "clean 主/弃", "overlap 主/弃", "｜base 主/弃", "k 变场次"))
    for lam in penalties:
        vals = [r[lam][arm] for r in rows.values()]
        bvals = [r[lam]["base"] for r in rows.values()]
        nch = sum(1 for m in rows
                  if rows[m][lam][arm].get("k") != rows[m][penalties[0]][arm].get("k"))
        print("%-9g %-13s %-13s %-13s %-13s %s" % (
            lam, fmt_pair(vals, "overall"), fmt_pair(vals, "clean"),
            fmt_pair(vals, "overlap"), fmt_pair(bvals, "overall"), "%d/%d" % (nch, len(rows))))

    print()
    print("—— clean 桶分解（判据：clean 的**弃权**对 λ 是否敏感）——")
    print("%-9s %-16s %-16s %s" % ("penalty", "clean 主指标", "clean 内弃权", "clean 内错答"))
    for lam in penalties:
        ps = [r[lam][arm] for r in rows.values()]
        ps = [c for c in ps if c.get("k") is not None and c["clean"]["per_sentence"] is not None]
        if not ps:
            print("%-9g %s" % (lam, "（无 clean 句）"))
            continue
        cp = float(np.mean([c["clean"]["per_sentence"] for c in ps]))
        ca = float(np.mean([c["clean"]["abstain"] for c in ps]))
        print("%-9g %-16s %-16s %s" % (lam, "%.1f%%" % (100 * cp), "%.1f%%" % (100 * ca),
                                       "%.1f%%" % (100 * (1.0 - cp - ca))))

    print()
    print("—— 逐场 k 随 λ 的变化（只看 unlocked 臂）——")
    print("%-26s %s" % ("场次", " ".join("λ=%-7g" % lam for lam in penalties)))
    changed = 0
    for meeting in sorted(rows):
        ks = [rows[meeting][lam]["mw%d" % mw].get("k") for lam in penalties]
        if len(set(ks)) > 1:
            changed += 1
        print("%-26s %s" % (meeting, " ".join("%-9s" % k for k in ks)))
    print()
    print("  k 随 λ 变化的场次: %d / %d" % (changed, len(rows)))

    print()
    print("—— base 臂是否随 λ 变化（坑 22：基准臂被同时动过时，收益是跨口径比）——")
    base_changed = 0
    for meeting in sorted(rows):
        ks = [rows[meeting][lam]["base"]["k"] for lam in penalties]
        if len(set(ks)) > 1:
            base_changed += 1
            print("  %-26s %s" % (meeting, " ".join("%-9s" % k for k in ks)))
    print("  base k 随 λ 变化的场次: %d / %d" % (base_changed, len(rows)))

    prod = default_penalty
    parts = [tag, "lamscan", "mw%d" % mw, args.mode,
             "sel%d-%s" % (len(meetings), _fingerprint(meetings))]
    out = ROOT / "artifacts" / ("v13_clean_bucket_%s.json" % "_".join(parts))
    out.write_text(json.dumps(
        {"schema": "v13-clean-bucket-lambda-v1", "cohort_tag": tag, "split": args.split,
         "min_win": mw, "mode": args.mode, "penalties": penalties,
         "production_penalty": prod, "meetings": list(rows),
         "base_k_changed_meetings": base_changed,
         "path": "cache+real_path_labels(句级已验证≡diarize)",
         "note": "clean 桶含弃权句且算未答对（abstain 与 per_sentence 同分母）",
         "skipped": skipped, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入 %s" % out)
    if skipped:
        print("⚠ 跳过 %d 场: %s" % (len(skipped), skipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--min-wins", default="2,3,4")
    ap.add_argument("--mode", default="objective",
                    choices=("objective", "largest_eligible", "largest"))
    ap.add_argument("--penalty", type=float, default=None,
                    help="MULTI_K_PENALTY 扫描；默认=生产值（只影响 unlocked 支线）")
    ap.add_argument("--selfcheck", action="store_true",
                    help="与真模型产物 v13_e2e_unlocked2_dev_mw2_objective.json 对齐 k")
    ap.add_argument("--root", default=None,
                    help="数据根（真值 TextGrid 所在）。train_L 等另一个 cohort 用这个；"
                         "需先用 build_probe_cache.py --root 建好窗缓存")
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="显式场次列表。train_L 的场次名带日期前缀，"
                         "`split_meetings` 的分组函数会抛异常，必须走这条")
    ap.add_argument("--tag", default=None,
                    help="产物文件名里的标记；默认 = split（给了 --root 时用 root 目录名）")
    ap.add_argument("--lambda-scan", dest="lambda_scan", default=None,
                    help="逗号分隔的 MULTI_K_PENALTY 网格。给了就走扫描模式（只跑 --min-wins 的"
                         "第一档），每场每 λ 报 base 与 unlocked 的 clean/overlap 桶。"
                         "例：--lambda-scan 0,0.003,0.005,0.008,0.01,0.015,0.03")
    args = ap.parse_args()

    default_penalty = v11.MULTI_K_PENALTY
    penalty = default_penalty if args.penalty is None else args.penalty
    min_wins = [int(x) for x in args.min_wins.split(",") if x.strip()]
    if not min_wins:
        # 空表会在下面 `primary = keys[1]` 处 IndexError，且是跑完所有场次之后才爆。
        raise SystemExit("--min-wins 不能为空")
    if args.root:
        # `probe_ab_dev` 的真值根也在这里读（`load_sentences`）。
        AB.DATA = Path(args.root)
    tag = args.tag or (Path(args.root).name if args.root else args.split)
    meetings = args.meetings or split_meetings(AB.DATA, args.split)

    if args.lambda_scan:
        run_lambda_scan(args, tag, meetings, min_wins, default_penalty)
        return

    keys = ["base"] + ["mw%d" % w for w in min_wins]

    rows, skipped = {}, []
    for meeting in meetings:
        try:
            r = run_meeting(meeting, min_wins, args.mode, penalty)
        except FileNotFoundError as exc:      # 缺缓存 → 显式记录，不静默跳过
            skipped.append("%s: %s" % (meeting, exc))
            continue
        if r is None:
            skipped.append("%s: 窗数或句子数不足" % meeting)
            continue
        rows[meeting] = r

    if not rows:
        # 非零退出：缓存未建 / --root 写错时，批处理不能把"什么都没跑"当成功。
        raise SystemExit("无可用场次：%s" % skipped)

    print("mode=%s  cohort=%s  场次=%d  penalty=%g（生产 %g）  真路径口径（缓存重建，句级主指标）"
          % (args.mode, tag, len(rows), penalty, default_penalty))
    head = "%-16s" % "会议"
    for k in keys:
        head += "│ %-17s" % (k + " k/主指标")
    # Δ 列跟随**实际跑的那一档**，不写死 mw2：`--min-wins 3,4` 时 mw2 根本不在结果里。
    primary = keys[1]
    print(head + "│ Δ%s−base" % primary)
    for meeting in sorted(rows):
        r = rows[meeting]
        line = "%-16s" % meeting
        for k in keys:
            c = r[k]
            line += "│ k=%-2d %5.1f%% 弃%4.1f" % (
                c["k"], 100 * c["per_sentence"], 100 * c["abstain"])
        d = r[primary]["per_sentence"] - r["base"]["per_sentence"]
        print(line + "│ %+7.1f pts" % (100 * d))

    print()
    for k in keys:
        acc = np.mean([r[k]["per_sentence"] for r in rows.values()])
        levels = Counter(r[k]["level"] for r in rows.values())
        print("  %-6s 主指标均值 %.1f%%  弃权 %.1f%%  level=%s"
              % (k, 100 * acc, 100 * np.mean([r[k]["abstain"] for r in rows.values()]),
                 dict(levels)))
    base = np.mean([r["base"]["per_sentence"] for r in rows.values()])
    for k in keys[1:]:
        cur = np.mean([r[k]["per_sentence"] for r in rows.values()])
        diffs = [r[k]["per_sentence"] - r["base"]["per_sentence"] for r in rows.values()]
        print("  %-6s vs base: %+.1f pts  改善 %d / 恶化 %d / 持平 %d  (逐场Δ中位 %+.1f)"
              % (k, 100 * (cur - base),
                 sum(1 for d in diffs if d > 0.001), sum(1 for d in diffs if d < -0.001),
                 sum(1 for d in diffs if abs(d) <= 0.001), 100 * float(np.median(diffs))))

    # 真模型产物是「生产 λ + objective 口径」的；两项任一不同，k 本来就该不同，比它只会误报。
    comparable = penalty == default_penalty and args.mode == "objective"
    if args.selfcheck and not comparable:
        print("\n  selfcheck 跳过：penalty=%g（生产 %g）、mode=%s 与真模型产物口径不同。"
              % (penalty, default_penalty, args.mode))
    if args.selfcheck and comparable:
        # 独立来源：真模型产物。缓存路径若复现不出它的 k，本脚本的口径就不成立。
        # ponytail: 该产物还是 λ=0.015 时跑的，dev 的 `L_R003S03C02` 会报 mw2 k 不一致（按设计）；
        # 重跑该场真模型（≈21 分钟）并合回后删掉本注释。见 HANDOFF §4.12。
        if 2 not in min_wins:
            # 真模型产物只有 mw2 一臂，缺了就只能比 base；说清楚，别静默只比一半。
            print("\n  selfcheck 限定：--min-wins 未含 2，只比 base 臂（真模型产物没有这一档的 mw）")
        real = {r["meeting"]: r for r in json.loads(E2E.read_text(encoding="utf-8"))["rows"]}
        # 所选场次全不在产物里时，下面的 `continue` 会让 bad 恒为空——"全部一致 ✓" 是空的。
        # 与 build_probe_cache --check-only 同口径：0 场可比就必须非零退出。
        compared = [m for m in rows if m in real]
        if not compared:
            raise SystemExit("selfcheck 无可比场次：所选场次均不在真模型产物中")
        bad = []
        for meeting, r in rows.items():
            if meeting not in real:
                continue
            if r["base"]["k"] != real[meeting]["base"]["k"]:
                bad.append("%s base k %d≠%d" % (meeting, r["base"]["k"], real[meeting]["base"]["k"]))
            if "mw2" in r and r["mw2"]["k"] != real[meeting]["fixed"]["k"]:
                bad.append("%s mw2 k %d≠%d" % (meeting, r["mw2"]["k"], real[meeting]["fixed"]["k"]))
        print("\n  selfcheck vs 真模型产物（比对 %d/%d 场）: %s"
              % (len(compared), len(rows),
                 "全部一致 ✓" if not bad else "不一致 " + "; ".join(bad)))
        if bad:
            raise SystemExit(1)
    # 覆盖守卫：名字必须唯一确定调用口径（档位/模式/选场/λ），否则一次窄跑会静默盖掉整批产物。
    # λ 曾是隐含项：默认值从 0.015 改成 0.005 后，裸跑会盖掉 λ=0.015 的基线产物。
    # 场次集合**无条件**进名字：--split 与 --meetings 哪个定的场次都能区分（只记数量会撞）。
    parts = [tag, "mw" + "-".join(str(w) for w in min_wins), args.mode, "penalty%g" % penalty,
             "sel%d-%s" % (len(meetings), _fingerprint(meetings))]
    out = ROOT / "artifacts" / ("v13_unlocked_sentence_%s.json" % "_".join(parts))
    out.write_text(json.dumps({"schema": "v13-unlocked-sentence-v1", "split": args.split,
                               "penalty": penalty, "penalty_source":
                               "default" if args.penalty is None else "explicit",
                               "cohort_tag": tag, "meetings": list(rows),
                               "mode": args.mode, "min_wins": min_wins, "cap": CAP,
                               "path": "cache+real_path_labels(句级已验证≡diarize)",
                               "production_penalty": default_penalty,
                               "skipped": skipped, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入 %s" % out)
    if skipped:
        print("⚠ 跳过 %d 场: %s" % (len(skipped), skipped))


if __name__ == "__main__":
    main()
