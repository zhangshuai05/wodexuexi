# -*- coding: utf-8 -*-
"""k 网格上的**真实路径**句子归属率曲线：oracle / 双射 / 消重复簇三个口径。

## 为什么需要它

现行免真值判据（`unlocked_selector.build_unlocked`）在 train_L 上 **5/30** 选不中真 K，
但这**不是**缺陷的证据——README 明确写着「"K 应等于真值人数"未证明」：
K 同时承担「说话人身份数」与「时间边界锚点」两个角色，所以 **k 偏大常常有益**。

所以本脚本把优化目标从「选中真 K」换成「选中**句级归属率最高**的划分」，
真 K 降级为诊断量。要这么做，先得知道**每条 k 上的归属率是多少**——
现行三个判据都只在自己的读数上取极值，没人报过整条曲线。

## 三个口径，缺一不可

`probe_over_seg_gain` 已证明句级 oracle 是**多对一**：一个真值人被切成 5 个簇，
5 个簇仍全能映射回他，**归属率不降**。所以：

  * `oracle`        —— 多对一上界。**不可直接当目标**：它对过分割不扣分，
                       dev 实测 oracle 峰值 9/12 场在 5–8，但「恒选 k=10」regret=0.00。
  * `bijective`     —— 匈牙利**双射**。一个真值人只能被一个簇占用 ⇒ **对过分割扣分**。
                       这是「接近真划分」的诚实读数，也是本脚本推荐的选 k 目标。
  * `merged_oracle` —— 把 dup 簇（同一人的重复副本）并回主簇后再打分。
                       区分「真拆分收益」与「重复簇造的假收益」。

判据若直接追 oracle 最大值，会**一路选到 cap**（同 `largest_eligible` 的死法：
三场全选 10，是常数不是判据）。所以三条曲线都报，选 k 目标定为 **bijective 峰值**。

## 与「真 K」的关系

产物里同时有 `true_k` 与 `bijectiveK`（差值现算）；**不作为判据**。
`visibleK`（窗内实际出现过的真值人数）比 `trueK` 更该用来判「k 是否偏小」——
短发言者 15% 在窗内从未出现，判据不可能选中一个看不见的人。

## 口径

走**真实路径**：`probe_ab_dev.real_path_labels`（两级细化），不是粗窗代理。
理由见 `sentence_k_rule` 的 docstring——粗窗代理缺碎簇惩罚，系统性偏高
（dev 12 场里 9 场偏高），拿它标的判据不能迁移。

    python -u -B scripts/evaluation/probe_oracle_k_curve.py --selfcheck
    python -u -B scripts/evaluation/probe_oracle_k_curve.py --split dev
    python -u -B scripts/evaluation/probe_oracle_k_curve.py \
        --root "D:/yinp/AISHELL-4/data/train_L" --meetings 20200706_L_R001S07C01
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

import probe_ab_dev as AB  # noqa: E402
from der import read_rttm  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_mapping_objective import score_oracle  # noqa: E402
from probe_over_seg_gain import classify_dup, merge_dups  # noqa: E402
from probe_small_cluster_audit import truth_flags  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

K_GRID = tuple(range(2, 11))   # 与生产 cap=10 对齐


def truth_coverage(labels, names):
    """每个预测簇 → 主导真值人（复用 `probe_over_seg_gain` 同法，不另立口径）。"""
    rows = []
    for lab in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == lab)
        got = [str(names[i]) for i in idx if str(names[i]) != ""]
        comp = Counter(got)
        top, _ = comp.most_common(1)[0] if comp else ("", 0)
        rows.append({"cluster": lab, "n_windows": int(len(idx)), "dominant": top})
    return rows


def leg(sents, emb, times, rE, rtimes, names, k):
    """一条 k：真实路径细化 → 三个口径的句级归属率。"""
    labels = P.clusters_for(emb, k)
    flabels, ftimes, level = AB.real_path_labels(np.asarray(labels), emb, times, rE, rtimes)
    win = AB._win(level)
    end = max(s["end"] for s in sents)
    segs = P.segment_map(ftimes, flabels, np.full(len(ftimes), win), end)
    pred = P.sentence_dict_predictions(sents, segs)

    # dup 判定在主窗层做（与 probe_over_seg_gain 一致：那是分簇发生的地方）
    rows = classify_dup(truth_coverage(labels, names))

    # 匈牙利只解一次（同一 (sents, pred) 解两次是白算）。
    # `split()` 里的三个比率都取整到 4 位了 ⇒ 错答不能拿它们相减（残差会把三份取整误差
    # 全算到错答头上，最高偏 1.5e-4）；用精确计数 `n_wrong`，分母与 overall 一致。
    att = P.attribution(sents, pred)
    ov = att["overall"]
    bijective = float(ov["per_sentence"])
    out = {
        "k": int(k), "level": level,
        "n_clusters": len(set(int(x) for x in flabels)),
        "oracle": round(float(score_oracle(sents, pred)["overall"]["per_sentence"]), 4),
        "bijective": round(bijective, 4),
    }
    # 缺口成分：峰值处「没答对」的那部分里，有多少是弃权（无覆盖）、
    # 未映射（有预测但标签没配上名字）、还是真的错答。
    # 三者分开，才能判「换后端有没有用」——弃权与未映射不是聚类质量问题。
    out["n_sentences"] = ov["n"]
    out["abstain"] = ov["abstain"]
    out["unmapped"] = ov["unmapped"]
    # 错答用精确计数（四者互斥完备，见 attribution 的记账规则）：`n_wrong` 只计「有映射却
    # 没对上人」的句子。**不要**写成 1−对−弃权−未映射——那会把三份取整误差都算到错答头上。
    # 代价：四项各自取整后可能显示 99.99%，但每个值本身都是准确的。
    out["wrong"] = round(att["n_wrong"] / ov["n"], 4) if ov["n"] else None
    # 消重复簇：dup 并回主簇后**重走真实路径**再打分（不是只在标签上合并）
    merged = merge_dups(labels, rows)
    if len(set(int(x) for x in merged)) != len(set(int(x) for x in labels)):
        ml, mt, mlevel = AB.real_path_labels(np.asarray(merged), emb, times, rE, rtimes)
        mwin = AB._win(mlevel)
        msegs = P.segment_map(mt, ml, np.full(len(mt), mwin), end)
        mpred = P.sentence_dict_predictions(sents, msegs)
        out["merged_oracle"] = round(
            float(score_oracle(sents, mpred)["overall"]["per_sentence"]), 4)
        out["merged_bijective"] = round(
            float(P.attribution(sents, mpred)["overall"]["per_sentence"]), 4)
        out["merged_clusters"] = len(set(int(x) for x in ml))
    else:
        out.update(merged_oracle=out["oracle"], merged_bijective=out["bijective"],
                   merged_clusters=out["n_clusters"])
    out["dup_clusters"] = int(sum(1 for r in rows if r.get("is_dup")))
    return out


def argmax_k(curve, key):
    """曲线峰值 k。平局取**小 k**（与 `sentence_k_rule.pick_best_k` 同规）。"""
    if not curve:
        return None
    best = max(c[key] for c in curve)
    return min(c["k"] for c in curve if c[key] == best)


def probe_meeting(meeting):
    E, times = PE.cached_stage(meeting, "main")
    rE, rtimes = PE.cached_stage(meeting, "refine")
    sents = AB.load_sentences(meeting)
    if len(E) < 2 or not sents:
        return None
    ref = [s for s in read_rttm(AB.DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < AB.SECONDS]
    names = window_reference(ref, times, P.WIN)
    bnd, ovl = truth_flags(ref, times, P.WIN)

    curve = [leg(sents, E, times, rE, rtimes, names, k) for k in K_GRID]
    truth_speakers = {str(s["speaker"]) for s in ref}
    visible = {str(v) for v in names if str(v) != ""}
    item = {
        "meeting": meeting, "n_windows": int(len(E)), "n_sentences": len(sents),
        "true_k": len(truth_speakers), "visible_k": len(visible),
        "n_boundary_windows": int(bnd.sum()), "n_overlap_windows": int(ovl.sum()),
        "curve": curve,
        "oracleK": argmax_k(curve, "oracle"),
        "bijectiveK": argmax_k(curve, "bijective"),
        "merged_oracleK": argmax_k(curve, "merged_oracle"),
        "merged_bijectiveK": argmax_k(curve, "merged_bijective"),
    }
    for name in ("oracle", "bijective", "merged_oracle", "merged_bijective"):
        item["peak_" + name] = max(c[name] for c in curve)
    # 判据最远能差多少：oracle 峰值与 bijective 峰值的分数差（= 过分割可白拿的空间）
    item["oracle_minus_bijective_at_peak"] = round(
        item["peak_oracle"] - item["peak_bijective"], 4)
    return item


def selfcheck():
    """退化输入：一条只有单簇的曲线不应抛异常，argmax 取小 k。"""
    fake = [{"k": 2, "oracle": 0.8, "bijective": 0.7},
            {"k": 3, "oracle": 0.8, "bijective": 0.7}]
    assert argmax_k(fake, "oracle") == 2, argmax_k(fake, "oracle")
    assert argmax_k([], "oracle") is None
    print("selfcheck ✓  峰值平局取小 k；空曲线返回 None")
    # 真实缓存的一场，确认走的是真实路径（level 应出现 refined）
    item = probe_meeting("L_R003S01C02")
    assert item is not None, (
        "L_R003S01C02 没跑出曲线（缓存缺失，或窗/句数不足）——先 build_probe_cache.py 建缓存")
    levels = {c["level"] for c in item["curve"]}
    assert "refined" in levels, levels
    print("selfcheck ✓  L_R003S01C02 网格 %d 点，level=%s，oracleK=%s bijectiveK=%s"
          % (len(item["curve"]), sorted(levels), item["oracleK"], item["bijectiveK"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--root", default=None, help="数据根；train_L 等另一 cohort 用这个")
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()

    if args.root:
        AB.DATA = Path(args.root)
    tag = args.tag or (Path(args.root).name if args.root else args.split)
    meetings = args.meetings or split_meetings(AB.DATA, args.split)

    report, skipped = [], []
    for meeting in meetings:
        try:
            item = probe_meeting(meeting)
        except FileNotFoundError as exc:
            skipped.append("%s: %s" % (meeting, exc))
            continue
        if item is None:
            skipped.append("%s: 窗数或句子数不足" % meeting)
            continue
        report.append(item)
        print("[%d/%d] %-24s oracleK=%s(%s) bijK=%s(%s) trueK=%s visK=%s"
              % (len(report), len(meetings), meeting,
                 item["oracleK"], item["peak_oracle"],
                 item["bijectiveK"], item["peak_bijective"],
                 item["true_k"], item["visible_k"]), flush=True)

    if not report:
        # 非零退出：缓存未建 / --root 写错时，批处理不能把"什么都没跑"当成功。
        raise SystemExit("无可用场次：%s" % skipped)

    print("\n%-24s %5s %5s %7s %7s %8s %8s %7s" % (
        "会议", "真K", "可见", "oracleK", "双射K", "oracle峰", "双射峰", "峰差"))
    for m in report:
        print("%-24s %5d %5d %7s %7s %7.1f%% %7.1f%% %+7.1f" % (
            m["meeting"], m["true_k"], m["visible_k"], m["oracleK"], m["bijectiveK"],
            100 * m["peak_oracle"], 100 * m["peak_bijective"],
            100 * m["oracle_minus_bijective_at_peak"]))
    n = len(report)
    print("\n合计 %d 场：oracleK 中位 %s，双射K 中位 %s，真K 中位 %s，可见K 中位 %s"
          % (n,
             np.median([m["oracleK"] for m in report]),
             np.median([m["bijectiveK"] for m in report]),
             np.median([m["true_k"] for m in report]),
             np.median([m["visible_k"] for m in report])))
    print("oracleK ≥ 双射K 的场次：%d/%d（差即过分割可白拿的空间）"
          % (sum(1 for m in report if m["oracleK"] >= m["bijectiveK"]), n))

    # 覆盖守卫：场次集合无条件进文件名（单场跑不得盖掉整批产物；
    # `--split` 与 `--meetings` 谁定的场次都能区分，光记数量会撞）。
    import hashlib

    sel = "_sel%d-%s" % (len(meetings), hashlib.sha1(
        ",".join(sorted(meetings)).encode("utf-8")).hexdigest()[:8])
    out = Path(args.output) if args.output else (
        ROOT / "artifacts" / ("v13_oracle_k_curve_%s%s.json" % (tag, sel)))
    out.write_text(json.dumps({
        "schema": "v13-oracle-k-curve-v1", "cohort_tag": tag, "grid": list(K_GRID),
        "path": "real_path_labels(两级细化)", "path_verified": "probe_path_equiv 句级≡diarize",
        "meetings": report, "skipped": skipped},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入 %s" % out)
    if skipped:
        print("⚠ 跳过 %d 场: %s" % (len(skipped), skipped))


if __name__ == "__main__":
    main()
