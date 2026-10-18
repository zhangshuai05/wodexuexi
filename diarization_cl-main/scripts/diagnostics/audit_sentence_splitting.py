# -*- coding: utf-8 -*-
"""句级「切句」审计：现在有多少句被系统切成了两个说话人？

**起因**：验收标准是「一句一标签、句内标签不许翻转」——
真值 A、B、C 各一句，识别成 A、B、A **可以接受**（漏了 C 但 A/B 分开了）；
但一句被切成「A 占 1.5 句、B 占 1 句、A 占 0.5 句」**不可接受**。

这条硬约束此前**没有任何人量过**。本脚本只读盘上已有的三份东西，不跑模型、秒级：

    TextGrid（每说话人一个 tier，句子自带真值说话人）
    reference_rttm/*.rttm（真值）
    predictions/<场次>.v12.full.rttm（生产路径预测，含细化）

对每句 [start, end] 求预测段表的覆盖，按占比分类：

    切句      : 预测在这句里覆盖 >=2 个说话人（占比都 > THRESH）-> 违反硬约束
    单标签    : 只有一个说话人覆盖 -> 再看是否等于真值（需全局映射）
    无预测    : 预测完全没覆盖这句（段表塌陷或空缺时会大量出现）

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_sentence_splitting.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr", REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, PREDICTIONS, RESULTS, der_tools  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)

_, read_rttm = der_tools()
ARMS = ("v12", "baseline")
THRESH_MAIN = 0.10
THRESH_STRICT = 0.30


def coverage(segments, start, end):
    """各说话人与 [start, end] 的交集时长（秒）。"""
    per = defaultdict(float)
    for seg in segments:
        a, b = max(start, seg["start"]), min(end, seg["end"])
        if b > a:
            per[seg["speaker"]] += b - a
    return dict(per)


def global_mapping(truth, pred):
    """贪心最大重叠：真值说话人 -> 预测说话人（整场一次，不逐段重配）。"""
    pair = defaultdict(float)
    for t in truth:
        for p in pred:
            ov = min(t["end"], p["end"]) - max(t["start"], p["start"])
            if ov > 0:
                pair[(t["speaker"], p["speaker"])] += ov
    mapping, used = {}, set()
    for (t_spk, p_spk), ov in sorted(pair.items(), key=lambda kv: -kv[1]):
        if t_spk in mapping or p_spk in used:
            continue
        mapping[t_spk] = p_spk
        used.add(p_spk)
    return mapping


def audit(meeting, arm, thresh):
    tiers = read_textgrid(ALI_TG / f"{meeting}.TextGrid")
    sentences = sentences_from_textgrid(tiers)
    truth = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    pred_path = PREDICTIONS / f"{meeting}.{arm}.full.rttm"
    if not pred_path.exists():
        return None
    pred = read_rttm(pred_path)
    mapping = global_mapping(truth, pred)

    stat = {"meeting": meeting, "arm": arm, "n_sentences": len(sentences),
            "split": 0, "single": 0, "no_pred": 0, "correct": 0,
            "n_pred_speakers": len({s["speaker"] for s in pred}),
            "n_truth_speakers": len({s["speaker"] for s in truth})}
    split_examples = []
    for sent in sentences:
        span = sent["end"] - sent["start"]
        if span <= 0:
            continue
        cov = coverage(pred, sent["start"], sent["end"])
        shares = {spk: t / span for spk, t in cov.items()}
        sig = sorted((spk for spk, r in shares.items() if r > thresh),
                     key=lambda spk: -shares[spk])
        if len(sig) >= 2:
            stat["split"] += 1
            if len(split_examples) < 3:
                split_examples.append({
                    "start": round(sent["start"], 2), "end": round(sent["end"], 2),
                    "truth": sent["speaker"],
                    "pred": {spk: round(shares[spk], 3) for spk in sig[:3]}})
        elif len(sig) == 1:
            stat["single"] += 1
            if mapping.get(sent["speaker"]) == sig[0]:
                stat["correct"] += 1
        else:
            stat["no_pred"] += 1
    stat["split_examples"] = split_examples
    return stat


def main():
    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    rows = []
    print("%-14s %-5s %6s %7s %7s %7s %8s %7s %7s"
          % ("meeting", "arm", "真值人", "预测人", "句数", "切句", "切句率",
             "单标签", "无预测"),
          flush=True)
    for meeting in meetings:
        for arm in ARMS:
            stat = audit(meeting, arm, THRESH_MAIN)
            if stat is None:
                continue
            stat["split_rate"] = stat["split"] / max(stat["n_sentences"], 1)
            stat["acc_single"] = stat["correct"] / max(stat["single"], 1)
            rows.append(stat)
            print("%-14s %-5s %6d %7d %7d %7d %7.1f%% %7d %7d"
                  % (meeting, arm, stat["n_truth_speakers"], stat["n_pred_speakers"],
                     stat["n_sentences"], stat["split"], stat["split_rate"] * 100,
                     stat["single"], stat["no_pred"]), flush=True)

    v12 = [r for r in rows if r["arm"] == "v12"]
    if v12:
        n_sent = sum(r["n_sentences"] for r in v12)
        n_split = sum(r["split"] for r in v12)
        n_single = sum(r["single"] for r in v12)
        n_correct = sum(r["correct"] for r in v12)
        n_nopred = sum(r["no_pred"] for r in v12)
        print("\n=== v12（生产路径）汇总，%d 场 ===" % len(v12))
        print("  句数            %5d" % n_sent)
        print("  切句（>=2 标签） %5d   %.1f%%   <- 你的硬约束违反率" % (n_split, 100.0 * n_split / max(n_sent, 1)))
        print("  单标签          %5d   其中归属正确 %d（%.1f%%）"
              % (n_single, n_correct, 100.0 * n_correct / max(n_single, 1)))
        print("  无预测覆盖      %5d   %.1f%%" % (n_nopred, 100.0 * n_nopred / max(n_sent, 1)))
        print("\n  严格阈值（占比 > %.0f%% 才算第二人）：" % (THRESH_STRICT * 100))
        strict_split = 0
        for meeting in meetings:
            stat = audit(meeting, "v12", THRESH_STRICT)
            if stat:
                strict_split += stat["split"]
        print("    切句 %d（%.1f%%）" % (strict_split, 100.0 * strict_split / max(n_sent, 1)))

        print("\n  塌陷检测（预测说话人数 == 1 的场次）：")
        for r in v12:
            if r["n_pred_speakers"] <= 1:
                print("    %-14s 真值 %d 人 -> 预测 %d 人，句数 %d，无预测覆盖 %d"
                      % (r["meeting"], r["n_truth_speakers"], r["n_pred_speakers"],
                         r["n_sentences"], r["no_pred"]))

        # 分组：塌陷场会把"归属正确率"拉到接近随机，必须分开报
        healthy = [r for r in v12 if r["n_pred_speakers"] >= 2]
        collapsed = [r for r in v12 if r["n_pred_speakers"] < 2]
        print("\n  分组（全归一个人的场次会污染总正确率，必须分开看）：")
        for label, group in (("预测>=2人", healthy), ("预测==1人(塌陷)", collapsed)):
            if not group:
                continue
            n_sent = sum(r["n_sentences"] for r in group)
            n_split = sum(r["split"] for r in group)
            n_single = sum(r["single"] for r in group)
            n_correct = sum(r["correct"] for r in group)
            print("    %-16s %d 场  句数 %5d  切句 %4d (%4.1f%%)  单标签归属正确 %4d/%4d (%.1f%%)"
                  % (label, len(group), n_sent, n_split,
                     100.0 * n_split / max(n_sent, 1), n_correct, n_single,
                     100.0 * n_correct / max(n_single, 1)))

        print("\n  切句样例（前 3 条/场）：")
        for r in v12:
            for ex in r["split_examples"][:2]:
                print("    %-14s 真值 %s  %.1f-%.1fs  预测分布 %s"
                      % (r["meeting"], ex["truth"], ex["start"], ex["end"], ex["pred"]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "sentence_splitting_audit.json"
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval",
                   "thresh_main": THRESH_MAIN, "thresh_strict": THRESH_STRICT,
                   "note": "切句 = 预测在一句内覆盖 >=2 个说话人；无需模型，只读真值+句表+已有预测"},
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 %s" % out)


if __name__ == "__main__":
    main()
