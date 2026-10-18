# -*- coding: utf-8 -*-
"""句级尺子 + 滞后平滑对照 + R8008 根因。三件事一次跑完，不需要模型。

**为什么需要这把尺子**：DER 会被"单标签表达不了重叠"这条结构性天花板主导
（实测 DER ≈ 0.85 × 重叠率，r=0.979），拿它当目标等于拿噪声当信号。工程上真正
关心的是两件事——**每句话有没有对上人**、**有没有被切开**。

指标定义（全部只读盘上已有数据：真值 RTTM / TextGrid / 现有预测 RTTM）：

    turn_acc_sentence  句级：每句取覆盖最多的预测说话人，映射后与真值比，一句一票
    turn_acc_frame     帧级：映射后同标签的交集时长 / 真值语音总时长（时长加权）
    fragments_mean     一个真值句平均被切成几段（原始定义）
    split_rate         该句里"显著说话人 >=2"（占比 > 10%）的比例 —— 硬约束违反率

⚠️ **必须做全局 ID 映射**：预测标签是 `v12_0`，真值标签是 `N_SPK8013`，两个 ID 空间
不同源，直接 `p['spk'] == gt['spk']` 永远不相等，会输出 0% 准确率。

平滑（对治 0.6s 细化造成的碎句）：
    切换必须维持 min_hold 秒，否则并入更长的邻居；同人相邻段间隙 < merge_gap 直接缝合。
    实测依据：v12（含细化）的切句率是 baseline（无细化）的 2 倍以上。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_turn_metrics.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr",
           REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, PREDICTIONS, RESULTS, der_tools  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)

_, read_rttm = der_tools()

SIG_THRESH = 0.10          # 占比超过多少才算"这句里有这个人"
MIN_HOLD = 1.2             # 说话人切换的最短维持时间（秒）
MERGE_GAP = 0.4            # 同说话人相邻段的最大缝合间隙（秒）


# ── 几何 ────────────────────────────────────────────────────────────────

def coverage(segments, start, end):
    """各说话人与 [start, end] 的交集时长（秒）。"""
    per = defaultdict(float)
    for seg in segments:
        a, b = max(start, seg["start"]), min(end, seg["end"])
        if b > a:
            per[seg["speaker"]] += b - a
    return dict(per)


def n_covering(segments, start, end):
    """覆盖该区间的预测段**数量**（用户定义的 fragments）。"""
    return sum(1 for seg in segments
               if min(end, seg["end"]) - max(start, seg["start"]) > 0)


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


# ── 滞后平滑（修掉两处 bug：不原地改输入；用迭代而非单遍扫描）────────────

def _compress(segs, merge_gap):
    out = []
    for s in segs:
        if out and out[-1]["speaker"] == s["speaker"] and s["start"] - out[-1]["end"] <= merge_gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def smooth_turns(segments, min_hold=MIN_HOLD, merge_gap=MERGE_GAP):
    """切换需维持 min_hold 秒，否则并入更长的邻居；同人相邻段缝合。

    不修改输入（返回新对象）。
    """
    segs = _compress(sorted((dict(s) for s in segments), key=lambda x: x["start"]),
                     merge_gap)
    for _ in range(64):
        if len(segs) <= 1:
            break
        runs, i = [], 0
        while i < len(segs):
            j = i
            while j + 1 < len(segs) and segs[j + 1]["speaker"] == segs[i]["speaker"]:
                j += 1
            runs.append((i, j, segs[j]["end"] - segs[i]["start"]))
            i = j + 1
        short = [r for r in runs if r[2] < min_hold]
        if not short:
            break
        a, b, _ = min(short, key=lambda r: r[2])
        prev_d = segs[a - 1]["end"] - segs[a - 1]["start"] if a > 0 else -1.0
        next_d = segs[b + 1]["end"] - segs[b + 1]["start"] if b + 1 < len(segs) else -1.0
        if prev_d < 0 and next_d < 0:
            break
        target = segs[a - 1]["speaker"] if prev_d >= next_d else segs[b + 1]["speaker"]
        for k in range(a, b + 1):
            segs[k]["speaker"] = target
        segs = _compress(segs, merge_gap)
    return segs


# ── 指标 ────────────────────────────────────────────────────────────────

def turn_metrics(pred, gt_turns, mapping, thresh=SIG_THRESH):
    ok = n = 0
    correct = total = 0.0
    frags, splits, empty = [], 0, 0
    for gt in gt_turns:
        span = gt["end"] - gt["start"]
        if span <= 0:
            continue
        total += span
        n += 1
        want = mapping.get(gt["speaker"])
        cov = coverage(pred, gt["start"], gt["end"])
        frags.append(n_covering(pred, gt["start"], gt["end"]))
        if not cov:
            empty += 1
            continue
        if want is not None:
            correct += cov.get(want, 0.0)
        dominant = max(cov, key=lambda k: cov[k])
        ok += int(dominant == want)
        if len([k for k, v in cov.items() if v / span > thresh]) >= 2:
            splits += 1
    return {"n_turns": n, "ok": ok, "empty": empty,
            "acc_sentence": ok / n if n else 0.0,
            "acc_frame": correct / total if total else 0.0,
            "fragments_mean": sum(frags) / len(frags) if frags else 0.0,
            "split_rate": splits / n if n else 0.0}


def confusion(truth, pred, mapping):
    """真值说话人 × 预测说话人 的重叠时长矩阵（看是映射错乱还是簇混人）。"""
    rows = defaultdict(lambda: defaultdict(float))
    for t in truth:
        for p in pred:
            ov = min(t["end"], p["end"]) - max(t["start"], p["start"])
            if ov > 0:
                rows[t["speaker"]][p["speaker"]] += ov
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="v12",
                    help="对比哪条预测臂：v12 含 0.6s 细化 / baseline 无细化")
    args = ap.parse_args()
    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    rows = []
    print("%-14s %5s %8s %9s %9s %8s %8s | %9s %9s %8s %8s"
          % ("meeting", "句数", "句级准确", "帧级准确", "碎段均值", "切句率",
             "映射数", "平滑后句级", "平滑后帧级", "平滑后碎", "平滑后切"),
          flush=True)
    for meeting in meetings:
        pred_path = PREDICTIONS / f"{meeting}.{args.arm}.full.rttm"
        if not pred_path.exists():
            continue
        truth = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        pred = read_rttm(pred_path)
        gt_turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        mapping = global_mapping(truth, pred)
        base = turn_metrics(pred, gt_turns, mapping)
        smooth = turn_metrics(smooth_turns(pred), gt_turns, mapping)
        row = {"meeting": meeting, "n_pred_speakers": len({p["speaker"] for p in pred}),
               "n_truth_speakers": len({t["speaker"] for t in truth}),
               "mapping": mapping, "base": base, "smooth": smooth}
        rows.append(row)
        print("%-14s %5d %7.1f%% %8.1f%% %9.2f %7.1f%% %8d | %8.1f%% %8.1f%% %8.2f %7.1f%%"
              % (meeting, base["n_turns"], base["acc_sentence"] * 100,
                 base["acc_frame"] * 100, base["fragments_mean"],
                 base["split_rate"] * 100, len(mapping),
                 smooth["acc_sentence"] * 100, smooth["acc_frame"] * 100,
                 smooth["fragments_mean"], smooth["split_rate"] * 100), flush=True)

    def agg(group):
        n = sum(r["base"]["n_turns"] for r in group) or 1
        return {
            "n": n,
            "acc_sentence": sum(r["base"]["ok"] for r in group) / n,
            "acc_frame": (sum(r["base"]["acc_frame"] * r["base"]["n_turns"] for r in group) / n),
            "fragments_mean": sum(r["base"]["fragments_mean"] * r["base"]["n_turns"] for r in group) / n,
            "split_rate": sum(r["base"]["split_rate"] * r["base"]["n_turns"] for r in group) / n,
            "s_acc_sentence": sum(r["smooth"]["ok"] for r in group) / n,
            "s_split_rate": sum(r["smooth"]["split_rate"] * r["smooth"]["n_turns"] for r in group) / n,
            "s_fragments_mean": sum(r["smooth"]["fragments_mean"] * r["smooth"]["n_turns"] for r in group) / n,
        }

    print("\n=== 汇总（%d 场 %d 句）===" % (len(rows), sum(r["base"]["n_turns"] for r in rows)))
    a = agg(rows)
    print("  句级 Turn Accuracy   %6.1f%%" % (a["acc_sentence"] * 100))
    print("  帧级 Turn Accuracy   %6.1f%%   （时长加权，含重叠罚）" % (a["acc_frame"] * 100))
    print("  平均碎段数/句        %6.2f" % a["fragments_mean"])
    print("  切句率（显著>=2人）  %6.1f%%   <- 硬约束违反率" % (a["split_rate"] * 100))
    print("  --- 滞后平滑后 (min_hold=%.1fs, merge_gap=%.1fs) ---" % (MIN_HOLD, MERGE_GAP))
    print("  句级 Turn Accuracy   %6.1f%%   (%+.1f pt)"
          % (a["s_acc_sentence"] * 100, (a["s_acc_sentence"] - a["acc_sentence"]) * 100))
    print("  平均碎段数/句        %6.2f   (%+.2f)"
          % (a["s_fragments_mean"], a["s_fragments_mean"] - a["fragments_mean"]))
    print("  切句率               %6.1f%%   (%+.1f pt)"
          % (a["s_split_rate"] * 100, (a["s_split_rate"] - a["split_rate"]) * 100))

    # 塌陷场单列（它们会把总准确率拉到接近随机）
    collapsed = [r for r in rows if r["n_pred_speakers"] <= 1]
    healthy = [r for r in rows if r["n_pred_speakers"] >= 2]
    if collapsed:
        ch = agg(collapsed)
        hh = agg(healthy)
        print("\n  分组（塌陷场会把总准确率拉到接近随机）：")
        print("    预测>=2人 %d 场: 句级 %.1f%%  切句 %.1f%%"
              % (len(healthy), hh["acc_sentence"] * 100, hh["split_rate"] * 100))
        print("    预测==1人 %d 场: 句级 %.1f%%  切句 %.1f%%"
              % (len(collapsed), ch["acc_sentence"] * 100, ch["split_rate"] * 100))

    # R8008 根因：人数已选对（3），DER 仍 79% —— 看映射矩阵
    target = "R8008_M8013"
    row = next((r for r in rows if r["meeting"] == target), None)
    if row:
        truth = read_rttm(ALI_RTTM / f"{target}.rttm")
        pred = read_rttm(PREDICTIONS / f"{target}.{args.arm}.full.rttm")
        print("\n=== %s 根因：真值 %d 人 / 预测 %d 簇（人数已选对，DER 仍 79%%）==="
              % (target, row["n_truth_speakers"], row["n_pred_speakers"]))
        print("  映射: %s" % row["mapping"])
        mat = confusion(truth, pred, row["mapping"])
        preds = sorted({p["speaker"] for p in pred})
        print("  %-14s %s" % ("真值 \\ 预测", "  ".join("%12s" % p for p in preds)))
        for t_spk in sorted(mat):
            total = sum(mat[t_spk].values()) or 1.0
            cells = "  ".join("%11.1f%%" % (100 * mat[t_spk].get(p, 0.0) / total)
                              for p in preds)
            print("  %-14s %s" % (t_spk, cells))
        print("  （每行 = 该真值说话人的语音被分到各预测簇的比例；对角线之外越大 = 混得越厉害）")

    # min_hold 扫描：切句率与归属准确率是一对权衡，必须显式找折中点
    print("\n=== min_hold 扫描（切句率 vs 句级准确率的折中）===")
    print("  %9s %11s %10s %11s" % ("min_hold", "句级准确", "切句率", "碎段均值"))
    cache = []
    for meeting in meetings:
        pred_path = PREDICTIONS / f"{meeting}.{args.arm}.full.rttm"
        if not pred_path.exists():
            continue
        truth = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        cache.append((read_rttm(pred_path),
                      sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid")),
                      global_mapping(truth, read_rttm(pred_path))))
    for mh in (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0):
        ok = n = 0
        splits = frags = 0.0
        for pred, gt_turns, mapping in cache:
            m = turn_metrics(smooth_turns(pred, min_hold=mh), gt_turns, mapping)
            ok += m["ok"]
            n += m["n_turns"]
            splits += m["split_rate"] * m["n_turns"]
            frags += m["fragments_mean"] * m["n_turns"]
        print("  %9.1f %10.1f%% %9.1f%% %11.2f"
              % (mh, 100.0 * ok / max(n, 1), 100.0 * splits / max(n, 1),
                 frags / max(n, 1)), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "turn_metrics_audit.json"
    out.write_text(json.dumps({
        "config": {"sig_thresh": SIG_THRESH, "min_hold": MIN_HOLD,
                   "merge_gap": MERGE_GAP,
                   "note": "句级尺子含全局 ID 映射；平滑作用于预测段表，不改标签"},
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 %s" % out)


if __name__ == "__main__":
    main()
