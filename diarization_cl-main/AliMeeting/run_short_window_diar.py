# -*- coding: utf-8 -*-
"""短窗 + 纯度过滤对照：把"缩短声纹窗"和"剔除骑墙派样本"分开验证。

**为什么要做**：我之前的句级实验只证明了"用真值排除重叠窗"有效（pure 臂 R8007_M8010
80.2% vs 现状 22.2%），但没有证明**能不能自动判出哪些样本是脏的**——而生产里没有真值。
另外我一直用 2.0s 窗，短窗（0.9s）是唯一还没试过的关键变量。

四个臂**共用同一次嵌入提取**（0.9s 窗 / 0.45s 步长），唯一差别是"用哪些样本聚类"：

    short   全部窗直接聚类（只改窗长）
    margin  按"Top1-Top2 余弦差 < margin"自动剔除骑墙派样本后再聚类
    two     用 margin 臂的中心，把**全部**窗按最近中心分配（两阶段，覆盖率 100%）
    oracle  用真值剔除"窗内混了两个人"的样本后再聚类（诊断上界，非可达）

评测：窗级标签 -> 每句取主导说话人 -> 全局映射后算句级准确率（与现状同一把尺子）。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_short_window_diar.py --meetings R8007_M8010
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "streaming_asr", HERE.parent / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, RESULTS, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)
from run_der import load_models  # noqa: E402

WIN_S, HOP_S = 0.9, 0.45        # 用户方案建议的短窗
MAX_SPEAKERS = 10
MIN_CHUNK = 0.35
OVERLAP_TOL = 0.05
GATE = (2, 0.010)

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def slide(start, end):
    wins, t = [], start
    while t + WIN_S <= end + 1e-9:
        wins.append((t, t + WIN_S))
        t += HOP_S
    if not wins and end - start >= MIN_CHUNK:
        wins.append((start, min(end, start + WIN_S)))
    return wins


def window_is_pure(a, b, segments, tol=OVERLAP_TOL):
    """窗内是否只坐着一个人（需要真值，仅用于 oracle 臂）。"""
    speakers = set()
    for seg in segments:
        if min(b, seg["end"]) - max(a, seg["start"]) > tol:
            speakers.add(seg["speaker"])
    return len(speakers) <= 1


def extract_all(model, audio, rate, vad_segments, truth_segments):
    """按 **VAD 段**滑短窗（与生产同源），真值只用来给 oracle 臂标纯度。

    ⚠️ 不能用真值段滑窗：那样每个窗都落在语音内，等于偷偷用了真值，与现状不可比。
    """
    vecs, wins, pure = [], [], []
    for seg_start, seg_end in vad_segments:
        for a, b in slide(seg_start, seg_end):
            chunk = audio[int(a * rate):int(b * rate)]
            if len(chunk) < int(MIN_CHUNK * rate):
                continue
            try:
                emb = model.generate(input=chunk)[0].get("spk_embedding")
            except Exception:
                continue
            if emb is None:
                continue
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy()
            vecs.append(_norm(emb))
            wins.append((a, b))
            pure.append(window_is_pure(a, b, truth_segments))
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32),
            wins, np.asarray(pure))


def cluster(E, gate=GATE):
    """复用生产的 AHC + 选 K（簇门换成句/窗通用的宽松档）。"""
    import claude_a2_v11 as v11
    if len(E) < 2:
        return np.zeros(len(E), dtype=int), 1
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = gate
    try:
        cands, _ = v11._candidate_clusters(E, max_speakers=min(MAX_SPEAKERS, len(E)))
        best = v11._select_candidate(cands)
    finally:
        v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = old
    return best["labels"], int(best["k"])


def centroids(E, labels, keep_ratio=0.7):
    """每个簇只用"离簇心最近的前 keep_ratio"样本算质心（滤掉污染样本）。"""
    out = []
    for c in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            continue
        raw = _norm(E[idx].mean(axis=0))
        sims = E[idx] @ raw
        cut = np.percentile(sims, (1.0 - keep_ratio) * 100.0)
        keep = idx[sims >= cut]
        out.append(_norm(E[keep].mean(axis=0)))
    return np.stack(out) if out else np.empty((0, E.shape[1]), np.float32)


def margin_filter(E, margin=0.065):
    """剔除骑墙派：先用全体样本定粗中心，再把 Top1-Top2 差 < margin 的样本踢掉。

    返回 (保留掩码, 迭代后的中心)。这就是用户方案里"余弦中心度过滤"的可测版本。
    """
    labels, k = cluster(E)
    if k < 2:
        return np.ones(len(E), dtype=bool), None, k
    cents = centroids(E, labels)
    for _ in range(3):
        sims = E @ cents.T
        order = np.sort(sims, axis=1)
        keep = (order[:, -1] - order[:, -2]) >= margin
        if keep.sum() < max(2, k) or keep.all():
            break
        labels2, k2 = cluster(E[keep])
        if k2 < 2:
            break
        cents = centroids(E[keep], labels2)
    return keep, cents, k


def score_by_turn(windows, labels, turns, mapping_segments):
    """窗级标签 -> 句级主导投票 -> 全局映射 -> 句级准确率。"""
    covered = defaultdict(float)
    for (a, b), lab in zip(windows, labels.tolist()):
        covered[(a, b)] = lab
    gt, pred, durs = [], [], []
    for turn in turns:
        tally = defaultdict(float)
        for (a, b), lab in covered.items():
            ov = min(b, turn["end"]) - max(a, turn["start"])
            if ov > 0:
                tally[lab] += ov
        if not tally:
            continue
        gt.append(turn["speaker"])
        pred.append(max(tally, key=lambda x: tally[x]))
        durs.append(turn["end"] - turn["start"])
    pair = defaultdict(float)
    for g, p, d in zip(gt, pred, durs):
        pair[(g, p)] += d
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gt, pred) if mapping.get(g) == p)
    return {"acc": ok / max(len(gt), 1), "k": len(set(pred)), "n_turns": len(gt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=["R8007_M8010"])
    ap.add_argument("--out", default="short_window_diar.json")
    args = ap.parse_args()

    import soundfile as sf
    models = load_models()
    rows = []
    print("%-14s %6s | %-22s | %-22s | %-22s | %s"
          % ("meeting", "窗数", "short(全窗)", "margin(剔骑墙)", "two(两阶段)",
             "oracle(真值剔重叠·上界)"), flush=True)
    for meeting in args.meetings:
        t0 = time.time()
        audio, rate = sf.read(str(resolve_wav(meeting)), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        import claude_a2_v11 as v11
        vad_segments = v11._vad_segs(models["vad"], str(resolve_wav(meeting)))
        E, wins, pure = extract_all(models["spk"], audio, rate, vad_segments, segments)
        row = {"meeting": meeting, "n_windows": int(len(E)),
               "n_vad_segments": len(vad_segments),
               "pure_ratio": float(pure.mean()) if len(pure) else 0.0}

        lab_s, k_s = cluster(E)
        row["short"] = {**score_by_turn(wins, lab_s, turns, segments), "k_cluster": k_s}

        keep, cents, k_m = margin_filter(E)
        row["margin_ratio"] = float(keep.mean()) if len(keep) else 0.0
        if cents is not None and k_m >= 2:
            lab_m, _ = cluster(E[keep])
            row["margin"] = score_by_turn([w for w, k in zip(wins, keep) if k],
                                          lab_m, turns, segments)
            lab_two = (E @ cents.T).argmax(axis=1)
            row["two"] = score_by_turn(wins, lab_two, turns, segments)
        else:
            row["margin"] = row["two"] = {"acc": 0.0, "k": k_m, "note": "未分出 >=2 簇"}

        if pure.sum() >= 2:
            lab_o, _ = cluster(E[pure])
            row["oracle"] = score_by_turn([w for w, p in zip(wins, pure) if p],
                                          lab_o, turns, segments)
        else:
            row["oracle"] = {"acc": 0.0, "note": "纯净窗不足"}

        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print("%-14s %6d | acc %.1f%% k=%d        | acc %.1f%% k=%s (留%.0f%%) | acc %.1f%% k=%d       | acc %.1f%% (纯窗%.0f%%)"
              % (meeting, len(E),
                 row["short"]["acc"] * 100, row["short"]["k_cluster"],
                 row["margin"]["acc"] * 100, row["margin"].get("k"),
                 row["margin_ratio"] * 100,
                 row["two"]["acc"] * 100, row["two"].get("k", 0),
                 row["oracle"]["acc"] * 100, row["pure_ratio"] * 100), flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"win_s": WIN_S, "hop_s": HOP_S,
                                   "note": "oracle 臂用真值剔重叠窗，是诊断上界"},
                        "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print("\n对照：现状（2.0s 窗 + AHC 段表 + 句级投票）R8007_M8010 = 22.2%，R8008_M8013 = 38.1%")
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
