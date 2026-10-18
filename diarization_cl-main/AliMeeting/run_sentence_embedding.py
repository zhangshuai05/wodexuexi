# -*- coding: utf-8 -*-
"""句级嵌入聚类：把聚类单位从"窗"换成"句"，验证它能否解决 R8008 的簇内混人。

**动机（实测）**：R8008_M8013 人数选对（3）、映射也对，但每个真值说话人只有
59.3% / 82.6% / 63.7% 落在自己的簇里 —— 同一人三分之一的语音跑进了别人的簇。
根因不是聚类算法，是**固定网格窗不管说话人边界**：切换窗、重叠窗、短发言窗把
每个人的声纹污染了。（佐证：同一 CAM++ 在**纯净段内**的窗上留一准确率 100%。）

**做法**：每句提一个嵌入（句内各 2.0s 窗的 L2 平均），直接对**句子**聚类。
好处有三：① 嵌入变纯 ② 一句一标签、结构上不可能切句（切句率恒为 0）
③ 样本从几千个脏窗变成几百个干净句。

**两臂只差一样东西 —— 用不用"有别人在说"的那部分音频**：

    full  句内全部窗（现实可行，不知道谁在什么时候说话）
    pure  只用句内"不与其他说话人重叠"的窗（用真值排除，**上界**，不是可达收益）

两者之差 = 重叠污染值多少。**pure 是诊断上界，不得当作可达收益引用。**

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_sentence_embedding.py --limit 1
    python run_sentence_embedding.py --out sentence_embedding.json [--resume]
"""

import argparse
import json
import math
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

WIN_S, HOP_S = 2.0, 1.0
MIN_CHUNK = 0.4            # 短于这个秒数的音频不提嵌入
MAX_SPEAKERS = 10
OVERLAP_TOL = 0.05

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def sentence_windows(start, end):
    """句内完全落得下的 2.0s 窗；一句都放不下时退化成整句一段。"""
    wins, t = [], start
    while t + WIN_S <= end + 1e-9:
        wins.append((t, t + WIN_S))
        t += HOP_S
    if not wins and end - start >= MIN_CHUNK:
        wins.append((start, min(end, start + WIN_S)))
    return wins


def pure_windows(start, end, segments, self_speaker):
    """句内**不与其他说话人重叠**的窗（需要真值，诊断上界）。"""
    others = [s for s in segments if s["speaker"] != self_speaker]
    out = []
    for a, b in sentence_windows(start, end):
        clash = any(min(b, o["end"]) - max(a, o["start"]) > OVERLAP_TOL for o in others)
        if not clash:
            out.append((a, b))
    return out


def embed(model, audio, rate, windows):
    """窗嵌入的 L2 平均 -> 句嵌入。"""
    vecs = []
    for a, b in windows:
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
    if not vecs:
        return None
    return _norm(np.mean(vecs, axis=0))


def greedy_mapping(gt_speakers, pred_labels, durations):
    """真值说话人 -> 预测标签 的全局贪心最大时长映射。"""
    pair = defaultdict(float)
    for g, p, d in zip(gt_speakers, pred_labels, durations):
        pair[(g, p)] += d
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    return mapping


def cluster_and_score(embeddings, gt_speakers, durations, gate=(2, 0.010)):
    """对句嵌入聚类（复用生产的 AHC + 选 K），返回句级准确率。

    ⚠️ 簇门必须换：`MIN_CLUSTER_WINDOWS=4 / MIN_CLUSTER_SHARE=0.015` 是给**窗**定的。
    句级上 668 句 × 1.5% = 10 句的门，会让"某人只说了 9 句"否掉整个候选 -> K 塌陷到 1
    （实测：未换门时 full 臂 K=1、准确率 32.3%）。句级样本少得多，门要相应收紧到
    (2 句, 1%)：既允许小说话人存在，又不放进 1 句的噪声簇。
    """
    import claude_a2_v11 as v11

    E = np.stack(embeddings)
    n = len(E)
    if n < 2:
        return {"k": 1, "acc": 0.0, "n": n}
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = gate
    try:
        cands, _ = v11._candidate_clusters(E, max_speakers=min(MAX_SPEAKERS, n))
        best = v11._select_candidate(cands)
    finally:
        v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = old
    labels = best["labels"]
    mapping = greedy_mapping(gt_speakers, labels.tolist(), durations)
    ok = sum(1 for g, p in zip(gt_speakers, labels.tolist())
             if mapping.get(g) == p)
    return {"k": int(best["k"]), "acc": ok / n, "n": n,
            "k_true": len(set(gt_speakers)), "gate": list(gate)}


def seed_and_assign(pure_data, full_data, gate=(2, 0.010)):
    """两阶段：**纯句聚类得说话人中心 -> 所有句（含重叠句）按中心分配**。

    为什么必须分两步（实测）：把含重叠的句子和纯句混在一起聚类，轮廓系数会变成负数
    （`_select_candidate` 里 `best["score"] > 0.0` 兜底 -> 退回 k=1），整场塌陷。
    实测 full 臂两场都是 k=1；而 pure 臂单独跑，R8007_M8010 能到 80.2%（现状 22.2%）。
    ⇒ 纯度决定聚类能不能立起来，覆盖率则靠第二阶段补回。
    """
    import claude_a2_v11 as v11

    P, gts_p, durs_p = pure_data
    A, gts_a, durs_a = full_data
    if len(P) < 2 or len(A) < 2:
        return {"error": "数据不足", "n": len(A)}
    Pm, Am = np.stack(P), np.stack(A)
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = gate
    try:
        cands, _ = v11._candidate_clusters(Pm, max_speakers=min(MAX_SPEAKERS, len(Pm)))
        best = v11._select_candidate(cands)
    finally:
        v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = old
    k = int(best["k"])
    if k < 2:
        return {"k": k, "acc": 0.0, "n": len(Am), "k_true": len(set(gts_a)),
                "note": "纯句臂自身塌陷"}
    labels_p = best["labels"]
    cents = np.stack([_norm(Pm[labels_p == c].mean(axis=0)) for c in range(k)])
    lab_all = (Am @ cents.T).argmax(axis=1)
    mapping = greedy_mapping(gts_a, lab_all.tolist(), durs_a)
    ok = sum(1 for g, p in zip(gts_a, lab_all.tolist()) if mapping.get(g) == p)
    # 顺带算：Top1/Top2 接近的句子占比（模块三"熵双标签"的候选率）
    sims = Am @ cents.T
    if k >= 2:
        part = np.sort(sims, axis=1)
        ratio = part[:, -2] / (part[:, -1] + 1e-9)
        ambiguous = float((ratio >= 0.82).mean())
    else:
        ambiguous = 0.0
    return {"k": k, "acc": ok / len(Am), "n": len(Am),
            "k_true": len(set(gts_a)), "coverage": 1.0,
            "ambiguous_rate": ambiguous}


def run_meeting(models, meeting, write_rttm=False):
    import soundfile as sf

    wav = resolve_wav(meeting)
    audio, rate = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))

    out = {"meeting": meeting, "n_turns": len(turns),
           "n_truth_speakers": len({t["speaker"] for t in turns})}
    collected = {}
    for arm in ("full", "pure"):
        vecs, gts, durs = [], [], []
        for turn in turns:
            wins = (sentence_windows(turn["start"], turn["end"]) if arm == "full"
                    else pure_windows(turn["start"], turn["end"], segments, turn["speaker"]))
            if not wins:
                continue
            v = embed(models["spk"], audio, rate, wins)
            if v is None:
                continue
            vecs.append(v)
            gts.append(turn["speaker"])
            durs.append(turn["end"] - turn["start"])
        collected[arm] = (vecs, gts, durs)
        if len(vecs) < 2:
            out[arm] = {"error": "可用句不足", "n": len(vecs)}
            continue
        res = cluster_and_score(vecs, gts, durs)
        res["coverage"] = len(vecs) / len(turns)
        out[arm] = res
    out["hybrid"] = seed_and_assign(collected["pure"], collected["full"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--out", default="sentence_embedding.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if args.limit:
        meetings = meetings[:args.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out
    done = []
    if args.resume and out_path.exists():
        done = [r for r in json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])
                if "error" not in r]
        finished = {r["meeting"] for r in done}
        meetings = [m for m in meetings if m not in finished]
        print("续跑：已完成 %d 场，本轮剩 %d 场" % (len(finished), len(meetings)), flush=True)

    models = load_models()
    rows = list(done)
    print("%-14s %5s %6s | %7s %7s %5s | %7s %7s %5s | %8s %5s %7s"
          % ("meeting", "句数", "真值人", "full句", "full准确", "fullK",
             "pure句", "pure准确", "pureK", "hyb准确", "hybK", "歧义率"), flush=True)
    for meeting in meetings:
        t0 = time.time()
        try:
            row = run_meeting(models, meeting)
        except Exception as exc:
            print("%-14s FAILED %s: %s" % (meeting, type(exc).__name__, exc), flush=True)
            continue
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        f, p, h = row.get("full", {}), row.get("pure", {}), row.get("hybrid", {})
        print("%-14s %5d %6d | %7s %6s%% %5s | %7s %6s%% %5s | %7s%% %5s %6s%%"
              % (meeting, row["n_turns"], row["n_truth_speakers"],
                 f.get("n", "-"), ("%.1f" % (f["acc"] * 100)) if "acc" in f else "-",
                 f.get("k", "-"),
                 p.get("n", "-"), ("%.1f" % (p["acc"] * 100)) if "acc" in p else "-",
                 p.get("k", "-"),
                 ("%.1f" % (h["acc"] * 100)) if "acc" in h else "-",
                 h.get("k", "-"),
                 ("%.1f" % (h["ambiguous_rate"] * 100)) if h.get("ambiguous_rate") is not None else "-"),
              flush=True)
        out_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    ok = [r for r in rows if "full" in r and "acc" in r.get("full", {})]
    if ok:
        print("\n=== 汇总（%d 场）===" % len(ok))
        for arm, label in (("full", "full（现实：句内全部窗）"),
                           ("pure", "pure（上界：只用无重叠窗）"),
                           ("hybrid", "hybrid（纯句定中心→全部句分配）")):
            sub = [r for r in ok if "acc" in r.get(arm, {})]
            if not sub:
                continue
            accs = [r[arm]["acc"] for r in sub]
            hits = sum(1 for r in sub if r[arm].get("k") == r[arm].get("k_true"))
            cov = np.mean([r[arm].get("coverage", 0.0) for r in sub])
            print("  %-30s 句级 %.1f%%   K 命中 %d/%d   可用句占比 %.0f%%"
                  % (label, 100 * float(np.mean(accs)), hits, len(sub), 100 * float(cov)))
        print("\n  逐场（句级准确率）：")
        print("    %-14s %7s %8s %8s %8s %7s" % ("meeting", "真值K", "现状", "full", "pure", "hybrid"))
        base = {"R8001_M8004": 45.0, "R8003_M8001": 42.3, "R8007_M8010": 22.2,
                "R8007_M8011": 34.8, "R8008_M8013": 38.1, "R8009_M8018": 77.1,
                "R8009_M8019": 85.7, "R8009_M8020": 84.2}
        for r in ok:
            print("    %-14s %7d %6.1f%% %7.1f%% %7.1f%% %7.1f%%"
                  % (r["meeting"], r["n_truth_speakers"], base.get(r["meeting"], float("nan")),
                     r["full"]["acc"] * 100,
                     r.get("pure", {}).get("acc", float("nan")) * 100,
                     r.get("hybrid", {}).get("acc", float("nan")) * 100))
        print("\n  '现状' = 固定网格窗段表 + 句级主导投票（同一把尺子），全体 53.2%% / 正常场 62.1%%")
        print("  注：pure 用了真值排除重叠，是诊断上界；hybrid 的第二阶段不用真值。")

    out_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("\n已写出 %s" % out_path)


if __name__ == "__main__":
    main()
