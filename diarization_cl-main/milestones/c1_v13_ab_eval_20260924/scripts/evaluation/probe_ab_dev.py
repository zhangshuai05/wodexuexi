# -*- coding: utf-8 -*-
"""A/B 选 K 对照（dev 12 场，真实路径）——见修订规格 rev1。

## 规格要点（`artifacts/v13_ab_dev_spec_rev1_20260924.md`）

- **A**：AHC 距离阈值切树，阈值在 **dev 12 场**上选**一个全局阈值**。
- **B**：归一化谱间隙（`W = max(cos,0)`，`K = argmax(gaps) + 1`），cap **固定 10**。
- 两者**共用**：VAD 门控窗、主窗嵌入、**密窗嵌入**（`_refine_windows` 不依赖 K，
  每场只提取一次——这是成本控制的关键，见 §R5/R6）。
- 细化策略**固定为 v12 现有一级**，A/B 只替换"选 K"。
- 指标：**总体含全部参考句，弃权算未答对**（§R1）；主指标按句平均，辅为时长加权；
  弃权率与未匹配率**分开报**。
- 失败场次**显式记录**（§R7），不填 NaN、不静默跳过。
- **不作迁移结论**：只跑 dev，不碰 holdout / train_L。

    python -u -B scripts/evaluation/probe_ab_dev.py
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from k_candidates import cut_by_distance, spectral_gap_k  # noqa: E402
from probe_sentence_attribution import (  # noqa: E402
    CACHE, DATA, MIN_VAD, VAD_CACHE, WIN, annotate_overlap, attribution,
    read_textgrid, segment_map, sentence_dict_predictions,
    sentences_from_textgrid)

SECONDS = 600.0
B_CAP = 10                      # 修订规格 §R4：B 的 K 上限固定
A_THRESHOLDS = [round(0.2 + 0.1 * i, 1) for i in range(11)]   # 0.2 … 1.2


# ───────────────────── 参考句（与 K 无关，每场只算一次） ─────────────────────

def load_sentences(meeting):
    tiers = read_textgrid(DATA / "TextGrid" / f"{meeting}.TextGrid")
    return annotate_overlap(sentences_from_textgrid(tiers, SECONDS), tiers)


# ───────────────────── 真实路径：注入候选标签，固定两级细化 ─────────────────────

def real_path_labels(labels, E, times, rE, rtimes):
    """给定主窗簇标签，走 v12 的**同一套**细化逻辑，返回最终 (labels, times)。

    与 `v12.diarize()` 的差别只有一处：选 K 由外部给，不调 `_select_candidate`。

    ⚠️ **k<2 时也必须返回原 `times`**，不能返回 None。`times` 是 VAD 门控后的
    真实窗起点（稀疏、覆盖到窗口末尾），而 `arange(n)*hop` 会重建出一条
    从 0 开始的等距网格——本函数初版这么做，导致 422 个窗只铺到 423 秒，
    600 秒参考的后半段全被判成"无覆盖"，弃权假报 37%（离线版同 k 是 5.9%）。
    """
    k = len(set(int(x) for x in labels))
    if k < 2:
        return labels, times, "2s-only"
    anchor_labels = sorted(set(int(x) for x in labels))
    anchors = v11._centroids(E, labels)
    rlabels = v12._refine_labels(rE, anchors, anchor_labels, rtimes)
    if rlabels is not None and len(set(rlabels.tolist())) >= 2:
        return rlabels, rtimes, "refined"
    return labels, times, "2s-only"


def _win(level):
    """细化后是 0.6s 网格，2s-only 是 2.0s 主窗。"""
    return v12.REFINE_WIN_S if level == "refined" else v12.WIN_S


def score(sentences, labels, times, win):
    """把最终标签铺成段表并算句子级归属率（§R1 口径）。

    **永远用传入的 `times`**（真实窗起点），不用 `arange` 重建。
    """
    end = max(s["end"] for s in sentences)
    w = np.full(len(labels), win)
    segs = segment_map(times, labels, w, end)
    pred = sentence_dict_predictions(sentences, segs)
    r = attribution(sentences, pred)["overall"]
    return {"per_sentence": r["per_sentence"], "duration_weighted": r["duration_weighted"],
            "abstain": r["abstain"], "unmapped": r["unmapped"], "n": r["n"],
            "n_sentences": len(sentences)}


# ───────────────────── 主流程 ─────────────────────

def run_meeting(meeting, models, do_a=True):
    """返回该场全部候选的结果。K 无关的中间结果只算一次。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    import soundfile as sf

    wav = str(DATA / "wav" / f"{meeting}.flac")
    info = sf.info(wav)
    stop = min(int(SECONDS * info.samplerate), info.frames)
    audio, rate = sf.read(wav, dtype="float32", stop=stop)
    if audio.ndim > 1:
        audio = audio[:, 0]

    sentences = load_sentences(meeting)

    # ── K 无关：VAD + 主窗 + 密窗（密窗是大头，只付一次）
    vad_segments = v11._vad_segs(models["vad"], wav)
    vad_segments = [(s, min(e, SECONDS)) for s, e in vad_segments if s < SECONDS]
    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E, times, _ = v11._extract_windows(audio, rate, vad_segments, models["spk"])
    rE, rtimes = v12._refine_windows(audio, rate, vad_segments, models["spk"])
    if len(E) < 2:
        return {"meeting": meeting, "insufficient": True, "n_windows": int(len(E))}

    tree = linkage(E, method="average", metric="cosine")
    out = {"meeting": meeting, "insufficient": False, "n_windows": int(len(E)),
           "n_dense": int(len(rE)), "n_sentences": len(sentences), "candidates": {}}

    # ── 基线（现状 v12 的选 K）
    candidates, _ = v11._candidate_clusters(E)
    selected = v11._select_candidate(candidates)
    labels = selected["labels"].copy()
    flabels, ftimes, level = real_path_labels(labels, E, times, rE, rtimes)
    out["candidates"]["base"] = {"k": int(selected["k"]), "level": level,
                                 "sizes": sorted([int((labels == l).sum())
                                                  for l in set(labels.tolist())], reverse=True),
                                 **score(sentences, flabels, ftimes, _win(level))}

    # ── B：谱间隙（cap 固定）
    kb, gaps = spectral_gap_k(E, cap=B_CAP)
    if kb < 2:
        blabels = np.zeros(len(E), dtype=int)
    else:
        blabels = fcluster(tree, t=kb, criterion="maxclust") - 1
        blabels = v11._relabel_by_first_seen(blabels)
    flabels, ftimes, level = real_path_labels(blabels, E, times, rE, rtimes)
    out["candidates"]["B"] = {"k": int(len(set(blabels.tolist()))), "level": level,
                              "spectral_k": int(kb),
                              "sizes": sorted([int((blabels == l).sum())
                                               for l in set(blabels.tolist())], reverse=True),
                              "gap_ratio": (float(gaps.max() / (gaps.sum() + 1e-12))
                                            if len(gaps) else 0.0),
                              **score(sentences, flabels, ftimes, _win(level))}

    # ── A：11 个阈值（共用 tree 与 rE）
    if do_a:
        for t in A_THRESHOLDS:
            # **同时报告 cap 前的原始簇数**：若多个阈值都被 cap 压到同一个 k，
            # 那是 cap 造成的平台期，不是判据在起作用——必须让它在表里可见，
            # 否则会把"常数"误读成"阈值有效"（本项目已有 `largest_eligible`
            # 在 cap 足够大时恒等于 cap 的前车之鉴）。
            raw_k = int(cut_by_distance(tree, t).max()) + 1
            alabels = cut_by_distance(tree, t, n_clusters_cap=v12.MAX_SPEAKERS)
            alabels = v11._relabel_by_first_seen(alabels)
            k = len(set(alabels.tolist()))
            flabels, ftimes, level = real_path_labels(alabels, E, times, rE, rtimes)
            out["candidates"][f"A@{t}"] = {
                "k": int(k), "raw_k_before_cap": raw_k,
                "capped": raw_k > v12.MAX_SPEAKERS, "level": level,
                "sizes": sorted([int((alabels == l).sum())
                                 for l in set(alabels.tolist())], reverse=True),
                **score(sentences, flabels, ftimes, _win(level))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/v13_ab_dev_results.json")
    ap.add_argument("--skip-a", action="store_true")
    args = ap.parse_args()

    from v12_aishell4 import split_meetings
    meetings = split_meetings(DATA, args.split)
    if args.limit:
        meetings = meetings[:args.limit]
    print(f"dev 场次 {len(meetings)}: {meetings}\n", flush=True)

    from web_server import get_models
    models = get_models(lambda m: None, need_stream=False, need_offline=False)

    results, failed = [], []
    for i, meeting in enumerate(meetings, 1):
        t0 = time.perf_counter()
        try:
            r = run_meeting(meeting, models, do_a=not args.skip_a)
            r["elapsed_s"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            c = r["candidates"]
            print(f"[{i}/{len(meetings)}] {meeting}  {r['elapsed_s']:6.1f}s  "
                  f"窗 {r['n_windows']}/{r['n_dense']} 句 {r['n_sentences']}  "
                  f"base k={c['base']['k']} {100*c['base']['per_sentence']:.1f}%  "
                  f"B k={c['B']['k']} {100*c['B']['per_sentence']:.1f}%", flush=True)
        except Exception as exc:  # noqa: BLE001
            # §R7：失败场次**显式记录**，不静默跳过
            failed.append({"meeting": meeting, "error": repr(exc),
                           "traceback": traceback.format_exc()})
            print(f"[{i}/{len(meetings)}] {meeting}  ***失败***  {exc!r}", flush=True)

    payload = {"schema": "v13-ab-dev-v1",
               "spec": "v13_ab_dev_spec_rev1_20260924.md",
               "split": args.split, "seconds": SECONDS, "b_cap": B_CAP,
               "a_thresholds": A_THRESHOLDS,
               "results": results, "failed": failed}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n写入 {args.out}")
    print(f"成功 {len(results)} 场，失败 {len(failed)} 场"
          + (f"：{[f['meeting'] for f in failed]}" if failed else ""))


if __name__ == "__main__":
    main()
