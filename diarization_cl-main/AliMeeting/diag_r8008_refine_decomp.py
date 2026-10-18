# -*- coding: utf-8 -*-
"""困境六·逐段比对：R8008_M8013 细化 −10.2 的归因（2026-10-15）。

confine 实验已证伪"跨边界污染窗"（confine 40.0% ≈ 裸细化 40.3%），
接线嫌疑已排除（7/8 场细化为正；_centroids 与 anchor_labels 同序）。
剩下三个候选机制，本脚本一次跑完并排：

  A. 句级映射翻转：chunk 票与 refine 票各自做贪心映射，若两者把同一个簇
     对到不同真值人，−10.2 可能只是评分假象而非窗标签变差。
  B. 坏质心：逐簇看 chunk 窗上的真值帧分布，某个簇混了两个人 ⇒ 细化锚点被拖走。
  C. 内部噪声：0.6s 窗在区间内部也大量误判（margin 分布 + 帧级 GT 对照）。

输出：AliMeeting/results/diag_r8008_refine_decomp.json + 控制台汇总。
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav
from probe_sentence_attribution import (read_textgrid,
                                       sentences_from_textgrid)
from run_3dspeaker_pipeline import (class_chunks, embed_chunks, osd_probs,
                                    spectral_cluster)

_, read_rttm = der_tools()


def greedy_mapping(pairs):
    """chunks_to_turns 同款贪心映射（按时长），返回 {gt: pred}。"""
    mapping, used = {}, set()
    for (g, p), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="R8008_M8013")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meeting = args.meeting
    t0 = time.time()
    from run_der import load_models
    import onnxruntime as ort
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12mod

    models = load_models()
    osd_sess = ort.InferenceSession(r"D:\yinp\pyannote_seg3\model.onnx",
                                    providers=["CPUExecutionProvider"])
    wav = resolve_wav(meeting)
    audio, rate = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    turns = sentences_from_textgrid(
        read_textgrid(Path(r"D:\yinp\AliMeeting\Eval_Ali\Eval_Ali_far\textgrid_dir")
                      / f"{meeting}.TextGrid"))
    ref = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    vad = v11._vad_segs(models["vad"], str(wav))

    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins, _ = class_chunks(probs, fs)
    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    labels, k, _ = spectral_cluster(E, None)

    anchors = v11._centroids(E, labels)
    alabs = sorted(set(int(x) for x in labels.tolist()))
    rE, rtimes = v12mod._refine_windows(audio, rate, vad, models["spk"])
    rlabels = v12mod._refine_labels(rE, anchors, alabs, rtimes)
    scores = rE @ anchors.T
    margins = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]

    # ── 机制 B：逐簇纯度（chunk 窗 span 上的真值帧分布）─────────────────
    gt_speakers = sorted({s["speaker"] for s in ref})
    cluster_gt = {c: defaultdict(float) for c in alabs}
    for (a, b), lab in zip(kept, labels.tolist()):
        for seg in ref:
            ov = min(b, seg["end"]) - max(a, seg["start"])
            if ov > 0:
                cluster_gt[lab][seg["speaker"]] += ov
    purity = {}
    for c in alabs:
        tot = sum(cluster_gt[c].values())
        best = max(cluster_gt[c].values()) if tot else 0
        purity[c] = {"total_s": round(tot, 1),
                     "top_gt_share": round(best / tot, 3) if tot else None,
                     "dist": {g: round(v, 1) for g, v in
                              sorted(cluster_gt[c].items(), key=lambda kv: -kv[1])}}

    # ── 机制 A：两臂各自的句级映射是否一致 ─────────────────────────────
    def sentence_votes(windows, win_labels):
        """每句 (主标签, 各簇时长)。"""
        out = []
        for turn in turns:
            tally = defaultdict(float)
            for (a, b), lab in zip(windows, win_labels):
                ov = min(b, turn["end"]) - max(a, turn["start"])
                if ov > 0:
                    tally[lab] += ov
            if not tally:
                mid = (turn["start"] + turn["end"]) / 2.0
                near = min(range(len(windows)),
                           key=lambda i: abs((windows[i][0] + windows[i][1]) / 2.0 - mid))
                tally[win_labels[near]] = 1.0
            out.append(tally)
        return out

    cw_votes = sentence_votes(kept, labels.tolist())
    rw_votes = sentence_votes([(float(t), float(t) + v12mod.REFINE_WIN_S)
                               for t in rtimes], rlabels.tolist())

    def arm_map(votes):
        pairs = defaultdict(float)
        for turn, tally in zip(turns, votes):
            for lab, dur in tally.items():
                pairs[(turn["speaker"], lab)] += dur
        return greedy_mapping(pairs)

    map_chunk, map_refine = arm_map(cw_votes), arm_map(rw_votes)

    # ── 逐句分解：改错 / 改对 / 双错 / 双对 ────────────────────────────
    # map_*: {真值说话人: 簇}；句对 ⟺ 该句主导簇 == map.get(真值说话人)
    flips = defaultdict(int)
    flip_examples = []
    for i, turn in enumerate(turns):
        okc = (max(cw_votes[i], key=lambda x: cw_votes[i][x])
               == map_chunk.get(turn["speaker"]))
        okr = (max(rw_votes[i], key=lambda x: rw_votes[i][x])
               == map_refine.get(turn["speaker"]))
        key = ("chunk对→refine错" if okc and not okr else
               "chunk错→refine对" if not okc and okr else
               "双对" if okc else "双错")
        flips[key] += 1
        if key == "chunk对→refine错" and len(flip_examples) < 8:
            flip_examples.append({"i": i, "start": round(turn["start"], 2),
                                  "end": round(turn["end"], 2),
                                  "gt": turn["speaker"],
                                  "chunk_tally": {str(kk): round(v, 2) for kk, v in cw_votes[i].items()},
                                  "refine_tally": {str(kk): round(v, 2) for kk, v in rw_votes[i].items()}})

    out = {
        "meeting": meeting,
        "k_auto": int(k),
        "map_chunk": {g: int(p) for g, p in map_chunk.items()},
        "map_refine": {g: int(p) for g, p in map_refine.items()},
        "mapping_same": map_chunk == map_refine,
        "cluster_purity": {str(c): v for c, v in purity.items()},
        "margin_stats": {"p10": round(float(np.percentile(margins, 10)), 3),
                         "p50": round(float(np.percentile(margins, 50)), 3),
                         "lt_0.18_share": round(float((margins < 0.18).mean()), 3)},
        "sentence_flips": dict(flips),
        "flip_examples": flip_examples,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = RESULTS / (args.out or f"diag_refine_decomp_{meeting}.json")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== %s 细化归因 ===" % meeting)
    print("映射一致：%s" % out["mapping_same"])
    print("  chunk 臂映射：%s" % out["map_chunk"])
    print("  refine 臂映射：%s" % out["map_refine"])
    print("逐簇纯度：")
    for c, v in out["cluster_purity"].items():
        print("  簇%s：top 真值占比 %s，分布 %s" % (c, v["top_gt_share"], v["dist"]))
    print("margin：p10=%(p10)s p50=%(p50)s <0.18 占比 %(lt_0.18_share)s" % out["margin_stats"])
    print("逐句分解：%s" % out["sentence_flips"])
    print("已写出 %s" % out_path)


if __name__ == "__main__":
    main()
