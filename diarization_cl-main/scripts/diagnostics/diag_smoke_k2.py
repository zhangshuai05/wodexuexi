# -*- coding: utf-8 -*-
"""第二轮诊断：smoke 的 3 簇到底是三个真人，还是把一个人切成两半？

第一轮已知：
  - eigengap 在 λ2≈λ3 处没有间隙，结构上只能选 K=2；
  - 强制 K=3 能分出均衡 3 簇，但句序是「长块切分」，切换数 4–7，不是记忆中的 11。

本轮补三类证据：
  1) 谱聚类 K=2..6：原始嵌入空间的 cosine silhouette、质心两两余弦、簇时间跨度/连续块数；
  2) 旧生产链路 v11（0.6s/0.15s 滑窗 + AHC/silhouette）在同一音频上选几人；
  3) v12 生产链路（2.0s/1.0s 长窗 + AHC/silhouette）选几人。
     —— 如果旧链路选 3，说明回归来自「AliMeeting class-chunk + eigengap」这条新链；
        如果旧链路也选 2，那 3 人说法需要重新核对。

用法：
    PYTHONIOENCODING=utf-8 C:/Users/31817/miniconda3/envs/funasr/python.exe \
        scripts/diagnostics/diag_smoke_k2.py > output/diag_smoke_k2.log 2>&1; echo EXIT=$?
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / ".zhang", ROOT / "AliMeeting", ROOT / "streaming_asr",
           ROOT / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import claude_a2_v11 as v11                    # noqa: E402
import claude_a2_v12 as v12                    # noqa: E402
from run_3dspeaker_pipeline import (            # noqa: E402
    class_chunks, embed_chunks, osd_probs, spectral_cluster, turns_to_speakers)
from run_der import load_models                 # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")
OSD = Path(r"D:\yinp\pyannote_seg3\model.onnx")
UTT_FILES = {
    "s_ocrfix_34": ROOT / "output/sessions/s_ocrfix/utterances.jsonl",
    "s_smoke_33": ROOT / "output/sessions/s_smoke/utterances.jsonl",
}


def load_utts(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append({"utt_id": row["utt_id"],
                        "start": row["start_ms"] / 1000.0,
                        "end": row["end_ms"] / 1000.0})
    return out


def time_structure(times, labels, step_s):
    """每个簇的时间结构：总窗数、时间跨度、连续块数、最长块秒数。"""
    order = np.argsort(times)
    times = np.asarray(times)[order]
    labels = np.asarray(labels)[order]
    out = {}
    for lab in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == lab)
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1 or times[b] - times[a] > step_s * 1.5:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        spans = [(float(times[e]) + step_s - float(times[s])) for s, e in runs]
        out[lab] = {
            "n_windows": int(len(idx)),
            "first_s": round(float(times[idx[0]]), 2),
            "last_s": round(float(times[idx[-1]] + step_s), 2),
            "n_runs": len(runs),
            "longest_run_s": round(max(spans), 2) if spans else 0.0,
        }
    return out


def centroid_cos(E, labels):
    cents = v11._centroids(E, np.asarray(labels))
    return np.round(cents @ cents.T, 4).tolist()


def utterance_seq(kept, labels, rlabels, rtimes, turns, win_s):
    if rlabels is not None and len(set(rlabels.tolist())) >= 2:
        rwins = [(float(t), float(t) + win_s) for t in rtimes]
        pred, _ = turns_to_speakers(rwins, rlabels, turns, fallback=(kept, labels))
    else:
        pred, _ = turns_to_speakers(kept, labels, turns)
    switches = sum(1 for a, b in zip(pred, pred[1:]) if a != b)
    return {"n_speakers": len({int(p) for p in pred if p != -1}),
            "switches": switches, "seq": [int(p) for p in pred]}


def lane_report(tag, E, times, step_s, utt_sets):
    print("\n--- %s: %d 窗" % (tag, len(E)), flush=True)
    cands, _ = v11._candidate_clusters(E, max_speakers=min(10, len(E)))
    best = v11._select_candidate(cands)
    print("  AHC 选择 k=%d; 候选: %s"
          % (best["k"], ", ".join(
              "k%d:%.3f%s" % (c["k"], c["score"], "" if c["eligible"] else "(inelig)")
              for c in cands if 2 <= c["k"] <= 6)), flush=True)
    print("  AHC 最优簇时间结构: %s"
          % time_structure(times, best["labels"], step_s), flush=True)
    print("  AHC 质心余弦: %s" % centroid_cos(E, best["labels"]), flush=True)
    out = {"ahc_k": int(best["k"]),
           "ahc_candidates": [{"k": c["k"], "score": round(float(c["score"]), 4),
                               "eligible": bool(c["eligible"]),
                               "sizes": c["sizes"]} for c in cands],
           "ahc_time": time_structure(times, best["labels"], step_s),
           "ahc_centroid_cos": centroid_cos(E, best["labels"]),
           "utt_seq": {}}
    for name, turns in utt_sets.items():
        m = utterance_seq(times, best["labels"], None, None, turns, step_s)
        out["utt_seq"][name] = m
        print("    %s -> %d 人 / %d 切换 seq=%s"
              % (name, m["n_speakers"], m["switches"], m["seq"]), flush=True)
    return out


def main():
    import onnxruntime as ort
    models = load_models()
    osd_sess = ort.InferenceSession(str(OSD), providers=["CPUExecutionProvider"])
    audio, rate = sf.read(str(SMOKE), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    utt_sets = {k: load_utts(p) for k, p in UTT_FILES.items() if p.exists()}
    vad = v11._vad_segs(models["vad"], str(SMOKE))

    # ---------- Part A: AliMeeting class-chunk 链路的 K 几何 ----------
    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins, _ = class_chunks(probs, fs)
    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    kept_times = np.asarray([a for a, _ in kept])
    part_a = {"k_geometry": {}}
    print("=" * 78)
    print("[class-chunk] E=%d fs=%.3f" % (len(E), fs), flush=True)
    for k in range(2, 7):
        labels, k_used, _ = spectral_cluster(E, k)
        sil = float(v11._silhouette(E, labels))
        sizes = [int((labels == x).sum()) for x in sorted(set(labels.tolist()))]
        entry = {"silhouette": round(sil, 4), "sizes": sizes,
                 "centroid_cos": centroid_cos(E, labels),
                 "time": time_structure(kept_times, labels, fs)}
        part_a["k_geometry"][k] = entry
        print("  K=%d silhouette=%.4f sizes=%s 时间结构=%s"
              % (k, sil, sizes, entry["time"]), flush=True)
        print("        质心余弦=%s" % entry["centroid_cos"], flush=True)
    # 细化层在 K=2/3 下的句序
    rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
    part_a["utt_seq"] = {}
    for k in (2, 3):
        labels, _, _ = spectral_cluster(E, k)
        anchors = v11._centroids(E, labels)
        alabs = sorted({int(x) for x in labels.tolist()})
        rlabels = v12._refine_labels(rE, anchors, alabs, rtimes)
        for name, turns in utt_sets.items():
            m = utterance_seq(kept, labels, rlabels, rtimes, turns, v12.REFINE_WIN_S)
            part_a["utt_seq"]["k%d/%s" % (k, name)] = m
            print("  [refine K=%d] %s -> %d 人 / %d 切换 seq=%s"
                  % (k, name, m["n_speakers"], m["switches"], m["seq"]), flush=True)

    # ---------- Part B: 旧生产链路 v11（0.6s/0.15s） ----------
    print("=" * 78)
    E_v11, t_v11, err_v11 = v11._extract_windows(audio, rate, vad, models["spk"])
    part_b = lane_report("v11 0.6s/0.15s", E_v11, t_v11, v11.HOP_S, utt_sets)

    # ---------- Part C: 旧生产链路 v12（2.0s/1.0s） ----------
    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E_v12, t_v12, err_v12 = v11._extract_windows(audio, rate, vad, models["spk"])
    part_c = lane_report("v12 2.0s/1.0s", E_v12, t_v12, v12.HOP_S, utt_sets)

    out = {"class_chunk": part_a, "v11_lane": part_b, "v12_lane": part_c,
           "embedding_errors": {"v11": err_v11, "v12": err_v12}}
    (ROOT / "output" / "diag_smoke_k2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 output/diag_smoke_k2.json", flush=True)


if __name__ == "__main__":
    main()
