# -*- coding: utf-8 -*-
"""一次性诊断：smoke_clip_180s.wav 为何从 3 人退化成 2 人。

对比两组条件：
  1) 原始电平 / OSD class 窗
  2) peak 归一化到 0.85 / OSD class 窗
并分别检查：
  - 谱聚类 eigengap 自选 K vs 强制 K=3 的簇大小
  - v11 AHC 候选（silhouette）会不会选到 3
  - 细化层（0.6s 短窗）在两种 K 下的标签分布
  - 用 s_smoke / s_ocrfix 两份真实 utt 列表做句级映射，数说话人与切换数

用法（模型加载约 90s，全脚本 5-10 分钟）：
    PYTHONIOENCODING=utf-8 C:/Users/31817/miniconda3/envs/funasr/python.exe \
        scripts/diagnostics/diag_smoke_k.py > output/diag_smoke_k.log 2>&1; echo EXIT=$?
"""
import json
import sys
import tempfile
from collections import Counter
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
    class_chunks, chunk_windows, eigengap_k, embed_chunks, osd_probs,
    spectral_cluster, turns_to_speakers)
from run_der import load_models                 # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")
OSD = Path(r"D:\yinp\pyannote_seg3\model.onnx")
UTT_FILES = {
    "s_smoke_33": ROOT / "output/sessions/s_smoke/utterances.jsonl",
    "s_ocrfix_34": ROOT / "output/sessions/s_ocrfix/utterances.jsonl",
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


def clusters_info(labels):
    return {int(k): int(v) for k, v in Counter(labels.tolist()).items()}


def map_and_count(kept, labels, rwins, rlabels, turns):
    """句级映射 + 说话人数 + 切换数（只用预测，不读真值）。"""
    if rwins is not None:
        pred, cover = turns_to_speakers(rwins, rlabels, turns,
                                        fallback=(kept, labels))
    else:
        pred, cover = turns_to_speakers(kept, labels, turns)
    speakers = sorted({int(p) for p in pred if p != -1})
    switches = sum(1 for a, b in zip(pred, pred[1:]) if a != b)
    return {
        "n_speakers": len(speakers),
        "speakers": speakers,
        "switches": switches,
        "coverage": cover / max(len(turns), 1),
        "seq": [int(p) for p in pred],
    }


def analyze(tag, audio, rate, models, osd_sess, utt_sets):
    print("\n" + "=" * 78)
    print("[%s] peak=%.4f rms=%.4f dur=%.1fs"
          % (tag, float(np.max(np.abs(audio))),
             float(np.sqrt(np.mean(audio ** 2))), len(audio) / rate), flush=True)
    tmp = Path(tempfile.gettempdir()) / ("diag_smoke_%s.wav" % tag)
    sf.write(tmp, audio, rate, subtype="PCM_16")
    vad = v11._vad_segs(models["vad"], str(tmp))
    wins_vad = chunk_windows(vad)
    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins_cls, n_intervals = class_chunks(probs, fs)
    E, kept = embed_chunks(models["spk"], audio, rate, wins_cls)

    labels, k_auto, eigs = spectral_cluster(E, None)
    vals = np.sort(eigs)[:9]
    gaps = np.diff(vals)
    print("  VAD 段=%d / VAD 窗=%d / class 区间=%d / class 窗=%d / 嵌入=%d"
          % (len(vad), len(wins_vad), n_intervals, len(wins_cls), len(E)),
          flush=True)
    print("  谱聚类 k_auto=%d  簇大小=%s" % (k_auto, clusters_info(labels)))
    print("  eigvals[:9]=%s" % np.round(vals, 5).tolist())
    print("  gaps[:8]   =%s  -> eigengap_k=%d"
          % (np.round(gaps, 5).tolist(), eigengap_k(eigs)), flush=True)

    # v11 AHC 候选（旧链路同款 silhouette 选择）
    cands, _ = v11._candidate_clusters(E, max_speakers=min(15, len(E)))
    best = v11._select_candidate(cands)
    print("  AHC 选择 k=%d（eligible=%s）; 候选: %s"
          % (best["k"], best.get("eligible"),
             ", ".join("k%d:%.3f%s" % (c["k"], c["score"], "" if c["eligible"] else "(inelig)")
                       for c in cands if 2 <= c["k"] <= 6)), flush=True)

    labels3, _, _ = spectral_cluster(E, min(3, len(E) - 1))
    print("  强制 K=3 簇大小=%s" % clusters_info(labels3), flush=True)

    rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
    rwins = [(float(t), float(t) + v12.REFINE_WIN_S) for t in rtimes]
    print("  细化窗=%d" % len(rtimes), flush=True)

    results = {}
    for k_name, lab in (("auto_k%d" % k_auto, labels),
                        ("forced_k3", labels3)):
        anchors = v11._centroids(E, lab)
        alabs = sorted({int(x) for x in lab.tolist()})
        rlabels = v12._refine_labels(rE, anchors, alabs, rtimes)
        use_rw, use_rl = (rwins, rlabels) if (
            rlabels is not None and len(set(rlabels.tolist())) >= 2) else (None, None)
        info = {
            "chunk_clusters": clusters_info(lab),
            "refine_ok": rlabels is not None,
            "refine_clusters": clusters_info(rlabels) if rlabels is not None else None,
            "per_utt_set": {},
        }
        for utt_name, turns in utt_sets.items():
            m = map_and_count(kept, lab, use_rw, use_rl, turns)
            info["per_utt_set"][utt_name] = m
            print("    %-10s %-9s -> %d 人 / %d 切换 / 覆盖 %.0f%%  seq(前20)=%s"
                  % (k_name, utt_name, m["n_speakers"], m["switches"],
                     m["coverage"] * 100, m["seq"][:20]), flush=True)
        results[k_name] = info
    return {
        "vad_segments": len(vad), "vad_windows": len(wins_vad),
        "class_intervals": n_intervals, "class_windows": len(wins_cls),
        "embeddings": len(E), "k_auto": int(k_auto),
        "eigvals": vals.tolist(), "gaps": gaps.tolist(),
        "ahc_best_k": int(best["k"]),
        "ahc_candidates": [{"k": c["k"], "score": round(float(c["score"]), 4),
                            "eligible": bool(c["eligible"])} for c in cands],
        "forced_k3_clusters": clusters_info(labels3),
        "refine_windows": len(rtimes),
        "variants": results,
    }


def main():
    import onnxruntime as ort
    models = load_models()
    osd_sess = ort.InferenceSession(str(OSD), providers=["CPUExecutionProvider"])
    audio, rate = sf.read(str(SMOKE), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    utt_sets = {name: load_utts(path) for name, path in UTT_FILES.items() if path.exists()}
    print("utt 列表: %s" % {k: len(v) for k, v in utt_sets.items()}, flush=True)

    out = {
        "raw": analyze("raw", audio, rate, models, osd_sess, utt_sets),
    }
    peak = float(np.max(np.abs(audio)))
    gain = 0.85 / max(peak, 1e-9)
    if gain > 1.0:  # 只提拉弱音，不衰减
        norm = audio * gain
        print("\n[normalize] gain=%.3f（peak %.4f -> 0.85）" % (gain, peak), flush=True)
        out["normalized"] = analyze("norm085", norm, rate, models, osd_sess, utt_sets)
    else:
        print("\n[normalize] peak 已 >=0.85，跳过", flush=True)

    out_path = ROOT / "output" / "diag_smoke_k.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
