# -*- coding: utf-8 -*-
"""密集窗谱几何：0.6s / 2.0s 窗集里到底能不能看出 3 个说话人。

背景：class-chunk（1.5s，只取 OSD 单说话人区间）的嵌入里只有 2 个身份，
K=3 的轮廓系数从 0.30 掉到 0.17，且两簇质心余弦 0.725（= 把一个人切成两半）。
本脚本跳过 OSD，直接用 VAD 滑窗（v11 0.6s / v12 2.0s）看：
  - eigengap 自选 K
  - K=2..7 的 silhouette、质心两两余弦、簇大小
  - 用最优 K 的标签直接做句级映射，数说话人与切换数

用法：
    PYTHONIOENCODING=utf-8 C:/Users/31817/miniconda3/envs/funasr/python.exe \
        scripts/diagnostics/diag_smoke_dense.py > output/diag_smoke_dense.log 2>&1; echo EXIT=$?
"""
import json
import sys
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
    spectral_cluster, turns_to_speakers)
from run_der import load_models                 # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")
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


def geometry(tag, E, times, win_s, utt_sets):
    print("\n" + "=" * 78)
    print("[%s] E=%d  win=%.2fs" % (tag, len(E), win_s), flush=True)
    labels_auto, k_auto, eigs = spectral_cluster(E, None)
    vals = np.sort(eigs)[: 9]
    print("  eigengap k_auto=%d eigvals=%s gaps=%s"
          % (k_auto, np.round(vals, 5).tolist(), np.round(np.diff(vals), 5).tolist()),
          flush=True)
    out = {"n_windows": len(E), "win_s": win_s, "k_auto": int(k_auto),
           "eigvals": vals.tolist(), "k_geometry": {}, "utt_seq": {}}
    for k in range(2, 8):
        if k >= len(E):
            break
        labels, _, _ = spectral_cluster(E, k)
        sil = float(v11._silhouette(E, labels))
        sizes = {int(x): int((labels == x).sum()) for x in sorted(set(labels.tolist()))}
        cents = v11._centroids(E, labels)
        cos = np.round(cents @ cents.T, 4).tolist()
        print("  K=%d silhouette=%.4f sizes=%s centroid_cos=%s"
              % (k, sil, sizes, cos), flush=True)
        out["k_geometry"][k] = {"silhouette": round(sil, 4), "sizes": sizes,
                                "centroid_cos": cos}
    for k in (k_auto, 3):
        labels, _, _ = spectral_cluster(E, k)
        wins = [(float(t), float(t) + win_s) for t in times]
        for name, turns in utt_sets.items():
            pred, _ = turns_to_speakers(wins, labels, turns)
            sw = sum(1 for a, b in zip(pred, pred[1:]) if a != b)
            item = {"n_speakers": len({int(p) for p in pred if p != -1}),
                    "switches": sw, "seq": [int(p) for p in pred]}
            out["utt_seq"]["k%d/%s" % (k, name)] = item
            print("    K=%d %s -> %d 人 / %d 切换 seq=%s"
                  % (k, name, item["n_speakers"], item["switches"], item["seq"]),
                  flush=True)
    return out


def main():
    models = load_models()
    audio, rate = sf.read(str(SMOKE), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    utt_sets = {k: load_utts(p) for k, p in UTT_FILES.items() if p.exists()}
    vad = v11._vad_segs(models["vad"], str(SMOKE))

    E11, t11, err11 = v11._extract_windows(audio, rate, vad, models["spk"])
    part11 = geometry("v11 0.6s/0.15s", E11, t11, v11.WIN_S, utt_sets)

    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E12, t12, err12 = v11._extract_windows(audio, rate, vad, models["spk"])
    part12 = geometry("v12 2.0s/1.0s", E12, t12, v12.WIN_S, utt_sets)

    out = {"v11_06": part11, "v12_20": part12,
           "embedding_errors": {"v11": err11, "v12": err12}}
    (ROOT / "output" / "diag_smoke_dense.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 output/diag_smoke_dense.json", flush=True)


if __name__ == "__main__":
    main()
