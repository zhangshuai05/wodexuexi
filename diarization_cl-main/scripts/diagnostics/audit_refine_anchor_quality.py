# -*- coding: utf-8 -*-
"""细化层接线定位：两组质心在 0.6s 细化窗上的判定质量对比。

**要回答的问题**（困境六 · 第 2 步，判据指向"接线问题"之后）：
同一份细化代码（`v12._refine_windows` / `_refine_labels`），在生产链路上值 **+23.8**
（R8008: 14.3% → 38.1%），在新链路上却 **−10.2**（50.5% → 40.3%）。
两处唯一的实质差别是**喂给细化的簇质心**：

    A. 生产口径：2.0s/1.0s 网格窗 + AHC 聚类 的质心
    B. 新链路　：1.5s class 区间窗 + 谱聚类   的质心

细化是「0.6s 短窗按最近质心判归属」，所以质心代表性直接决定它的成败。
本脚本在同一批 0.6s 细化窗上，让两组质心各判一次，比匈牙利映射后的准确率。

判据（事前写死）
    A 明显高于 B（≥8 pt） ⇒ 质心是元凶，改法是给细化单独配更稳的锚点（或主窗换回 2.0s）
    A ≈ B（<8 pt）        ⇒ 质心无罪，嫌疑转向 `_refine_labels` 里从 v7 继承的
                             SLIVER_MIN_DUR / SLIVER_MARGIN（为"2.0s 主窗"调的）

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_refine_anchor_quality.py --meetings R8008_M8013 R8007_M8011
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr",
           REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav  # noqa: E402
from run_der import load_models  # noqa: E402
from run_3dspeaker_pipeline import (class_chunks, osd_probs,  # noqa: E402
                                    spectral_cluster)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

_, read_rttm = der_tools()
OSD_MODEL = Path(r"D:\yinp\pyannote_seg3\model.onnx")
PROD_WIN, PROD_HOP = 2.0, 1.0
JUDGE_GAP = 0.08          # A 比 B 高这么多点才算"质心是元凶"


def prod_windows(audio, rate, vad, spk):
    """生产口径主窗（2.0s/1.0s）：复用 v11._extract_windows，临时改它的模块全局。

    ⚠️ v11 的窗长是**模块级全局**，`_extract_windows` 在函数体里读它 —— 所以改全局有效；
    但绝不能用默认参数传（那在 def 时就绑定了，Task 2 踩过）。
    """
    old = (v11.WIN_S, v11.HOP_S, v11.MAX_SPEAKERS)
    v11.WIN_S, v11.HOP_S, v11.MAX_SPEAKERS = PROD_WIN, PROD_HOP, 15
    try:
        out = v11._extract_windows(audio, rate, vad, spk)
    finally:
        v11.WIN_S, v11.HOP_S, v11.MAX_SPEAKERS = old
    return out[0], out[1]


def ahc_centroids(E):
    cands, _ = v11._candidate_clusters(E, max_speakers=15)
    best = v11._select_candidate(cands)
    return v11._centroids(E, best["labels"]), int(best["k"])


def truth_at_windows(ref, times, win_s, tol=0.05):
    """每个细化窗的真值说话人（窗内只属一人时才可标，否则 None）。"""
    out = []
    for t in times:
        a, b = float(t), float(t) + win_s
        owners = {s["speaker"] for s in ref if min(b, s["end"]) - max(a, s["start"]) > tol}
        out.append(next(iter(owners)) if len(owners) == 1 else None)
    return out


def judge(centroids, rE, truth, speakers):
    """用给定质心判细化窗归属 -> 匈牙利映射 -> 准确率。"""
    scores = rE @ centroids.T
    pred = scores.argmax(axis=1)
    cats = sorted(set(pred.tolist()))
    if not cats or not speakers:
        return 0.0, 0
    cost = np.zeros((len(cats), len(speakers)))
    for i, c in enumerate(cats):
        idx = np.flatnonzero(pred == c)
        for j, s in enumerate(speakers):
            cost[i, j] = sum(1 for k in idx if truth[k] == s)
    rows, cols = linear_sum_assignment(-cost)
    mp = {cats[i]: speakers[j] for i, j in zip(rows, cols)}
    ok = sum(1 for k in range(len(pred)) if mp.get(int(pred[k])) == truth[k])
    return ok / len(pred), len(cats)


def run_meeting(meeting, models, osd_sess):
    wav = resolve_wav(meeting)
    audio, rate = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    ref = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    speakers = sorted({s["speaker"] for s in ref})
    vad = v11._vad_segs(models["vad"], str(wav))

    # A. 生产口径质心（2.0s 窗 + AHC）
    Ea, _ = prod_windows(audio, rate, vad, models["spk"])
    cents_a, k_a = ahc_centroids(Ea)

    # B. 新链路质心（1.5s class 区间窗 + 谱聚类）
    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins, _ = class_chunks(probs, fs)
    Eb, _ = None, None
    from run_3dspeaker_pipeline import embed_chunks
    Eb, kept_b = embed_chunks(models["spk"], audio, rate, wins)
    labels_b, k_b, _ = spectral_cluster(Eb, None)
    cents_b = v11._centroids(Eb, labels_b)

    # 同一批 0.6s 细化窗
    rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
    truth_all = truth_at_windows(ref, rtimes, v12.REFINE_WIN_S)
    idx = [i for i, t in enumerate(truth_all) if t is not None]
    if len(idx) < 20:
        return None
    rE2 = rE[idx]
    truth2 = [truth_all[i] for i in idx]

    acc_a, _ = judge(cents_a, rE2, truth2, speakers)
    acc_b, _ = judge(cents_b, rE2, truth2, speakers)
    return {"meeting": meeting, "n_refine_windows": int(len(rtimes)),
            "n_labelled": len(idx), "n_speakers": len(speakers),
            "k_prod_ahc": k_a, "k_new_spectral": int(k_b),
            "acc_prod_centroids": acc_a, "acc_new_centroids": acc_b,
            "gap": acc_a - acc_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*",
                    default=["R8008_M8013", "R8007_M8011"])
    ap.add_argument("--out", default="refine_anchor_quality.json")
    args = ap.parse_args()

    models = load_models()
    osd_sess = ort.InferenceSession(str(OSD_MODEL), providers=["CPUExecutionProvider"])
    rows = []
    print("%-14s %7s %7s | %8s %8s | %8s %8s | %s"
          % ("meeting", "细化窗", "可标", "生产质心", "新链质心", "Kprod", "Knew", "gap"),
          flush=True)
    for meeting in args.meetings:
        t0 = time.time()
        r = run_meeting(meeting, models, osd_sess)
        if r is None:
            print("  %s 可标窗不足，跳过" % meeting, flush=True)
            continue
        r["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print("%-14s %7d %7d | %7.1f%% %7.1f%% | %8d %8d | %+.1f pt"
              % (meeting, r["n_refine_windows"], r["n_labelled"],
                 r["acc_prod_centroids"] * 100, r["acc_new_centroids"] * 100,
                 r["k_prod_ahc"], r["k_new_spectral"], r["gap"] * 100), flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"note": "同一批 0.6s 细化窗上，两组质心的判定准确率；"
                                           "匈牙利映射后统计；judge_gap=%.2f" % JUDGE_GAP},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        gap = float(np.mean([r["gap"] for r in rows]))
        print("\n=== 判据 ===")
        print("  平均 gap = %+.1f pt（生产质心 − 新链质心）" % (gap * 100))
        print("  结论：%s" % ("质心是元凶 —— 给细化单独配更稳的锚点，或主窗换回 2.0s"
                             if gap >= JUDGE_GAP else
                             "质心无罪 —— 嫌疑转向 _refine_labels 的 SLIVER_* 参数"
                             "（为 2.0s 主窗调的，换到 1.5s 主窗失配）"))
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
