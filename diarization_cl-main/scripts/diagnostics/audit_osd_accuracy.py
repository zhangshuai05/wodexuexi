# -*- coding: utf-8 -*-
"""pyannote/segmentation-3.0 ONNX 的重叠检测精度验证（先验 OSD，再接架构）。

**背景**：我们实测"在声纹空间里判纯度"全部失败 ——
  · 余弦 margin（骑墙派）：K 塌陷时无从下手
  · 熵（Top1/Top2 比值）：真值重叠 57.4% 的场只标出 13%，召回仅约 1/4
所以句级 pure 臂那个 80.2% 上界一直无法自动逼近。

`csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/model.onnx`（5.99 MB，免 gated）
在**声学层**直接输出 7 分类 powerset。本脚本只回答一件事：**它的重叠检测准不准**。
不接架构、不改流程，先用真值把 precision/recall 量出来，再决定要不要用。

输出语义（pyannote 3.0 powerset，本脚本先核实 exp 后是否归一化）：
    class 0 静音 | class 1-3 单人 | class 4-6 双人重叠

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_osd_accuracy.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav  # noqa: E402

MODEL = Path(r"D:\yinp\pyannote_seg3\model.onnx")
WIN, STEP, FRAMES = 160000, 80000, 589          # 10s 窗 / 5s 步长 / 589 帧
FRAME_S = (WIN / FRAMES) / 16000.0              # 16.98 ms
DEFAULT = ["R8007_M8010", "R8008_M8013", "R8009_M8020"]
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

_, read_rttm = der_tools()


def predict_probs(sess, audio):
    """整场音频 -> 每帧 7 类概率（重叠窗取平均）。"""
    total = len(audio)
    n_frames = int(np.ceil(total / (WIN / FRAMES))) + 8
    acc = np.zeros((n_frames, 7), np.float64)
    cnt = np.zeros(n_frames, np.float64)
    name = sess.get_inputs()[0].name
    for start in range(0, max(1, total), STEP):
        chunk = audio[start:start + WIN]
        valid = len(chunk)
        if valid < WIN:
            chunk = np.pad(chunk, (0, WIN - valid))
        out = sess.run(None, {name: chunk[None, None, :].astype(np.float32)})[0][0]
        probs = np.exp(out) if out.max() <= 0 else out        # log-softmax 还是已归一化
        vf = min(FRAMES, int(round(valid / (WIN / FRAMES))))
        sf_ = int(round(start / (WIN / FRAMES)))
        ef = min(n_frames, sf_ + vf)
        if ef <= sf_:
            continue
        acc[sf_:ef] += probs[:ef - sf_]
        cnt[sf_:ef] += 1.0
        if start + WIN >= total:
            break
    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return acc, mask


def truth_overlap(segments, times):
    """每个时间点（帧中心）的活跃说话人数。"""
    out = np.zeros(len(times), np.int8)
    for i, t in enumerate(times):
        out[i] = len({s["speaker"] for s in segments if s["start"] <= t < s["end"]})
    return out


def prf(pred, true):
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def class_purity(probs, times, segments, win_s=10.0):
    """powerset 的局部说话人类别（class 1/2/3）可不可信？

    **为什么测这个**：chunk 有 90% 跨说话人边界，根因是 VAD 段只认"有没有语音"、
    不认"是谁在说"。而 powerset 的 class 1/2/3 是"局部 speaker 1/2/3 单独说话"——
    如果这些类别真的对应某个人，就能**在窗内按类别切出单人片段**，
    把跨界率从 90% 压下去。跨窗编号不连续没关系，匹配只在窗内做。

    做法：每个 10s 窗内用匈牙利匹配把类别对到真值说话人，再算逐帧一致率。
    一致率高 = 类别可信 = 值得用它切 chunk。
    """
    from scipy.optimize import linear_sum_assignment

    argmax = probs.argmax(axis=1)
    n = len(probs)
    win = max(1, int(win_s / FRAME_S))
    ok = tot = 0
    for s in range(0, n, win):
        e = min(n, s + win)
        cls, ts = argmax[s:e], times[s:e]
        truth = []
        for t in ts:
            spk = None
            for seg in segments:
                if seg["start"] <= t < seg["end"]:
                    spk = seg["speaker"]
                    break
            truth.append(spk)
        single = np.isin(cls, (1, 2, 3))
        cats = sorted(set(cls[single].tolist()))
        spks = sorted({x for x in truth if x})
        if len(cats) < 2 or len(spks) < 2:
            continue
        cost = np.zeros((len(cats), len(spks)))
        for i, c in enumerate(cats):
            idx = np.flatnonzero(single & (cls == c))
            for j, sp in enumerate(spks):
                cost[i, j] = sum(1 for k in idx if truth[k] == sp)
        rows, cols = linear_sum_assignment(-cost)
        mp = {cats[i]: spks[j] for i, j in zip(rows, cols)}
        for k in np.flatnonzero(single):
            tot += 1
            ok += int(mp.get(int(cls[k])) == truth[k])
    return ok / max(tot, 1), tot


def main():
    if not MODEL.exists():
        raise SystemExit("缺 %s（先用 hf_hub 下载 csukuangfj/sherpa-onnx-pyannote-segmentation-3-0）"
                         % MODEL)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    rows = []
    for meeting in DEFAULT:
        audio, rate = sf.read(str(resolve_wav(meeting)), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        assert rate == 16000, rate
        probs, mask = predict_probs(sess, audio)
        times = (np.arange(len(probs)) + 0.5) * FRAME_S
        keep = mask & (times < len(audio) / rate)
        p = probs[keep]
        t = times[keep]
        # 归一化自检
        norm_err = float(np.abs(p.sum(axis=1) - 1.0).mean())
        segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        active = truth_overlap(segments, t)
        speech_pred = p[:, 1:7].sum(axis=1)
        overlap_pred = p[:, 4:7].sum(axis=1)
        true_speech = active >= 1
        true_overlap = active >= 2
        sp, sr_, sf1, *_ = prf(speech_pred > 0.5, true_speech)
        purity, n_single_frames = class_purity(p, t, segments)
        row = {"meeting": meeting, "n_frames": int(keep.sum()),
               "norm_err": round(norm_err, 5),
               "class_purity": round(purity, 4),
               "n_single_frames": int(n_single_frames),
               "truth_overlap_rate": float(true_overlap.mean()),
               "speech_prf": [round(sp, 3), round(sr_, 3), round(sf1, 3)],
               "osd": {}}
        print("\n=== %s ===" % meeting)
        print("  帧数 %d  概率归一化误差 %.5f  真值重叠帧占比 %.1f%%"
              % (keep.sum(), norm_err, 100 * true_overlap.mean()))
        print("  语音检测(class1-6) P=%.3f R=%.3f F1=%.3f" % (sp, sr_, sf1))
        print("  **powerset 类别纯度（class1-3 对到真值后的逐帧一致率）= %.1f%%**  (%d 单人帧)"
              % (purity * 100, n_single_frames))
        print("  重叠检测(class4-6) 阈值扫描：")
        for th in THRESHOLDS:
            pp, rr, ff, tp, fp, fn = prf(overlap_pred > th, true_overlap)
            row["osd"][str(th)] = {"p": round(pp, 3), "r": round(rr, 3),
                                   "f1": round(ff, 3), "tp": tp, "fp": fp, "fn": fn}
            print("     th=%.1f  P=%.3f  R=%.3f  F1=%.3f   (tp=%d fp=%d fn=%d)"
                  % (th, pp, rr, ff, tp, fp, fn))
        rows.append(row)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "osd_accuracy_audit.json").write_text(
            json.dumps({"config": {"model": str(MODEL), "frame_s": FRAME_S,
                                   "note": "class 0=静音 1-3=单人 4-6=双人重叠；与真值帧级对比"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n判据参考：熵法在重叠率 57.4%% 的场上召回只有约 0.13。")
    print("若 OSD 召回 >0.7，就值得接进句级 pure 架构；否则这条路也不通。")
    print("已写出 %s" % (RESULTS / "osd_accuracy_audit.json"))


if __name__ == "__main__":
    main()
