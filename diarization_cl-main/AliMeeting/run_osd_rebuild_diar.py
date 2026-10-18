# -*- coding: utf-8 -*-
"""用 OSD 的帧级输出**重建窗**：只在"单人且连续"的区间上提嵌入，再聚类。

**为什么换这个思路**（前两轮实测的教训）：
  · 固定网格窗不管说话人边界 -> 窗里混两个人的声纹 -> 毒化整个嵌入空间
    （R8007_M8010 上窗级/句级聚类全部塌陷到 K=1，silhouette ≤ 0）
  · "先找纯句再聚类"这场不可行：句重叠率中位 1.00，<0.30 的只有 3.3%，
    OSD 剥离后仍 ≥1.2s 纯净音频的句子 **0 个**
  · 但 OSD 本身很准（重叠召回 0.890 / F1 0.877，语音 F1 0.99，13 秒/场）

**所以把 OSD 用对地方**：不用它剔样本，而是用它**决定窗在哪**。
`pyannote/segmentation-3.0` 的 7 类 powerset 里，class 1-3 是单人、
class 4-6 是双人重叠 —— 取"单人帧的连续区间"，在这些区间里取 2.0s 窗，
窗与窗之间就不会横跨说话人切换点。

    臂 grid   : 固定网格窗（现状做法，复现塌陷）
    臂 osdwin : OSD 单人连续区间内的窗

评测：窗级标签 -> 句级主导投票 -> 全局映射 -> 句级准确率（两臂同一把尺子）。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_osd_rebuild_diar.py --meetings R8007_M8010
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "streaming_asr", HERE.parent / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, RESULTS, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)
from run_der import load_models  # noqa: E402

OSD_MODEL = Path(r"D:\yinp\pyannote_seg3\model.onnx")
WIN_W, STEP_W, FRAMES = 160000, 80000, 589
FRAME_STEP = WIN_W / FRAMES               # 271.65 采样点 = 16.98 ms
FRAME_STEP_S = FRAME_STEP / 16000.0
WIN_S, HOP_S = 2.0, 1.0
MIN_CHUNK = 0.35
MIN_RUN_S = 0.60          # 单人连续区间的最短长度
OSD_TH = 0.30             # 重叠帧判定阈值
SPEECH_TH = 0.50
MAX_SPEAKERS = 10
GATE = (2, 0.010)

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def osd_frames(sess, audio):
    """整场 -> 帧级 (speech_prob, overlap_prob, valid, argmax_class)。"""
    total = len(audio)
    n = int(np.ceil(total / FRAME_STEP)) + 8
    acc = np.zeros((n, 7), np.float64)
    cnt = np.zeros(n, np.float64)
    name = sess.get_inputs()[0].name
    for start in range(0, max(1, total), STEP_W):
        chunk = audio[start:start + WIN_W]
        valid = len(chunk)
        if valid < WIN_W:
            chunk = np.pad(chunk, (0, WIN_W - valid))
        out = sess.run(None, {name: chunk[None, None, :].astype(np.float32)})[0][0]
        probs = np.exp(out) if out.max() <= 0 else out
        vf = min(FRAMES, int(round(valid / FRAME_STEP)))
        sf_ = int(round(start / FRAME_STEP))
        ef = min(n, sf_ + vf)
        if ef <= sf_:
            continue
        acc[sf_:ef] += probs[:ef - sf_]
        cnt[sf_:ef] += 1.0
        if start + WIN_W >= total:
            break
    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return (acc[:, 1:7].sum(axis=1), acc[:, 4:7].sum(axis=1), mask,
            acc.argmax(axis=1))


def single_runs(speech, overlap, valid, cls=None, min_len_s=MIN_RUN_S):
    """**同一局部说话人**的连续单人区间（秒）。

    ⚠️ 只判"是否重叠"是不够的：A 说完 B 立刻接话（中间没有静音）时，OSD 看到的是
    连续单人语音，区间会跨过 A/B 边界，窗里仍然混两个人。实测（R8007_M8010）：
    只用 overlap 判据时，"单人窗"仍有 **48%** 跨真值说话人边界。
    powerset 的 class 1/2/3 是"局部 speaker 1/2/3 单独说话"，**类别一变就是换人**。
    """
    single = valid & (speech > SPEECH_TH) & (overlap < OSD_TH)
    if cls is not None:
        single = single & np.isin(cls, (1, 2, 3))
    runs, start, cur = [], None, None
    for i, flag in enumerate(single):
        now = int(cls[i]) if cls is not None else 0
        if flag and (start is None or now != cur):
            if start is not None and (i - start) * FRAME_STEP_S >= min_len_s:
                runs.append((start * FRAME_STEP_S, i * FRAME_STEP_S))
            start, cur = i, now
        elif not flag and start is not None:
            if (i - start) * FRAME_STEP_S >= min_len_s:
                runs.append((start * FRAME_STEP_S, i * FRAME_STEP_S))
            start, cur = None, None
    if start is not None and (len(single) - start) * FRAME_STEP_S >= min_len_s:
        runs.append((start * FRAME_STEP_S, len(single) * FRAME_STEP_S))
    return runs


def windows_in(start, end):
    """区间内取 2.0s 窗；区间太短就整段当一个窗。"""
    wins, t = [], start
    while t + WIN_S <= end + 1e-9:
        wins.append((t, t + WIN_S))
        t += HOP_S
    if not wins and end - start >= MIN_CHUNK:
        wins.append((start, end))
    return wins


def embed_windows(model, audio, rate, wins):
    vecs, kept = [], []
    for a, b in wins:
        chunk = audio[int(a * rate):int(b * rate)]
        if len(chunk) < int(MIN_CHUNK * rate):
            continue
        try:
            e = model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if e is None:
            continue
        if hasattr(e, "detach"):
            e = e.detach().cpu().numpy()
        vecs.append(_norm(e))
        kept.append((a, b))
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32)), kept


def cluster(E, gate=GATE):
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


def turns_accuracy(windows, labels, turns):
    """窗级标签 -> 句级主导投票（未覆盖的句用时间最近的窗）-> 句级准确率。"""
    if not len(windows):
        return {"acc": 0.0, "k": 0, "n": 0, "cover": 0.0}
    gt, pred, durs, covered = [], [], [], 0
    for turn in turns:
        tally = defaultdict(float)
        for (a, b), lab in zip(windows, labels.tolist()):
            ov = min(b, turn["end"]) - max(a, turn["start"])
            if ov > 0:
                tally[lab] += ov
        if not tally:                      # 未被任何窗覆盖 -> 用最近的窗
            mid = (turn["start"] + turn["end"]) / 2.0
            near = min(range(len(windows)),
                       key=lambda i: abs((windows[i][0] + windows[i][1]) / 2.0 - mid))
            tally[labels[near]] = 1.0
        else:
            covered += 1
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
    return {"acc": ok / max(len(gt), 1), "k": len(set(pred)),
            "n": len(gt), "cover": covered / max(len(gt), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=["R8007_M8010"])
    ap.add_argument("--out", default="osd_rebuild_diar.json")
    args = ap.parse_args()

    sess = ort.InferenceSession(str(OSD_MODEL), providers=["CPUExecutionProvider"])
    models = load_models()
    rows = []
    for meeting in args.meetings:
        t0 = time.time()
        wav = resolve_wav(meeting)
        audio, rate = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        speech, overlap, valid, cls = osd_frames(sess, audio)
        runs = single_runs(speech, overlap, valid, cls)
        osd_wins = [w for r in runs for w in windows_in(*r)]
        # grid 臂也过 VAD（与生产 `_extract_windows` 的 MIN_VAD_FRACTION=0.5 同口径），
        # 否则它塌陷可能是因为混进了静音窗，而不是混合声纹 —— 那就不是公平对照。
        import claude_a2_v11 as v11
        vad_segs = v11._vad_segs(models["vad"], str(wav))
        grid_all = [(t, t + WIN_S)
                    for t in np.arange(0, max(0.0, len(audio) / rate - WIN_S), HOP_S)]
        grid_wins = [(a, b) for a, b in grid_all
                     if sum(max(0.0, min(b, e) - max(a, s)) for s, e in vad_segs)
                     >= 0.5 * (b - a)]

        row = {"meeting": meeting, "n_turns": len(turns),
               "single_runs": len(runs),
               "single_speech_s": round(sum(e - s for s, e in runs), 1),
               "osd_windows": len(osd_wins), "grid_windows": len(grid_wins)}
        print("\n=== %s ===" % meeting, flush=True)
        print("  单人连续区间 %d 个，合计 %.0f 秒；OSD 窗 %d 个，网格窗 %d 个"
              % (len(runs), row["single_speech_s"], len(osd_wins), len(grid_wins)), flush=True)

        for arm, wins in (("grid", grid_wins), ("osdwin", osd_wins)):
            E, kept = embed_windows(models["spk"], audio, rate, wins)
            if len(E) < 2:
                row[arm] = {"error": "窗不足", "n": len(E)}
                continue
            labels, k = cluster(E)
            res = turns_accuracy(kept, labels, turns)
            res["n_windows"] = len(E)
            row[arm] = res
            print("  %-7s 窗 %5d  聚类K=%d  句级 %5.1f%%  覆盖 %4.0f%%"
                  % (arm, len(E), k, res["acc"] * 100, res["cover"] * 100), flush=True)
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"osd_th": OSD_TH, "min_run_s": MIN_RUN_S,
                                   "win_s": WIN_S, "hop_s": HOP_S,
                                   "note": "grid=固定网格窗（现状做法）；osdwin=OSD 单人区间内取窗"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n对照：现状（2.0s 网格窗段表 + 句级投票）R8007_M8010 = 22.2%，R8008_M8013 = 38.1%")
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
