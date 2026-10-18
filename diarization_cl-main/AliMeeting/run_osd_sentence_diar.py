# -*- coding: utf-8 -*-
"""OSD 掩码 + 句级 CAM++：用免 gated 的 pyannote 3.0 ONNX 自动逼近 pure 臂上界。

**为什么是这个组合**（全部有实测支撑）：
  · 窗级（2.0s / 0.9s）都塌陷到 K=1；句级平均是唯一立得住的粒度
  · 句级 pure（**真值**剔重叠句）能到 80.2%，但生产没有真值
  · 声纹空间里判纯度全部失败（余弦 margin 死锁、熵召回仅 0.13）
  · `pyannote/segmentation-3.0` 的免 gated ONNX 在**声学层**判重叠，实测召回
    0.75–0.89（重叠率最高的 R8007_M8010 上 F1=0.877），CPU 约 13 秒/场

**三臂对照（唯一变量 = 用什么剔污染）**：

    full    不剔，整句提嵌入（复现已知的 K=1 塌陷）
    osd     用 OSD 剔句内重叠帧后提嵌入；只有"重叠率<0.30 且纯净>=1.2s"的句子
            当锚点进聚类，其余句子按余弦分配给锚点质心
    oracle  用**真值**剔重叠（诊断上界，非可达收益）

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_osd_sentence_diar.py --meetings R8007_M8010
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
WIN, STEP, FRAMES = 160000, 80000, 589
FRAME_STEP = WIN / FRAMES                     # 271.65 采样点/帧 ≈ 16.98ms
FRAME_STEP_S = FRAME_STEP / 16000.0
WIN_S, HOP_S = 2.0, 1.0
MAX_SPEAKERS = 10
MIN_CHUNK = 0.35
GATE = (2, 0.010)
OSD_TH = 0.30          # 重叠判定阈值（实测 th=0.2~0.3 召回/精度折中最好）
ANCHOR_OV = 0.30       # 锚点句允许的最大重叠率
ANCHOR_PURE_S = 1.2    # 锚点句要求的最短纯净语音
OVERLAP_TOL = 0.05

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def osd_probs(sess, audio):
    """整场 -> 每帧 7 类概率。"""
    total = len(audio)
    n = int(np.ceil(total / FRAME_STEP)) + 8
    acc = np.zeros((n, 7), np.float64)
    cnt = np.zeros(n, np.float64)
    name = sess.get_inputs()[0].name
    for start in range(0, max(1, total), STEP):
        chunk = audio[start:start + WIN]
        valid = len(chunk)
        if valid < WIN:
            chunk = np.pad(chunk, (0, WIN - valid))
        out = sess.run(None, {name: chunk[None, None, :].astype(np.float32)})[0][0]
        probs = np.exp(out) if out.max() <= 0 else out
        vf = min(FRAMES, int(round(valid / FRAME_STEP)))
        sf_ = int(round(start / FRAME_STEP))
        ef = min(n, sf_ + vf)
        if ef <= sf_:
            continue
        acc[sf_:ef] += probs[:ef - sf_]
        cnt[sf_:ef] += 1.0
        if start + WIN >= total:
            break
    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return acc[:, 4:7].sum(axis=1), mask      # (overlap_prob, valid)


def pure_slice(audio, ov, start_ms, end_ms, th=OSD_TH):
    """句内剥离重叠帧，返回 (纯净音频, 句内重叠率, 纯净秒数)。

    帧号 = 采样点 / FRAME_STEP；重叠率按"活跃帧里重叠帧的占比"算。
    """
    s_f = max(0, int(start_ms * 16.0 / FRAME_STEP))
    e_f = min(len(ov), int(end_ms * 16.0 / FRAME_STEP))
    raw = audio[int(start_ms * 16):int(end_ms * 16)]
    if e_f <= s_f:
        return raw, 0.0, len(raw) / 16000.0
    seg_ov = ov[s_f:e_f]
    active = seg_ov >= 0.0                     # 该窗内有语音（OSD 已在语音内）
    if not active.any():
        return raw, 0.0, len(raw) / 16000.0
    dirty = active & (seg_ov >= th)
    ratio = float(dirty.sum()) / float(max(active.sum(), 1))
    keep = np.flatnonzero(active & ~dirty) + s_f
    if not len(keep):
        # 整句都被判重叠：退回取重叠概率最低的前 40% 帧（保底，不是干净样本）
        order = np.argsort(seg_ov)[: max(1, len(seg_ov) * 2 // 5)]
        keep = np.sort(order) + s_f
    pieces = [audio[int(f * FRAME_STEP):int((f + 1) * FRAME_STEP)] for f in keep]
    pure = np.concatenate(pieces) if pieces else raw
    return pure, ratio, len(pure) / 16000.0


def slide(start, end):
    wins, t = [], start
    while t + WIN_S <= end + 1e-9:
        wins.append((t, t + WIN_S))
        t += HOP_S
    if not wins and end - start >= MIN_CHUNK:
        wins.append((start, min(end, start + WIN_S)))
    return wins


def embed(model, audio, rate, start, end):
    """句嵌入 = 句内各 2.0s 窗嵌入的 L2 平均。"""
    vecs = []
    for a, b in slide(start, end):
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
    return _norm(np.mean(vecs, axis=0)) if vecs else None


def truth_overlap_ratio(turn, segments, tol=OVERLAP_TOL):
    """真值口径：该句时长里"有别人同时在说"的比例（给 oracle 臂用）。"""
    span = turn["end"] - turn["start"]
    if span <= 0:
        return 1.0
    others = [s for s in segments if s["speaker"] != turn["speaker"]]
    cov = 0.0
    step = 0.02
    t = turn["start"]
    while t < turn["end"]:
        if any(o["start"] <= t < o["end"] for o in others):
            cov += step
        t += step
    return min(1.0, cov / span)


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


def score(embs, gts, durs, labels):
    pair = defaultdict(float)
    for g, p, d in zip(gts, labels.tolist(), durs):
        pair[(g, p)] += d
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gts, labels.tolist()) if mapping.get(g) == p)
    return ok / max(len(gts), 1), len(set(labels.tolist()))


def run_meeting(sess, models, meeting):
    audio, rate = sf.read(str(resolve_wav(meeting)), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
    ov, _ = osd_probs(sess, audio)

    out = {"meeting": meeting, "n_turns": len(turns)}
    tr = [truth_overlap_ratio(t, segments) for t in turns]
    print("  [诊断] 真值句重叠率: <0.30 的 %d/%d (%.1f%%)  中位 %.2f  均值 %.2f"
          % (sum(1 for x in tr if x < ANCHOR_OV), len(tr),
             100.0 * sum(1 for x in tr if x < ANCHOR_OV) / max(len(tr), 1),
             float(np.median(tr)), float(np.mean(tr))), flush=True)
    data = {}
    for arm in ("full", "osd", "oracle"):
        embs, gts, durs, anchors = [], [], [], []
        for turn in turns:
            if arm == "osd":
                src, ratio, pure_s = pure_slice(audio, ov, turn["start"], turn["end"])
                if len(src) / rate < 1.0:      # 纯片段太短 -> 退回整句，保证能提出嵌入
                    src = audio[int(turn["start"] * rate):int(turn["end"] * rate)]
                e = embed(models["spk"], src, rate, 0, len(src) / rate)
            else:
                e = embed(models["spk"], audio, rate, turn["start"], turn["end"])
                if arm == "full":
                    ratio, pure_s = 0.0, turn["end"] - turn["start"]
                else:
                    ratio = truth_overlap_ratio(turn, segments)
                    pure_s = turn["end"] - turn["start"]
            if e is None:
                continue
            embs.append(e)
            gts.append(turn["speaker"])
            durs.append(turn["end"] - turn["start"])
            anchors.append((ratio < ANCHOR_OV) and (pure_s >= ANCHOR_PURE_S))
        data[arm] = (embs, gts, durs, anchors)
        print("    [%s] 可用句 %d  锚点 %d" % (arm, len(embs), int(np.sum(anchors))),
              flush=True)

    for arm in ("full", "osd", "oracle"):
        embs, gts, durs, anchors = data[arm]
        if len(embs) < 2:
            out[arm] = {"error": "样本不足", "n": len(embs)}
            continue
        E = np.stack(embs)
        A = np.asarray(anchors)
        if arm == "full" or A.sum() < 2:
            labels, k = cluster(E)
            acc, kk = score(E, gts, durs, labels)
            out[arm] = {"acc": acc, "k": k, "n_anchor": int(A.sum()),
                        "coverage": 1.0, "note": "全量直接聚类" if arm == "full" else "锚点不足"}
            continue
        lab_a, k = cluster(E[A])
        if k < 2:
            out[arm] = {"acc": 0.0, "k": k, "n_anchor": int(A.sum()),
                        "coverage": 1.0, "note": "锚点聚类塌陷"}
            continue
        cents = np.stack([_norm(E[A][lab_a == c].mean(axis=0)) for c in range(k)])
        labels = (E @ cents.T).argmax(axis=1)      # 全量句子按中心分配（覆盖 100%）
        acc, _ = score(E, gts, durs, labels)
        acc_a, _ = score(E[A], [g for g, a in zip(gts, A) if a],
                         [d for d, a in zip(durs, A) if a], lab_a)
        out[arm] = {"acc": acc, "anchor_acc": acc_a, "k": k, "n_anchor": int(A.sum()),
                    "anchor_ratio": float(A.mean()), "coverage": 1.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=["R8007_M8010"])
    ap.add_argument("--out", default="osd_sentence_diar.json")
    args = ap.parse_args()
    sess = ort.InferenceSession(str(OSD_MODEL), providers=["CPUExecutionProvider"])
    models = load_models()
    rows = []
    for meeting in args.meetings:
        t0 = time.time()
        row = run_meeting(sess, models, meeting)
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print("\n=== %s（%d 句）===" % (meeting, row["n_turns"]), flush=True)
        for arm, label in (("full", "full  整句不剔（复现已知塌陷）"),
                           ("osd", "osd   声学 OSD 剔重叠 + 锚点聚类"),
                           ("oracle", "oracle 真值剔重叠（上界）")):
            r = row.get(arm, {})
            print("  %-34s 全量 %6.1f%%  锚点集 %s  K=%s  锚点数 %s  %s"
                  % (label, r.get("acc", 0) * 100,
                     ("%5.1f%%" % (r["anchor_acc"] * 100)) if "anchor_acc" in r else "   -  ",
                     r.get("k", "-"), r.get("n_anchor", "-"), r.get("note", "")),
                  flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"osd_model": str(OSD_MODEL), "osd_th": OSD_TH,
                                   "anchor_ov": ANCHOR_OV, "anchor_pure_s": ANCHOR_PURE_S,
                                   "note": "oracle 用真值剔重叠，是诊断上界"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n对照基准：现状（2.0s 窗段表）R8007_M8010 = 22.2%，R8008_M8013 = 38.1%")
    print("已知上界：句级 pure（真值剔重叠句）R8007_M8010 = 80.2%")
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
