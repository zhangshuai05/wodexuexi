# -*- coding: utf-8 -*-
"""3D-Speaker pipeline 移植 · 第一步：验证谱聚类能否把 K 立起来。

**为什么要移植**：我们自己的链路在 R8007_M8010 上窗级/句级聚类全部塌陷到 K=1，
而诊断显示**纯窗上 CAM++ 留一准确率 100%**（间隔 0.377）——声纹没问题，
是"取窗"和"聚类"这两步合起来出了问题。3D-Speaker（Apache 2.0）的
`infer_diarization.py` 是这套组合的成熟版本，本脚本照它的配方做第一步：

    VAD → 1.5s/0.75s chunk → CAM++ 嵌入 → 谱聚类（k-NN 图 + eigengap 自动定 K）

与我们的旧链路相比，只有三处不同（其余尽量对齐以便归因）：
    1. chunk 1.5s/0.75s（旧链路 2.0s/1.0s）
    2. 谱聚类 + k-NN 稀疏化（旧链路 AHC + 轮廓系数选 K）
    3. 用 eigengap 自动定 K（旧链路用 objective = silhouette - λ(k-2)）

**判据**：K 能否自动接近真值、句级准确率能否超过现状（R8007_M8010 = 22.2%，
R8008_M8013 = 38.1%）。**这一步不过，post_process（匈牙利匹配多标签）就没有意义。**

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_3dspeaker_pipeline.py --meetings R8007_M8010 R8008_M8013
"""

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "streaming_asr", HERE.parent / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, RESULTS, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)
from run_der import load_models  # noqa: E402

CHUNK_S, STEP_S = 1.5, 0.75        # 3D-Speaker 的配方
MIN_CHUNK = 0.40
MAX_K = 15                          # 3D-Speaker: max_num_spks=15
MIN_K = 1
KNN_K = 8                           # k-NN 稀疏化的邻居数
SPECTRAL_SEED = 42

# 密集窗身份复核（2026-10-18）：class-chunk 的 1.5s 长窗塞不进 <1.5s 的短插话，
# 短插话说话人可能整段缺席，eigengap 只能选更小的 K（smoke_clip_180s 实测 3 人，
# 长窗只看到 2 人；该片段与 R8008 同剧本但音频不是同一段）。用覆盖全部 VAD
# 语音的 0.6s 密集窗复核 K。
DENSE_MAX_WINDOWS = 1500            # 超过则等间隔抽样，控制 eigh/linkage 成本
DENSE_MAX_SPEAKERS = 10
# 判据（2026-10-18 smoke 实测标定）：
#   K=3 若只是把一个人切成两半，两簇质心余弦 >=0.72（class-chunk 实测 0.725）；
#   真三人时三簇质心余弦 <=0.56（0.6s 密集窗实测 0.374/0.525/0.555）。
#   所以用"质心可区分 + 轮廓质量接近最优时取更大 K"替代纯 silhouette/ eigengap。
DENSE_CENTROID_MAX_COS = 0.65
DENSE_SIL_TOLERANCE = 0.03
DENSE_MIN_CLUSTER_SHARE = 0.05

# 2026-10-18 封板：用户拍板默认关闭 K=2 -> K=3 密集窗救援。
# 理由：0.6s 短窗方差较大，真实 2 人对有被过切成 3 人的风险（R8001 上曾把
# K=4 误升为 5，细化 47.85→41.9）；餐饮段（hash b2d1…）长窗本身就是
# 3 人 / 11 切换，不依赖救援。物业段（hash b8f0…）回到自然 2 人。
DENSE_RESCUE_ENABLED = False

_, read_rttm = der_tools()


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)


def chunk_windows(vad_segments):
    """VAD 段内滑 1.5s/0.75s 的 chunk（3D-Speaker 的 chunk()）。"""
    out = []
    for start, end in vad_segments:
        t = start
        while t + CHUNK_S < end + STEP_S:
            out.append((t, min(t + CHUNK_S, end)))
            t += STEP_S
    return [(a, b) for a, b in out if b - a >= MIN_CHUNK]


def embed_chunks(model, audio, rate, wins):
    vecs, kept = [], []
    failed = 0
    for a, b in wins:
        chunk = audio[int(a * rate):int(b * rate)]
        if len(chunk) < int(MIN_CHUNK * rate):
            continue
        try:
            e = model.generate(input=chunk)[0].get("spk_embedding")
        except Exception as exc:
            failed += 1
            if failed == 1:          # 首个失败喊一次，别逐块刷屏
                print("  [embed] 首个 chunk 取嵌入失败（%s: %s），继续试剩余"
                      % (type(exc).__name__, str(exc)[:80]), flush=True)
            continue
        if e is None:
            continue
        if hasattr(e, "detach"):
            e = e.detach().cpu().numpy()
        vecs.append(_norm(e))
        kept.append((a, b))
    if failed:
        print("  [embed] %d/%d 个窗取嵌入失败" % (failed, len(wins)), flush=True)
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32)), kept


def eigengap_k(eigenvalues, max_k=MAX_K):
    """最大特征间隙定 K（谱聚类标准做法）。"""
    vals = np.sort(eigenvalues)[: max_k + 1]
    gaps = np.diff(vals)
    if not len(gaps):
        return 1
    return int(np.argmax(gaps) + 1)


def spectral_cluster(E, n_clusters=None, knn_k=KNN_K):
    """归一化谱聚类 + k-NN 稀疏化（纯 scipy，不引 sklearn）。

    亲和矩阵用 k-NN 截断而非全局 p-percentile：全局阈值在簇大小不均时
    会把两个簇"搭桥"连通（这正是 K 塌陷的常见机制）。
    """
    from scipy.cluster.vq import kmeans2
    from scipy.linalg import eigh

    n = len(E)
    if n < 3:
        return np.zeros(n, dtype=int), 1, np.array([0.0])
    S = np.clip(E @ E.T, 0.0, None)
    np.fill_diagonal(S, 0.0)
    k = min(knn_k, n - 1)
    W = np.zeros_like(S)
    for i in range(n):
        idx = np.argpartition(S[i], -k)[-k:]
        W[i, idx] = S[i, idx]
    W = np.maximum(W, W.T)
    deg = W.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    L = np.eye(n) - (W * dinv[:, None]) * dinv[None, :]
    vals, vecs = eigh(L)
    k_auto = max(MIN_K, min(eigengap_k(vals), MAX_K, n - 1))
    k_use = n_clusters if n_clusters else k_auto
    U = vecs[:, :k_use]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
    try:
        _, labels = kmeans2(U, k_use, minit="++", seed=SPECTRAL_SEED)
    except Exception as exc:
        # round-robin 标签是**语义随机**的：下游所有准确率都会失去意义，必须让人看见
        print("  [cluster] kmeans2 失败（%s: %s），退回 round-robin 标签，"
              "本次结果不可信" % (type(exc).__name__, str(exc)[:80]), flush=True)
        labels = np.arange(n) % k_use
    return labels.astype(int), k_use, vals[: MAX_K + 1]


def dense_identity_estimate(E, max_windows=DENSE_MAX_WINDOWS,
                            max_speakers=DENSE_MAX_SPEAKERS):
    """0.6s 密集窗上的独立 K 估计；返回 anchors/labels/k，或 None。

    为什么需要独立估 K：AliMeeting 的 class-chunk 只从 OSD 单说话人区间切
    1.5s 长窗，短于 1.5s 的插话会被并入前一个区间或整段丢掉。若短插话来自
    第三个人，长窗嵌入里根本没有这个身份，eigengap 只能选 2（smoke_clip_180s
    实测：3 人，长窗 K=2；强制 K=3 会把一人切成两半，轮廓系数从 0.30 掉到
    0.17）。密集窗覆盖全部 VAD 语音，能补回短插话身份。

    调用方只在长窗 K=2、密集窗 K=3 时采用（R8001_M8004 真值 4 人时，
    密集窗 K=5 会让细化 47.85→41.9，故 K>=3 不做"多一人"救援）。
    抽样上限默认 1500 窗，对 2000s 会议约 1.35s 一个样本，仍能覆盖 >=1s 的插话。
    """
    import claude_a2_v11 as v11
    n = len(E)
    if n < 8:
        return None
    stride = max(1, int(np.ceil(n / float(max_windows))))
    sub = E[::stride]
    if len(sub) < 8:
        return None

    candidates = []
    for k in range(2, min(max_speakers, len(sub)) + 1):
        labels, _, _ = spectral_cluster(sub, k)
        present = sorted({int(x) for x in labels.tolist()})
        sizes = [int((labels == x).sum()) for x in present]
        if len(sizes) < 2 or min(sizes) < max(
                4, int(DENSE_MIN_CLUSTER_SHARE * len(sub))):
            continue
        sil = float(v11._silhouette(sub, labels))
        if sil <= 0.0:
            continue
        cents = v11._centroids(sub, labels)
        cos = cents @ cents.T
        off = cos[~np.eye(len(cos), dtype=bool)]
        max_cos = float(off.max()) if off.size else 1.0
        # K>2 时新簇必须与所有旧簇都可区分；否则只是把一个人切成两半。
        if len(sizes) > 2 and max_cos > DENSE_CENTROID_MAX_COS:
            continue
        candidates.append({"k": len(sizes), "labels": labels,
                           "silhouette": sil, "sizes": sizes,
                           "max_centroid_cos": max_cos})
    if not candidates:
        return None
    best_sil = max(c["silhouette"] for c in candidates)
    viable = [c for c in candidates
              if c["silhouette"] >= best_sil - DENSE_SIL_TOLERANCE]
    best = max(viable, key=lambda c: c["k"])
    anchors = v11._centroids(sub, best["labels"])
    anchor_labels = sorted({int(x) for x in best["labels"].tolist()})
    return {"anchors": anchors, "anchor_labels": anchor_labels,
            "k": int(best["k"]), "silhouette": float(best["silhouette"]),
            "n_sampled": int(len(sub)), "stride": int(stride),
            "sizes": best["sizes"],
            "max_centroid_cos": float(best["max_centroid_cos"])}


def osd_probs(sess, audio):
    """pyannote/segmentation-3.0 ONNX -> 帧级 7 类概率（10s 窗 / 5s 步长，重叠窗取平均）。"""
    W, STEP, FR = 160000, 80000, 589
    fs = (W / FR) / 16000.0
    total = len(audio)
    n = int(np.ceil(total / (W / FR))) + 8
    acc = np.zeros((n, 7), np.float64)
    cnt = np.zeros(n, np.float64)
    name = sess.get_inputs()[0].name
    for start in range(0, max(1, total), STEP):
        chunk = audio[start:start + W]
        valid = len(chunk)
        if valid < W:
            chunk = np.pad(chunk, (0, W - valid))
        out = sess.run(None, {name: chunk[None, None, :].astype(np.float32)})[0][0]
        p = np.exp(out) if out.max() <= 0 else out
        vf = min(FR, int(round(valid / (W / FR))))
        i0 = int(round(start / (W / FR)))
        i1 = min(n, i0 + vf)
        if i1 <= i0:
            continue
        acc[i0:i1] += p[: i1 - i0]
        cnt[i0:i1] += 1.0
        if start + W >= total:
            break
    mask = cnt > 0
    acc[mask] /= cnt[mask, None]
    return acc, mask, fs


def post_process(chunk_labels, chunk_spans, probs, k, frame_step_s):
    """3D-Speaker 的 post_process：逐 OSD 窗匈牙利匹配，把局部 powerset 类别对到全局簇。

    为什么必须逐窗：powerset 的 class 1/2/3 是**窗内局部**说话人，跨 10s 窗会重新编号，
    所以匹配只能在窗内做（这正是它能绕开"跨窗一致性"死结的原因）。
    输出帧级多标签：class 4/5/6 是双人重叠，匹配后会激活两个全局说话人。
    """
    from scipy.optimize import linear_sum_assignment

    n = len(probs)
    cluster_frames = np.zeros((n, k), dtype=np.float32)
    for (a, b), lab in zip(chunk_spans, chunk_labels.tolist()):
        i0, i1 = int(a / frame_step_s), min(n, int(b / frame_step_s))
        if i1 > i0:
            cluster_frames[i0:i1, lab] = 1.0

    win = max(1, int(10.0 / frame_step_s))
    act = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, win):
        e = min(n, s + win)
        data, ccf = probs[s:e, :], cluster_frames[s:e, :]
        if ccf.sum() == 0:
            continue
        cost = np.full((7, k), -1.0)
        for j in range(7):
            dj = data[:, j] > 0.5
            if dj.sum():
                cost[j] = [float((dj & (ccf[:, c] > 0.5)).sum()) for c in range(k)]
        rows, cols = linear_sum_assignment(-cost)
        for r, c in zip(rows, cols):
            if cost[r, c] > 0:
                act[s:e, c] = np.maximum(act[s:e, c], data[:, r])

    # 每帧取"期望说话人数"个（powerset 期望值：单人1 + 双人2）
    exp_n = np.rint(probs[:, 1:4].sum(axis=1) + 2 * probs[:, 4:7].sum(axis=1)).astype(int)
    order = np.argsort(-act, axis=1)
    binary = np.zeros_like(act)
    for t in range(n):
        for i in range(min(k, max(1, int(exp_n[t])))):
            if act[t, order[t, i]] > 0:
                binary[t, order[t, i]] = 1.0
        if binary[t].sum() == 0 and cluster_frames[t].sum() > 0:
            binary[t] = cluster_frames[t]
    return binary


def binary_to_turns(binary, frame_step_s):
    """帧级多标签 -> 段表（每段带可能的次说话人）。"""
    out = []
    k = binary.shape[1]
    for spk in range(k):
        col = binary[:, spk] > 0.5
        start = None
        for i, flag in enumerate(col):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                out.append({"start": start * frame_step_s, "end": i * frame_step_s,
                            "speaker": "C%d" % spk})
                start = None
        if start is not None:
            out.append({"start": start * frame_step_s, "end": len(col) * frame_step_s,
                        "speaker": "C%d" % spk})
    out.sort(key=lambda x: x["start"])
    return out


def turns_multilabel(binary, turns, frame_step_s, thresh=0.25):
    """句级多标签统计：主标签 + 是否检出重叠。"""
    k = binary.shape[1]
    pred, gt, ok2, has2, cov = [], [], 0, 0, 0
    for turn in turns:
        i0 = max(0, int(turn["start"] / frame_step_s))
        i1 = min(len(binary), int(turn["end"] / frame_step_s))
        if i1 <= i0:
            pred.append(-1); gt.append(turn["speaker"]); continue
        seg = binary[i0:i1]
        share = seg.mean(axis=0)
        if share.max() <= 0:
            # 全零 = 本句没有任何激活簇。旧代码仍把 argsort 的 0 当预测，
            # 映射到簇 0 的说话人会白送正确句（acc 虚高、coverage 不涨）。
            pred.append(-1)              # -1 = 未预测，与 chunks_to_turns 同哨兵
            gt.append(turn["speaker"])
            continue
        order = np.argsort(-share)
        pred.append(int(order[0]))
        gt.append(turn["speaker"])
        cov += 1
        if len(order) > 1 and share[order[1]] > thresh:
            has2 += 1
    pair = defaultdict(float)
    for g, p in zip(gt, pred):
        if p != -1:                      # -1 不参与映射，否则会占掉别人的簇
            pair[(g, p)] += 1.0
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gt, pred) if mapping.get(g) == p)
    return {"acc": ok / max(len(gt), 1), "multilabel_rate": has2 / max(len(gt), 1),
            "coverage": cov / max(len(gt), 1)}


def class_intervals(probs, frame_step_s, min_run_s=0.30, merge_under_s=None):
    """powerset class 1/2/3 的连续区间（短于 merge_under_s 的并入前一个）。

    class_chunks（切 chunk）与 --refine-confine（限制细化窗）共用这一份区间，
    保证"chunk 从哪切"和"细化窗留哪些"的口径完全一致。

    区间先按 min_run_s（0.30s）全收，**不能**按 chunk_s 过滤：实测那样 R8007_M8010
    只剩 668 个 chunk、覆盖掉到 45%、准确率反降 4.7 点。短区间**并入前一个**，
    否则它塞不下一个完整窗、只能整段丢弃。
    """
    argmax = probs.argmax(axis=1)
    single = np.isin(argmax, (1, 2, 3))
    raw, start, cur = [], None, None
    for i, (flag, c) in enumerate(zip(single, argmax)):
        now = int(c)
        if flag and (start is None or now != cur):
            if start is not None and (i - start) * frame_step_s >= min_run_s:
                raw.append([start * frame_step_s, i * frame_step_s])
            start, cur = i, now
        elif not flag and start is not None:
            if (i - start) * frame_step_s >= min_run_s:
                raw.append([start * frame_step_s, i * frame_step_s])
            start, cur = None, None
    if start is not None and (len(single) - start) * frame_step_s >= min_run_s:
        raw.append([start * frame_step_s, len(single) * frame_step_s])

    merged = []
    for a, b in raw:
        if merged and merge_under_s and (b - a) < merge_under_s:
            merged[-1][1] = b                  # 短区间并入前一个，保住覆盖
        else:
            merged.append([a, b])
    return merged


def confine_refine(times, intervals, win_s):
    """困境六验证：只保留**完整落在某个 class 区间内**的细化窗。

    假设：细化（0.6s 短窗按最近质心重判）把 class 切分已经切对的地方重判散了 ——
    跨区间的窗里坐着两个人，嵌入是两人平均，最近质心判归属等于掷硬币。
    全包含（不是中心在内）才把这类污染窗干净地剔除。返回布尔掩码。
    """
    keep = np.zeros(len(times), dtype=bool)
    j = 0
    for i, t in enumerate(times):
        a, b = float(t), float(t) + win_s
        while j < len(intervals) and intervals[j][1] < b:
            j += 1
        if j < len(intervals) and intervals[j][0] <= a:
            keep[i] = True
    return keep


def class_chunks(probs, frame_step_s, chunk_s=None, step_s=None,
                 min_run_s=0.30):
    """在 powerset 的"同一局部说话人"区间内切 chunk，而不是从 VAD 段盲滑。

    **为什么**：实测 2.0s 网格窗有 **90%** 跨说话人边界（1851 个里只有 194 个能标上
    唯一说话人），因为 VAD 段只认"有没有语音"、不认"是谁在说"，A 说完 B 接话就落进
    同一段。powerset 的 class 1/2/3 是"局部 speaker 1/2/3 单独说话"，实测类别纯度
    77–83%，按它切分能把"窗内只属于一个人"的比例从 10.5% 提到 52%。
    跨窗编号不连续没关系 —— 聚类本来就要解决跨窗对齐。
    """
    # ⚠️ 不能写成 chunk_s=CHUNK_S：默认参数在 def 时就绑定，改模块常量改不动它
    # （Task 2 在 `_candidate_clusters(max_speakers=MAX_SPEAKERS)` 上踩过同一个坑）。
    chunk_s = CHUNK_S if chunk_s is None else chunk_s
    step_s = STEP_S if step_s is None else step_s
    merged = class_intervals(probs, frame_step_s, min_run_s=min_run_s,
                             merge_under_s=chunk_s)
    wins = []
    for a, b in merged:
        t = a
        while t + chunk_s <= b + 1e-9:
            wins.append((t, t + chunk_s))
            t += step_s
    return wins, len(merged)


def ahc_cluster(E, forced_k=None):
    """复用生产的 AHC + 选 K，用于在**同一份嵌入**上和谱聚类并排对照。

    为什么要并排跑：2 人场上谱聚类比现状低约 3 点，但那个对比跨了口径
    （现状是 v12 生产段表、含细化）。只有同嵌入换聚类，才能把
    "聚类算法的差" 与 "chunk 切分的差" 分开。
    """
    import claude_a2_v11 as v11
    if len(E) < 2:
        return np.zeros(len(E), dtype=int), 1
    old = (v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE)
    v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = 2, 0.010
    try:
        cands, _ = v11._candidate_clusters(E, max_speakers=min(MAX_K, len(E)))
        best = v11._select_candidate(cands, forced_k=forced_k)
    finally:
        v11.MIN_CLUSTER_WINDOWS, v11.MIN_CLUSTER_SHARE = old
    return best["labels"], int(best["k"])


def turns_to_speakers(windows, labels, turns, fallback=None):
    """每个 turn 的主导标签 —— **只出标签，不读真值**。

    fallback=(fb_windows, fb_labels)：没有窗覆盖的句子退回主窗（长窗）投票，
    而不是拿"时间最近"的窗瞎猜。R8008_M8013 上细化窗只覆盖 51.8% 的句子，
    其余 48.2% 落到旧的最近窗兜底——0.15s 步长的密集网格里"最近"的窗很可能
    在几秒之外、属于别人，这是细化 −10.2 的真凶（2026-10-15 定位）。
    两级窗的本意就是"长窗定身份、短窗找边界"，没短窗覆盖就该退回长窗的答案。

    返回 (pred, cover)。cover 只计本臂自己的窗，fallback 覆盖不算。
    chunks_to_turns 与 offline_diarizer.backfill_speakers 共用这一段。
    """
    pred, cover = [], 0
    fb_windows, fb_labels = fallback if fallback is not None else (None, None)
    for turn in turns:
        tally = defaultdict(float)
        for (a, b), lab in zip(windows, labels.tolist()):
            ov = min(b, turn["end"]) - max(a, turn["start"])
            if ov > 0:
                tally[lab] += ov
        own_cover = bool(tally)
        if not tally and fb_windows is not None:
            for (a, b), lab in zip(fb_windows, fb_labels.tolist()):
                ov = min(b, turn["end"]) - max(a, turn["start"])
                if ov > 0:
                    tally[lab] += ov
        if not tally:
            if not windows:
                pred.append(-1)      # 无窗可用：记未覆盖，别让 min(range(0)) 崩掉整场
                continue
            mid = (turn["start"] + turn["end"]) / 2.0
            near = min(range(len(windows)),
                       key=lambda i: abs((windows[i][0] + windows[i][1]) / 2.0 - mid))
            tally[labels[near]] = 1.0
        if own_cover:
            cover += 1
        pred.append(max(tally, key=lambda x: tally[x]))
    return pred, cover


def chunks_to_turns(windows, labels, turns, fallback=None):
    """chunk 标签 -> 句级主导投票 -> 全局映射 -> 句级准确率（**读真值**）。"""
    pred, cover = turns_to_speakers(windows, labels, turns, fallback=fallback)
    gt = [t["speaker"] for t in turns]
    durs = [t["end"] - t["start"] for t in turns]
    pair = defaultdict(float)
    for g, p, d in zip(gt, pred, durs):
        if p != -1:              # -1 = "整场无窗可用"哨兵，不是"预测成簇 0"
            pair[(g, p)] += d
    mapping, used = {}, set()
    for (g, p), _ in sorted(pair.items(), key=lambda kv: -kv[1]):
        if g in mapping or p in used:
            continue
        mapping[g] = p
        used.add(p)
    ok = sum(1 for g, p in zip(gt, pred) if mapping.get(g) == p)
    return {"acc": ok / max(len(gt), 1), "n_pred_speakers": len(set(pred)),
            "coverage": cover / max(len(gt), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*",
                    default=["R8007_M8010", "R8008_M8013", "R8009_M8020"])
    ap.add_argument("--out", default="three_d_speaker_step1.json")
    ap.add_argument("--arm", choices=["vad", "class"], default="class",
                    help="chunk 从哪来：vad=从 VAD 段盲滑（旧做法）；"
                         "class=powerset 同一局部说话人的区间内滑（新）")
    ap.add_argument("--post", action="store_true",
                    help="启用 OSD+匈牙利匹配后处理（实测 2/3 场有害，默认关；"
                         "只有需要重叠多标签时才开）")
    ap.add_argument("--chunk-s", type=float, default=None,
                    help="chunk 窗长（默认 1.5，即 3D-Speaker 配方；现状生产用 2.0）")
    ap.add_argument("--step-s", type=float, default=None, help="chunk 步长（默认 0.75）")
    ap.add_argument("--normalize-peak", type=float, default=0.0,
                    help=">0 时把弱音峰值提拉到该值再跑 VAD（只提拉不衰减）。"
                         "实测 VAD 对绝对电平敏感：R8008_M8013 peak=0.26 时"
                         "语音覆盖仅 24%，归一化到 0.85 后升到 51%（2026-10-17）")
    ap.add_argument("--refine-confine", action="store_true",
                    help="困境六验证：细化窗只保留完整落在 class 区间内的，"
                         "排除跨边界（窗内两人）的污染窗")
    args = ap.parse_args()
    global CHUNK_S, STEP_S
    if args.chunk_s:
        CHUNK_S = args.chunk_s
    if args.step_s:
        STEP_S = args.step_s
    print("chunk = %.2fs / hop = %.2fs" % (CHUNK_S, STEP_S), flush=True)

    import onnxruntime as ort
    import claude_a2_v11 as v11
    models = load_models()
    osd_path = Path(r"D:\yinp\pyannote_seg3\model.onnx")
    osd_sess = (ort.InferenceSession(str(osd_path), providers=["CPUExecutionProvider"])
                if osd_path.exists() else None)
    rows = []
    print("%-14s %5s %6s | %7s %7s %6s | %9s %9s"
          % ("meeting", "真值K", "chunk", "自动K", "真值K", "覆盖",
             "自动K准确", "真值K准确"), flush=True)
    for meeting in args.meetings:
        t0 = time.time()
        wav = resolve_wav(meeting)
        audio, rate = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        vad_wav = str(wav)
        if args.normalize_peak:
            # 与生产接口 offline_diarizer._normalize_audio 同口径：只提拉弱音，
            # 不衰减已经够响的录音（peak >= target 时逐位不变）。
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-9 and peak < args.normalize_peak:
                audio = audio * (args.normalize_peak / peak)
                vad_wav = str(Path(tempfile.gettempdir()) / f"norm_{meeting}.wav")
                sf.write(vad_wav, audio, rate, subtype="PCM_16")
        turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        vad = v11._vad_segs(models["vad"], vad_wav)
        # OSD 只跑一次：既用来切 chunk 的类别区间，也喂给 post_process
        probs = None
        if osd_sess is not None:
            probs_all, mask_all, fs = osd_probs(osd_sess, audio)
            probs = probs_all[mask_all]
        wins_vad = chunk_windows(vad)
        # 缺 OSD 时 class 臂会静默退化成 VAD 臂，却仍打印 [class]、JSON 里仍写
        # arm="class" —— 归因实验由此得出"class 无增益"的伪结论。宁可当场退出。
        if (args.arm == "class" or args.post) and osd_sess is None:
            raise SystemExit("--arm class / --post 需要 OSD 模型：%s" % osd_path)
        wins_cls = class_chunks(probs, fs)[0] if probs is not None else wins_vad
        arm = args.arm
        wins = wins_cls if arm == "class" else wins_vad
        E, kept = embed_chunks(models["spk"], audio, rate, wins)
        true_k = len({t["speaker"] for t in turns})

        labels_auto, k_auto, eigs = spectral_cluster(E, None)
        labels_true, _, _ = spectral_cluster(E, min(true_k, len(E) - 1))
        r_auto = chunks_to_turns(kept, labels_auto, turns)
        r_true = chunks_to_turns(kept, labels_true, turns)
        # 同一份嵌入换聚类，把"聚类算法的差"与"chunk 切分的差"分开
        la_ahc, ka_ahc = ahc_cluster(E)
        lt_ahc, _ = ahc_cluster(E, forced_k=min(true_k, len(E) - 1))
        r_ahc = chunks_to_turns(kept, la_ahc, turns)
        r_ahc_true = chunks_to_turns(kept, lt_ahc, turns)
        # 细化层：复用 v12 的"长窗定身份、0.6s 密集窗找边界"。
        # 实测这一层单独值 +7.6 pt（baseline 45.6% -> v12 53.2%），所以必须接上。
        import claude_a2_v12 as v12mod
        r_ref = r_ref_c = None
        confine_keep = None
        dense = None
        if k_auto >= 2:
            rE, rtimes = v12mod._refine_windows(audio, rate, vad, models["spk"])
            anchors = v11._centroids(E, labels_auto)
            alabs = sorted(set(int(x) for x in labels_auto.tolist()))
            fb_labels = labels_auto
            # 长窗 K 可能漏掉短插话说话人（class-chunk 塞不下 <1.5s 区间）。
            # 救援默认关闭（DENSE_RESCUE_ENABLED=False，2026-10-18 拍板）；
            # 打开时也只在长窗 K=2、密集窗给出 3 时接管（R8001 K=4→5 会 -5.9）。
            dense = (dense_identity_estimate(rE)
                     if DENSE_RESCUE_ENABLED and len(rE) else None)
            if (DENSE_RESCUE_ENABLED and dense is not None and k_auto == 2
                    and k_auto < dense["k"] <= k_auto + 1):
                anchors = dense["anchors"]
                alabs = dense["anchor_labels"]
                fb_labels = np.asarray([alabs[i] for i in
                                        np.argmax(E @ anchors.T, axis=1)])
                print("  [dense] %s 长窗 K=2 -> 密集窗 K=%d（sil=%.3f, %d/%d 窗）"
                      % (meeting, dense["k"], dense["silhouette"],
                         dense["n_sampled"], len(rE)), flush=True)
            rlabels = v12mod._refine_labels(rE, anchors, alabs, rtimes)
            if rlabels is not None and len(set(rlabels.tolist())) >= 2:
                rwins = [(float(t), float(t) + v12mod.REFINE_WIN_S) for t in rtimes]
                # ★ fallback=主窗：细化窗只盖住一半句子（R8008_M8013 实测 51.8%），
                # 落在外面的句子退回长窗投票，而不是拿几秒外别人的短窗瞎猜。
                r_ref = chunks_to_turns(rwins, rlabels, turns,
                                        fallback=(kept, fb_labels))
            # confine 臂（困境六）：细化窗只留完整落在 class 区间内的。
            # 结论（2026-10-15）：只挪 0.3 点，"细化把 class 切对的地方重判散了"被**证伪**；
            # 它的轻微变差其实来自"少了 564 个窗 ⇒ 更多句子落兜底"，与覆盖率假说同向。
            # 保留此臂作为覆盖率假说的旁证，别再拿它当机制冲突的证据。
            if args.refine_confine and probs is not None:
                intervals = class_intervals(probs, fs, merge_under_s=CHUNK_S)
                confine_keep = confine_refine(rtimes, intervals, v12mod.REFINE_WIN_S)
                if confine_keep.sum() >= v12mod.REFINE_MIN_WINDOWS:
                    rc_labels = v12mod._refine_labels(
                        rE[confine_keep], anchors, alabs, rtimes[confine_keep])
                    if (rc_labels is not None
                            and len(set(rc_labels.tolist())) >= 2):
                        rcwins = [(float(t), float(t) + v12mod.REFINE_WIN_S)
                                  for t in rtimes[confine_keep]]
                        r_ref_c = chunks_to_turns(rcwins, rc_labels, turns,
                                                  fallback=(kept, fb_labels))
        pa = pt = None
        if args.post and probs is not None:        # 复用上面那一次 OSD，不重复跑
            k_t = min(true_k, len(E) - 1)
            pa = turns_multilabel(
                post_process(labels_auto, kept, probs, k_auto, fs), turns, fs)
            pt = turns_multilabel(
                post_process(labels_true, kept, probs, k_t, fs), turns, fs)
        row = {"meeting": meeting, "true_k": true_k, "n_chunks": len(E),
               "k_auto": k_auto,
               "dense_k": (dense["k"] if dense is not None else None),
               "eigvals": [round(float(v), 4) for v in eigs[:8]],
               "auto": r_auto, "true_k_forced": r_true,
               "ahc": r_ahc, "ahc_true": r_ahc_true, "k_ahc": ka_ahc,
               "refined": r_ref,
               "refined_confined": r_ref_c,
               "confine_kept": ([int(confine_keep.sum()), int(len(confine_keep))]
                                if confine_keep is not None else None),
               "post_auto": pa, "post_true": pt,
               "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        print("[%s] %-14s %5d %6d | K 谱%2d/真%2d AHC%2d | 覆盖%3.0f%% | "
              "谱聚类 %5.1f%% | AHC %5.1f%% | **+细化 %s** | confine %s(%s) | +OSD %5.1f%%"
              % (arm, meeting, true_k, len(E), k_auto, true_k, ka_ahc,
                 r_auto["coverage"] * 100, r_auto["acc"] * 100, r_ahc["acc"] * 100,
                 ("%5.1f%%" % (r_ref["acc"] * 100)) if r_ref else "  -  ",
                 ("%5.1f%%" % (r_ref_c["acc"] * 100)) if r_ref_c else "  -  ",
                 ("%d/%d" % tuple(row["confine_kept"])) if row["confine_kept"] else "-",
                 (pt or {}).get("acc", 0) * 100),
              flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"chunk_s": CHUNK_S, "step_s": STEP_S,
                                   "knn_k": KNN_K, "max_k": MAX_K,
                                   "arm": args.arm, "post": args.post,
                                   "normalize_peak": args.normalize_peak,
                                   "refine_confine": args.refine_confine,
                                   "note": "谱聚类 k-NN 图 + eigengap 定 K；纯 scipy"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n对照：现状（2.0s 网格窗 + AHC）R8007_M8010 = 22.2%，R8008_M8013 = 38.1%，"
          "R8009_M8020 = 84.2%")
    if rows:
        import statistics as st
        BASE = {"R8001_M8004": 45.0, "R8003_M8001": 42.3, "R8007_M8010": 22.2,
                "R8007_M8011": 34.8, "R8008_M8013": 38.1, "R8009_M8018": 77.1,
                "R8009_M8019": 85.7, "R8009_M8020": 84.2}
        quads = [(r["meeting"], r["auto"]["acc"] * 100,
                  (r["refined"]["acc"] * 100
                   if r.get("refined") is not None else None),
                  ((r.get("refined_confined") or {}).get("acc") or 0) * 100
                  if r.get("refined_confined") else None,
                  BASE[r["meeting"]])
                 for r in rows if r["meeting"] in BASE]
        if quads:
            has_c = any(x[3] is not None for x in quads)
            ref_vals = [x[2] for x in quads if x[2] is not None]
            print("\n=== 汇总（%d 场，臂=%s）===" % (len(rows), args.arm))
            cols = ("meeting", "谱聚类", "+细化") + (("confine",) if has_c else ()) + ("现状", "细化增益")
            print("  %-14s" % cols[0] + "".join(" %8s" % c for c in cols[1:]))
            for m, sp, rf, rc, b in quads:
                rf_txt = ("%7.1f%%" % rf) if rf is not None else ("%8s" % "-")
                gain_txt = ("%+9.1f" % (rf - sp)) if rf is not None else ("%9s" % "-")
                line = "  %-14s %7.1f%% %s" % (m, sp, rf_txt)
                if has_c:
                    line += " %7.1f%%" % (rc if rc is not None else float("nan"))
                line += " %7.1f%% %s" % (b, gain_txt)
                print(line)
            avg_ref = ("%7.1f%%" % st.mean(ref_vals)) if ref_vals else ("%8s" % "-")
            avg_gain = ("%+9.1f" % st.mean([x[2] - x[1] for x in quads
                                            if x[2] is not None])) if ref_vals else ("%9s" % "-")
            print("  %-14s %7.1f%% %s" % ("平均", st.mean([x[1] for x in quads]), avg_ref)
                  + (" %7.1f%%" % st.mean([x[3] for x in quads if x[3] is not None]) if has_c else "")
                  + " %7.1f%% %s" % (st.mean([x[4] for x in quads]), avg_gain))
            if ref_vals:
                print("  含细化 vs 现状：%+.1f pt（%d/%d 场有细化结果）"
                      % (st.mean([x[2] - x[4] for x in quads if x[2] is not None]),
                         len(ref_vals), len(quads)))
            else:
                print("  含细化 vs 现状：无细化结果，未计算")
            if has_c:
                print("  困境六判据（R8008_M8013）：confine 恢复 ≥45% ⇒ 机制冲突坐实；"
                      "仍 ~40% ⇒ 接线问题" )
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
