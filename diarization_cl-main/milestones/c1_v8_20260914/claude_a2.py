# -*- coding: utf-8 -*-
"""
Claude 项目代码 —— A2 说话人分离核心（嵌入 DSH 测试台）
================================================================
源自 Claude 工作区（C:/Users/EDY/Desktop/new）的 A2-v2/v3 算法。
为嵌入本前端而做独立模块。本版本只保留 Claude 版本 1（短答细化 + 密集窗救援）。

运行链路：VAD 门控 → 1.2s/0.6s 滑窗 CAM++ 嵌入 → 跳变分段 → AHC 聚类 → 段标签。
关键增强（c1）：
  1. auto 塌缩判据：段级切换很少 + 主导占比在 (0.7, 0.95) 之间 → 判定为"两人被揉成一团"。
  2. 密集窗救援（dense_rescue）：塌缩文件用 0.6s/0.15s 密集窗重提嵌入，
     按"距主导质心边距"重建远端质心、逐窗判决，把被 1.2s 大窗平均吞掉的远端说话人挖出来。
  3. 普通段级路径以两簇质心为参考, 用短窗细化换人边界; 单段孤立簇回退。
     默认 merge=False, 与离线评估保持一致。

对 web_server.py 的接口：`run_claude_lane(wav_path, models, lane_key)`  →  lane_key 为 "c1"。
- wav_path: 16k 单声道 wav 的绝对路径
- models：{"vad":..., "spk":..., "asr":...}，由 web_server.get_models 提供（已加载）
返回：{"sentence_info": [{start,end,spk,text}], "diag": {...}, "verdict": ...,
      "method": ..., "elapsed": ...}

标签契约：spk ∈ {0,1,...,n-1}，对应前端 speaker_N。
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

# ---- 加载控制路径（与 web_server.py 的 VAD_LOCAL 一致，避免重复加载）----
VAD_LOCAL = os.path.join(HERE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                         "snapshots", "master")
# SPK 用本地快照（同请求/modelscope id，能省则省）
SPK_LOCAL = os.path.join(r"C:\Users\EDY\.cache\modelscope\models",
                         "iic--speech_campplus_sv_zh-cn_16k-common", "snapshots", "master")

WIN_S, HOP_S, JUMP_OFF = 1.2, 0.6, 0.20
MIN_WIN_OVERLAP = 0.5
CLUST_N = (2, 2)
GAP_UNRELIABLE = 0.06
# 救援门的切换数上限：救援输出切换数超过 window 回退的该倍数即判为过切，弃用救援。
RESCUE_SWITCH_MAX = 1.5

# ---- Claude 版本：c1 为当前 v6，c2 保留冻结 v4 供对比 ----
VERSIONS = {
    "c1": {
        "label": "Claude 版本 1 · v7 救援门+碎段吸收",
        "method": "claude-a2-v7-c1",
        "cluster_level": "auto", "link_mode": "auto", "dom_max": 0.95,
        "desc": "稳定身份锚 + 低置信碎段吸收 + 救援过切门 + 短窗换人边界",
    },
    "c2": {
        "label": "Claude 版本 2 · v4 短答细化",
        "method": "claude-a2-v4-c2",
        "cluster_level": "auto", "link_mode": "auto", "dom_max": 0.95,
        "desc": "v4 段级身份参考 + 短窗换人边界 + 塌缩救援",
    },
    # ---- v8：三项致命缺陷的结构性修复（2026-09-14 评审）----
    # 只新增开关，不动 c1 的任何默认值：c1 路径字节不变，v7 冻结基线继续可比。
    # 人数三态 / 外生救援门 / ASR 契约，证据层见 claude_a2_v8.py。
    "v8": {
        "label": "Claude 版本 3 · v8 外生证据+人数三态",
        "method": "claude-a2-v8-c3",
        "cluster_level": "auto", "link_mode": "auto", "dom_max": 0.95,
        "desc": "VAD 外生边界证据 + 人数三态(1/2/uncertain) + ASR 幻觉契约",
        "v8": True,
    },
}
DEFAULT_LANES = ["c1", "c2"]


def cs(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"


def _stabilize_labels(labels, margins):
    """Remove an isolated weak flip while preserving a confident short reply."""
    stable = labels.copy()
    for i in range(1, len(labels) - 1):
        if (labels[i - 1] == labels[i + 1] != labels[i]
                and abs(margins[i]) < min(abs(margins[i - 1]),
                                          abs(margins[i + 1]))):
            stable[i] = labels[i - 1]
    return stable


def _vad_segs(vad, wav):
    res = vad.generate(input=wav, dynamic_silence=False)
    segs = []
    for r in (res if isinstance(res, list) else [res]):
        for it in r.get("value", []):
            segs.append((float(it[0]) / 1000.0, float(it[1]) / 1000.0))
    return segs


def _segment_bounds(times, start_i, end_i, win_s):
    """Place adjacent segment edges between window centers, preserving gaps."""
    if start_i == 0 or times[start_i] - times[start_i - 1] > win_s:
        start = times[start_i]
    else:
        start = (times[start_i - 1] + times[start_i] + win_s) / 2
    if end_i == len(times) or times[end_i] - times[end_i - 1] > win_s:
        end = times[end_i - 1] + win_s
    else:
        end = (times[end_i - 1] + times[end_i] + win_s) / 2
    return start, end


def _dense_rescue(x, sr, vsegs, spk_model, win_s=0.6, hop_s=0.15,
                  min_frac=0.3, n_iter=3, rescue_merge=0.35,
                  prune_sliver=0.0, centroids=None):
    """密集窗边距判决：段级塌缩文件里，用 0.6s/0.15s 密集窗重提嵌入，
    以"距主导质心最远"的窗重建远端质心，逐窗按 cosine 边距分类，切成微段。
    返回 [(start,end,spk)]（已合并），失败返回 None。"""
    win = win_s * sr
    embs, times = [], []
    for t in np.arange(0.0, len(x) / sr - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < min_frac * win_s:
            continue
        chunk = x[int(s0 * sr):int(e0 * sr)]
        if len(chunk) < 1600:
            continue
        try:
            e = spk_model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)
                        if hasattr(e, "detach") else np.asarray(e).reshape(-1))
            times.append(float(t))
    if len(embs) < 8:
        return None
    E = np.stack(embs)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def _cosv(cent):
        c = np.asarray(cent, dtype=np.float64)
        c = c / (np.linalg.norm(c) + 1e-9)
        return En @ c

    # 主导质心 ≈ 全窗均值（近单说话人时约等于主导说话人）
    c_main = En.mean(axis=0)
    cos_main = _cosv(c_main)
    n_far = max(3, int(0.30 * len(E)))
    c_far = En[np.argsort(cos_main)[:n_far]].mean(axis=0)
    if centroids is not None:
        c_main, c_far = centroids
    else:
        # 无身份锚时才自更新; 已有段级质心时保持身份参考稳定。
        for _ in range(n_iter):
            delta = _cosv(c_main) - _cosv(c_far)
            k = max(3, len(E) // 10)
            c_main_new = En[np.argsort(-delta)[:k]].mean(axis=0)
            c_far_new = En[np.argsort(delta)[:k]].mean(axis=0)
            if cs(c_main_new, c_far_new) > 0.98 or cs(c_main_new, c_main) < 0.7:
                break
            c_main, c_far = c_main_new, c_far_new
    ca, cb = _cosv(c_main), _cosv(c_far)
    margins = ca - cb
    labels = (margins >= 0).astype(int)
    if centroids is not None:
        labels = _stabilize_labels(labels, margins)
        labels, margins = _absorb_slivers(labels, margins, np.asarray(times), win_s)
        # Preserve the caller's speaker IDs and place boundaries between centers.
        labels = 1 - labels
    else:
        n1 = int((labels == 1).sum())
        if n1 < 8 or n1 / len(E) < 0.10 or cs(c_main, c_far) > 0.90:
            return None
    seg_idx = [0]
    for i in range(1, len(times)):
        if labels[i] != labels[i - 1] or times[i] - times[i - 1] > win_s:
            seg_idx.append(i)
    seg_idx.append(len(times))
    out = []
    for a, b in zip(seg_idx[:-1], seg_idx[1:]):
        start, end = _segment_bounds(times, a, b, win_s)
        out.append((start, end, int(labels[a])))
    if centroids is not None:
        return out
    if not out:
        return None
    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < rescue_merge:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    return merged


def _absorb_slivers(labels, margins, times, win_s, min_dur=0.75,
                    weak_margin=0.18, passes=3):
    """把"很短的、边距很弱的"标签 run 吸进邻居，消除低置信度假碎段。

    只对 weak_margin 以下的短 run 动手：真短答（如 0.38s 的"对。"）边距在
    0.19~0.41，远高于此，因此不会被吞。
    """
    n = len(labels)
    if n == 0:
        return labels, margins
    lab = list(labels)
    for _ in range(passes):
        # run 的时长按窗中心间距累积（末窗补 win_s）
        i = 0
        runs = []
        while i < n:
            j = i
            while j + 1 < n and lab[j + 1] == lab[i] and \
                    times[j + 1] - times[j] <= win_s:
                j += 1
            span = times[j] - times[i] + win_s
            runs.append((i, j, span, float(np.mean(np.abs(margins[i:j + 1])))))
            i = j + 1
        changed = False
        for k, (a, b, span, conf) in enumerate(runs):
            if span >= min_dur or conf >= weak_margin:
                continue
            left = runs[k - 1] if k > 0 else None
            right = runs[k + 1] if k + 1 < len(runs) else None
            if left is None and right is None:
                continue
            if left is None:
                tgt = right
            elif right is None:
                tgt = left
            else:
                tgt = left if left[3] >= right[3] else right
            for m in range(a, b + 1):
                lab[m] = lab[tgt[0]]
                margins[m] = np.sign(margins[tgt[0]] or 1.0) * abs(margins[m])
            changed = True
            break
        if not changed:
            break
    return np.asarray(lab), np.asarray(margins)


def _build_seg(E, times, win_s, jump_off=JUMP_OFF):
    adj = np.array([cs(E[i], E[i + 1]) for i in range(len(E) - 1)])
    th_jump = float(np.median(adj)) - jump_off
    seg_idx = [0]
    for i in range(len(E) - 1):
        if adj[i] < th_jump:
            seg_idx.append(i + 1)
    seg_idx.append(len(E))
    segs = []
    for a, b in zip(seg_idx[:-1], seg_idx[1:]):
        rep = E[a] if b - a < 2 else E[a:b].mean(axis=0)
        start, end = _segment_bounds(times, a, b, win_s)
        segs.append((start, end, rep))
    return np.stack([s[2] for s in segs]), [s[0] for s in segs], segs


def _gap2_of(Z):
    d = Z[:, 2]
    k = len(d) - 2
    return float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None


def _merge_runs(out, gap_merge):
    """相邻同标签且间隔小于 gap_merge 的段合并。v7/v8 共用，保持行为一致。"""
    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < gap_merge:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    return merged


def _rescue_gate(cand, seqw, times, win_s):
    """Return dense/window switch counts and whether rescue stays within budget."""
    n_sw_win = sum(seqw[i] != seqw[i - 1] or times[i] - times[i - 1] > win_s
                   for i in range(1, len(seqw)))
    n_sw_cand = sum(cand[i][2] != cand[i - 1][2]
                    for i in range(1, len(cand)))
    allowed = max(RESCUE_SWITCH_MAX * max(n_sw_win, 1), n_sw_win + 2)
    return n_sw_cand, n_sw_win, n_sw_cand <= allowed


def a2_segments(wav, spk_model, vad_model, win_s=WIN_S, hop_s=HOP_S,
                jump_off=JUMP_OFF, clust_n=CLUST_N, merge=False,
                link_mode="auto", cluster_level="auto", gap_merge=0.3,
                dom_max=0.95, v8=False):
    """说话人段 → [{start,end,spk}]（与 a2_v2.py 同逻辑，嵌入版）。

    v8=True 时在塌缩分支改用外生证据（VAD 静音结构）判救援取舍，并在段级
    结果上做人数三态判定；v8=False 时行为与 v7 完全一致（冻结基线）。
    """
    import soundfile as sf
    from scipy.cluster.hierarchy import linkage, fcluster
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad_model, wav)
    if v8:
        return _a2_segments_v8(x, sr, dur, vsegs, spk_model, win_s, hop_s,
                               jump_off, clust_n, link_mode, cluster_level,
                               gap_merge, dom_max)
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < MIN_WIN_OVERLAP * win_s:
            continue
        chunk = x[int(s0 * sr):int(e0 * sr)]
        if len(chunk) < 1600:
            continue
        try:
            r = spk_model.generate(input=chunk)
            e = r[0].get("spk_embedding")
        except Exception:
            continue
        if e is not None:
            e = e.detach().cpu().numpy().reshape(-1) if hasattr(e, "detach") else np.asarray(e).reshape(-1)
            embs.append(e)
            times.append(float(t))
    if len(embs) < 2:
        return []
    E = np.stack(embs)

    # 决定 cluster_level
    rescue = None
    if cluster_level == "auto":
        seg_reps, seg_wst, segs_tmp = _build_seg(E, times, win_s)
        # 单一代表段 = 整段语音被判成同一人，没有可分的聚类结构。
        # 此时 linkage 拿不到距离矩阵（会抛 ValueError），直接输出单人。
        if len(seg_reps) < 2:
            return [{"start": float(segs_tmp[0][0]), "end": float(segs_tmp[0][1]),
                     "spk": 0}]
        Zseg = linkage(seg_reps, method="average", metric="cosine")
        dseg = Zseg[:, 2]
        kseg = len(dseg) - 2
        seg_gap = float(dseg[kseg + 1] - dseg[kseg]) if 0 <= kseg + 1 < len(dseg) else None
        lab_seg = fcluster(Zseg, t=2, criterion="maxclust")
        lab2s = {int(l): i for i, l in enumerate(sorted(set(lab_seg)))}
        seqs = [lab2s[int(l)] for l in lab_seg]
        n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
        dom_ratio = max(np.bincount(seqs)) / len(seqs) if len(seqs) else 0
        # 塌缩判定：切换很少 + 主导占比在 (0.7, dom_max)。dom 顶到 0.95 以上就是
        # 近单说话人（音频质量差/远场），不强上窗口，交给 dense 救援尝试。
        collapsed = (n_switch <= 2 and 0.7 < dom_ratio < dom_max)
        if collapsed:
            cand = _dense_rescue(x, sr, vsegs, spk_model)
            # 救援门：救援的远端质心是"距主导质心最远的 30% 窗"，在近单说话人
            # 音频里那批窗就是噪声，逐窗边距会在零轴附近反复穿越，切出一堆假段。
            # 判据：救援输出的切换数若显著超过 window 回退，说明它在过切，弃用。
            # 实测（7 个塌缩文件，DER 口径）1.5x 最优：平均 DER 40.75%，
            # 优于全开救援 45.52% 与全关救援 44.56%，且保住 11745/11423 的收益。
            # 已知漏判：10836（救援 DER 30.0% 优于关救援 39.2%，但切换 62 vs 22
            # 会被判过切）—— 该文件救援以 32 个碎段换 9pp DER，按"去碎段"优先舍弃。
            # 基线必须与 cand 同层级：window 回退是在"窗"上聚类后再合成连续段，
            # 不能拿段级标签序列（长度=段数）当基线，两者元素数不同、不可比。
            if cand is not None and len(set(r[2] for r in cand)) >= 2:
                Zw = linkage(E, method="average", metric="cosine")
                labw = fcluster(Zw, t=2, criterion="maxclust")
                labw2 = {int(l): i for i, l in enumerate(sorted(set(labw)))}
                seqw = [labw2[int(l)] for l in labw]
                n_sw_cand, n_sw_win, rescue_ok = _rescue_gate(
                    cand, seqw, times, win_s)
                if rescue_ok:
                    rescue = cand
                    cluster_level = "dense_rescue"
                else:
                    cluster_level = "window"
            else:
                cluster_level = "window"
        else:
            cluster_level = "segment"

    if cluster_level == "window":
        reps, wstarts, segs = E, list(times), None
    elif cluster_level == "dense_rescue":
        reps, wstarts, segs = None, None, None
    else:
        reps, wstarts, segs = _build_seg(E, times, win_s)

    # 选链接
    if cluster_level != "dense_rescue":
        Za = linkage(reps, method="average", metric="cosine")
        Zc = linkage(reps, method="complete", metric="cosine")
        ga, gc = _gap2_of(Za), _gap2_of(Zc)
        if link_mode == "average":
            Z = Za
        elif link_mode == "complete":
            Z = Zc
        else:
            Z = Zc if ((ga and gc and gc > ga * 1.5) or (ga and gc and gc - ga > 0.05)) else Za
        d = Z[:, 2]

        def _gap(cand):
            k = len(d) - cand
            return float(d[k + 1] - d[k]) if (0 <= k and k + 1 < len(d)) else None

        n = clust_n[0]
        best_gap = -1.0
        for cand in range(clust_n[0], clust_n[1] + 1):
            k = len(d) - cand
            if k < 0 or k + 1 >= len(d):
                continue
            jump = d[k + 1] - d[k]
            if jump > best_gap:
                best_gap, n = jump, cand
        labels = fcluster(Z, t=n, criterion="maxclust")
        lab2 = {int(l): i for i, l in enumerate(sorted(set(labels)))}
        labseq = [lab2[int(l)] for l in labels]
        if (cluster_level == "segment" and len(set(labseq)) == 2
                and min(np.bincount(labseq)) >= 2):
            # Long windows anchor identity; short windows recover embedded replies.
            # A singleton segment cannot provide a stable speaker anchor.
            centroids = [reps[np.asarray(labseq) == k].mean(axis=0) for k in (0, 1)]
            refined = _dense_rescue(x, sr, vsegs, spk_model, centroids=centroids)
            if refined is not None:
                return [{"start": float(a), "end": float(b), "spk": int(k)}
                        for a, b, k in refined]

    if cluster_level == "dense_rescue":
        out = [(s0, s1, lb) for (s0, s1, lb) in rescue]
        merged = _merge_runs(out, gap_merge)
        return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
                for (s0, s1, lab) in merged]

    if cluster_level == "window":
        out = []
        cur = None
        for st, lb in zip(wstarts, labseq):
            if cur is not None and cur[2] == lb:
                cur = (cur[0], st + win_s, lb)  # 连续同标签
            else:
                if cur is not None:
                    out.append(cur)
                cur = (st, st + win_s, lb)
        if cur is not None:
            out.append(cur)
    else:
        out = [(s0, s1, lb) for (s0, s1, _), lb in zip(segs, labseq)]

    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < gap_merge:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    if not merge:
        merged = out
    return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
            for (s0, s1, lab) in merged]


# ---------------------------------------------------------------------------
# v8：外生证据路径
# ---------------------------------------------------------------------------
# 三项缺陷的共同形状是"判据的参考量由被判决的对象自己产生"。v8 把证据移到
# CAM++ 之外（VAD 静音结构 + 跨段预测），并把人数从常量 2 改成三态。
# 只在 v8 lane 下生效；c1 路径保持 v7 行为不变。
ISO_GAP = 0.02          # k=2 段级间隙低于此 = 没有可分的第二人
PURE_RESCUE_MAX = 0.05  # 救援候选的无静音支撑切换占比上限
SINGLE_WIN_MAX = 0.10   # 判单人的无静音支撑切换占比上限

_A2D = {}


def a2_segments_diag():
    """最近一次 v8 路径的诊断（人数判据 / 救援门证据），供报告使用。"""
    return dict(_A2D)


def _v8_embs(x, sr, dur, vsegs, spk_model, win_s, hop_s):
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < MIN_WIN_OVERLAP * win_s:
            continue
        chunk = x[int(s0 * sr):int(e0 * sr)]
        if len(chunk) < 1600:
            continue
        try:
            e = spk_model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)
                        if hasattr(e, "detach") else np.asarray(e).reshape(-1))
            times.append(float(t))
    return embs, times


def _v8_out(out):
    return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
            for (s0, s1, lab) in out]


def _a2_segments_v8(x, sr, dur, vsegs, spk_model, win_s, hop_s, jump_off,
                    clust_n, link_mode, cluster_level, gap_merge, dom_max):
    """与 a2_segments 同结构，但人数判定与救援取舍改用外生证据。"""
    import claude_a2_v8 as V8
    from scipy.cluster.hierarchy import linkage, fcluster
    _A2D.clear()

    embs, times = _v8_embs(x, sr, dur, vsegs, spk_model, win_s, hop_s)
    _A2D["n_win"] = len(embs)
    if len(embs) < 2:
        _A2D.update(n_speakers=1, count_reason="窗数不足")
        return []
    E = np.stack(embs)

    seg_reps, seg_wst, segs_tmp = _build_seg(E, times, win_s)
    seg_tuples = [(s0, s1, 0) for s0, s1, _ in segs_tmp]

    if len(seg_reps) < 2:
        _A2D.update(n_speakers=1, count_reason="单一代表段")
        return [{"start": float(segs_tmp[0][0]), "end": float(segs_tmp[-1][1]),
                 "spk": 0}]

    tree = V8.fit_multi_k(seg_reps, kmax=4)
    _A2D["seg_tree"] = tree
    n_spk, why = V8.count_gate(len(seg_reps), tree, vsegs, seg_tuples,
                               gap_single=ISO_GAP)
    _A2D["n_speakers"] = n_spk
    _A2D["count_reason"] = "；".join(why)
    if n_spk == 1:
        return [{"start": float(segs_tmp[0][0]), "end": float(segs_tmp[-1][1]),
                 "spk": 0}]

    Zseg = linkage(seg_reps, method="average", metric="cosine")
    lab_seg = fcluster(Zseg, t=2, criterion="maxclust")
    lab2s = {int(l): i for i, l in enumerate(sorted(set(lab_seg)))}
    seqs = [lab2s[int(l)] for l in lab_seg]
    n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
    dom_ratio = max(np.bincount(seqs)) / len(seqs) if len(seqs) else 0
    collapsed = (n_switch <= 2 and 0.7 < dom_ratio < dom_max)

    rescue = None
    if collapsed:
        cand = _dense_rescue(x, sr, vsegs, spk_model)
        cand_ok = cand is not None and len(set(r[2] for r in cand)) >= 2
        if cand_ok:
            cand_m = _merge_runs([(a, b, l) for a, b, l in cand], gap_merge)
            seg_m = V8.run_stats(seg_tuples, vsegs)
            cand_st = V8.run_stats(cand_m, vsegs)
            keys = ("n_sw", "n_unsupported", "unsupported_ratio", "n_short",
                    "short_ratio", "median_dur")
            _A2D["rescue_gate"] = {
                "seg": {k: seg_m[k] for k in keys},
                "dense": {k: cand_st[k] for k in keys},
            }
            # 外生判据：dense 的切换必须大多落在真实静音上。v7 是拿同模型的
            # window 欠切结果比切换数（基线本身偏少切），那条路已废弃。
            if cand_st["unsupported_ratio"] <= PURE_RESCUE_MAX:
                rescue, cluster_level = cand, "dense_rescue"
                _A2D["rescue_decision"] = "dense_rescue（切换有静音支撑）"
            else:
                cluster_level = "window"
                _A2D["rescue_decision"] = (
                    f"回退 window（dense 无静音支撑切换 "
                    f"{cand_st['unsupported_ratio']:.0%} > {PURE_RESCUE_MAX:.0%}）")
        else:
            cluster_level = "window"
            _A2D["rescue_decision"] = "回退 window（救援无有效输出）"
    else:
        cluster_level = "segment"
        _A2D["rescue_decision"] = "非塌缩，走段级"

    if cluster_level == "dense_rescue":
        return _v8_out(_merge_runs([(a, b, l) for a, b, l in rescue],
                                   gap_merge))
    if cluster_level == "window":
        reps, segs = E, None
    else:
        reps, segs = _build_seg(E, times, win_s)[0], seg_tuples

    Za = linkage(reps, method="average", metric="cosine")
    Zc = linkage(reps, method="complete", metric="cosine")
    ga, gc = _gap2_of(Za), _gap2_of(Zc)
    if link_mode == "average":
        Z = Za
    elif link_mode == "complete":
        Z = Zc
    else:
        Z = Zc if ((ga and gc and gc > ga * 1.5)
                   or (ga and gc and gc - ga > 0.05)) else Za
    labels = fcluster(Z, t=2, criterion="maxclust")
    lab2 = {int(l): i for i, l in enumerate(sorted(set(labels)))}
    labseq = [lab2[int(l)] for l in labels]

    if (cluster_level == "segment" and len(set(labseq)) == 2
            and min(np.bincount(labseq)) >= 2):
        centroids = [reps[np.asarray(labseq) == k].mean(axis=0) for k in (0, 1)]
        refined = _dense_rescue(x, sr, vsegs, spk_model, centroids=centroids)
        if refined is not None:
            _A2D["rescue_decision"] = "短窗细化（段级身份锚）"
            return _v8_out(_merge_runs([(a, b, l) for a, b, l in refined],
                                       gap_merge))

    if cluster_level == "window":
        out, cur = [], None
        for st, lb in zip(times, labseq):
            if cur is not None and cur[2] == lb:
                cur = (cur[0], st + win_s, lb)
            else:
                if cur is not None:
                    out.append(cur)
                cur = (st, st + win_s, lb)
        if cur is not None:
            out.append(cur)
    else:
        out = [(s0, s1, lb) for (s0, s1, _), lb in zip(segs, labseq)]
    return _v8_out(_merge_runs(out, gap_merge))


def run_claude_lane(wav, models, lane_key, progress=None):
    """供 web_server 调用。models 为 {"vad","spk","asr"}（已加载 dict）。"""
    if lane_key == "c2":
        from claude_a2_v4 import run_claude_lane as run_v4
        return run_v4(wav, models, "c1", progress=progress)
    cfg = VERSIONS.get(lane_key)
    if cfg is None:
        raise ValueError(f"unknown claude lane {lane_key}")
    import time
    import soundfile as sf
    t0 = time.time()
    is_v8 = bool(cfg.get("v8"))
    if progress:
        progress(f"{cfg['label']}：VAD 门控 + 嵌入中…")
    segs = a2_segments(wav, models["spk"], models["vad"],
                       merge=False,
                       cluster_level=cfg["cluster_level"],
                       link_mode=cfg["link_mode"], dom_max=cfg["dom_max"],
                       v8=is_v8)
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    vsegs = _vad_segs(models["vad"], wav)
    sents = []
    for i, s in enumerate(segs):
        chunk = x[int(s["start"] * sr):int(s["end"] * sr)]
        text = ""
        if len(chunk) >= int(0.3 * sr):
            try:
                r = models["asr"].generate(input=chunk)
                text = r[0].get("text", "") or ""
            except Exception:
                text = ""
        sents.append({"start": round(float(s["start"]), 2),
                      "end": round(float(s["end"]), 2),
                      "spk": int(s["spk"]), "text": text})
        if progress and (i % 5 == 0 or i == len(segs) - 1):
            progress(f"{cfg['label']}：逐段转写 {i + 1}/{len(segs)}")
    n_clusters = len(set(s["spk"] for s in segs)) if segs else 0

    # ASR/人数契约：v7 这里是无条件 reliable，现在改成算出来的量。
    import claude_a2_asr_contract as C
    flags = []
    for e in sents:
        c = C.segment_contract(e["text"], e["start"], e["end"], vsegs)
        e["flag"] = c["flag"]
        e["asr_chars"] = c["chars"]
        e["asr_cps"] = c["cps"]
        if c["reasons"]:
            e["asr_note"] = "；".join(c["reasons"])
        flags.append(c["flag"])
    n_spk_decl = a2_segments_diag().get("n_speakers", n_clusters) if is_v8 else n_clusters
    # v8 的 verdict 是算出来的三态；c1 保持字面 "reliable"（v7 冻结归档的
    # 返回契约不能变，逐段 flag 仍附加），新语义只在 v8 lane 下生效。
    if is_v8:
        verdict, v_reasons = C._verdict_from(flags, n_spk_decl)
    else:
        verdict, v_reasons = "reliable", []
    diag = {"n_segments": len(segs), "n_clusters": n_clusters,
            "n_speakers": n_spk_decl,
            "n_flag_uncertain": sum(1 for f in flags if f != "ok")}
    if is_v8:
        diag.update(a2_segments_diag())
    return {
        "method": cfg["method"],
        "file": os.path.basename(wav),
        "elapsed": round(time.time() - t0, 1),
        "diag": diag,
        "verdict": verdict,
        "verdict_reasons": v_reasons,
        "sentence_info": sents,
    }
