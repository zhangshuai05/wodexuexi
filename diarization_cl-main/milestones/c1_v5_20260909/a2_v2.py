# -*- coding: utf-8 -*-
"""A2-v2: VAD 门控 + 跳变分段 + 段代表 AHC 聚类(替代贪心质心)。

当前 c1: 段级质心作为身份参考, 0.6s/0.15s 短窗细化换人边界。
每簇至少有两段才启用细化; 默认不把同标签的原始段拼成长段。

相对 v1 的改动:
  1. VAD(dynamic_silence=False) 门控, 只保留与 VAD 段重叠>=50% 的窗口 -> 消除静音/噪声垃圾嵌入
  2. 跳变分段后, 段代表用 AHC(平均连接, cosine)聚类, 不再用增量质心
  3. 簇数用 "top-gap" 判据: 只在树顶(候选 n=2/3)里找最大间隙, 把 n_spks 逼向 2/3

用法: python a2_v2.py <wav> <out_suffix.json>
"""
import os, sys, json, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
VAD = os.path.join(BASE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"
SEACO = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


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
    from funasr import AutoModel
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
                  min_frac=0.3, n_iter=3, rescue_merge=0.35, dom_cap=0.93,
                  prune_sliver=0.0, centroids=None):
    """Dense 窗边距判决: 在段级塌缩文件中重新提取 0.6s/0.15s 密集窗 embedding,
    用"距主导质心最远"的窗重建远端质心, 逐窗按 cos 边距分类, 切成微段。
    centroids 给定时沿用段级身份, 在短窗上细化; 否则执行原塌缩救援。
    返回段列表[(start,end,spk)] (已 merge), 失败返回 None。"""
    import soundfile as sf
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
            embs.append(e.detach().cpu().numpy().reshape(-1))
            times.append(float(t))
    if len(embs) < 8:
        return None
    E = np.stack(embs)
    # 单位化便于向量化余弦
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def _cosv(cent):
        c = np.asarray(cent, dtype=np.float64)
        c = c / (np.linalg.norm(c) + 1e-9)
        return En @ c
    # 主导质心 = 全窗均值(近单说话人时约等于主导说话人)
    c_main = En.mean(axis=0)
    cos_main = _cosv(c_main)
    # 远端种子: 距主导质心最远的 30% 窗
    n_far = max(3, int(0.30 * len(E)))
    far_idx = np.argsort(cos_main)[:n_far]
    c_far = En[far_idx].mean(axis=0)
    if centroids is not None:
        c_main, c_far = centroids
    else:
        # 无身份锚时才自更新; 已有段级质心时保持身份参考稳定。
        for _ in range(n_iter):
            ca = _cosv(c_main)
            cb = _cosv(c_far)
            delta = ca - cb
            k = max(3, len(E) // 10)
            idxA = np.argsort(-delta)[:k]
            idxB = np.argsort(delta)[:k]
            c_main_new = En[idxA].mean(axis=0)
            c_far_new = En[idxB].mean(axis=0)
            if cs(c_main_new, c_far_new) > 0.98:
                break
            if cs(c_main_new, c_main) < 0.7:
                break
            c_main, c_far = c_main_new, c_far_new
    # 逐窗判决: 近主导的一边 vs 远端一边
    ca = _cosv(c_main)
    cb = _cosv(c_far)
    margins = ca - cb
    labels = (margins >= 0).astype(int)
    if centroids is not None:
        labels = _stabilize_labels(labels, margins)
        # Preserve the caller's speaker IDs and place boundaries between centers.
        labels = 1 - labels
    else:
        n1 = int((labels == 1).sum())
        # 严守: 远端占比过低(假分离)或两质心太像(单说话人), 放弃
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
    # 相邻同标签合并
    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < rescue_merge:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    # 最短段长 prune: 单段 < prune_sliver 且夹在两个同标签主导段之间 -> 并入主导
    # (远端短暂可见瞬间是真实信号; 过长会吞噬. 0.8s 上限仅清"孤影噪声")
    if prune_sliver > 0:
        changed = True
        while changed:
            changed = False
            for i in range(len(merged)):
                s0, s1, lab = merged[i]
                if (s1 - s0) >= prune_sliver:
                    continue
                left = merged[i - 1] if i > 0 else None
                right = merged[i + 1] if i + 1 < len(merged) else None
                if left and right and left[2] == right[2] and lab != left[2]:
                    # 夹在两个主导段之间 -> 并入较近者
                    if (s0 - left[1]) <= (right[0] - s1):
                        merged[i] = (left[0], s1, left[2])
                    else:
                        merged[i] = (s0, right[1], right[2])
                    changed = True
                elif left and lab != left[2] and (s0 - left[1]) < rescue_merge:
                    merged[i] = (left[0], s1, left[2])
                    changed = True
                elif right and lab != right[2] and (right[0] - s1) < rescue_merge:
                    merged[i] = (s0, right[1], right[2])
                    changed = True
                if changed:
                    break
        # 重新合并一次
        merged2 = []
        for (s0, s1, lab) in merged:
            if merged2 and merged2[-1][2] == lab and (s0 - merged2[-1][1]) < rescue_merge:
                merged2[-1] = (merged2[-1][0], s1, lab)
            else:
                merged2.append((s0, s1, lab))
        merged = merged2
    return merged


def a2_segments_v2(wav, spk_model, vad_model, win_s=1.2, hop_s=0.6,
                   jump_off=0.20, clust_n=(2, 2), merge=False, link_mode="auto",
                   cluster_level="auto", gap_merge=0.3, dom_max=0.95):
    import soundfile as sf
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad_model, wav)
    win_n = int(win_s * sr)
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < 0.5 * win_s:
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
            embs.append(e.detach().cpu().numpy().reshape(-1))
            times.append(float(t))
    if len(embs) < 2:
        return []
    E = np.stack(embs)

    from scipy.cluster.hierarchy import linkage, fcluster

    def _gap2_of(Z):
        d = Z[:, 2]
        k = len(d) - 2
        return float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None

    def _build_seg():
        """跳变分段 -> (reps, wstarts, seg_info)。 seg_info为None表示非段级"""
        adj = np.array([cs(E[i], E[i + 1]) for i in range(len(E) - 1)])
        th_jump = float(np.median(adj)) - jump_off
        seg_idx = [0]
        for i in range(len(E) - 1):
            if adj[i] < th_jump:
                seg_idx.append(i + 1)
        seg_idx.append(len(E))
        segs = []
        for a, b in zip(seg_idx[:-1], seg_idx[1:]):
            if b - a < 2:
                rep = E[a]
            else:
                rep = E[a:b].mean(axis=0)
            start, end = _segment_bounds(times, a, b, win_s)
            segs.append((start, end, rep))
        return np.stack([s[2] for s in segs]), [s[0] for s in segs], segs

    # 决定 cluster_level
    rescue = None
    if cluster_level == "auto":
        # 先跑段级看标签: 若段级塌缩(切换很少+某标签占绝对主导)则改用window
        seg_reps, seg_wst, segs_tmp = _build_seg()
        # 段级avg聚类看标签结构
        Zseg = linkage(seg_reps, method="average", metric="cosine")
        dseg = Zseg[:, 2]
        kseg = len(dseg) - 2
        seg_gap = float(dseg[kseg + 1] - dseg[kseg]) if 0 <= kseg + 1 < len(dseg) else None
        lab_seg = fcluster(Zseg, t=2, criterion="maxclust")
        lab2s = {int(l): i for i, l in enumerate(sorted(set(lab_seg)))}
        seqs = [lab2s[int(l)] for l in lab_seg]
        n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
        dom_ratio = max(np.bincount(seqs)) / len(seqs) if len(seqs) else 0
        # 塌缩判定: 切换很少 且 主导标签占优势(但非近乎单说话人独占).
        # dom>0.7 且 dom<0.95: 段级把两人揉成一团(真塌缩, 窗口可救).
        # dom>=0.95: 段级本就近单说话人(音频质量差/远场, 切了反而碎), 不强上窗口.
        collapsed = (n_switch <= 2 and 0.7 < dom_ratio < dom_max)
        print(f"  [level] seg_gap={_fmt(seg_gap)} switches={n_switch} "
              f"dom={dom_ratio:.2f} collapsed={collapsed}", flush=True)
        if collapsed:
            # 塌缩 = 段级把两人揉成一团。先试 dense 窗边距判决(专救远端被平均吞掉),
            # 成功且确认分成两说话人则用之; 失败回退窗口级聚类。
            rescue = _dense_rescue(x, sr, vsegs, spk_model)
            if rescue is not None:
                l2 = sorted(set(r[2] for r in rescue))
                if len(l2) >= 2:
                    print(f"  [rescue] dense 切分 -> {len(rescue)} 段 spk={l2}", flush=True)
                    cluster_level = "dense_rescue"
                else:
                    rescue = None
            if rescue is None:
                cluster_level = "window"
            else:
                cluster_level = "dense_rescue"
        else:
            cluster_level = "segment"
    if cluster_level == "window":
        reps = E
        wstarts = list(times)
        segs = None
    elif cluster_level == "dense_rescue":
        reps, wstarts, segs = None, None, None
    else:
        reps, wstarts, segs = _build_seg()

    # 选链接
    if cluster_level != "dense_rescue":
        Za = linkage(reps, method="average", metric="cosine")
        Zc = linkage(reps, method="complete", metric="cosine")
        ga, gc = _gap2_of(Za), _gap2_of(Zc)
        if link_mode == "average":
            Z, chosen = Za, "average"
        elif link_mode == "complete":
            Z, chosen = Zc, "complete"
        else:
            # auto: complete 能挖出被平均抹平的结构(comp_gap2 明显更大)时用 complete
            if ga and gc and gc > ga * 1.5:
                Z, chosen = Zc, "complete"
            elif ga and gc and gc - ga > 0.05:
                Z, chosen = Zc, "complete"
            else:
                Z, chosen = Za, "average"
        d = Z[:, 2]
        def _gap(cand):
            k = len(d) - cand
            if k >= 0 and k + 1 < len(d):
                return float(d[k + 1] - d[k])
            return None
        g2, g3 = _gap(2), _gap(3)
        print(f"  [聚类diag] link={chosen} nsegs={len(d)+1} "
              f"gap2_avg={_fmt(ga)} gap2_comp={_fmt(gc)} "
              f"final_gap2={_fmt(g2)} gap3={_fmt(g3)}", flush=True)
        # top-gap: 只在候选簇数范围内找最大间隙
        n = clust_n[0]
        best_gap = -1.0
        for cand in range(clust_n[0], clust_n[1] + 1):
            # 要得到 cand 簇, 切在 d[k] 与 d[k+1] 之间, k = len(d) - cand
            k = len(d) - cand
            if k >= 0 and k + 1 < len(d):
                jump = d[k + 1] - d[k]
            else:
                continue
            if jump > best_gap:
                best_gap = jump
                n = cand
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
        # 密集窗边距判决结果: 直接用 rescue 段(已含连续合并)
        out = [(s0, s1, lb) for (s0, s1, lb) in rescue]
        merged = []
        for (s0, s1, lab) in out:
            if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < gap_merge:
                merged[-1] = (merged[-1][0], s1, lab)
            else:
                merged.append((s0, s1, lab))
        if not merge:
            return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
                    for (s0, s1, lab) in out]
        return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
                for (s0, s1, lab) in merged]
    if cluster_level == "window":
        # 窗口级: 把连续同标签窗口合并成段 (start=组首窗start, end=组末窗start+win_s)
        out = []
        cur = None
        for st, lb in zip(wstarts, labseq):
            if cur is not None and cur[2] == lb and st - cur[1] < gap_merge:
                # 同标签且与上一窗近, 延伸end
                cur = (cur[0], st + win_s, lb)
            elif cur is not None and cur[2] == lb:
                cur = (cur[0], st + win_s, lb)  # 同标签但间隔大, 仍作为同段延伸(不换人)
            else:
                if cur is not None:
                    out.append(cur)
                cur = (st, st + win_s, lb)
        if cur is not None:
            out.append(cur)
    else:
        out = []
        for (s0, s1, _), lb in zip(segs, labseq):
            out.append((s0, s1, lb))
    # 相邻同标签合并(可选): 时间gap < gap_merge
    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and (s0 - merged[-1][1]) < gap_merge:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    if not merge:
        return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
                for (s0, s1, lab) in out]
    return [{"start": float(s0), "end": float(s1), "spk": int(lab)}
            for (s0, s1, lab) in merged]


def main(wav, outjson):
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    asr = AutoModel(model=SEACO, model_revision="master", disable_update=True, disable_pbar=True,
                    vad_model=None)
    segs = a2_segments_v2(wav, spk, vad)
    print(f"A2-v2 段数: {len(segs)}, spk={sorted(set(s['spk'] for s in segs))}", flush=True)
    import soundfile as sf
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    sents = []
    for s in segs:
        chunk = x[int(s["start"] * sr):int(s["end"] * sr)]
        if len(chunk) < int(0.3 * sr):
            sents.append({"start": s["start"], "end": s["end"], "spk": s["spk"], "text": ""})
            continue
        try:
            r = asr.generate(input=chunk)
            text = r[0].get("text", "") or ""
        except Exception:
            text = ""
        sents.append({"start": s["start"], "end": s["end"], "spk": s["spk"], "text": text})
    out = {"sentence_info": sents}
    with open(outjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved:", outjson, flush=True)
    for s in sents:
        print(f"  [{s['start']:6.2f}-{s['end']:6.2f}] spk={s['spk']} {s['text'][:28]}", flush=True)


if __name__ == "__main__":
    wav = sys.argv[1]
    outjson = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, "diar_result", "a2_v2", "_test.json")
    os.makedirs(os.path.dirname(outjson), exist_ok=True)
    main(wav, outjson)
