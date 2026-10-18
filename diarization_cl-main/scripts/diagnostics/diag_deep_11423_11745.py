# -*- coding: utf-8 -*-
"""深入诊断 11423/11745 塌缩: 窗口嵌入的 AHC 树 + 段代表的两两距离, 找能不能分开"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

def diag(fname, device):
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    wav = os.path.join(AUDIO, device, fname)
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad, wav)
    win_s, hop_s = 1.2, 0.6
    win_n = int(win_s * sr)
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < 0.5 * win_s: continue
        chunk = x[int(s0*sr):int(e0*sr)]
        if len(chunk) < 1600: continue
        try:
            r = spk.generate(input=chunk)
            e = r[0].get("spk_embedding")
        except Exception: continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
    E = np.stack(embs)
    print(f"\n{'='*60}\n{fname} ({device}) 窗口数={len(E)} 时长={dur:.1f}s")
    # VAD段
    print(f"VAD段: {[(round(a,1),round(b,1)) for a,b in vsegs]}")
    # 跳变分段(与a2_v2一致)
    adj = np.array([cs(E[i], E[i+1]) for i in range(len(E)-1)])
    th_jump = float(np.median(adj)) - 0.20
    seg_idx = [0]
    for i in range(len(E)-1):
        if adj[i] < th_jump: seg_idx.append(i+1)
    seg_idx.append(len(E))
    segs = []
    for a, b in zip(seg_idx[:-1], seg_idx[1:]):
        rep = E[a] if b-a < 2 else E[a:b].mean(axis=0)
        segs.append((times[a], times[min(b, len(times)-1)] + win_s, rep))
    print(f"跳变段数={len(segs)} th_jump={th_jump:.3f}")
    reps = np.stack([s[2] for s in segs])
    # 段距离矩阵
    D = squareform(pdist(reps, metric="cosine"))
    n = len(reps)
    print("段两两cos距离(下三角, 只列<0.9的):")
    for i in range(n):
        row = [f"{D[i,j]:.2f}" for j in range(n)]
        print(f"  seg{i} {segs[i][0]:.1f}-{segs[i][1]:.1f}: " + " ".join(row))
    # 段内平均邻接 vs 跨段
    intra_adj = np.mean([adj[i] for i in range(len(adj))]) if len(adj) else 0
    # 最大段内方差
    print(f"段代表平均cos距离: {np.mean(D[np.triu_indices(n,1)]):.3f}")
    print(f"段代表最小cos距离: {D[np.triu_indices(n,1)].min():.3f}")

if __name__=="__main__":
    diag("Call20260718211423.wav", "Device116")
    diag("Call20260718211745.wav", "Device116")
