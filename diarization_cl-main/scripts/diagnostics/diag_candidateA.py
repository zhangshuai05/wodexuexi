# -*- coding: utf-8 -*-
"""验证候选A: 段级 complete vs average 链接, 看能否把塌缩文件的离群短答分离出来
输出: 每个文件的 段标签序列(2簇), 比较 average/complete 是否交错"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster

def segs_of(wav):
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad, wav)
    win_s, hop_s = 1.2, 0.6
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < 0.5 * win_s: continue
        chunk = x[int(s0*sr):int(e0*sr)]
        if len(chunk) < 1600: continue
        try:
            e = spk.generate(input=chunk)[0].get("spk_embedding")
        except Exception: continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
    E = np.stack(embs)
    adj = np.array([cs(E[i], E[i+1]) for i in range(len(E)-1)])
    th = float(np.median(adj)) - 0.20
    idx = [0]
    for i in range(len(E)-1):
        if adj[i] < th: idx.append(i+1)
    idx.append(len(E))
    segs = []
    for a, b in zip(idx[:-1], idx[1:]):
        rep = E[a] if b-a < 2 else E[a:b].mean(axis=0)
        segs.append((times[a], times[min(b, len(times)-1)] + win_s, rep))
    return segs

def run(fname, device):
    wav = os.path.join(AUDIO, device, fname)
    segs = segs_of(wav)
    reps = np.stack([s[2] for s in segs])
    print(f"\n=== {fname} ({device}) 段数={len(segs)} ===")
    for meth in ("average", "complete"):
        Z = linkage(reps, method=meth, metric="cosine")
        labels = fcluster(Z, t=2, criterion="maxclust")
        lab2 = {int(l): i for i, l in enumerate(sorted(set(labels)))}
        seq = "".join(str(lab2[int(l)]) for l in labels)
        d = Z[:, 2]; k = len(d) - 2
        gap2 = float(d[k+1]-d[k]) if 0 <= k+1 < len(d) else None
        print(f"  {meth:9s} gap2={gap2:.3f} 标签序列={seq}")
        print(f"             {'  '.join(f'{s[0]:.1f}~{s[1]:.1f}:{lab2[int(l)]}' for s,l in zip(segs,labels))}")

if __name__ == "__main__":
    run("Call20260718211745.wav", "Device116")
    run("Call20260718211423.wav", "Device116")
    run("Note20260803112655.wav", "Device118")
