# -*- coding: utf-8 -*-
"""诊断: 11745 强制2簇下 合并前的原始段标签序列, 看是否交错"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster

def main():
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    fname = "Call20260718211745.wav"
    wav = os.path.join(AUDIO, "Device116", fname)
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
    reps = np.stack([s[2] for s in segs])
    Z = linkage(reps, method="average", metric="cosine")
    for n in (2, 3):
        labels = fcluster(Z, t=n, criterion="maxclust")
        lab2 = {int(l): i for i, l in enumerate(sorted(set(labels)))}
        seq = "".join(str(lab2[int(l)]) for l in labels)
        print(f"n={n} 标签序列: {seq}")
        print(f"  {'  '.join(f'{s[0]:.1f}-{s[1]:.1f}:{lab2[int(l)]}' for s,l in zip(segs,labels))}")

if __name__=="__main__":
    main()
