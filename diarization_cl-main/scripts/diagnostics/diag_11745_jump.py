# -*- coding: utf-8 -*-
"""诊断 11745: 聚类前的原始分段(跳变分段), 看段边界和每段 spk 分配"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf

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
    print(f"窗口数={len(E)}, VAD段={[(round(a,1),round(b,1)) for a,b in vsegs]}")
    adj = np.array([cs(E[i], E[i+1]) for i in range(len(E)-1)])
    print("adj(相邻窗口cos):", [round(float(a),2) for a in adj])
    th_jump = float(np.median(adj)) - 0.20
    print(f"跳变阈值={th_jump:.3f} (median={np.median(adj):.3f})")
    seg_idx = [0]
    for i in range(len(E)-1):
        if adj[i] < th_jump: seg_idx.append(i+1)
    seg_idx.append(len(E))
    print(f"跳变段边界(window idx): {seg_idx}")
    for a, b in zip(seg_idx[:-1], seg_idx[1:]):
        print(f"  段 {times[a]:.2f}-{times[min(b, len(times)-1)]+win_s:.2f}  ({b-a}窗)")

if __name__=="__main__":
    main()
