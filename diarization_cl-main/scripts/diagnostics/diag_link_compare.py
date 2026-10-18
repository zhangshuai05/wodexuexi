# -*- coding: utf-8 -*-
"""段级 complete vs average gap2 全量对比: 找自适应选链接的判据
也输出按complete/candidate2簇的matched(后面用)"""
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

FILES = [
    ("1m","Call20260718211745.wav","Device116"),("1m","Note20260803112655.wav","Device118"),
    ("1.5m","Call20260718211423.wav","Device116"),("1.5m","Call20260718211842.wav","Device116"),
    ("2m","Call20260718211506.wav","Device116"),("2m","Call20260718211938.wav","Device116"),
    ("1.5m","Note20260803112329.wav","Device118"),("2m","Note20260803112415.wav","Device118"),
    ("1.5m","Note20260803112752.wav","Device118"),("2m","Note20260803112841.wav","Device118"),
]
def main():
    print(f"{'file':<26}{'nsegs':>6}{'avg_gap2':>9}{'comp_gap2':>10}  标签序列(avg/comp)")
    for dist, f, dev in FILES:
        wav = os.path.join(AUDIO, dev, f)
        segs = segs_of(wav)
        reps = np.stack([s[2] for s in segs])
        res = {}
        for meth in ("average","complete"):
            Z = linkage(reps, method=meth, metric="cosine")
            d = Z[:, 2]; k = len(d)-2
            g = float(d[k+1]-d[k]) if 0 <= k+1 < len(d) else 0
            labels = fcluster(Z, t=2, criterion="maxclust")
            lab2 = {int(l): i for i,l in enumerate(sorted(set(labels)))}
            res[meth] = (g, "".join(str(lab2[int(l)]) for l in labels))
        ag, aseq = res["average"]; cg, cseq = res["complete"]
        print(f"{f:<26}{len(segs):>6}{ag:>9.3f}{cg:>10.3f}  {aseq} / {cseq}")

if __name__=="__main__":
    main()
