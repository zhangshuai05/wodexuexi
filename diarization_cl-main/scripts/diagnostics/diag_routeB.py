# -*- coding: utf-8 -*-
"""诊断路线B: 对比 11745(塌缩) vs 12655(成功) 的窗级嵌入结构
指标:
  1. 每个窗口嵌入到其最近 k 邻居的距离(局部紧度)分布 -> 是否有紧核
  2. 三种 AHC 链接(average/complete/ward)的 top-gap2
  3. 嵌入是否有"一紧核一松散外围"结构
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

def get_embs(wav):
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
    return np.stack(embs), times

def diag(fname, device, label):
    wav = os.path.join(AUDIO, device, fname)
    E, times = get_embs(wav)
    D = squareform(pdist(E, metric="cosine"))
    n = len(E)
    np.fill_diagonal(D, np.inf)
    # 局部紧度: 每点到其最近k邻居的平均距离
    k = min(3, n-1)
    knn = np.sort(D, axis=1)[:, :k].mean(axis=1)
    # 全局分布
    print(f"\n{'='*64}\n{label}: {fname} ({device})  窗数={n}")
    print(f"  kNN{k}距离: 均值={knn.mean():.3f} 中位={np.median(knn):.3f} "
          f"P10={np.percentile(knn,10):.3f} P90={np.percentile(knn,90):.3f} "
          f"min={knn.min():.3f} max={knn.max():.3f}")
    # 紧核刻画: 多少点落在 kNN<0.35 (紧区)
    for th in (0.3, 0.35, 0.4, 0.5):
        cnt = (knn < th).sum()
        print(f"    kNN<{th}: {cnt}点 ({cnt/n*100:.0f}%)")
    # 三种链接的 gap 序列(候选2/3)
    for meth in ("average", "complete", "ward"):
        metric = "euclidean" if meth == "ward" else "cosine"
        Z = linkage(E, method=meth, metric=metric)
        d = Z[:, 2]
        def gap(cand):
            k = len(d) - cand
            return float(d[k+1]-d[k]) if 0 <= k+1 < len(d) else None
        print(f"    {meth:9s}: gap2={fmt(gap(2))} gap3={fmt(gap(3))} "
              f"top3_d={[round(float(v),3) for v in d[-3:][::-1]]}")

def fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"

if __name__ == "__main__":
    diag("Call20260718211745.wav", "Device116", "塌缩-顾客麦戴ash")
    diag("Note20260803112655.wav", "Device118", "成功-店员麦戴wzy")
    diag("Call20260718211423.wav", "Device116", "塌缩-顾客麦")
    diag("Call20260718211842.wav", "Device116", "成功-顾客麦")
