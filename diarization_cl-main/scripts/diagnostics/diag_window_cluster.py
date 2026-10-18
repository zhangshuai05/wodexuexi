# -*- coding: utf-8 -*-
"""11745: 窗口级 complete 聚类 vs 段级(跳变)complete 聚类, 看哪个保留了交替结构
思路: 跨视角实验已证明窗口嵌入含说话人信息(30s后大多ash,中间有wzy).
  若窗口级2簇标签序列比段级更交错, 说明分段(段代表均值)丢失了切换点.
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster

def win_embs(wav):
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad, wav)
    win_s, hop_s = 1.2, 0.6
    embs, starts = [], []
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
            embs.append(e.detach().cpu().numpy().reshape(-1)); starts.append(s0)
    return np.stack(embs), starts, sr, x

def main():
    wav = os.path.join(AUDIO, "Device116", "Call20260718211745.wav")
    E, starts, sr, x = win_embs(wav)
    print(f"窗口数={len(E)}")
    # 窗口级 complete/average 2簇
    for meth in ("average", "complete"):
        Z = linkage(E, method=meth, metric="cosine")
        labels = fcluster(Z, t=2, criterion="maxclust")
        lab2 = {int(l): i for i, l in enumerate(sorted(set(labels)))}
        seq = [lab2[int(l)] for l in labels]
        # 转成时间序列(每窗), 计算切换次数
        switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        d = Z[:, 2]; k = len(d)-2
        gap2 = float(d[k+1]-d[k]) if 0 <= k+1 < len(d) else None
        print(f"\n  {meth} 窗口级 gap2={gap2:.3f} 切换数={switches}")
        # 打印时间+标签
        line = ""
        for st, lb in zip(starts, seq):
            line += f"{st:.1f}:{lb} "
        print(f"  {line}")
    # 对照: 段级(跳变) 标签
    vsegs = _vad_segs.__wrapped__ if False else None
    print("\n对照段级(跳变分段)标签: 000111111111 (从前面诊断)")

if __name__ == "__main__":
    main()
