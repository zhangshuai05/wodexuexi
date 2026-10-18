# -*- coding: utf-8 -*-
"""诊断 c1(auto) 在嘈杂1m的失败机制。
对比: 几个嘈杂1m文件窗口级 vs 段级的标签结构。
关键: 若段级主导占比≥0.95(判近单说话人) 但在窗口级能分出2人 → 说明是判据把信号吞了, 可优化。
      若窗口级也揉成一团 → 嵌入真分不开, 得动嵌入/检测层。
"""
import os, sys, json, numpy as np
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from scipy.cluster.hierarchy import linkage, fcluster
from a2_v2 import cs, VAD, SPK

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")

# 嘈杂1m: Call10836(c1 23段2人) Note11741(c1 19段2人)
# 对照 安静2m(11506) / 嘈杂2m(11938) / 嘈杂1.5m(11436)
CASES = [
    ("Device116", "Call20260718210836.wav", "嘈杂1m"),
    ("Device118", "Note20260803111741.wav", "嘈杂1m"),
    ("Device116", "Call20260718211506.wav", "安静2m"),
    ("Device116", "Call20260718211938.wav", "嘈杂2m"),
    ("Device118", "Note20260803111436.wav", "嘈杂1.5m"),
]


def seg_stats(E, times, win_s, jump_off=0.20):
    adj = np.array([cs(E[i], E[i + 1]) for i in range(len(E) - 1)])
    th_jump = float(np.median(adj)) - jump_off
    idx = [0]
    for i in range(len(E) - 1):
        if adj[i] < th_jump:
            idx.append(i + 1)
    idx.append(len(E))
    reps = []
    for a, b in zip(idx[:-1], idx[1:]):
        reps.append(E[a] if b - a < 2 else E[a:b].mean(axis=0))
    reps = np.stack(reps)
    # 段级 k=2 聚类
    Z = linkage(reps, method="average", metric="cosine")
    lab = fcluster(Z, t=2, criterion="maxclust")
    lab2 = {int(l): i for i, l in enumerate(sorted(set(lab)))}
    seqs = [lab2[int(l)] for l in lab]
    n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
    dom = max(np.bincount(seqs)) / len(seqs)
    return n_switch, dom, len(reps)


def win_stats(E, jump_off=0.0):
    """窗口级直接 k=2 聚类看分得开吗"""
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(E, method="average", metric="cosine")
    d = Z[:, 2]
    k = len(d) - 2
    g2 = float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None
    lab = fcluster(Z, t=2, criterion="maxclust")
    lab2 = {int(l): i for i, l in enumerate(sorted(set(lab)))}
    seqs = [lab2[int(l)] for l in lab]
    n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
    dom = max(np.bincount(seqs)) / len(seqs)
    return g2, n_switch, dom


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    win_s = 1.2; hop = 0.6; win_n = int(win_s * 16000)
    print("%-28s %-8s | 段级 sw/dom | 级内gap2 sw/dom | 判定" % ("file", "场景"), flush=True)
    for sub, fname, tag in CASES:
        wav = os.path.join(AUDIO, sub, fname)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        vsegs = []
        dur = len(x) / sr
        embs, times = [], []
        for t in np.arange(0.0, dur - win_s, hop):
            s0, e0 = t, t + win_s
            chunk = x[int(s0 * sr):int(e0 * sr)]
            if len(chunk) < 1600: continue
            try:
                r = spk.generate(input=chunk); e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
        E = np.stack(embs)
        sw, dom0, nseg = seg_stats(E, times, win_s)
        # 窗口级(高频VAD门控): 这里简化直接全部窗口
        if len(E) >= 6:
            g2, sw2, dom2 = win_stats(E)
            winstr = f"{g2:.3f} {sw2}/{dom2:.2f}"
        else:
            winstr = "n/a"
        print("%-28s %-8s |   %2d /%.2f | %s" % (fname, tag, sw, dom0, winstr), flush=True)


if __name__ == "__main__":
    main()
