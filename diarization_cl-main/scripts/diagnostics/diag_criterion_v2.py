# -*- coding: utf-8 -*-
"""离线验证新判据(auto 改进): 段级聚类 -> 若 (切换少 & 主导占比在合理区间 & gap大) 才用窗口。
与原判据差异: 给 dom 加了一个上界 (dom_max=0.95)，排除"段级几乎纯单说话人"(假塌缩)的情况。
对 23 文件输出: 原判据选什么 / 新判据选什么 / 各自最优匹配。
"""
import os, sys, json
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import cs, VAD, SPK

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")

FILES = [
    ("Device116", "Call20260718210228.wav"), ("Device116", "Call20260718210454.wav"),
    ("Device116", "Call20260718210657.wav"), ("Device116", "Call20260718210836.wav"),
    ("Device116", "Call20260718210948.wav"), ("Device116", "Call20260718211031.wav"),
    ("Device118", "Note20260803111143.wav"), ("Device118", "Note20260803111436.wav"),
    ("Device118", "Note20260803111605.wav"), ("Device118", "Note20260803111741.wav"),
    ("Device118", "Note20260803111855.wav"), ("Device118", "Note20260803111943.wav"),
    ("Device116", "Call20260718211423.wav"), ("Device116", "Call20260718211506.wav"),
    ("Device116", "Call20260718211745.wav"), ("Device116", "Call20260718211842.wav"),
    ("Device116", "Call20260718211938.wav"),
    ("Device118", "Note20260803112220.wav"), ("Device118", "Note20260803112329.wav"),
    ("Device118", "Note20260803112415.wav"), ("Device118", "Note20260803112655.wav"),
    ("Device118", "Note20260803112752.wav"), ("Device118", "Note20260803112841.wav"),
]
QUIET = set('''Call20260718211423.wav Call20260718211506.wav Call20260718211745.wav
Call20260718211842.wav Call20260718211938.wav Note20260803112329.wav
Note20260803112415.wav Note20260803112655.wav Note20260803112752.wav
Note20260803112841.wav'''.split())
# 最优: 从之前 auto vs seg matched 得来 (auto>seg=window, seg>auto=segment, 相等=tie)
BEST = {'Call20260718210228.wav':'seg','Call20260718210454.wav':'tie','Call20260718210657.wav':'tie',
        'Call20260718210836.wav':'window','Call20260718210948.wav':'seg','Call20260718211031.wav':'tie',
        'Note20260803111143.wav':'seg','Note20260803111436.wav':'tie','Note20260803111605.wav':'tie',
        'Note20260803111741.wav':'seg','Note20260803111855.wav':'tie','Note20260803111943.wav':'tie',
        'Call20260718211423.wav':'window','Call20260718211506.wav':'window','Call20260718211745.wav':'window',
        'Call20260718211842.wav':'tie','Call20260718211938.wav':'tie','Note20260803112220.wav':'tie',
        'Note20260803112329.wav':'tie','Note20260803112415.wav':'tie','Note20260803112655.wav':'tie',
        'Note20260803112752.wav':'tie','Note20260803112841.wav':'tie'}

# --- 重跑: 用与原判据相同的锚点(段级avg的sw/dom/gap) ---
def segment_stats(E, times, win_s, jump_off=0.20):
    adj = np.array([cs(E[i], E[i + 1]) for i in range(len(E) - 1)])
    th_jump = float(np.median(adj)) - jump_off
    idx = [0]
    for i in range(len(E) - 1):
        if adj[i] < th_jump:
            idx.append(i + 1)
    idx.append(len(E))
    reps = []
    for a, b in zip(idx[:-1], idx[1:]):
        rep = E[a] if b - a < 2 else E[a:b].mean(axis=0)
        reps.append(rep)
    reps = np.stack(reps)
    Z = linkage(reps, method="average", metric="cosine")
    d = Z[:, 2]
    lab = fcluster(Z, t=2, criterion="maxclust")
    lab2 = {int(l): i for i, l in enumerate(sorted(set(lab)))}
    seqs = [lab2[int(l)] for l in lab]
    n_switch = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
    dom = max(np.bincount(seqs)) / len(seqs)
    k = len(d) - 2
    seg_gap = float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None
    return n_switch, dom, seg_gap


def criterion(n_switch, dom, seg_gap, dom_max):
    """新判据: 段级塌缩(切换少+主导占优) 但 没有出现极端单说话人独占(dom<dom_max)。"""
    collapsed = (n_switch <= 2 and dom > 0.7 and dom < dom_max)
    return "window" if collapsed else "segment"


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    # 统计
    from collections import Counter
    orig_correct = new_correct = 0
    sw_diffs = Counter()
    for sub, fname in FILES:
        wav = os.path.join(AUDIO, sub, fname)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        win_s = 1.2; hop = 0.6; win_n = int(win_s * sr)
        embs, times, rms, _t = [], [], [], []
        for t in np.arange(0, len(x)/sr - win_s, hop):
            s0 = int(t * sr); e0 = s0 + win_n
            if e0 > len(x): break
            ch = x[s0:e0]
            if len(ch) < 1600: continue
            rms.append(float(np.sqrt(np.mean(ch**2)))); _t.append(float(t))
        rms = np.array(rms); smask = rms >= np.percentile(rms, 25) if len(rms) else rms >= 0
        for t, m in zip(_t, smask):
            if not m: continue
            s0 = int(t * sr); e0 = s0 + win_n
            try:
                r = spk.generate(input=x[s0:e0]); e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
        if len(embs) < 6:
            print("%-28s not enough embs" % fname, flush=True); continue
        E = np.stack(embs)
        n_switch, dom, seg_gap = segment_stats(E, times, win_s)
        best = BEST.get(fname, '?')
        # 原判据 (auto 之前用 n_switch<=2 and dom>0.7, 无上界)
        orig_use = "window" if (n_switch <= 2 and dom > 0.7) else "segment"
        # 新 1: 加 dom 上界 0.95
        new_use = criterion(n_switch, dom, seg_gap, 0.95)
        def is_correct(u):
            return (u == best) or best == 'tie'
        if is_correct(orig_use): orig_correct += 1
        if is_correct(new_use): new_correct += 1
        if orig_use != new_use: sw_diffs[fname] = (orig_use, new_use, best, n_switch, dom, seg_gap)
        print("%-28s sw=%2d dom=%.2f gap=%.3f | orig=%-8s new=%-8s best=%s" % (
            fname, n_switch, dom, seg_gap if seg_gap is not None else -1, orig_use, new_use, best), flush=True)
    print("\n原判据 选对 = %d/23 ; 新判据(dom<0.95) 选对 = %d/23" % (orig_correct, new_correct))
    print("判据切换的文件:", dict(sw_diffs))


if __name__ == "__main__":
    main()
