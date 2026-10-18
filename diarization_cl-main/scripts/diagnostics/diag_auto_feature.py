# -*- coding: utf-8 -*-
"""探针: 对全 23 文件输出 auto 判据的三要素(seg switches / dom / seg_gap) vs 最优模式。
目的: 找一个能把「真塌缩(该用窗口)」和「假塌缩(该用段级)」分开的单调阈值。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from scipy.cluster.hierarchy import linkage, fcluster
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
BEST = {'Call20260718210228.wav':'seg','Call20260718210454.wav':'tie','Call20260718210657.wav':'tie',
        'Call20260718210836.wav':'window','Call20260718210948.wav':'seg','Call20260718211031.wav':'tie',
        'Note20260803111143.wav':'seg','Note20260803111436.wav':'tie','Note20260803111605.wav':'tie',
        'Note20260803111741.wav':'seg','Note20260803111855.wav':'tie','Note20260803111943.wav':'tie',
        'Call20260718211423.wav':'window','Call20260718211506.wav':'window','Call20260718211745.wav':'window',
        'Call20260718211842.wav':'tie','Call20260718211938.wav':'tie','Note20260803112220.wav':'tie',
        'Note20260803112329.wav':'tie','Note20260803112415.wav':'tie','Note20260803112655.wav':'tie',
        'Note20260803112752.wav':'tie','Note20260803112841.wav':'tie'}


def seg_stats(E, times, win_s, jump_off=0.20):
    adj = np.array([cs(E[i], E[i + 1]) for i in range(len(E) - 1)])
    th_jump = float(np.median(adj)) - jump_off
    seg_idx = [0]
    for i in range(len(E) - 1):
        if adj[i] < th_jump:
            seg_idx.append(i + 1)
    seg_idx.append(len(E))
    reps, wstarts, segs = [], [], []
    for a, b in zip(seg_idx[:-1], seg_idx[1:]):
        rep = E[a] if b - a < 2 else E[a:b].mean(axis=0)
        reps.append(rep); wstarts.append(times[a])
        segs.append((times[a], times[min(b, len(times) - 1)] + win_s, rep))
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
    return n_switch, dom, seg_gap, len(reps)


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    rows = []
    for sub, fname in FILES:
        wav = os.path.join(AUDIO, sub, fname)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        win_s = 1.2; hop = 0.6; win_n = int(win_s * sr)
        # 重算窗口嵌入(与 a2_v2 一致但不用 VAD 门控, 简化; 仅用非静音窗)
        embs, times = [], []
        rms = []; _t = []
        for t in np.arange(0, len(x)/sr - win_s, hop):
            s0 = int(t*sr); e0 = s0+win_n
            if e0 > len(x): break
            ch = x[s0:e0]
            if len(ch) < 1600: continue
            rms.append(float(np.sqrt(np.mean(ch**2)))); _t.append(float(t))
        rms = np.array(rms); spk_mask = rms >= np.percentile(rms, 25)
        for t, m in zip(_t, spk_mask):
            if not m: continue
            s0 = int(t*sr); e0 = s0+win_n
            try:
                r = spk.generate(input=x[s0:e0])
                e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
        if len(embs) < 6:
            rows.append({"file": fname, "env": env, "switches": 0, "dom": 0, "seg_gap": None, "nseg": 0, "best": BEST.get(fname, '?')})
            print("%-28s %4s sw=%2d dom=%.2f gap=%.3f nseg=%-3d best=%s" % (fname, '?', 0, 0.0, 0.0, 0, '?'), flush=True)
            continue
        E = np.stack(embs)
        env = "安静" if fname in QUIET else "嘈杂"
        n_switch, dom, seg_gap, nseg = seg_stats(E, times, win_s)
        rows.append({"file": fname, "env": env, "switches": n_switch, "dom": dom, "seg_gap": seg_gap, "nseg": nseg, "best": BEST.get(fname, '?')})
        print("%-28s %4s sw=%2d dom=%.2f gap=%.3f nseg=%-3d best=%s" % (
            fname, env, n_switch, dom, seg_gap if seg_gap is not None else -1, nseg, BEST.get(fname, '?')), flush=True)
    json.dump(rows, open(os.path.join(BASE, "artifacts", "diag_auto_feature.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
