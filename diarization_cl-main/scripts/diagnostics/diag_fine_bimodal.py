# -*- coding: utf-8 -*-
"""细粒度双峰探测: 0.4s 子窗(0.3s hop)重提 CAM++ embedding, 看嘈杂1m 能否挖出远端.
对每个文件报告: 子窗总数, 2-cluster vs 主导占比, cosine gap, 各簇占用.
"""
import os, sys, json, numpy as np

os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
BASE = r"c:\Users\EDY\Desktop\new"
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"


def cs(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import soundfile as sf
    from funasr import AutoModel
    from scipy.cluster.hierarchy import linkage, fcluster
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)

    files = [
        ("Call20260718210836", "Audio/Device116/Call20260718210836.wav"),
        ("Note20260803111741", "Audio/Device118/Note20260803111741.wav"),
        ("Call20260718211745", "Audio/Device116/Call20260718211745.wav"),
        ("Note20260803112655", "Audio/Device118/Note20260803112655.wav"),
        # 对照组: 安静2m(现在 c1 最好的), 验证不会被误救
        ("Call20260718211506", "Audio/Device116/Call20260718211506.wav"),
        ("Note20260803112415", "Audio/Device118/Note20260803112415.wav"),
        # 嘈杂2m(现在 c1 0.517 也弱)
        ("Call20260718210657", "Audio/Device116/Call20260718210657.wav"),
        ("Note20260803111943", "Audio/Device118/Note20260803111943.wav"),
    ]
    for name, rel in files:
        wav = os.path.join(BASE, rel)
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x[:, 0]
        dur = len(x) / sr
        win_s, hop_s = 0.4, 0.3
        win_n = int(win_s * sr)
        embs, times = [], []
        for t in np.arange(0.0, dur - win_s, hop_s):
            c = x[int(t * sr):int((t + win_s) * sr)]
            if len(c) < 1600:
                continue
            try:
                r = spk.generate(input=c)
                e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1))
                times.append(float(t))
        if len(embs) < 4:
            print(f"{name}: only {len(embs)} subwins, skip"); continue
        E = np.stack(embs)
        # 2-cluster AHC(平均) + 余弦
        Z = linkage(E, method="average", metric="cosine")
        lab2 = fcluster(Z, t=2, criterion="maxclust")
        lab2s = {int(l): i for i, l in enumerate(sorted(set(lab2)))}
        seqs = [lab2s[int(l)] for l in lab2]
        cnt = np.bincount(seqs)
        dom = cnt.max() / len(seqs)
        # gap: 树顶两个merge高度差
        d = Z[:, 2]
        k = len(d) - 2
        gap2 = float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None
        # 2-cluster 簇间余弦距离(中心)
        c0 = E[[i for i, s in enumerate(seqs) if s == 0]].mean(axis=0)
        c1 = E[[i for i, s in enumerate(seqs) if s == 1]].mean(axis=0)
        inter = cs(c0, c1)
        # 跳变次数(子窗时间序列切换)
        n_sw = sum(1 for i in range(1, len(seqs)) if seqs[i] != seqs[i - 1])
        print(f"{name:<20} subwins={len(E)} spk2_dom={dom:.3f} "
              f"gap2={gap2:.3f} inter_cos={inter:.3f} n_sw={n_sw}")


if __name__ == "__main__":
    main()
