# -*- coding: utf-8 -*-
"""用真值 speaker 时段验证: 0.6s/0.15s 密集窗上, 远端说话人是否可分离.
对每个 1m 文件:
  - 从 ground_truth 的 turns 得到每说话人的时间区间(用文本对齐到句子的时间窗)
  - 用真值说话人时段内的密集窗均值做质心 cA/cB
  - 看每个窗口 cosA-cosB 边距是否把真值说话人分开(可分离性上界)
真值本身没有逐句时间, 但 ground_truth.turns 无时间. 因此改用:
  a2_v3_improved 的句段(已带时间+spk)作为"可信时段"来源? 不行, 那是 c1 标签.
  所以此法不可行——真值无时间.
  替代: 直接对密集窗做 2-cluster, 看是否有 ≥2 个可靠簇且在时间上交替(远/近双说话人).
  这是"不依赖真值"的可分离性上界.
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

    # 1m 文件: 嘈杂4 + 安静4
    files = [
        ("Call20260718210836", "Audio/Device116/Call20260718210836.wav"),
        ("Note20260803111741", "Audio/Device118/Note20260803111741.wav"),
        ("Call20260718211745", "Audio/Device116/Call20260718211745.wav"),
        ("Note20260803112655", "Audio/Device118/Note20260803112655.wav"),
        ("Call20260718210228", "Audio/Device116/Call20260718210228.wav"),
        ("Note20260803111143", "Audio/Device118/Note20260803111143.wav"),
        ("Call20260718211938", "Audio/Device116/Call20260718211938.wav"),
        ("Note20260803112415", "Audio/Device118/Note20260803112415.wav"),
    ]
    for name, rel in files:
        wav = os.path.join(BASE, rel)
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x[:, 0]
        dur = len(x) / sr
        win, hop = 0.6, 0.15
        win_n = int(win * sr)
        embs, times = [], []
        for t in np.arange(0.0, dur - win, hop):
            c = x[int(t * sr):int((t + win) * sr)]
            if len(c) < 1600:
                continue
            try:
                e = spk.generate(input=c)[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1))
                times.append(float(t))
        if len(embs) < 8:
            print(f"{name}: few dense windows {len(embs)}"); continue
        E = np.stack(embs)
        # 2-cluster AHC 看两簇是否分离且交替
        Z = linkage(E, method="average", metric="cosine")
        lab = fcluster(Z, t=2, criterion="maxclust")
        labm = {int(l): i for i, l in enumerate(sorted(set(lab)))}
        seq = [labm[int(l)] for l in lab]
        n1 = seq.count(0)
        n2 = seq.count(1)
        if min(n1, n2) < 5:
            print(f"{name}: too lopsided n1={n1} n2={n2}"); continue
        m0 = E[[i for i, s in enumerate(seq) if s == 0]].mean(axis=0)
        m1 = E[[i for i, s in enumerate(seq) if s == 1]].mean(axis=0)
        d = Z[:, 2]
        k = len(d) - 2
        gap2 = float(d[k + 1] - d[k]) if 0 <= k + 1 < len(d) else None
        # 交替次数
        n_sw = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        inter = cs(m0, m1)
        dom = max(n1, n2) / len(seq)
        print(f"{name:<20} dense={len(E)} n1={n1} n2={n2} dom={dom:.3f} "
              f"gap2={gap2:.3f} inter={inter:.3f} n_sw={n_sw} dur={dur:.1f}s")


if __name__ == "__main__":
    main()
