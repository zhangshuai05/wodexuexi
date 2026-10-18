# -*- coding: utf-8 -*-
"""诊断: 提取每窗 CAM++ 嵌入, 统计相邻 cos 分布, 测试不同阈值下的段数/簇数。
输出 ASCII 为主, 避免 GBK 控制台乱码。用法: python diag_cos.py <wav>...
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

def cos_matrix(E):
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E @ E.T

def extract(wav, spk, win_s=1.2, hop_s=0.6):
    import soundfile as sf
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    win_n = int(win_s * sr)
    embs, times = [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, s1 = int(t * sr), int(t * sr) + win_n
        chunk = x[s0:s1]
        if len(chunk) < 1600: continue
        try:
            r = spk.generate(input=chunk)
            e = r[0].get("spk_embedding")
        except Exception:
            continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
    if not embs: return None, None, dur
    return np.stack(embs), times, dur

def main():
    from funasr import AutoModel
    spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", model_revision="master",
                    disable_update=True, disable_pbar=True)
    for wav in sys.argv[1:]:
        name = os.path.basename(wav)
        E, times, dur = extract(wav, spk)
        if E is None:
            print(f"{name}: NO EMBEDS"); continue
        S = np.triu(cos_matrix(E), 1)
        off = S[S > 0]
        print(f"\n=== {name}  dur={dur:.1f}s  n_win={len(E)}")
        sims = [cos_matrix(E[:2])[0,1]] if len(E) > 1 else []
        adj = [float(cos_matrix(E[i:i+2])[0,1]) for i in range(len(E)-1)]
        adj = np.array(adj)
        print(f"  adjacent cos: min={adj.min():.3f} med={np.median(adj):.3f} max={adj.max():.3f} "
              f"<0.5:{np.mean(adj<0.5)*100:.0f}% <0.6:{np.mean(adj<0.6)*100:.0f}%")
        # 全部窗口对 cos 直方图 (粗略分桶)
        qs = [0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        hist, edges = np.histogram(off, bins=[-1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
        print("  all-pairs cos hist:", " ".join(f"{int(h)}" for h in hist), "(<0.3|0.3-0.5|0.5-0.6|0.6-0.7|0.7-0.8|0.8-0.9|0.9+)")
        for th in (0.10, 0.15, 0.20, 0.25, 0.30):
            thresh = float(np.median(adj)) - th
            # 模拟当前 A2: 相邻 sims<thresh 则切段
            seg = []
            for i in range(len(E)-1):
                v = adj[i]
                if v < thresh:
                    seg.append(i+1)
            nseg = len(seg) + 1
            print(f"  med-{th:.2f} -> thresh={thresh:.3f}  n_seg={nseg}")
        # 粗聚类测试: 把所有窗口按 th 阈值在线分配 (单遍, 无段级)
        for th_assign in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            # 简单顺序聚类: 每窗与已有质心最大 cos >= th_assign 则归入, 否则新簇
            cs = []
            labs = []
            for e in E:
                if not cs:
                    cs.append(e.copy()); labs.append(0); continue
                c = np.array(cs)
                sim = (c @ e) / (np.linalg.norm(c, axis=1) * np.linalg.norm(e) + 1e-9)
                b = int(np.argmax(sim))
                if sim[b] >= th_assign:
                    cs[b] = 0.9*cs[b] + 0.1*e
                    labs.append(b)
                else:
                    cs.append(e.copy()); labs.append(len(cs)-1)
            # 去掉只出现一次的簇?
            from collections import Counter
            cnt = Counter(labs)
            keep = sum(1 for k,v in cnt.items() if v >= 3)
            print(f"  窗级 th_assign={th_assign:.2f} -> n_clu={len(cnt)} (>=3窗: {keep})")

if __name__ == "__main__":
    main()
