# -*- coding: utf-8 -*-
"""A2-v2 诊断: VAD 门控 + 窗口嵌入 + 跳变分段 + 段代表 AHC 合并阶梯。
对每个安静文件打印: 段数、AHC 平均连接合并距离阶梯、各阈值下簇数。
用数据决定聚类阈值/规则。用法: python a2_v2_diag.py <wav>...
"""
import os, sys, json, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAD = os.path.join(r"C:\Users\EDY\Desktop\new",
                   "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")

def cs(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def main():
    from funasr import AutoModel
    import soundfile as sf
    from scipy.cluster.hierarchy import linkage, fcluster
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", model_revision="master",
                    disable_update=True, disable_pbar=True)
    win_s, hop_s = 1.2, 0.6
    for wav in sys.argv[1:]:
        base = os.path.basename(wav)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        dur = len(x) / sr
        res = vad.generate(input=wav, dynamic_silence=False)
        vsegs = []
        for r in (res if isinstance(res, list) else [res]):
            for it in r.get("value", []):
                vsegs.append((float(it[0]) / 1000.0, float(it[1]) / 1000.0))
        # 门控: 窗与 VAD 段交集 >= 50%
        embs, times = [], []
        for t in np.arange(0.0, dur - win_s, hop_s):
            s0, e0 = t, t + win_s
            inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
            if inter < 0.5 * win_s: continue
            chunk = x[int(s0*sr):int(e0*sr)]
            try:
                r = spk.generate(input=chunk)
                e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
        if len(embs) < 2:
            print(f"{base}: gated windows={len(embs)} <2, skip"); continue
        E = np.stack(embs)
        # 跳变分段(用 gated 序列)
        adj = np.array([cs(E[i], E[i+1]) for i in range(len(E)-1)])
        th_jump = float(np.median(adj)) - 0.20
        seg_idx = [0]
        for i in range(len(E)-1):
            if adj[i] < th_jump:
                seg_idx.append(i+1)
        seg_idx.append(len(E))
        reps = []
        for a, b in zip(seg_idx[:-1], seg_idx[1:]):
            reps.append(E[a:b].mean(axis=0))
        print(f"\n=== {base} dur={dur:.1f}s vad_segs={len(vsegs)} gated_win={len(E)} "
              f"th_jump={th_jump:.3f} segs={len(reps)}")
        if len(reps) < 2: continue
        # AHC 平均连接, 距离 = 1 - cos
        Z = linkage(reps, method="average", metric="cosine")
        # 合并距离阶梯 (从近到远): Z[:,2] 第 k 次合并的距离
        d = Z[:, 2]
        print(f"  merge distances: " + " ".join(f"{v:.3f}" for v in d))
        # 相邻差最大的位置 -> 自然簇数
        gaps = np.diff(d)
        gi = int(np.argmax(gaps)) if len(gaps) else -1
        n_gap = len(d) - gi  # 切在最大间隙处时的簇数
        print(f"  largest gap at idx={gi} (0-based) -> natural n_clu={n_gap} (gap={gaps[gi]:.3f})")
        for th in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
            labels = fcluster(Z, t=th, criterion="distance")
            print(f"    th_dist={th:.2f} -> n_clu={len(set(labels))}")

if __name__ == "__main__":
    main()
