# -*- coding: utf-8 -*-
"""VAD 门控诊断: FSMN-VAD(fix) 段 vs CAM++ 窗口, 验证门控是否消除静音窗垃圾嵌入。
用法: python vad_gate_diag.py <wav>...
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
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", model_revision="master",
                    disable_update=True, disable_pbar=True)
    for wav in sys.argv[1:]:
        base = os.path.basename(wav)
        res = vad.generate(input=wav, dynamic_silence=False)
        if isinstance(res, list):
            segs = []
            for r in res:
                for it in r.get("value", []):
                    segs.append((float(it[0]) / 1000.0, float(it[1]) / 1000.0))
        else:
            segs = [(float(it[0]) / 1000.0, float(it[1]) / 1000.0) for it in res.get("value", [])]
        tot = sum(b - a for a, b in segs)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        print(f"\n=== {base}  dur={len(x)/sr:.1f}s  vad_segs={len(segs)} speech={tot:.1f}s ({tot/(len(x)/sr)*100:.0f}%)")
        # 窗口只取 VAD 段内的: 每窗与 VAD 段交集 >= 50% 才保留
        win_s, hop_s = 1.2, 0.6
        embs, times = [], []
        for t in np.arange(0.0, len(x)/sr - win_s, hop_s):
            s0, e0 = t, t + win_s
            inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in segs)
            if inter < 0.5 * win_s: continue
            chunk = x[int(s0*sr):int(e0*sr)]
            try:
                r = spk.generate(input=chunk)
                e = r[0].get("spk_embedding")
            except Exception:
                continue
            if e is not None:
                embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
        E = np.stack(embs)
        print(f"  gated windows: {len(E)} (原 79)")
        adj = np.array([cs(E[i], E[i+1]) for i in range(len(E)-1)])
        print(f"  gated adjacent cos: min={adj.min():.3f} med={np.median(adj):.3f} max={adj.max():.3f} <0.5:{np.mean(adj<0.5)*100:.0f}%")
        allp = (E / (np.linalg.norm(E, axis=1, keepdims=True)+1e-9)) @ (E / (np.linalg.norm(E, axis=1, keepdims=True)+1e-9)).T
        off = allp[np.triu_indices(len(E), 1)]
        h, _ = np.histogram(off, bins=[-1,0.3,0.5,0.6,0.7,0.8,0.9,1.01])
        print("  all-pairs cos:", " ".join(map(str, h)), "(<0.3|.3-.5|.5-.6|.6-.7|.7-.8|.8-.9|.9+)")
        # 段代表嵌入 AHC: 先把 VAD 段合并(间隙<1s), 每段窗口平均
        if not segs: continue
        merged = []
        for a, b in segs:
            if merged and a - merged[-1][1] < 1.0:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        reps = []
        for a, b in merged:
            w = [i for i, t in enumerate(times) if a <= t < b]
            if w:
                reps.append((a, b, E[w].mean(axis=0)))
        print(f"  merged VAD segs={len(merged)}  with-speech reps={len(reps)}")
        # 相邻代表间 cos
        if len(reps) >= 2:
            rc = [cs(reps[i][2], reps[i+1][2]) for i in range(len(reps)-1)]
            print(f"  adj rep cos: min={min(rc):.3f} med={np.median(rc):.3f} max={max(rc):.3f}")

if __name__ == "__main__":
    main()
