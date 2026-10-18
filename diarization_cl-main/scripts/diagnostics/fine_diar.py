# -*- coding: utf-8 -*-
"""
A2: 细粒度说话人分离 (Fine-grained Speaker Diarization)
不依赖 VAD 粗段; 用 CAM++ 对滑动窗逐窗嵌入, 得 embedding 时间序列,
再检测"说话人切换点" => 精细分段。

方法(三选一, 主用前两):
  1) 嵌入跳变检测: 相邻窗 cos 相似度骤降 => 切换点
  2) 增量质心追踪(复用 centroid_track 的 CentroidTracker)
  3) 对比: 固定窗长窗距

用法: python fine_diar.py <wav> [win_s=1.0 hop_s=0.5]
输出: 分段 [t0-t1] spk=X (基于跳变后重新聚类), 及与真值对比提示
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"

class CentroidTracker:
    """复用增量质心分配(与 centroid_track 一致)"""
    def __init__(self, alpha=0.90, thresh_new=0.70, thresh_assign=0.45):
        self.alpha = alpha; self.thresh_new = thresh_new; self.thresh_assign = thresh_assign
        self.centroids = []; self.counts = []
    def cos(self, a, b):
        a = a/(np.linalg.norm(a)+1e-9); b = b/(np.linalg.norm(b)+1e-9)
        return float(np.dot(a, b))
    def assign(self, emb):
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if not self.centroids:
            self.centroids.append(emb); self.counts.append(1); return 0
        sims = [self.cos(emb, c) for c in self.centroids]
        best = int(np.argmax(sims)); bs = sims[best]
        if bs >= self.thresh_assign:
            self.centroids[best] = self.alpha*self.centroids[best] + (1-self.alpha)*emb
            self.centroids[best] /= (np.linalg.norm(self.centroids[best])+1e-9)
            self.counts[best] += 1
            return best
        else:
            self.centroids.append(emb); self.counts.append(1)
            return len(self.centroids)-1

def main():
    from funasr import AutoModel
    import soundfile as sf
    wav = sys.argv[1]
    win_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    hop_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    print(f"加载 CAM++ ...", flush=True)
    spk = AutoModel(model=SPK, model_revision="master",
                    disable_update=True, disable_pbar=True)
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    print(f"音频: {os.path.basename(wav)}  dur={dur:.1f}s  win={win_s}s hop={hop_s}s", flush=True)

    # 1) 逐窗嵌入序列 (窗长/hop 正确, 过滤过短窗)
    embs, times = [], []
    win_n = int(win_s * sr); hop_n = int(hop_s * sr)
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0 = int(t * sr); s1 = s0 + win_n
        if s1 > len(x): break
        chunk = x[s0:s1]
        if len(chunk) < 1600:  # 至少 100ms
            continue
        try:
            r = spk.generate(input=chunk)
            e = r[0].get("spk_embedding")
        except Exception as ex:
            continue  # 跳过坏窗, 不中断
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1))
            times.append(float(t))
    print(f"嵌入数: {len(embs)} (原总窗≈{int((dur-win_s)/hop_s)+1})", flush=True)
    if not embs:
        print("无嵌入"); return
    E = np.stack(embs)

    # 2) 相邻 cos 相似度序列 => 跳变检测 (粗滤: 低能量窗标记为非语音)
    def cosv(a, b):
        a = a/(np.linalg.norm(a)+1e-9); b = b/(np.linalg.norm(b)+1e-9)
        return float(np.dot(a, b))
    # 用窗内 RMS 能量过滤：安静窗不参与跳变
    win_rms = []
    for t in times:
        s0 = int(t*sr); s1 = s0 + win_n
        seg = x[s0:min(s1, len(x))]
        win_rms.append(float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0)
    rms_arr = np.array(win_rms)
    # 语音门限: 取 RMS 的前 60% 分位视为有语音(低能量=噪声/静音, 跳过)
    speech_mask = rms_arr >= np.percentile(rms_arr, 25)
    print(f"语音窗占比: {speech_mask.mean():.2f} (非语音窗跳过)", flush=True)

    sims = [cosv(E[i], E[i+1]) for i in range(len(E)-1)]
    med = float(np.median(sims))
    thresh = med - 0.20
    print(f"相邻相似度: min={min(sims):.3f} med={med:.3f} max={max(sims):.3f}  切换阈值≈{thresh:.3f}", flush=True)

    # 3) 分段: 只在"语音窗"内找跳变
    seg = []
    cur_start = times[0]; cur_emb_top = E[0]
    for i in range(len(E)-1):
        # 只考虑两窗都有语音
        if not (speech_mask[i] and speech_mask[i+1]):
            continue
        if sims[i] < thresh:
            seg.append((cur_start, times[i+1], cur_emb_top))
            cur_start = times[i+1]; cur_emb_top = E[i+1]
    seg.append((cur_start, times[-1]+win_s, cur_emb_top))
    # 每段用其中所有窗嵌入的重心做代表, 再对段分配质心
    tracker = CentroidTracker()
    out = []
    # 用段内平均嵌入作为该段代表
    seg_reps = []
    idx = 0
    for (s0, s1, etop) in seg:
        # 找落在 [s0,s1) 的窗
        wids = [i for i,t0 in enumerate(times) if s0 <= t0 < s1]
        reps = E[wids].mean(axis=0) if wids else etop
        seg_reps.append((s0, s1, reps))
    # 增量质心分配 (段级)
    for (s0, s1, rep) in seg_reps:
        lab = tracker.assign(rep)
        out.append((s0, s1, lab))
    # 合并相邻同标号
    merged = []
    for (s0, s1, lab) in out:
        if merged and merged[-1][2] == lab and merged[-1][1] >= s0 - 0.01:
            merged[-1] = (merged[-1][0], s1, lab)
        else:
            merged.append((s0, s1, lab))
    from collections import Counter
    cnt = Counter(l for _,_,l in merged)
    print(f"切换点数: {len(seg)-1}  分段数: {len(merged)}  说话人簇: {dict(cnt)}", flush=True)
    for (s0, s1, lab) in merged:
        print(f"  [{s0:6.2f}-{s1:6.2f}] spk={lab}", flush=True)

if __name__ == "__main__":
    main()
