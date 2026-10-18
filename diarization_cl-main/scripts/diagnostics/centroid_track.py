# -*- coding: utf-8 -*-
"""
在线增量声纹质心追踪 v2
改进（借鉴 UIS-RNN 思想但无监督可用）:
  1) VAD 调细：短句不被合并、长段被拆，让说话人切换点更贴近句边界。
  2) 每个说话人"质心" = 最近 k 个嵌入的滑动均值，而不是全量均值
     （更像 UIS-RNN 的隐状态：状态随最近说话动态走，而不是整个历史平均）。
  3) 时间连续性先验：相邻段倾向于同一说话人，仅当相似度显著低于切换阈值才换人。
用法: python centroid_track.py <wav...>
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
VAD = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"

VAD_KWARGS = dict(
    max_single_segment_time=8000,   # 单段最长 8s，避免长段混多轮
    speech_to_sil_time_thres=120,   # 语音转静音阈值调小 -> 更早切段，保留短句边界
    sil_to_speech_time_thres=120,   # 静音转语音阈值调小 -> 更早开新段
    max_end_silence_time=400,       # 段尾静音容忍变短
)

class CentroidTracker:
    def __init__(self, alpha=0.85, thresh_new=0.70, thresh_switch=0.62,
                 buffer_k=6, time_cont_boost=0.0):
        self.alpha = alpha           # 质心(滑动均值)更新率
        self.thresh_new = thresh_new     # cos>=TH 归入并更新；否则考虑开新簇
        self.thresh_switch = thresh_switch  # 相邻连续段切换阈值（时间先验）
        self.buffer_k = buffer_k     # 每说话人保留最近 k 个嵌入算加权质心
        self.time_cont_boost = time_cont_boost
        self.buffers = []            # list[list[np.ndarray]]  每个说话人最近嵌入
        self.counts = []
    def centroid(self, spkid):
        b = self.buffers[spkid]
        if not b: return None
        # 加权：越近权重越大（时间衰减）
        w = np.linspace(0.5, 1.0, len(b))
        c = np.average(np.stack(b), axis=0, weights=w)
        return c / (np.linalg.norm(c) + 1e-9)
    def cos(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b))
    def assign(self, emb, prev_spk=None):
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if not self.buffers:
            self.buffers.append([emb.copy()]); self.counts.append(1); return 0
        sims = []
        for i in range(len(self.buffers)):
            c = self.centroid(i)
            s = self.cos(emb, c)
            # 时间连续先验：若与"上一段说话人"是同一人，给一点加成
            if prev_spk is not None and i == prev_spk:
                s = s + self.time_cont_boost
            sims.append(s)
        best = int(np.argmax(sims)); bs = sims[best]
        # 换人阈值（时间先验下更保守）：与最大相似 < thresh_switch 才不沿用
        # 若有上一段且最大不是上个说话人，则需显著超过 thresh_new 才允许换
        if bs >= self.thresh_switch:
            # 归入并更新
            self.buffers[best].append(emb.copy())
            if len(self.buffers[best]) > self.buffer_k:
                self.buffers[best].pop(0)
            self.counts[best] += 1
            return best
        else:
            self.buffers.append([emb.copy()]); self.counts.append(1)
            return len(self.buffers) - 1

def main():
    from funasr import AutoModel
    import soundfile as sf
    print("加载 VAD(细) + CAM++ ...", flush=True)
    vad = AutoModel(model=VAD, model_revision="master",
                    vad_kwargs=VAD_KWARGS, disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master",
                    disable_update=True, disable_pbar=True)

    wavs = sys.argv[1:]
    if not wavs:
        print("用法: python centroid_track.py <wav...>"); return
    for wav in wavs:
        print(f"\n########### {os.path.basename(wav)} ###########", flush=True)
        x, sr = sf.read(wav)
        if x.ndim > 1: x = x[:, 0]
        vr = vad.generate(input=wav)
        segs = []
        for r in vr:
            v = r.get("value")
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                for it in v:
                    if len(it) >= 2:
                        segs.append((float(it[0])/1000.0, float(it[1])/1000.0))
        segs = [(s,e) for (s,e) in segs if e-s >= 0.25]
        print(f"  VAD 段数(细化后): {len(segs)}", flush=True)

        tracker = CentroidTracker()
        labels = []
        prev = None
        for (st, en) in segs:
            chunk = x[int(st*sr):int(en*sr)]
            if len(chunk) < int(0.2*sr): continue
            try:
                r = spk.generate(input=chunk)
                e = r[0].get("spk_embedding")
            except Exception as ex:
                print("  err", ex); continue
            if e is None: continue
            e = e.detach().cpu().numpy().reshape(-1)
            lab = tracker.assign(e, prev_spk=prev)
            labels.append((st, en, lab))
            prev = lab

        # 合并相邻同簇段
        from collections import Counter
        merged = []
        for (st, en, lab) in labels:
            if merged and merged[-1][2] == lab and merged[-1][1] >= st - 0.05:
                merged[-1] = (merged[-1][0], en, lab)
            else:
                merged.append((st, en, lab))
        cnt = Counter(l for _,_,l in merged)
        print(f"  说话人簇数: {len(cnt)}  分布: {dict(cnt)}  段数(合并后): {len(merged)}", flush=True)
        for st, en, lab in merged:
            print(f"    [{st:6.2f}-{en:6.2f}] spk={lab}", flush=True)

if __name__ == "__main__":
    main()
