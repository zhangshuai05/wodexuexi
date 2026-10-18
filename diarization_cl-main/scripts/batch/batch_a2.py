# -*- coding: utf-8 -*-
"""
B1: A2 细粒度说话人分离 - 全量 22 文件 (11对 x 顾客麦/店员麦)
对每个 wav 跑 fine_diar 逻辑(滑窗嵌入+语音门控+跳变+增量质心),
输出: 每文件 分段数 / spk数 / 每段[起-止 spk]
汇总到 diar_result/a2_all/summary.json + _matrix.txt
funasr 环境。
"""
import os, sys, json, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_all")
os.makedirs(OUT, exist_ok=True)
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"

class CentroidTracker:
    def __init__(self, alpha=0.90, thresh_new=0.70, thresh_assign=0.45):
        self.alpha=alpha; self.thresh_new=thresh_new; self.thresh_assign=thresh_assign
        self.centroids=[]; self.counts=[]
    def cos(self,a,b):
        a=a/(np.linalg.norm(a)+1e-9); b=b/(np.linalg.norm(b)+1e-9)
        return float(np.dot(a,b))
    def assign(self,emb):
        emb=np.asarray(emb,dtype=np.float32).reshape(-1)
        if not self.centroids:
            self.centroids.append(emb); self.counts.append(1); return 0
        sims=[self.cos(emb,c) for c in self.centroids]
        best=int(np.argmax(sims)); bs=sims[best]
        if bs>=self.thresh_assign:
            self.centroids[best]=self.alpha*self.centroids[best]+(1-self.alpha)*emb
            self.centroids[best]/= (np.linalg.norm(self.centroids[best])+1e-9)
            self.counts[best]+=1; return best
        else:
            self.centroids.append(emb); self.counts.append(1)
            return len(self.centroids)-1

def fine_diar(wav, spk_model, win_s=1.2, hop_s=0.6):
    import soundfile as sf
    x, sr = sf.read(wav)
    if x.ndim>1: x=x[:,0]
    dur = len(x)/sr
    win_n=int(win_s*sr)
    embs,times=[],[]
    for t in np.arange(0.0, dur-win_s, hop_s):
        s0=int(t*sr); s1=s0+win_n
        if s1>len(x): break
        chunk=x[s0:s1]
        if len(chunk)<1600: continue
        try:
            r=spk_model.generate(input=chunk)
            e=r[0].get("spk_embedding")
        except Exception:
            continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)); times.append(float(t))
    if not embs: return []
    E=np.stack(embs)
    # 语音门控
    win_rms=[]
    for t in times:
        s0=int(t*sr); s1=min(s0+win_n,len(x))
        seg=x[s0:s1]
        win_rms.append(float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0)
    rms_arr=np.array(win_rms)
    speech_mask = rms_arr >= np.percentile(rms_arr,25)
    def cosv(a,b):
        a=a/(np.linalg.norm(a)+1e-9); b=b/(np.linalg.norm(b)+1e-9)
        return float(np.dot(a,b))
    sims=[cosv(E[i],E[i+1]) for i in range(len(E)-1)]
    med=float(np.median(sims))
    thresh=med-0.20
    seg=[]; cur_start=times[0]; cur_top=E[0]
    for i in range(len(E)-1):
        if not (speech_mask[i] and speech_mask[i+1]): continue
        if sims[i]<thresh:
            seg.append((cur_start,times[i+1],cur_top)); cur_start=times[i+1]; cur_top=E[i+1]
    seg.append((cur_start, times[-1]+win_s, cur_top))
    # 段代表 -> 增量分配
    tracker=CentroidTracker()
    out=[]
    for (s0,s1,_) in seg:
        wids=[i for i,t0 in enumerate(times) if s0<=t0<s1]
        rep=E[wids].mean(axis=0) if wids else E[0]
        lab=tracker.assign(rep)
        out.append((s0,s1,lab))
    # 合并相邻同标号
    merged=[]
    for (s0,s1,lab) in out:
        if merged and merged[-1][2]==lab and merged[-1][1]>=s0-0.01:
            merged[-1]=(merged[-1][0],s1,lab)
        else:
            merged.append((s0,s1,lab))
    return [{"start":round(s0,2),"end":round(s1,2),"spk":int(lab)} for (s0,s1,lab) in merged]

def main():
    from funasr import AutoModel
    print("加载 CAM++ ...", flush=True)
    spk = AutoModel(model=SPK, model_revision="master",
                    disable_update=True, disable_pbar=True)
    PAIRS = [
        ("1m","noisy","zjl","Call20260718210228.wav","Note20260803111143.wav"),
        ("1m","noisy","ash","Call20260718210836.wav","Note20260803111741.wav"),
        ("1m","quiet","ash","Call20260718211745.wav","Note20260803112655.wav"),
        ("1.5m","noisy","zjl","Call20260718210454.wav","Note20260803111436.wav"),
        ("1.5m","noisy","ash","Call20260718210948.wav","Note20260803111855.wav"),
        ("1.5m","quiet","zjl","Call20260718211423.wav","Note20260803112329.wav"),
        ("1.5m","quiet","ash","Call20260718211842.wav","Note20260803112752.wav"),
        ("2m","noisy","zjl","Call20260718210657.wav","Note20260803111605.wav"),
        ("2m","noisy","ash","Call20260718211031.wav","Note20260803111943.wav"),
        ("2m","quiet","zjl","Call20260718211506.wav","Note20260803112415.wav"),
        ("2m","quiet","ash","Call20260718211938.wav","Note20260803112841.wav"),
    ]
    summary=[]
    for (dist,env,cust,ca,no) in PAIRS:
        row={"dist":dist,"env":env,"cust":cust,"call":ca,"note":no}
        for chan, fname in [("customer",ca),("staff",no)]:
            wav = os.path.join(AUDIO, "Device116" if fname.startswith("Call") else "Device118", fname)
            try:
                segs = fine_diar(wav, spk)
            except Exception as e:
                print("  ERR", fname, e, flush=True); segs=[]
            from collections import Counter
            cnt = Counter(s["spk"] for s in segs)
            row[chan]={"n_seg":len(segs), "n_spk":len(cnt), "spk_cnt":dict(cnt),
                       "segs":segs}
            print(f"[{dist}|{env}|{cust}|{chan}] {fname}: seg={len(segs)} spk={len(cnt)} {dict(cnt)}", flush=True)
        json.dump(row, open(os.path.join(OUT, f"a2_{dist}_{env}_{cust}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        summary.append(row)
    json.dump(summary, open(os.path.join(OUT,"summary.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print("完成 -> ", os.path.join(OUT,"summary.json"), flush=True)

if __name__=="__main__":
    main()
