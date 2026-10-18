# -*- coding: utf-8 -*-
"""批量 A2+ASR: 22 文件 -> sentence_info(带文本) -> 存 diar_result/a2_all/{file}_diar.json"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_all")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts", "audio"))
import json
from a2_with_asr import a2_segments  # 复用 A2 分段

def main():
    import soundfile as sf
    from funasr import AutoModel
    spk = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", model_revision="master",
                    disable_update=True, disable_pbar=True)
    asr = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    model_revision="master", disable_update=True, disable_pbar=True)
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
    for (dist,env,cust,ca,no) in PAIRS:
        if env != "quiet":
            continue
        for fname in (ca,no):
            wav = os.path.join(AUDIO, "Device116" if fname.startswith("Call") else "Device118", fname)
            outjson = os.path.join(OUT, fname.rsplit(".",1)[0] + "_diar.json")
            if os.path.exists(outjson):
                print(f"skip {fname}", flush=True); continue
            print(f"--- {fname} ---", flush=True)
            try:
                segs = a2_segments(wav, spk)
                x, sr = sf.read(wav)
                if x.ndim>1: x=x[:,0]
                sents=[]
                for s in segs:
                    chunk=x[int(s["start"]*sr):int(s["end"]*sr)]
                    if len(chunk)<int(0.3*sr):
                        sents.append({"start":s["start"],"end":s["end"],"spk":s["spk"],"text":""}); continue
                    try:
                        r=asr.generate(input=chunk)
                        text=r[0].get("text","") or ""
                    except Exception:
                        text=""
                    sents.append({"start":s["start"],"end":s["end"],"spk":s["spk"],"text":text})
                json.dump({"sentence_info":sents}, open(outjson,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
                print(f"  {len(sents)} 段 saved", flush=True)
            except Exception as e:
                print(f"  ERR {fname}: {e}", flush=True)

if __name__=="__main__":
    main()
