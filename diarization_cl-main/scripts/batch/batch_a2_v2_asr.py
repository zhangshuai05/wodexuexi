# -*- coding: utf-8 -*-
"""批量 A2-v2 + ASR: 10 个安静文件 -> sentence_info(带文本) -> diar_result/a2_v2/{file}_diar.json"""
import os, sys, json, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_v2")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BASE)
from a2_v2 import a2_segments_v2, VAD, SPK, SEACO  # 复用 v2 分段
# 可选参数: link_mode (average/complete/auto), cluster_level (segment/window), 默认 auto/segment
LINK_MODE = "auto"
CLUSTER_LEVEL = "auto"
# 按位置: argv[1]=link_mode, argv[2]=cluster_level
if len(sys.argv) > 1 and sys.argv[1] in ("average", "complete", "auto"):
    LINK_MODE = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] in ("segment", "window", "auto"):
    CLUSTER_LEVEL = sys.argv[2]
print(f"link_mode={LINK_MODE} cluster_level={CLUSTER_LEVEL}")

def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    asr = AutoModel(model=SEACO, model_revision="master", disable_update=True, disable_pbar=True,
                    vad_model=None)
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
                segs = a2_segments_v2(wav, spk, vad, merge=False, link_mode=LINK_MODE,
                                      cluster_level=CLUSTER_LEVEL)
                print(f"  段数: {len(segs)}, spk={sorted(set(s['spk'] for s in segs))}", flush=True)
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
