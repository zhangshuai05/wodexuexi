# -*- coding: utf-8 -*-
"""仅诊断 A2-v2 聚类 gap: 不跑 ASR, 打印每文件的 gap2/gap3 和真实簇数选择"""
import os, sys
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import a2_segments_v2, VAD, SPK

def main():
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    files = [
        "Call20260718211745.wav","Note20260803112655.wav",
        "Call20260718211423.wav","Note20260803112329.wav",
        "Call20260718211842.wav","Note20260803112752.wav",
        "Call20260718211506.wav","Note20260803112415.wav",
        "Call20260718211938.wav","Note20260803112841.wav",
    ]
    for fname in files:
        wav = os.path.join(AUDIO, "Device116" if fname.startswith("Call") else "Device118", fname)
        print(f"=== {fname} ===", flush=True)
        segs = a2_segments_v2(wav, spk, vad)
        print(f"   最终 spk={sorted(set(s['spk'] for s in segs))} 段数={len(segs)}", flush=True)

if __name__=="__main__":
    main()
