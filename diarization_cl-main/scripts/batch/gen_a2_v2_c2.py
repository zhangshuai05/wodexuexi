# -*- coding: utf-8 -*-
"""生成 c2 对照版(段级聚类 segment-level)。输出到 diar_result/a2_v2_c2/{file}_diar.json。
与 c1 唯一差异: cluster_level="segment"(旧版,过分割/1切换), 其余参数一致(link=auto, merge=False)。
用于在网页上对比「自动判别(窗口级) vs 段级」的差异。
"""
import os, sys, json
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import a2_segments_v2, VAD, SPK, SEACO

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_v2_c2")
os.makedirs(OUT, exist_ok=True)

FILES = [
    "Call20260718211423.wav", "Call20260718211506.wav", "Call20260718211745.wav",
    "Call20260718211842.wav", "Call20260718211938.wav",
    "Note20260803112329.wav", "Note20260803112415.wav", "Note20260803112655.wav",
    "Note20260803112752.wav", "Note20260803112841.wav",
]


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    asr = AutoModel(model=SEACO, model_revision="master", disable_update=True, disable_pbar=True,
                    vad_model=None)
    for fname in FILES:
        sub = "Device116" if fname.startswith("Call") else "Device118"
        wav = os.path.join(AUDIO, sub, fname)
        outjson = os.path.join(OUT, fname.rsplit(".", 1)[0] + "_diar.json")
        if os.path.exists(outjson):
            print(f"skip {fname}", flush=True); continue
        print(f"--- {fname} ---", flush=True)
        try:
            segs = a2_segments_v2(wav, spk, vad, merge=False, link_mode="auto",
                                  cluster_level="segment")
            x, sr = sf.read(wav)
            if x.ndim > 1: x = x[:, 0]
            sents = []
            for s in segs:
                chunk = x[int(s["start"] * sr):int(s["end"] * sr)]
                if len(chunk) < int(0.3 * sr):
                    sents.append({"start": s["start"], "end": s["end"], "spk": s["spk"], "text": ""}); continue
                try:
                    r = asr.generate(input=chunk)
                    text = r[0].get("text", "") or ""
                except Exception:
                    text = ""
                sents.append({"start": s["start"], "end": s["end"], "spk": s["spk"], "text": text})
            json.dump({"sentence_info": sents}, open(outjson, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"  {len(sents)} 段 saved", flush=True)
        except Exception as e:
            print(f"  ERR {fname}: {e}", flush=True)


if __name__ == "__main__":
    main()
