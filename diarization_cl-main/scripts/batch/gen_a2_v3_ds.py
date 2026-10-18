# -*- coding: utf-8 -*-
"""c1+dense_rescue 全量 23 文件推理。输出 diar_result/a2_v3_ds。
复用 gen_a2_v3_improved 但调用新的 a2_v2.a2_segments_v2 (已含 dense_rescue)。
"""
import os, sys, json, time
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import a2_segments_v2, VAD, SPK, SEACO

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_v3_ds")
os.makedirs(OUT, exist_ok=True)

FILES = [
    ("Device116", "Call20260718210228.wav"), ("Device116", "Call20260718210454.wav"),
    ("Device116", "Call20260718210657.wav"), ("Device116", "Call20260718210836.wav"),
    ("Device116", "Call20260718210948.wav"), ("Device116", "Call20260718211031.wav"),
    ("Device118", "Note20260803111143.wav"), ("Device118", "Note20260803111436.wav"),
    ("Device118", "Note20260803111605.wav"), ("Device118", "Note20260803111741.wav"),
    ("Device118", "Note20260803111855.wav"), ("Device118", "Note20260803111943.wav"),
    ("Device116", "Call20260718211423.wav"), ("Device116", "Call20260718211506.wav"),
    ("Device116", "Call20260718211745.wav"), ("Device116", "Call20260718211842.wav"),
    ("Device116", "Call20260718211938.wav"),
    ("Device118", "Note20260803112220.wav"), ("Device118", "Note20260803112329.wav"),
    ("Device118", "Note20260803112415.wav"), ("Device118", "Note20260803112655.wav"),
    ("Device118", "Note20260803112752.wav"), ("Device118", "Note20260803112841.wav"),
]


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    asr = AutoModel(model=SEACO, model_revision="master", disable_update=True, disable_pbar=True,
                    vad_model=None)
    done = 0
    t0 = time.time()
    for sub, fname in FILES:
        outjson = os.path.join(OUT, fname.rsplit(".", 1)[0] + "_diar.json")
        if os.path.exists(outjson):
            print(f"skip {fname}", flush=True); done += 1; continue
        wav = os.path.join(AUDIO, sub, fname)
        print(f"--- {fname} ---", flush=True)
        try:
            segs = a2_segments_v2(wav, spk, vad, merge=False)
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
            done += 1
        except Exception as e:
            import traceback
            print(f"  ERR {fname}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n完成: {done}/{len(FILES)} -> {OUT}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
