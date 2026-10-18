# -*- coding: utf-8 -*-
"""全量 23 个标注数据集推理 (A2 v3 auto 版)。输出 diar_result/a2_v3_full/{file}_diar.json。
auto = link_mode auto + cluster_level auto (窗口级自动判据, 修复塌缩)。
与 batch_a2_v2_asr.py 同一套代码, 只是覆盖全部 23 个 ground_truth 文件。
"""
import os, sys, json
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import a2_segments_v2, VAD, SPK, SEACO

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "a2_v3_full")
os.makedirs(OUT, exist_ok=True)

# DSH ground_truth.json 的全部 23 个文件, 按 (dir, wav) 顺序
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
ALL_WAV = [f for _, f in FILES]


def main():
    import soundfile as sf
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    asr = AutoModel(model=SEACO, model_revision="master", disable_update=True, disable_pbar=True,
                    vad_model=None)
    done = 0
    for sub, fname in FILES:
        base = fname.rsplit(".", 1)[0]
        outjson = os.path.join(OUT, base + "_diar.json")
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
            print(f"  ERR {fname}: {e}", flush=True)
    print(f"\n完成: {done}/{len(FILES)} -> {OUT}")


if __name__ == "__main__":
    main()
