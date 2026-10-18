# -*- coding: utf-8 -*-
"""
批量跑 说话人分离+识别，输出逐文件报告 + 距离汇总
funasr 环境。用法: python batch_run.py
会先加载一次模型，然后逐个跑 25 个文件，结果写 result/*.json + summary.csv
"""
import os, sys, json, glob, csv
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
RESULT = os.path.join(BASE, "diar_result")
os.makedirs(RESULT, exist_ok=True)

VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SEACO_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PUNC_MODEL = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
SPK_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"

def main():
    from funasr import AutoModel
    print("加载模型 ...")
    m = AutoModel(
        model=SEACO_MODEL, model_revision="master",
        vad_model=VAD_MODEL, vad_kwargs={"max_single_segment_time": 30000},
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL, spk_model_revision="master",
        spk_mode="punc_segment",
        disable_update=True, disable_pbar=True,
    )

    wavs = sorted(glob.glob(os.path.join(AUDIO, "Device116", "*.wav"))) + \
           sorted(glob.glob(os.path.join(AUDIO, "Device118", "*.wav")))
    # 可选: 传入部分文件名只跑这些
    if len(sys.argv) > 1:
        want = set(os.path.basename(a).lower() for a in sys.argv[1:])
        wavs = [w for w in wavs if os.path.basename(w).lower() in want]
    print(f"待处理 {len(wavs)} 个文件")

    summary = []
    for wav in wavs:
        name = os.path.basename(wav)
        print(f"--- {name} ---")
        try:
            res = m.generate(input=wav, batch_size=1, sentence_timestamp=True,
                             output_dir=RESULT, merge_vad=True, merge_length_s=15)
            out = res[0]
            si = out.get("sentence_info", [])
            rec = {"file": name, "text": out.get("text", ""),
                   "sentences": [{"start": s.get("start"), "end": s.get("end"),
                                  "spk": s.get("spk"), "text": s.get("text")} for s in si]}
            with open(os.path.join(RESULT, name + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            n = len(si)
            nspk = len(set(s.get("spk") for s in si))
            summary.append({"file": name, "device": "116" if "Device116" in wav else "118",
                            "n_sent": n, "n_spk": nspk, "ok": True})
        except Exception as e:
            summary.append({"file": name, "device": "116" if "Device116" in wav else "118",
                            "n_sent": 0, "n_spk": 0, "ok": False, "err": str(e)[:200]})
            print("  ERR", e)

    # 汇总
    with open(os.path.join(RESULT, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "device", "n_sent", "n_spk", "ok", "err"])
        w.writeheader(); w.writerows(summary)
    print("\n完成，结果在:", RESULT)

if __name__ == "__main__":
    main()
