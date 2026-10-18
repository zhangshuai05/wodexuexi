# -*- coding: utf-8 -*-
"""全量双设备融合：11 对 (距离 x 环境 x 顾客) 各自 fusion，输出按距离/环境汇总。
funasr 环境。用法: python batch_fusion.py [--limit N]
"""
import os, sys, json, re, collections
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
OUT = os.path.join(BASE, "diar_result", "fusion_all")
os.makedirs(OUT, exist_ok=True)

VAD = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SEACO = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PUNC = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
SPK = "iic/speech_campplus_sv_zh-cn_16k-common"

def make_pairs():
    # 11 对已核定: (距离, 环境, 顾客) -> (Call文件=顾客麦116, Note文件=店员麦118)
    PAIRS = [
        ("1m",  "noisy", "zjl", "Call20260718210228.wav", "Note20260803111143.wav"),
        ("1m",  "noisy", "ash", "Call20260718210836.wav", "Note20260803111741.wav"),
        ("1m",  "quiet", "ash", "Call20260718211745.wav", "Note20260803112655.wav"),
        ("1.5m","noisy", "zjl", "Call20260718210454.wav", "Note20260803111436.wav"),
        ("1.5m","noisy", "ash", "Call20260718210948.wav", "Note20260803111855.wav"),
        ("1.5m","quiet", "zjl", "Call20260718211423.wav", "Note20260803112329.wav"),
        ("1.5m","quiet", "ash", "Call20260718211842.wav", "Note20260803112752.wav"),
        ("2m",  "noisy", "zjl", "Call20260718210657.wav", "Note20260803111605.wav"),
        ("2m",  "noisy", "ash", "Call20260718211031.wav", "Note20260803111943.wav"),
        ("2m",  "quiet", "zjl", "Call20260718211506.wav", "Note20260803112415.wav"),
        ("2m",  "quiet", "ash", "Call20260718211938.wav", "Note20260803112841.wav"),
    ]
    return [{"dist": d, "env": e, "cust": c, "call": ca, "note": no}
            for d, e, c, ca, no in PAIRS]

def run_diar(wav, model):
    res = model.generate(input=wav, batch_size=1, sentence_timestamp=True,
                         output_dir=OUT, merge_vad=True, merge_length_s=15)
    ss = res[0].get("sentence_info", [])
    return [{"start": s.get("start")/1000.0 if isinstance(s.get("start"),(int,float)) else 0.0,
             "end": s.get("end")/1000.0 if isinstance(s.get("end"),(int,float)) else 0.0,
             "text": s.get("text","")} for s in ss]

def main(limit=None):
    from funasr import AutoModel
    print("加载模型 ...", flush=True)
    m = AutoModel(model=SEACO, model_revision="master",
                  vad_model=VAD, vad_kwargs={"max_single_segment_time": 30000},
                  punc_model=PUNC,
                  spk_model=SPK, spk_model_revision="master",
                  spk_mode="punc_segment",
                  disable_update=True, disable_pbar=True)
    pairs = make_pairs()
    if limit:
        pairs = pairs[:limit]
    print(f"共 {len(pairs)} 对", flush=True)
    summary = []
    for i, pr in enumerate(pairs, 1):
        tag = f"[{pr['dist']}|{pr['env']}|{pr['cust']}]"
        print(f"--- {tag} ({i}/{len(pairs)}) call={pr['call']} note={pr['note']} ---", flush=True)
        ca = os.path.join(AUDIO, "Device116", pr["call"])
        no = os.path.join(AUDIO, "Device118", pr["note"])
        try:
            segsA = run_diar(ca, m)   # 顾客麦 -> 顾客
            segsB = run_diar(no, m)   # 店员麦 -> 店员
        except Exception as e:
            print("  ERR", e, flush=True)
            continue
        for s in segsA: s["speaker"] = "顾客"
        for s in segsB: s["speaker"] = "店员"
        merged = sorted(segsA + segsB, key=lambda x: x["start"])
        nA, nB = len(segsA), len(segsB)
        summary.append({"dist": pr["dist"], "env": pr["env"], "cust": pr["cust"],
                        "call": pr["call"], "note": pr["note"],
                        "n_book": nA, "n_staff": nB, "n_merged": len(merged),
                        "cust_text": " ".join(s["text"] for s in segsA)[:120],
                        "staff_text": " ".join(s["text"] for s in segsB)[:120]})
        # 存每对结果
        json.dump({"dist": pr["dist"], "env": pr["env"], "cust": pr["cust"],
                   "merged": merged}, open(
                    os.path.join(OUT, f"fusion_{pr['dist']}_{pr['env']}_{pr['cust']}.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n完成，summary ->", os.path.join(OUT, "summary.json"), flush=True)

if __name__ == "__main__":
    lim = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--limit":
        lim = int(sys.argv[2])
    main(lim)
