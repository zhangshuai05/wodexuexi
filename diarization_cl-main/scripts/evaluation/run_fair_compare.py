# -*- coding: utf-8 -*-
"""公平对比：c1 vs DSH 当前版，两边都吃 FRCRN 降噪音频。

模拟 web_server 生产流程（denoise=auto 嘈杂档必然降噪）：
  同一 wav_eff = frcrn 降噪音频，分别喂 DSH a2_ovl.run 和 c1 run_claude_lane。
输出到本地 diar_result/fair/，不碰 DSH output/。
评分：eval_der.eval_one，collar0.5+trim，GT 用 gt_ts_diag(11文件)。
"""
import os, sys, json
import numpy as np

DSH = r"D:\Deepseek_herness\2026-08\diarization"
sys.path.insert(0, DSH)
os.chdir(DSH)

from a2_ovl import run as dsh_run, SEACO_LOCAL  # noqa: E402
from a2_port import load_models as dsh_load  # noqa: E402
from claude_a2 import run_claude_lane, VERSIONS  # noqa: E402
from funasr import AutoModel  # noqa: E402
from eval_der import eval_one  # noqa: E402
from eval_gt import ENV  # noqa: E402

OUT = r"C:\Users\EDY\Desktop\new\diar_result\fair"
DSH_DEN = os.path.join(DSH, "output", "denoised")
MINE = r"C:\Users\EDY\Desktop\new\diar_result\fair_frcrn_audio"
DIAG = ["Call20260718210657", "Call20260718211031", "Call20260718211506",
        "Call20260718211745", "Call20260718211938", "Note20260803111605",
        "Note20260803111943", "Note20260803112220", "Note20260803112415",
        "Note20260803112655", "Note20260803112841"]
COLLAR, TRIM = 0.5, True


def wav_of(base):
    for p in (os.path.join(DSH_DEN, base + "_frcrn.wav"),
              os.path.join(MINE, base + "_frcrn.wav")):
        if os.path.exists(p):
            return p
    return None


def main():
    os.makedirs(os.path.join(OUT, "dsh"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "c1"), exist_ok=True)

    print("loading models (vad+spk+dsh_asr)...", flush=True)
    vad, spk = dsh_load()
    asr = AutoModel(model=SEACO_LOCAL, disable_update=True, disable_pbar=True)
    cm = {"vad": vad, "spk": spk, "asr": asr}

    for base in DIAG:
        w = wav_of(base)
        if w is None:
            print(f"[skip] {base}: no frcrn wav", flush=True)
            continue
        # DSH 当前版
        pd = os.path.join(OUT, "dsh", base + "_dsh.json")
        if not os.path.exists(pd):
            try:
                dsh_run(w, vad, spk, asr, outdir=os.path.join(OUT, "dsh"),
                        suffix="_dsh")
                print(f"[dsh-ok] {base}", flush=True)
            except Exception as e:
                print(f"[dsh-FAIL] {base}: {type(e).__name__}: {e}", flush=True)
        # c1
        pc = os.path.join(OUT, "c1", base + "_c1.json")
        if not os.path.exists(pc):
            try:
                r = run_claude_lane(w, cm, "c1")
                json.dump(r, open(pc, "w", encoding="utf-8"),
                          ensure_ascii=False)
                print(f"[c1-ok] {base}", flush=True)
            except Exception as e:
                print(f"[c1-FAIL] {base}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
