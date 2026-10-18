# -*- coding: utf-8 -*-
"""口径对照实验：a2_ovl.run() 当前版，同一文件喂 raw vs frcrn 降噪。
输出到本地 diar_result/dsh_probe/，不碰 DSH output/。
"""
import os, sys, json

DSH = r"D:\Deepseek_herness\2026-08\diarization"
sys.path.insert(0, DSH)
os.chdir(DSH)

from a2_ovl import run  # noqa: E402
from a2_ovl import SEACO_LOCAL  # noqa: E402
from a2_port import load_models  # noqa: E402
from funasr import AutoModel  # noqa: E402

OUT = r"C:\Users\EDY\Desktop\new\diar_result\dsh_probe"
os.makedirs(OUT, exist_ok=True)

base = "Note20260803112841"
RAW = os.path.join(DSH, "Audio", "Device118", base + ".wav")
FRC = os.path.join(DSH, "output", "denoised", base + "_frcrn.wav")


def main():
    print("loading models...", flush=True)
    vad, spk = load_models()
    asr = AutoModel(model=SEACO_LOCAL, disable_update=True, disable_pbar=True)

    for tag, path in (("raw", RAW), ("frcrn", FRC)):
        print(f"=== run {tag}: {path} ===", flush=True)
        try:
            run(path, vad, spk, asr, outdir=OUT, suffix=f"_{tag}")
        except Exception as e:
            print(f"[FAILED] {tag}: {type(e).__name__}: {e}", flush=True)

    # 对比 new vs old
    old = json.load(open(os.path.join(DSH, "output", "a2ovl", base + "_a2ovl.json"),
                         encoding="utf-8"))["sentence_info"]
    for tag in ("raw", "frcrn"):
        p = os.path.join(OUT, base + f"_{tag}.json")
        if not os.path.exists(p):
            print(f"{tag}: no output", flush=True)
            continue
        new = json.load(open(p, encoding="utf-8"))["sentence_info"]
        if len(new) != len(old):
            print(f"{tag}: N old={len(old)} new={len(new)}", flush=True)
            continue
        nd = sum(1 for a, b in zip(old, new)
                 if a.get("spk") != b.get("spk")
                 or abs(a.get("start", -1) - b.get("start", -1)) > 0.01
                 or abs(a.get("end", -1) - b.get("end", -1)) > 0.01)
        print(f"{tag}: n={len(new)} spk/bound diff vs old = {nd}", flush=True)
    print("DONE_PROBE", flush=True)


if __name__ == "__main__":
    main()
