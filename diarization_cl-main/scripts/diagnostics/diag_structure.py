# -*- coding: utf-8 -*-
"""验证核心判据: 窗口级聚类的 gap2 是否能干净区分 真塌缩(该用窗口) / 假塌缩(该用段级) / 正常。
对每个文件跑两种 cluster_level, 记录 window-gap2 / seg-gap2, 并与最优模式对齐。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import a2_segments_v2, VAD, SPK

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")

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
QUIET = set('''Call20260718211423.wav Call20260718211506.wav Call20260718211745.wav
Call20260718211842.wav Call20260718211938.wav Note20260803112329.wav
Note20260803112415.wav Note20260803112655.wav Note20260803112752.wav
Note20260803112841.wav'''.split())


def main():
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    out = []
    for sub, fname in FILES:
        wav = os.path.join(AUDIO, sub, fname)
        rec = {"file": fname, "env": "安静" if fname in QUIET else "嘈杂"}
        for lvl in ("window", "segment"):
            try:
                segs = a2_segments_v2(wav, spk, vad, merge=False, cluster_level=lvl)
                nspk = len(set(s["spk"] for s in segs))
                rec[lvl + "_nseg"] = len(segs)
                rec[lvl + "_nspk"] = nspk
            except Exception as e:
                rec[lvl + "_nseg"] = -1
        print("%-28s %4s win_nseg=%-3d win_nspk=%d seg_nseg=%-3d" % (
            fname, rec["env"], rec.get("window_nseg", -1), rec.get("window_nspk", -1),
            rec.get("segment_nseg", -1)), flush=True)
        out.append(rec)
    json.dump(out, open(os.path.join(BASE, "artifacts", "diag_structure.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
