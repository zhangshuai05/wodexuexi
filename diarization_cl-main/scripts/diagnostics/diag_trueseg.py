# -*- coding: utf-8 -*-
"""诊断: 真塌缩(安静, 窗口能救) vs 假塌缩(嘈杂 11143, 窗口碎切) 段级聚类结构差异。
目标: 找到一个判据, 在窗口级之前区分这两种情况, 避免 11143 被误切。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, r"c:\Users\EDY\Desktop\new")
from a2_v2 import a2_segments_v2, VAD, SPK

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")

# (dir, file, 预期)  真塌缩=安静里段级塌但窗口能救; 假塌缩=嘈杂里段级就是碎的
CASES = [
    ("Device116", "Call20260718211423.wav", "真塌缩"),
    ("Device116", "Call20260718211506.wav", "真塌缩"),
    ("Device116", "Call20260718211745.wav", "真塌缩"),
    ("Device118", "Note20260803111143.wav", "假塌缩"),
    ("Device116", "Call20260718211842.wav", "正常"),
    ("Device118", "Note20260803112655.wav", "正常"),
    ("Device118", "Note20260803112841.wav", "正常"),
]


def cs(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    for sub, fname, tag in CASES:
        wav = os.path.join(AUDIO, sub, fname)
        segs = a2_segments_v2(wav, spk, vad, merge=False, cluster_level="segment")
        labels = [s["spk"] for s in segs]
        n = len(set(labels))
        # 段级切换数
        switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        # 主导标签占比
        import collections
        cnt = collections.Counter(labels)
        dom = max(cnt.values()) / len(labels)
        print(f"{fname:28s} [{tag}] nseg={n:3d} 段级spk={sorted(set(labels))} 切换={switches:2d} dom={dom:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
