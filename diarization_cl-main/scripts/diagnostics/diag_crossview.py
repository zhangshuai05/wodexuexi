# -*- coding: utf-8 -*-
"""关键实验: 用 12655(分对)的标签当伪标签, 验证 11745(分错)的嵌入能否区分同一对说话人
思路: 12655/11745 是同一场对话的两个视角(店员麦/顾客麦).
  从 12655 窗口嵌入取 ash/wzy 两个簇 -> 各自质心; 再算 11745 每段嵌入离两个质心的距离,
  -> 若 11745 嵌入能分清远近, 则应稳定地靠近某一个质心(跟文本真值一致).
  输出: 11745 每段的 [离ash质心cos, 离wzy质心cos] 及据此的预测标签, 与文本真值对照.
"""
import os, sys, numpy as np
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")
sys.path.insert(0, BASE)
from a2_v2 import cs, _vad_segs, VAD, SPK
import soundfile as sf
from collections import Counter

def win_embs(wav):
    """返回窗口级 (嵌入, 起始时间, 结束时间)"""
    from funasr import AutoModel
    vad = AutoModel(model=VAD, model_revision="master", disable_update=True, disable_pbar=True)
    spk = AutoModel(model=SPK, model_revision="master", disable_update=True, disable_pbar=True)
    x, sr = sf.read(wav)
    if x.ndim > 1: x = x[:, 0]
    dur = len(x) / sr
    vsegs = _vad_segs(vad, wav)
    win_s, hop_s = 1.2, 0.6
    embs, starts, ends = [], [], []
    for t in np.arange(0.0, dur - win_s, hop_s):
        s0, e0 = t, t + win_s
        inter = sum(max(0.0, min(e0, b) - max(s0, a)) for a, b in vsegs)
        if inter < 0.5 * win_s: continue
        chunk = x[int(s0*sr):int(e0*sr)]
        if len(chunk) < 1600: continue
        try:
            e = spk.generate(input=chunk)[0].get("spk_embedding")
        except Exception: continue
        if e is not None:
            embs.append(e.detach().cpu().numpy().reshape(-1)); starts.append(s0); ends.append(e0)
    return np.stack(embs), starts, ends, sr, x

def main():
    # 1) 12655 (店员麦, 分对). 用它的 A2-v2 结果标签作伪标签 (ash/wzy)
    ref = json_read("diar_result/a2_v2/Note20260803112655_diar.json")
    print("12655 结果段:", [(round(s['start'],1),round(s['end'],1)) for s in ref['sentence_info']])
    # 需要知道哪个spk是ash哪个是wzy: 从真值匹配脚本输出得知 ash->1, wzy->0
    ref_segs = ref['sentence_info']
    E2, st2, en2, sr2, x2 = win_embs(os.path.join(AUDIO, "Device118", "Note20260803112655.wav"))
    # 对12655每窗嵌入, 用它的段标签(时间覆盖)给伪标签
    labels = []
    for st, en in zip(st2, en2):
        lab = None
        for s in ref_segs:
            if st >= s['start'] - 0.3 and en <= s['end'] + 0.3:
                lab = s['spk']; break
        labels.append(lab)
    # 用真值判断 spk1=ash?  真值: ash 是顾客(短答), wzy 是店员(主导). 12655 ash->1, wzy->0.
    # 质心
    lab_arr = np.array([l if l is not None else -1 for l in labels])
    id1 = np.where(lab_arr == 1)[0]; id0 = np.where(lab_arr == 0)[0]
    c1 = E2[id1].mean(axis=0) if len(id1) else None   # spk1 (ash)
    c0 = E2[id0].mean(axis=0) if len(id0) else None   # spk0 (wzy)
    print(f"12655 窗口: spk1(ash)={len(id1)}窗 spk0(wzy)={len(id0)}窗")
    # 2) 11745 (顾客麦, 分错). 每段嵌入离两质心的距离
    import json as _json
    d = _json.load(open("diar_result/a2_v2/Call20260718211745_diar.json", encoding='utf-8'))
    print("\n11745 段 -> 离 ash/wzy 质心cos (cos越大越像该人):")
    for s in d['sentence_info']:
        # 直接对该段ASR文本; 用段代表嵌入
        pass
    # 重新算11745段嵌入
    E1, st1, en1, sr1, x1 = win_embs(os.path.join(AUDIO, "Device116", "Call20260718211745.wav"))
    # 与12655段对齐: 用窗口交叉验证复杂, 这里直接用12655两质心打分11745窗口
    best1, best0 = [], []
    for e in E1:
        b1 = cs(e, c1) if c1 is not None else 0
        b0 = cs(e, c0) if c0 is not None else 0
        best1.append(b1); best0.append(b0)
    best1 = np.array(best1); best0 = np.array(best0)
    # 每个11745窗: 预测标签 = 更接近的质心
    for st, en, b1, b0 in zip(st1, en1, best1, best0):
        pred = 'ash' if b1 > b0 else 'wzy'
        print(f"  [{st:5.2f}-{en:5.2f}] cos_ash={b1:.3f} cos_wzy={b0:.3f}  pred={pred}")

def json_read(p):
    import json
    return json.load(open(p, encoding='utf-8'))

if __name__ == "__main__":
    main()
