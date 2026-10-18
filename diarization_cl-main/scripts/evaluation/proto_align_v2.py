# -*- coding: utf-8 -*-
"""GT 对齐修复原型 v2：引入"时间单调 + 间隙唯一分配"约束。
针对 build_gt_auto.align_turns 的 gap 兜底灾难：
  多 gap 轮挤同一个候选间隙 + 尾部整体错位。
改进：未匹配 gap 轮按 GT 顺序 + 相邻 asr 锚轮边界，单调分配不重叠。
"""
import os, re, json
import numpy as np
import soundfile as sf
from difflib import SequenceMatcher

DSH = r"D:\Deepseek_herness\2026-08\diarization"
PUNC = re.compile(r"[\s，。？！、；：,.?!;:'\"“”‘’（）()\[\]【】\-_…～~·]+")
GAP_T, CLUST_T = 0.5, 0.12


def norm(s):
    return PUNC.sub("", s or "")


def speech_segments(wav):
    x, sr = sf.read(wav)
    x = x[:, 0] if x.ndim > 1 else x
    hop = int(0.05 * sr)
    env = np.array([float(np.sqrt((x[i:i + hop] ** 2).mean()))
                    for i in range(0, len(x) - hop, hop)])
    env = np.convolve(env, np.ones(6) / 6, mode="same")
    noise = float(np.median(env[env < np.quantile(env, 0.25)]))
    thr = max(3 * noise, 0.012)
    sp = env > thr
    runs, i = [], 0
    while i < len(sp):
        if sp[i]:
            j = i
            while j < len(sp) and sp[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    runs = [(a, b) for a, b in runs if (b - a) * 0.05 >= 0.25]
    runs_s = [(a * 0.05, b * 0.05) for a, b in runs]
    merged = []
    for s, e in runs_s:
        if merged and s - merged[-1][1] < 0.4:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return len(x) / sr, merged


def align_v2(toks, ts, turns, audio_len, energy):
    """改进版：asr 锚轮精确钉死；gap 轮按 GT 顺序在相邻锚轮之间单调分配。"""
    asr_text = "".join(toks)
    spans_gt, gt_stream = [], ""
    for t in turns:
        n = norm(t["text"])
        spans_gt.append((len(gt_stream), len(gt_stream) + len(n)))
        gt_stream += n
    sm = SequenceMatcher(None, gt_stream, asr_text, autojunk=False)
    owner = {}
    for a, b, n in sm.get_matching_blocks():
        if n == 0:
            continue
        for k in range(n):
            gi = a + k
            for ti, (s0, s1) in enumerate(spans_gt):
                if s0 <= gi < s1:
                    owner[b + k] = ti
                    break
    owned = [[] for _ in turns]
    for j in range(len(toks)):
        if j in owner:
            owned[owner[j]].append(j)
    for j in range(len(toks)):
        if j in owner:
            continue
        left = right = None
        for jj in range(j - 1, -1, -1):
            if jj in owner:
                left = (jj, owner[jj]); break
        for jj in range(j + 1, len(toks)):
            if jj in owner:
                right = (jj, owner[jj]); break
        mid = (ts[j][0] + ts[j][1]) / 2
        if left is None and right is None:
            continue
        if left is None:
            ti = 0
        elif right is None:
            if (ts[j][0] - ts[left[0]][1]) / 1000.0 < CLUST_T \
                    and owned[left[1]] and owned[left[1]][-1] == left[0]:
                ti = left[1]
            elif left[1] + 1 < len(turns):
                ti = left[1] + 1
            else:
                ti = left[1]
        else:
            a_t = (ts[left[0]][0] + ts[left[0]][1]) / 2
            b_t = (ts[right[0]][0] + ts[right[0]][1]) / 2
            if right[1] == left[1]:
                ti = left[1]
            elif right[1] == left[1] + 1:
                ti = left[1] if abs(mid - a_t) <= abs(mid - b_t) + 0.05 \
                    else right[1]
            else:
                ti = left[1] if abs(mid - a_t) < 0.5 else right[1]
        owned[ti].append(j)
    for lst in owned:
        lst.sort()
    matched_idx = set(owner.keys())
    for ti, lst in enumerate(owned):
        if len(lst) < 2:
            continue
        clusters = []
        cur = [lst[0]]
        for a, b in zip(lst, lst[1:]):
            if (ts[b][0] - ts[a][1]) / 1000.0 <= 0.8:
                cur.append(b)
            else:
                clusters.append(cur); cur = [b]
        clusters.append(cur)
        anchored = [c for c in clusters if any(idx in matched_idx for idx in c)]
        if not anchored:
            continue
        keep = []
        for c in clusters:
            if any(idx in matched_idx for idx in c):
                keep.extend(c)
            else:
                nearest = min(
                    (min(abs((ts[x][1] - ts[y][0]) / 1000.0) for x in c
                         for y in anc) for anc in anchored), default=99.0)
                if nearest <= 1.0:
                    keep.extend(c)
        owned[ti] = sorted(keep)
    rows = []
    for ti, lst in enumerate(owned):
        if lst:
            start = ts[lst[0]][0] / 1000.0
            end = ts[lst[-1]][1] / 1000.0
            rows.append([start, end, "asr"])
        else:
            rows.append([None, None, None])
    # ---- 改进：# 用"锚轮边界 + 能量段"给 gap 轮单调分配，不重叠 ----
    # 锚序 = 有 asr 锚的轮
    anchor_pos = [i for i, r in enumerate(rows) if r[2] == "asr"]
    # 每个 gap 轮在前一锚(end)与后一锚(start)之间，能量段内选
    # 单调：同一锚间 gap 轮按顺序均分剩余区间
    for g0 in range(len(rows)):
        if rows[g0][2] is not None:
            continue
        # 向前找锚
        kp = max([i for i in anchor_pos if i < g0], default=None)
        kn = min([i for i in anchor_pos if i > g0], default=None)
        lo = rows[kp][1] if kp is not None else 0.0
        hi = rows[kn][0] if kn is not None else audio_len
        # 同锚区间内 gap 轮按 GT 顺序单调分配 [lo,hi]，严格递增
        group = [i for i in range(len(rows)) if rows[i][2] is None
                 and (kp is None or i > kp) and (kn is None or i < kn)]
        pos_in_group = group.index(g0)
        span = max(0.0, hi - lo)
        # 每个 gap 轮大致占 span/len(group)，但边界单调：前一 gap 的 e ≤ 后一 gap 的 s
        # 用累计比例确定，确保 start < end 且与相邻不重叠
        n_g = len(group)
        cum_lo = lo + span * (pos_in_group / n_g)
        cum_hi = lo + span * ((pos_in_group + 1) / n_g)
        # 前一 gap 轮的 end（同 group 内）
        prev_end = None
        if pos_in_group > 0:
            prev_end = rows[group[pos_in_group - 1]][1]
        s = max(cum_lo, prev_end) if prev_end else cum_lo
        e = cum_hi if cum_hi > s else s + 0.15
        # 用能量段收紧：找覆盖 s 起点附近的能量段，取其前边界
        es = [x for x in energy if x[0] <= s + 0.1 and x[1] > s]
        if es:
            best = min(es, key=lambda x: abs(x[0] - s))
            s2 = max(lo, best[0])
            s = max(s, min(s2, e))  # 不越过 e
        rows[g0] = [round(s, 2), round(e, 2), "gap"]
    for r in rows:
        if r[2] == "asr":
            pass
        if r[1] <= r[0]:
            r[1] = r[0] + 0.15
    return rows


def main():
    base = "Call20260718210657"
    c = json.load(open(os.path.join(DSH, "output", f"_chars_{base}.json"),
                       encoding="utf-8"))
    toks = [t for t in c["toks"] if t]
    ts = c["ts"][:len(toks)]
    gt = json.load(open(os.path.join(DSH, "output", "ground_truth.json"),
                        encoding="utf-8"))
    g = [x for x in gt if x["file"] == base + ".wav"][0]
    turns = g["turns"]
    wav = os.path.join(DSH, "Audio", "Device116", base + ".wav")
    audio_len, energy = speech_segments(wav)
    rows = align_v2(toks, ts, turns, audio_len, energy)
    print(f"== {base} align_v2 (audio_len={audio_len:.1f}) ==")
    for i, (s, e, q) in enumerate(rows):
        print(f"  {i:>2} [{s:>6} .. {e:>6}] {q:<4} {turns[i]['spk']:<4} "
              f"{turns[i]['text'][:14]}")
    out = {"rows": rows, "turns": [{"spk": t["spk"], "text": t["text"]} for t in turns]}
    open(r"C:\Users\EDY\Desktop\new\diar_result\proto_10657_v2.json", "w",
         encoding="utf-8").write(json.dumps(out, ensure_ascii=False,
                                            indent=2))


if __name__ == "__main__":
    main()
