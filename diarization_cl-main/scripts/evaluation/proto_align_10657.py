# -*- coding: utf-8 -*-
"""GT 修复原型（本地，不写 DSH output/）：对嘈杂文件 10657，用 build_gt_auto
的 align_turns 原生逻辑跑一遍，先看它把 GT 标成什么样，再决定怎么改。
复现 build_gt_auto.align_turns + speech_segments，输入用现有 SEACO 缓存。
"""
import os, re, json, sys
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


def align_turns(toks, ts, turns, audio_len, energy):
    asr_text = "".join(toks)
    spans_gt, gt_stream = [], ""
    for t in turns:
        n = norm(t["text"])
        spans_gt.append((len(gt_stream), len(gt_stream) + len(n)))
        gt_stream += n
    sm = SequenceMatcher(None, gt_stream, asr_text, autojunk=False)
    owner = {}
    matched_gt = set()
    for a, b, n in sm.get_matching_blocks():
        if n == 0:
            continue
        for k in range(n):
            gi = a + k
            for ti, (s0, s1) in enumerate(spans_gt):
                if s0 <= gi < s1:
                    owner[b + k] = ti
                    matched_gt.add(gi)
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
            gt_len = spans_gt[ti][1] - spans_gt[ti][0]
            m = sum(1 for g in range(spans_gt[ti][0], spans_gt[ti][1])
                    if g in matched_gt)
            rows.append([start, end, m / max(1, gt_len), "asr"])
        else:
            rows.append([None, None, 0.0, None])
    char_gaps = []
    for j in range(1, len(toks)):
        dt = (ts[j][0] - ts[j - 1][1]) / 1000.0
        if dt > GAP_T:
            char_gaps.append((ts[j - 1][1] / 1000.0, ts[j][0] / 1000.0))
    if toks:
        char_gaps.append((0.0, ts[0][0] / 1000.0))
        char_gaps.append((ts[-1][1] / 1000.0, audio_len))
    for ti, r in enumerate(rows):
        if r[3] is not None:
            continue
        prev_end, kp = 0.0, None
        for k in range(ti - 1, -1, -1):
            if rows[k][3]:
                prev_end, kp = rows[k][1], k; break
        next_start, kn = audio_len, None
        for k in range(ti + 1, len(turns)):
            if rows[k][3]:
                next_start, kn = rows[k][0], k; break
        pstart = rows[kp][0] if kp is not None else 0.0
        nend = rows[kn][1] if kn is not None else audio_len
        cands = [g for g in char_gaps if g[0] >= pstart - 1.0
                 and g[1] <= nend + 1.0]
        wmid = (prev_end + next_start) / 2
        cands.sort(key=lambda g: (g[1] - g[0] < 0.8,
                                  abs((g[0] + g[1]) / 2 - wmid)))
        pick = cands[0] if cands else (prev_end, next_start)
        es = [s for s in energy if s[0] < pick[1] and s[1] > pick[0]]
        if es:
            best = max(es, key=lambda s: min(s[1], pick[1]) - max(s[0], pick[0]))
            inter = (max(best[0], pick[0]), min(best[1], pick[1]))
            if inter[1] - inter[0] >= 0.3:
                pick = inter
        start, end = pick
        if kn is not None and end > next_start + 0.05:
            end = next_start
        if end <= start:
            start, end = max(0, wmid - 0.3), wmid + 0.3
        rows[ti] = [round(start, 2), round(end, 2), 0.0, "gap"]
    if rows[0][3]:
        es = [s for s in energy if s[1] > rows[0][0] - 0.5 and s[0] < rows[0][1]]
        if es:
            cand = min(es, key=lambda s: abs(s[0] - rows[0][0]))
            if abs(cand[0] - rows[0][0]) > 1.0:
                rows[0][0] = round(max(0.0, cand[0] - 0.1), 2)
    if rows[-1][3]:
        es = [s for s in energy if s[0] < rows[-1][1] + 0.5 and s[1] > rows[-1][0]]
        if es:
            cand = max(es, key=lambda s: s[1])
            if abs(cand[1] - rows[-1][1]) > 1.0:
                rows[-1][1] = round(min(audio_len, cand[1] + 0.15), 2)
    for r in rows:
        if r[3] and r[1] <= r[0]:
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
    rows = align_turns(toks, ts, turns, audio_len, energy)
    print(f"== {base} 原生 align_turns 输出 (audio_len={audio_len:.1f}) ==")
    for i, (s, e, ratio, q) in enumerate(rows):
        print(f"  {i:>2} [{s:>6} .. {e:>6}] r={ratio:.2f} q={q:<4} "
              f"{turns[i]['spk']:<4} {turns[i]['text'][:14]}")
    out = {"rows": rows, "turns": turns}
    open(r"C:\Users\EDY\Desktop\new\diar_result\proto_10657.json", "w",
         encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
