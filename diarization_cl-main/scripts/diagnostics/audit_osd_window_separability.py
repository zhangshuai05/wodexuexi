# -*- coding: utf-8 -*-
"""OSD 重建的窗到底纯不纯？用**真值标签**直接测可分性，把两种病因分开。

**背景**：用 OSD 单人区间重建窗之后，聚类**仍然塌陷到 K=1**（R8007_M8010，
537 窗、341 个"单人连续区间"）。这分不清两件事：

    (a) 窗还是脏的        -> 换聚类算法没救，得换嵌入或上端到端
    (b) 窗干净了、聚类不行 -> 才值得动聚类

本脚本用**真值**给每个窗打标签（窗中心落在哪个说话人的段内，跨人的窗丢弃），
直接算可分性 —— 绕开聚类、绕开选 K、绕开准入，只看嵌入本身。

两臂对照（同一场、同一模型、同一窗长，唯一差别是**窗在哪**）：

    grid    VAD 过滤后的固定网格窗（现状做法）
    osdwin  OSD"单人且连续"区间内的窗

判据（与最早那次嵌入诊断保持一致）：
    留一最近质心准确率 > 90%  → 可分，问题在聚类
    准确率 < 75%              → 不可分，声纹/嵌入是瓶颈

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_osd_window_separability.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr",
           REPO / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav  # noqa: E402
from run_der import load_models  # noqa: E402
from run_osd_rebuild_diar import (OSD_MODEL, WIN_S, HOP_S,  # noqa: E402
                                  osd_frames, single_runs, _norm)

_, read_rttm = der_tools()
MAX_PAIRS = 200000


def label_of_window(segments, a, b, tol=0.05):
    """窗中心落在谁的段内；若窗内混了两个说话人就返回 None（丢弃）。"""
    mid = (a + b) / 2.0
    owners = {s["speaker"] for s in segments if min(b, s["end"]) - max(a, s["start"]) > tol}
    if len(owners) != 1:
        return None
    for s in segments:
        if s["speaker"] in owners and s["start"] <= mid < s["end"]:
            return s["speaker"]
    return None


def embed(model, audio, rate, wins):
    vecs, labs = [], []
    for a, b in wins:
        chunk = audio[int(a * rate):int(b * rate)]
        if len(chunk) < int(0.35 * rate):
            continue
        try:
            e = model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if e is None:
            continue
        if hasattr(e, "detach"):
            e = e.detach().cpu().numpy()
        vecs.append(_norm(e))
        labs.append((a, b))
    return (np.stack(vecs) if vecs else np.empty((0, 0), np.float32)), labs


def loo_accuracy(E, labels):
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    if len(uniq) < 2:
        return 0.0, 0.0, 0.0
    sums = {u: E[labels == u].sum(axis=0) for u in uniq}
    cnts = {u: int((labels == u).sum()) for u in uniq}
    ok = tot = 0
    for i in range(len(E)):
        own = labels[i]
        d_own = d_other = None
        for u in uniq:
            n = cnts[u] - (1 if u == own else 0)
            if n <= 0:
                continue
            c = (sums[u] - (E[i] if u == own else 0.0)) / n
            c = c / (np.linalg.norm(c) + 1e-9)
            d = 1.0 - float(E[i] @ c)
            if u == own:
                d_own = d
            else:
                d_other = d if d_other is None else min(d_other, d)
        if d_own is None or d_other is None:
            continue
        ok += int(d_own < d_other)
        tot += 1
    return ok / max(tot, 1), 0.0, 0.0


def distance_split(E, labels, seed=20261014):
    rng = np.random.default_rng(seed)
    n = len(E)
    total = n * (n - 1) // 2
    if total <= MAX_PAIRS:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        ii = rng.integers(0, n, MAX_PAIRS)
        jj = rng.integers(0, n, MAX_PAIRS)
        pairs = [(i, j) for i, j in zip(ii, jj) if i != j]
    same, diff = [], []
    for i, j in pairs:
        d = 1.0 - float(E[i] @ E[j])
        (same if labels[i] == labels[j] else diff).append(d)
    return np.asarray(same), np.asarray(diff)


def main():
    import claude_a2_v11 as v11
    meetings = ["R8007_M8010", "R8008_M8013"]
    sess = ort.InferenceSession(str(OSD_MODEL), providers=["CPUExecutionProvider"])
    models = load_models()
    rows = []
    for meeting in meetings:
        wav = resolve_wav(meeting)
        audio, rate = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        segments = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        speech, overlap, valid, cls = osd_frames(sess, audio)
        runs = single_runs(speech, overlap, valid, cls)
        osd_wins = []
        for s, e in runs:
            t = s
            while t + WIN_S <= e + 1e-9:
                osd_wins.append((t, t + WIN_S))
                t += HOP_S
        vad = v11._vad_segs(models["vad"], str(wav))
        grid_all = [(t, t + WIN_S)
                    for t in np.arange(0, max(0.0, len(audio) / rate - WIN_S), HOP_S)]
        grid_wins = [(a, b) for a, b in grid_all
                     if sum(max(0.0, min(b, e2) - max(a, s2)) for s2, e2 in vad)
                     >= 0.5 * (b - a)]

        print("\n=== %s ===" % meeting, flush=True)
        for arm, wins in (("grid", grid_wins), ("osdwin", osd_wins)):
            E, kept = embed(models["spk"], audio, rate, wins)
            labs = [label_of_window(segments, a, b) for a, b in kept]
            idx = [i for i, L in enumerate(labs) if L is not None]
            if len(idx) < 4:
                print("  %-7s 可用窗 %d（真值可标 %d）—— 太少，跳过"
                      % (arm, len(E), len(idx)), flush=True)
                continue
            Es = E[idx]
            ls = [labs[i] for i in idx]
            same, diff = distance_split(Es, ls)
            acc, _, _ = loo_accuracy(Es, ls)
            s_med = float(np.median(same)) if len(same) else float("nan")
            d_med = float(np.median(diff)) if len(diff) else float("nan")
            verdict = ("可分（问题在聚类）" if acc > 0.90
                       else "不可分（声纹/嵌入是瓶颈）" if acc < 0.75 else "边缘")
            rows.append({"meeting": meeting, "arm": arm, "n_windows": int(len(E)),
                         "n_labelled": len(idx), "n_speakers": len(set(ls)),
                         "within_median": s_med, "between_median": d_med,
                         "gap": d_med - s_med, "loo_acc": acc, "verdict": verdict})
            print("  %-7s 窗 %5d（可标 %4d，%d 人）  类内 %.3f  类间 %.3f  间隔 %.3f  "
                  "留一准确率 %5.1f%%  -> %s"
                  % (arm, len(E), len(idx), len(set(ls)), s_med, d_med,
                     d_med - s_med, acc * 100, verdict), flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "osd_window_separability.json").write_text(
            __import__("json").dumps({"rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print("\n说明：判据与最早那次嵌入诊断一致（>90% 可分 / <75% 不可分）。")
    print("若 osdwin 与 grid 的留一准确率都低，说明换聚类算法救不了，要换嵌入或上端到端。")


if __name__ == "__main__":
    main()
