# -*- coding: utf-8 -*-
"""细化层分级拆解：L1 判定 / L2 稳定化 / L3 碎段吸收，哪一级把准确率吃掉的。

**背景**：困境六第 2 步的两次反转 ——
  · confine 只挪 0.3 点，但那是**盲区**：它过滤的是"跨 class 区间的窗"，
    而 class 区间 ≠ 真值边界（powerset 纯度 77–83%），真正的跨人窗没被剔掉。
  · 质心对照显示 R8008 上两组质心完全等价（95.1% vs 95.1%），
    而且**0.6s 细化窗的最近质心判定本身有 95.1% 准**。
⇒ 判定准、句级却从 50.5% 掉到 40.3%，问题只能在**判定之后**。

`_refine_labels` 的后半段是两个从 v7 继承的函数，作用都是"把短段并进邻居"：

    assigned = argmax(短窗 @ 质心)                    # ← L1，已知 ~95% 准
    stable   = _stabilize_labels(assigned, margins)   # ← L2
    final    = _absorb_slivers(stable, ...)           # ← L3（现状）

一个跨人窗判错，会被这两个函数**传播到相邻的正确窗**，把边界推错。
而它们的参数（SLIVER_MIN_DUR / SLIVER_MARGIN）是为"2.0s 主窗"调的。

判据（事前写死）
    哪一级的句级 acc 掉得最多，元凶就是它：
      L1 就低        ⇒ 判定本身不行（与上一个诊断的 95.1% 冲突，需复查）
      L2 掉          ⇒ _stabilize_labels 的 margin 阈值在 1.5s 主窗上失配
      L3 掉          ⇒ _absorb_slivers 的碎段吸收在推错边界

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_refine_levels.py --meetings R8008_M8013 R8007_M8011
"""

import argparse
import json
import sys
import time
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

from common import ALI_RTTM, ALI_TG, RESULTS, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                        sentences_from_textgrid)
from run_der import load_models  # noqa: E402
from run_3dspeaker_pipeline import (class_chunks, chunks_to_turns,  # noqa: E402
                                    embed_chunks, osd_probs, spectral_cluster)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
import claude_a2_v7_v8 as v7  # noqa: E402

_, read_rttm = der_tools()
OSD_MODEL = Path(r"D:\yinp\pyannote_seg3\model.onnx")


def truth_at_windows(ref, times, win_s, tol=0.05):
    """每个细化窗的真值说话人（窗内只属一人时才可标，否则 None）。"""
    out = []
    for t in times:
        a, b = float(t), float(t) + win_s
        owners = {s["speaker"] for s in ref if min(b, s["end"]) - max(a, s["start"]) > tol}
        out.append(next(iter(owners)) if len(owners) == 1 else None)
    return out


def window_acc(labels, truth, mapping):
    """可标窗上的窗级准确率。"""
    idx = [i for i, t in enumerate(truth) if t is not None]
    if not idx:
        return 0.0, 0
    ok = sum(1 for i in idx if mapping.get(int(labels[i])) == truth[i])
    return ok / len(idx), len(idx)


def run_meeting(meeting, models, osd_sess):
    wav = resolve_wav(meeting)
    audio, rate = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    ref = read_rttm(ALI_RTTM / f"{meeting}.rttm")
    turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
    vad = v11._vad_segs(models["vad"], str(wav))

    # 主窗（新链路：class 区间 + 谱聚类）—— 身份与质心的来源
    probs_all, mask_all, fs = osd_probs(osd_sess, audio)
    probs = probs_all[mask_all]
    wins, _ = class_chunks(probs, fs)
    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    labels_main, k_main, _ = spectral_cluster(E, None)
    anchors = v11._centroids(E, labels_main)
    alabs = sorted(set(int(x) for x in labels_main.tolist()))

    # 细化窗与三级标签（逐级复用生产函数，不重写）
    rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
    scores = rE @ anchors.T
    assigned = np.asarray([alabs[i] for i in np.argmax(scores, axis=1)])
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    stable = np.asarray(v7._stabilize_labels(assigned, margins))
    absorbed = np.asarray(v7._absorb_slivers(
        stable, margins, rtimes, v12.REFINE_WIN_S,
        min_dur=v12.SLIVER_MIN_DUR, weak_margin=v12.SLIVER_MARGIN)[0])

    rwins = [(float(t), float(t) + v12.REFINE_WIN_S) for t in rtimes]
    truth = truth_at_windows(ref, rtimes, v12.REFINE_WIN_S)

    row = {"meeting": meeting, "k_main": int(k_main), "n_refine_windows": int(len(rtimes)),
           "n_main_chunks": len(E),
           "churn_L1_L2": int((assigned != stable).sum()),
           "churn_L2_L3": int((stable != absorbed).sum()),
           "churn_L1_L3": int((assigned != absorbed).sum()),
           "levels": {}}

    # 基准：不细化（主窗直接句级投票）
    base = chunks_to_turns(kept, labels_main, turns)
    row["levels"]["no_refine"] = {"sent_acc": base["acc"], "win_acc": None}

    for name, lab in (("L1_assign", assigned), ("L2_stabilize", stable),
                      ("L3_absorb", absorbed)):
        sent = chunks_to_turns(rwins, lab, turns)
        # 窗级准确率：用该级标签自己的匈牙利映射（与句级映射独立，只看标签质量）
        m = build_mapping(lab, truth)
        wacc, n_lab = window_acc(lab, truth, m)
        row["levels"][name] = {"sent_acc": sent["acc"], "win_acc": wacc,
                               "n_labelled_windows": n_lab,
                               "coverage": sent["coverage"]}
    return row


def build_mapping(labels, truth):
    """标签 -> 真值说话人（在可标窗上按共现时长做匈牙利双射）。"""
    from scipy.optimize import linear_sum_assignment
    idx = [i for i, t in enumerate(truth) if t is not None]
    if not idx:
        return {}
    cats = sorted({int(labels[i]) for i in idx})
    spks = sorted({truth[i] for i in idx})
    cost = np.zeros((len(cats), len(spks)))
    for a, c in enumerate(cats):
        sel = [i for i in idx if int(labels[i]) == c]
        for b, s in enumerate(spks):
            cost[a, b] = sum(1 for i in sel if truth[i] == s)
    rows, cols = linear_sum_assignment(-cost)
    return {cats[a]: spks[b] for a, b in zip(rows, cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=["R8008_M8013", "R8007_M8011"])
    ap.add_argument("--out", default="refine_levels.json")
    args = ap.parse_args()

    models = load_models()
    osd_sess = ort.InferenceSession(str(OSD_MODEL), providers=["CPUExecutionProvider"])
    rows = []
    for meeting in args.meetings:
        t0 = time.time()
        row = run_meeting(meeting, models, osd_sess)
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print("\n=== %s（主窗 K=%d，%d 窗；细化窗 %d）==="
              % (meeting, row["k_main"], row["n_main_chunks"], row["n_refine_windows"]),
              flush=True)
        print("  标签改动：L1→L2 %d 窗 | L2→L3 %d 窗 | L1→L3 %d 窗"
              % (row["churn_L1_L2"], row["churn_L2_L3"], row["churn_L1_L3"]), flush=True)
        print("  %-16s %9s %9s %9s" % ("级", "句级acc", "窗级acc", "覆盖"), flush=True)
        for name in ("no_refine", "L1_assign", "L2_stabilize", "L3_absorb"):
            lv = row["levels"][name]
            print("  %-16s %8.1f%% %9s %9s"
                  % (name, lv["sent_acc"] * 100,
                     ("%8.1f%%" % (lv["win_acc"] * 100)) if lv["win_acc"] is not None else "    -   ",
                     ("%8.1f%%" % (lv["coverage"] * 100)) if lv.get("coverage") is not None else "    -   "),
                  flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / args.out).write_text(
            json.dumps({"config": {"note": "L1=最近质心判定；L2=+_stabilize_labels；"
                                           "L3=+_absorb_slivers（现状）。"
                                           "窗级acc只统计窗内单人的可标窗"},
                        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n判据：哪一级句级 acc 掉得最多，元凶就是它")
    print("已写出 %s" % (RESULTS / args.out))


if __name__ == "__main__":
    main()
