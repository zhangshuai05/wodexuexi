# -*- coding: utf-8 -*-
"""嵌入可分性诊断：CAM++ 在同信道会议音频上到底能不能把人分开？

**要回答的问题**：`oracle_k` 显示「强制真值 K」后 DER 仍有 36.99% —— 错的不是选 K。
但还分不清是**嵌入不可分**，还是**聚类不会分**。本脚本用**真值标签**直接测嵌入本身，
绕开选 K、绕开聚类、绕开准入。

方法：
  1. 读真值 RTTM 的语音段，**丢掉与其他说话人重叠的段**（重叠段的窗是两个人的混合
     声纹，会污染类内统计，把可分性低报）
  2. 对每段取**完全落在段内**的 2.0s/1.0s 窗（避开边界混合窗），CAM++ 提嵌入
     （与生产同一模型、同一窗长、同一 L2 归一化）
  3. 按 `(场次, 说话人)` 分组，算类内 / 类间余弦距离分布
  4. **留一最近质心**：每个窗到"本人质心（排除自身）"与"最近他人质心"比大小

判读（预注册，跑之前写定）：
    类间中位 − 类内中位 > 0.30  且  留一准确率 > 90%   → 嵌入**可分**，问题在聚类
    间隔 0.10–0.30  或  准确率 75–90%                  → 边缘，嵌入与聚类都要动
    间隔 < 0.10     或  准确率 < 75%                   → 嵌入**不可分**，换嵌入是唯一出路

**自检（本脚本的判别性所在）**：4 场里有两场已知行为相反 ——
`R8007_M8011` 在真值 K 下能到 22.45%（说明嵌入**分得开**，错在选 K），
`R8008_M8013` 在真值 K 下仍是 79.18%（疑似嵌入**分不开**）。
若本诊断给这两场相同结论，说明指标设计有毛病，结论不可用。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/diagnose_embedding_separability.py --meetings R8007_M8011 R8008_M8013
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "AliMeeting", REPO / "streaming_asr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, RESULTS, der_tools, resolve_wav  # noqa: E402
from run_der import load_models  # noqa: E402

WIN_S, HOP_S = 2.0, 1.0
MAX_PER_SEGMENT = 2            # 每段最多取几个窗（见 windows_in 的说明：必须跨段）
MAX_PER_SPEAKER = 200          # 每人最多采样多少窗（两两距离是 O(n^2)，必须封顶）
OVERLAP_TOL = 0.05             # 判"与其他说话人重叠"的容差（秒）
DEFAULT_MEETINGS = ["R8009_M8020", "R8001_M8004", "R8007_M8011", "R8008_M8013"]

_, read_rttm = der_tools()


def clean_segments(segments):
    """只保留不与其他说话人重叠的段（保证窗内只有一个人）。"""
    out = []
    for i, seg in enumerate(segments):
        clash = False
        for j, other in enumerate(segments):
            if i == j or other["speaker"] == seg["speaker"]:
                continue
            if min(seg["end"], other["end"]) - max(seg["start"], other["start"]) > OVERLAP_TOL:
                clash = True
                break
        if not clash:
            out.append(seg)
    return out


def windows_in(start, end, win=WIN_S, hop=HOP_S, max_per_segment=MAX_PER_SEGMENT):
    """完全落在 [start, end] 内的窗起点，**每段最多取 max_per_segment 个**。

    为什么必须限采样：同一段内的相邻窗高度相关（时间相邻、声学条件几乎相同），
    全取会让类内距离被严重低估、把可分性**高报**（实测过：不限采样时留一准确率
    是 100%，而真实 DER 是 22–79%，两者矛盾）。类内样本必须来自不同段。
    """
    all_windows, t = [], start
    while t + win <= end + 1e-9:
        all_windows.append(round(t, 3))
        t += hop
    if len(all_windows) <= max_per_segment:
        return all_windows
    picks = np.linspace(0, len(all_windows) - 1, max_per_segment).round().astype(int)
    return [all_windows[i] for i in sorted(set(picks.tolist()))]


def embed_windows(model, audio, rate, times, win=WIN_S):
    vecs = []
    for t in times:
        chunk = audio[int(t * rate):int((t + win) * rate)]
        if len(chunk) < int(0.25 * rate):
            continue
        emb = model.generate(input=chunk)[0].get("spk_embedding")
        if emb is None:
            continue
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(vecs) if vecs else np.empty((0, 0), np.float32)


def loo_nearest_centroid(E, labels):
    """留一最近质心：到本人质心（排除自身）vs 到最近他人质心。

    返回 (准确率, margins)。margin = d_other - d_own，>0 判对。
    """
    uniq = sorted(set(labels.tolist()))
    sums = {u: E[labels == u].sum(axis=0) for u in uniq}
    cnts = {u: int((labels == u).sum()) for u in uniq}
    correct, margins = 0, []
    for i in range(len(E)):
        own = labels[i]
        d_own = d_other = None
        for u in uniq:
            n = cnts[u] - (1 if u == own else 0)
            if n <= 0:
                continue
            c = sums[u] - (E[i] if u == own else 0.0)
            c = c / n
            c = c / (np.linalg.norm(c) + 1e-9)
            d = 1.0 - float(E[i] @ c)
            if u == own:
                d_own = d
            else:
                d_other = d if d_other is None else min(d_other, d)
        if d_own is None or d_other is None:
            continue
        correct += int(d_own < d_other)
        margins.append(d_other - d_own)
    return correct / max(len(margins), 1), np.asarray(margins)


def distance_split(E, labels, max_pairs=400000, seed=20261014):
    """类内 / 类间余弦距离（随机采样对，避免 O(n^2) 爆炸）。"""
    rng = np.random.default_rng(seed)
    n = len(E)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        ii = rng.integers(0, n, max_pairs)
        jj = rng.integers(0, n, max_pairs)
        idx = [(i, j) for i, j in zip(ii, jj) if i != j]
    same, diff = [], []
    for i, j in idx:
        d = 1.0 - float(E[i] @ E[j])
        (same if labels[i] == labels[j] else diff).append(d)
    return np.asarray(same), np.asarray(diff)


def auc_between(same, diff, n=200000, seed=20261015):
    """AUC = P(类间距离 > 类内距离)，随机采样估计。"""
    if not len(same) or not len(diff):
        return None
    rng = np.random.default_rng(seed)
    a = same[rng.integers(0, len(same), n)]
    b = diff[rng.integers(0, len(diff), n)]
    return float((b > a).mean() + 0.5 * (b == a).mean())


def verdict(gap, acc):
    if gap > 0.30 and acc > 0.90:
        return "可分（问题在聚类）"
    if gap < 0.10 or acc < 0.75:
        return "不可分（换嵌入是唯一出路）"
    return "边缘（嵌入与聚类都要动）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    ap.add_argument("--out", default="embedding_separability.json")
    args = ap.parse_args()

    import soundfile as sf
    models = load_models()
    rows = []
    print("%-14s %4s %5s %7s %6s %9s %9s %8s %8s %s"
          % ("meeting", "真值", "人", "窗数", "段数", "类内中位", "类间中位",
             "间隔", "准确率", "判定"),
          flush=True)
    for meeting in args.meetings:
        t0 = time.time()
        wav = resolve_wav(meeting)
        audio, rate = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        segments = clean_segments(read_rttm(ALI_RTTM / f"{meeting}.rttm"))

        by_speaker, n_segments = {}, 0
        for seg in segments:
            windows = windows_in(seg["start"], seg["end"])
            if not windows:
                continue
            by_speaker.setdefault(seg["speaker"], []).extend(windows)
            n_segments += 1

        vecs, labels = [], []
        for spk, times in sorted(by_speaker.items()):
            if not times:
                continue
            times = times[:MAX_PER_SPEAKER]
            E = embed_windows(models["spk"], audio, rate, times)
            if not len(E):
                continue
            vecs.append(E)
            labels.extend([spk] * len(E))
        if not vecs:
            print("%-14s  FAILED 没有可用窗" % meeting, flush=True)
            continue
        E = np.vstack(vecs)
        labels = np.asarray(labels)

        same, diff = distance_split(E, labels)
        acc, margins = loo_nearest_centroid(E, labels)
        s_med = float(np.median(same)) if len(same) else float("nan")
        d_med = float(np.median(diff)) if len(diff) else float("nan")
        gap = d_med - s_med
        row = {"meeting": meeting, "true_k": len(by_speaker), "n_windows": int(len(E)),
               "n_segments": n_segments,
               "within_median": s_med, "between_median": d_med, "gap": gap,
               "loo_accuracy": acc, "auc": auc_between(same, diff),
               "margin_median": float(np.median(margins)) if len(margins) else None,
               "verdict": verdict(gap, acc), "elapsed_s": round(time.time() - t0, 1)}
        rows.append(row)
        print("%-14s %4d %5d %7d %6d %9.3f %9.3f %8.3f %7.1f%% %s"
              % (meeting, row["true_k"], len(by_speaker), row["n_windows"],
                 n_segments, s_med, d_med, gap, acc * 100, row["verdict"]), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "window": f"{WIN_S}s/{HOP_S}s", "max_per_speaker": MAX_PER_SPEAKER,
                   "method": "真值段内窗 + CAM++ 嵌入 + 留一最近质心；丢弃重叠段",
                   "thresholds": {"可分": "gap>0.30 且 acc>90%",
                                  "不可分": "gap<0.10 或 acc<75%"}},
        "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已写出 %s" % out)

    ok = [r for r in rows if "verdict" in r]
    if len(ok) >= 2:
        print("\n自检（两场已知行为相反，结论必须不同）：")
        for r in ok:
            if r["meeting"] in ("R8007_M8011", "R8008_M8013"):
                print("  %-14s 真值K上界已知 %s  -> 诊断: %s"
                      % (r["meeting"],
                         "22.45%（嵌入可分）" if r["meeting"] == "R8007_M8011"
                         else "79.18%（疑似不可分）", r["verdict"]))


if __name__ == "__main__":
    main()
