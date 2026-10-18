# -*- coding: utf-8 -*-
"""分阶段成本实测（修订规格 §R6）——为正式 dev 12 场实验定预算。

分三段，**分开计时**：
  C1  模型首次加载（CAM++ / FSMN 各一次）
  C2  单场 VAD + 主窗 + 密窗特征提取
  C3  缓存就绪后，换一个不同 K 的细化与评分

**不是**"一场耗时 × 12"：模型在会议循环外加载（`probe_e2e_unlocked2.py` 第 62 行），
主窗嵌入在 K 之间共用。

        python -u -B scripts/evaluation/probe_ab_cost.py --meeting L_R003S01C02
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` 与本地目录。**不要插仓库根**（会遮住同名的
# `claude_a2.py`）。回归测试见 tests/test_evaluator_path_hygiene.py
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
SECONDS = 600.0


def _relabel(labels):
    """与 `v11._relabel_by_first_seen` 同义的本地实现（C3 只用于计时）。"""
    import claude_a2_v11 as v11
    return v11._relabel_by_first_seen(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="L_R003S01C02")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    marks = {}

    # ── C1 模型加载
    t0 = time.perf_counter()
    from web_server import get_models
    models = get_models(lambda m: None, need_stream=False, need_offline=False)
    marks["C1 模型加载"] = time.perf_counter() - t0
    print(f"C1 模型加载            : {marks['C1 模型加载']:7.1f}s")

    import soundfile as sf
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12

    wav = DATA / "wav" / f"{args.meeting}.flac"
    if not wav.exists():
        print(f"找不到 {wav}")
        return
    wav = str(wav)

    # ── C2 单场：VAD + 主窗 + 密窗特征提取
    from web_server import get_models as _gm  # noqa: F401  (已加载)
    t0 = time.perf_counter()
    info = sf.info(wav)
    stop = min(int(args.seconds * info.samplerate), info.frames)
    audio, rate = sf.read(wav, dtype="float32", stop=stop)
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = v11._vad_segs(models["vad"], wav)
    vad_segments = [(s, min(e, args.seconds)) for s, e in vad_segments if s < args.seconds]
    t_vad = time.perf_counter() - t0

    t0 = time.perf_counter()
    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E, times, _ = v11._extract_windows(audio, rate, vad_segments, models["spk"])
    t_main = time.perf_counter() - t0

    t0 = time.perf_counter()
    rE, rtimes = v12._refine_windows(audio, rate, vad_segments, models["spk"])
    t_dense = time.perf_counter() - t0

    marks["C2 VAD"] = t_vad
    marks["C2 主窗嵌入"] = t_main
    marks["C2 密窗嵌入"] = t_dense
    marks["C2 合计"] = t_vad + t_main + t_dense
    print(f"C2 VAD                 : {t_vad:7.1f}s  ({len(vad_segments)} 段)")
    print(f"C2 主窗嵌入            : {t_main:7.1f}s  ({len(E)} 窗)")
    print(f"C2 密窗嵌入            : {t_dense:7.1f}s  ({len(rE)} 窗)")
    print(f"C2 合计                : {marks['C2 合计']:7.1f}s")

    # ── C3 缓存就绪后，换一个不同 K 的细化与评分（主开销在细化，不在聚类）
    from scipy.cluster.hierarchy import fcluster, linkage

    t0 = time.perf_counter()
    labels_k3 = _relabel(fcluster(linkage(E, method="average", metric="cosine"),
                                  t=3, criterion="maxclust"))
    anchor_labels = sorted(set(int(x) for x in labels_k3))
    anchors = v11._centroids(E, labels_k3)
    rlabels = v12._refine_labels(rE, anchors, anchor_labels, rtimes)
    t_refine = time.perf_counter() - t0
    marks["C3 一个不同 K 的细化+评分"] = t_refine
    n_after = len(set(rlabels.tolist())) if rlabels is not None else 0
    print(f"C3 换 K 的细化+评分    : {t_refine:7.1f}s  (k=3, 细化后 {n_after} 簇)")

    print()
    print("─── 预算推算 ───")
    c1 = marks["C1 模型加载"]
    c2 = marks["C2 合计"]
    c3 = marks["C3 一个不同 K 的细化+评分"]
    n_meet = 12
    print(f"  固定成本（一次）      : C1 = {c1:.0f}s")
    print(f"  **每场主开销**        : C2 = {c2:.0f}s  其中密窗嵌入 {marks['C2 密窗嵌入']:.0f}s")
    print(f"      × {n_meet} 场        = {c2*n_meet/60:.1f} 分钟")
    print(f"  换一个 K 的增量       : C3 = {c3:.1f}s（纯标签传播，与 K 无关的部分已在 C2 付过）")
    print(f"")
    print(f"  关键事实：`_refine_windows(audio, rate, vad_segments, spk_model)` "
          f"**不依赖 K**，")
    print(f"  所以密窗嵌入每场只算一次，**A / B / 全部 11 个阈值共用**。")
    print(f"  换 K 只重跑 `_refine_labels`（标签传播），近零成本。")
    print(f"")
    print(f"  **实际预算**          : {(c1 + c2*n_meet + c3*n_meet*11)/60:.1f} 分钟"
          f"（最坏：每场每阈值都换 K）")
    print(f"  **主项是 C2 的 {c2*n_meet/60:.1f} 分钟**，阈值数量几乎不影响总成本。")


if __name__ == "__main__":
    main()
