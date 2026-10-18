# -*- coding: utf-8 -*-
"""会后说话人回填 —— 复用 AliMeeting 新链路，不自研聚类。

链路（`run_3dspeaker_pipeline.py`，HANDOFF_1015 验证过的那条）：
    VAD → class 区间切分 → CAM++ 嵌入 → 谱聚类（k-NN 图 + eigengap 定 K）
        → 0.6s 密集窗细化（长窗定身份、短窗找边界）
        → 无短窗覆盖的句子退回长窗投票（`fallback=`）

本文件原先那套自研 `OverlapAwareDiarizer`（0.9s/0.45s 滑窗 + 谱聚类）已删除：
它在高重叠场塌陷（R8007_M8010 窗级/句级全塌到 K=1），而上面的链路 8 场无一亏损。
删除时无调用方 —— `.zhang/streaming_asr_engine.py` 已不 import 本模块。

唯一入口：`backfill_speakers(wav_path, utterances, models=None, mode="auto",
                           normalize_peak=0.85, with_segments=False)`

2026-10-18 解冻后新增：
  - 弱音提拉默认开启（只提拉不衰减），解决远场低电平导致 VAD/细化窗覆盖率崩盘；
  - `with_segments=True` 时返回句内 `segments` / `sub_turns`，保留细化层找到的
    句内换人；默认返回仍是扁平 {utt_id: "Speaker_N"}，Meeting_main 契约不变。
"""
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# run_3dspeaker_pipeline / claude_a2_v1x 的导入前提（与该脚本自身的 bootstrap 同源）
for _p in (_ROOT / "AliMeeting", _ROOT / "streaming_asr", _ROOT / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# OSD 只用来切 class 区间。缺它时 mode="auto" 退化成纯 VAD 窗。
# 这是本机资产（约 1.2GB 的 pyannote 导出），换机器用环境变量指过去，别改代码。
OSD_ONNX = Path(os.environ.get("DIAR_OSD_ONNX",
                               r"D:\yinp\pyannote_seg3\model.onnx"))

# 弱音提拉目标（只提拉、不衰减）：VAD 是绝对能量门限，peak≈0.26 的远场录音
# 语音覆盖会从 100% 掉到 24%，细化窗随之只剩 51.8%（R8008_M8013，2026-10-17 实测）。
# 0 = 关闭。阈值参考 HANDOFF_1015 §8 的 --normalize-peak 0.85 实验。
DEFAULT_NORMALIZE_PEAK = 0.85

# 句内切分：主标签之外的连续片段至少这么长，才认定为"稳定的第二说话人"。
MIN_SUBTURN_S = 1.0
# segments 里可见片段的最短长度（更短的抖动不输出）。
MIN_SEGMENT_S = 0.3


def _normalize_audio(audio: np.ndarray, rate: int, target_peak: float):
    """弱音提拉到 target_peak；已足够响的录音原样返回（不衰减）。

    返回 (audio, temp_wav_path_or_None)。临时 WAV 只给 VAD 用（它按文件路径
    读），调用方负责删除；峰归一化对 CAM++（L2 归一化 + CMN）近似不变，增益
    主要作用在 VAD 的绝对能量门限上。
    """
    if not target_peak or target_peak <= 0 or len(audio) == 0:
        return audio, None
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-9 or peak >= target_peak:
        return audio, None
    scaled = (audio * (target_peak / peak)).astype(np.float32, copy=False)
    fd, path = tempfile.mkstemp(prefix="diar_norm_", suffix=".wav")
    os.close(fd)
    sf.write(path, scaled, rate, subtype="PCM_16")
    return scaled, path


def _refined_turn_segments(turns, windows, labels, main_labels, hop_s,
                           min_subturn_s, min_segment_s):
    """细化短窗 -> 句内时间片。

    把与句子有重叠的短窗按时间排序，连续同标签（窗起点间隔 <= 1.5*hop）
    合并成 run；run 边界取相邻窗起点中点 + hop/2（与 v12._segment_windows
    同口径），再裁到句子边界。短于 min_segment_s 的片段丢弃。

    返回 (segments, sub_turns)：
      segments  每句的完整时间片（含主说话人），供下游切 Turn；
      sub_turns 只含非主说话人且 >= min_subturn_s 的稳定片段。
    """
    segments, sub_turns = {}, {}
    label_list = labels.tolist()
    for turn, main in zip(turns, main_labels):
        main = int(main)
        if main == -1:                       # 整场无窗哨兵，没有可切分的短窗
            continue
        picked = []
        for (a, b), lab in zip(windows, label_list):
            s = max(float(a), turn["start"])
            e = min(float(b), turn["end"])
            if e - s <= 1e-6:
                continue
            picked.append((float(a), float(b), int(lab)))
        if not picked:
            continue
        picked.sort(key=lambda item: item[0])

        runs = []
        for idx, (a, b, lab) in enumerate(picked):
            if (runs and runs[-1]["label"] == lab
                    and a - picked[idx - 1][0] <= hop_s * 1.5 + 1e-9):
                runs[-1]["last_start"] = a
                runs[-1]["end"] = max(runs[-1]["end"], b)
            else:
                runs.append({"label": lab, "first_start": a,
                             "last_start": a, "end": b})

        pieces = []
        for idx, run in enumerate(runs):
            start = max(turn["start"], run["first_start"])
            end = min(turn["end"], run["end"])
            if idx > 0:
                prev = runs[idx - 1]
                start = max(start, (prev["last_start"] + run["first_start"]) / 2.0
                            + hop_s / 2.0)
            if idx + 1 < len(runs):
                nxt = runs[idx + 1]
                end = min(end, (run["last_start"] + nxt["first_start"]) / 2.0
                          + hop_s / 2.0)
            if end - start < min_segment_s:
                continue
            pieces.append({"start_ms": int(round(start * 1000)),
                           "end_ms": int(round(end * 1000)),
                           "speaker": "Speaker_%d" % (run["label"] + 1),
                           "is_main": run["label"] == main,
                           "_dur": end - start})
        if not pieces:
            continue
        if not any(p["is_main"] for p in pieces):
            max(pieces, key=lambda p: p["_dur"])["is_main"] = True
        for piece in pieces:
            piece.pop("_dur", None)
        segments[turn["utt_id"]] = pieces
        subs = [{"start_ms": p["start_ms"], "end_ms": p["end_ms"],
                 "speaker": p["speaker"]}
                for p in pieces
                if (not p["is_main"]
                    and p["end_ms"] - p["start_ms"] >= int(round(min_subturn_s * 1000)))]
        if subs:
            sub_turns[turn["utt_id"]] = subs
    return segments, sub_turns


def _osd_chunk_windows(audio: np.ndarray, wins_vad):
    """有 OSD 就用 class 区间切窗，否则原样返回 VAD 窗。"""
    from run_3dspeaker_pipeline import class_chunks, osd_probs
    if not OSD_ONNX.exists():
        return wins_vad, "vad(无OSD)"
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OSD_ONNX), providers=["CPUExecutionProvider"])
    probs_all, mask_all, fs = osd_probs(sess, audio)
    return class_chunks(probs_all[mask_all], fs)[0], "class"


def backfill_speakers(
    wav_path: str,
    utterances: List[Dict],
    models: Optional[Dict] = None,
    mode: str = "auto",
    normalize_peak: float = DEFAULT_NORMALIZE_PEAK,
    with_segments: bool = False,
) -> Dict:
    """整场音频 + 流式阶段锁死的句子 -> {utt_id: "Speaker_N"}。

    utterances: [{"utt_id": "utt_0001", "start_ms": 1200, "end_ms": 4500}, ...]
                start_ms/end_ms 必须是流式阶段按采样点计数锁死的绝对值（见
                .zhang/streaming_asr_engine.py）；本函数只按这两个数对齐。
    models:     web_server.get_models() 的返回（含 "vad"/"spk"）。None 时自取。
    mode:       "auto"（有 OSD 用 class，否则退化 vad）| "class" | "vad"
    normalize_peak: 弱音提拉目标峰值（只提拉不衰减；0 = 关闭）。VAD 有绝对
                能量门限，低电平录音会整段漏检，细化窗覆盖率随之崩掉。
    with_segments: False（默认）返回扁平 {utt_id: "Speaker_N"}，与 Meeting_main
                的既有契约兼容；True 返回下面的富结构。

    返回（with_segments=False）：{utt_id: "Speaker_N"}；N 是聚类标签 +1，全局一致。
    返回（with_segments=True）：
      mapping    {utt_id: "Speaker_N"}
      segments   {utt_id: [{"start_ms", "end_ms", "speaker", "is_main"}, ...]}
                 细化层给出的句内时间片（0.6s/0.15s 短窗合并），供下游切 Turn。
      sub_turns  {utt_id: [{"start_ms", "end_ms", "speaker"}, ...]}
                 仅非主说话人且持续 >= MIN_SUBTURN_S（1.0s）的稳定片段。
    """
    from run_3dspeaker_pipeline import (DENSE_RESCUE_ENABLED, chunk_windows,
                                        dense_identity_estimate, embed_chunks,
                                        spectral_cluster, turns_to_speakers)
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12

    if mode not in ("auto", "class", "vad"):
        raise ValueError(f"mode 只支持 auto/class/vad，收到 {mode!r}")
    if not utterances:
        return {}

    if models is None:
        from run_der import load_models
        models = load_models()

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 句子键名是 *_ms，链路内部一律用秒
    turns = [{"utt_id": u["utt_id"],
              "start": u["start_ms"] / 1000.0,
              "end": u["end_ms"] / 1000.0} for u in utterances]

    # 弱音提拉：VAD 是绝对能量门限，低电平录音的语音段会被整段漏掉。
    # 只提拉不衰减——已足够响的录音保持逐位不变（与 1015 基线同口径）。
    audio, norm_wav = _normalize_audio(audio, rate, normalize_peak)
    try:
        vad = v11._vad_segs(models["vad"], norm_wav or str(wav_path))
    finally:
        if norm_wav:
            try:
                os.unlink(norm_wav)
            except OSError:
                pass
    wins = chunk_windows(vad)
    if mode in ("auto", "class"):
        if mode == "class" and not OSD_ONNX.exists():
            raise FileNotFoundError(f"mode='class' 需要 OSD 模型：{OSD_ONNX}")
        wins, arm = _osd_chunk_windows(audio, wins)
        if arm != "class":
            # 降级必须是可观测的：否则一次 class 臂的结果会被当成 VAD 臂（反之亦然）
            print("[diarizer] 未找到 OSD（%s），本次退化为纯 VAD 切窗" % OSD_ONNX,
                  flush=True)
    if not wins:
        return {u["utt_id"]: "Speaker_1" for u in utterances}

    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    if len(E) == 0:
        return {u["utt_id"]: "Speaker_1" for u in utterances}

    labels, k, _ = spectral_cluster(E, None)

    # 细化层：长窗定身份、0.6s 密集窗找边界；没短窗盖住的句子退回长窗（fallback）。
    # 长窗 K 可能漏掉短插话说话人（class-chunk 塞不下 <1.5s 区间），此时用
    # 0.6s 密集窗独立估 K；只有密集窗 K 更大时才接管身份锚点。
    rwins = rlabels = None
    fb_labels = labels
    if k >= 1:
        anchors = v11._centroids(E, labels)
        alabs = sorted({int(x) for x in labels.tolist()})
        rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
        # 密集窗救援默认关闭（DENSE_RESCUE_ENABLED=False，2026-10-18 拍板）：
        # 0.6s 短窗方差较大，真实 2 人对有被过切成 3 人的风险；餐饮段
        # （b2d1…）长窗本身就是 3 人/11 切换，不依赖救援。打开时也只在
        # 长窗 K=2、密集窗给出 3 时接管。
        dense = (dense_identity_estimate(rE)
                 if DENSE_RESCUE_ENABLED and len(rE) else None)
        if (DENSE_RESCUE_ENABLED and dense is not None and k == 2
                and k < dense["k"] <= k + 1):
            anchors = dense["anchors"]
            alabs = dense["anchor_labels"]
            fb_labels = np.asarray([alabs[i] for i in
                                    np.argmax(E @ anchors.T, axis=1)])
            print("[diarizer] 长窗 K=2 漏掉短插话说话人；密集窗复核 K=%d"
                  "（sil=%.3f, %d/%d 窗）"
                  % (dense["k"], dense["silhouette"],
                     dense["n_sampled"], len(rE)), flush=True)
        rlabels = v12._refine_labels(rE, anchors, alabs, rtimes)
        if rlabels is None or len(set(rlabels.tolist())) < 2:
            rwins = rlabels = None
        else:
            rwins = [(float(t), float(t) + v12.REFINE_WIN_S) for t in rtimes]

    if rwins is not None:
        pred, _ = turns_to_speakers(rwins, rlabels, turns, fallback=(kept, fb_labels))
    else:
        pred, _ = turns_to_speakers(kept, labels, turns)

    mapping = {t["utt_id"]: f"Speaker_{int(p) + 1}" for t, p in zip(turns, pred)}
    if not with_segments:
        return mapping

    # 句内切分：细化短窗已经找出了换人边界，但 turns_to_speakers 只保留句级
    # 主导标签，短的一方会被整句吞掉。这里把连续同标签的短窗合并成片段返回。
    segments, sub_turns = {}, {}
    if rwins is not None:
        segments, sub_turns = _refined_turn_segments(
            turns, rwins, rlabels, pred,
            hop_s=v12.REFINE_HOP_S,
            min_subturn_s=MIN_SUBTURN_S, min_segment_s=MIN_SEGMENT_S)
    return {"mapping": mapping, "segments": segments, "sub_turns": sub_turns}


# --------------------------------------------------------------------------
# 自检：确认 K 不塌陷 + 打印说话人分布（不需要真值）
#   python .zhang/offline_diarizer.py R8008_M8013
# 准确率对账走 eval_multilabel.py（已有），不要在这里再写一个评分器。
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from collections import Counter

    from common import resolve_wav
    from eval_multilabel import vad_sentences

    meeting = sys.argv[1] if len(sys.argv) > 1 else "R8008_M8013"
    wav = resolve_wav(meeting)

    t0 = time.time()
    from run_der import load_models
    models = load_models()
    # 1000ms = StreamingSession.max_end_silence_time 的默认值；同口径（少一次 ASR 二次切分）
    utts = [{"utt_id": f"utt_{i:04d}",
             "start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000)}
            for i, s in enumerate(
                vad_sentences(models["vad"], wav, max_end_silence_ms=1000)[0], 1)]

    mapping = backfill_speakers(wav, utts, models=models, mode="auto")
    hist = Counter(mapping.values())

    print(f"\n{meeting}: {len(utts)} 句 -> {len(hist)} 个说话人  "
          f"({time.time() - t0:.1f}s)")
    for spk, n in hist.most_common():
        print(f"  {spk:12s} {n:5d} 句  {n / len(utts):6.1%}")
    print("K 塌陷为 1" if len(hist) == 1 else f"K = {len(hist)}（未塌陷）")
