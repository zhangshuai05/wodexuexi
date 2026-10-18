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

# 句内切分门控：避免普通短句被切成词中两段。
MIN_SUBTURN_S = float(os.environ.get("DIAR_MIN_SUBTURN_S", "2.5"))
MIN_SEGMENT_S = float(os.environ.get("DIAR_MIN_SEGMENT_S", "0.5"))
MIN_SPLITTABLE_TURN_S = float(os.environ.get("DIAR_MIN_SPLITTABLE_TURN_S", "4.5"))
MIN_MAIN_REMAINDER_S = float(os.environ.get("DIAR_MIN_MAIN_REMAINDER_S", "1.2"))
SUBTURN_MIN_MARGIN = float(os.environ.get("DIAR_SUBTURN_MIN_MARGIN", "0.08"))

DEFAULT_MERGE_COS_TH = float(os.environ.get("DIAR_MERGE_COS_TH", "0.58"))
ISLAND_MAX_DUR_S = float(os.environ.get("DIAR_ISLAND_MAX_DUR_S", "3.0"))
ISLAND_MAX_GAP_S = float(os.environ.get("DIAR_ISLAND_MAX_GAP_S", "3.5"))
HYSTERESIS_MARGIN = float(os.environ.get("DIAR_HYSTERESIS_MARGIN", "0.10"))


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


def _turn_embedding_scores(
    turns: List[Dict],
    win_spans: List[tuple],
    win_embeds: np.ndarray,
    anchors: np.ndarray,
    anchor_labels: List[int],
) -> np.ndarray:
    """按窗与 turn 的重叠时长，计算每句对各说话人质心的平均余弦分数。"""
    scores = np.zeros((len(turns), len(anchor_labels)), dtype=np.float32)
    if not turns or not anchor_labels or not len(win_spans) or not len(win_embeds):
        return scores
    if len(win_spans) != len(win_embeds):
        raise ValueError("声纹嵌入数与长窗数不一致")
    win_norm = win_embeds / (np.linalg.norm(win_embeds, axis=1, keepdims=True) + 1e-9)
    anchor_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
    similarities = win_norm @ anchor_norm.T
    for i, turn in enumerate(turns):
        total = 0.0
        weighted = np.zeros(len(anchor_labels), dtype=np.float64)
        for j, (start, end) in enumerate(win_spans):
            overlap = min(float(end), turn["end"]) - max(float(start), turn["start"])
            if overlap > 0:
                weighted += overlap * similarities[j]
                total += overlap
        if total:
            scores[i] = (weighted / total).astype(np.float32)
    return scores


def _smooth_turn_speakers(
    turns: List[Dict],
    pred: List[int],
    turn_scores: Optional[np.ndarray],
    anchor_labels: List[int],
    *,
    island_max_dur_s: float = ISLAND_MAX_DUR_S,
    island_max_gap_s: float = ISLAND_MAX_GAP_S,
    hysteresis_margin: float = HYSTERESIS_MARGIN,
) -> List[int]:
    """用迟滞规则吸收长独白中置信度不足的短说话人孤岛。"""
    out = list(pred)
    if len(out) < 2:
        return out
    label_to_col = {int(label): i for i, label in enumerate(anchor_labels)}

    def margin(i: int, current: int, target: int) -> float:
        if (turn_scores is None or i >= len(turn_scores)
                or current not in label_to_col or target not in label_to_col):
            return float("inf")
        row = turn_scores[i]
        if not np.all(np.isfinite(row)) or not np.any(row):
            # 无长窗覆盖时没有声纹证据；保留 fallback 投票给出的身份。
            return float("inf")
        return float(row[label_to_col[current]] - row[label_to_col[target]])

    # 一句夹心：A -> B -> A。
    for i in range(1, len(out) - 1):
        prev, current, nxt = out[i - 1], out[i], out[i + 1]
        if prev < 0 or current == prev or nxt != prev:
            continue
        duration = float(turns[i]["end"]) - float(turns[i]["start"])
        left_gap = float(turns[i]["start"]) - float(turns[i - 1]["end"])
        right_gap = float(turns[i + 1]["start"]) - float(turns[i]["end"])
        if (duration <= island_max_dur_s and left_gap <= island_max_gap_s
                and right_gap <= island_max_gap_s
                and margin(i, current, prev) < hysteresis_margin):
            out[i] = prev

    # 两句夹心：A -> B/C -> A；合计时长必须仍处在短孤岛范围内。
    for i in range(1, len(out) - 2):
        prev, nxt = out[i - 1], out[i + 2]
        if prev < 0 or nxt != prev or (out[i] == prev and out[i + 1] == prev):
            continue
        duration = float(turns[i + 1]["end"]) - float(turns[i]["start"])
        left_gap = float(turns[i]["start"]) - float(turns[i - 1]["end"])
        right_gap = float(turns[i + 2]["start"]) - float(turns[i + 1]["end"])
        if (duration <= island_max_dur_s and left_gap <= island_max_gap_s
                and right_gap <= island_max_gap_s
                and max(margin(i, out[i], prev), margin(i + 1, out[i + 1], prev))
                < hysteresis_margin):
            out[i] = out[i + 1] = prev

    # 极短句紧邻上一句时，只有声纹优势足够大才确认换人。
    for i in range(1, len(out)):
        prev, current = out[i - 1], out[i]
        if prev < 0 or current < 0 or current == prev:
            continue
        duration = float(turns[i]["end"]) - float(turns[i]["start"])
        gap = float(turns[i]["start"]) - float(turns[i - 1]["end"])
        if (duration <= 1.5 and gap <= 1.5
                and margin(i, current, prev) < hysteresis_margin * 0.8):
            out[i] = prev
    return out


def _refined_turn_segments(turns, windows, labels, main_labels, hop_s,
                           min_subturn_s, min_segment_s, *, rE=None,
                           anchors=None, anchor_labels=None,
                           min_splittable_turn_s=MIN_SPLITTABLE_TURN_S,
                           min_main_remainder_s=MIN_MAIN_REMAINDER_S,
                           subturn_min_margin=SUBTURN_MIN_MARGIN,
                           enable_subturns=True):
    """细化短窗转成句内片段；只有长句中的稳定、高置信度换人进入 sub_turns。"""
    segments, sub_turns = {}, {}
    if len(windows) != len(labels):
        raise ValueError("细化短窗数与标签数不一致")
    label_list = np.asarray(labels, dtype=int).tolist()
    label_values = [] if anchor_labels is None else anchor_labels
    label_to_col = {int(label): i for i, label in enumerate(label_values)}
    win_sims = None
    if (rE is not None and anchors is not None and len(rE) == len(windows)
            and len(anchors)):
        r_norm = rE / (np.linalg.norm(rE, axis=1, keepdims=True) + 1e-9)
        a_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-9)
        win_sims = r_norm @ a_norm.T

    for turn, main_raw in zip(turns, main_labels):
        main = int(main_raw)
        if main < 0:
            continue
        turn_start, turn_end = float(turn["start"]), float(turn["end"])
        if not enable_subturns or turn_end - turn_start < min_splittable_turn_s:
            segments[turn["utt_id"]] = [{
                "start_ms": int(round(turn_start * 1000)),
                "end_ms": int(round(turn_end * 1000)),
                "speaker": "Speaker_%d" % (main + 1),
                "is_main": True,
            }]
            continue

        picked = []
        for idx, ((a, b), label) in enumerate(zip(windows, label_list)):
            if min(float(b), turn_end) - max(float(a), turn_start) > 1e-6:
                picked.append((float(a), float(b), label, idx))
        if not picked:
            continue
        picked.sort(key=lambda row: row[0])

        runs = []
        for idx, (a, b, label, win_idx) in enumerate(picked):
            if (runs and runs[-1]["label"] == label
                    and a - picked[idx - 1][0] <= hop_s * 1.5 + 1e-9):
                runs[-1]["last_start"] = a
                runs[-1]["end"] = max(runs[-1]["end"], b)
                runs[-1]["indices"].append(win_idx)
            else:
                runs.append({"label": label, "first_start": a, "last_start": a,
                             "end": b, "indices": [win_idx]})

        # 若短窗对异类质心的平均余弦优势不足，先归并回主说话人，再合并相邻段。
        if win_sims is not None and main in label_to_col:
            for run in runs:
                label = run["label"]
                if label == main or label not in label_to_col:
                    continue
                idxs = run["indices"]
                sub_score = float(np.mean(win_sims[idxs, label_to_col[label]]))
                main_score = float(np.mean(win_sims[idxs, label_to_col[main]]))
                if sub_score - main_score < subturn_min_margin:
                    run["label"] = main

        merged = []
        for run in runs:
            if merged and merged[-1]["label"] == run["label"]:
                merged[-1]["last_start"] = run["last_start"]
                merged[-1]["end"] = max(merged[-1]["end"], run["end"])
                merged[-1]["indices"].extend(run["indices"])
            else:
                merged.append(run)

        pieces = []
        for idx, run in enumerate(merged):
            start = max(turn_start, run["first_start"])
            end = min(turn_end, run["end"])
            if idx > 0:
                prev = merged[idx - 1]
                start = max(start, (prev["last_start"] + run["first_start"]) / 2.0
                            + hop_s / 2.0)
            if idx + 1 < len(merged):
                nxt = merged[idx + 1]
                end = min(end, (run["last_start"] + nxt["first_start"]) / 2.0
                          + hop_s / 2.0)
            if end - start >= min_segment_s:
                pieces.append({
                    "start_ms": int(round(start * 1000)),
                    "end_ms": int(round(end * 1000)),
                    "speaker": "Speaker_%d" % (run["label"] + 1),
                    "is_main": run["label"] == main,
                    "_dur": end - start,
                })
        if not pieces:
            continue
        if not any(piece["is_main"] for piece in pieces):
            max(pieces, key=lambda piece: piece["_dur"])["is_main"] = True
        main_duration = sum(piece["_dur"] for piece in pieces if piece["is_main"])
        for piece in pieces:
            piece.pop("_dur", None)
        segments[turn["utt_id"]] = pieces

        if main_duration < min_main_remainder_s:
            continue
        stable_subs = [{
            "start_ms": piece["start_ms"], "end_ms": piece["end_ms"],
            "speaker": piece["speaker"],
        } for piece in pieces if (not piece["is_main"]
             and piece["end_ms"] - piece["start_ms"] >= int(round(min_subturn_s * 1000)))]
        if stable_subs:
            sub_turns[turn["utt_id"]] = stable_subs
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


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off")


def backfill_speakers(
    wav_path: str,
    utterances: List[Dict],
    models: Optional[Dict] = None,
    mode: str = "auto",
    normalize_peak: float = DEFAULT_NORMALIZE_PEAK,
    with_segments: bool = False,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    merge_cos_th: float = DEFAULT_MERGE_COS_TH,
    enable_subturns: Optional[bool] = None,
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
                 仅长句中持续 >= MIN_SUBTURN_S 且声纹优势足够的稳定片段。

    num_speakers: 已知现场人数时锁定精确簇数；也可用 DIAR_NUM_SPEAKERS 设置。
    min_speakers / max_speakers: 自动人数估计的范围约束。
    merge_cos_th: 同一说话人被分裂时的质心合并阈值（默认 0.58）。
    enable_subturns: 是否允许回传稳定的句内换人；默认 True，可用
                      DIAR_ENABLE_SUBTURNS=0 关闭。
    """
    from run_3dspeaker_pipeline import (DENSE_RESCUE_ENABLED, chunk_windows,
                                        dense_identity_estimate, embed_chunks,
                                        spectral_cluster, turns_to_speakers)
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12

    if mode not in ("auto", "class", "vad"):
        raise ValueError(f"mode 只支持 auto/class/vad，收到 {mode!r}")
    if not utterances:
        return {} if not with_segments else {"mapping": {}, "segments": {}, "sub_turns": {}}

    if num_speakers is None:
        num_speakers = _env_int("DIAR_NUM_SPEAKERS")
    if min_speakers is None:
        min_speakers = _env_int("DIAR_MIN_SPEAKERS")
    if max_speakers is None:
        max_speakers = _env_int("DIAR_MAX_SPEAKERS")
    if enable_subturns is None:
        enable_subturns = _env_bool("DIAR_ENABLE_SUBTURNS", default=True)
    for name, value in (("num_speakers", num_speakers),
                        ("min_speakers", min_speakers),
                        ("max_speakers", max_speakers)):
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} 必须为正整数")
    if min_speakers is not None and max_speakers is not None \
            and min_speakers > max_speakers:
        raise ValueError("min_speakers 不能大于 max_speakers")

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
        # Even the no-voice fallback must honor explicit speaker bounds.
        spectral_cluster(np.empty((0, 0), dtype=np.float32), num_speakers,
                         min_speakers=min_speakers, max_speakers=max_speakers)
        default_map = {u["utt_id"]: "Speaker_1" for u in utterances}
        return default_map if not with_segments else {
            "mapping": default_map, "segments": {}, "sub_turns": {}
        }

    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    if len(E) == 0:
        spectral_cluster(np.empty((0, 0), dtype=np.float32), num_speakers,
                         min_speakers=min_speakers, max_speakers=max_speakers)
        default_map = {u["utt_id"]: "Speaker_1" for u in utterances}
        return default_map if not with_segments else {
            "mapping": default_map, "segments": {}, "sub_turns": {}
        }

    labels, k, _ = spectral_cluster(
        E,
        num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        merge_cos_th=merge_cos_th,
    )

    # 细化层：长窗定身份、0.6s 密集窗找边界；没短窗盖住的句子退回长窗（fallback）。
    # 长窗 K 可能漏掉短插话说话人（class-chunk 塞不下 <1.5s 区间），此时用
    # 0.6s 密集窗独立估 K；只有密集窗 K 更大时才接管身份锚点。
    rwins = rlabels = None
    fb_labels = labels
    anchors = None
    alabs: List[int] = []
    if k >= 1:
        anchors = v11._centroids(E, labels)
        alabs = sorted({int(x) for x in labels.tolist()})
        rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
        # 密集窗救援默认关闭（DENSE_RESCUE_ENABLED=False，2026-10-18 拍板）：
        # 0.6s 短窗方差较大，真实 2 人对有被过切成 3 人的风险；餐饮段
        # （b2d1…）长窗本身就是 3 人/11 切换，不依赖救援。打开时也只在
        # 长窗 K=2、密集窗给出 3 时接管。
        # An explicit speaker count is a hard constraint; dense rescue may
        # otherwise replace the requested K anchors with a larger estimate.
        dense = (dense_identity_estimate(rE)
                 if DENSE_RESCUE_ENABLED and num_speakers is None and len(rE) else None)
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

    # 长窗是句级身份主票；短窗只在整句没有长窗覆盖时兜底。
    if rwins is not None:
        pred, _ = turns_to_speakers(kept, fb_labels, turns, fallback=(rwins, rlabels))
    else:
        pred, _ = turns_to_speakers(kept, fb_labels, turns)

    if anchors is not None and len(alabs) >= 2:
        turn_scores = _turn_embedding_scores(turns, kept, E, anchors, alabs)
        pred = _smooth_turn_speakers(turns, pred, turn_scores, alabs)

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
            min_subturn_s=MIN_SUBTURN_S,
            min_segment_s=MIN_SEGMENT_S,
            rE=rE,
            anchors=anchors,
            anchor_labels=alabs,
            enable_subturns=bool(enable_subturns),
        )
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
