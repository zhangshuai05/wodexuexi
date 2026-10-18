# -*- coding: utf-8 -*-
"""v12 泳道：2.0 秒 CAM++ 窗 + 自动人数聚类（8735 在线版）。

与 v11 的关系：同一套 VAD/CAM++ 模型与同一套 AHC/silhouette 选择逻辑，换掉的是
窗口参数。v12 在 AISHELL-4 真实会议（12 场开发 / 8 场身份隔离留出）上测出
**2.0 秒纯说话人窗**的余弦 EER 显著优于 1.2/0.6 秒（开发 8.97% 对 14.58%/26.37%），
所以这里把窗长从 v11 的 0.6 秒改成 2.0 秒。ERes2NetV2 与 CAM++ 在 2.0 秒上持平
（开发 8.97% 对 8.84%），但 CAM++ 推理快 2.28 倍，故不换模型。

已知局限（AISHELL-4 留出集实测，见 V12_EXPERIMENT.md）：
  - 已知真实人数时平均聚类上界 92.77%，自动人数 8 场里对 5 场；
  - 失败全部发生在把一位说话人**拆成两人**，不是多估；
  - 2.0 秒窗比 v11 的 0.6 秒窗**时间分辨率低**，短插话会被相邻窗吸收。
前两条是 v12 下一阶段要解决的缺口（人数估计、重叠处理），这个泳道就是用来在真实
长会议上暴露它们的，不是已验收能力。
"""

import os
import threading

import numpy as np

# v11 的窗口类常量是模块级全局（它自己也是按 0.6 秒写死的），临时换掉再换回来。
# 加锁是因为 8735 用 ThreadingHTTPServer，/api/compare 之间可能并发。
_V11_PARAMS_LOCK = threading.Lock()
_V11_WINDOW_PARAMS = ("WIN_S", "HOP_S", "MAX_SPEAKERS")

import claude_a2_v11 as v11

METHOD = "claude-a2-v12-meeting"
WIN_S = 2.0
HOP_S = 1.0
MIN_VAD_FRACTION = 0.5
MAX_SPEAKERS = 10
MIN_CLUSTER_WINDOWS = 3
MIN_CLUSTER_SHARE = 0.02
MULTI_K_PENALTY = 0.015
MAX_ASR_SECONDS = 8.0
SCENE_GAP_SECONDS = 2.0

# 低于这个窗数的簇：标签只在一两个窗上出现，2.0 秒窗下更可能是离群窗而非真人。
SLIVER_WINDOWS = 2

# 两级窗。2.0 秒窗在安静长会议上 GT 单人窗占比 73%，但店铺点单素材轮次中位仅 1.57 秒，
# 2.0 秒窗只有 46--72% 落在单人区，实测 56% 的窗里坐着两个人 —— CAM++ 给出的就是这两个
# 人的平均声纹。所以长窗只用来**定身份**，换人边界必须靠短窗找（v7 的既有结论）。
REFINE_WIN_S = 0.6
REFINE_HOP_S = 0.15
REFINE_MIN_VAD = 0.3
REFINE_MIN_WINDOWS = 4
# 边距绝对值低于此值的短段可以并进邻居（沿用 v7 `_absorb_slivers` 的 weak_margin）。
SLIVER_MARGIN = 0.18
SLIVER_MIN_DUR = 0.75


class _v11_window_params:
    """临时把 v11 的窗口常量换成指定值（_extract_windows/_candidate_clusters 读它们）。"""

    def __init__(self, win_s, hop_s, max_speakers):
        self._values = {"WIN_S": win_s, "HOP_S": hop_s, "MAX_SPEAKERS": max_speakers}

    def __enter__(self):
        self._lock = _V11_PARAMS_LOCK
        self._lock.acquire()
        self._saved = {name: getattr(v11, name) for name in _V11_WINDOW_PARAMS}
        for name, value in self._values.items():
            setattr(v11, name, value)
        return self

    def __exit__(self, *exc_info):
        for name, value in self._saved.items():
            setattr(v11, name, value)
        self._lock.release()
        return False


def _segment_windows(times, labels, margins, hop_s=HOP_S):
    """把逐窗标签合成段边界。边界落在相邻窗重叠中点，步长由 hop_s 给。

    v11 的 `_segments_from_windows` 把中点公式写死成它自己的 HOP/2；v12 主窗和救援窗
    步长不同（1.0 / 0.15），所以这里按实际步长算。相邻同标签窗直接连通，避免同一人
    被切成很多小段。
    """
    if not len(times):
        return []
    starts = [0]
    for index in range(1, len(times)):
        if labels[index] != labels[index - 1] or times[index] - times[index - 1] > hop_s * 1.5:
            starts.append(index)
    starts.append(len(times))
    segments = []
    for first, last in zip(starts[:-1], starts[1:]):
        start = 0.0 if first == 0 else float((times[first - 1] + times[first]) / 2 + hop_s / 2)
        end = float(times[last - 1] + hop_s) if last == len(times) else float(
            (times[last - 1] + times[last]) / 2 + hop_s / 2)
        span = slice(first, last)
        segments.append({
            "start": start,
            "end": end,
            "spk": int(labels[first]),
            "n_windows": int(last - first),
            "margin": round(float(np.mean(margins[span])), 4) if len(margins) else None,
        })
    return segments


def _refine_windows(audio, rate, vad_segments, spk_model):
    """0.6 秒 / 0.15 秒密集窗的归一化嵌入与窗起点（只用于找换人边界）。"""
    duration = len(audio) / rate
    vectors, times = [], []
    for start in np.arange(0.0, max(0.0, duration - REFINE_WIN_S) + 1e-9, REFINE_HOP_S):
        end = start + REFINE_WIN_S
        if v11._overlap(start, end, vad_segments) < REFINE_MIN_VAD * REFINE_WIN_S:
            continue
        chunk = audio[int(start * rate):int(end * rate)]
        if len(chunk) < int(0.25 * rate):
            continue
        try:
            value = spk_model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            continue
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        vectors.append(v11._normalized(value))
        times.append(float(start))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), np.asarray(times)
    return np.stack(vectors), np.asarray(times)


def _refine_labels(E, anchors, anchor_labels, times):
    """长窗定身份、短窗找边界：每个短窗按最近锚点判，再用 v7 的碎段吸收去噪。

    `anchors[i]` 是长窗聚类里标签 `anchor_labels[i]` 的质心。短窗不重新聚类 ——
    重聚类会让身份漂移，而身份正是长窗的强项。返回逐短窗标签，或 None（窗太少）。
    """
    if len(E) < REFINE_MIN_WINDOWS or len(anchors) < 2:
        return None
    scores = E @ anchors.T
    assigned = np.asarray([anchor_labels[index] for index in np.argmax(scores, axis=1)])
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    from claude_a2_v7_v8 import _absorb_slivers, _stabilize_labels

    stable = _stabilize_labels(assigned, margins)
    return _absorb_slivers(stable, margins, times, REFINE_WIN_S,
                           min_dur=SLIVER_MIN_DUR, weak_margin=SLIVER_MARGIN)[0]


def run_claude_lane(wav, models, lane_key, progress=None):
    if lane_key != "v12":
        raise ValueError(f"unknown v12 lane {lane_key}")
    import soundfile as sf
    from claude_a2_v8_contract import segment_contract

    if progress:
        progress("v12：提取 2.0 秒 CAM++ 窗…")
    audio, rate = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = v11._vad_segs(models["vad"], wav)
    with _v11_window_params(WIN_S, HOP_S, MAX_SPEAKERS):
        E, times, embedding_errors = v11._extract_windows(
            audio, rate, vad_segments, models["spk"])
        if len(E) < 2:
            return {
                "method": METHOD, "file": os.path.basename(wav),
                "elapsed": 0.0, "verdict": "experimental",
                "verdict_reasons": ["v12 窗数不足，无法做多人聚类"],
                "diag": {"algorithm": "v12_meeting_2s", "n_windows": int(len(E)),
                         "window_seconds": WIN_S, "n_speakers": 0,
                         "cluster_level": "insufficient"},
                "sentence_info": [],
            }
        candidates, _ = v11._candidate_clusters(E)
        selected = v11._select_candidate(candidates)
        labels = selected["labels"].copy()

    # 两级：长窗已定好身份（k 与质心），短窗只负责找换人边界。
    # 长窗塌成单人或短窗不足时退回长窗自己的分段。
    refined = False
    if selected["k"] >= 2:
        if progress:
            progress(f"v12：{selected['k']} 人身份已定，改用 {REFINE_WIN_S} 秒短窗细化边界…")
        rE, rtimes = _refine_windows(audio, rate, vad_segments, models["spk"])
        anchor_labels = sorted(set(int(value) for value in labels))
        anchors = v11._centroids(E, labels)
        rlabels = _refine_labels(rE, anchors, anchor_labels, rtimes)
        if rlabels is not None and len(set(rlabels.tolist())) >= 2:
            E, times, labels = rE, rtimes, rlabels
            refined = True

    # 长窗判成单人时，用密集窗复核身份：只认分出更多人的结果，否则保持单人。
    rescued = False
    if not refined and selected["k"] == 1:
        if progress:
            progress("v12：2.0 秒窗判为单人，改用 0.6 秒密集窗复核…")
        with _v11_window_params(REFINE_WIN_S, REFINE_HOP_S, MAX_SPEAKERS):
            rE, rtimes, rescue_errors = v11._extract_windows(
                audio, rate, vad_segments, models["spk"])
            rcandidates, _ = v11._candidate_clusters(rE)
            rselected = v11._select_candidate(rcandidates) if len(rE) >= 2 else None
        if rselected is not None and rselected["k"] > 1:
            E, times, labels = rE, rtimes, rselected["labels"].copy()
            candidates, selected = rcandidates, rselected
            embedding_errors += rescue_errors
            rescued = True

    # 相对所选簇质心的余弦 margin：给每段一个"这个标签有多确定"的可核验量。
    centroids = v11._centroids(E, labels)
    scores = E @ centroids.T
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2] if scores.shape[1] >= 2 else np.zeros(len(E))

    segments = _segment_windows(times, labels, margins,
                                hop_s=REFINE_HOP_S if (refined or rescued) else HOP_S)
    tiny = [segment for segment in segments if segment["n_windows"] <= SLIVER_WINDOWS]
    segments = [segment for segment in segments if segment["n_windows"] > SLIVER_WINDOWS]
    for segment in segments:
        segment.pop("n_windows", None)
    asr_segments = [piece for segment in segments
                    for piece in v11._split_for_asr(segment, vad_segments,
                                                    max_seconds=MAX_ASR_SECONDS)]

    if progress:
        progress(f"v12：选择 {selected['k']} 人候选，"
                 f"逐段转写 {len(asr_segments)} 段（剔除 {len(tiny)} 个孤立短段）…")
    sentences, flags = [], []
    for index, segment in enumerate(asr_segments):
        chunk = audio[int(segment["start"] * rate):int(segment["end"] * rate)]
        text = ""
        if len(chunk) >= int(0.3 * rate):
            try:
                text = models["asr"].generate(input=chunk)[0].get("text", "") or ""
            except Exception:
                text = ""
        item = {
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "spk": int(segment["spk"]),
            "text": text,
            "asr_split": bool(segment.get("asr_split")),
        }
        contract = segment_contract(text, item["start"], item["end"], vad_segments)
        item.update({"flag": contract["flag"], "asr_chars": contract["chars"],
                     "asr_cps": contract["cps"], "speech_sec": contract["speech_sec"]})
        if contract["reasons"]:
            item["asr_note"] = "；".join(contract["reasons"])
        sentences.append(item)
        flags.append(contract["flag"])
        if progress and (index % 5 == 0 or index == len(asr_segments) - 1):
            progress(f"v12：逐段转写 {index + 1}/{len(asr_segments)}")

    actual_k = len({item["spk"] for item in sentences})
    used_win = REFINE_WIN_S if (refined or rescued) else WIN_S
    used_hop = REFINE_HOP_S if (refined or rescued) else HOP_S
    reasons = [
        f"v12 会议实验泳道：{WIN_S} 秒窗定身份 + {REFINE_WIN_S} 秒窗找边界，"
        f"自动人数 1–{MAX_SPEAKERS}",
        "AISHELL-4 留出集上自动人数 8 场对 5 场，失败多为把一人拆成两人；"
        "此处结果同样可能多出说话人标签，需人工核对",
    ]
    if refined:
        reasons.append("已用短窗细化换人边界（身份来自 2.0 秒长窗，边界来自 "
                       f"{REFINE_WIN_S} 秒短窗）")
    elif rescued:
        reasons.append("2.0 秒窗判为单人，已改用 0.6 秒密集窗复核并分出多人；"
                       "该路径未经 AISHELL-4 验收，仅作诊断")
    else:
        reasons.append("短窗细化不可用，退回 2.0 秒长窗分段："
                       "短插话可能被相邻窗吸收，边界偏粗")
    if tiny:
        reasons.append(f"剔除 {len(tiny)} 个孤立短段（≤{SLIVER_WINDOWS} 窗），"
                       "它们可能是真实短插话，也可能是离群窗")
    suspect = sorted({flag for flag in flags if flag in ("empty", "no_speech", "too_fast", "repeated")})
    if suspect:
        reasons.append(f"{sum(flag in suspect for flag in flags)} 段文本可疑（{'/'.join(suspect)}）")
    return {
        "method": METHOD,
        "file": os.path.basename(wav),
        "elapsed": 0.0,
        "diag": {
            "algorithm": "v12_meeting_2s",
            "cluster_level": ("dense_refine_2level" if refined
                              else "dense_rescue_ahc" if rescued
                              else "dense_global_ahc_2s"),
            "window_seconds": used_win,
            "hop_seconds": used_hop,
            "primary_window_seconds": WIN_S,
            "refined_boundaries": refined,
            "rescued_from_single": rescued,
            "n_windows": int(len(E)),
            "embedding_errors": embedding_errors,
            "assumed_k": int(selected["k"]),
            "n_speakers": actual_k,
            "n_clusters": actual_k,
            "k_candidates": [
                {"k": item["k"], "silhouette": round(float(item["score"]), 4),
                 "objective": round(float(item["objective"]), 4), "sizes": item["sizes"],
                 "eligible": bool(item["eligible"])}
                for item in candidates
            ],
            "scenes": v11._scene_summary(times, labels),
            "largest_vad_gaps": v11._largest_window_gaps(times, labels),
            "n_sliver_segments_dropped": len(tiny),
            "max_asr_seconds": MAX_ASR_SECONDS,
            "n_asr_splits": sum(bool(item.get("asr_split")) for item in sentences),
            "n_flag_uncertain": sum(flag != "ok" for flag in flags),
        },
        "verdict": "experimental",
        "verdict_reasons": reasons,
        "sentence_info": sentences,
    }
