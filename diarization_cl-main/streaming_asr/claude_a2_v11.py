# -*- coding: utf-8 -*-
"""Claude v11: independent multi-speaker offline diarization experiment.

v7 is deliberately frozen as a two-speaker baseline.  v11 uses the same
VAD/CAM++ models but makes speaker count a measured choice (1..4), keeps the
candidate evidence in ``diag``, and caps ASR chunks so a collapsed label does
not turn an entire conversation into one decoder input.
"""

import os

import numpy as np


METHOD = "claude-a2-v11-multispeaker"
WIN_S = 0.6
HOP_S = 0.15
MIN_VAD_FRACTION = 0.5
MAX_SPEAKERS = 4
MIN_CLUSTER_WINDOWS = 4
MIN_CLUSTER_SHARE = 0.015
# 0.005：原生产值 0.015 比轮廓信号还大（网格跨度 0.099，惩罚走 8 步就是 0.12），平坦曲线上是常数在选 k。
# 降 λ：train_L 30 场零回归、dev +10.9（in-sample）、holdout 非劣；λ≥0.03 两个 cohort 都翻车。
# base 臂已实测：dev/holdout 逐场 k 不变，train_L 5/30 变且全为正。依据与限定见 docs/HANDOFF.md §4.12。
MULTI_K_PENALTY = 0.005
MAX_ASR_SECONDS = 8.0
SCENE_GAP_SECONDS = 2.0
BLOCK_GAP_SECONDS = 0.75


def _vad_segs(vad, wav):
    result = vad.generate(input=wav, dynamic_silence=False)
    return [
        (float(item[0]) / 1000.0, float(item[1]) / 1000.0)
        for row in (result if isinstance(result, list) else [result])
        for item in row.get("value", [])
    ]


def _overlap(start, end, segments):
    return sum(max(0.0, min(end, right) - max(start, left))
               for left, right in segments)


def _normalized(value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value / (np.linalg.norm(value) + 1e-9)


def _extract_windows(audio, rate, vad_segments, spk_model):
    """Return normalized CAM++ vectors and their 0.6 s window starts."""
    duration = len(audio) / rate
    vectors, times, errors = [], [], 0
    last = max(0.0, duration - WIN_S)
    for start in np.arange(0.0, last + 1e-9, HOP_S):
        end = start + WIN_S
        if _overlap(start, end, vad_segments) < MIN_VAD_FRACTION * WIN_S:
            continue
        chunk = audio[int(start * rate):int(end * rate)]
        if len(chunk) < int(0.25 * rate):
            continue
        try:
            embedding = spk_model.generate(input=chunk)[0].get("spk_embedding")
        except Exception:
            errors += 1
            continue
        if embedding is None:
            continue
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        vectors.append(_normalized(embedding))
        times.append(float(start))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), np.asarray(times), errors
    return np.stack(vectors), np.asarray(times), errors


def _relabel_by_first_seen(labels):
    order = []
    for label in labels.tolist():
        if label not in order:
            order.append(label)
    remap = {label: index for index, label in enumerate(order)}
    return np.asarray([remap[label] for label in labels], dtype=int)


def _silhouette(E, labels, max_samples=512):
    """Deterministic cosine silhouette without adding a sklearn dependency."""
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        return -1.0
    sample = np.linspace(0, len(E) - 1, min(len(E), max_samples), dtype=int)
    values = []
    for index in sample:
        own = labels[index]
        distances = 1.0 - (E[index] @ E.T)
        same = groups[own]
        same = same[same != index]
        intra = float(distances[same].mean()) if len(same) else 0.0
        other = min(float(distances[group].mean())
                    for label, group in enumerate(groups) if label != own)
        values.append((other - intra) / max(other, intra, 1e-9))
    return float(np.mean(values))


def _candidate_clusters(E, max_speakers=MAX_SPEAKERS):
    """Build all K candidates once from the same hierarchy and score support."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(E)
    if n == 0:
        return [], np.empty((0,), dtype=int)
    if n == 1:
        return [{"k": 1, "labels": np.zeros(1, dtype=int), "score": 0.0,
                 "objective": 0.0, "sizes": [1], "eligible": True}], np.zeros(1, dtype=int)

    tree = linkage(E, method="average", metric="cosine")
    candidates = [{"k": 1, "labels": np.zeros(n, dtype=int), "score": 0.0,
                   "objective": 0.0, "sizes": [n], "eligible": True}]
    for k in range(2, min(max_speakers, n) + 1):
        labels = _relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
        sizes = [int((labels == label).sum()) for label in range(k)]
        eligible = (min(sizes) >= MIN_CLUSTER_WINDOWS
                    and min(sizes) / n >= MIN_CLUSTER_SHARE)
        score = _silhouette(E, labels) if eligible else -1.0
        candidates.append({
            "k": k,
            "labels": labels,
            "score": score,
            "objective": score - MULTI_K_PENALTY * (k - 2),
            "sizes": sizes,
            "eligible": eligible,
        })
    return candidates, tree


def _select_candidate(candidates, forced_k=None):
    """Select the supported K with the best silhouette after a small K penalty."""
    if forced_k is not None:
        forced = next((item for item in candidates if item["k"] == forced_k), None)
        if forced is not None:
            return forced
    eligible = [item for item in candidates if item["eligible"] and item["k"] >= 2]
    if not eligible:
        return candidates[0]
    best = max(eligible, key=lambda item: (item["objective"], -item["k"]))
    return best if best["score"] > 0.0 else candidates[0]


def _centroids(E, labels):
    return np.stack([_normalized(E[labels == label].mean(axis=0))
                     for label in sorted(set(labels))])


def _stabilize_isolated(labels, scores, times):
    """Only absorb a one-window flip if its own cluster margin is weak."""
    stable = labels.copy()
    if scores.shape[1] < 2:
        return stable
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    for index in range(1, len(labels) - 1):
        contiguous = (times[index] - times[index - 1] <= HOP_S * 1.5
                      and times[index + 1] - times[index] <= HOP_S * 1.5)
        if (contiguous and labels[index - 1] == labels[index + 1] != labels[index]
                and margins[index] < 0.04):
            stable[index] = labels[index - 1]
    return _relabel_by_first_seen(stable)


def _segments_from_windows(times, labels):
    if not len(times):
        return []
    starts = [0]
    for index in range(1, len(times)):
        if labels[index] != labels[index - 1] or times[index] - times[index - 1] > WIN_S:
            starts.append(index)
    starts.append(len(times))
    output = []
    for first, last in zip(starts[:-1], starts[1:]):
        start = times[first] if first == 0 or times[first] - times[first - 1] > WIN_S else (
            times[first - 1] + times[first] + WIN_S) / 2
        end = times[last - 1] + WIN_S if last == len(times) or times[last] - times[last - 1] > WIN_S else (
            times[last - 1] + times[last] + WIN_S) / 2
        output.append({"start": float(start), "end": float(end), "spk": int(labels[first])})
    return output


def _gap_midpoints(vad_segments, start, end):
    clipped = [(max(start, left), min(end, right)) for left, right in vad_segments
               if right > start and left < end]
    clipped.sort()
    return [(left_end + right_start) / 2 for (_, left_end), (right_start, _) in zip(clipped, clipped[1:])
            if right_start > left_end]


def _split_for_asr(segment, vad_segments, max_seconds=MAX_ASR_SECONDS):
    """Keep a decoder input bounded; prefer a nearby VAD silence when available."""
    start, end = segment["start"], segment["end"]
    if end - start <= max_seconds:
        return [dict(segment, asr_split=False)]
    points, cursor = [], start
    while end - cursor > max_seconds:
        target = cursor + max_seconds
        candidates = [point for point in _gap_midpoints(vad_segments, cursor, end)
                      if cursor + 0.6 <= point <= target]
        cut = max(candidates) if candidates else target
        points.append((cursor, cut))
        cursor = cut
    points.append((cursor, end))
    return [{"start": float(left), "end": float(right), "spk": segment["spk"],
             "asr_split": True} for left, right in points]


def _scene_summary(times, labels):
    scenes, first = [], 0
    for index in range(1, len(times) + 1):
        if index == len(times) or times[index] - times[index - 1] > SCENE_GAP_SECONDS:
            scenes.append({"start": round(float(times[first]), 2),
                           "end": round(float(times[index - 1] + WIN_S), 2),
                           "labels": sorted({int(label) for label in labels[first:index]})})
            first = index
    return scenes


def _largest_window_gaps(times, labels, limit=12):
    gaps = []
    for index in range(1, len(times)):
        gap = float(times[index] - times[index - 1] - WIN_S)
        if gap > 0:
            gaps.append({"after": round(float(times[index - 1] + WIN_S), 2),
                         "before": round(float(times[index]), 2),
                         "gap": round(gap, 2),
                         "left_label": int(labels[index - 1]),
                         "right_label": int(labels[index])})
    return sorted(gaps, key=lambda item: item["gap"], reverse=True)[:limit]


def _block_candidates(E, times, gap_seconds=BLOCK_GAP_SECONDS):
    """Diagnostic only: local K evidence across long VAD gaps."""
    if not len(times):
        return []
    first, blocks = 0, []
    for index in range(1, len(times) + 1):
        gap = index == len(times) or times[index] - times[index - 1] - WIN_S >= gap_seconds
        if not gap:
            continue
        part = E[first:index]
        candidates, _ = _candidate_clusters(part)
        selected = _select_candidate(candidates)
        blocks.append({
            "start": round(float(times[first]), 2),
            "end": round(float(times[index - 1] + WIN_S), 2),
            "n_windows": int(len(part)),
            "selected_k": int(selected["k"]),
            "candidates": [
                {"k": item["k"], "silhouette": round(float(item["score"]), 4),
                 "objective": round(float(item["objective"]), 4), "sizes": item["sizes"],
                 "eligible": bool(item["eligible"])}
                for item in candidates
            ],
        })
        first = index
    return blocks


def run_claude_lane(wav, models, lane_key, progress=None):
    if lane_key not in ("v11", "v11_3"):
        raise ValueError(f"unknown v11 lane {lane_key}")
    import soundfile as sf
    from claude_a2_v8_contract import segment_contract

    if progress:
        progress("v11：提取密集 CAM++ 窗并评估 1–4 人候选…")
    audio, rate = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = _vad_segs(models["vad"], wav)
    E, times, embedding_errors = _extract_windows(audio, rate, vad_segments, models["spk"])
    if len(E) < 2:
        return {
            "method": METHOD,
            "file": os.path.basename(wav),
            "elapsed": 0.0,
            "diag": {"n_windows": int(len(E)), "n_speakers": 0, "cluster_level": "insufficient"},
            "verdict": "experimental",
            "verdict_reasons": ["v11 窗数不足，无法做多人聚类"],
            "sentence_info": [],
        }

    candidates, _ = _candidate_clusters(E)
    forced_k = 3 if lane_key == "v11_3" else None
    selected = _select_candidate(candidates, forced_k=forced_k)
    labels = selected["labels"].copy()
    if forced_k is None:
        scores = E @ _centroids(E, labels).T
        labels = _stabilize_isolated(labels, scores, times)
    segments = _segments_from_windows(times, labels)
    asr_segments = [piece for segment in segments
                    for piece in _split_for_asr(segment, vad_segments)]

    if progress:
        mode_text = f"固定 {forced_k} 人" if forced_k else f"选择 {selected['k']} 人候选"
        progress(f"v11：{mode_text}，逐段转写 {len(asr_segments)} 段…")
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
            progress(f"v11：逐段转写 {index + 1}/{len(asr_segments)}")

    actual_k = len({item["spk"] for item in sentences})
    reasons = ["v11 多人实验泳道；未替换冻结 v7 基线"]
    if forced_k and not selected["eligible"]:
        reasons.append(f"固定 K={forced_k} 的最小簇仅 {min(selected['sizes'])} 个窗，"
                       "支持不足；第三标签仅作诊断，不能当作已验证说话人")
    if actual_k < selected["k"]:
        reasons.append(f"候选为 {selected['k']} 人，但平滑后只保留 {actual_k} 个标签")
    suspect = sorted({flag for flag in flags if flag in ("empty", "no_speech", "too_fast", "repeated")})
    if suspect:
        reasons.append(f"{sum(flag in suspect for flag in flags)} 段文本可疑（{'/'.join(suspect)}）")
    return {
        "method": METHOD,
        "file": os.path.basename(wav),
        "elapsed": 0.0,
        "diag": {
            "algorithm": "v11_multispeaker",
            "cluster_level": "dense_global_ahc",
            "n_windows": int(len(E)),
            "embedding_errors": embedding_errors,
            "assumed_k": int(selected["k"]),
            "k_mode": "forced" if forced_k else "auto",
            "n_speakers": actual_k,
            "n_clusters": actual_k,
            "k_candidates": [
                {"k": item["k"], "silhouette": round(float(item["score"]), 4),
                 "objective": round(float(item["objective"]), 4), "sizes": item["sizes"],
                 "eligible": bool(item["eligible"])}
                for item in candidates
            ],
            "scenes": _scene_summary(times, labels),
            "largest_vad_gaps": _largest_window_gaps(times, labels),
            "local_block_candidates": _block_candidates(E, times),
            "max_asr_seconds": MAX_ASR_SECONDS,
            "n_asr_splits": sum(bool(item.get("asr_split")) for item in sentences),
            "n_flag_uncertain": sum(flag != "ok" for flag in flags),
        },
        "verdict": "experimental",
        "verdict_reasons": reasons,
        "sentence_info": sentences,
    }
