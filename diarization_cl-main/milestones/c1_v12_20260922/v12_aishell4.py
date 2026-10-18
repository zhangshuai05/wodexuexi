# -*- coding: utf-8 -*-
"""v12 phase 1: speaker-embedding A/B on real AISHELL-4 meetings.

The benchmark uses channel 0, ignores the first 10 seconds of spoken IDs, and
extracts only windows covered by exactly one RTTM speaker.  It compares speaker
information before any clustering/backend choice is introduced.
"""

import argparse
import json
import platform
import re
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment


MODEL_IDS = {
    "campplus": "iic/speech_campplus_sv_zh-cn_16k-common",
    "eres2netv2": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
}
DEV_GROUPS = {"L_R003C02", "M_R003C01", "S_R004C01"}
HOLDOUT_GROUPS = {"L_R004C01", "S_R003C01"}


def meeting_group(meeting_id):
    match = re.fullmatch(r"([LMS]_R\d{3})S\d{2}(C\d{2})", meeting_id)
    if not match:
        raise ValueError(f"invalid AISHELL-4 meeting id: {meeting_id}")
    return match.group(1) + match.group(2)


def split_meetings(root, split):
    meetings = sorted(path.stem for path in (Path(root) / "TextGrid").glob("*.rttm"))
    groups = DEV_GROUPS if split == "dev" else HOLDOUT_GROUPS
    selected = [meeting for meeting in meetings if meeting_group(meeting) in groups]
    if split == "pilot":
        selected = ["L_R003S01C02"]
    return selected


def read_rttm(path):
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start = float(parts[3])
            segments.append({"start": start, "end": start + float(parts[4]),
                             "speaker": parts[7]})
    return segments


def pure_windows(segments, window_seconds, per_speaker, skip_seconds=10.0):
    """Return evenly sampled windows where exactly one RTTM speaker is active."""
    by_speaker = {}
    for segment in segments:
        speaker = segment["speaker"]
        start = max(skip_seconds, segment["start"])
        last = segment["end"] - window_seconds
        while start <= last + 1e-9:
            end = start + window_seconds
            active = {item["speaker"] for item in segments
                      if item["start"] < end and item["end"] > start}
            if active == {speaker}:
                by_speaker.setdefault(speaker, []).append((start, end))
            start += max(0.3, window_seconds / 2)
    selected = []
    for speaker, windows in sorted(by_speaker.items()):
        if len(windows) <= per_speaker:
            chosen = windows
        else:
            chosen = [windows[index] for index in
                      np.linspace(0, len(windows) - 1, per_speaker, dtype=int)]
        selected.extend((speaker, start, end) for start, end in chosen)
    return selected


def read_channel0(path, start, end, sample_rate=16000):
    with sf.SoundFile(path) as file:
        if file.samplerate != sample_rate or file.channels != 8:
            raise ValueError(f"expected 16 kHz / 8 channels: {path}")
        file.seek(round(start * sample_rate))
        audio = file.read(round((end - start) * sample_rate), dtype="float32", always_2d=True)
    return audio[:, 0]


def load_model(name, device="cpu"):
    if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
        platform._wmi_query = lambda *args: (_ for _ in ()).throw(OSError("WMI disabled"))
    from funasr import AutoModel
    return AutoModel(model=MODEL_IDS[name], device=device,
                     disable_update=True, disable_pbar=True)


def embedding(model, samples):
    value = model.generate(input=samples)[0].get("spk_embedding")
    if value is None:
        raise RuntimeError("speaker model returned no embedding")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(value).all():
        raise RuntimeError("speaker embedding contains non-finite values")
    return value / (np.linalg.norm(value) + 1e-9)


def pair_scores(vectors, labels, pair_rate=1.0, seed=12):
    """Cosine scores for every same-speaker and different-speaker pair.

    Both classes are subsampled at the *same* rate when pair_rate < 1, so FRR and
    FAR keep a common denominator and their comparison stays meaningful.
    """
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)
    scores = vectors @ vectors.T
    rows, cols = np.triu_indices(len(vectors), k=1)
    values = scores[rows, cols]
    same_mask = labels[rows] == labels[cols]
    same, different = values[same_mask], values[~same_mask]
    if pair_rate < 1.0:
        rng = np.random.default_rng(seed)
        same = _subsample(same, pair_rate, rng)
        different = _subsample(different, pair_rate, rng)
    return same, different


def _subsample(values, pair_rate, rng):
    keep = max(1, int(round(len(values) * pair_rate)))
    if len(values) <= keep:
        return values
    return values[rng.choice(len(values), keep, replace=False)]


def threshold_metrics(same, different, threshold):
    """FRR/FAR at a frozen threshold. `None` when a class has no pairs."""
    if not len(same) or not len(different):
        return None
    frr = float(np.mean(same < threshold))
    far = float(np.mean(different >= threshold))
    return {"threshold": round(float(threshold), 6), "frr": round(frr, 6),
            "far": round(far, 6), "eer": round(max(frr, far), 6)}


def eer(same, different):
    """In-sample EER: the threshold is chosen on the same pairs it is scored on.

    Use it to compare models, not to quote an operating point. For a quotable
    number freeze the threshold on the dev split and score holdout with
    `threshold_metrics` via `--locked-thresholds`.
    """
    if not len(same) or not len(different):
        return None
    thresholds = np.unique(np.concatenate([same, different]))
    best = (1.0, None, None, None)
    for threshold in thresholds:
        frr = float(np.mean(same < threshold))
        far = float(np.mean(different >= threshold))
        candidate = (max(frr, far), float(threshold), frr, far)
        if candidate[0] < best[0]:
            best = candidate
    return {"eer": round(best[0], 6), "threshold": round(best[1], 6),
            "frr": round(best[2], 6), "far": round(best[3], 6)}


def silhouette(vectors, labels):
    groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        return -1.0
    group_of = {label: index for index, label in enumerate(sorted(set(labels)))}
    values = []
    for index in range(len(vectors)):
        distances = 1.0 - vectors[index] @ vectors.T
        own = group_of[labels[index]]
        same = groups[own]
        same = same[same != index]
        intra = float(distances[same].mean())
        other = min(float(distances[group].mean())
                    for label, group in enumerate(groups) if label != own)
        values.append((other - intra) / max(intra, other, 1e-9))
    return float(np.mean(values))


def cluster_metrics(vectors, reference_labels, max_k=10):
    if len(vectors) < 2:
        return None
    true_names = sorted(set(reference_labels))
    true_k = len(true_names)
    # The oracle branch below only fires at k == true_k, so the sweep has to
    # reach it or oracle_k_accuracy would stay None.
    top_k = min(max(max_k, true_k), len(vectors))
    tree = linkage(vectors, method="average", metric="cosine")
    true_index = {name: index for index, name in enumerate(true_names)}
    candidates = []
    oracle_accuracy = None
    for k in range(2, top_k + 1):
        predicted = fcluster(tree, t=k, criterion="maxclust") - 1
        sizes = [int((predicted == label).sum()) for label in range(k)]
        score = silhouette(vectors, predicted)
        candidates.append({"k": k, "silhouette": round(score, 6), "sizes": sizes})
        if k == true_k:
            confusion = np.zeros((true_k, k), dtype=int)
            for truth, guess in zip(reference_labels, predicted):
                confusion[true_index[truth], guess] += 1
            rows, cols = linear_sum_assignment(-confusion)
            oracle_accuracy = float(confusion[rows, cols].sum() / len(reference_labels))
    if not candidates:
        return None
    selected = max(candidates, key=lambda item: item["silhouette"])
    return {"true_k": true_k, "auto_k": selected["k"],
            "oracle_k_accuracy": round(oracle_accuracy, 6), "candidates": candidates}


def file_sha256(path):
    digest = sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def thresholds_from_report(report):
    """Freeze the dev split's tuned thresholds so holdout can be scored at them."""
    return {f"{item['model']}@{item['window_seconds']}": item["metric"]
            for item in report["results"] if item.get("metric")}


def benchmark(root, split, model_names, window_seconds, windows_per_speaker, device,
              pair_rate=1.0, locked_thresholds=None):
    root = Path(root)
    meetings = split_meetings(root, split)
    samples = {duration: [] for duration in window_seconds}
    manifest = []
    for meeting in meetings:
        wav = root / "wav" / f"{meeting}.flac"
        rttm = root / "TextGrid" / f"{meeting}.rttm"
        segments = read_rttm(rttm)
        manifest.append({"meeting": meeting, "group": meeting_group(meeting),
                         "speakers": sorted({item["speaker"] for item in segments}),
                         "wav_sha256": file_sha256(wav), "rttm_sha256": file_sha256(rttm)})
        for duration in window_seconds:
            for speaker, start, end in pure_windows(
                    segments, duration, windows_per_speaker):
                samples[duration].append({"meeting": meeting,
                                          "speaker": f"{meeting_group(meeting)}:{speaker}",
                                          "start": start, "end": end, "wav": str(wav)})

    results = []
    for model_name in model_names:
        model = load_model(model_name, device=device)
        for duration in window_seconds:
            key = f"{model_name}@{duration}"
            if not samples[duration]:
                results.append({"model": model_name, "model_id": MODEL_IDS[model_name],
                                "window_seconds": duration, "embeddings": 0,
                                "speakers": 0, "same_pairs": 0, "different_pairs": 0,
                                "metric": None, "locked_metric": None,
                                "same_mean": None, "different_mean": None,
                                "inference_seconds": 0.0, "audio_rtf": None,
                                "meeting_clusters": {}})
                continue
            vectors, labels, sample_meetings = [], [], []
            started = time.perf_counter()
            for item in samples[duration]:
                vectors.append(embedding(model, read_channel0(
                    item["wav"], item["start"], item["end"])))
                labels.append(item["speaker"])
                sample_meetings.append(item["meeting"])
            seconds = time.perf_counter() - started
            matrix = np.stack(vectors)
            same, different = pair_scores(matrix, labels, pair_rate=pair_rate)
            meeting_clusters = {}
            for meeting in meetings:
                indices = [index for index, value in enumerate(sample_meetings) if value == meeting]
                metrics = cluster_metrics(
                    matrix[indices], [labels[index] for index in indices])
                if metrics is not None:
                    meeting_clusters[meeting] = metrics
            locked = None
            if locked_thresholds and key in locked_thresholds:
                locked = threshold_metrics(same, different,
                                           locked_thresholds[key]["threshold"])
            results.append({
                "model": model_name, "model_id": MODEL_IDS[model_name],
                "window_seconds": duration, "embeddings": len(vectors),
                "speakers": len(set(labels)), "same_pairs": len(same),
                "different_pairs": len(different), "metric": eer(same, different),
                "locked_metric": locked,
                "same_mean": round(float(same.mean()), 6),
                "different_mean": round(float(different.mean()), 6),
                "inference_seconds": round(seconds, 3),
                "audio_rtf": round(seconds / max(1e-9, duration * len(vectors)), 6),
                "meeting_clusters": meeting_clusters,
            })
        del model
    return {"schema": "v12-embedding-ab-v1", "split": split, "channel": 0,
            "skip_seconds": 10.0, "pair_rate": pair_rate, "meetings": meetings,
            "manifest": manifest, "windows_per_speaker": windows_per_speaker,
            "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    parser.add_argument("--split", choices=("pilot", "dev", "holdout"), default="pilot")
    parser.add_argument("--models", default="campplus,eres2netv2")
    parser.add_argument("--windows", default="0.6,1.2,2.0")
    parser.add_argument("--windows-per-speaker", type=int, default=8)
    parser.add_argument("--pair-rate", type=float, default=1.0,
                        help="shared same/different pair subsample rate; 1.0 = all pairs")
    parser.add_argument("--locked-thresholds",
                        help="dev summary JSON from an earlier run; scores this split at "
                             "the thresholds frozen there instead of re-tuning")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    locked = None
    if args.locked_thresholds:
        locked = thresholds_from_report(json.loads(
            Path(args.locked_thresholds).read_text(encoding="utf-8")))
    report = benchmark(args.root, args.split, args.models.split(","),
                       [float(value) for value in args.windows.split(",")],
                       args.windows_per_speaker, args.device,
                       pair_rate=args.pair_rate, locked_thresholds=locked)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
