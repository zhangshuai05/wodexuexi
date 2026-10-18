# -*- coding: utf-8 -*-
"""Time-only evaluator for the v11 multi-speaker experiment.

The score never reads ASR text. One Hungarian label mapping is selected for the
full file from trusted GT rows, then held fixed for every scene.
"""

import argparse
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment


def _valid(item):
    return float(item.get("end", 0)) > float(item.get("start", 0))


def _merge(intervals):
    merged = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in _merge(intervals))


def _intersection(left, right):
    left, right = _merge(left), _merge(right)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _contains(intervals, point):
    return any(start <= point < end for start, end in intervals)


def _sentences(result):
    return [sentence for sentence in result.get("sentence_info", [])
            if "spk" in sentence and _valid(sentence)]


def _quality(turn):
    return "trusted" if turn.get("q") == "asr" and not turn.get("clamped") else "gap_or_clamped"


def _by_speaker(turns):
    speakers = sorted({turn["spk"] for turn in turns})
    return {speaker: _merge((turn["start"], turn["end"])
                            for turn in turns if turn["spk"] == speaker)
            for speaker in speakers}


def _by_label(sentences):
    labels = sorted({int(sentence["spk"]) for sentence in sentences})
    return {label: _merge((sentence["start"], sentence["end"])
                          for sentence in sentences if int(sentence["spk"]) == label)
            for label in labels}


def _mapping(turns, sentences, speakers, labels):
    reference = _by_speaker(turns)
    predicted = _by_label(sentences)
    matrix = [[_intersection(reference[speaker], predicted[label]) for label in labels]
              for speaker in speakers]
    size = max(len(speakers), len(labels))
    padded = [[0.0] * size for _ in range(size)]
    for row, values in enumerate(matrix):
        padded[row][:len(values)] = values
    rows, cols = linear_sum_assignment([[-value for value in row] for row in padded])
    label_to_speaker = {}
    for row, col in zip(rows, cols):
        if row < len(speakers) and col < len(labels) and matrix[row][col] > 0:
            label_to_speaker[labels[col]] = speakers[row]
    return matrix, label_to_speaker


def _turn_counts(turns, predicted, mapping):
    result = {"turns": len(turns), "correct": 0, "wrong": 0, "miss": 0}
    for turn in turns:
        overlaps = {label: _intersection([(turn["start"], turn["end"])], intervals)
                    for label, intervals in predicted.items()}
        active = [label for label, seconds in overlaps.items() if seconds > 0]
        if not active:
            result["miss"] += 1
        elif any(mapping.get(label) == turn["spk"] for label in active):
            result["correct"] += 1
        else:
            result["wrong"] += 1
    result["accuracy"] = round(result["correct"] / max(1, result["turns"]), 4)
    return result


def _speaker_metrics(turns, sentences, mapping):
    reference = _by_speaker(turns)
    predicted = _by_label(sentences)
    reverse = {speaker: label for label, speaker in mapping.items()}
    all_predicted = _merge(interval for intervals in predicted.values() for interval in intervals)
    output = {}
    for speaker, ref in reference.items():
        mapped = predicted.get(reverse.get(speaker), [])
        boundaries = sorted({point for intervals in [ref, *predicted.values()]
                             for interval in intervals for point in interval})
        correct = miss = confusion = 0.0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            midpoint = (start + end) / 2
            if not _contains(ref, midpoint):
                continue
            active = [label for label, intervals in predicted.items() if _contains(intervals, midpoint)]
            if not active:
                miss += end - start
            elif _contains(mapped, midpoint):
                correct += end - start
            else:
                confusion += end - start
        ref_seconds = _duration(ref)
        output[speaker] = {
            "ref_seconds": round(ref_seconds, 3),
            "correct_seconds": round(correct, 3),
            "miss_seconds": round(miss, 3),
            "confusion_seconds": round(confusion, 3),
            "coverage": round(correct / max(ref_seconds, 1e-9), 4),
            "mapped_label": reverse.get(speaker),
        }
    reference_union = _merge(interval for intervals in reference.values() for interval in intervals)
    outside = max(0.0, _duration(all_predicted) - _intersection(all_predicted, reference_union))
    return output, round(outside, 3)


def _scope(turns, sentences, mapping):
    predicted = _by_label(sentences)
    speaker_metrics, outside = _speaker_metrics(turns, sentences, mapping)
    return {
        "turns": _turn_counts(turns, predicted, mapping),
        "speaker_metrics": speaker_metrics,
        "predicted_outside_annotated_seconds": outside,
    }


def _dominant_label(turns, sentences):
    labels = _by_label(sentences)
    reference = _merge((turn["start"], turn["end"]) for turn in turns)
    votes = {label: _intersection(reference, intervals) for label, intervals in labels.items()}
    return max(votes, key=votes.get) if votes and max(votes.values()) > 0 else None


def evaluate(gt, result):
    sentences = _sentences(result)
    turns = [turn for turn in gt.get("turns", []) if _valid(turn)]
    speakers = sorted({turn["spk"] for turn in turns})
    labels = sorted({int(sentence["spk"]) for sentence in sentences})
    trusted = [turn for turn in turns if _quality(turn) == "trusted"]
    mapping_turns = trusted if {turn["spk"] for turn in trusted} == set(speakers) else turns
    matrix, mapping = _mapping(mapping_turns, sentences, speakers, labels)
    scope_turns = {
        "all": turns,
        "asr_unclamped": trusted,
        "gap_or_clamped": [turn for turn in turns if _quality(turn) == "gap_or_clamped"],
    }
    scopes = {name: _scope(items, sentences, mapping) for name, items in scope_turns.items()}
    scenes = {}
    for scene in sorted({turn.get("seg") for turn in turns if turn.get("seg")}):
        scene_turns = [turn for turn in turns if turn.get("seg") == scene]
        scene_speakers = sorted({turn["spk"] for turn in scene_turns})
        _, local = _mapping(scene_turns, sentences, scene_speakers, labels)
        scenes[scene] = {
            "global": _scope(scene_turns, sentences, mapping),
            "local_oracle": _scope(scene_turns, sentences, local),
            "local_mapping": {str(label): speaker for label, speaker in local.items()},
        }
    shared = {}
    for speaker in speakers:
        by_scene = {scene: _dominant_label(
            [turn for turn in turns if turn.get("seg") == scene and turn["spk"] == speaker], sentences)
            for scene in scenes}
        present = [label for label in by_scene.values() if label is not None]
        if len(by_scene) > 1:
            shared[speaker] = {
                "dominant_labels": by_scene,
                "status": "consistent" if len(present) == len(by_scene) and len(set(present)) == 1
                else "changed_label" if len(present) == len(by_scene) else "unresolved",
            }
    return {
        "schema": "v11-time-diar-v1",
        "file": gt.get("name"),
        "input": {"method": result.get("method"), "gt_speakers": speakers,
                  "predicted_labels": labels},
        "mapping_scope": "asr_unclamped" if mapping_turns is trusted else "all",
        "global_mapping": {str(label): speaker for label, speaker in mapping.items()},
        "missing_ref_speakers": [speaker for speaker in speakers if speaker not in mapping.values()],
        "unmapped_output_labels": [label for label in labels if label not in mapping],
        "counts": {"reference": len(speakers), "predicted_active": len(labels),
                   "shortfall": max(0, len(speakers) - len(labels))},
        "overlap_seconds": [[round(value, 3) for value in row] for row in matrix],
        "scopes": scopes,
        "by_source_segment": scenes,
        "cross_scene": shared,
    }


def _load_result(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
        done = [event["result"] for event in events if event.get("t") == "done"]
        if not done:
            raise ValueError("NDJSON 中没有 done 结果")
        return done[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="v11 导出的 JSON 或 NDJSON")
    parser.add_argument("ground_truth", help="Merged3Spk 的 .gt.json")
    args = parser.parse_args()
    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(gt, _load_result(args.result)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
