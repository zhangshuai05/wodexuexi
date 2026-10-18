# -*- coding: utf-8 -*-
"""Frame-level DER between a predicted RTTM and AISHELL-4 reference RTTM.

DER = (missed speech + false alarm + speaker confusion) / total reference speech.

Ported from the v13 spike's throwaway `der.py` after its five self-check cases
were verified by hand.  It lives here because "one DER number" is a deliverable,
and a deliverable's measurement tool cannot live in a scratch directory.

**Read these two conventions before quoting a number from this file.**

1. The denominator is the count of frames where the **reference has any speech**,
   counted once per frame however many speakers are active. Overlapped reference
   speech therefore contributes one frame to the denominator, not one per
   speaker — this is what md-eval does, and what makes a DER here comparable to a
   published number. The spike's `der.py` counted `len(R)` (the number of active
   speakers) instead, so every overlapped frame weighted its denominator up and
   its figures are not directly comparable to md-eval or pyannote.

2. Label mapping is **greedy by overlap**, not Hungarian. pyannote's DER solves
   a bipartite assignment first; this file sorts overlapping (hypothesis,
   reference) pairs by frame count and takes them while both sides are unused.
   With the union convention a merge costs one point of DER per merged frame, in
   `confusion` — the frames where A and B speak and the hypothesis keeps saying
   "X". Nothing lands in `missed` for it, which is the opposite of what the
   multiplicity convention did. A frame where the reference has more than one
   active speaker is scored as correct only when the hypothesis covers all of
   them; a hypothesis that gets one of two is booked as confusion for the frame.

`der()` returns the counts behind the rates; the decomposition satisfies
    der == (missed + false_alarm + confusion) / total_frames
by construction.
"""

import argparse
from pathlib import Path

FRAME = 0.01


def read_rttm(path):
    segments = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start = float(parts[3])
            segments.append({"start": start, "end": start + float(parts[4]),
                             "speaker": parts[7]})
    return segments


def _frames(segments, duration):
    """frame index -> set of speakers active in that frame."""
    n = int(duration / FRAME) + 1
    out = [set() for _ in range(n)]
    for segment in segments:
        a = max(0, int(round(segment["start"] / FRAME)))
        b = min(n, int(round(segment["end"] / FRAME)))
        for i in range(a, b):
            out[i].add(segment["speaker"])
    return out


def _collar_mask(reference, collar, n):
    """True where the frame lies within `collar` seconds of a reference boundary.

    Frames near a boundary are excluded from both sides.  Shrinking the segments
    instead would leave the two sides with different gaps and book the
    difference as false alarm -- worth about 6 DER points at collar=0.25.
    """
    mask = [False] * n
    if collar <= 0:
        return mask
    span = int(round(collar / FRAME))
    for segment in reference:
        for boundary in (segment["start"], segment["end"]):
            centre = int(round(boundary / FRAME))
            for i in range(max(0, centre - span), min(n, centre + span + 1)):
                mask[i] = True
    return mask


def der(reference, predicted, collar=0.0, denominator="union"):
    """Score `predicted` against `reference` over a single pass, frame by frame.

    Accounting is the union-frame convention md-eval uses, not a per-speaker
    multiplicity convention:

        total  = number of frames, outside the collar, where the reference has
                 any speech at all        <- the DER denominator
        misses = reference frames with no hypothesis speech
        FA     = hypothesis speech frames where the reference is silent
        conf   = frames where both have speech but the mapped speakers differ

    A merge then costs once, in `conf`: the 10s where A and B talk and the
    hypothesis says "X" is 10s of confusion, not 10s of miss *plus* 10s of
    false alarm.

    `denominator="speakers"` reproduces the v13 spike's convention instead: the
    total counts one frame per active reference speaker (`len(R)` summed), so an
    overlapped frame weights the denominator up. It exists to line the two
    scorers up against each other and to quote the spike-era numbers in their
    own units -- do not use it for a number that will be compared against
    published DER.
    """
    duration = max([s["end"] for s in reference] + [s["end"] for s in predicted] + [0.0])
    n = int(duration / FRAME) + 1
    ref = _frames(reference, duration)
    hyp = _frames(predicted, duration)
    mask = _collar_mask(reference, collar, n)

    overlap = {}
    for i in range(n):
        if mask[i]:
            continue
        for h in hyp[i]:
            for r in ref[i]:
                overlap[(h, r)] = overlap.get((h, r), 0) + 1
    mapping, used = {}, set()
    for (h, r), _ in sorted(overlap.items(), key=lambda item: -item[1]):
        if h not in mapping and r not in used:
            mapping[h] = r
            used.add(r)

    missed = false_alarm = confusion = total = 0
    for i in range(n):
        if mask[i]:
            continue
        R, H = ref[i], hyp[i]
        if denominator == "speakers":
            # Spike-era accounting: one unit per active reference speaker, and
            # only frames where the two sides disagree are charged (a frame the
            # hypothesis explains is correct whether or not the hypothesis has
            # extra labels on it).
            total += len(R)
            if not R:
                false_alarm += len(H)
            elif R <= {mapping[h] for h in H if h in mapping}:
                pass
            elif not H:
                missed += len(R)
            else:
                confusion += len(R)
            continue
        total += 1 if R else 0
        if not R:
            false_alarm += len(H)
            continue
        mapped_here = {mapping[h] for h in H if h in mapping and mapping[h] in R}
        if R <= mapped_here:
            continue
        if not H:
            missed += 1
        else:
            # ponytail: a partial attribution is charged as one point of
            # confusion per frame regardless of how many reference speakers are
            # uncovered.  md-eval's speaker-time accounting would charge two
            # points when the hypothesis covers one of three simultaneous
            # speakers.  Reach for that only if a table starts depending on it;
            # AISHELL-4 overlap is ~2% of speech, so it moves the mean by
            # hundredths of a point.
            confusion += 1
    divisor = max(1, total)
    return {
        "total_frames": total,
        "missed": missed, "false_alarm": false_alarm, "confusion": confusion,
        "miss_rate": round(missed / divisor, 4),
        "false_alarm_rate": round(false_alarm / divisor, 4),
        "confusion_rate": round(confusion / divisor, 4),
        "der": round((missed + false_alarm + confusion) / divisor, 4),
        "n_ref_speakers": len({s["speaker"] for s in reference}),
        "n_pred_speakers": len({s["speaker"] for s in predicted}),
        "mapping": mapping,
    }


def main():
    ap = argparse.ArgumentParser(description="score predicted RTTMs against reference RTTMs")
    ap.add_argument("--gt", required=True, help="reference .rttm file or directory")
    ap.add_argument("--hyp", required=True, help="predicted .rttm file or directory")
    ap.add_argument("--collar", type=float, default=0.25)
    args = ap.parse_args()
    gt_path, hyp_path = Path(args.gt), Path(args.hyp)
    files = sorted(gt_path.glob("*.rttm")) if gt_path.is_dir() else [gt_path]
    for gt_file in files:
        hyp_file = hyp_path / gt_file.name if hyp_path.is_dir() else hyp_path
        if not hyp_file.exists():
            print(f"  [skip] no prediction for {gt_file.name}")
            continue
        result = der(read_rttm(gt_file), read_rttm(hyp_file), collar=args.collar)
        print(f"{gt_file.stem:20s} DER={result['der'] * 100:6.2f}%  "
              f"miss={result['miss_rate'] * 100:5.2f}% "
              f"fa={result['false_alarm_rate'] * 100:5.2f}% "
              f"conf={result['confusion_rate'] * 100:5.2f}%  "
              f"spk {result['n_pred_speakers']}/{result['n_ref_speakers']}")


if __name__ == "__main__":
    main()
