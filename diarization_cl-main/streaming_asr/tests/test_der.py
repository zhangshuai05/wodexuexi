"""Self-checks for der.py.

The v13 spike's first DER implementation reported 35.68% where the truth was
22.79% -- a wrong number that raised no error, on the metric the whole roadmap
rests on.  These cases exist so that a rewrite of `der()` fails loudly instead.

The convention these cases pin down (see the module docstring): the denominator
is reference **speech frames**, counted once per frame, and a frame where two
reference speakers talk at once and the hypothesis labels it with one speaker is
confusion -- one point, not miss *plus* false alarm.
"""

import pytest

from der import der, read_rttm


def segment(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def score(reference, predicted, collar=0.0):
    return der(reference, predicted, collar=collar)["der"]


def test_identical_is_zero():
    assert score([segment(0, 10, "A")], [segment(0, 10, "A")]) == 0.0


def test_half_the_reference_missing_is_half():
    reference = [segment(0, 10, "A"), segment(10, 20, "B")]
    assert score(reference, [segment(0, 10, "A")]) == 0.5


def test_swapped_labels_map_back_to_zero():
    reference = [segment(0, 10, "A"), segment(10, 20, "B")]
    predicted = [segment(0, 10, "B"), segment(10, 20, "A")]
    assert score(reference, predicted) == 0.0


def test_speech_past_the_end_of_the_reference_is_false_alarm():
    # Scope is the whole predicted span, so a prediction that keeps talking past
    # the last reference segment is charged for it.  Half the hypothesis frames
    # have no reference speech under them.
    reference = [segment(0, 10, "A")]
    predicted = [segment(0, 10, "A"), segment(10, 20, "Z")]
    result = der(reference, predicted, collar=0.0)
    assert result["false_alarm"] == 1000
    assert result["total_frames"] == 1000
    assert result["der"] == 1.0


def test_collar_does_not_change_a_perfect_match():
    assert score([segment(0, 10, "A")], [segment(0, 10, "A")], collar=0.25) == 0.0


def test_overlapping_reference_counts_one_frame_in_the_denominator():
    # Two speakers talk over each other for 10s and the prediction matches both,
    # so nothing is wrong -- and the denominator is 1000 frames, not 2000.
    reference = [segment(0, 10, "A"), segment(0, 10, "B")]
    predicted = [segment(0, 10, "X"), segment(0, 10, "Y")]
    result = der(reference, predicted, collar=0.0)
    assert result["der"] == 0.0
    assert result["total_frames"] == 1000


def test_single_label_over_two_speakers_is_confusion():
    # A and B talk over each other; the hypothesis emits one label covering both.
    # That is a merge: one point of DER per frame, booked as confusion -- not as
    # miss plus false alarm.  This is the case the spike's denominator convention
    # got wrong (it reported 2.0 here).
    reference = [segment(0, 10, "A"), segment(0, 10, "B")]
    predicted = [segment(0, 10, "X")]
    result = der(reference, predicted, collar=0.0)
    assert (result["missed"], result["false_alarm"]) == (0, 0)
    assert result["confusion"] == 1000
    assert result["der"] == 1.0


def test_redundant_prediction_speaker_is_not_charged():
    # One reference speaker and the hypothesis emits both "A" (correct) and "Z"
    # on the same speech.  Every reference frame is covered by its mapped label,
    # so nothing is wrong -- an extra label on already-explained speech is
    # redundant, not phantom.  A lane that double-labels one speaker this way is
    # invisible to DER; it is a transcript-quality problem, not a DER problem.
    reference = [segment(0, 10, "A")]
    predicted = [segment(0, 10, "A"), segment(0, 10, "Z")]
    result = der(reference, predicted, collar=0.0)
    assert (result["missed"], result["false_alarm"], result["confusion"]) == (0, 0, 0)
    assert result["der"] == 0.0


def test_three_way_overlap_with_one_covered_is_confusion():
    # A, B and C all talk over each other for 10s; the hypothesis labels the span
    # with two fresh labels that the mapping pins to two of the three.  Every
    # frame has an uncovered reference speaker, so every frame is a point of
    # confusion.  (See the `ponytail:` note in der.py: two uncovered speakers
    # still count once here, unlike md-eval's per-speaker accounting.)
    reference = [segment(0, 10, "A"), segment(0, 10, "B"), segment(0, 10, "C")]
    predicted = [segment(0, 10, "X"), segment(0, 10, "Y")]
    result = der(reference, predicted, collar=0.0)
    assert (result["missed"], result["false_alarm"]) == (0, 0)
    assert result["confusion"] == 1000
    assert result["der"] == 1.0


def test_merge_across_two_turns_costs_one_der_point_of_two():
    # Two turns, one speaker each, one hypothesis label spanning both.  The
    # mapping pins "X" to whichever turn it overlaps more, so half the reference
    # speech is right and half is confusion.
    reference = [segment(0, 10, "A"), segment(10, 20, "B")]
    predicted = [segment(0, 20, "X")]
    result = der(reference, predicted, collar=0.0)
    assert result["total_frames"] == 2000
    assert result["confusion"] == 1000
    assert result["der"] == 0.5


def test_collar_excludes_frames_near_reference_boundaries():
    # Reference A runs 0-10; the prediction puts the switch 0.2s early.  With no
    # collar that boundary error is charged; a 0.25s collar forgives it.
    reference = [segment(0, 10, "A"), segment(10, 20, "B")]
    predicted = [segment(0, 9.8, "A"), segment(9.8, 20, "B")]
    assert score(reference, predicted) > 0.0
    assert score(reference, predicted, collar=0.25) == 0.0


def test_predicting_silence_everywhere_is_full_miss():
    assert score([segment(0, 10, "A")], []) == 1.0


def test_read_rttm_ignores_non_speaker_lines(tmp_path):
    path = tmp_path / "sample.rttm"
    path.write_text("SPEAKER m 1 1.50 2.25 <NA> <NA> spk1 <NA> <NA>\n"
                    "# a comment\n"
                    "SPEAKER m 1 4.00 1.00 <NA> <NA> spk2 <NA> <NA>\n",
                    encoding="utf-8")
    segments = read_rttm(path)
    assert [(s["start"], s["end"], s["speaker"]) for s in segments] == [
        (1.5, 3.75, "spk1"), (4.0, 5.0, "spk2")]


@pytest.mark.parametrize("collar", [0.0, 0.05, 0.25])
def test_decomposition_sums_to_der(collar):
    reference = [segment(0, 4, "A"), segment(3, 7, "B"), segment(9, 12, "C")]
    predicted = [segment(0.2, 4.1, "p"), segment(3.9, 8, "q")]
    result = der(reference, predicted, collar=collar)
    total = result["missed"] + result["false_alarm"] + result["confusion"]
    assert result["der"] == round(total / max(1, result["total_frames"]), 4)
