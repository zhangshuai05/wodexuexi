from v11_eval import evaluate


def test_v11_eval_counts_a_missing_third_speaker_as_wrong():
    gt = {"name": "three", "turns": [
        {"spk": "a", "start": 0.0, "end": 1.0, "seg": "A", "q": "asr"},
        {"spk": "b", "start": 1.0, "end": 2.0, "seg": "A", "q": "asr"},
        {"spk": "c", "start": 2.0, "end": 3.0, "seg": "B", "q": "asr"},
    ]}
    result = {"sentence_info": [
        {"start": 0.0, "end": 1.0, "spk": 0},
        {"start": 1.0, "end": 3.0, "spk": 1},
    ]}
    score = evaluate(gt, result)
    assert len(score["missing_ref_speakers"]) == 1
    assert score["scopes"]["all"]["turns"]["correct"] == 2
    assert score["scopes"]["all"]["turns"]["wrong"] == 1


def test_v11_eval_counts_a_prediction_that_ends_early_as_miss():
    gt = {"name": "early", "turns": [
        {"spk": "a", "start": 0.0, "end": 2.0, "q": "asr"},
        {"spk": "b", "start": 2.0, "end": 4.0, "q": "asr"},
    ]}
    result = {"sentence_info": [{"start": 0.0, "end": 2.0, "spk": 7}]}
    score = evaluate(gt, result)
    assert score["scopes"]["all"]["speaker_metrics"]["b"]["miss_seconds"] == 2.0
