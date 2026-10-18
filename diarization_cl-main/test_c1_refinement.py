"""Run without loading speech models: python test_c1_refinement.py."""
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

import a2_v2
import claude_a2
from compare_c1 import score


class Speaker:
    def generate(self, input):
        value = float(np.mean(input))
        return [{"spk_embedding": torch.tensor([1 - value, value])}]


def unexpected_centroid_update(*_):
    raise AssertionError("anchored centroids must not be self-updated")


def main():
    for module in (a2_v2, claude_a2):
        assert np.allclose(module._segment_bounds([0.0, 0.6, 1.2], 0, 1, 1.2), (0.0, 0.9))
        assert np.allclose(module._segment_bounds([0.0, 0.6, 3.0], 1, 2, 1.2), (0.9, 1.8))
        assert np.allclose(module._segment_bounds([0.0, 0.6, 3.0], 2, 3, 1.2), (3.0, 4.2))

    weak_flip = np.array([0, 1, 0])
    strong_reply = np.array([0, 1, 0])
    for module in (a2_v2, claude_a2):
        assert module._stabilize_labels(weak_flip, np.array([.4, -.1, .3])).tolist() == [0, 0, 0]
        assert module._stabilize_labels(strong_reply, np.array([.2, -.5, .3])).tolist() == [0, 1, 0]

    # A short run only gets absorbed when its margin is also weak; a confident
    # short reply and any run past min_dur both survive. margins is mutated in
    # place, so each case needs its own arrays.
    times = np.arange(7) * 0.15
    for module in (a2_v2, claude_a2):
        for margin, span_windows, expected in ((.005, 1, 0), (.40, 1, 1),
                                               (.005, 2, 1)):
            labels = np.zeros(7, dtype=int)
            margins = np.full(7, .3)
            labels[3:3 + span_windows] = 1
            margins[3:3 + span_windows] = margin
            stable, _ = module._absorb_slivers(labels, margins, times, 0.6)
            assert stable[3] == expected, (module.__name__, margin, span_windows,
                                           stable.tolist())

    seqw = [0, 0, 1, 1, 0, 0]
    times = np.arange(len(seqw)) * 0.6
    accepted = [(i, i + 1, i % 2) for i in range(5)]
    rejected = [(i, i + 1, i % 2) for i in range(6)]
    for module in (a2_v2, claude_a2):
        assert module.RESCUE_SWITCH_MAX == 1.5
        assert module._rescue_gate(accepted, seqw, times, 1.2) == (4, 2, True)
        assert module._rescue_gate(rejected, seqw, times, 1.2) == (5, 2, False)

    with TemporaryDirectory() as tmp:
        wav = Path(tmp) / "single.wav"
        sf.write(wav, np.zeros(3 * 16000), 16000)
        for module, function in ((a2_v2, a2_v2.a2_segments_v2),
                                 (claude_a2, claude_a2.a2_segments)):
            original_vad = module._vad_segs
            module._vad_segs = lambda *_: [(0, 3)]
            try:
                result = function(str(wav), Speaker(), None)
            finally:
                module._vad_segs = original_vad
            assert len(result) == 1 and result[0]["spk"] == 0, (module.__name__, result)

    sr = 16000
    x = np.repeat([0.0, 1.0, 0.0], 3 * sr)
    outputs = []
    for module, function in ((a2_v2, a2_v2.a2_segments_v2),
                             (claude_a2, claude_a2.a2_segments)):
        assert inspect.signature(function).parameters["merge"].default is False
        original_cs = module.cs
        module.cs = unexpected_centroid_update
        try:
            result = module._dense_rescue(x, sr, [(0, 9)], Speaker(),
                                          centroids=np.eye(2))
        finally:
            module.cs = original_cs
        assert [s[2] for s in result] == [0, 1, 0], result
        assert abs(result[0][1] - 3) <= 0.15
        assert abs(result[1][1] - 6) <= 0.15
        assert all(a[1] <= b[0] for a, b in zip(result, result[1:])), result
        assert all(0 <= a < b <= 9 for a, b, _ in result)
        outputs.append(result)
        gap = module._dense_rescue(np.zeros(9 * sr), sr,
                                  [(0, 2), (7, 9)], Speaker(), centroids=np.eye(2))
        assert len(gap) == 2 and gap[0][1] < 3 and gap[1][0] > 6, gap
        rescued = module._dense_rescue(x, sr, [(0, 9)], Speaker())
        assert rescued is None or all(a[1] <= b[0] for a, b in zip(rescued, rescued[1:])), rescued
        assert module._dense_rescue(x[:1000], sr, [(0, 1)], Speaker(),
                                    centroids=np.eye(2)) is None
    assert outputs[0] == outputs[1], outputs
    truth = {"speakers": ["a", "b"], "n_turns": 2,
             "turns": [{"spk": "a", "text": "hello"}, {"spk": "b", "text": "world"}]}
    sentences = [{"spk": 0, "text": "hello", "start": 0, "end": 1},
                 {"spk": 0, "text": "world", "start": 1, "end": 2}]
    assert score(truth, sentences)["matched"] == 2
    assert score(truth, sentences)["speaker_matched"] == 1
    print("PASS: rescue gate, single-speaker guard, speaker switches, nonoverlap, "
          "silence gaps, short input, parity, label scoring")


if __name__ == "__main__":
    main()
