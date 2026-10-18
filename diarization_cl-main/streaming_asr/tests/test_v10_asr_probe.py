import numpy as np
import pytest

from v10_asr_probe import stream_replay


class FakeStreamingModel:
    def __init__(self):
        self.calls = []

    def generate(self, input, **kwargs):
        self.calls.append((len(input), kwargs["is_final"]))
        return [{"text": str(len(self.calls))}]


@pytest.mark.parametrize("length", [19207, 19200])
def test_stream_replay_never_passes_future_audio_and_flushes_tail(length):
    now = [0.0]
    audio = np.arange(length, dtype=np.float32)
    seen = []
    caches = []

    class CausalModel(FakeStreamingModel):
        def generate(self, input, **kwargs):
            # Every sample must have arrived at wall time before inference.
            assert input[-1] + 1 <= round(now[0] * 16000)
            seen.extend(input.tolist())
            caches.append(kwargs["cache"])
            now[0] += 0.01
            return super().generate(input, **kwargs)

    model = CausalModel()
    def sleep(seconds):
        now[0] += seconds

    result = stream_replay(model, audio, clock=lambda: now[0], sleep=sleep)

    assert model.calls[-1][1] is True
    assert sum(final for _, final in model.calls) == 1
    assert seen == audio.tolist()
    assert all(cache is caches[0] for cache in caches)
    assert result["events"][-1]["end_sample"] == length
    assert result["first_partial_ms"] == 610.0
