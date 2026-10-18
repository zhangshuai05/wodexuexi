import numpy as np

import denoise_frcrn


def test_chunked_infer_preserves_length_and_overlap():
    sr = 100
    audio = np.arange(3301, dtype=np.float32)
    calls = []

    def identity(chunk):
        calls.append(len(chunk))
        return chunk.copy()

    output = denoise_frcrn._chunked_infer(audio, sr, identity)
    assert len(calls) > 1
    assert calls[-1] == denoise_frcrn.CHUNK_SECONDS * sr
    assert len(output) == len(audio)
    np.testing.assert_allclose(output, audio, atol=1e-4)
