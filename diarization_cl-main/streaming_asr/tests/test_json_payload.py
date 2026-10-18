"""A numpy scalar in a lane result must not kill the stream.

v7 crashed with "Object of type bool_ is not JSON serializable" because
diag.collapsed was a numpy.bool_. The guard lives at the single serialization
exit (web_server._dumps), so new algorithm versions are covered automatically.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import web_server


def test_encode_handles_numpy_scalars():
    payload = {"t": "done", "result": {
        "diag": {"collapsed": np.bool_(True)},
        "ratio": np.float32(0.5),
        "count": np.int64(3),
    }}
    parsed = json.loads(web_server._dumps(payload))["result"]
    assert parsed["diag"]["collapsed"] is True
    assert parsed["count"] == 3
    assert abs(parsed["ratio"] - 0.5) < 1e-6