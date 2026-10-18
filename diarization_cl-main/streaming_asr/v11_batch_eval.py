# -*- coding: utf-8 -*-
"""Print a compact, reproducible score table for v11 Merged3Spk traces."""

import json
from hashlib import sha256
from pathlib import Path

from v11_eval import _load_result, evaluate


HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "Audio" / "Merged3Spk"


def main():
    for trace in sorted((HERE / "output").glob("v11_20260917_CallMerge_*.ndjson")):
        result = _load_result(trace)
        wav = AUDIO / result["file"]
        gt = json.loads((AUDIO / f"{wav.stem}.gt.json").read_text(encoding="utf-8"))
        score = evaluate(gt, result)
        print(json.dumps({
            "file": score["file"],
            "method": result["method"],
            "input_sha256": sha256(wav.read_bytes()).hexdigest(),
            "selected_k": result["diag"]["assumed_k"],
            "output_k": result["diag"]["n_speakers"],
            "missing_ref_speakers": score["missing_ref_speakers"],
            "trusted_turn_accuracy": score["scopes"]["asr_unclamped"]["turns"]["accuracy"],
            "all_turn_accuracy": score["scopes"]["all"]["turns"]["accuracy"],
            "wzy_cross_scene": score["cross_scene"].get("wzy"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
