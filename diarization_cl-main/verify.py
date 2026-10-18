"""Verify copied files; optionally reproduce the two archived live cases."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--audio-dir", type=Path,
                        help="Directory containing Note20260803112329.wav and Note20260803111855.wav")
    args = parser.parse_args()
    if bool(args.models_config) != bool(args.audio_dir):
        parser.error("Supply both --models-config and --audio-dir for inference checks.")
    manifest = json.loads((HERE / "SHA256.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        assert actual == expected, f"Checksum mismatch: {name}"
    print(f"PASS: {len(manifest)} archived files verified", flush=True)
    if args.models_config:
        from run import load_models, run
        models = load_models(args.models_config)
        for name in ("Note20260803112329", "Note20260803111855"):
            result = run(args.audio_dir / (name + ".wav"), models)
            expected = json.loads((HERE / "evidence" / ("c1_api_" + name + ".json")).read_text(encoding="utf-8"))["c1"]
            assert result["method"] == expected["method"]
            assert result["env"]["verdict"] == expected["env"]["verdict"]
            old, new = expected["sentence_info"], result["sentence_info"]
            assert len(old) == len(new), (name, len(old), len(new))
            text_differences = []
            for a, b in zip(old, new):
                assert a["spk"] == b["spk"], (name, a, b)
                assert abs(a["start"] - b["start"]) <= 0.16, (name, a, b)
                assert abs(a["end"] - b["end"]) <= 0.16, (name, a, b)
                if a["text"] != b["text"]:
                    text_differences.append({"expected": a, "actual": b})
            print("PASS: segment/label check", name, "denoised=", result["denoised"],
                  "text_differences=", len(text_differences), flush=True)
            for difference in text_differences:
                print("TEXT_DIFFERENCE", json.dumps(difference, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
