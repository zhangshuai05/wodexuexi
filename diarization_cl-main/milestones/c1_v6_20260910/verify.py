"""Verify the v6 archive; optionally run the two archived audio cases."""
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
            segs = result["sentence_info"]
            assert result["method"] == "claude-a2-v6-c1"
            assert result["diag"]["n_clusters"] == 2
            assert segs and all(a["start"] < a["end"] for a in segs)
            assert all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:]))
            print("PASS: v6 segment/label check", name,
                  "denoised=", result["denoised"], "segments=", len(segs), flush=True)


if __name__ == "__main__":
    main()
