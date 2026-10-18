"""Portable entry point for the frozen v9 algorithm and DSH FRCRN."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
# Set this before importing the unchanged snapshots with legacy cache defaults.
os.environ.setdefault("MODELSCOPE_CACHE", str(HERE / ".cache/modelscope"))


def resolve_models(config):
    config = Path(config).resolve()
    values = json.loads(config.read_text(encoding="utf-8"))
    paths = {}
    for key in ("vad", "spk", "asr", "frcrn"):
        path = Path(values[key]).expanduser()
        path = (config.parent / path).resolve() if not path.is_absolute() else path
        if not path.is_dir():
            raise FileNotFoundError(f"{key} model directory does not exist: {path}")
        paths[key] = path
    if not (paths["frcrn"] / "pytorch_model.bin").is_file():
        raise FileNotFoundError(paths["frcrn"] / "pytorch_model.bin")
    return paths


def load_models(config, device=None):
    paths = resolve_models(config)
    import denoise_frcrn
    from funasr import AutoModel
    denoise_frcrn.FRCRN_BIN = str(paths["frcrn"] / "pytorch_model.bin")
    denoise_frcrn._model = None
    options = {"disable_update": True, "disable_pbar": True}
    if device:
        options["device"] = device
    return {name: AutoModel(model=str(paths[name]), **options)
            for name in ("vad", "spk", "asr")}


def run(wav, models, denoise="auto"):
    if denoise not in ("auto", "on", "off"):
        raise ValueError("denoise must be auto, on, or off")
    wav = Path(wav).resolve()
    if not wav.is_file():
        raise FileNotFoundError(wav)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install the ffmpeg system package and add it to PATH.")
    import claude_a2
    import denoise_frcrn
    from env_detect import detect_env
    with tempfile.TemporaryDirectory(prefix="c1_v9_") as temp:
        normalized = str(Path(temp) / "input_16k.wav")
        subprocess.run([ffmpeg, "-y", "-i", str(wav), "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", normalized], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        env = detect_env(normalized, models["vad"])
        enhanced = denoise == "on" or (denoise == "auto" and env["verdict"] == "\u5608\u6742")
        effective = normalized
        if enhanced:
            effective = str(Path(temp) / "input_frcrn.wav")
            # Fail visibly if denoising fails; do not silently change the pipeline.
            denoise_frcrn.denoise_file(normalized, effective)
        result = claude_a2.run_claude_lane(effective, models, "v9", progress=print)
        result.update(file=wav.name, env=env, denoise_mode=denoise, denoised=enhanced)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--models-config", required=True, type=Path)
    parser.add_argument("--denoise", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--device", help="FunASR device, e.g. cpu or cuda:0; FRCRN auto-selects CUDA if available")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "output" / (args.audio.stem + ".json")
    if output.resolve() == args.audio.resolve():
        parser.error("Output must not overwrite the input audio.")
    result = run(args.audio, load_models(args.models_config, args.device), args.denoise)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", output.resolve())


if __name__ == "__main__":
    main()
