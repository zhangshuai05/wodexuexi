# -*- coding: utf-8 -*-
"""A 路线：FRCRN 降噪前置（离线批量）。
用法: python denoise_frcrn.py <wav...> [--outdir DIR]
输出: <outdir>/<base>_frcrn.wav（16k 单声道 float→int16，与原文件等长）
实现：绕过 modelscope pipeline（缺 addict/datasets），直接构造 FRCRN 类 + 加载
pytorch_model.bin。GPU 可用则走 GPU；长音频切块前向，避免单次张量过大后耗时暴涨。
"""
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor_pkgs"))  # vendor addict
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

FRCRN_CFG = dict(complex=True, model_complexity=45, model_depth=14,
                 log_amp=False, padding_mode="zeros",
                 win_len=640, win_inc=320, fft_len=640, win_type="hann")
FRCRN_BIN = os.path.join(r"C:\Users\EDY\.cache\modelscope\models",
                         "iic--speech_frcrn_ans_cirm_16k",
                         "snapshots", "master", "pytorch_model.bin")

_model = None
CHUNK_SECONDS = 15
OVERLAP_SECONDS = 0.5


def get_model():
    global _model
    if _model is None:
        from modelscope.models.audio.ans.frcrn import FRCRN
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = FRCRN(**FRCRN_CFG).to(dev).eval()
        ckpt = torch.load(FRCRN_BIN, map_location="cpu")
        missing, unexpected = _model.load_state_dict(ckpt, strict=False)
        print(f"FRCRN loaded on {dev} (missing={len(missing)}, "
              f"unexpected={len(unexpected)})", flush=True)
    return _model


def _chunked_infer(x, sr, infer):
    """Long-file inference with a short crossfade; short files stay unchanged."""
    chunk = int(CHUNK_SECONDS * sr)
    overlap = int(OVERLAP_SECONDS * sr)
    if len(x) <= chunk:
        return infer(x)[:len(x)]

    step = chunk - overlap
    estimate = np.zeros(len(x), dtype=np.float32)
    weight = np.zeros(len(x), dtype=np.float32)
    for start in range(0, len(x), step):
        stop = min(start + chunk, len(x))
        size = stop - start
        samples = x[start:stop]
        if size < chunk:
            samples = np.pad(samples, (0, chunk - size))
        enhanced = infer(samples)
        if len(enhanced) < size:
            raise ValueError("FRCRN output shorter than input")
        enhanced = enhanced[:size]
        fade = np.ones(size, dtype=np.float32)
        width = min(overlap, size)
        if start:
            fade[:width] = np.linspace(0, 1, width, endpoint=False)
        if stop < len(x):
            fade[-width:] *= np.linspace(1, 0, width, endpoint=False)
        estimate[start:stop] += enhanced * fade
        weight[start:stop] += fade
    return estimate / np.maximum(weight, 1e-8)


def denoise_file(wav, out_path):
    import soundfile as sf
    model = get_model()
    dev = next(model.parameters()).device
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    t0 = time.time()

    def infer(samples):
        xt = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0).to(dev)
        with torch.inference_mode():
            out = model(xt)
        return out[4].squeeze(0).cpu().numpy()  # wav_l2（两级 mask 叠加后的估计）

    est = _chunked_infer(x, sr, infer)
    # 长度对齐 + 防削波
    est = est[:len(x)]
    est = np.clip(est, -1.0, 1.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, (est * 32767).astype(np.int16), sr)
    print(f"{os.path.basename(wav)}: {len(x)/sr:.1f}s -> "
          f"{time.time()-t0:.1f}s ({dev})", flush=True)


def main():
    args = sys.argv[1:]
    outdir = os.path.join(HERE, "output", "denoised")
    wavs = []
    i = 0
    while i < len(args):
        if args[i] == "--outdir" and i + 1 < len(args):
            outdir = args[i + 1]; i += 2
        else:
            wavs.append(args[i]); i += 1
    if not wavs:
        print(__doc__)
        sys.exit(0)
    for wav in wavs:
        base = os.path.basename(wav).rsplit(".", 1)[0]
        try:
            denoise_file(wav, os.path.join(outdir, base + "_frcrn.wav"))
        except Exception as e:
            print(f"[FAILED] {wav}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
