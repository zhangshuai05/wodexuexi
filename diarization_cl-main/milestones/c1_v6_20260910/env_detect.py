# -*- coding: utf-8 -*-
"""自动环境检测（零训练，全离线）：
NIST STNR 风格分位数法 + FSMN-VAD 辅助分离语音帧/噪声帧。
- 噪声底 = 非语音帧能量 P50（dB）
- 语音电平 = 语音帧能量 P90（dB）
- SNR ≈ 语音P90 - 噪声底P50
- verdict：SNR < SNR_TH 判"嘈杂"→ 触发降噪前置（FRCRN）；否则"安静"。
无 VAD 段时退化为全文件能量分位数（P90-P10）。
文献：Kim & Stern, "Robust SNR Estimation Based on Waveform Amplitude
Distribution Analysis" (Interspeech 2008, WADA-SNR)；NIST STNR 分位数法。
用法: python env_detect.py <wav...> [--vad-model PATH]
"""
import os, sys
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
VAD_LOCAL = os.path.join(HERE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                         "snapshots", "master")
FRAME = 0.01   # 10ms 帧
NOISE_TH = -70.0  # 噪声底（非语音帧能量 P50，dB）高于此值判"嘈杂"
# 注：SNR 定义（语音P90-噪声底P50）会因静音帧拉低噪声底而虚高，不作主判据；
# 实测 26 段录音：嘈杂组噪声底 -43~-68dB、安静组 -72~-84dB，-70dB 分界 100% 干净。


def _frame_db(x, sr):
    hop = int(FRAME * sr)
    rms = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2) + 1e-12)
                    for i in range(0, len(x) - hop, hop)])
    return 20 * np.log10(rms)


def detect_env(wav, vad=None):
    """返回 dict: snr_db/noise_db/speech_db/verdict/method/n_vad_frames。"""
    x, sr = sf.read(wav)
    if x.ndim > 1:
        x = x[:, 0]
    db = _frame_db(x, sr)
    vsegs = []
    if vad is not None:
        try:
            res = vad.generate(input=wav, dynamic_silence=False)
            vsegs = [(float(it[0]) / 1000.0, float(it[1]) / 1000.0)
                     for r in (res if isinstance(res, list) else [res])
                     for it in r.get("value", [])]
        except Exception:
            vsegs = []
    if vsegs:
        speech_mask = np.zeros(len(db), dtype=bool)
        for a, b in vsegs:
            speech_mask[int(a / FRAME):int(b / FRAME) + 1] = True
        if speech_mask.sum() >= 5 and (~speech_mask).sum() >= 5:
            noise_db = float(np.percentile(db[~speech_mask], 50))
            speech_db = float(np.percentile(db[speech_mask], 90))
            snr = speech_db - noise_db
            return {"snr_db": round(snr, 1), "noise_db": round(noise_db, 1),
                    "speech_db": round(speech_db, 1),
                    "verdict": "嘈杂" if noise_db > NOISE_TH else "安静",
                    "method": "vad+percentile", "n_vad_frames": int(speech_mask.sum())}
    # 退化：全文件分位数
    noise_db = float(np.percentile(db, 10))
    speech_db = float(np.percentile(db, 90))
    snr = speech_db - noise_db
    return {"snr_db": round(snr, 1), "noise_db": round(noise_db, 1),
            "speech_db": round(speech_db, 1),
            "verdict": "嘈杂" if noise_db > NOISE_TH else "安静",
            "method": "percentile_fallback", "n_vad_frames": 0}


def main():
    args = sys.argv[1:]
    vad_model = VAD_LOCAL
    wavs = []
    i = 0
    while i < len(args):
        if args[i] == "--vad-model" and i + 1 < len(args):
            vad_model = args[i + 1]; i += 2
        else:
            wavs.append(args[i]); i += 1
    if not wavs:
        print(__doc__)
        sys.exit(0)
    from funasr import AutoModel
    vad = AutoModel(model=vad_model, disable_update=True, disable_pbar=True)
    print(f"{'file':<30} {'snr':>6} {'noise':>7} {'speech':>7} {'verdict':>4} "
          f"{'method':<20}")
    for wav in wavs:
        d = detect_env(wav, vad)
        print(f"{os.path.basename(wav):<30} {d['snr_db']:>6} {d['noise_db']:>7} "
              f"{d['speech_db']:>7} {d['verdict']:>4} {d['method']:<20}")


if __name__ == "__main__":
    main()
