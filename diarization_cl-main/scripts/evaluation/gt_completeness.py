# -*- coding: utf-8 -*-
"""GT 完整性校验：gt_ts_diag 的每个文件，用 VAD 检测"GT 声称静音但实际有语音"时长。
若巨大（如 >3s）= GT 漏标远场轻声，DER 在该文件不可信（>100% 的根源）。
输出每文件的：VAD 总语音时长、GT 语音时长、GT-静音区 VAD 重合时长、漏标率。
"""
import os, sys, json
import soundfile as sf

DSH = r"D:\Deepseek_herness\2026-08\diarization"
sys.path.insert(0, DSH)
os.chdir(DSH)

from a2_port import load_models  # noqa: E402


def main():
    gts = json.load(open(os.path.join(DSH, "output", "gt_ts_diag.json"),
                         encoding="utf-8"))["files"]
    vad, spk = load_models()
    print(f"{'file':<22} {'dur':>6} {'VAD语音':>7} {'GT语音':>7} "
          f"{'GT静音区VAD':>12} {'漏标秒':>7}")
    print("-" * 66)
    recs = []
    for base in sorted(gts):
        turns = gts[base]["turns"]
        wav = None
        for cand in ("Audio/Device116/%s.wav", "Audio/Device118/%s.wav"):
            p = cand % base
            if os.path.exists(p):
                wav = p
                break
        if wav is None:
            print(base, "NO WAV")
            continue
        x, sr = sf.read(wav)
        x = x[:, 0] if x.ndim > 1 else x
        dur = len(x) / sr
        res = vad.generate(input=wav, dynamic_silence=False)
        vsegs = [(float(it[0]) / 1000., float(it[1]) / 1000.)
                 for r in (res if isinstance(res, list) else [res])
                 for it in r.get("value", [])]
        gtspeech = sum(tr.get("end", 0) - tr.get("start", 0)
                       for tr in turns if tr.get("start") is not None)
        vad_total = sum(b - a for a, b in vsegs)
        # GT 静音区（不在任何 GT turn 内）与 VAD 段的重合时长
        miss = 0.0
        for a, b in vsegs:
            covin = 0.0
            for tr in turns:
                if tr.get("start") is None:
                    continue
                covin += max(0.0, min(b, tr["end"]) - max(a, tr["start"]))
            miss += max(0.0, (b - a) - covin)
        missrate = miss / max(1e-9, vad_total)
        recs.append((base, dur, vad_total, gtspeech, miss, missrate))
        print(f"{base:<22} {dur:>6.1f} {vad_total:>7.1f} {gtspeech:>7.1f} "
              f"{miss:>12.1f} {missrate:>7.1%}")
    print("\n=== 剔除顺序（漏标率从高到低，判断 GT 可信度） ===")
    for base, dur, vt, gt, miss, missrate in sorted(recs, key=lambda r: -r[5]):
        print(f"  {base:<22} 漏标率={missrate:>6.1%}  GT静音区VAD={miss:>5.1f}s")


if __name__ == "__main__":
    main()
