# -*- coding: utf-8 -*-
"""VAD 阈值探针 v2：改写本地 VAD 模型 config.yaml 的 speech_noise_thres 并关闭动态静音调度"""
import sys, os, re

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
VAD_LOCAL = os.path.join(HERE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch",
                         "snapshots", "master")


def set_thres(t):
    cfg = os.path.join(VAD_LOCAL, "config.yaml")
    with open(cfg, encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r"speech_noise_thres: [0-9.]+", f"speech_noise_thres: {t}", txt)
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(txt)


def probe(wav, thres_list, dynamic_silence=False):
    from funasr import AutoModel
    import soundfile as sf
    dur = len(sf.read(wav)[0]) / 16000
    print(f"file={os.path.basename(wav)} dur={dur:.1f}s", flush=True)
    for t in thres_list:
        set_thres(t)
        m = AutoModel(model=VAD_LOCAL, disable_update=True, disable_pbar=True)
        res = m.generate(input=wav, dynamic_silence=dynamic_silence)
        v = res[0].get("value") or []
        tot = sum(e - s for s, e in v) / 1000.0
        print(f"  thres={t} dynamic_silence={dynamic_silence}: "
              f"segs={len(v)} speech_total={tot:.1f}s", flush=True)


if __name__ == "__main__":
    wavs = sys.argv[1:]
    thres = [0.6, 0.7, 0.8, 0.9, 0.95]
    for w in wavs:
        probe(w, thres, dynamic_silence=False)
