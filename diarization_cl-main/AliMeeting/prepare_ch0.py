# -*- coding: utf-8 -*-
"""远场 8 通道 → ch0 单声道派生文件。

为什么要这一步：主仓生产链路里，波形由 `sf.read(wav)[:, 0]` 取 ch0（单声道化），
但 VAD 走 FunASR 自己的文件读取（`v11._vad_segs` → `vad.generate(input=wav)`）。
两条路对 8 通道文件的处理是否一致**没有验证过**。与其赌，不如先把 ch0 落成
单声道文件，让"只读 ch0"这件事在整条链路上无歧义。

     python prepare_ch0.py
"""

from common import ALI_CH0, ALI_FAR


def main():
    import soundfile as sf

    audio_dir = ALI_FAR / "audio_dir"
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"没找到远场音频：{audio_dir}")

    ALI_CH0.mkdir(parents=True, exist_ok=True)
    for path in wavs:
        info = sf.info(path)
        target = ALI_CH0 / path.name
        if target.exists() and target.stat().st_size > 0:
            print(f"  [skip] {path.stem} 已存在")
            continue
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] != info.channels:
            raise SystemExit(f"{path.name}: 读到的通道数 {audio.shape[1]} "
                             f"与 info 的 {info.channels} 不符")
        sf.write(target, audio[:, 0], rate, subtype="PCM_16")
        peak = float(abs(audio[:, 0]).max())
        print(f"  {path.stem}: {info.channels}ch -> 1ch  "
              f"{len(audio) / rate:7.1f}s  峰值={peak:.4f}")

    print(f"\nch0 单声道已就绪 -> {ALI_CH0}")


if __name__ == "__main__":
    main()
