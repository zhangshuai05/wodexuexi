# -*- coding: utf-8 -*-
"""
核心模块 1：流式 VAD + 流式 Paraformer 双写引擎
适配 web_server.py 的本地模型路径与 Windows WMI 补丁
"""
import os
import sys
import wave
import platform
import numpy as np
from typing import List, Dict, Optional

# ---- 1. Windows Python 3.12 WMI 防卡死补丁（源自 web_server.py）----
def _no_wmi(*args):
    raise OSError("WMI unavailable in this service")

if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
    platform._wmi_query = _no_wmi

from funasr import AutoModel

# ---- 2. 对齐本地 ModelScope 缓存路径（源自 web_server.py / claude_a2.py）----
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
MODEL_ROOT = os.environ.get("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
if not MODEL_ROOT.endswith("models"):
    MODEL_ROOT_MODELS = os.path.join(MODEL_ROOT, "models")
else:
    MODEL_ROOT_MODELS = MODEL_ROOT

VAD_LOCAL = os.path.join(
    PROJECT_ROOT, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master"
)
if not os.path.isdir(VAD_LOCAL):
    VAD_LOCAL = "fsmn-vad"

STREAM_ASR_LOCAL = os.path.join(
    MODEL_ROOT_MODELS,
    "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "snapshots",
    "v2.0.4",
)
if not os.path.isdir(STREAM_ASR_LOCAL):
    STREAM_ASR_LOCAL = "paraformer-zh-streaming"


class StreamingSession:
    """
    流式会议转写会话状态机：
    - 边推流边落盘 16kHz 单声道 session_audio.wav
    - 严格按采样点计数 (samples // 16) 锁定每句话的物理 [start_ms, end_ms]
    - 输出 asr_partial（前端渲染斜体）与 asr_final（触发 DeepSeek 异步纠错）
    """

    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2

    def __init__(
        self,
        session_wav_path: str,
        asr_model: Optional[AutoModel] = None,
        vad_model: Optional[AutoModel] = None,
        max_end_silence_time: int = 1000,  # 2026-10-17 拍板：1000ms（450 会把 ASR 切成碎片）
        max_utt_duration_ms: int = 12000,  # 单句超 12 秒遇换气即切分
        asr_reset_mode: str = "per_utt",   # per_utt=策略A(每句清cache) | continuous=策略B(跨句保留)
        reset_silence_ms: int = 2000,      # 策略B 下静音超过此值才重置 cache
        device: str = "cpu",
    ):
        self.session_wav_path = session_wav_path
        self.max_utt_duration_ms = max_utt_duration_ms
        self.max_end_silence_time = max_end_silence_time
        # 注入的 VAD（如 get_models 的）构造期没带 max_end_silence_time，只能调用期补；
        # 自建的已在构造期设好，不必重复传。
        self._vad_end_silence_in_call = vad_model is not None
        if asr_reset_mode not in ("per_utt", "continuous"):
            raise ValueError("asr_reset_mode 只支持 per_utt/continuous，收到 %r" % (asr_reset_mode,))
        self.asr_reset_mode = asr_reset_mode
        self.reset_silence_ms = reset_silence_ms
        self._last_speech_ms = 0

        # 600ms 统一流式步长 (9600 samples)
        self.chunk_size_asr = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        self.chunk_size_vad = 600
        self.chunk_stride_samples = 9600

        opts = {"disable_update": True, "disable_pbar": True}
        self.asr_model = asr_model or AutoModel(model=STREAM_ASR_LOCAL, device=device, **opts)
        self.vad_model = vad_model or AutoModel(
            model=VAD_LOCAL, max_end_silence_time=max_end_silence_time, device=device, **opts
        )

        os.makedirs(os.path.dirname(os.path.abspath(session_wav_path)), exist_ok=True)
        self._wav_file = wave.open(self.session_wav_path, "wb")
        self._wav_file.setnchannels(1)
        self._wav_file.setsampwidth(self.BYTES_PER_SAMPLE)
        self._wav_file.setframerate(self.SAMPLE_RATE)

        self.total_samples_processed = 0
        self._pcm_buffer = np.zeros(0, dtype=np.float32)

        self.asr_cache = {}
        self.vad_cache = {}
        self.in_vad_speech = False

        self.utt_counter = 0
        self.current_utt: Optional[Dict] = None
        self.completed_utterances: List[Dict] = []

    @property
    def processed_ms(self) -> int:
        return int(self.total_samples_processed * 1000 // self.SAMPLE_RATE)

    def _new_utt_id(self) -> str:
        self.utt_counter += 1
        return f"utt_{self.utt_counter:04d}"

    def feed_pcm_chunk(self, pcm_bytes: bytes, is_final: bool = False) -> List[Dict]:
        """喂入 16kHz int16 单声道字节流，返回当前触发的流式事件列表"""
        events: List[Dict] = []

        if pcm_bytes:
            if self._wav_file is None:
                # 已收尾（is_final 调过）：丢弃迟到/重复的音频包，别抛 AttributeError
                # 把调用方的推流循环打断（客户端重试、重复 stop 都会撞上）。
                return events
            self._wav_file.writeframes(pcm_bytes)
            self._wav_file._patchheader()
            int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            float32_data = int16_data.astype(np.float32) / 32768.0
            self._pcm_buffer = np.concatenate([self._pcm_buffer, float32_data])

        while len(self._pcm_buffer) >= self.chunk_stride_samples:
            chunk = self._pcm_buffer[: self.chunk_stride_samples]
            self._pcm_buffer = self._pcm_buffer[self.chunk_stride_samples :]
            is_last = is_final and (len(self._pcm_buffer) == 0)
            events.extend(self._step_chunk(chunk, is_stream_end=is_last))

        if is_final:
            if len(self._pcm_buffer) > 0:
                tail = self._pcm_buffer
                self._pcm_buffer = np.zeros(0, dtype=np.float32)
                events.extend(self._step_chunk(tail, is_stream_end=True))
            elif self.current_utt is not None:
                ev = self._finalize_utt(exact_end_ms=self.processed_ms, flush_asr=True)
                if ev:
                    events.append(ev)
            self.close_wav()

        return events

    def _step_chunk(self, chunk: np.ndarray, is_stream_end: bool = False) -> List[Dict]:
        events: List[Dict] = []
        chunk_start_ms = self.processed_ms
        self.total_samples_processed += len(chunk)
        chunk_end_ms = self.processed_ms

        # 1. 流式 fsmn-vad 检测毫秒级起止点
        vad_kw = dict(input=chunk, cache=self.vad_cache,
                      is_final=is_stream_end, chunk_size=self.chunk_size_vad)
        if self._vad_end_silence_in_call:
            vad_kw["max_end_silence_time"] = self.max_end_silence_time
        try:
            vad_res = self.vad_model.generate(**vad_kw)
        except TypeError:
            if not self._vad_end_silence_in_call:
                raise          # 自建的 VAD 在构造期就带了，这里失败是真 bug
            self._vad_end_silence_in_call = False   # 只提示一次，别每块都抛
            vad_kw.pop("max_end_silence_time")
            print("[vad] 本版 funasr 不接受 max_end_silence_time，改用模型默认断句",
                  flush=True)
            vad_res = self.vad_model.generate(**vad_kw)
        vad_segs = vad_res[0].get("value", []) if vad_res else []
        vad_start_ms, vad_end_ms = None, None
        for seg_beg, seg_end in vad_segs:
            if seg_beg != -1:
                self.in_vad_speech = True
                vad_start_ms = seg_beg
            if seg_end != -1:
                self.in_vad_speech = False
                vad_end_ms = seg_end

        if self.in_vad_speech:
            self._last_speech_ms = chunk_end_ms
        elif (self.asr_reset_mode == "continuous" and self.asr_cache
              and chunk_end_ms - self._last_speech_ms > self.reset_silence_ms):
            self.asr_cache = {}          # 长静音：才允许丢上下文

        # fsmn-vad 可能在同一块里同时给出 [-1, 上一段末] 与 [本段起, -1]。此时必须
        # **先**把上一句按它的真实末尾封口，**再**在 vad_start_ms 开新句 —— 否则本块的
        # ASR delta（其实属于新句开头）会记进上一句，新句只能拿 chunk_start_ms-150 顶替
        # start_ms，边界与说话人投票一起被带偏（OCR 2026-10-18）。
        if (vad_end_ms is not None and vad_start_ms is not None
                and self.current_utt is not None
                and vad_end_ms >= self.current_utt["start_ms"]):
            ev = self._finalize_utt(exact_end_ms=vad_end_ms, flush_asr=True)
            if ev:
                events.append(ev)

        if vad_start_ms is not None and self.current_utt is None:
            self.current_utt = {
                "utt_id": self._new_utt_id(),
                "start_ms": max(0, vad_start_ms),
                "end_ms": None,
                "raw_text": "",
            }

        if not self.in_vad_speech and self.current_utt is None and not is_stream_end:
            return events

        # 2. 流式 Paraformer 增量解码
        asr_res = self.asr_model.generate(
            input=chunk,
            cache=self.asr_cache,
            is_final=is_stream_end,
            chunk_size=self.chunk_size_asr,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
        )
        delta_text = asr_res[0].get("text", "") if asr_res else ""

        if delta_text and self.current_utt is None:
            self.current_utt = {
                "utt_id": self._new_utt_id(),
                "start_ms": max(0, chunk_start_ms - 150),
                "end_ms": None,
                "raw_text": "",
            }

        # 3. 推进状态：抛出斜体草稿或锁死句尾
        if self.current_utt is not None:
            if delta_text:
                self.current_utt["raw_text"] += delta_text
                events.append({
                    "t": "asr_partial",
                    "utt_id": self.current_utt["utt_id"],
                    "start_ms": self.current_utt["start_ms"],
                    "current_ms": chunk_end_ms,
                    "text": self.current_utt["raw_text"],
                })

            utt_dur = chunk_end_ms - self.current_utt["start_ms"]
            # vad_end_ms 可能属于本块刚关闭的**上一段**（fsmn-vad 会在一个 600ms
            # 结果里同时给出 [-1, 上一段末] 与 [本段起, -1]）。用它封口刚新建的句子
            # 会造出一个 ~250ms 的假碎片，且 end_ms 是别人的。只有 end 落在本句之后才认。
            hit_vad_end = (vad_end_ms is not None
                           and vad_end_ms >= self.current_utt["start_ms"])
            hit_long_split = (utt_dur >= self.max_utt_duration_ms) and (not delta_text)

            if hit_vad_end or hit_long_split or is_stream_end:
                end_anchor = vad_end_ms if hit_vad_end else chunk_end_ms
                ev = self._finalize_utt(exact_end_ms=end_anchor, flush_asr=(not is_stream_end))
                if ev:
                    events.append(ev)

        return events

    def _finalize_utt(self, exact_end_ms: int, flush_asr: bool = True) -> Optional[Dict]:
        if self.current_utt is None:
            return None

        if self.asr_reset_mode == "per_utt":
            # 策略 A：flush 尾字后清 cache，下一句从零开始（丢跨句上下文）
            if flush_asr and self.asr_cache:
                dummy = np.zeros(1600, dtype=np.float32)
                res = self.asr_model.generate(
                    input=dummy,
                    cache=self.asr_cache,
                    is_final=True,
                    chunk_size=self.chunk_size_asr,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
                tail = res[0].get("text", "") if res else ""
                if tail:
                    self.current_utt["raw_text"] += tail
            self.asr_cache = {}
        # 策略 B：flush 与清 cache 都不做 —— 保留 look_back 跨句记忆。
        # 代价：lookahead 尾字会溢出到下一句。长静音下由 _step_chunk 重置。
        raw_text = self.current_utt["raw_text"].strip()
        if not raw_text:
            self.utt_counter -= 1
            self.current_utt = None
            return None

        locked_end_ms = max(self.current_utt["start_ms"] + 250, exact_end_ms)
        final_utt = {
            "utt_id": self.current_utt["utt_id"],
            "start_ms": int(self.current_utt["start_ms"]),
            "end_ms": int(locked_end_ms),
            "raw_text": raw_text,
        }
        self.completed_utterances.append(final_utt)
        self.current_utt = None

        return {"t": "asr_final", **final_utt}

    def close_wav(self):
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception as exc:
                # 关不上 = 头部可能没 patch，后续 backfill 会读到截断的 WAV
                print("[session] 关闭会话 WAV 失败：%s" % exc, file=sys.stderr)
            self._wav_file = None


# --------------------------------------------------------------------------
# 验收：本地 WAV 按 600ms 切片喂流 → 草稿/锁死时间戳 → 任务 1 回填说话人
#   python .zhang/streaming_asr_engine.py <源.wav> [--slice-s 180]
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--session-wav", default="session_audio.wav")
    ap.add_argument("--slice-s", type=float, default=180.0, help="只喂前 N 秒；0=整段")
    ap.add_argument("--chunk-ms", type=int, default=600)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "streaming_asr"))
    from web_server import get_models

    # need_offline=False：只要 vad/spk/stream_asr，不拉 Fun-ASR-Nano（省一次加载）
    models = get_models(need_stream=True, need_offline=False)
    session = StreamingSession(args.session_wav,
                               asr_model=models["stream_asr"],
                               vad_model=models["vad"])

    info = sf.info(args.wav)
    if info.samplerate != session.SAMPLE_RATE:
        raise SystemExit(f"需要 16kHz，实际 {info.samplerate}Hz")
    audio, sr = sf.read(args.wav, dtype="int16", always_2d=True,
                        frames=int(args.slice_s * info.samplerate) if args.slice_s else -1)
    audio = audio[:, 0]
    print(f"喂入 {args.wav}: {len(audio) / sr:.1f}s @ {sr}Hz，步长 {args.chunk_ms}ms",
          flush=True)

    step = int(sr * args.chunk_ms / 1000)
    n_partial = 0
    for i in range(0, len(audio), step):
        is_end = (i + step) >= len(audio)
        for ev in session.feed_pcm_chunk(audio[i:i + step].tobytes(), is_final=is_end):
            if ev["t"] == "asr_partial":
                n_partial += 1
            else:
                print(f"  ✅ [{ev['start_ms']:6d} → {ev['end_ms']:6d} ms] {ev['raw_text']}",
                      flush=True)
    print(f"\n{len(session.completed_utterances)} 句锁定 / {n_partial} 次草稿更新",
          flush=True)

    from offline_diarizer import backfill_speakers

    mapping = backfill_speakers(args.session_wav, session.completed_utterances,
                                models=models, mode="auto")

    print("\n" + "=" * 62)
    for u in session.completed_utterances:
        print(f"[{mapping.get(u['utt_id'], 'Speaker_?')}] "
              f"{u['start_ms'] / 1000.0:7.1f}s  {u['raw_text']}")
    print("=" * 62)