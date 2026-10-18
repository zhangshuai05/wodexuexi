# -*- coding: utf-8 -*-
"""会后说话人回填 —— 复用 AliMeeting 新链路，不自研聚类。

链路（`run_3dspeaker_pipeline.py`，HANDOFF_1015 验证过的那条）：
    VAD → class 区间切分 → CAM++ 嵌入 → 谱聚类（k-NN 图 + eigengap 定 K）
        → 0.6s 密集窗细化（长窗定身份、短窗找边界）
        → 无短窗覆盖的句子退回长窗投票（`fallback=`）

本文件原先那套自研 `OverlapAwareDiarizer`（0.9s/0.45s 滑窗 + 谱聚类）已删除：
它在高重叠场塌陷（R8007_M8010 窗级/句级全塌到 K=1），而上面的链路 8 场无一亏损。
删除时无调用方 —— `.zhang/streaming_asr_engine.py` 已不 import 本模块。

唯一入口：`backfill_speakers(wav_path, utterances, models=None, mode="auto")`
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
# run_3dspeaker_pipeline / claude_a2_v1x 的导入前提（与该脚本自身的 bootstrap 同源）
for _p in (_ROOT / "AliMeeting", _ROOT / "streaming_asr", _ROOT / "scripts" / "evaluation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# OSD 只用来切 class 区间。缺它时 mode="auto" 退化成纯 VAD 窗。
# 这是本机资产（约 1.2GB 的 pyannote 导出），换机器用环境变量指过去，别改代码。
OSD_ONNX = Path(os.environ.get("DIAR_OSD_ONNX",
                               r"D:\yinp\pyannote_seg3\model.onnx"))


def _osd_chunk_windows(audio: np.ndarray, wins_vad):
    """有 OSD 就用 class 区间切窗，否则原样返回 VAD 窗。"""
    from run_3dspeaker_pipeline import class_chunks, osd_probs
    if not OSD_ONNX.exists():
        return wins_vad, "vad(无OSD)"
    import onnxruntime as ort
    sess = ort.InferenceSession(str(OSD_ONNX), providers=["CPUExecutionProvider"])
    probs_all, mask_all, fs = osd_probs(sess, audio)
    return class_chunks(probs_all[mask_all], fs)[0], "class"


def backfill_speakers(
    wav_path: str,
    utterances: List[Dict],
    models: Optional[Dict] = None,
    mode: str = "auto",
) -> Dict[str, str]:
    """整场音频 + 流式阶段锁死的句子 -> {utt_id: "Speaker_N"}。

    utterances: [{"utt_id": "utt_0001", "start_ms": 1200, "end_ms": 4500}, ...]
                start_ms/end_ms 必须是流式阶段按采样点计数锁死的绝对值（见
                .zhang/streaming_asr_engine.py）；本函数只按这两个数对齐。
    models:     web_server.get_models() 的返回（含 "vad"/"spk"）。None 时自取。
    mode:       "auto"（有 OSD 用 class，否则退化 vad）| "class" | "vad"

    返回 {utt_id: "Speaker_N"}；N 是聚类标签 +1，全局一致。
    """
    from run_3dspeaker_pipeline import (chunk_windows, embed_chunks,
                                        spectral_cluster, turns_to_speakers)
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12

    if mode not in ("auto", "class", "vad"):
        raise ValueError(f"mode 只支持 auto/class/vad，收到 {mode!r}")
    if not utterances:
        return {}

    if models is None:
        from run_der import load_models
        models = load_models()

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 句子键名是 *_ms，链路内部一律用秒
    turns = [{"utt_id": u["utt_id"],
              "start": u["start_ms"] / 1000.0,
              "end": u["end_ms"] / 1000.0} for u in utterances]

    vad = v11._vad_segs(models["vad"], str(wav_path))
    wins = chunk_windows(vad)
    if mode in ("auto", "class"):
        if mode == "class" and not OSD_ONNX.exists():
            raise FileNotFoundError(f"mode='class' 需要 OSD 模型：{OSD_ONNX}")
        wins, arm = _osd_chunk_windows(audio, wins)
        if arm != "class":
            # 降级必须是可观测的：否则一次 class 臂的结果会被当成 VAD 臂（反之亦然）
            print("[diarizer] 未找到 OSD（%s），本次退化为纯 VAD 切窗" % OSD_ONNX,
                  flush=True)
    if not wins:
        return {u["utt_id"]: "Speaker_1" for u in utterances}

    E, kept = embed_chunks(models["spk"], audio, rate, wins)
    if len(E) == 0:
        return {u["utt_id"]: "Speaker_1" for u in utterances}

    labels, k, _ = spectral_cluster(E, None)

    # 细化层：长窗定身份、0.6s 密集窗找边界；没短窗盖住的句子退回长窗（fallback）。
    # k<2 或细化后塌成 1 人时不接细化 —— 与 main() 的守卫一致。
    rwins = rlabels = None
    if k >= 2:
        anchors = v11._centroids(E, labels)
        rE, rtimes = v12._refine_windows(audio, rate, vad, models["spk"])
        rlabels = v12._refine_labels(rE, anchors,
                                     sorted({int(x) for x in labels.tolist()}), rtimes)
        if rlabels is None or len(set(rlabels.tolist())) < 2:
            rwins = rlabels = None
        else:
            rwins = [(float(t), float(t) + v12.REFINE_WIN_S) for t in rtimes]

    if rwins is not None:
        pred, _ = turns_to_speakers(rwins, rlabels, turns, fallback=(kept, labels))
    else:
        pred, _ = turns_to_speakers(kept, labels, turns)

    return {t["utt_id"]: f"Speaker_{int(p) + 1}" for t, p in zip(turns, pred)}


# --------------------------------------------------------------------------
# 自检：确认 K 不塌陷 + 打印说话人分布（不需要真值）
#   python .zhang/offline_diarizer.py R8008_M8013
# 准确率对账走 eval_multilabel.py（已有），不要在这里再写一个评分器。
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from collections import Counter

    from common import resolve_wav
    from eval_multilabel import vad_sentences

    meeting = sys.argv[1] if len(sys.argv) > 1 else "R8008_M8013"
    wav = resolve_wav(meeting)

    t0 = time.time()
    from run_der import load_models
    models = load_models()
    # 1000ms = StreamingSession.max_end_silence_time 的默认值；同口径（少一次 ASR 二次切分）
    utts = [{"utt_id": f"utt_{i:04d}",
             "start_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000)}
            for i, s in enumerate(
                vad_sentences(models["vad"], wav, max_end_silence_ms=1000)[0], 1)]

    mapping = backfill_speakers(wav, utts, models=models, mode="auto")
    hist = Counter(mapping.values())

    print(f"\n{meeting}: {len(utts)} 句 -> {len(hist)} 个说话人  "
          f"({time.time() - t0:.1f}s)")
    for spk, n in hist.most_common():
        print(f"  {spk:12s} {n:5d} 句  {n / len(utts):6.1%}")
    print("K 塌陷为 1" if len(hist) == 1 else f"K = {len(hist)}（未塌陷）")
