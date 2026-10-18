# -*- coding: utf-8 -*-
"""SortFormer ONNX 最小验证：一场跑通 + 用已有的句级尺子评测。

模型权重（474MB）和 Python 实现都已在本地，这是 ponytail 阶梯第 5 级
（已装依赖能解决就别自己造）—— 跑不通再回去移植 3D-Speaker。

用法：
    $env:PYTHONIOENCODING="utf-8"
    python run_sortformer.py R8007_M8010
"""

import os
import sys
from pathlib import Path

# 代理变量里的 NO_PROXY=[::1] 会让 httpx 抛 InvalidURL，挡住模型下载
for _k in list(os.environ):
    if "proxy" in _k.lower():
        del os.environ[_k]

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "streaming_asr", HERE.parent / "scripts" / "evaluation",
           HERE.parent / "scripts" / "diagnostics"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import ALI_RTTM, ALI_TG, der_tools, resolve_wav  # noqa: E402
from probe_sentence_attribution import (read_textgrid,  # noqa: E402
                                       sentences_from_textgrid)
from audit_turn_metrics import global_mapping, turn_metrics  # noqa: E402

_, read_rttm = der_tools()
BASE = {"R8001_M8004": 45.0, "R8003_M8001": 42.3, "R8007_M8010": 22.2,
        "R8007_M8011": 34.8, "R8008_M8013": 38.1, "R8009_M8018": 77.1,
        "R8009_M8019": 85.7, "R8009_M8020": 84.2}


def main():
    meetings = sys.argv[1:] or ["R8007_M8010"]
    from sortformer import Sortformer
    # 本地已有同源导出（soniqo 版，几何参数与 cgus 版一致），优先用它，避免重复下载 500MB
    local = Path(r"D:\yinp\sortformer_onnx\sortformer-default.onnx")
    model = Sortformer(str(local) if local.exists()
                       else "cgus/diar_streaming_sortformer_4spk-v2.1-onnx")
    for meeting in meetings:
        wav = resolve_wav(meeting)
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        segs = model.diarize(audio, sr, 1)
        pred = [{"start": float(s.start), "end": float(s.end),
                 "speaker": "SF_%d" % s.speaker_id} for s in segs]
        truth = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        gt_turns = sentences_from_textgrid(read_textgrid(ALI_TG / f"{meeting}.TextGrid"))
        mapping = global_mapping(truth, pred)
        m = turn_metrics(pred, gt_turns, mapping)
        n_pred = len({p["speaker"] for p in pred})
        print("\n=== %s ===" % meeting)
        print("  SortFormer: %d 段, %d 个说话人 (真值 %d 人)"
              % (len(pred), n_pred, len({t["speaker"] for t in gt_turns})))
        print("  句级准确 %5.1f%%   帧级 %5.1f%%   切句率 %5.1f%%   碎段 %.2f"
              % (m["acc_sentence"] * 100, m["acc_frame"] * 100,
                 m["split_rate"] * 100, m["fragments_mean"]))
        print("  对照（现状网格窗+AHC）: %.1f%%" % BASE.get(meeting, float("nan")))


if __name__ == "__main__":
    main()
