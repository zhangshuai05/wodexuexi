# -*- coding: utf-8 -*-
"""口径敏感性：把已有预测 RTTM 按 M2MeT 官方的口径重评一遍。

为什么必须做：M2MeT 官方口径与本仓库 `der.py` 有 4 处已知差异，不逐项量化就说不清
"35.39% 与官方 15.24% 差 20 个点"里有多少是口径、多少是真差距。

| 轴 | 本仓库 `der.py` | M2MeT 官方 |
|---|---|---|
| 评分分母 T_Total | 参考**有语音**的帧数 | 论文公式写 "total time duration"（整场时长） |
| 标签映射 | greedy by overlap | **Hungarian**（论文 §3.1 明写） |
| collar | 0.25（默认，可传） | 0.25（正式）/ 0（论文并列给出） |
| VAD | **自研 FSMN-VAD** | 主办方**直接给 oracle SAD**，不需要 VAD 模块 |
| 通道 | ch0 单通道 | 官基线 `Ali-far` = 单通道；冠军/Royalflush = 多通道 |

本脚本覆盖前 3 轴（都能从 RTTM 重算）。VAD 那一轴无法事后补偿——official 是 oracle，
我们的是真跑出来的，只能把 miss+FA 那一份单独列出来看。

    python rescore_conventions.py
"""

import json
from itertools import product

import numpy as np
from scipy.optimize import linear_sum_assignment

from common import ALI_RTTM, PREDICTIONS, RESULTS, der_tools, resolve_wav

der, read_rttm = der_tools()
FRAME = 0.01


def frame_sets(segments, duration):
    n = int(duration / FRAME) + 1
    out = [set() for _ in range(n)]
    for seg in segments:
        a = max(0, int(round(seg["start"] / FRAME)))
        b = min(n, int(round(seg["end"] / FRAME)))
        for i in range(a, b):
            out[i].add(seg["speaker"])
    return out


def collar_mask(reference, collar, n):
    mask = [False] * n
    if collar <= 0:
        return mask
    span = int(round(collar / FRAME))
    for seg in reference:
        for boundary in (seg["start"], seg["end"]):
            centre = int(round(boundary / FRAME))
            for i in range(max(0, centre - span), min(n, centre + span + 1)):
                mask[i] = True
    return mask


def score(reference, predicted, duration, collar, mapping_kind, denominator):
    """返回 der / miss / fa / conf。mapping_kind ∈ {greedy, hungarian}；
    denominator ∈ {speech, duration}。"""
    n = int(duration / FRAME) + 1
    ref = frame_sets(reference, duration)
    hyp = frame_sets(predicted, duration)
    mask = collar_mask(reference, collar, n)

    ref_speakers = sorted({s["speaker"] for s in reference})
    hyp_speakers = sorted({s["speaker"] for s in predicted})
    ref_index = {s: i for i, s in enumerate(ref_speakers)}
    hyp_index = {s: i for i, s in enumerate(hyp_speakers)}

    # 共现矩阵：行 = 预测说话人，列 = 参考说话人
    co = np.zeros((len(hyp_speakers), len(ref_speakers)), dtype=np.int64)
    for i in range(n):
        if mask[i]:
            continue
        for h in hyp[i]:
            for r in ref[i]:
                co[hyp_index[h], ref_index[r]] += 1

    mapping = {}
    if len(hyp_speakers) and len(ref_speakers):
        if mapping_kind == "hungarian":
            rows, cols = linear_sum_assignment(-co)
        else:  # greedy by overlap，与 der.py 一致
            order = sorted(((co[i, j], i, j) for i in range(co.shape[0])
                            for j in range(co.shape[1])), reverse=True)
            rows, cols, used_r, used_c = [], [], set(), set()
            for _value, i, j in order:
                if i in used_r or j in used_c:
                    continue
                rows.append(i)
                cols.append(j)
                used_r.add(i)
                used_c.add(j)
        for i, j in zip(rows, cols):
            if co[i, j] > 0:
                mapping[hyp_speakers[i]] = ref_speakers[j]

    speech_frames = 0
    missed = false_alarm = confusion = 0
    for i in range(n):
        if mask[i]:
            continue
        R, H = ref[i], hyp[i]
        if R:
            speech_frames += 1
        if not R:
            false_alarm += len(H)
            continue
        mapped = {mapping[h] for h in H if h in mapping and mapping[h] in R}
        if R <= mapped:
            continue
        if not H:
            missed += 1
        else:
            confusion += 1

    divisor = speech_frames if denominator == "speech" else n
    divisor = max(1, divisor)
    return {
        "der": round((missed + false_alarm + confusion) / divisor, 4),
        "miss_rate": round(missed / divisor, 4),
        "false_alarm_rate": round(false_alarm / divisor, 4),
        "confusion_rate": round(confusion / divisor, 4),
        "divisor_frames": divisor,
    }


def main():
    import soundfile as sf

    meetings = sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    cache = {}
    for meeting in meetings:
        reference = read_rttm(ALI_RTTM / f"{meeting}.rttm")
        for arm in ("v12", "baseline"):
            path = PREDICTIONS / f"{meeting}.{arm}.full.rttm"
            if path.exists():
                cache[(meeting, arm)] = (reference, read_rttm(path))

    configs = list(product(("greedy", "hungarian"), ("speech", "duration"), (0.25, 0.0)))
    table = {}
    print(f"{'arm':9s} {'映射':10s} {'分母':9s} {'collar':>6s} {'DER':>8s} "
          f"{'miss':>7s} {'FA':>7s} {'conf':>7s}")
    for arm in ("v12", "baseline"):
        for mapping_kind, denominator, collar in configs:
            per_meeting = []
            for meeting in meetings:
                if (meeting, arm) not in cache:
                    continue
                reference, predicted = cache[(meeting, arm)]
                duration = sf.info(resolve_wav(meeting)).duration
                per_meeting.append(score(reference, predicted, duration, collar,
                                         mapping_kind, denominator))
            if not per_meeting:
                continue
            avg = {k: float(np.mean([p[k] for p in per_meeting]))
                   for k in ("der", "miss_rate", "false_alarm_rate", "confusion_rate")}
            table[f"{arm}|{mapping_kind}|{denominator}|{collar}"] = avg
            print(f"{arm:9s} {mapping_kind:10s} {denominator:9s} {collar:6.2f} "
                  f"{avg['der'] * 100:7.2f}% {avg['miss_rate'] * 100:6.2f}% "
                  f"{avg['false_alarm_rate'] * 100:6.2f}% {avg['confusion_rate'] * 100:6.2f}%")
        print()

    official = {"baseline_eval_far_collar025": 0.1524, "baseline_eval_far_collar0": 0.2452,
                "royalflush_fusion_eval_far": 0.0630, "winner_A41_test_far": 0.0298}
    key = "v12|hungarian|duration|0.25"
    ours = table[key]["der"]
    print(f"官方数字（论文，均为 collar 0.25）：")
    print(f"  官基线 Eval-Ali-far（单通道 + oracle VAD）= "
          f"{official['baseline_eval_far_collar025'] * 100:.2f}%"
          f"（collar0 对照值 {official['baseline_eval_far_collar0'] * 100:.2f}%）")
    print(f"  Royalflush 四系统融合 Eval（多通道）= "
          f"{official['royalflush_fusion_eval_far'] * 100:.2f}%")
    print(f"  冠军 A41 Test（多通道 + TS-VAD + 跳通道自注意力）= "
          f"{official['winner_A41_test_far'] * 100:.2f}%")
    print()
    print(f"最贴近官方口径的我们那条腿（hungarian + 整场时长分母 + collar 0.25）= "
          f"{ours * 100:.2f}%")
    print(f"  -> 对官基线（单通道 + oracle VAD）差 {(ours - official['baseline_eval_far_collar025']) * 100:+.2f} pts")
    print(f"  -> 对 Royalflush 融合（多通道）差 {(ours - official['royalflush_fusion_eval_far']) * 100:+.2f} pts")
    print(f"  -> 对冠军 A41（多通道，Test 集）差 {(ours - official['winner_A41_test_far']) * 100:+.2f} pts")

    ours_speech = table["v12|greedy|speech|0.25"]["der"]
    print(f"\n口径最大影响：本仓库默认口径 {ours_speech * 100:.2f}% "
          f"-> 官方口径 {ours * 100:.2f}%，差 {(ours_speech - ours) * 100:+.2f} pts")
    print(f"其中可事后补偿的只有上面这些；VAD 那一轴补不了——官方给 oracle SAD，"
          f"我们自研 VAD 的 miss+FA = "
          f"{(table['v12|greedy|speech|0.25']['miss_rate'] + table['v12|greedy|speech|0.25']['false_alarm_rate']) * 100:.2f} pts")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "convention_sensitivity.json"
    out.write_text(json.dumps({"table": table, "official": official}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
