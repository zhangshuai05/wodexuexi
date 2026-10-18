# -*- coding: utf-8 -*-
"""平均语音重叠率：AliMeeting Eval vs AISHELL-4 test，两种口径。

这个脚本回答的不是"我的系统能得多少分"，而是一个**标定**问题：
两组数据的重叠难度差多少？不先量这个，后面 DER 变差就分不清
"数据变难了"还是"算法失效了"。

口径（两者都算，因为它们不是一回事，且相差可达数个百分点）：

  A 区域口径    = |S(f)| >= 2 的帧数 / |S(f)| >= 1 的帧数
                  "有多少比例的语音时间里是多人在同时说"

  B 说话人-时间 = (Σ_f |S(f)| - 语音帧数) / 语音帧数
                  = 说话人总时长 / 语音总时长 - 1
                  "多出来的那条说话人时间占语音时间的多少"

  C 说话人重叠 = Σ_f |S(f)|·[|S(f)| >= 2] / 语音帧数
                  "处在重叠区里的说话人时长之和占语音时间的多少"

恒有 C >= B >= A（三人同时说话时 A 记 1、B 记 2、C 记 3）。
**引用任何一个数字前必须说明是哪个口径**，也不要把此处算出的数字与官方文档里
不同定义的数字直接比。官方 AliMeeting 文档给出 Eval 平均重叠率 34.76%、Train
42.27%，但**没有写明定义**——本脚本三种口径都算出来跟它对，看哪个能对上。

     python overlap_stats.py
"""

import json

import numpy as np

from common import (AISHELL_TG, ALI_RTTM, FRAME, RESULTS, der_tools)

_, read_rttm = der_tools()


def overlap_profile(segments, frame=FRAME):
    """把参考段铺到 10 ms 栅格上，统计两种口径的重叠。"""
    if not segments:
        return None
    duration = max(seg["end"] for seg in segments)
    n = int(duration / frame) + 1
    counts = np.zeros(n, dtype=np.int16)
    for seg in segments:
        a = max(0, int(round(seg["start"] / frame)))
        b = min(n, int(round(seg["end"] / frame)))
        if b > a:
            counts[a:b] += 1

    speech = int((counts >= 1).sum())
    overlap_region = int((counts >= 2).sum())
    speaker_time = int(counts.sum())
    speaker_overlap = int(counts[counts >= 2].sum())
    if speech == 0:
        return None
    return {
        "duration_s": round(duration, 3),
        "speech_s": round(speech * frame, 3),
        "overlap_region_s": round(overlap_region * frame, 3),
        "speaker_time_s": round(speaker_time * frame, 3),
        "speaker_overlap_s": round(speaker_overlap * frame, 3),
        "n_speakers": len({seg["speaker"] for seg in segments}),
        "ratio_A_region": round(overlap_region / speech, 4),
        "ratio_B_speaker_time": round((speaker_time - speech) / speech, 4),
        "ratio_C_speaker_overlap": round(speaker_overlap / speech, 4),
    }


def collect(pairs):
    """pairs: [(name, rttm_path)] -> {name: profile}"""
    out = {}
    for name, path in pairs:
        profile = overlap_profile(read_rttm(path))
        if profile:
            out[name] = profile
    return out


def summarize(profiles):
    """macro = 逐场比值的平均；micro = 总量比值。两者不同，都报。"""
    values_a = [p["ratio_A_region"] for p in profiles.values()]
    values_b = [p["ratio_B_speaker_time"] for p in profiles.values()]
    values_c = [p["ratio_C_speaker_overlap"] for p in profiles.values()]
    speech = sum(p["speech_s"] for p in profiles.values())
    overlap_region = sum(p["overlap_region_s"] for p in profiles.values())
    speaker_time = sum(p["speaker_time_s"] for p in profiles.values())
    speaker_overlap = sum(p["speaker_overlap_s"] for p in profiles.values())
    return {
        "n_meetings": len(profiles),
        "speech_hours": round(speech / 3600, 2),
        "macro_A_region": round(float(np.mean(values_a)), 4),
        "macro_B_speaker_time": round(float(np.mean(values_b)), 4),
        "macro_C_speaker_overlap": round(float(np.mean(values_c)), 4),
        "micro_A_region": round(overlap_region / speech, 4),
        "micro_B_speaker_time": round((speaker_time - speech) / speech, 4),
        "micro_C_speaker_overlap": round(speaker_overlap / speech, 4),
    }


def main():
    ali = collect(sorted((p.stem, p) for p in ALI_RTTM.glob("*.rttm")))
    aishell = collect(sorted((p.stem, p) for p in AISHELL_TG.glob("*.rttm")))
    if not ali or not aishell:
        raise SystemExit(f"参考 RTTM 缺失：ali={len(ali)} aishell={len(aishell)}")

    header = (f"{'meeting':22s} {'时长':>7s} {'语音':>7s} {'人数':>4s} "
              f"{'A 区域':>8s} {'B 说话人-时间':>13s} {'C 说话人重叠':>13s}")

    def report(label, corpus):
        print(f"\n=== {label} ===")
        print(header)
        for name, p in corpus.items():
            print(f"{name:22s} {p['duration_s'] / 60:6.1f}m {p['speech_s'] / 60:6.1f}m "
                  f"{p['n_speakers']:>4d} {p['ratio_A_region'] * 100:7.2f}% "
                  f"{p['ratio_B_speaker_time'] * 100:12.2f}% "
                  f"{p['ratio_C_speaker_overlap'] * 100:12.2f}%")
        s = summarize(corpus)
        print(f"{'平均':22s} {s['speech_hours']:6.2f}h "
              f"{'':17s} {s['macro_A_region'] * 100:7.2f}% "
              f"{s['macro_B_speaker_time'] * 100:12.2f}% "
              f"{s['macro_C_speaker_overlap'] * 100:12.2f}%   <- 逐场平均(macro)")
        print(f"{'合计':22s} {'':9s} {'':9s} "
              f"{s['micro_A_region'] * 100:7.2f}% "
              f"{s['micro_B_speaker_time'] * 100:12.2f}% "
              f"{s['micro_C_speaker_overlap'] * 100:12.2f}%   <- 总量比值(micro)")
        return s

    ali_s = report("AliMeeting Eval 远场（本子项目）", ali)
    aishell_s = report("AISHELL-4 test 全部 20 场（主仓）", aishell)

    # 主仓的结论全部建立在 dev 12 场口径上，所以这一层单独拉出来看。
    # 这两个子集汇总**必须一并落盘**：文档引用的 5.09% 是 dev 12 场的值，
    # 不是全部 20 场的 4.89%；只存全量汇总的话文档就引了一个 JSON 里没有的数。
    from v12_aishell4 import DEV_GROUPS, HOLDOUT_GROUPS, meeting_group
    subsets = {}
    for label, groups in (("dev 12 场", DEV_GROUPS), ("holdout 8 场", HOLDOUT_GROUPS)):
        subset = {k: v for k, v in aishell.items() if meeting_group(k) in groups}
        if subset:
            s = report(f"AISHELL-4 {label}（主仓实际口径）", subset)
            subsets[label] = {"per_meeting": subset, "summary": s}
            if label.startswith("dev"):
                dev_s = s

    official = 0.3476
    print(f"\n官方 AliMeeting Eval 平均语音重叠率声明值: {official * 100:.2f}%（定义未写明）")
    closest = None
    for key in ("macro_A_region", "macro_B_speaker_time", "macro_C_speaker_overlap",
                "micro_A_region", "micro_B_speaker_time", "micro_C_speaker_overlap"):
        delta = abs(ali_s[key] - official) * 100
        if closest is None or delta < closest[1]:
            closest = (key, delta)
        print(f"  与本项目 {key:24s} = {ali_s[key] * 100:6.2f}%   差 {delta:5.2f} pts")
    print(f"  最接近的是 {closest[0]}（差 {closest[1]:.2f} pts）——"
          f"{'可以认为官方用的就是这个口径' if closest[1] < 1.5 else '三种口径都对不上，官方定义是第四种，引用时必须标明'}")

    print(f"\n难度差（AliMeeting - AISHELL-4 dev，A 区域口径）: "
          f"{(ali_s['macro_A_region'] - dev_s['macro_A_region']) * 100:+.2f} pts"
          f"  倍数: {ali_s['macro_A_region'] / dev_s['macro_A_region']:.2f}x")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "overlap_stats.json"
    out.write_text(json.dumps({
        "frame_s": FRAME,
        "ali": {"per_meeting": ali, "summary": ali_s},
        "aishell4_test": {"per_meeting": aishell, "summary": aishell_s,
                          "subsets": subsets},
        "official_ali_eval_overlap": official,
        "closest_convention": closest[0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
