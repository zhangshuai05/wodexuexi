# -*- coding: utf-8 -*-
"""选 K 解锁的**诚实复核**：加了 DER 才敢下结论。

`probe_selector_locks.py` 报出"再拆轮廓系数"帧级均值 +6.8 点、最差 47.2%->63.6%。
但有一个危险的读法必须先排除：

  * `M_R003S05C01` 有效 K=4，却选了 **k=7**，帧级反而 47.2% -> 97.7%。
  * 帧级正确率**不惩罚碎段与边界错**：把一个人切成两簇，只要贪心映射把两簇都指到
    正确的人，帧级照样算对；而 DER 会把那段边界差、以及映射表漏掉的簇记成错。
  * 所以"拆掉轮廓系数锁"可能只是**让规则偏爱更高的 k**，而高 k 在帧级口径上免费。

本脚本对同样的四条规则同时报：
  * 选出的 k
  * 帧级说话人正确率
  * **DER**（collar=0.25，用 v12 真正的分段函数 `_segment_windows`）
  * 该场各簇的窗数分布（看是不是碎成渣）

零推理；纯缓存。

    python scripts/evaluation/probe_selector_locks_der.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_selector_locks import RULES, build, select  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0


def predicted_rttm(times, labels, hop_s):
    """用 v12 自己的分段函数把窗标签变成段（与生产路径同一支）。"""
    segments = v12._segment_windows(times, labels, np.zeros(len(times)), hop_s=hop_s)
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "speaker": "p_%d" % int(s["spk"])} for s in segments]


def main():
    rows = []
    print(f"{'会议':<16}{'有效K':>6}{'规则':>14}{'选k':>5}{'帧级':>8}{'DER':>8}  簇窗数分布")
    for meeting in split_meetings(ROOT_DATA, "dev"):
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / f"{meeting}.rttm")
                     if s["start"] < SECONDS]
        reference = [{"start": s["start"], "end": min(s["end"], SECONDS),
                      "speaker": s["speaker"]} for s in reference]
        emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=SECONDS), WIN, 0.5)
        duration = Counter()
        for s in reference:
            duration[s["speaker"]] += s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sorted(k for k, v in duration.items() if v / grand >= 0.05)
        if len(emb) < 2:
            continue
        win = np.full(len(times), WIN)
        for name, cap, unlock in RULES:
            best = build(emb, cap, unlock)
            chosen = select(best, unlock)
            labels = chosen["labels"]
            named = map_by_frames(reference, times, labels, win)
            accuracy = frame_accuracy(reference, times, named, win)["accuracy"]
            pred = predicted_rttm(times, labels, WIN * 0.25)
            value = der(reference, pred, collar=0.25)["der"]
            sizes = sorted(chosen["sizes"], reverse=True)
            tiny = sum(1 for size in sizes if size == 1)
            shape = ("碎" if tiny >= 2 else "细" if tiny == 1 else "整")
            rows.append({"meeting": meeting, "rule": name, "k": chosen["k"],
                         "accuracy": accuracy, "der": value, "sizes": sizes,
                         "eff_k": len(effective)})
            print(f"{meeting:<16}{len(effective):>6}{name:>14}{chosen['k']:>5}"
                  f"{accuracy * 100:>7.1f}%{value * 100:>7.1f}%  "
                  f"{shape} {sizes[:6]}")

    print()
    print(f"{'规则':<16}{'帧级均值':>9}{'DER均值':>9}{'选的k':>9}{'高估场数':>9}{'碎簇场':>7}")
    for name, _, _ in RULES:
        part = [r for r in rows if r["rule"] == name]
        accuracy = np.mean([r["accuracy"] for r in part])
        value = np.mean([r["der"] for r in part])
        over = sum(1 for r in part if r["k"] > r["eff_k"])
        crushed = sum(1 for r in part if sum(1 for s in r["sizes"] if s == 1) >= 2)
        print(f"{name:<16}{accuracy * 100:>8.1f}%{value * 100:>8.1f}%{over:>9}{crushed:>7}")

    import json
    output = ROOT / "artifacts" / "v13_selector_locks_der.json"
    output.write_text(json.dumps({"schema": "v13-selector-locks-der-v1", "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {output}")


if __name__ == "__main__":
    main()
