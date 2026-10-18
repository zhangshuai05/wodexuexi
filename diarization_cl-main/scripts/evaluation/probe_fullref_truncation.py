# -*- coding: utf-8 -*-
"""整场复跑 v12，并证明「参考是否截断」是控制台那四个数的唯一差别。

背景：`交接文档_0923.md` §4.6/§4.7 曾把控制台报的
`v12=67.12% / 窗标签上界=73.51% / probe=83.83% / 真值边界上界=0.00%`
当成"另一个脚本"的产物、甚至当成一处自相矛盾（"v12 优于天花板"）。

实测结论：**同一个 `streaming_asr/der.py`，同一场会议，唯一差别是参考 RTTM
有没有截断到分析窗口。**

    预测覆盖 [0,600]   + 参考未截断(全场 2267s)  -> winOR = 73.51%
    预测覆盖 [0,2330]  + 参考未截断(全场 2267s)  -> v12   = 67.12%
    预测覆盖 [0,2330]  + 参考截断到 600s         -> DER   = 421%（多出的全是假警报）
    预测 = 参考本身（两种口径）                  -> bdOR  = 0.00%（自检）

机制：参考不截断时，落在预测窗口之外的参考语音全部记成 `missed`；
预测超出参考截断点时，多出的帧全部记成 `false_alarm`。
**两种截断必须一致，否则 DER 差 69 个点。**

    # 整场（约 10 分钟推理）
    python -u -B scripts/evaluation/probe_fullref_truncation.py
    # 只验口径、不跑推理
    python -u -B scripts/evaluation/probe_fullref_truncation.py --skip-v12
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
from der import der, read_rttm  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
MEETING = "L_R003S01C02"


def truncated(segs, seconds):
    return [{"start": s["start"], "end": min(s["end"], seconds), "speaker": s["speaker"]}
            for s in segs if s["start"] < seconds]


def score(reference, predicted, collar=0.25, denominator="union"):
    return der(reference, predicted, collar=collar, denominator=denominator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default=MEETING)
    ap.add_argument("--skip-v12", action="store_true",
                    help="只验口径差（零推理），不跑 v12.diarize()")
    args = ap.parse_args()

    wav = ROOT_DATA / "wav" / f"{args.meeting}.flac"
    segs = read_rttm(ROOT_DATA / "TextGrid" / f"{args.meeting}.rttm")
    full_end = max(s["end"] for s in segs)
    print(f"参考：{len(segs)} 段 / 全场 {full_end/60:.3f} 分钟", flush=True)

    # ---- 腿 1：winOR（缓存窗取参考多数说话人） ----
    data = np.load(CACHE / args.meeting / "w2_h1.npz", allow_pickle=False)
    times, speakers = data["times"], data["speakers"].astype(str)
    window_oracle = [{"start": float(t), "end": float(t) + 2.0, "speaker": f"gt_{n}"}
                     for t, n in zip(times, speakers) if n]
    print(f"\n缓存窗 {len(times)} 个，覆盖 [{times.min():.0f}, {times.max()+2:.0f}]s")

    # ---- 腿 2：probe（缓存窗 AHC@k=3 + 窗粒度分段） ----
    from scipy.cluster.hierarchy import fcluster, linkage
    tree = linkage(data["emb"], method="average", metric="cosine")
    labels = v11._relabel_by_first_seen(fcluster(tree, t=3, criterion="maxclust"))
    starts = [0]
    for index in range(1, len(times)):
        if labels[index] != labels[index - 1] or times[index] - times[index - 1] > 0.5:
            starts.append(index)
    starts.append(len(times))
    probe = []
    for first, last in zip(starts[:-1], starts[1:]):
        start = 0.0 if first == 0 else float((times[first - 1] + times[first]) / 2 + 0.25)
        end = (float(times[last - 1] + 0.5) if last == len(times)
               else float((times[last - 1] + times[last]) / 2 + 0.25))
        probe.append({"start": start, "end": end, "speaker": f"p_{int(labels[first])}"})

    # ---- 腿 3：bdOR（预测 = 参考本身） ----
    boundaries = [{"start": s["start"], "end": s["end"], "speaker": f"gt_{s['speaker']}"}
                  for s in segs]

    # ---- 腿 4：v12 真实路径（整场） ----
    v12_pred = None
    if not args.skip_v12:
        import claude_a2_v12 as v12
        from web_server import get_models
        models = get_models(lambda m: print(f"  [models] {m}", flush=True),
                            need_stream=False, need_offline=False)
        print("跑 v12.diarize(max_seconds=0) —— 整场，约 10 分钟…", flush=True)
        result = v12.diarize(str(wav), models, max_seconds=0)
        v12_pred = [{"start": float(s["start"]), "end": float(s["end"]),
                     "speaker": f"v_{int(s['spk'])}"} for s in result["segments"]]
        span = (min(p["start"] for p in v12_pred), max(p["end"] for p in v12_pred))
        print(f"v12：k={result['k']} refined={result['refined']} 段数={len(v12_pred)} "
              f"覆盖 [{span[0]:.0f},{span[1]:.0f}]s", flush=True)

    # ---- 打印：每条腿 × 两种参考 × 两种 collar ----
    legs = [("winOR", window_oracle, 73.51), ("probe@k3", probe, 83.83),
            ("bdOR", boundaries, 0.00)]
    if v12_pred is not None:
        legs.append(("v12", v12_pred, 67.12))

    print(f"\n{'腿':<10}{'参考':<12}{'collar':>7}{'分母':>10}{'DER':>9}"
          f"{'miss':>8}{'fa':>8}{'conf':>8}   控制台")
    for name, predicted, reported in legs:
        for refname, reference in (("未截断", segs),
                                   ("截断600s", truncated(segs, 600.0))):
            for collar in (0.0, 0.25):
                for den in ("union", "speakers"):
                    r = score(reference, predicted, collar=collar, denominator=den)
                    hit = "  <<<" if abs(r["der"] * 100 - reported) < 0.005 else ""
                    shown = f"{reported:.2f}%" if hit else ""
                    print(f"{name:<10}{refname:<12}{collar:>7.2f}{den:>10}"
                          f"{r['der']*100:>8.2f}%{r['miss_rate']*100:>7.2f}%"
                          f"{r['false_alarm_rate']*100:>7.2f}%"
                          f"{r['confusion_rate']*100:>7.2f}%   {shown}{hit}")

    print("\n参考各说话人时长占比：")
    total = sum(s["end"] - s["start"] for s in segs)
    by = {}
    for s in segs:
        by[s["speaker"]] = by.get(s["speaker"], 0.0) + s["end"] - s["start"]
    for name, value in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {name:<8} {value/total*100:5.2f}%")


if __name__ == "__main__":
    main()
