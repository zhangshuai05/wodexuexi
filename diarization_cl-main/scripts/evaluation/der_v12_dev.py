# -*- coding: utf-8 -*-
"""v12 真实产出的 DER，与探针口径逐场并列（AISHELL-4 dev/holdout）。

为什么需要它：交接文档报的 22.79% 基线来自 `_v13_spike/der.py` 对
"2.0 秒窗缓存 + AHC + `labels_to_rttm`" 的评分，而 `labels_to_rttm` **不是 v12 的产出
路径**——v12 走 `claude_a2_v12._segment_windows`，边界公式、碎段过滤、细化回退都不同。
在这个脚本之前，仓库里没有任何一个 DER 数字是在 v12 真实输出上算的。

每场会议跑三条：

    probe  缓存的 2.0 秒窗 + 现行选 K + `labels_to_rttm`（复现 22.79% 的口径）
    v12    `claude_a2_v12.diarize()` 的真实产出（跑推理）
    oracle 参考窗标签 + 参考边界（上界，不跑推理）

    python scripts/evaluation/der_v12_dev.py --limit 1        # 冒烟
    python scripts/evaluation/der_v12_dev.py --split dev

`--seconds 600` 与探针口径对齐（探针全部在 600 秒上）；`--seconds 0` 跑整场。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASR = ROOT / "streaming_asr"
sys.path.insert(0, str(ASR))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import der, read_rttm  # noqa: E402
from v12_aishell4 import HOLDOUT_GROUPS, DEV_GROUPS, meeting_group  # noqa: E402

SKIP_SECONDS = 10.0


def segments_to_rttm(segments, name):
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "speaker": f"{name}_{int(s['spk'])}"} for s in segments]


def probe_rttm(cache_dir, meeting, window, hop_ratio=0.5, split_at=0.25):
    """缓存的嵌入 + 探针的选 K + 探针的合并规则。

    这条腿是**复现 22.79% 的口径**，两处必须照抄探针，不能顺手套用 v12 的现行选择器：

    * 选 K = 穷举 2..min(10, n) 取轮廓系数最大（`diag_der_truek.py` 的做法），
      **不是** `v11._select_candidate`（它带簇尺寸门槛与小 K 惩罚）。
    * 边界步长 = 窗长的一半（`labels_to_rttm` 的 `win_s / 4` 加在中点上，
      等价于 1.0 秒窗下的半步）。
    """
    path = Path(cache_dir) / meeting / f"w{window:g}_h{window * hop_ratio:g}.npz"
    if not path.exists():
        return None, {"error": "no cache"}
    data = np.load(path, allow_pickle=False)
    E, times = data["emb"], data["times"]
    if len(E) < 8:
        return None, {"error": "too few windows"}
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(E, method="average", metric="cosine")
    best_k, best_score = None, None
    for k in range(2, min(10, len(E)) + 1):
        labels = fcluster(tree, t=k, criterion="maxclust") - 1
        silhouette = v11._silhouette(E, labels)
        if best_score is None or silhouette > best_score:
            best_k, best_score = k, silhouette
    labels = fcluster(tree, t=best_k, criterion="maxclust") - 1
    segments = v12._segment_windows(times, labels, np.zeros(len(times)),
                                    hop_s=window * split_at)
    return segments_to_rttm(segments, "probe"), {"k": int(best_k)}


def reference_window_oracle(cache_dir, meeting, window, hop_ratio=0.5):
    """每个窗取参考的多数说话人当标签，边界仍按窗的步长走。

    这条衡量的是"标签全对、但时间粒度锁死在 `window` 秒"能到多少 —— 与
    `reference_boundaries` 之差就是 2.0 秒粒度本身的代价。
    """
    path = Path(cache_dir) / meeting / f"w{window:g}_h{window * hop_ratio:g}.npz"
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=False)
    times, speakers = data["times"], data["speakers"].astype(str)
    return [{"start": float(start), "end": float(start + window),
             "speaker": f"gt_{speaker}"}
            for start, speaker in zip(times, speakers) if speaker]


def reference_boundaries(rttm_segments, seconds):
    """真值边界上界：拿参考段的起止当预测，不经过任何窗。

    衡量的是"边界和身份全对能到多少"。它必然得 0:DER 的分子只剩参考窗被截断
    到 `seconds` 的那一点——留着是为了确认同一把尺子对完美输入给出 0，
    以及确认重叠帧在**新分母**下不再被重复计费。
    """
    return [{"start": float(s["start"]), "end": min(float(s["end"]), seconds),
             "speaker": f"gt_{s['speaker']}"}
            for s in rttm_segments if s["start"] < seconds]


def score(reference, predicted, collar):
    if not predicted:
        return None
    return der(reference, predicted, collar=collar)["der"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=600.0, help="0 = 整场")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-v12", action="store_true", help="只跑探针与上界，不跑推理")
    args = ap.parse_args()

    root = Path(args.root)
    groups = DEV_GROUPS if args.split == "dev" else HOLDOUT_GROUPS
    meetings = sorted(p.stem for p in (root / "TextGrid").glob("*.rttm")
                      if meeting_group(p.stem) in groups)
    if args.limit:
        meetings = meetings[:args.limit]

    models = None
    if not args.skip_v12:
        from web_server import get_models

        models = get_models(lambda message: None, need_stream=False, need_offline=False)

    header = (f"{'meeting':16s} {'gtK':>4} {'selK':>5} {'level':>20s} "
              f"{'probe':>8} {'v12':>8} {'winOR':>8} {'bdOR':>8}")
    print(header, flush=True)
    rows = []
    for meeting in meetings:
        rttm_path = root / "TextGrid" / f"{meeting}.rttm"
        reference_all = read_rttm(rttm_path)
        names = sorted({s["speaker"] for s in reference_all})
        seconds = args.seconds if args.seconds > 0 else max(
            s["end"] for s in reference_all)
        reference = [{"start": s["start"], "end": min(s["end"], seconds),
                      "speaker": s["speaker"]}
                     for s in reference_all if s["start"] < seconds]

        probe_segments, probe_meta = probe_rttm(args.cache, meeting, args.window)
        probe_der = score(reference, probe_segments, args.collar)
        sel_k = probe_meta.get("k") if isinstance(probe_meta, dict) else None

        window_segments = reference_window_oracle(args.cache, meeting, args.window)
        window_oracle = score(reference, window_segments, args.collar)
        # `reference` 已截到 `seconds`，这条腿的预测必须同样截断：不截的话
        # 预测会伸到全场末尾，尾部那段的假警报会被记进来（600s 口径下 +0.24%）。
        # 两条腿都截断后它才是纯粹的自检——同一把尺子对完美输入给 0。
        boundary_oracle = score(reference,
                                reference_boundaries(reference, seconds),
                                args.collar)

        v12_der, level, v12_k = None, "-", None
        if models is not None:
            wav = root / "wav" / f"{meeting}.flac"
            result = v12.diarize(str(wav), models, max_seconds=args.seconds)
            v12_segments = segments_to_rttm(result["segments"], "v12")
            v12_der = score(reference, v12_segments, args.collar)
            level = ("refined" if result["refined"] else
                     "rescued" if result["rescued"] else
                     "2s-only" if not result["insufficient"] else "insufficient")
            v12_k = result["k"]

        fmt = lambda value: "   -   " if value is None else f"{value * 100:6.2f}%"
        print(f"{meeting:16s} {len(names):>4} {str(v12_k if v12_k else sel_k):>5} "
              f"{level:>20s} {fmt(probe_der)} {fmt(v12_der)} "
              f"{fmt(window_oracle)} {fmt(boundary_oracle)}", flush=True)
        rows.append({"meeting": meeting, "gtK": len(names), "selK": v12_k or sel_k,
                     "level": level, "probe": probe_der, "v12": v12_der,
                     "window_oracle": window_oracle,
                     "boundary_oracle": boundary_oracle})

    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return f"{np.mean(values) * 100:.2f}%" if values else "-"

    print(f"\n平均（{len(rows)} 场，{args.seconds or '整场'}秒，collar={args.collar}）: "
          f"probe={mean('probe')}  v12={mean('v12')}  "
          f"窗标签上界={mean('window_oracle')}  真值边界上界={mean('boundary_oracle')}",
          flush=True)
    refined = sum(1 for row in rows if row["level"] == "refined")
    print(f"两级细化生效 {refined}/{len(rows)} 场", flush=True)


if __name__ == "__main__":
    main()
