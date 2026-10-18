# -*- coding: utf-8 -*-
"""把"选 K 的代价"和"聚类本身的代价"分开。

生产路径 `claude_a2_v12.diarize()` 在全长 AliMeeting 上会把人数判成 k=1
（见 `results/der_alimeeting_full.json`），而 600 秒口径下同一场能判出 k=5。
塌成单簇时 DER 里全是 confusion，看不出到底是"选 K 错了"还是"嵌入本身就分不开"。

三条腿，**只有 K 不同**，其余完全一致（同 VAD、同 CAM++、同 2.0 秒窗、
同 `_segment_windows` 边界公式、同 sliver 过滤、同 `der.der()`）：

    auto    生产选择器 `v11._select_candidate`（带簇尺寸门槛与小 K 惩罚）
    true K  强制成参考说话人数（参考人数先验，**不是系统能拿到的信息**，
            只作为诊断上界，不得当作可达收益引用）
    ref     参考边界直接当预测（同一把尺子的自检，恒为 0）

    auto → trueK 的差  = 选 K 策略的代价
    trueK → ref 的差   = 聚类 + 分段本身的代价

    python run_oracle_k.py --limit 1                 # 冒烟
    python run_oracle_k.py --out oracle_k_alimeeting.json
"""

import argparse
import json
import time

import numpy as np

from common import (ALI_RTTM, COLLAR, PREDICTIONS, RESULTS, der_tools,
                    resolve_wav)
from run_der import (load_models, overlap_attribution, truncate_reference,
                     write_rttm)

der, read_rttm = der_tools()


def extract(model_bundle, wav_path):
    """生产前端的嵌入提取（2.0 秒窗 / 1.0 秒步长），返回 (E, times, 窗数)。"""
    import soundfile as sf

    import claude_a2_v11 as v11
    import claude_a2_v12 as v12

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = v11._vad_segs(model_bundle["vad"], str(wav_path))
    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        E, times, _errors = v11._extract_windows(audio, rate, vad_segments,
                                                 model_bundle["spk"])
    return E, times


def candidates_for(E):
    import claude_a2_v11 as v11
    import claude_a2_v12 as v12
    with v12._v11_window_params(v12.WIN_S, v12.HOP_S, v12.MAX_SPEAKERS):
        # max_speakers 必须显式传：v11 里它是 def 时绑定的默认参数（=4），
        # `_v11_window_params` 改模块变量改不动它。生产 `diarize` 两处调用点
        # 也是显式传的（K 上限静默截断 bug 的修法）。
        candidates, _tree = v11._candidate_clusters(E, max_speakers=v12.MAX_SPEAKERS)
    return candidates


def segments_from(labels, times, prefix):
    import claude_a2_v12 as v12
    raw = v12._segment_windows(times, labels, np.zeros(len(times)), hop_s=v12.HOP_S)
    kept = [s for s in raw if s["n_windows"] > v12.SLIVER_WINDOWS]
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "speaker": f"{prefix}_{int(s['spk'])}"} for s in kept], len(raw) - len(kept)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--collar", type=float, default=COLLAR)
    parser.add_argument("--meetings", nargs="*", default=None)
    parser.add_argument("--out", default="oracle_k_alimeeting.json")
    args = parser.parse_args()

    meetings = args.meetings or sorted(p.stem for p in ALI_RTTM.glob("*.rttm"))
    if args.limit:
        meetings = meetings[:args.limit]

    models = load_models()
    rows = []
    print(f"{'meeting':16s} {'真值K':>5s} {'autoK':>5s} {'autoDER':>8s} "
          f"{'trueKDER':>8s} {'选K代价':>8s} {'聚类代价':>8s}", flush=True)
    for meeting in meetings:
        try:
            wav = resolve_wav(meeting)
            reference_all = read_rttm(ALI_RTTM / f"{meeting}.rttm")
            reference = truncate_reference(reference_all, args.seconds)
            true_k = len({s["speaker"] for s in reference})

            started = time.time()
            E, times = extract(models, wav)
            if len(E) < 2:
                raise RuntimeError(f"窗数不足: {len(E)}")
            candidates = candidates_for(E)
            elapsed = time.time() - started

            import claude_a2_v11 as v11
            auto = v11._select_candidate(candidates)
            forced = next((c for c in candidates if c["k"] == true_k), None)

            auto_segments, auto_dropped = segments_from(
                auto["labels"], times, "auto")
            row = {"meeting": meeting, "true_k": true_k, "auto_k": int(auto["k"]),
                   "n_windows": int(len(E)), "elapsed_s": round(elapsed, 1),
                   "dropped": auto_dropped,
                   "candidates": [{"k": c["k"], "score": round(float(c["score"]), 4),
                                   "objective": round(float(c["objective"]), 4),
                                   "eligible": bool(c["eligible"]),
                                   "sizes": c["sizes"]}
                                  for c in candidates]}
            row["der_auto"] = der(reference, auto_segments, collar=args.collar)
            row["der_auto"].update(
                overlap_attribution(reference, auto_segments, args.collar))

            if forced is None:
                row["der_true_k"] = {"note": f"候选里没有 k={true_k}（上限 {len(candidates) - 1}）"}
            else:
                forced_segments, _ = segments_from(forced["labels"], times, "truek")
                row["der_true_k"] = der(reference, forced_segments, collar=args.collar)
                row["der_true_k"].update(
                    overlap_attribution(reference, forced_segments, args.collar))

            reference_segments = [{"start": s["start"], "end": s["end"],
                                   "speaker": f"ref_{s['speaker']}"}
                                  for s in reference]
            row["der_reference"] = der(reference, reference_segments, collar=args.collar)

            auto_der = row["der_auto"]["der"]
            true_der = row["der_true_k"].get("der")
            print(f"{meeting:16s} {true_k:>5d} {row['auto_k']:>5d} "
                  f"{auto_der * 100:7.2f}% "
                  f"{'   -   ' if true_der is None else f'{true_der * 100:7.2f}%'} "
                  f"{'   -   ' if true_der is None else f'{(auto_der - true_der) * 100:+7.2f}%'} "
                  f"{'   -   ' if true_der is None else f'{true_der * 100:7.2f}%'}", flush=True)
            rows.append(row)
        except Exception as exc:
            print(f"{meeting:16s}  FAILED {type(exc).__name__}: {exc}", flush=True)
            rows.append({"meeting": meeting, "error": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in rows if "error" not in r and r["der_true_k"].get("der") is not None]
    if ok:
        auto_der = [r["der_auto"]["der"] for r in ok]
        true_der = [r["der_true_k"]["der"] for r in ok]
        print(f"\n平均（{len(ok)}/{len(rows)} 场）")
        print(f"  auto K      DER = {float(np.mean(auto_der)) * 100:6.2f}%")
        print(f"  强制真值 K  DER = {float(np.mean(true_der)) * 100:6.2f}%")
        print(f"  -> 选 K 的代价 = {(float(np.mean(auto_der)) - float(np.mean(true_der))) * 100:+.2f} pts")
        auto_k_hit = sum(1 for r in ok if r["auto_k"] == r["true_k"])
        print(f"  auto K 命中真值的场次: {auto_k_hit}/{len(ok)}")

        # 按真值人数分组：这是本批数据里最强的规律，必须显式算出来而不是靠眼睛看表格。
        print("\n按真值人数分组：")
        for size in sorted({r["true_k"] for r in ok}):
            group = [r for r in ok if r["true_k"] == size]
            g_auto = float(np.mean([r["der_auto"]["der"] for r in group]))
            g_true = float(np.mean([r["der_true_k"]["der"] for r in group]))
            hits = sum(1 for r in group if r["auto_k"] == r["true_k"])
            print(f"  {size} 人场（{len(group)} 场）: auto {g_auto * 100:6.2f}%  "
                  f"真值K {g_true * 100:6.2f}%  选K命中 {hits}/{len(group)}")

        print("\n候选表（k:轮廓系数，! = 不合格簇）：")
        for r in ok:
            table = "  ".join(f"k{c['k']}:{c['score']:+.3f}{'' if c['eligible'] else '!'}"
                              for c in r["candidates"])
            print(f"  {r['meeting']:16s} 真值{r['true_k']} 选中{r['auto_k']}  {table}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps({
        "config": {"corpus": "AliMeeting Eval (far-field, ch0, mono)",
                   "seconds": args.seconds or "full", "collar": args.collar,
                   "note": "true-K 是诊断上界，用了参考人数先验，不得当作可达收益"},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
