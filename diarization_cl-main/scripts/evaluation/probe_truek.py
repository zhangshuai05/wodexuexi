# -*- coding: utf-8 -*-
"""复现 spike 的 der@selK / der@trueK，逐场并与缓存口径对照。

交接文档 §4.3 报的 22.79%（现行选 K）与 21.98%（已知人数）来自
`_v13_spike/diag_der_truek.py`。那条脚本**不复用缓存**——它每场重新抽 embedding，
VAD 用 RTTM 段本身（`_extract_windows(audio, rate, vsegs)`，无 10 秒掩蔽），
而缓存 `cache/*/w2_h1.npz` 用的是 RTTM 并集 + 前 10 秒掩蔽。两者是不同的输入。

所以复现必须照抄 spike 的取材，而不是读缓存：

    GT 段 VAD → v11._extract_windows → linkage(average/cosine)
      selK  = 穷举 2..min(10,n) 取 silhouette 最大
      trueK = 参考说话人数
      两者都过 compare_backends.labels_to_rttm 落盘再评分

    python scripts/evaluation/probe_truek.py --seconds 600
    python scripts/evaluation/probe_truek.py --from-rttm   # 只评 spike 留下的 rttm
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
from claude_a2_v12 import MAX_SPEAKERS, _v11_window_params  # noqa: E402
from der import der, read_rttm  # noqa: E402

SPIKE = Path(r"D:\yinp\AISHELL-4\_v13_spike")
DEV_GROUPS = {"L_R003C02", "M_R003C01", "S_R004C01"}


def meeting_group(meeting):
    match = re.fullmatch(r"([LMS]_R\d{3})S\d{2}(C\d{2})", meeting)
    if not match:
        raise ValueError(f"invalid AISHELL-4 meeting id: {meeting}")
    return match.group(1) + match.group(2)


def labels_to_rttm(times, labels, win_s, out_path, name):
    """照抄 `_v13_spike/compare_backends.py`，含 `win_s / 4` 这个边界偏置。"""
    labels = np.asarray(labels).reshape(-1)
    lines = []
    if len(times):
        starts = [0]
        for i in range(1, len(times)):
            if labels[i] != labels[i - 1] or times[i] - times[i - 1] > win_s * 1.5:
                starts.append(i)
        starts.append(len(times))
        for a, b in zip(starts[:-1], starts[1:]):
            start = 0.0 if a == 0 else (times[a - 1] + times[a]) / 2 + win_s / 4
            end = (times[b - 1] + win_s if b == len(times)
                   else (times[b - 1] + times[b]) / 2 + win_s / 4)
            lines.append(f"SPEAKER {name} 1 {start:.3f} {max(0.0, end - start):.3f} "
                         f"<NA> <NA> {int(labels[a]) + 1} <NA> <NA>")
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def select_k(E, cap=10):
    """spike 的选 K：穷举 2..min(cap,n) 取 silhouette 最大。"""
    tree = linkage(E, method="average", metric="cosine")
    best_k, best_score = None, None
    for k in range(2, min(cap, len(E)) + 1):
        labels = fcluster(tree, t=k, criterion="maxclust") - 1
        score = v11._silhouette(E, labels)
        if best_score is None or score > best_score:
            best_k, best_score = k, score
    return tree, best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--out", default=str(SPIKE / "work" / "der_truek"))
    ap.add_argument("--from-rttm", action="store_true",
                    help="不跑推理，只评 spike 已落盘的 rttm（对照用）")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    meetings = sorted(p.stem for p in (root / "TextGrid").glob("*.rttm")
                      if meeting_group(p.stem) in DEV_GROUPS)
    models = None
    if not args.from_rttm:
        from web_server import get_models
        models = get_models(lambda message: None, need_stream=False, need_offline=True)

    print(f"{'meeting':16s} {'trueK':>5} {'selK':>5} {'der@selK':>9} "
          f"{'der@trueK':>10} {'gap':>7}", flush=True)
    rows = []
    for meeting in meetings:
        gt = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        turns = [(s["start"], s["end"], s["speaker"]) for s in gt if s["start"] < args.seconds]
        names = sorted({speaker for _, _, speaker in turns})
        reference = [{"start": a, "end": min(b, args.seconds), "speaker": speaker}
                     for a, b, speaker in turns]

        def score_file(path):
            return der(reference, read_rttm(path), collar=args.collar)["der"] if path.exists() else None

        if args.from_rttm:
            scored = {int(f.stem.split(".k")[1]): score_file(f)
                      for f in out.glob(f"{meeting}.k*.rttm")}
            if not scored:
                continue
            sel_k = min(scored)          # spike 按文件名排序，第一个 k 就是"现行选 K"
            sel_der, true_der = scored[sel_k], scored.get(len(names))
            best_k = min(scored, key=lambda k: scored[k])
            fmt = lambda value: "   -   " if value is None else f"{value * 100:6.2f}%"
            print(f"{meeting:16s} {len(names):>5} {sel_k:>5} {fmt(sel_der)} "
                  f"{'   ' + fmt(true_der)} {(sel_der - true_der) * 100 if true_der else 0:>6.2f}% "
                  f"  (best=k{best_k} {fmt(scored[best_k])})", flush=True)
            continue

        audio, rate = sf.read(root / "wav" / f"{meeting}.flac", dtype="float32",
                              start=0, stop=int(16000 * args.seconds), always_2d=True)
        audio = audio[:, 0]
        with _v11_window_params(args.window, args.window / 2, MAX_SPEAKERS):
            E, times, _ = v11._extract_windows(audio, rate, [(a, b) for a, b, _ in turns],
                                               models["spk"])
        if len(E) < 8:
            print(f"{meeting:16s} 窗数不足", flush=True)
            continue
        tree, sel_k = select_k(E)
        out.mkdir(parents=True, exist_ok=True)

        def der_at(k):
            labels = fcluster(tree, t=k, criterion="maxclust") - 1
            path = out / f"{meeting}.k{k}.rttm"
            labels_to_rttm(times, labels, args.window, path, meeting)
            return der(reference, read_rttm(path), collar=args.collar)["der"]

        sel_der = der_at(sel_k)
        true_der = der_at(len(names)) if len(names) != sel_k else sel_der
        rows.append((meeting, len(names), sel_k, sel_der, true_der))
        print(f"{meeting:16s} {len(names):>5} {sel_k:>5} {sel_der*100:>8.2f}% "
              f"{true_der*100:>9.2f}% {(sel_der-true_der)*100:>6.2f}%", flush=True)

    if rows:
        arr = np.array([[r[3], r[4]] for r in rows])
        print(f"\n平均（{len(rows)} 场，{args.seconds:g}秒，collar={args.collar}）: "
              f"der@selK={arr[:, 0].mean()*100:.2f}%  der@trueK={arr[:, 1].mean()*100:.2f}%  "
              f"选K代价={(arr[:, 0] - arr[:, 1]).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
