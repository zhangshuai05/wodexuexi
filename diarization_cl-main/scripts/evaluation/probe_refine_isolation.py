# -*- coding: utf-8 -*-
"""隔离两级细化：§4.1f 那条机制的**直接检验**（当前未验证）。

§4.1f 声称：离线候选表（2.0s 主窗 + AHC）能预测真实路径的低 k 行为，
但在高 k 上失效——因为高 k 切出的 2–5 窗碎簇被细化阶段（0.6s/0.15s）当作锚点，
反过来把大片区域判给了错的锚点。

那条结论当时是**推出来的**（两腿只差细化这一步），不是测出来的。
本脚本把它测出来：同一场、同一个选 K 结果，只切换"细化开/关"。

做法（不改任何冻结代码）：
  * `with_refine`   —— 正常 `v12.diarize()`
  * `without_refine` —— 上下文管理器临时把 `v12.REFINE_WIN_S` 抬到主窗长度，
    使 `_refine_windows` 产出的窗与主窗等价、`refined` 分支自然不成立；
    退出还原。**不写 v11 的模块默认值**（那会破坏录音卡历史数字的可复现性）。

判据：若在 k=8 这类高 k 上 `without_refine` 的 DER 明显低于 `with_refine`，
则细化确实在破坏高 k 结果，§4.1f 成立；否则§4.1f 要被推翻。

    python scripts/evaluation/probe_refine_isolation.py --split holdout --meeting L_R004S01C01
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v12 as v12  # noqa: E402

from der import der, read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
from unlocked_selector import unlocked  # noqa: E402

ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
SECONDS = 600.0


@contextlib.contextmanager
def refinement_disabled():
    """把 `diarize` 的细化分支整块跳过：身份用主窗选定，边界也走主窗分段。

    为什么不能靠改 `REFINE_WIN_S`：源码里 `refined = True` 的条件只是
    `rlabels is not None and len(set(rlabels)) >= 2`，与窗长无关。把细化窗长抬到
    主窗长度后，那条分支**照样会跑**（在 2.0s/0.15s 的网格上重判边缘），
    于是"禁用细化"的臂其实仍在细化，只是换了个网格——探针会静默地什么都没测到。

    正确做法是替换 `_refine_windows`，让它返回空，使 `rlabels is None` 而自然落回主窗。
    """
    original = v12._refine_windows

    def empty(*args, **kwargs):
        return np.empty((0, 0), dtype=np.float32), np.asarray([])

    v12._refine_windows = empty
    try:
        yield
    finally:
        v12._refine_windows = original


def score(reference, result):
    predicted = [{"start": float(s["start"]), "end": float(s["end"]),
                  "speaker": "v_%d" % int(s["spk"])} for s in result["segments"]]
    detail = der(reference, predicted, collar=0.25)
    level = ("refined" if result["refined"] else
             "rescued" if result["rescued"] else
             "2s-only" if not result["insufficient"] else "insufficient")
    return detail, level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout"), default="holdout")
    ap.add_argument("--meeting", help="只跑一场（不填则跑整个划分）")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--min-win", type=int, default=2)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--mode", default="largest_eligible",
                    choices=("objective", "largest_eligible", "largest"))
    ap.add_argument("--output")
    args = ap.parse_args()

    from web_server import get_models

    models = get_models(lambda m: None, need_stream=False, need_offline=False)

    meetings = [args.meeting] if args.meeting else split_meetings(ROOT_DATA, args.split)
    rows = []
    print("mode=%s cap=%d min_win=%d  分=%s" % (args.mode, args.cap, args.min_win, args.split))
    print("%-16s%5s | %-26s | %-26s | %s" % ("会议", "真K", "细化开 (k/DER/level)", "细化关 (k/DER/level)", "ΔDER(开-关)"))
    for meeting in meetings:
        reference = [s for s in read_rttm(ROOT_DATA / "TextGrid" / ("%s.rttm" % meeting))
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        duration = {}
        for s in reference:
            duration[s["speaker"]] = duration.get(s["speaker"], 0.0) + s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sum(1 for v in duration.values() if v / grand >= 0.05)

        wav = str(ROOT_DATA / "wav" / ("%s.flac" % meeting))
        with unlocked(min_win=args.min_win, cap=args.cap, mode=args.mode):
            on = v12.diarize(wav, models, max_seconds=args.seconds)
        with unlocked(min_win=args.min_win, cap=args.cap, mode=args.mode):
            with refinement_disabled():
                off = v12.diarize(wav, models, max_seconds=args.seconds)
        if off["refined"]:
            raise SystemExit(
                "关臂仍然 refined=True —— 隔离没生效，探针会静默测错。"
                "检查 _refine_windows 是否被别处缓存/预绑定。")

        don, lon = score(reference, on)
        doff, loff = score(reference, off)
        delta = don["der"] - doff["der"]
        rows.append({"meeting": meeting, "eff_k": effective,
                     "on": {**don, "k": on["k"], "level": lon},
                     "off": {**doff, "k": off["k"], "level": loff}})
        print("%-16s%5d | k=%-2d %5.1f%% %-12s | k=%-2d %5.1f%% %-12s | %+6.1f" % (
            meeting, effective, on["k"], don["der"] * 100, lon,
            off["k"], doff["der"] * 100, loff, delta * 100), flush=True)

    d_on = np.mean([r["on"]["der"] for r in rows])
    d_off = np.mean([r["off"]["der"] for r in rows])
    print("\n  DER 均值：细化开 %.1f%%  细化关 %.1f%%  差 %+.1f 点（正 = 细化有害）" % (
        d_on * 100, d_off * 100, (d_on - d_off) * 100))
    worse = sum(1 for r in rows if r["on"]["der"] > r["off"]["der"] + 0.001)
    better = sum(1 for r in rows if r["on"]["der"] < r["off"]["der"] - 0.001)
    print("  细化有害 %d 场 / 有益 %d 场 / 无差 %d 场" % (
        worse, better, len(rows) - worse - better))

    # 默认名带上会议名与 mode，避免按 --meeting 连跑时互相覆盖（本会话已踩过一次）。
    stem = args.meeting if args.meeting else args.split
    output = Path(args.output) if args.output else (
        ROOT / "artifacts" / ("v13_refine_isolation_%s_%s.json" % (stem, args.mode)))
    output.write_text(json.dumps({"schema": "v13-refine-isolation-v1",
                                  "split": args.split, "mode": args.mode,
                                  "cap": args.cap, "min_win": args.min_win,
                                  "rows": rows}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("写入 %s" % output)


if __name__ == "__main__":
    main()
