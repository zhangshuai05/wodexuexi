# -*- coding: utf-8 -*-
"""端到端复核：用 `unlocked_selector` 的**同一份实现**跑 v12 真实路径。

为什么需要第二个端到端脚本：`probe_e2e_unlocked.py` 用的是自己内联的一份规则
（且继承的是 `min_win=4`）。在线下扫描证明 `min_win=2` 更好之后，探针与端到端
若各持一份实现，数字就无法互相印证。这里直接 import 共享模块，避免两份实现漂移。

不改动任何冻结代码：用 `unlocked_selector.unlocked()` 在调用期间替换 v11 的
两个函数，退出还原。

    python scripts/evaluation/probe_e2e_unlocked2.py --split dev --min-win 2
"""
import argparse
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
from unlocked_selector import CAP, MIN_WIN, unlocked  # noqa: E402

ROOT_DATA = Path(r"D:\yinp\AISHELL-4\data\test")
SECONDS = 600.0


def score(reference, result):
    predicted = [{"start": float(s["start"]), "end": float(s["end"]),
                  "speaker": "v_%d" % int(s["spk"])} for s in result["segments"]]
    detail = der(reference, predicted, collar=0.25)
    level = ("refined" if result["refined"] else
             "rescued" if result["rescued"] else
             "2s-only" if not result["insufficient"] else "insufficient")
    return detail, level, len(result["segments"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout", "all"), default="dev",
                    help="all = 该 root 下全部 rttm，不经身份组分组（train_L 的 id 带日期前缀，"
                         "分组函数会抛异常）")
    ap.add_argument("--root", default=str(ROOT_DATA), help="数据根，默认 test 划分")
    ap.add_argument("--meeting", default=None,
                    help="只跑指定场次；与 --limit 互斥，输出写入独立 meeting 后缀")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 场（0 = 全部）")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--min-win", type=int, default=MIN_WIN)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--mode", default="objective",
                    choices=("objective", "largest_eligible", "largest"),
                    help="选 K 规则；VAD 门扫描显示 largest_eligible 优于 objective")
    args = ap.parse_args()

    rows = []
    root = Path(args.root)
    if args.meeting and args.limit:
        ap.error("--meeting 与 --limit 不能同时使用")
    if args.split == "all":
        meetings = sorted(p.stem for p in (root / "TextGrid").glob("*.rttm"))
    else:
        meetings = split_meetings(root, args.split)
    if args.meeting:
        if args.meeting not in meetings:
            ap.error("场次不在所选 split/root 中: %s" % args.meeting)
        meetings = [args.meeting]
    if args.limit:
        meetings = meetings[:args.limit]
    from web_server import get_models

    models = get_models(lambda m: None, need_stream=False, need_offline=False)
    print("min_win=%d cap=%d mode=%s  分=%s  root=%s  场次=%d"
          % (args.min_win, args.cap, args.mode, args.split, root, len(meetings)), flush=True)
    print("%-16s%5s | %-30s | %-30s | %7s" % ("会议", "真K", "基线 k/DER/conf", "解锁 k/DER/conf", "ΔDER"))
    for meeting in meetings:
        reference = [s for s in read_rttm(root / "TextGrid" / ("%s.rttm" % meeting))
                     if s["start"] < args.seconds]
        reference = [{"start": s["start"], "end": min(s["end"], args.seconds),
                      "speaker": s["speaker"]} for s in reference]
        duration = {}
        for s in reference:
            duration[s["speaker"]] = duration.get(s["speaker"], 0.0) + s["end"] - s["start"]
        grand = sum(duration.values()) or 1.0
        effective = sum(1 for v in duration.values() if v / grand >= 0.05)

        wav = str(root / "wav" / ("%s.flac" % meeting))
        base = v12.diarize(wav, models, max_seconds=args.seconds)
        with unlocked(min_win=args.min_win, cap=args.cap, mode=args.mode):
            fixed = v12.diarize(wav, models, max_seconds=args.seconds)

        bd, bl, bs = score(reference, base)
        fd, fl, fs = score(reference, fixed)
        delta = bd["der"] - fd["der"]

        def cell(d, k, level, segs):
            return "k=%-2d %5.1f%% c%5.1f f%5.1f %s" % (
                k, d["der"] * 100, d["confusion_rate"] * 100,
                d["false_alarm_rate"] * 100, level)

        rows.append({"meeting": meeting, "eff_k": effective,
                     "base": {**bd, "k": base["k"], "level": bl, "n_segments": bs},
                     "fixed": {**fd, "k": fixed["k"], "level": fl, "n_segments": fs}})
        print("%-16s%5d | %-30s | %-30s | %+7.1f" % (
            meeting, effective, cell(bd, base["k"], bl, bs),
            cell(fd, fixed["k"], fl, fs), -delta * 100), flush=True)

    b = np.mean([r["base"]["der"] for r in rows])
    f = np.mean([r["fixed"]["der"] for r in rows])
    imp = sum(1 for r in rows if r["fixed"]["der"] < r["base"]["der"] - 0.001)
    wor = sum(1 for r in rows if r["fixed"]["der"] > r["base"]["der"] + 0.001)
    print("\n  %s：DER 均值 %.1f%% -> %.1f%%  (%.1f 点)  改善 %d / 恶化 %d / 不变 %d" % (
        args.split, b * 100, f * 100, (b - f) * 100, imp, wor, len(rows) - imp - wor))

    # 文件名不带 `--limit`，所以「跑一场看看」会静默覆盖整批产物
    # （实测过：把 12 场的 dev 产物写成了 1 场）。限量跑写到另一个名字。
    suffix = ("_meeting_%s" % args.meeting if args.meeting else
              "_limit%d" % args.limit if args.limit else "")
    output = ROOT / "artifacts" / (
        "v13_e2e_unlocked2_%s_mw%d_%s%s.json" % (args.split, args.min_win, args.mode, suffix))
    output.write_text(json.dumps({"schema": "v13-e2e-unlocked2-v1",
                                  "split": args.split, "root": str(root),
                                  "min_win": args.min_win,
                                  "cap": args.cap, "mode": args.mode, "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入 %s" % output)


if __name__ == "__main__":
    main()
