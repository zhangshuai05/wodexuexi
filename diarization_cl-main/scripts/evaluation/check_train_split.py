# -*- coding: utf-8 -*-
"""train_L 核验：它能不能当**真正的留出集**，以及能不能直接喂给现有评测链路。

背景：本项目此前所有结论都卡在"dev 与 holdout 都是样本内"——AISHELL-4 的 test 划分
只有 20 场，`min_win` 等参数是在全部 20 场上选的。SLR111 另发有 train 划分
（`train_S` 14G / `train_M` 25G / `train_L` 7G），但从未下载。

到货后要回答三件事，**顺序不能颠倒**：

  1. **身份组是否重叠**（最关键）：test 的 `R` 号只有 R003/R004，而 train_L 的
     前若干条是 R001/R002（带日期前缀）。若 train 里出现 `L_R003*C02`，
     它就含 dev 的同一批说话人 —— **那就不是留出集，不能用**。
  2. **规模与人数分布**：多少场、几通道、说话人数范围，是否与 test 同量级。
  3. **标注布局是否同构**：`<dir>/TextGrid/<meeting>.rttm` 能否被 `der.read_rttm`
     直接读、`speaker_id` 是否形如 `001-M`。不同构则要走 `sd/scripts/` 的转换。

⚠ **前提（未经证实）**：本脚本按 test 的布局假设 train 也是
`wav/<meeting>.flac` + `TextGrid/<meeting>.rttm`。这个假设**没有 train 侧来源**——
`DATA_SPEC.md` §6.6 明写"当前只有测试集"，train 三个包从未下载过。
**train_L.tar.gz 一解开就先 `tar tzf | head -50` 看真实布局**，别让本脚本的假设
变成"核验过"的假象。布局不符时下面会直接报"找不到 TextGrid"，那是**真结论**，
不是脚本坏了。

    python -u -B scripts/evaluation/check_train_split.py --dir D:\\yinp\\AISHELL-4\\data\\train_L
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))

from der import read_rttm  # noqa: E402

# 本项目现用的划分（来自 v12_aishell4.py / build_cache.py），train 不得与它们撞身份组
DEV_GROUPS = {"L_R003C02", "M_R003C01", "S_R004C01"}
HOLDOUT_GROUPS = {"L_R004C01", "S_R003C01"}


def meeting_group(meeting_id):
    """去掉日期前缀与 S## 段，得到身份组。

    test 的 id 形如 `L_R003S01C02`；train 的形如 `20200706_L_R001S08C01`。
    两种都归一到 `L_R001C01` 这种形状。
    """
    body = re.sub(r"^\d{8}_", "", meeting_id)
    match = re.fullmatch(r"([LMS]_R\d{3})S\d{2}(C\d{2})", body)
    return (match.group(1) + match.group(2)) if match else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="解包后的 train 划分目录")
    ap.add_argument("--limit", type=int, default=0, help="只查前 N 场（0 = 全部）")
    args = ap.parse_args()

    root = Path(args.dir)
    tg = root / "TextGrid"
    if not tg.is_dir():
        raise SystemExit(f"找不到 {tg}；先解包 train_*.tar.gz")

    rttms = sorted(tg.glob("*.rttm"))
    if args.limit:
        rttms = rttms[:args.limit]
    print(f"  目录 {root}")
    print(f"  rttm {len(rttms)} 个")
    if not rttms:
        raise SystemExit("没有 rttm，标注布局与 test 不同构")

    # ---- 1. 身份组重叠（决定性）----
    groups, speaker_counts, speaker_ids, spans = {}, {}, set(), []
    for path in rttms:
        segs = read_rttm(path)
        if not segs:
            continue
        g = meeting_group(path.stem)
        groups.setdefault(g, []).append(path.stem)
        names = sorted({s["speaker"] for s in segs})
        speaker_counts[path.stem] = len(names)
        speaker_ids |= set(names)
        spans.append(max(s["end"] for s in segs))

    overlap_dev = sorted(set(groups) & DEV_GROUPS)
    overlap_hold = sorted(set(groups) & HOLDOUT_GROUPS)
    print(f"\n=== 1. 身份组重叠（决定性）===")
    print(f"  本划分共 {len(groups)} 个身份组")
    print(f"  与 **dev** 撞的组: {overlap_dev or '无'}")
    print(f"  与 **holdout** 撞的组: {overlap_hold or '无'}")
    if overlap_dev or overlap_hold:
        print("  -> **不能当留出集**：含本项目已用过的同一批说话人")
    else:
        print("  -> **可以当留出集**（身份组完全不重叠）")

    # ---- 2. 规模与人数 ----
    print(f"\n=== 2. 规模 ===")
    if speaker_counts:
        counts = Counter(speaker_counts.values())
        print(f"  场次 {len(speaker_counts)}")
        print(f"  每场说话人数: 范围 {min(counts)}–{max(counts)}，分布 {dict(sorted(counts.items()))}")
    print(f"  speaker_id 样例: {sorted(speaker_ids)[:8]}")
    if spans:
        total = sum(spans)
        print(f"  单场时长范围 {min(spans)/60:.1f}–{max(spans)/60:.1f} 分钟；合计 {total/3600:.1f} 小时")

    # ---- 3. 布局同构 ----
    print(f"\n=== 3. 布局同构 ===")
    has_wav = (root / "wav").is_dir()
    flacs = list((root / "wav").glob("*.flac")) if has_wav else []
    print(f"  wav/ 存在: {has_wav}   flac 数: {len(flacs)}")
    print(f"  TextGrid/ 存在: {tg.is_dir()}")
    sample = sorted(speaker_ids)[:1]
    looks_like_ref = bool(sample) and re.fullmatch(r"\d{3}-[MF]", sample[0]) is not None
    print(f"  speaker_id 形如 `001-M`: {looks_like_ref}")
    print(f"  -> read_rttm 可直接读: True（上面已用）")
    if not looks_like_ref:
        print("     ⚠ 命名不同构，需先确认与 test 的对应关系再评测")

    # ---- 4. 与 test 的组级对照 ----
    print(f"\n=== 4. 组级对照 ===")
    print(f"  train 的组: {sorted(g for g in groups if g)}")
    print(f"  dev 的组:   {sorted(DEV_GROUPS)}")
    print(f"  holdout 的组: {sorted(HOLDOUT_GROUPS)}")


if __name__ == "__main__":
    main()
