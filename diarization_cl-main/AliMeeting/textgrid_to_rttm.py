# -*- coding: utf-8 -*-
"""TextGrid → RTTM 转换器。

口径见 README §2。核心只有一条：**剥掉 `<...>` 标记后剩余文本非空才算语音段**。
这条规则由 AISHELL-4 test 20 场与主仓现行参考 RTTM 逐段比对验证（20/20 一致），

    python textgrid_to_rttm.py --validate            # 自校验，期望 20/20
    python textgrid_to_rttm.py --corpus alimeeting   # 生成 AliMeeting 参考 RTTM

生成的文件格式刻意与主仓既有参考完全一致（`der.read_rttm` 能吃）：
    SPEAKER <uri> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
"""

import argparse
import re
from pathlib import Path

from common import (ALI_RTTM, ALI_TG, AISHELL_TG, PROJECT,
                    read_textgrid)

TAG = re.compile(r"<[^>]*>")


def is_speech(text):
    """剥掉 `<...>` 标记后是否还剩文本。

    不能简单判 `text.startswith("<")`：`<#>我我不我不是说这个<sil>` 是真实语音段，
    整段以 `<` 开头但它有内容；而 `<%>` / `<$>` 剥完是空串，是非语音标记。
    """
    return bool(TAG.sub("", text).strip())


def segments_from_textgrid(path):
    """返回 [{start, end, speaker}]，按 (start, end, speaker) 排序。"""
    tiers = read_textgrid(path)
    out = [{"start": float(start), "end": float(end), "speaker": name}
           for name, intervals in tiers.items()
           for start, end, text in intervals if is_speech(text)]
    out.sort(key=lambda seg: (seg["start"], seg["end"], seg["speaker"]))
    return out


def write_rttm(path, segments, uri):
    lines = [f"SPEAKER {uri} 1 {seg['start']:.4f} {seg['end'] - seg['start']:.4f} "
             f"<NA> <NA> {seg['speaker']} <NA> <NA>"
             for seg in segments]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _reference_segments(rttm_path):
    """按主仓既有参考的读法读回，用于比对（不复用 der.read_rttm 是为了保留原始四位小数）。"""
    out = []
    for line in Path(rttm_path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            out.append((round(float(parts[3]), 4), round(float(parts[4]), 4), parts[7]))
    return sorted(out)


def _to_tuples(segments):
    return sorted((round(seg["start"], 4), round(seg["end"] - seg["start"], 4),
                   seg["speaker"]) for seg in segments)


def validate(count=0):
    """AISHELL-4 test：转换结果必须与主仓现行参考 RTTM 逐段一致。"""
    rttms = sorted(AISHELL_TG.glob("*.rttm"))
    if count:
        rttms = rttms[:count]
    passed, failures = 0, []
    for rttm in rttms:
        mine = _to_tuples(segments_from_textgrid(AISHELL_TG / f"{rttm.stem}.TextGrid"))
        reference = _reference_segments(rttm)
        if mine == reference:
            passed += 1
        else:
            only_mine = sorted(set(mine) - set(reference))
            only_ref = sorted(set(reference) - set(mine))
            failures.append((rttm.stem, len(mine), len(reference), only_mine, only_ref))

    total = len(rttms)
    print(f"AISHELL-4 test TextGrid->RTTM 自校验: {passed}/{total} 逐段完全一致")
    for stem, n_mine, n_ref, only_mine, only_ref in failures:
        print(f"  不一致 {stem}: mine={n_mine} ref={n_ref} "
              f"多人={len(only_mine)} 少人={len(only_ref)}")
        for label, items in (("仅我的", only_mine), ("仅参考", only_ref)):
            for item in items[:2]:
                print(f"    {label}: {item}")
    return passed == total


def build_alimeeting():
    """生成 AliMeeting Eval 远场的参考 RTTM。"""
    ALI_RTTM.mkdir(parents=True, exist_ok=True)
    textgrids = sorted(ALI_TG.glob("*.TextGrid"))
    if not textgrids:
        raise SystemExit(f"没找到 TextGrid，确认已解压到 {ALI_TG}")

    total_segments, speakers = 0, set()
    for path in textgrids:
        segments = segments_from_textgrid(path)
        count = write_rttm(ALI_RTTM / f"{path.stem}.rttm", segments, path.stem)
        total_segments += count
        speakers |= {seg["speaker"] for seg in segments}
        duration = max(seg["end"] for seg in segments)
        print(f"  {path.stem}: {count:5d} 段  {duration / 60:6.1f} 分钟  "
              f"{len({s['speaker'] for s in segments})} 人")
    print(f"\n共 {len(textgrids)} 场 / {total_segments} 段 / {len(speakers)} 位说话人 "
          f"-> {ALI_RTTM}")
    # 官方文档：Eval 共 25 位说话人。对不上说明 tier 名或解析出了问题。
    print(f"说话人数校验（官方 25）: {len(speakers)} -> "
          f"{'一致' if len(speakers) == 25 else '不一致，需排查'}")


def main():
    parser = argparse.ArgumentParser(description="TextGrid -> RTTM")
    parser.add_argument("--validate", action="store_true",
                        help="用 AISHELL-4 现有参考自校验")
    parser.add_argument("--corpus", choices=("alimeeting",), default=None)
    args = parser.parse_args()

    if args.validate:
        raise SystemExit(0 if validate() else 1)
    if args.corpus == "alimeeting":
        build_alimeeting()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
