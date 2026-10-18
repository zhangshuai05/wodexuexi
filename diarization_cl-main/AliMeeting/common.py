# -*- coding: utf-8 -*-
"""路径与主仓复用入口。

这个子项目**不复制**任何评分或解析代码：DER 评分器与 TextGrid 读取器都从主仓导入，
保证本目录的数字和主仓的数字走的是同一份实现（主仓 README 反复强调口径一致性，
两份会分叉的实现比没有实现更糟）。
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
ASR_DIR = REPO / "streaming_asr"
EVAL_DIR = REPO / "scripts" / "evaluation"

for _path in (str(ASR_DIR), str(EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ── 数据路径 ──────────────────────────────────────────────────────────
ALI_ROOT = Path(r"D:\yinp\AliMeeting\Eval_Ali")
ALI_FAR = ALI_ROOT / "Eval_Ali_far"
ALI_TG = ALI_FAR / "textgrid_dir"
ALI_CH0 = ALI_ROOT / "Eval_Ali_ch0"
ALI_RTTM = ALI_ROOT / "reference_rttm"

AISHELL_ROOT = Path(r"D:\yinp\AISHELL-4\data\test")
AISHELL_TG = AISHELL_ROOT / "TextGrid"

RESULTS = PROJECT / "results"
PREDICTIONS = PROJECT / "predictions"

# 与主仓 der.py 一致的时间量化栅格（10 ms）
FRAME = 0.01

# 主仓 DER 口径的默认 collar
COLLAR = 0.25


def der_tools():
    """主仓的 DER 评分器与 RTTM 读写。"""
    from der import der, read_rttm  # noqa: E402  (主仓 streaming_asr/der.py)
    return der, read_rttm


def resolve_wav(meeting):
    """会议 id → ch0 单声道音频。

    目录里音频名带会话后缀（`R8001_M8004_MS801.wav`），而 TextGrid 与参考 RTTM
    的名字是 `R8001_M8004`。按前缀解析，并且**要求唯一命中**——多命中说明命名
    规则不是我以为的那样，此时静静地取第一个会拿错音频，必须直接报错。
    """
    matches = sorted(ALI_CH0.glob(f"{meeting}_*.wav"))
    if len(matches) != 1:
        raise SystemExit(f"{meeting}: 在 {ALI_CH0} 里命中 {len(matches)} 个音频，"
                         f"期望 1 个。先跑 prepare_ch0.py，或检查命名。")
    return matches[0]


def read_textgrid(path):
    """主仓的 ooTextFile IntervalTier 读取器。"""
    from probe_sentence_attribution import read_textgrid as _read  # noqa: E402
    return _read(path)
