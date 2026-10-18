# -*- coding: utf-8 -*-
"""锁住 2026-09-28 OCR 复审发现的四个静默失效。

四个问题都不报错、只出错数，是本会话反复出现的同一类陷阱：

1. **段标签类型不匹配**（`probe_path_equiv`）。`diarize` 的 `spk` 是 `int`，
   探针侧 `segment_map` 经 `_name()` 是 `str`。`0 != "0"` 恒真，
   于是"逐段不同数"**恒等于段数**——看起来在正常报差异，实则比较完全失效。
   实证：v1 产物里 `n_prefix_mismatch == n_prefix_compared`（132/132、231/231）。

2. **`--seconds` 只写 manifest、不影响计算**（`build_probe_cache`）。
   `--seconds 300` 会记录 "300" 却按 600 实算，manifest 与实际窗集不一致。

3. **`insufficient` 早返回键不齐**（`probe_ab_dev`）。缺 `candidates` 等键，
   调用方 `r["candidates"]` 抛 `KeyError` 被 `except` 吞掉、记成 `***失败***`，
   真因（窗数不足）被伪装成假异常——违反 §R7。

4. **`archive/` 下 `parents[2]` 深度错**。搬目录后没改，ROOT 落在 `scripts/`
   而非仓库根，导入静默失败。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_path_equiv as equiv  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402


# ───────────────────── 1. 段标签类型 ─────────────────────

def test_segments_of_returns_string_labels_matching_the_probe_side():
    """`segments_of` 的标签必须是 `str`，与 `segment_map`/`_name()` 同类型。

    回归：v1 用 `int(s["spk"])`，与探针侧的字符串比 `!=` 恒真，
    导致"逐段不同数"永远等于比较的段数。
    """
    fake = {"segments": [{"start": 0.0, "end": 1.0, "spk": 0},
                         {"start": 1.0, "end": 2.0, "spk": 3}]}
    got = equiv.segments_of(fake)
    assert got == [(0.0, 1.0, "0"), (1.0, 2.0, "3")], got
    assert all(isinstance(label, str) for _, _, label in got)


def test_integer_zero_label_compares_equal_across_the_two_sides():
    """0 号簇必须**跨口径相等**。这是"整数 0 falsy"陷阱的同族第四条。"""
    diarize_side = [(0.0, 1.0, "0")]
    probe_side = [(0.0, 1.0, P._name(0))]          # `_name(0)` -> "0"
    assert diarize_side == probe_side, "0 号簇两侧不等价，段级比较会假报差异"


def test_identical_segment_tables_compare_equal():
    """两侧同源段表**必须**判等——否则自证断言失去意义。"""
    segs = [(0.0, 1.0, "0"), (1.0, 2.5, "1")]
    assert equiv.segments_of({"segments": [{"start": s, "end": e, "spk": int(l)}
                                           for s, e, l in segs]}) == segs


# ───────────────────── 2. --seconds 必须真的生效 ─────────────────────

def test_live_vad_uses_the_seconds_argument_not_the_module_constant():
    """`live_vad` 的截断长度必须来自参数，不是模块级 `SECONDS`。

    回归：`--seconds 300` 只写进 manifest，VAD 截断仍按 600——
    manifest 元信息与实际窗集不一致，且无任何提示。
    """
    import build_probe_cache as bpc

    class FakeVad:
        """返回跨过两个截断点的区间，以便区分 300 与 600。"""

    intervals = [(0.0, 100.0), (200.0, 350.0), (500.0, 700.0)]
    orig = bpc.v11._vad_segs
    bpc.v11._vad_segs = lambda *a, **k: list(intervals)
    try:
        got300 = bpc.live_vad({"vad": None}, "x.flac", 300.0)
        got600 = bpc.live_vad({"vad": None}, "x.flac", 600.0)
    finally:
        bpc.v11._vad_segs = orig

    assert got300 == [(0.0, 100.0), (200.0, 300.0)], got300
    assert got600 == [(0.0, 100.0), (200.0, 350.0), (500.0, 600.0)], got600
    assert got300 != got600, "两种截断长度给出同一结果，说明参数没生效"


def test_build_meeting_forwards_seconds_to_the_truncation():
    """`build_meeting` 必须把 `seconds` 传下去。签名检查，避免又只写不传。"""
    import inspect
    import build_probe_cache as bpc

    params = inspect.signature(bpc.build_meeting).parameters
    assert "seconds" in params, "build_meeting 缺 seconds 参数"
    assert "seconds" in inspect.signature(bpc.live_vad).parameters, \
        "live_vad 缺 seconds 参数"


# ───────────────────── 3. insufficient 早返回键必须齐 ─────────────────────

def test_insufficient_early_return_has_all_keys_the_caller_reads():
    """窗数不足时返回的字典必须含调用方会读的**全部**键。

    回归：早先只给 `meeting/insufficient/n_windows`，`main` 里
    `r["candidates"]` 抛 KeyError → 被 `except` 吞 → 记成 `***失败***`，
    真因被伪装。这里用真 `diarize` 的不足分支产出，再按调用方的方式读一遍。
    """
    import inspect
    import probe_ab_dev as ab

    src = inspect.getsource(ab.run_meeting)
    # 早返回那一行必须与完整返回同构：抽查关键键出现在同一个 return 语句里
    early = src.split('insufficient": True')[1].split("\n\n")[0]
    for key in ('"n_dense"', '"candidates"', '"n_sentences"'):
        assert key in early, f"insufficient 早返回缺 {key}"


# ───────────────────── 4. archive/ 的 ROOT 深度 ─────────────────────

def test_archive_scripts_resolve_repo_root_not_scripts_dir():
    """`archive/` 下脚本的 ROOT 必须是**仓库根**。

    回归：搬进 archive/ 后仍用 `parents[2]`，指向 `scripts/`，
    `ROOT/"streaming_asr"` 不存在，导入静默失败。
    """
    import re
    archive = ROOT / "scripts" / "evaluation" / "archive"
    scripts = sorted(archive.glob("*.py"))
    assert scripts, "archive/ 下没有脚本，测试失去对象"
    for path in scripts:
        src = path.read_text(encoding="utf-8")
        m = re.search(r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]",
                      src, re.M)
        assert m, f"{path.name}: 找不到 ROOT 定义"
        depth = int(m.group(1))
        assert depth == 3, (
            f"{path.name}: parents[{depth}] —— archive/ 下应是 parents[3]，"
            "否则 ROOT 落在 scripts/ 而非仓库根"
        )
        # 实证：按该深度解析出来的目录必须真含这两个子目录
        resolved = path.resolve().parents[depth]
        assert (resolved / "streaming_asr").is_dir(), f"{path.name}: ROOT 下无 streaming_asr"
        assert (resolved / "scripts" / "evaluation").is_dir(), \
            f"{path.name}: ROOT 下无 scripts/evaluation"
