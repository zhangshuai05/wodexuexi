# Real-Mic Reference Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--ref-txt`, `--window`, and `--dump-draft` to `cer_eval.py` so the 2-minute real-meeting window can be annotated and scored with exact Levenshtein Sub/Del/Ins.

**Architecture:** Extend the existing `cer_eval.py` with a text-reference parser (`[mm:ss.s-mm:ss.s] speaker: text`) and a midpoint window filter. Keep the TextGrid path unchanged. `--dump-draft` reads `session.json` and writes editable draft lines. No ASR models are loaded by this tool or its tests.

**Tech Stack:** Python 3.12, rapidfuzz 3.14.6 (already installed), pytest, standard library only.

## Global Constraints

- Use `C:\Users\31817\miniconda3\envs\funasr\python.exe` (not system Python 3.13, which lacks `soundfile`).
- Run pytest from `D:\Deepseek\Meeting_main`, never from `D:\Deepseek`.
- Do not load ASR models in `cer_eval.py` or its tests.
- Do not change the existing TextGrid path's numbers or behavior.
- All text files are UTF-8; times are integer milliseconds.
- `--window` uses midpoint `(start_ms + end_ms) // 2`, inclusive on both ends.
- `--window` requires every reference line to carry a timestamp.
- `--dump-draft` must work without TextGrid or `--ref-txt`.
- Do not modify `D:\Deepseek\diarization_cl-main` (read-only upstream).
- Never hardcode `DEEPSEEK_API_KEY`; this feature does not call the API.

---

### Task 1: Time helpers and text-reference parser

**Files:**
- Modify: `cer_eval.py` (add helpers and `read_ref_text`, extend `RefInterval`)
- Test: `tests/test_cer_ref_txt.py` (create)

**Interfaces:**
- Produces:
  - `parse_time_ms(value: str) -> int`
  - `format_time_ms(ms: int) -> str`
  - `parse_window(value: str) -> tuple[int, int]`
  - `read_ref_text(path) -> list[RefInterval]`
  - `RefInterval(start_ms: int, end_ms: int, text: str, tier: str, timed: bool = True)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cer_ref_txt.py`:

```python
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import bootstrap

bootstrap.install()

from cer_eval import (  # noqa: E402
    format_time_ms,
    parse_time_ms,
    parse_window,
    read_ref_text,
)


def test_parse_time_ms_accepts_mm_ss_and_ms():
    assert parse_time_ms("01:04.2") == 64200
    assert parse_time_ms("1:02:03.5") == 3723500
    assert parse_time_ms("60000") == 60000


def test_format_time_ms_round_trip():
    assert format_time_ms(64200) == "01:04.2"
    assert format_time_ms(3723500) == "01:02:03.5"


def test_parse_window():
    assert parse_window("60000-180000") == (60000, 180000)
    assert parse_window("01:00.0-03:00.0") == (60000, 180000)


def test_read_ref_text_with_timestamps_and_speakers(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text(
        "# comment\n"
        "[00:01.0-00:02.5] speaker_1: 你好世界\n"
        "[00:03.0-00:04.0] 再见\n"
        "[00:05.0-00:06.0] speaker_?: 第三句\n",
        encoding="utf-8",
    )

    items = read_ref_text(path)

    # normalize() converts the Chinese numeral 三 -> 3, so the stored text is "第3句"
    assert [(i.start_ms, i.end_ms, i.text, i.tier, i.timed) for i in items] == [
        (1000, 2500, "你好世界", "speaker_1", True),
        (3000, 4000, "再见", "ref", True),
        (5000, 6000, "第3句", "speaker_?", True),
    ]


def test_read_ref_text_plain_lines_are_untimed(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("你好世界\n再见\n", encoding="utf-8")

    items = read_ref_text(path)

    assert [i.text for i in items] == ["你好世界", "再见"]
    assert all(i.timed is False for i in items)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
cd D:\Deepseek\Meeting_main
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `ImportError: cannot import name 'format_time_ms'` (or similar).

- [ ] **Step 3: Implement the helpers and parser**

In `cer_eval.py`, add `timed: bool = True` to `RefInterval`:

```python
@dataclass(frozen=True)
class RefInterval:
    start_ms: int
    end_ms: int
    text: str
    tier: str
    timed: bool = True
```

Add these helpers immediately before `read_textgrid`:

```python
_TIME_RE = re.compile(r"^\s*(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)\s*$")


def parse_time_ms(value: str) -> int:
    """把 mm:ss.s / hh:mm:ss.s / 纯毫秒解析成 int 毫秒。"""
    text = str(value).strip()
    if not text:
        raise ValueError("空时间")
    if re.fullmatch(r"\d+", text):
        return int(text)
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"无法解析时间: {value}")
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = float(match.group("s"))
    return int(round((hours * 3600 + minutes * 60 + seconds) * 1000))


def format_time_ms(ms: int) -> str:
    """格式化为 mm:ss.s（超过 1 小时用 hh:mm:ss.s）。"""
    total = max(0, int(ms))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis // 100}"
    return f"{minutes:02d}:{seconds:02d}.{millis // 100}"


def parse_window(value: str) -> Tuple[int, int]:
    """解析 START-END，支持 mm:ss.s / hh:mm:ss.s / 纯毫秒。"""
    text = str(value).strip()
    if "-" not in text:
        raise ValueError("窗口格式应为 START-END")
    start, end = text.split("-", 1)
    start_ms, end_ms = parse_time_ms(start), parse_time_ms(end)
    if end_ms < start_ms:
        raise ValueError("窗口结束早于开始")
    return start_ms, end_ms
```

Add the parser immediately after `reference_intervals`:

```python
_REF_TS_RE = re.compile(
    r"^\s*\[(?P<start>[^\]]+?)\s*-\s*(?P<end>[^\]]+?)\]\s*(?P<rest>.*)$")
_REF_SPK_RE = re.compile(
    r"^(?P<spk>[A-Za-z][A-Za-z0-9_?-]{0,20})\s*[:：]\s*(?P<text>.*)$")


def read_ref_text(path) -> List[RefInterval]:
    """读取 UTF-8 文本参考。

    带时间戳的行：`[mm:ss.s-mm:ss.s] speaker: text`；speaker 可省略。
    不带时间戳的行：整行都是文本，timed=False（只能全片评测）。
    """
    out: List[RefInterval] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        ts_match = _REF_TS_RE.match(line)
        if ts_match:
            start_ms = parse_time_ms(ts_match.group("start"))
            end_ms = parse_time_ms(ts_match.group("end"))
            if end_ms < start_ms:
                raise ValueError(f"参考区间结束早于开始: {line}")
            rest = ts_match.group("rest")
            spk_match = _REF_SPK_RE.match(rest)
            if spk_match:
                tier = spk_match.group("spk")
                text = spk_match.group("text")
            else:
                tier = "ref"
                text = rest
            timed = True
        else:
            start_ms = end_ms = 0
            tier = "ref"
            text = line
            timed = False
        normalized = normalize(text)
        if normalized:
            out.append(RefInterval(start_ms, end_ms, normalized, tier, timed))
    if out and all(item.timed for item in out):
        out.sort(key=lambda item: (item.start_ms, item.end_ms, item.tier))
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```powershell
git add cer_eval.py tests/test_cer_ref_txt.py
git commit -m "feat: parse text references for real-mic CER"
```

---

### Task 2: Window filtering and `evaluate` integration

**Files:**
- Modify: `cer_eval.py` (`_text_versions`, `evaluate`, `_print_report`)
- Test: `tests/test_cer_ref_txt.py` (append)

**Interfaces:**
- Consumes: `RefInterval.timed`, `read_ref_text`, `parse_window` from Task 1.
- Produces:
  - `_midpoint(start_ms: int, end_ms: int) -> int`
  - `_in_window(start_ms: int, end_ms: int, window: Optional[Tuple[int, int]]) -> bool`
  - `evaluate(session_dir, textgrid_path=None, *, ref_txt=None, window=None, skip_start_ms=0, legacy=False, pad_ms=80) -> Dict`
  - `_text_versions(obj, *, window=None) -> Dict[str, str]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cer_ref_txt.py`:

```python
from cer_eval import evaluate  # noqa: E402


def test_read_ref_text_accepts_utf8_bom(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text(
        "[00:00.0-00:01.0] speaker_1: 你好\n", encoding="utf-8-sig")

    items = read_ref_text(path)

    assert items[0].start_ms == 0
    assert items[0].text == "你好"


def _session(tmp_path, turns, utterances=None):
    d = tmp_path / "s"
    d.mkdir()
    (d / "session.json").write_text(json.dumps({
        "session_id": "s",
        "status": "transcript",
        "duration_ms": 20000,
        "utterances": utterances or [],
        "turns": turns,
    }, ensure_ascii=False), encoding="utf-8")
    return d


def test_evaluate_ref_txt_full_and_window(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好世界", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
        {"turn_id": "turn_0002", "text": "第二段内容", "start_ms": 10000,
         "end_ms": 12000, "spk_label": "speaker_2"},
    ])
    ref = tmp_path / "ref.txt"
    ref.write_text(
        "[00:00.0-00:02.0] speaker_1: 你好世界\n"
        "[00:10.0-00:12.0] speaker_2: 第二段内容\n",
        encoding="utf-8",
    )

    full = evaluate(d, ref_txt=ref)
    windowed = evaluate(d, ref_txt=ref, window=(9000, 13000))

    assert full["reference"]["chars"] == len("你好世界第二段内容")
    assert full["versions"]["turns"]["cer"] == 0.0
    assert windowed["reference"]["chars"] == len("第二段内容")
    assert windowed["versions"]["turns"]["cer"] == 0.0
    assert windowed["turns"] == 1


def test_window_uses_utterances_when_turn_spans_window(tmp_path):
    d = _session(
        tmp_path,
        turns=[{"turn_id": "turn_0001", "text": "第一段第二段",
                "start_ms": 0, "end_ms": 20000,
                "spk_label": "speaker_1"}],
        utterances=[
            {"utt_id": "u1", "start_ms": 12000, "end_ms": 14000,
             "final_text": "第二段", "raw_text": "第二段"},
            {"utt_id": "u2", "start_ms": 16000, "end_ms": 18000,
             "final_text": "内容", "raw_text": "内容"},
        ],
    )
    ref = tmp_path / "ref.txt"
    ref.write_text(
        "[00:12.0-00:14.0] speaker_1: 第二段\n"
        "[00:16.0-00:18.0] speaker_1: 内容\n",
        encoding="utf-8",
    )

    result = evaluate(d, ref_txt=ref, window=(11000, 19000))

    assert result["turns"] == 0
    assert result["primary_version"] == "final_fallback"
    assert result["versions"]["final"]["cer"] == 0.0


def test_evaluate_ref_txt_counts_substitution(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好世男", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
    ])
    ref = tmp_path / "ref.txt"
    ref.write_text("[00:00.0-00:02.0] speaker_1: 你好世界\n", encoding="utf-8")

    item = evaluate(d, ref_txt=ref)["versions"]["turns"]

    assert (item["sub"], item["del"], item["ins"]) == (1, 0, 0)
    assert item["cer"] == 0.25


def test_window_requires_timestamps(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
    ])
    ref = tmp_path / "ref.txt"
    ref.write_text("你好\n", encoding="utf-8")

    try:
        evaluate(d, ref_txt=ref, window=(0, 5000))
    except ValueError as exc:
        assert "时间戳" in str(exc)
    else:
        raise AssertionError("window + untimed ref must raise")
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `TypeError: evaluate() got an unexpected keyword argument 'ref_txt'`.

- [ ] **Step 3: Implement window helpers and integrate into `evaluate`**

Also harden `read_ref_text` against Windows BOM: change its
`Path(path).read_text(encoding="utf-8")` to `encoding="utf-8-sig"`.
This keeps the Task 1 parser unchanged for non-BOM files and prevents a
BOM-prefixed first line from being silently treated as untimed text.

Add before `_text_versions`:

```python
def _midpoint(start_ms: int, end_ms: int) -> int:
    return (int(start_ms) + int(end_ms)) // 2


def _in_window(start_ms: int, end_ms: int,
               window: Optional[Tuple[int, int]]) -> bool:
    if window is None:
        return True
    mid = _midpoint(start_ms, end_ms)
    return window[0] <= mid <= window[1]
```

Replace `_text_versions` with:

```python
def _text_versions(obj: Dict, *,
                   window: Optional[Tuple[int, int]] = None) -> Dict[str, str]:
    utterances = [
        u for u in (obj.get("utterances") or [])
        if _in_window(int(u.get("start_ms") or 0),
                      int(u.get("end_ms") or 0), window)
    ]
    raw = "".join(normalize(u.get("raw_text") or "") for u in utterances)
    polished = "".join(
        normalize(u.get("polished_text") or u.get("raw_text") or "")
        for u in utterances
    )
    final_strict = "".join(normalize(u.get("final_text") or "") for u in utterances)
    turns = [
        t for t in (obj.get("turns") or [])
        if _in_window(int(t.get("start_ms") or 0),
                      int(t.get("end_ms") or 0), window)
    ]
    turns_text = "".join(normalize(t.get("text") or "") for t in turns)
    return {
        "raw": raw,
        "polished": polished,
        "final": final_strict,
        "turns": turns_text,
    }
```

Replace the signature and the top of `evaluate` with:

```python
def evaluate(
    session_dir,
    textgrid_path=None,
    *,
    ref_txt=None,
    window: Optional[Tuple[int, int]] = None,
    skip_start_ms: int = 0,
    legacy: bool = False,
    pad_ms: int = 80,
) -> Dict:
    if (textgrid_path is None) == (ref_txt is None):
        raise ValueError("需要且只能提供 TextGrid 或 --ref-txt")
    session_dir = Path(session_dir)
    obj = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    if ref_txt is not None:
        ref_intervals = read_ref_text(ref_txt)
        if window is not None and not all(item.timed for item in ref_intervals):
            raise ValueError("--window 要求参考行全部带时间戳")
        source = "txt"
    else:
        ref_intervals = reference_intervals(
            textgrid_path, skip_start_ms=skip_start_ms)
        source = "textgrid"
    if window is not None:
        ref_intervals = [
            item for item in ref_intervals
            if _in_window(item.start_ms, item.end_ms, window)
        ]
    ref = "".join(item.text for item in ref_intervals)
```

In the same function, replace the utterance collection with:

```python
    utterances_all = obj.get("utterances") or []
    kept_utts = [
        u for u in utterances_all
        if _in_window(int(u.get("start_ms") or 0),
                      int(u.get("end_ms") or 0), window)
    ]
    kept_turns = [
        t for t in (obj.get("turns") or [])
        if _in_window(int(t.get("start_ms") or 0),
                      int(t.get("end_ms") or 0), window)
    ]
    utt_intervals = [
        (int(u.get("start_ms") or 0), int(u.get("end_ms") or 0))
        for u in kept_utts
    ]
```

Replace the attribution/coverage block with:

```python
    primary_version = "final_fallback" if window is not None else "turns"
    timed_ref = bool(ref_intervals) and all(item.timed for item in ref_intervals)
    if timed_ref:
        attribution = attribute_edits(
            ref_intervals, texts[primary_version], utt_intervals, pad_ms=pad_ms)
        raw_cover = coverage(ref_intervals, utt_intervals, pad_ms=0)
        padded_cover = coverage(ref_intervals, utt_intervals, pad_ms=pad_ms)
        overlap_chars, overlap_intervals = cross_speaker_overlap(ref_intervals)
    else:
        attribution = {}
        raw_cover = Coverage(len(ref), 0, 0.0, 0, 0.0)
        padded_cover = raw_cover
        overlap_chars = overlap_intervals = 0
```

Replace the returned `reference`, `utterances`, and `turns` fields with:

```python
        "reference": {
            "chars": len(ref),
            "intervals": len(ref_intervals),
            "ms": raw_cover.ref_ms,
            "filler_chars": sum(1 for ch in ref if ch in "呃嗯额哦"),
            "source": source,
            "timed": timed_ref,
            "textgrid": str(textgrid_path) if textgrid_path else None,
            "ref_txt": str(ref_txt) if ref_txt else None,
            "window": list(window) if window else None,
        },
        "utterances": len(kept_utts),
        "turns": len(kept_turns),
        "primary_version": primary_version,
```

Replace `_print_report` with the complete function:

```python
def _print_report(result: Dict) -> None:
    ref = result["reference"]
    print(f"session={result['session']}  ref={ref['chars']}  "
          f"ref_intervals={ref['intervals']}  utts={result['utterances']}  "
          f"turns={result['turns']}")
    if ref.get("source") == "txt":
        print(f"  ref_txt={ref['ref_txt']}")
    if ref.get("window"):
        start, end = ref["window"]
        print(f"  window={format_time_ms(start)}-{format_time_ms(end)}")
    if result.get("primary_version"):
        print(f"  主口径={result['primary_version']}")
    if result["skip_start_ms"]:
        print(f"  （已跳过前 {result['skip_start_ms'] / 1000:.1f}s）")
    if ref.get("filler_chars"):
        print(f"  参考口癖字符（呃/嗯/额/哦）: {ref['filler_chars']}"
              f" ({ref['filler_chars'] / ref['chars']:.2%})")

    print("  version    CER       Sub       Del       Ins     "
          "S       D       I    hyp")
    for name, item in result["versions"].items():
        print(f"  {name:9s} {item['cer']:8.2%} {item['sub_rate']:8.2%} "
              f"{item['del_rate']:8.2%} {item['ins_rate']:8.2%} "
              f"{item['sub']:6d}  {item['del']:6d}  {item['ins']:6d}  "
              f"{item['hyp_len']:6d}")
        if "legacy_lcs_cer" in item:
            print(f"  {'':9s} legacy difflib/LCS CER={item['legacy_lcs_cer']:.2%}"
                  f"（仅历史对照，不得与新口径混用）")

    if not ref.get("timed", True):
        print("  （参考无时间戳：跳过覆盖率与归因）")
    else:
        attribution = result.get("attribution") or {}
        if attribution:
            print("  错误归因（turns 口径，按参考区间精确归因）：")
            for dimension, title in (
                ("coverage", "VAD 覆盖"),
                ("overlap", "跨说话人"),
                ("tier", "说话人"),
            ):
                for key, item in attribution.get(dimension, {}).items():
                    print(f"    [{title}] {key:12s} ref={item['ref_chars']:5d} "
                          f"S={item['sub']:5d} D={item['del']:5d} I={item['ins']:5d} "
                          f"CER={item['cer']:7.2%} "
                          f"(Sub {item['sub_rate']:.2%} / Del {item['del_rate']:.2%} / "
                          f"Ins {item['ins_rate']:.2%})")

        cov = result["coverage"]
        print("  覆盖/重叠：")
        print(f"    VAD 原始未覆盖参考字符: {cov['vad_uncovered_chars']:.0f}"
              f" / {ref['chars']} ({cov['vad_uncovered_ratio']:.2%})，"
              f"时长 {cov['vad_uncovered_ms'] / 1000:.1f}s")
        print(f"    加 80ms pad 后未覆盖:   {cov['padded_uncovered_chars']:.0f}"
              f" ({cov['padded_uncovered_ratio']:.2%})，"
              f"时长 {cov['padded_uncovered_ms'] / 1000:.1f}s")
        print(f"    跨说话人重叠参考字符:   {cov['cross_speaker_overlap_chars']}"
              f" ({cov['cross_speaker_overlap_ratio']:.2%})，"
              f"{cov['cross_speaker_overlap_intervals']} 个区间")
        print("  分说话人：")
        for tier, stat in sorted(result["tiers"].items()):
            print(f"    {tier}: chars={stat['chars']} intervals={stat['intervals']} "
                  f"overlap_chars={stat['overlap_chars']}")
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `12 passed`.

- [ ] **Step 5: Commit**

```powershell
git add cer_eval.py tests/test_cer_ref_txt.py
git commit -m "feat: evaluate windowed text references"
```

---

### Task 3: `--dump-draft` and CLI wiring

**Files:**
- Modify: `cer_eval.py` (`dump_draft`, `main`)
- Test: `tests/test_cer_ref_txt.py` (append)

**Interfaces:**
- Consumes: `_in_window`, `format_time_ms`, `parse_window`.
- Produces:
  - `dump_draft(session_dir, output_path, *, window=None) -> int`
  - `main(argv=None) -> int` with `--ref-txt`, `--window`, `--dump-draft`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
from cer_eval import dump_draft, main  # noqa: E402


def test_dump_draft_writes_windowed_lines(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "第一段", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
        {"turn_id": "turn_0002", "text": "第二段", "start_ms": 10000,
         "end_ms": 12000, "spk_label": "speaker_2"},
    ])
    out = tmp_path / "draft.txt"

    count = dump_draft(d, out, window=(9000, 13000))

    assert count == 1
    assert out.read_text(encoding="utf-8") == (
        "[00:10.0-00:12.0] speaker_2: 第二段\n")


def test_dump_draft_prefers_utterances_and_strips_cjk_spaces(tmp_path):
    d = _session(
        tmp_path,
        turns=[{"turn_id": "turn_0001", "text": "整段", "start_ms": 0,
                "end_ms": 20000, "spk_label": "speaker_1"}],
        utterances=[
            {"utt_id": "u1", "start_ms": 10000, "end_ms": 12000,
             "final_text": "第 二 段", "raw_text": "第二段",
             "spk_label": "speaker_2"},
        ],
    )
    out = tmp_path / "draft.txt"

    count = dump_draft(d, out, window=(9000, 13000))

    assert count == 1
    assert out.read_text(encoding="utf-8") == (
        "[00:10.0-00:12.0] speaker_2: 第二段\n")


def test_main_dump_draft_does_not_require_reference(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
    ])
    out = tmp_path / "draft.txt"

    assert main([str(d), "--dump-draft", str(out)]) == 0
    assert out.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `ImportError: cannot import name 'dump_draft'`.

- [ ] **Step 3: Implement `dump_draft` and CLI**

Add before `main`:

```python
_CJK_SPACE_RE = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def _readable(text: str) -> str:
    """去掉汉字之间的空格（SEACO 输出为空格分隔 token）。"""
    return _CJK_SPACE_RE.sub("", text or "")


def dump_draft(session_dir, output_path, *,
               window: Optional[Tuple[int, int]] = None) -> int:
    """从 session.json 导出可人工修正的草稿；返回行数。

    优先 utterances（逐句），没有 utterances 时才回退 turns。
    """
    obj = json.loads(
        (Path(session_dir) / "session.json").read_text(encoding="utf-8"))
    utterances = obj.get("utterances") or []
    turns = obj.get("turns") or []
    use_utterances = bool(utterances)
    items = utterances if use_utterances else turns

    lines: List[str] = []
    for item in items:
        start_ms = int(item.get("start_ms") or 0)
        end_ms = int(item.get("end_ms") or 0)
        if not _in_window(start_ms, end_ms, window):
            continue
        if use_utterances:
            text = (item.get("final_text") or item.get("polished_text")
                    or item.get("raw_text") or "")
        else:
            text = item.get("text") or ""
        text = _readable(text)
        speaker = item.get("spk_label") or "speaker_?"
        lines.append(
            f"[{format_time_ms(start_ms)}-{format_time_ms(end_ms)}] "
            f"{speaker}: {text}")
    Path(output_path).write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
```

Replace `main` with:

```python
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CER（Levenshtein + Sub/Del/Ins，支持 TextGrid 或 --ref-txt）")
    parser.add_argument("session_dir")
    parser.add_argument("textgrid", nargs="?")
    parser.add_argument("--ref-txt", default=None,
                        help="纯文本参考；与 TextGrid 二选一")
    parser.add_argument("--window", default=None,
                        help="只评该时间窗，如 60000-180000 或 01:00.0-03:00.0")
    parser.add_argument("--dump-draft", default=None,
                        help="从 session.json 导出草稿后退出（不需要参考输入）")
    parser.add_argument("--skip-start-ms", type=int, default=0,
                        help="跳过开头 N 毫秒（TextGrid 路径）")
    parser.add_argument("--legacy", action="store_true",
                        help="额外打印旧 difflib/LCS 口径，仅作历史对照")
    parser.add_argument("--pad-ms", type=int, default=80,
                        help="离线二遍两侧 pad（覆盖率诊断用，默认 80ms）")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    window = parse_window(args.window) if args.window else None

    if args.dump_draft:
        count = dump_draft(args.session_dir, args.dump_draft, window=window)
        print(f"draft_lines={count} path={args.dump_draft}")
        return 0

    if args.textgrid and args.ref_txt:
        parser.error("TextGrid 与 --ref-txt 只能二选一")
    if not args.textgrid and not args.ref_txt:
        parser.error("需要 TextGrid 或 --ref-txt")

    # skip_start_ms 只适用于 TextGrid；ref-txt 路径置 0，避免 --json 误导
    skip_start_ms = args.skip_start_ms if args.textgrid else 0
    result = evaluate(
        args.session_dir,
        args.textgrid,
        ref_txt=args.ref_txt,
        window=window,
        skip_start_ms=skip_start_ms,
        legacy=args.legacy,
        pad_ms=args.pad_ms,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result)
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `15 passed`.

- [ ] **Step 5: Commit**

```powershell
git add cer_eval.py tests/test_cer_ref_txt.py
git commit -m "feat: add windowed draft export to cer_eval"
```

---

### Task 4: TextGrid regression, full suite, and real draft

**Files:**
- Modify: `tests/test_cer_ref_txt.py` (append regression test)
- Manual: `output/sessions/s_real_intel/`, `output/real_meeting/`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: no new code; verifies the TextGrid path and the real command.

- [ ] **Step 1: Add the TextGrid regression test**

Append:

```python
def test_textgrid_path_still_works(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好世界", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
    ])
    tg = tmp_path / "a.TextGrid"
    tg.write_text(
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n'
        'xmin = 0\n'
        'xmax = 2\n'
        'tiers? <exists>\n'
        'size = 1\n'
        'item [1]:\n'
        '    class = "IntervalTier"\n'
        '    name = "speaker_1"\n'
        '    xmin = 0\n'
        '    xmax = 2\n'
        '    intervals: size = 1\n'
        '    intervals [1]:\n'
        '        xmin = 0\n'
        '        xmax = 2\n'
        '        text = "你好世界"\n',
        encoding="utf-8",
    )

    result = evaluate(d, tg)

    assert result["reference"]["source"] == "textgrid"
    assert result["versions"]["turns"]["cer"] == 0.0
```

- [ ] **Step 2: Run the new test**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_cer_ref_txt.py -v
```

Expected: `16 passed`.

- [ ] **Step 3: Run the full fast suite**

```powershell
cd D:\Deepseek\Meeting_main
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest -q
```

Expected: all tests pass; count is at least `147 passed, 3 deselected`.

- [ ] **Step 4: Generate the real 60–180s draft**

After `output/sessions/s_real_intel/session.json` exists (the 10-minute
pipeline run must have finished):

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe cer_eval.py `
  output/sessions/s_real_intel `
  --window 60000-180000 `
  --dump-draft output/real_meeting/draft_60_180.txt
```

Expected: `draft_lines=22 path=output/real_meeting/draft_60_180.txt`
（当前真实 session 60–180s 内有 22 句；改窗口时行数会变）。

- [ ] **Step 5: Manual annotation (human step, not code)**

1. Copy `draft_60_180.txt` to `ref_60_180.txt`.
2. Listen to `output/real_meeting/intel_meeting_10min_16k.wav` from 60s to 180s.
3. Correct every line in `ref_60_180.txt`; keep the `[start-end] speaker:`
   format. Add speaker labels manually if the draft has `speaker_?`.
   The draft is per utterance and already has spaces between CJK characters
   removed; still correct every sentence against the audio.
4. Do one 30-second blind re-listen and fix missed errors.

- [ ] **Step 6: Score the real window**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe cer_eval.py `
  output/sessions/s_real_intel `
  --ref-txt output/real_meeting/ref_60_180.txt `
  --window 60000-180000
```

Expected: a table with `主口径=final_fallback` and `versions.final_fallback`
CER / Sub / Del / Ins, plus coverage/overlap attribution (the reference lines
carry timestamps). `versions.final` is the strict diagnostic (missing
`final_text` counts as empty).
`versions.turns` may be empty (0) because the real turn spans the window;
that is expected and why the window primary is `final`.

- [ ] **Step 7: Record the result and commit**

Add a short “真实麦克风基线” section to `HANDOFF.md` with:

- window (60–180s), device (Intel array), reference file path
- CER / Sub / Del / Ins
- the caveat that this is a 2-minute sample and an array upper bound

```powershell
git add cer_eval.py tests/test_cer_ref_txt.py HANDOFF.md
git commit -m "docs: record real-mic baseline and add ref-txt tooling"
```

---

## Self-Review

- **Spec coverage:** `--ref-txt`, `--window`, `--dump-draft`, midpoint
  filtering, timestamp requirement, **utterance-level window primary**,
  TextGrid regression, real draft/scoring, and the HANDOFF update all appear
  in Tasks 1–4.
- **Placeholder scan:** no TBD/TODO; every code step contains the exact code;
  every test step contains the exact assertion.
- **Type consistency:** `RefInterval.timed` is added in Task 1 and used by
  `read_ref_text`, `evaluate`, and `dump_draft`. `_in_window` is defined in
  Task 2 and reused in Task 3. `parse_window` is defined in Task 1 and called
  by `main` in Task 3. `primary_version` is produced by `evaluate` in Task 2
  and printed by `_print_report`; `dump_draft` prefers utterances in Task 3.
