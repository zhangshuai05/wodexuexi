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

    assert [(i.start_ms, i.end_ms, i.text, i.tier, i.timed) for i in items] == [
        (1000, 2500, "你好世界", "speaker_1", True),
        (3000, 4000, "再见", "ref", True),
        # 中文数字经 normalize 归一（三 -> 3），与 CER 口径一致
        (5000, 6000, "第3句", "speaker_?", True),
    ]


def test_read_ref_text_plain_lines_are_untimed(tmp_path):
    path = tmp_path / "ref.txt"
    path.write_text("你好世界\n再见\n", encoding="utf-8")

    items = read_ref_text(path)

    assert [i.text for i in items] == ["你好世界", "再见"]
    assert all(i.timed is False for i in items)


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
    assert result["versions"]["final_fallback"]["cer"] == 0.0


def test_window_final_fallback_uses_polished_or_raw(tmp_path):
    d = _session(
        tmp_path,
        turns=[{"turn_id": "turn_0001", "text": "你好", "start_ms": 12000,
                "end_ms": 14000, "spk_label": "speaker_1"}],
        utterances=[
            {"utt_id": "u1", "start_ms": 12000, "end_ms": 14000,
             "final_text": None, "polished_text": "你好", "raw_text": "你好"},
        ],
    )
    ref = tmp_path / "ref.txt"
    ref.write_text("[00:12.0-00:14.0] speaker_1: 你好\n", encoding="utf-8")

    result = evaluate(d, ref_txt=ref, window=(11000, 19000))

    assert result["primary_version"] == "final_fallback"
    assert result["versions"]["final"]["cer"] == 1.0
    assert result["versions"]["final_fallback"]["cer"] == 0.0
    assert evaluate(d, ref_txt=ref)["textgrid"] is None


def test_window_with_no_reference_intervals_reports_empty(tmp_path):
    d = _session(
        tmp_path,
        turns=[{"turn_id": "turn_0001", "text": "你好", "start_ms": 0,
                "end_ms": 1000, "spk_label": "speaker_1"}],
    )
    ref = tmp_path / "ref.txt"
    ref.write_text("[00:00.0-00:01.0] speaker_1: 你好\n", encoding="utf-8")

    result = evaluate(d, ref_txt=ref, window=(5000, 5500))

    assert result["reference"]["empty_reference"] is True
    assert result["reference"]["source_timed"] is True
    assert result["reference"]["timed"] is False


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


def test_no_filler_cer_ignores_fillers(tmp_path):
    d = _session(tmp_path, [
        {"turn_id": "turn_0001", "text": "你好世界", "start_ms": 0,
         "end_ms": 2000, "spk_label": "speaker_1"},
    ])
    ref = tmp_path / "ref.txt"
    ref.write_text(
        "[00:00.0-00:02.0] speaker_1: 你好呃世界\n", encoding="utf-8")

    result = evaluate(d, ref_txt=ref)

    assert result["reference_source"] == "txt"
    assert result["versions"]["turns"]["cer"] == 0.2
    assert result["versions"]["turns"]["no_filler"]["cer"] == 0.0


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
