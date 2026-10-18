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
