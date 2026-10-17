# -*- coding: utf-8 -*-
"""turn 合并层测试：join_texts / build_turns / StreamTurnTracker。"""
from turns import (
    FINAL_TURN_SPLIT_MS,
    STREAM_TURN_GAP_MS,
    StreamTurnTracker,
    build_turns,
    join_texts,
)


def _utt(uid, start, end, text, spk_label=None, spk=None, polished=None,
         corrected=False, bypassed=False):
    return {
        "utt_id": uid,
        "start_ms": start,
        "end_ms": end,
        "raw_text": text,
        "polished_text": text if polished is None else polished,
        "spk": spk,
        "spk_label": spk_label,
        "corrected": corrected,
        "bypassed": bypassed,
    }


# ---- join_texts ----

def test_join_texts_adds_comma_between_cjk_fragments():
    assert join_texts(["大家", "早上好"]) == "大家，早上好"
    # 已有标点的边界不再补
    assert join_texts(["你好，", "世界。"]) == "你好，世界。"
    assert join_texts(["  你好 ", "", "世界 "]) == "你好，世界"


def test_join_texts_single_space_between_ascii():
    assert join_texts(["hello", "world"]) == "hello world"
    assert join_texts(["hello", "  ", "again"]) == "hello again"
    # 任一侧是中文时补逗号而不是空格
    assert join_texts(["hello", "你好"]) == "hello，你好"
    assert join_texts(["你好", "world"]) == "你好，world"


# ---- build_turns ----

def test_build_turns_merges_same_speaker_and_splits_on_gap_or_speaker():
    utts = [
        _utt("utt_0001", 0, 1000, "大家", spk_label="speaker_0", spk=0),
        # 间隔 2000ms < 8000ms，同一说话人 -> 合并
        _utt("utt_0002", 3000, 4000, "早上好", spk_label="speaker_0", spk=0),
        # 间隔 9000ms >= 8000ms -> 同一说话人也断开
        _utt("utt_0003", 13000, 14000, "开始吧", spk_label="speaker_0", spk=0),
        # 说话人切换 -> 必然断开
        _utt("utt_0004", 14500, 15000, "好", spk_label="speaker_1", spk=1),
    ]

    turns = build_turns(utts)

    assert [turn["turn_id"] for turn in turns] == [
        "turn_0001", "turn_0002", "turn_0003",
    ]
    assert turns[0]["utterance_ids"] == ["utt_0001", "utt_0002"]
    assert turns[0]["text"] == "大家，早上好"
    assert turns[0]["start_ms"] == 0
    assert turns[0]["end_ms"] == 4000
    assert turns[0]["spk"] == 0
    assert turns[0]["spk_label"] == "speaker_0"
    assert turns[1]["utterance_ids"] == ["utt_0003"]
    assert turns[2]["utterance_ids"] == ["utt_0004"]


def test_build_turns_split_gap_is_strict():
    utts = [
        _utt("utt_0001", 0, 1000, "一", spk_label="speaker_0", spk=0),
        # 恰好等于阈值 -> 新 turn（只在 < 阈值时合并）
        _utt("utt_0002", 1000 + FINAL_TURN_SPLIT_MS, 12000, "二",
             spk_label="speaker_0", spk=0),
    ]

    turns = build_turns(utts)

    assert len(turns) == 2


def test_build_turns_without_labels_uses_stream_gap():
    utts = [
        _utt("utt_0001", 0, 1000, "一"),
        _utt("utt_0002", 2500, 3500, "二"),     # 间隔 1500 < 2000 -> 合并
        _utt("utt_0003", 5600, 6600, "三"),     # 间隔 2100 >= 2000 -> 断开
        _utt("utt_0004", 7000, 8000, "四"),     # 间隔 400 -> 合并
    ]

    turns = build_turns(utts)

    assert [turn["utterance_ids"] for turn in turns] == [
        ["utt_0001", "utt_0002"],
        ["utt_0003", "utt_0004"],
    ]
    assert len(turns) > 1, "无说话人信息时不得退化成整场一个 turn"
    assert turns[0]["spk_label"] is None
    assert turns[0]["spk"] is None


def test_build_turns_gap_threshold_matches_stream_constant():
    utts = [
        _utt("utt_0001", 0, 1000, "一"),
        _utt("utt_0002", 1000 + STREAM_TURN_GAP_MS, 5000, "二"),
    ]

    assert len(build_turns(utts)) == 2


def test_turn_ids_ordered_and_zero_padded():
    utts = [
        _utt(f"utt_{index:04d}", index * 10000, index * 10000 + 1000, "字",
             spk_label="speaker_0", spk=0)
        for index in range(1, 4)
    ]

    turns = build_turns(utts)

    assert [turn["turn_id"] for turn in turns] == [
        "turn_0001", "turn_0002", "turn_0003",
    ]


def test_build_turns_prefers_polished_text():
    utts = [
        _utt("utt_0001", 0, 1000, "你好", polished="你好。",
             spk_label="speaker_0", spk=0),
        _utt("utt_0002", 1500, 2500, "再见", polished="",
             spk_label="speaker_0", spk=0),
    ]

    turns = build_turns(utts)

    assert len(turns) == 1
    assert turns[0]["text"] == "你好。再见"


def test_build_turns_empty_input():
    assert build_turns([]) == []


def test_build_turns_propagates_corrected_and_bypassed_flags():
    utts = [
        # 同一说话人、间隔足够小 -> 合并为一个 turn
        _utt("utt_0001", 0, 1000, "一", spk_label="speaker_0", spk=0,
             corrected=True, bypassed=False),
        _utt("utt_0002", 1500, 2500, "二", spk_label="speaker_0", spk=0,
             corrected=False, bypassed=False),
        _utt("utt_0003", 3000, 4000, "三", spk_label="speaker_0", spk=0,
             corrected=False, bypassed=True),
        # 间隔 9000ms >= 8000ms -> 新 turn；组内全部 bypassed
        _utt("utt_0004", 13000, 14000, "四", spk_label="speaker_0", spk=0,
             corrected=False, bypassed=True),
        _utt("utt_0005", 14500, 15000, "五", spk_label="speaker_0", spk=0,
             corrected=False, bypassed=True),
    ]

    turns = build_turns(utts)

    assert len(turns) == 2
    # 任一句被修正 -> corrected=True；并非全部 bypass -> bypassed=False
    assert turns[0]["corrected"] is True
    assert turns[0]["bypassed"] is False
    # 全组都是短句 bypass -> bypassed=True；无修正 -> corrected=False
    assert turns[1]["corrected"] is False
    assert turns[1]["bypassed"] is True


# ---- StreamTurnTracker ----

def test_tracker_assigns_turn_and_cumulative_text():
    tracker = StreamTurnTracker()

    partial = tracker.assign({"t": "asr_partial", "utt_id": "utt_0001",
                              "start_ms": 0, "text": "你"})
    assert partial["turn_id"] == "turn_0001"
    assert partial["turn_text"] == "你"

    final = tracker.assign({"t": "asr_final", "utt_id": "utt_0001",
                            "start_ms": 0, "end_ms": 1000, "raw_text": "你好"})
    assert final["turn_id"] == "turn_0001"
    assert final["turn_text"] == "你好"

    confirmed = tracker.assign({"t": "llm_confirmed", "utt_id": "utt_0001",
                                "start_ms": 0, "end_ms": 1000,
                                "raw_text": "你好", "polished_text": "你好。"})
    assert confirmed["turn_id"] == "turn_0001"
    assert confirmed["turn_polished_text"] == "你好。"


def test_tracker_starts_new_turn_after_gap():
    tracker = StreamTurnTracker()
    tracker.assign({"t": "asr_final", "utt_id": "utt_0001",
                    "start_ms": 0, "end_ms": 1000, "raw_text": "一"})

    merged = tracker.assign({"t": "asr_final", "utt_id": "utt_0002",
                             "start_ms": 2999, "end_ms": 4000, "raw_text": "二"})
    assert merged["turn_id"] == "turn_0001"
    assert merged["turn_text"] == "一，二"

    split = tracker.assign({"t": "asr_final", "utt_id": "utt_0003",
                            "start_ms": 6000, "end_ms": 7000, "raw_text": "三"})
    assert split["turn_id"] == "turn_0002"
    assert split["turn_text"] == "三"


def test_tracker_unknown_llm_confirmed_falls_back_to_current_turn():
    tracker = StreamTurnTracker()
    tracker.assign({"t": "asr_final", "utt_id": "utt_0001",
                    "start_ms": 0, "end_ms": 1000, "raw_text": "一"})

    ev = tracker.assign({"t": "llm_confirmed", "utt_id": "utt_unknown",
                         "polished_text": "二"})

    assert ev["turn_id"] == "turn_0001"
    assert ev["turn_polished_text"] == "一，二"
    assert "utt_unknown" not in tracker.utt_turn, "llm_confirmed 不得回填归属"


def test_tracker_unknown_event_is_returned_unchanged():
    tracker = StreamTurnTracker()
    ev = {"t": "progress", "msg": "x"}

    assert tracker.assign(ev) is ev
    assert "turn_id" not in ev


def test_stream_turn_tracker_is_thread_safe():
    import threading

    tracker = StreamTurnTracker(gap_ms=2000)
    errors = []

    def feed(start):
        try:
            for i in range(200):
                tracker.assign({
                    "t": "asr_final", "utt_id": f"u_{start}_{i}",
                    "start_ms": start + i * 1000, "end_ms": start + i * 1000 + 500,
                    "raw_text": "你好",
                })
                tracker.assign({
                    "t": "llm_confirmed", "utt_id": f"u_{start}_{i}",
                    "start_ms": start + i * 1000, "end_ms": start + i * 1000 + 500,
                    "polished_text": "你好。",
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=feed, args=(n * 100000,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
