from llm_correct import build_prompt, extract_json, normalize


def test_llm_prompt_and_json_normalization():
    source = [
        {"start": 0.0, "end": 1.5, "spk": 0, "text": "你好"},
        {"start": 1.5, "end": 2.0, "spk": 1, "text": "谢谢"},
    ]
    prompt = build_prompt(source)
    assert "speaker_0: 你好" in prompt
    assert "speaker_1: 谢谢" in prompt

    parsed = extract_json(
        '```json\n{"sentence_info":[{"start":0,"end":2,"spk":1,"text":"你好"}]}\n```'
    )
    assert normalize(parsed) == [
        {"start": 0.0, "end": 2.0, "spk": 1, "text": "你好", "overlap": False}
    ]


def test_llm_normalize_rejects_invalid_entries():
    assert normalize({"sentence_info": [
        {"start": 2, "end": 1, "spk": 0, "text": "bad time"},
        {"start": 0, "end": 1, "spk": 4, "text": "bad speaker"},
        {"start": 0, "end": 1, "spk": 0, "text": ""},
    ]}) == []


def test_llm_accepts_existing_third_speaker_but_not_an_invented_one():
    parsed = {"sentence_info": [
        {"start": 0, "end": 1, "spk": 2, "text": "第三位说话人"},
        {"start": 1, "end": 2, "spk": 3, "text": "不在输入里的标签"},
    ]}
    assert normalize(parsed, allowed_speakers=(0, 1, 2)) == [
        {"start": 0.0, "end": 1.0, "spk": 2, "text": "第三位说话人", "overlap": False}
    ]
