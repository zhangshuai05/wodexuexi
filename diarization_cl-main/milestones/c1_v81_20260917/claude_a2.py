# -*- coding: utf-8 -*-
"""Claude v7/v8/v9/v10 的独立泳道分发器。"""

VERSIONS = {
    "v7": {
        "label": "Claude v7 · 救援门+碎段吸收",
        "method": "claude-a2-v7-c1",
    },
    "v8": {
        "label": "Claude v8 · 外生证据+人数三态",
        "method": "claude-a2-v8-c3",
    },
    "v9": {
        "label": "Claude v9 · 契约与救援门修正",
        "method": "claude-a2-v9-c3",
    },
    "v10": {
        "label": "Claude v10 · 两遍对照实验",
        "method": "claude-a2-v10-e1-two-pass-reference",
    },
}


def visible_versions():
    return VERSIONS.copy()


def run_claude_lane(wav, models, lane_key, progress=None):
    if lane_key in ("v7", "v8"):
        from claude_a2_v7_v8 import run_claude_lane as run_version

        frozen_key = "c1" if lane_key == "v7" else "v8"
    elif lane_key == "v9":
        from claude_a2_v9 import run_claude_lane as run_version

        frozen_key = "v9"
    elif lane_key == "v10":
        from claude_a2_v10 import run_claude_lane as run_version

        frozen_key = "v10"
    else:
        raise ValueError(f"unknown claude lane {lane_key}")
    return run_version(wav, models, frozen_key, progress=progress)
