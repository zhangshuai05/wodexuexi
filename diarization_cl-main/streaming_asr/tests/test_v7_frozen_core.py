"""Guard the deployed v7 core against accidental edits by future lanes."""

import ast
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
MILESTONE = HERE.parent / "milestones" / "c1_v7_20260910" / "claude_a2.py"
CURRENT = HERE / "claude_a2_v7_v8.py"
MILESTONE_SHA256 = "c8c37e5d8cca62868f6eca21a6456c10d7a0e040b77cf0164e4e57693ea05aa5"
FROZEN_HELPERS = (
    "cs",
    "_stabilize_labels",
    "_vad_segs",
    "_segment_bounds",
    "_dense_rescue",
    "_absorb_slivers",
    "_build_seg",
    "_gap2_of",
    "_rescue_gate",
)


def _functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_v7_milestone_hash_is_the_archived_release():
    assert hashlib.sha256(MILESTONE.read_bytes()).hexdigest() == MILESTONE_SHA256


def test_current_v7_helpers_match_the_archived_release():
    archived, current = _functions(MILESTONE), _functions(CURRENT)
    for name in FROZEN_HELPERS:
        assert current[name] == archived[name], f"v7 frozen helper changed: {name}"
