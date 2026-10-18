# -*- coding: utf-8 -*-
"""回归：评价器模块**不得**把仓库根插进 `sys.path`。

## 为什么

`scripts/evaluation/` 下的模块若把仓库根（`parents[2]`）插进 `sys.path`，
就会**遮住 `streaming_asr/claude_a2.py`**——因为仓库根有一份同名文件
（DSH 测试台用的 468 行旧版，没有 `visible_versions`）。

后果不是一处报错，而是：任何一个先导入了这些评价器的测试文件，
都会让**后续** `test_service.py` 的 `import claude_a2` 命中错的那份，
报 `ImportError: cannot import name 'visible_versions' from <repo-root>\\claude_a2.py`。

这曾真实发生：本次会话新增的三个测试使 pytest 的导入顺序改变，
暴露出这个隐患（全量 3 failed；排除新增测试则 77 passed）。
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

STREAMING = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAMING.parent
EVAL_DIR = REPO_ROOT / "scripts" / "evaluation"

# 本次会话新增/依赖的评价器模块。任何一个把仓库根插进 sys.path，
# 都会让后续 `import claude_a2` 命中仓库根那份（DSH 测试台用）旧模块。
EVALUATOR_MODULES = [
    "sentence_k_rule",
    "probe_sentence_attribution",
    "probe_sentence_vs_realpath",
    "probe_gap_decomp",
    "unlocked_selector",
    "probe_selector_locks",
]

# 源码扫描的范围：句子级评价器这一组（本次会话维护的）。
# `scripts/evaluation/` 下另有约 20 个更早的脚本也插仓库根，
# 它们是独立执行的、不在本组导入链上，统一修属于另一次改动。
SCANNED_SCRIPTS = [
    "sentence_k_rule.py",
    "probe_sentence_attribution.py",
    "probe_sentence_vs_realpath.py",
    "probe_gap_decomp.py",
    "unlocked_selector.py",
    "probe_selector_locks.py",
]


def _probe_sys_path(module_name):
    """在干净子进程里导入给定模块，回报 (sys.path, claude_a2 实际来源)。

    子进程是必须的：这些模块顶层会 import scipy/numpy，
    且本测试要观察的是**导入副作用**，在 pytest 进程内做会互相污染。
    """
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{STREAMING}')\n"
        f"sys.path.insert(0, r'{EVAL_DIR}')\n"
        f"import {module_name}\n"
        "import json\n"
        "sys.modules.pop('claude_a2', None)\n"
        "import claude_a2\n"
        "print(json.dumps({'path': sys.path, 'a2': claude_a2.__file__}))\n"
    )
    out = subprocess.run([sys.executable, "-B", "-c", code],
                         capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", EVALUATOR_MODULES)
def test_evaluator_does_not_shadow_streaming_asr_claude_a2(module):
    """导入评价器后，`claude_a2` 必须仍解析到 `streaming_asr/` 那一份。"""
    got = _probe_sys_path(module)
    a2 = Path(got["a2"]).resolve()
    assert a2.parent == STREAMING, (
        f"导入 {module} 后 claude_a2 解析到 {a2}，应为 {STREAMING}/claude_a2.py —— "
        "该模块把仓库根插进了 sys.path，遮住了 streaming_asr 版本"
    )


@pytest.mark.parametrize("module", EVALUATOR_MODULES)
def test_evaluator_does_not_put_repo_root_on_sys_path(module):
    """仓库根不得出现在评价器导入后的 `sys.path` 里。"""
    got = _probe_sys_path(module)
    offenders = [p for p in got["path"]
                 if p and os.path.abspath(p) == str(REPO_ROOT.resolve())]
    assert not offenders, (
        f"仓库根 {REPO_ROOT} 被 {module} 插进了 sys.path：{offenders}。"
        "那里有同名 claude_a2.py，会遮住 streaming_asr 版本。"
    )


def test_no_eval_script_inserts_repo_root_by_source_scan():
    """**源码扫描**：句子级评价器相关的脚本不得再有人把仓库根插进 sys.path。

    这是对上面 per-module 导入测试的补充。导入测试只能覆盖已知模块；
    源码扫描能拦住**将来新增**的句子级评价器脚本，不用为每个文件起子进程。

    合规：插 `streaming_asr/` 或本地目录——
        sys.path.insert(0, str(ROOT / "streaming_asr"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_asr"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    违规：把**仓库根本身**插进去——
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    ⚠️ 范围有意限制在句子级评价器这一组。`scripts/evaluation/` 下另有约 20 个
    更早的脚本（`probe_truek*`、`probe_selector_locks_*`、`check_train_split` 等）
    也插仓库根；它们是独立执行的脚本、不在本组导入链上，**统一修属于另一次改动**。
    """
    offenders = []
    for name in SCANNED_SCRIPTS:
        path = EVAL_DIR / name
        assert path.exists(), f"清单里的脚本不存在：{path}"
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or "sys.path.insert" not in stripped:
                continue
            # 合规：路径里带 streaming_asr 或指本地目录
            if "streaming_asr" in stripped or "resolve().parent)" in stripped \
                    or "__file__).resolve().parent)" in stripped:
                continue
            # 其余 insert 若提到 ROOT / parents[2]，即为插仓库根
            if "ROOT" in stripped or "parents[2]" in stripped:
                offenders.append(f"{name}:{lineno}: {stripped}")
    assert not offenders, (
        "以下句子级评价器脚本把仓库根插进了 sys.path，"
        "会遮住 streaming_asr/claude_a2.py：\n  " + "\n  ".join(offenders)
    )
