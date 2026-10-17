# -*- coding: utf-8 -*-
"""Meeting_main 与 diarization_cl-main 的唯一耦合点。

只往 sys.path 里加路径，不复制、不修改上游任何代码。
上游自身的 bootstrap 见 AliMeeting/run_3dspeaker_pipeline.py:34-37（同源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\Deepseek\diarization_cl-main")

# 本项目根目录：必须始终排在 sys.path 最前，避免被上游同名模块遮蔽
_LOCAL = Path(__file__).resolve().parent

_SUBPATHS: tuple[Path, ...] = (
    REPO / ".zhang",
    REPO / "AliMeeting",
    REPO / "streaming_asr",
    REPO / "scripts" / "evaluation",
)


def install() -> None:
    """幂等地把上游路径插到 sys.path 最前面。

    上游目录之后，本项目目录始终被强制放在 sys.path 首位，
    保证与上游同名的本地模块（如 run_offline、stream_llm_polish）不被遮蔽。
    """
    for path in _SUBPATHS:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    local = str(_LOCAL)
    if local in sys.path:
        sys.path.remove(local)
    sys.path.insert(0, local)


def check_paths() -> dict[str, bool]:
    """只检查目录存在，不触发任何 import。"""
    install()
    return {path.name: path.is_dir() for path in _SUBPATHS}


def check() -> dict[str, bool | str]:
    """真正尝试 import 上游入口。慢（会加载 funasr），仅 slow 测试用。"""
    install()
    result: dict[str, bool | str] = {}
    for name in ("offline_diarizer", "streaming_asr_engine", "web_server", "claude_a2"):
        try:
            __import__(name)
            result[name] = True
        except Exception as exc:  # noqa: BLE001 - 自检需要看到任何失败原因
            result[name] = f"{type(exc).__name__}: {exc}"
    return result
