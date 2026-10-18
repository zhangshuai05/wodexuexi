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
    REPO,
    REPO / ".zhang",
    REPO / "AliMeeting",
    REPO / "streaming_asr",
    REPO / "scripts" / "evaluation",
)

BASELINE = {
    # 2026-10-18 上游封板指纹（Agent A 最终版）：
    # streaming_asr_engine: VAD 同块结束+开始跨句吞字修复；1000ms 显式生效
    # offline_diarizer: 默认 normalize_peak=0.85 + with_segments=True 句内切分
    "streaming_asr_engine.py": ("5152424c04f4e572", 15831),
    "offline_diarizer.py": ("eb454a8bf70d2081", 15061),
}


def upstream_fingerprint() -> dict[str, str]:
    """返回与基线不一致的上游模块（空 dict = 一致）。缺失也算不一致。"""
    import hashlib

    drift: dict[str, str] = {}
    for name, (expected_hash, expected_size) in BASELINE.items():
        path = REPO / ".zhang" / name
        if not path.exists():
            drift[name] = "missing"
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            drift[name] = f"unreadable: {type(exc).__name__}: {exc}"
            continue
        digest = hashlib.sha256(data).hexdigest()[:16]
        if digest != expected_hash or len(data) != expected_size:
            drift[name] = f"changed: {digest} {len(data)}B (baseline {expected_hash} {expected_size}B)"
    return drift


def install() -> None:
    """幂等地把上游路径插到 sys.path 最前面。

    上游目录之后，本项目目录始终被强制放在 sys.path 首位，
    保证与上游同名的本地模块（如 run_offline、stream_llm_polish）不被遮蔽。
    """
    for path in reversed(_SUBPATHS):
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
