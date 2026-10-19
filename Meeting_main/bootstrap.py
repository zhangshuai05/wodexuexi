# -*- coding: utf-8 -*-
"""Meeting_main 与 diarization_cl-main 的唯一耦合点。

只往 sys.path 里加路径，不复制、不修改上游任何代码。
上游自身的 bootstrap 见 AliMeeting/run_3dspeaker_pipeline.py:34-37（同源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

def _resolve_repo(here: Path | None = None) -> Path:
    """定位 diarization_cl-main：monorepo 同级 → 项目内 vendored → 旧绝对路径。"""
    here = Path(here or __file__).resolve()
    candidates = (
        here.parent.parent / "diarization_cl-main",  # monorepo: <root>/Meeting_main/..
        here.parent / "diarization_cl-main",         # vendored inside Meeting_main
        Path(r"D:\Deepseek\diarization_cl-main"),    # legacy local checkout
    )
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[-1]


REPO = _resolve_repo()

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
    # 2026-10-19 上游身份合并与句级迟滞更新：
    # streaming_asr_engine: VAD 同块结束+开始跨句吞字修复；1000ms 显式生效
    # offline_diarizer: 长窗主投票、迟滞平滑、超短句邻域继承与句内门控的 LF 规范化指纹。
    "streaming_asr_engine.py": ("5152424c04f4e572", 15831),
    "offline_diarizer.py": ("fc06cdda7d27cebb", 26247),
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
        normalized = data.replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()[:16]
        if digest != expected_hash or len(normalized) != expected_size:
            drift[name] = (
                f"changed: {digest} {len(normalized)}B "
                f"(baseline {expected_hash} {expected_size}B)"
            )
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
