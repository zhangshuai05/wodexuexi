# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

import bootstrap


def test_upstream_paths_exist():
    paths = bootstrap.check_paths()
    assert paths, "check_paths 不应返回空"
    missing = [name for name, ok in paths.items() if not ok]
    assert not missing, f"缺失的上游路径: {missing}"
    assert (Path(r"D:\Deepseek\diarization_cl-main") / ".zhang").is_dir()


def test_install_is_idempotent():
    import sys
    bootstrap.install()
    before = list(sys.path)
    bootstrap.install()
    assert list(sys.path) == before, "重复 install 不应改变 sys.path"


@pytest.mark.slow
def test_upstream_modules_importable():
    result = bootstrap.check()
    bad = {k: v for k, v in result.items() if v is not True}
    assert not bad, f"导入失败: {bad}"


def test_local_modules_win_over_upstream():
    import importlib.util
    import sys
    from pathlib import Path

    import bootstrap

    bootstrap.install()
    for name in ("stream_llm_polish", "run_offline"):
        sys.modules.pop(name, None)
        spec = importlib.util.find_spec(name)
        assert spec is not None, f"{name} 不可导入"
        assert Path(spec.origin).resolve().parent == Path(bootstrap._LOCAL), \
            f"{name} 被上游同名模块遮蔽: {spec.origin}"


def test_claude_a2_resolves_to_repo_root():
    import importlib.util
    import sys

    bootstrap.install()
    sys.modules.pop("claude_a2", None)
    spec = importlib.util.find_spec("claude_a2")
    assert spec is not None, "claude_a2 不可导入"
    assert Path(spec.origin).resolve().parent == Path(bootstrap.REPO).resolve(), \
        f"claude_a2 被 streaming_asr 下的分发器遮蔽: {spec.origin}"


def test_upstream_fingerprint_survives_unreadable_file(monkeypatch):
    target_name = next(iter(bootstrap.BASELINE))
    real_read_bytes = Path.read_bytes

    def fake_read_bytes(self):
        if self.name == target_name:
            raise OSError("locked")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    drift = bootstrap.upstream_fingerprint()

    assert target_name in drift
    assert drift[target_name].startswith("unreadable:"), drift[target_name]
