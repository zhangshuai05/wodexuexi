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
