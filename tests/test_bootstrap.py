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
