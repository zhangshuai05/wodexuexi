# -*- coding: utf-8 -*-
"""测试引导：让本地 run_offline 优先于上游同名模块。

bootstrap.install() 会把 D:\\Deepseek\\diarization_cl-main\\.zhang 插到
sys.path 最前面，而那里有一个同名 run_offline.py。若测试模块在
bootstrap.install() 之后才首次 import run_offline，拿到的会是上游模块。
这里在收集测试前先导入本地模块并缓存到 sys.modules，保证被测的是本项目实现。
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_offline  # noqa: E402,F401 - 预导入并缓存本地模块
