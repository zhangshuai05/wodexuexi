# -*- coding: utf-8 -*-
"""把 `run_tiny_fix.py` 脱离宿主跑。

宿主后台任务会被连带杀掉：主仓 ledger 记过 Task 3 第一次全量只留下 1/30
（`[status: killed, killed before exit]`）。本启动器用
`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` 让 python 彻底脱离宿主，
日志写**文本文件**（不是管道），宿主/执行器何时退出都不影响它。

    python _launch_tiny_fix.py
    # 之后：Get-Content ..\artifacts\_tiny_fix_run.log -Tail 20
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.abspath(os.path.join(HERE, "..", "artifacts", "_tiny_fix_run.log"))


def main():
    cmd = [sys.executable, "-u", os.path.join(HERE, "run_tiny_fix.py"),
           "--out", "tiny_fix_alimeeting_full.json"]
    flags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        flags = (subprocess.DETACHED_PROCESS
                 | subprocess.CREATE_NEW_PROCESS_GROUP)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with open(LOG, "w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=handle,
                                stderr=subprocess.STDOUT, env=env,
                                creationflags=flags)
    print("detached pid=%d" % proc.pid)
    print("python=%s" % sys.executable)
    print("log=%s" % LOG)


if __name__ == "__main__":
    main()
