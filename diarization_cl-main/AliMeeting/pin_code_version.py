# -*- coding: utf-8 -*-
"""钉住"这批数字是哪一版代码跑出来的"。

主仓用 `SHA256.json` 给里程碑归档做完整性清单，这里是同一个思路：
主仓在被并行改动（本会话运行期间 `scripts/evaluation/build_probe_cache.py`、
`docs/HANDOFF.md` 都动过），所以本子项目的数字必须能归因到确切的代码版本。

跑完任何一批数字后都可以重跑本脚本，比对哈希是否仍然一致。

    python pin_code_version.py            # 写出 results/code_version.json
    python pin_code_version.py --check    # 与已写出的比对，不一致就退出 1
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from common import RESULTS, REPO

# 本子项目的结果直接依赖这些文件；任何一处变了，数字就必须重跑。
DEPENDENCIES = [
    "streaming_asr/der.py",                  # DER 评分器（口径的定义处）
    "streaming_asr/claude_a2_v11.py",        # VAD 调用、窗提取、候选簇、选 K
    "streaming_asr/claude_a2_v12.py",        # 生产入口 diarize()、分段、细化
    "streaming_asr/v12_aishell4.py",         # dev/holdout 分组（重叠率对照用）
    "streaming_asr/web_server.py",           # get_models()
    "scripts/evaluation/probe_sentence_attribution.py",  # TextGrid 读取器
]

# 算法行为的常量也钉住：哈希一致但常量被改，同样是换了一版算法。
CONSTANTS = {
    "streaming_asr/claude_a2_v11.py": [
        "WIN_S", "HOP_S", "MIN_VAD_FRACTION", "MAX_SPEAKERS",
        "MIN_CLUSTER_WINDOWS", "MIN_CLUSTER_SHARE", "MIN_CLUSTER_SIZE",
        "MULTI_K_PENALTY"],
    "streaming_asr/claude_a2_v12.py": [
        "WIN_S", "HOP_S", "MAX_SPEAKERS", "SLIVER_WINDOWS",
        "REFINE_WIN_S", "REFINE_HOP_S"],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constants_of(relative, names):
    """从模块源码里抠出 `NAME = value` 的字面量，不 import（import 会真的加载模型）。"""
    import re

    text = (REPO / relative).read_text(encoding="utf-8", errors="replace")
    out = {}
    for name in names:
        match = re.search(rf"^{re.escape(name)}\s*=\s*([^\n#]+)", text, re.M)
        if match:
            out[name] = match.group(1).strip()
    return out


def snapshot():
    return {
        "files": {rel: sha256(REPO / rel) for rel in DEPENDENCIES},
        "constants": {rel: constants_of(rel, names)
                      for rel, names in CONSTANTS.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "code_version.json"
    current = snapshot()

    for relative, digest in current["files"].items():
        print(f"  {digest[:16]}  {relative}")
    for relative, values in current["constants"].items():
        print(f"  常量 {Path(relative).name}: {values}")

    if args.check:
        if not target.exists():
            print(f"\n没有 {target}，先跑一次不带 --check。")
            return 1
        saved = json.loads(target.read_text(encoding="utf-8"))
        diffs = [rel for rel in current["files"]
                 if saved["files"].get(rel) != current["files"][rel]]
        diffs += [rel for rel in current["constants"]
                  if saved["constants"].get(rel) != current["constants"][rel]]
        if diffs:
            print(f"\n不一致：{diffs}\n依赖代码已变，results/ 下的数字必须重跑。")
            return 1
        print("\n一致：results/ 下的数字仍对应当前代码。")
        return 0

    target.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n已写出 {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
