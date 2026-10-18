# -*- coding: utf-8 -*-
"""入口文档自洽体检。

四份入口文档（README / 发现记录 / 交付合并表 / 交接文档）被反复追加更正，
而**更正天然要引用旧说法**。天真的字符串搜索会把引文当残留报假阳性
（本会话就报过一次：三处"残留"全是正当引文），所以这里的判据是
**命中点是否处在更正语境里**，而不是命中与否。

另加两道：
  2. 同一主张的数值跨文档一致性
  3. 文档里反引号引用的**仓库内**路径是否真实存在
     （D:\\ 下的数据/模型目录不入库，一律放行）

    python -u -B scripts/evaluation/check_entry_docs.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = [("README.md", ROOT / "README.md"),
        ("发现记录", ROOT / "artifacts/v13_会话发现_20260923.md"),
        ("交付合并表", ROOT / "artifacts/v13_交付合并表.md"),
        ("交接文档0924", ROOT / "docs/交接文档_0924.md"),
        ("交接文档0923", ROOT / "docs/交接文档_0923.md")]

# 被推翻的旧说法 -> 允许它们出现的语境标记
STALE = {
    "并人藏在 FA": ("更正", "错的", "旧"),
    "必须同时看 FA": ("更正", "错的", "旧"),
    "逐位相同、因此无效": ("更正", "错的", "推翻"),
    "尚未解决的自相矛盾": ("已解", "更正", "作废"),
    "无法从仓库侧追查": ("已解", "已复现", "更正"),
    "61 passed": ("过期", "已更正"),
}
CONTEXT = re.compile(r"已解|更正|推翻|错的|作废|过期|已复现|口径差")

# 同一主张必须一致的数值
CLAIMS = [("dev 基线", "24.4%"), ("dev 解锁", "14.8%"),
          ("holdout 基线", "19.8%"), ("holdout 解锁", "10.2%"),
          ("train_L 基线", "56.4%"), ("train_L 解锁", "48.7%"),
          ("train_L 改善计数", "11 / 0 / 19")]

# 仓库内的路径前缀。D:\ 下的数据/模型目录不入库，一律放行。
IN_REPO = ("artifacts/", "scripts/", "docs/", "streaming_asr/", "milestones/")


def main():
    text = {name: p.read_text(encoding="utf-8", errors="replace")
            for name, p in DOCS if p.exists()}
    bad = 0

    print("=== 1. 旧说法是否出现在没有更正标记的段落 ===")
    for phrase, _markers in STALE.items():
        for name, body in text.items():
            for m in re.finditer(re.escape(phrase), body):
                # 命中点前后 3 行内是否有更正标记
                start = body.rfind("\n", 0, max(0, m.start() - 300))
                window = body[max(0, start):m.end() + 300]
                if CONTEXT.search(window):
                    continue                     # 正当引文
                bad += 1
                line = body[:m.start()].count("\n") + 1
                print(f"  X {name}:{line}  「{phrase}」无更正语境")
    if bad == 0:
        print("  全部处在更正语境里（引文），无裸残留")

    print("\n=== 2. 同一主张的数值一致性 ===")
    for label, value in CLAIMS:
        have = [name for name, body in text.items() if value in body]
        ok = len(have) >= 2
        if not ok:
            bad += 1
        print(f"  {'OK' if ok else 'X '} {label:<12}{value:<8} 出现在 {', '.join(have) or '（无）'}")

    print("\n=== 3. 仓库内引用是否真实存在 ===")
    checked = skipped = 0
    for name, body in text.items():
        for m in re.finditer(r"`([^`]+)`", body):
            ref = m.group(1).strip()
            if not ref.startswith(IN_REPO):
                continue
            # 带 <占位> / 空格 / * 的不是可解析路径字面量（占位或带参数的命令）
            if re.search(r"[<>* ]", ref):
                skipped += 1
                continue
            checked += 1
            if not (ROOT / ref).exists():
                bad += 1
                line = body[:m.start()].count("\n") + 1
                print(f"  X {name}:{line}  引用不存在: {ref}")
    print(f"  检查 {checked} 条，跳过非字面量 {skipped} 条"
          + ("，全部存在" if bad == 0 else ""))

    print(f"\n  -> 问题 {bad} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
