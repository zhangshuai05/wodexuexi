# -*- coding: utf-8 -*-
"""离线代理的增益能否预测真实路径的增益（**相关性入口**）。

## 这是干什么的

`probe_sentence_vs_realpath.py` 只比"选出的 k 是否一致"，本脚本比**增益**：
拿真实路径已经从 `base.k` 换到 `fixed.k` 的场次，问
「代理在这两个 k 上的归属率差」与「真实路径的 DER 差」是否相关。

若相关 → 代理保留了排序信息，可以进一步谈阈值。
若无关/反向 → 代理与真实路径量的不是同一件事（§4.1e 的重演），作为 K 判据不可救。

## 血泪教训（务必先读）

本入口的前身 `_diag_proxy_vs_real_delta.py` 报出 **r=+0.893、放大 1.5 倍**，
但那是**污染尺子**上的数：当时 `segment_map` 用 `if nm` 判空，
把整数 `0` 簇整个丢掉。**该结论已撤回。** 下面的数字只有在本脚本
（以及 `probe_sentence_attribution` 的当前版本）下才成立。

        python -u -B scripts/evaluation/probe_sentence_proxy_gain.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` 与本地目录。**不要插仓库根**——那里有一份同名
# `claude_a2.py`（DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`。
# 回归测试见 tests/test_evaluator_path_hygiene.py
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_sentence_vs_realpath import sweep  # noqa: E402

E2E_DEFAULT = "v13_e2e_unlocked2_dev_mw2_objective.json"


def collect(e2e_path, cache):
    """逐场对比。返回 (可用项, 被排除项)。

    `sweep()` 要读嵌入并做层次聚类，开销不小，所以每场**只算一次**、
    把 base/fixed 两个 k 一起算出来复用（前身在同一场算了两次）。

    **两个边界必须显式处理（修订规格 §R7）**：
      * `sweep` 对无句子的场次返回 `{k: None}` → 直接相减会 `TypeError`
        并中断整批（前身 `_diag_proxy_vs_real_delta.py` 用 `np.nan` 降级，
        但 `nan - nan` 是**静默**传播，会污染相关系数）。
      * `sweep` 自身可能抛异常。
    两种情况都**记录下来**、返回给调用方报出，**不静默跳过**。
    """
    rows = json.loads(e2e_path.read_text(encoding="utf-8"))["rows"]
    usable, excluded = [], []
    for r in sorted(rows, key=lambda x: x["meeting"]):
        meeting = r["meeting"]
        bk, fk = r["base"]["k"], r["fixed"]["k"]
        if bk == fk:
            continue                      # 真实路径没换 k → 对照无信息量
        key = (meeting, (bk, fk))
        try:
            if key in cache:
                acc = cache[key]
            else:
                acc, _ = sweep(meeting, sorted({bk, fk}))
                cache[key] = acc
        except Exception as exc:          # noqa: BLE001
            excluded.append({"meeting": meeting, "reason": "sweep 抛异常",
                             "detail": repr(exc)})
            continue
        pb, pf = acc.get(bk), acc.get(fk)
        if pb is None or pf is None:
            excluded.append({"meeting": meeting, "reason": "该 k 无结论（值=None）",
                             "detail": f"k={bk} -> {pb}, k={fk} -> {pf}"})
            continue
        usable.append({"meeting": meeting, "bk": bk, "fk": fk,
                       "real_delta": r["base"]["der"] - r["fixed"]["der"],
                       "proxy_delta": pf - pb,
                       "proxy_at_base": pb, "proxy_at_fixed": pf})
    return usable, excluded


def report(items, min_n=3):
    print(f"{'会议':<22}{'真k变化':>10}{'真实ΔDER':>11}{'代理Δ归属':>11}")
    for it in items:
        move = str(it["bk"]) + "->" + str(it["fk"])
        print(f"{it['meeting']:<22}{move:>10}"
              f"{100*it['real_delta']:>10.1f}pts{100*it['proxy_delta']:>10.1f}pts")
    n = len(items)
    print(f"\n  n={n}（只含真实路径 k 发生变化的场次）")
    if n < min_n:
        print(f"  ⚠️ n<{min_n}：**不足以**支持任何相关系数或换算系数，"
              f"只能说方向是否一致。")
        agree = sum(1 for it in items
                    if np.sign(it["proxy_delta"]) == np.sign(it["real_delta"]))
        print(f"  方向一致: {agree}/{n}")
        return
    proxy = np.array([it["proxy_delta"] for it in items], dtype=float)
    real = np.array([it["real_delta"] for it in items], dtype=float)
    # Spearman 用 scipy 的实现：前身用 `argsort(argsort())` 近似，
    # 有并列值时不准，且拿不到 p 值——n 这么小，p 值比 r 更重要。
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(proxy, real)
        print(f"  Spearman rho = {rho:+.3f}  (p={pval:.3f})")
    except Exception as exc:  # noqa: BLE001
        print(f"  Spearman 计算失败：{exc}")
    if np.std(proxy) > 0 and np.std(real) > 0:
        print(f"  Pearson  r = {np.corrcoef(proxy, real)[0, 1]:+.3f}")
    else:
        print("  Pearson  r = n/a（某一列无变化）")
    if real.mean() != 0:
        print(f"  代理Δ 均 {100*proxy.mean():.1f}pts, 真实Δ 均 {100*real.mean():.1f}pts"
              f"  (放大 {proxy.mean()/real.mean():.1f} 倍)")
    else:
        print("  放大倍数 n/a（真实Δ 均值为 0，不能相除）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", default=E2E_DEFAULT)
    ap.add_argument("--min-n", type=int, default=3,
                    help="低于此样本量只报方向，不报相关系数")
    args = ap.parse_args()
    items, excluded = collect(ROOT / "artifacts" / args.e2e, {})
    if excluded:
        # §R7：被排除的场次**显式报出**，不静默跳过
        print("**被排除的场次（显式记录，未静默跳过）**：")
        for e in excluded:
            print(f"  - {e['meeting']}: {e['reason']}  [{e['detail']}]")
        print()
    if not items:
        print("没有可用的 k 变化场次，无从对比。")
        return
    report(items, args.min_n)

    print(f"\n  逐场：代理认为 base.k 有多好？（若 base 也高，代理就没有可用排序）")
    for it in items:
        flag = "  ← 代理同意 base" if it["proxy_at_base"] >= it["proxy_at_fixed"] - 0.02 else ""
        print(f"    {it['meeting']:<22} base k={it['bk']}: {100*it['proxy_at_base']:>5.1f}%  "
              f"fixed k={it['fk']}: {100*it['proxy_at_fixed']:>5.1f}%{flag}")


if __name__ == "__main__":
    main()
