# -*- coding: utf-8 -*-
"""真实路径预测检验：句子级指标（离线粗窗）选出的 k，能否预测
真实路径（`v12.diarize()`）的选 k 与改善方向。

**这是 §4.1e 那个错的复刻检验**：`largest_eligible` 在离线候选表上排第一，
真实路径上却更差且半数回退。若新指标也在同一方向上骗人，它就不能当判据。

两级细化会改变高 k 的收益（§4.1f：高 k 上细化**有害**），所以这里必须区分：
  * 低/中 k：离线与真实路径一致（§4.1f 实测 k=4 两腿一致到 0.2 点）
  * 高 k：离线可信度下降
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` 与本地目录。**不要插仓库根**——那里有一份同名
# `claude_a2.py`（DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`。
# 回归测试见 tests/test_evaluator_path_hygiene.py
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 直接复用探针的 run_meeting，但换 k 网格
import numpy as np  # noqa: E402

import probe_sentence_attribution as P  # noqa: E402
from probe_gap_decomp import cached_windows, gate_windows  # noqa: E402
from sentence_k_rule import pick_best_k  # noqa: E402


def sweep(meeting, ks, seconds=600.0):
    """在给定 k 网格上算句子归属率（离线粗窗）。返回 {k: 归属率} 与 {k: sizes}。

    **无句子的场次**（TextGrid 无非空 mark，或 `seconds` 截断后为空）返回
    `{k: None}` / `{k: []}`，不抛异常——`max()` 空序列会 `ValueError`，
    而上游（`probe_sentence_proxy_gain`）没有兜底，会中断整批。

    返回 `None` 表示"该 k 无结论"（与"成绩为 0"不同）。**调用方必须显式处理**，
    不得把它当 0 或静默跳过整个场次（修订规格 §R7）。
    """
    emb_all, times_all = cached_windows(str(P.CACHE), meeting, P.WIN, P.HOP)
    vad = json.loads((P.VAD_CACHE / f"{meeting}.json").read_text(encoding="utf-8"))
    intervals = [(a, b) for a, b in vad["intervals"]]
    emb, times = gate_windows(emb_all, times_all, intervals, P.WIN, P.MIN_VAD)
    tiers = P.read_textgrid(P.DATA / "TextGrid" / f"{meeting}.TextGrid")
    sents = P.annotate_overlap(P.sentences_from_textgrid(tiers, seconds), tiers)
    if not sents:
        return {k: None for k in ks}, {k: [] for k in ks}
    end = max(s["end"] for s in sents)
    acc, sizes = {}, {}
    for k in ks:
        labels = P.clusters_for(emb, k)
        segs = P.segment_map(times, labels, np.full(len(times), P.WIN), end)
        pred = P.sentence_dict_predictions(sents, segs)
        r = P.attribution(sents, pred)
        acc[k] = r["overall"]["duration_weighted"]
        sizes[k] = [int((labels == l).sum()) for l in range(k)]
    return acc, sizes


def main():
    rows = json.loads((ROOT / "artifacts" /
                       "v13_e2e_unlocked2_dev_mw2_objective.json").read_text(encoding="utf-8"))["rows"]
    real = {r["meeting"]: r for r in rows}
    ks = list(range(2, 13))
    print(f"{'会议':<20}{'真实fix':>9}{'真实ΔDER':>10}│"
          f"{'argmax':>8}{'加惩罚':>8}{'@真实k':>8}{'一致?':>7}")
    agree = {"argmax": [], "penalised": []}
    for meeting in sorted(real):
        r = real[meeting]
        fk = r["fixed"]["k"]
        dder = r["base"]["der"] - r["fixed"]["der"]
        try:
            acc, sizes = sweep(meeting, ks)
        except Exception as exc:  # noqa: BLE001
            print(f"{meeting:<20} 失败: {exc}")
            continue
        argmax = max(acc, key=lambda k: acc[k] if acc[k] is not None else -1)
        pen = pick_best_k(acc, sizes)
        if pen is None:
            # 所有 k 的归属率都是 None（例如全场弃权）→ 这个场次无结论，
            # 不能让它中断整批（`abs(None - fk)` 会 TypeError）。
            print(f"{meeting:<20}{fk:>9}{dder*100:>9.1f}pts│"
                  f"{'—':>8}{'—':>8}{'—':>8}{'无结论':>7}")
            continue
        for name, pick in (("argmax", argmax), ("penalised", pen)):
            agree[name].append(pick == fk)
        mark = "✓" if pen == fk else ("≈" if abs(pen - fk) <= 1 else "✗")
        print(f"{meeting:<20}{fk:>9}{dder*100:>9.1f}pts│"
              f"{argmax:>8}{pen:>8}{100*acc.get(fk, float('nan')):>7.1f}%{mark:>7}")

    for name in ("argmax", "penalised"):
        n = sum(agree[name])
        print(f"  {name:<10} 精确一致 {n}/{len(agree[name])}")
    # 选 k 的分布对比
    from collections import Counter
    print(f"  真实 fixed.k 分布: {dict(sorted(Counter(v['fixed']['k'] for v in real.values()).items()))}")


if __name__ == "__main__":
    main()
