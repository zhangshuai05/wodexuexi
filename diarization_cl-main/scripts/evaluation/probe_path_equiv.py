# -*- coding: utf-8 -*-
"""路径等价检查：`v12.diarize()`（未修改）喂**探针缓存**，与 A/B 探针逐项比对。

## 两类差异，一次全暴露

`probe_ab_dev` 与 `v12.diarize()` 有三处已知差异（本次会话实测）：

  1. **`k<2` 时探针不救援**。`real_path_labels` 见 `k<2` 直接返回主窗标签；
     而 `diarize:203` 在 `not refined and selected["k"] == 1` 时会重提 0.6s 窗、
     重跑 `_candidate_clusters`，分出多人就**整体替换**。
  2. **铺窗 vs `_segment_windows`**。探针按 `t + 窗长` 铺段；`_segment_windows`
     的边界落在相邻窗重叠中点 + `hop/2`，**约半个窗的系统偏移**。
  3. **基线身份**。探针用 `v11._candidate_clusters` + `_select_candidate`，
     与 `diarize` 相同--这一条**不是**差异，列出来是为了不再被误读成差异。

## ⚠️ 判据的盲区（2026-09-28 实测，必读）

三场 × k=2..10 共 26 个对比，结果**一边倒**：段级几乎 100% 不同
（如 597/597），句级**全部 0 差异**。

```
L_R003S04C02  k=10:  diarize 段=597  探针段=3222   段级不同 597/597   句级不同 0/119
```

这意味着两件事，不可只看其中一件：

* **好消息**：差异 2（铺窗 vs `_segment_windows`）在句级被"取最长重叠"**完全吸收**。
  这些 k 上归属率是 27.5%--85.5% 的真实跨度，不是平凡解，所以"0 差异"有信息量。
* **警告**：句级判据对**段边界不敏感**。所以它**证明不了段表等价**，
  只能在"下游按最长重叠取预测说话人"这个用途下证明两条路径给同一答案。
  若将来下游改用段边界本身（如按段切音频、算段级 DER），这里的 0 差异**不适用**。

**不要**把 `segments_equal` 当作等价判据——两边段粒度本就不同（探针不做段合并）。
等价结论请只引用 `n_sentence_diff`。

## 做法：换缓存 + 注入首候选，不碰生产代码

`diarize` 内部有自己的 `_select_candidate`。要让它产出指定 K 的结果而不改生产签名，
用**既有**的 `forced_k` 参数（`v11._select_candidate(candidates, forced_k=...)`）。

`sentence` 侧只读缓存里的 `times`（生产窗起点），不按 `arange` 重建。

    python -u -B scripts/evaluation/probe_path_equiv.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from probe_ab_dev import SECONDS, load_sentences, score  # noqa: E402
from probe_sentence_attribution import segment_map, sentence_dict_predictions  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
PROBE_CACHE = CACHE / "probe"
DATA = Path(r"D:\yinp\AISHELL-4\data\test")

# 三场覆盖三种路径（见 manifest 的 counts）
TARGETS = {
    "L_R003S01C02": {"path": "rescue", "note": "base k=1，探针差异 1 直接命中"},
    "L_R003S03C02": {"path": "refine", "note": "base k=3，走细化主路径"},
    "L_R003S04C02": {"path": "refine", "note": "A@0.6 回归 -9.24 点，反例对照"},
}


def cached_stage(meeting, stage):
    data = np.load(PROBE_CACHE / meeting / f"{stage}.npz", allow_pickle=False)
    return data["emb"], np.asarray(data["times"], dtype=float)


def manifest_intervals(meeting):
    """建缓存时记下的 VAD 区间（4 位小数）。替换 `_vad_segs` 用，省掉模型。"""
    payload = json.loads((PROBE_CACHE / "manifest.json").read_text(encoding="utf-8"))
    return [(a, b) for a, b in payload["meetings"][meeting]["vad_intervals"]]


def install_cache(meeting, keep_model=False):
    """把三处窗提取换成读缓存。返回还原函数。

    **按调用时刻的 `WIN_S` 分派**：主窗与救援窗共用 `v11._extract_windows`，
    只有 `_v11_window_params` 换过的常量不同，所以必须看 `v11.WIN_S` 当前值。
    细化窗走 `v12._refine_windows`（它有自己的门限），单独替换。

    VAD 也替换：`diarize` 会自己调 `v11._vad_segs(models["vad"], wav)`，
    而缓存是按**建缓存时那份真 VAD** 生成的。读 manifest 里记的区间，
    保证窗集与缓存同源；VAD 本身不参与任何其它计算。
    """
    main = cached_stage(meeting, "main")
    refine = cached_stage(meeting, "refine")
    rescue = cached_stage(meeting, "rescue")
    intervals = manifest_intervals(meeting)
    orig = {"extract": v11._extract_windows, "refine": v12._refine_windows,
            "vad": v11._vad_segs}

    calls = []

    def fake_vad(vad, wav):
        return list(intervals)

    def fake_extract(audio, rate, vad_segments, spk_model):
        if abs(v11.WIN_S - v12.WIN_S) < 1e-9:
            stage = "main"
        elif abs(v11.WIN_S - v12.REFINE_WIN_S) < 1e-9:
            stage = "rescue"
        else:
            raise AssertionError(f"未预期的 WIN_S={v11.WIN_S}")
        calls.append(stage)
        emb, times = {"main": main, "rescue": rescue}[stage]
        return emb.copy(), times.copy(), 0

    def fake_refine(audio, rate, vad_segments, spk_model):
        calls.append("refine")
        return refine[0].copy(), refine[1].copy()

    v11._extract_windows = fake_extract
    v12._refine_windows = fake_refine
    v11._vad_segs = fake_vad
    return calls, orig


def restore(orig):
    v11._extract_windows = orig["extract"]
    v12._refine_windows = orig["refine"]
    v11._vad_segs = orig["vad"]


def run_diarize(meeting, forced_k=None):
    """跑**未修改的** `v12.diarize()`；`forced_k` 时只换选候选那一步。"""
    calls, orig = install_cache(meeting)
    injected = {}
    try:
        if forced_k is not None:
            real_select = v11._select_candidate

            def select_with_forced(candidates, forced_k=None, **kw):
                """注入的是 `_candidate_clusters` 的返回值，不改 `diarize` 的控制流。"""
                return real_select(candidates, forced_k=injected["k"])

            injected["k"] = forced_k
            v11._select_candidate = select_with_forced
        wav = str(DATA / "wav" / f"{meeting}.flac")
        result = v12.diarize(wav, {"vad": None, "spk": None},
                             progress=None, max_seconds=SECONDS)
        return result, calls
    finally:
        if forced_k is not None:
            v11._select_candidate = locals().get("real_select", v11._select_candidate)
        restore(orig)


def probe_path(meeting, k):
    """探针口径：主窗缓存 + 相同 AHC，`real_path_labels` 的细化策略。"""
    from probe_ab_dev import real_path_labels, _win
    from scipy.cluster.hierarchy import fcluster, linkage

    main_emb, main_times = cached_stage(meeting, "main")
    rE, rtimes = cached_stage(meeting, "refine")
    if k <= 1 or len(main_emb) < 2:
        labels = np.zeros(len(main_emb), dtype=int)
    else:
        tree = linkage(main_emb, method="average", metric="cosine")
        labels = v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))
    flabels, ftimes, level = real_path_labels(labels, main_emb, main_times, rE, rtimes)
    return flabels, ftimes, level


def segments_of(result):
    """`diarize` 的段表，标签**统一成字符串**。

    ⚠️ `diarize` 的 `spk` 是 `int(labels[...])`，而探针侧 `segment_map` 经 `_name()`
    转出**字符串**。`0 != "0"` 在 Python 里恒为 True，两边直接比对会让
    "逐段不同数"**恒等于段数**--看起来在报差异，其实是类型不匹配的假象。
    这里统一成 `str`，与 `_name()` 的口径一致（该函数对 None 返回 `""`）。
    """
    return [(round(s["start"], 3), round(s["end"], 3), str(s["spk"]))
            for s in result["segments"]]


def _sentence_preds(sentences, segs):
    """段表 → 逐句预测。段表的标签类型必须与 `sentence_dict_predictions` 期望一致。"""
    return np.asarray(sentence_dict_predictions(sentences, segs), dtype=object)


def _compare_sentences(sentences, a_pred, b_pred):
    """逐句对比两个预测序列。返回 (总句数, 不同句数, 已弃权句数)。

    弃权（`None`/`""`）**计入分母**（项目负责人既有约束）：全弃权的两边算"相同"，
    但弃权数必须单列，以区分"两边都对"与"两边都不知道"。
    """
    n = len(sentences)
    diff = sum(1 for x, y in zip(a_pred, b_pred) if str(x) != str(y))
    abstain = sum(1 for x in a_pred if x is None or x == "")
    return n, diff, abstain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=list(TARGETS))
    ap.add_argument("--ks", default="2,3,4")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "v13_path_equiv.json"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    report = {}

    for meeting in args.meetings:
        info = TARGETS.get(meeting, {})
        sentences = load_sentences(meeting)
        print(f"\n{'='*90}\n{meeting}  预期路径 {info.get('path','?')}"
              f"  （{info.get('note','')}）\n{'='*90}")

        # ── ① 参照：base（`diarize` 自己选 K）
        base, calls = run_diarize(meeting)
        bsegs = segments_of(base)
        bpred = _sentence_preds(
            sentences,
            segment_map(base["times"], base["labels"],
                        np.full(len(base["labels"]), base["window_seconds"]), 1e9))
        print(f"  ① base（原路径）: k={base['k']} refined={base['refined']} "
              f"rescued={base['rescued']} 段={len(bsegs)} 窗调用={calls}")

        # ── ② 注入同一个 K：应**逐位相同**（免费的 harness 自证）
        # 判据是段表**逐段**相同，不是"成绩接近"--成绩接近可以是两处错误互相抵消。
        same, same_calls = run_diarize(meeting, forced_k=base["k"])
        ssegs = segments_of(same)
        identical = ssegs == bsegs
        # 自证还要覆盖**逐句**：段表相同但句子归属不同，说明下游有额外分歧。
        s_pred = _sentence_preds(
            sentences,
            segment_map(same["times"], same["labels"],
                        np.full(len(same["labels"]), same["window_seconds"]), 1e9))
        _n, _d, _a, = _compare_sentences(sentences, bpred, s_pred)
        self_check_sent_identical = (_d == 0)
        print(f"  ② 注入 k={base['k']}: 段表相同 = {identical}  "
              f"逐句相同 = {self_check_sent_identical}（不同 {_d}/{_n}）  窗调用={same_calls}")

        # ── ③ 探针路径跑同一个 K
        entry = {"path": info.get("path"), "note": info.get("note"),
                 "base": {"k": base["k"], "refined": base["refined"],
                          "rescued": base["rescued"], "n_segments": len(bsegs)},
                 "self_check_identical": bool(identical),
                 "self_check_sent_identical": bool(self_check_sent_identical),
                 "n_sentences": len(sentences), "cases": {}}

        for k in ks:
            if k == base["k"]:
                continue
            diag, dcalls = run_diarize(meeting, forced_k=k)
            dsegs = segments_of(diag)
            d_pred = _sentence_preds(
                sentences,
                segment_map(diag["times"], diag["labels"],
                            np.full(len(diag["labels"]), diag["window_seconds"]), 1e9))
            plabels, ptimes, plevel = probe_path(meeting, k)
            win = v12.REFINE_WIN_S if plevel == "refined" else v12.WIN_S
            psegs = segment_map(ptimes, plabels, np.full(len(plabels), win), 1e9)
            p_pred = _sentence_preds(sentences, psegs)

            n_sent, n_sent_diff, n_abstain = _compare_sentences(sentences, d_pred, p_pred)
            n_diff = sum(1 for a, b in zip(dsegs, psegs) if a != b)
            n_common = min(len(dsegs), len(psegs))
            print(f"  k={k}: diarize 段={len(dsegs)} ({diag['window_seconds']}s, "
                  f"refined={diag['refined']} rescued={diag['rescued']} 窗调用={dcalls})"
                  f" | 探针 段={len(psegs)} ({plevel})"
                  f" | 前 {n_common} 段中不同 {n_diff}"
                  f" | 逐句不同 {n_sent_diff}/{n_sent}（弃权 {n_abstain}）")
            entry["cases"][str(k)] = {
                "diarize": {"n_segments": len(dsegs),
                            "window_seconds": diag["window_seconds"],
                            "refined": diag["refined"], "rescued": diag["rescued"],
                            "calls": dcalls},
                "probe": {"n_segments": len(psegs), "level": plevel},
                "n_segments_diff": abs(len(dsegs) - len(psegs)),
                # ⚠️ 两边的段粒度本就不同（探针按 `t+窗长` 铺段并**不做段合并**，
                # `diarize` 走 `_segment_windows` 合并相邻同标签窗），所以
                # "前 N 段不同"**天然会高**，`segments_equal` 基本恒 False。
                # 判等价要看**逐句**：`n_sentence_diff`，段级只作辅助。
                "granularity_mismatch": len(dsegs) != len(psegs),
                "n_prefix_mismatch": int(n_diff),
                "n_prefix_compared": int(n_common),
                "segments_equal": dsegs == psegs,
                "n_sentences": int(n_sent),
                "n_sentence_diff": int(n_sent_diff),
                "n_sentence_abstain_diarize": int(n_abstain),
                "sentence_preds_equal": d_pred.tolist() == p_pred.tolist(),
            }
        report[meeting] = entry

    Path(args.out).write_text(
        json.dumps({"schema": "v13-path-equiv-v2", "seconds": SECONDS,
                    "note": "v2：段标签统一为 str（v1 用 int 比对，n_prefix_mismatch "
                            "恒等于 n_prefix_compared，逐段比较失效）；新增逐句比较字段。"
                            "v1 的段级数字作废。",
                    "reports": report}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
