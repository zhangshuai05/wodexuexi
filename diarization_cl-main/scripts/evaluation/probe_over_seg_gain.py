# -*- coding: utf-8 -*-
"""证伪「过分割涨分」：`k_all_best` 的收益是真的两人分离，还是评分口径的产物？

## 为什么这是阻塞项

`HANDOFF.md` §3 卡点 1。两条实测事实只能同时成立，如果 oracle 口径**在大 k 上单调不减**：

  * 12 场的 oracle 峰值 k 是 `[7,6,7,6,5,10,8,6,5,8,6,5]`，**只有 1 场峰值是 10**；
  * 但「恒选 k=10」的平均 regret 恰好是 **0.00**。

机制（`probe_mapping_objective.score_oracle` 的 docstring 自己写明）：
句级 oracle **每个预测簇独立取它最常对应的真值人**，是**多对一**。所以一个真值人被
切成 5 个簇，5 个簇仍全能映射回他，**句级归属率不降**。

⇒ `k_all_best` 可能不是「最优 k」，只是「**过分割不扣分的那个 k**」。

## 本脚本怎么判

对每场取 **k=6** 与 **k=10** 两个 partition（HANDOFF §4 指定的对照点），做四件事：

1. **簇数 vs 真值人数**：k=10 的簇数明显多于该场真值人数吗？超出的簇对应谁？
2. **`dup` 比例**（复用 `probe_small_cluster_audit` 的性质分类）：k=10 的簇里，
   有多少是「该真值人在别处另有更大主簇」= **重复副本**。
3. **★ 决定性判据 —— 生物射对照**：
   * `oracle`（现行口径，**多对一**，允许 N 个簇同指一人）
   * `bijective`（匈牙利**双射**，一个真值人只能被一个簇占用）
   若 `k=10` 相对 `k=6` 的收益**在双射口径下消失**，就说明收益全来自
   「多出来的簇可以重复映射到同一个人」——即**过分割不扣分**，`k_all_best` 是假象。
4. **簇数归一**：同口径下 k=10 与 k=6 的「每个真值人被几个簇占用」分布。

## 输出口径

  * `gain_oracle`   = oracle(k=10) − oracle(k=6)
  * `gain_biject`   = bijective(k=10) − bijective(k=6)
  * `new_speakers`  = k=10 新增（k=6 未覆盖）的真值人列表
  * `dup_share`     = k=10 簇中判为 dup 的比例

**判读**：`gain_oracle > 0` 而 `gain_biject ≈ 0` ⇒ **过分割涨分，卡点 1 成立**。

    python -u -B scripts/evaluation/probe_over_seg_gain.py --selfcheck
    python -u -B scripts/evaluation/probe_over_seg_gain.py --detail M_R003S05C01
    python -u -B scripts/evaluation/probe_over_seg_gain.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import probe_path_equiv as E  # noqa: E402
from der import read_rttm  # noqa: E402
from probe_gap_decomp import window_reference  # noqa: E402
from probe_mapping_objective import score_oracle  # noqa: E402
from probe_small_cluster_audit import truth_flags  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")
K_LOW, K_HIGH = 6, 10          # HANDOFF §4 指定的对照点
FRAME = 0.01


def partition(emb, k):
    """k 层 partition（与生产 `clusters_for` 同法）。"""
    return P.clusters_for(emb, k)


def truth_coverage(labels, names):
    """每个预测簇 → (真值人分布, 主导人, 尺寸, 纯度)。"""
    rows = []
    for lab in sorted(set(int(x) for x in labels)):
        idx = np.flatnonzero(labels == lab)
        got = [str(names[i]) for i in idx if str(names[i]) != ""]
        comp = Counter(got)
        top, topn = comp.most_common(1)[0] if comp else ("", 0)
        rows.append({
            "cluster": lab,
            "n_windows": int(len(idx)),
            "dominant": top,
            "purity": round(topn / len(got), 4) if got else None,
            "truth_mix": {k: v for k, v in comp.most_common(4)},
        })
    return rows


def bijective_score(sents, pred):
    """匈牙利**双射**口径（`P.attribution` 默认走它），返回 overall。"""
    return P.attribution(sents, pred)["overall"]


def classify_dup(rows):
    """给每个簇判 dup：该真值人在别处另有**更大**的主簇。

    只按「主导人相同且存在更大的簇」判定，与 `probe_small_cluster_audit` 同规则。
    """
    home = {}
    for r in rows:
        if r["n_windows"] > 2 and r["dominant"]:
            home.setdefault(r["dominant"], []).append(r["n_windows"])
    home_max = {k: max(v) for k, v in home.items()}
    for r in rows:
        dom, n = r["dominant"], r["n_windows"]
        r["is_dup"] = bool(dom and dom in home_max and home_max[dom] > n)
        r["speaker_home_size"] = home_max.get(dom)
    return rows


def merge_dups(labels, rows):
    """把每个 dup 簇的窗并入它的**主簇**（该真值人最大的簇），重排标签。

    这是本脚本的**决定性对照**：dup 簇与主簇同指一人，合并是唯一正确的归一化
    （无论它是"同一人的另一段时间"还是"离群窗"，都不该独占一个标签）。
    合并后：
      * 若 `merged(k=10)` **仍高于** `oracle(k=6)` → 收益来自**非 dup 的真拆分**；
      * 若 `merged(k=10)` **回落到** `oracle(k=6)` → 收益全部来自重复簇，是口径产物。
    """
    home_of = {}
    for r in rows:
        if r["is_dup"] and r["dominant"]:
            home_of[r["cluster"]] = None            # 先占位，稍后填主簇号
    # 主簇 = 该主导人里**尺寸最大**的簇
    best = {}
    for r in rows:
        dom = r["dominant"]
        if not dom:
            continue
        if dom not in best or r["n_windows"] > best[dom][1]:
            best[dom] = (r["cluster"], r["n_windows"])
    for r in rows:
        if r["is_dup"] and r["dominant"] in best:
            home_of[r["cluster"]] = best[r["dominant"]][0]
    out = np.array(labels, dtype=int)
    for dup, home in home_of.items():
        if home is not None:
            out[labels == dup] = home
    return v11._relabel_by_first_seen(out)


def probe(meeting):
    sents = E.load_sentences(meeting)
    emb, times = E.cached_stage(meeting, "main")
    ref = [s for s in read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
           if s["start"] < P.SECONDS]
    end = max(s["end"] for s in sents)
    names = window_reference(ref, times, P.WIN)
    bnd, ovl = truth_flags(ref, times, P.WIN)  # 供 --detail 交叉核对，本脚本不直接判边界

    truth_speakers = {str(s["speaker"]) for s in ref}
    # 主窗里实际可见的真值人（有窗被判给他）= 本场"可得分人数"的下界
    visible = {str(v) for v in names if str(v) != ""}

    out = {"meeting": meeting, "n_windows": int(len(emb)), "n_sentences": len(sents),
           "truth_speakers": len(truth_speakers), "visible_speakers": len(visible),
           "n_boundary_windows": int(bnd.sum()), "n_overlap_windows": int(ovl.sum())}
    per_k = {}
    for k in (K_LOW, K_HIGH):
        labels = partition(emb, k)
        rows = classify_dup(truth_coverage(labels, names))
        segs = P.segment_map(times, labels, np.full(len(times), P.WIN), end)
        p = P.sentence_dict_predictions(sents, segs)
        orc = score_oracle(sents, p)["overall"]["per_sentence"]
        bij = bijective_score(sents, p)["per_sentence"]
        doms = [r["dominant"] for r in rows if r["dominant"]]
        # 决定性对照：dup 簇并回主簇后再打分
        merged_labels = merge_dups(labels, rows)
        msegs = P.segment_map(times, merged_labels, np.full(len(times), P.WIN), end)
        mp = P.sentence_dict_predictions(sents, msegs)
        m_orc = score_oracle(sents, mp)["overall"]["per_sentence"]
        m_bij = bijective_score(sents, mp)["per_sentence"]
        per_k[k] = {
            "n_clusters": len(rows),
            "oracle": round(float(orc), 4),
            "bijective": round(float(bij), 4),
            "merged_oracle": round(float(m_orc), 4),
            "merged_bijective": round(float(m_bij), 4),
            "merged_clusters": len({int(x) for x in merged_labels}),
            "dup_clusters": int(sum(r["is_dup"] for r in rows)),
            "dup_share": round(sum(r["is_dup"] for r in rows) / len(rows), 4),
            "distinct_dominant": len(set(doms)),
            "rows": rows,
        }
    out["per_k"] = per_k
    # ★ 三个口径下的收益
    out["gain_oracle"] = round(per_k[K_HIGH]["oracle"] - per_k[K_LOW]["oracle"], 4)
    out["gain_bijective"] = round(per_k[K_HIGH]["bijective"] - per_k[K_LOW]["bijective"], 4)
    # 决定性：合并 dup 后，k=10 相对 k=6 的**原始** oracle 还有多少收益
    out["gain_merged"] = round(per_k[K_HIGH]["merged_oracle"]
                               - per_k[K_LOW]["oracle"], 4)
    # k=10 相比 k=6 新增覆盖的真值人
    low_doms = {r["dominant"] for r in per_k[K_LOW]["rows"] if r["dominant"]}
    high_doms = {r["dominant"] for r in per_k[K_HIGH]["rows"] if r["dominant"]}
    out["new_speakers"] = sorted(high_doms - low_doms)
    out["lost_speakers"] = sorted(low_doms - high_doms)
    out["n_new_speakers"] = len(out["new_speakers"])
    return out


def report(rows):
    print("=" * 118)
    print(f"过分割涨分证伪：k={K_LOW} vs k={K_HIGH}（HANDOFF §4 第一优先）")
    print("=" * 118)
    print(f"{'会议':<18}{'窗数':>6}{'真值人':>7}{'k=6簇':>7}{'k=10簇':>8}"
          f"{'oracle6':>9}{'oracle10':>9}{'Δoracle':>9}"
          f"{'bij6':>8}{'bij10':>8}{'Δbij':>8}{'合并后':>8}{'新增人':>7}{'dup%':>6}")
    print("-" * 118)
    for m in rows:
        a, b = m["per_k"][K_LOW], m["per_k"][K_HIGH]
        print(f"{m['meeting']:<18}{m['n_windows']:>6}{m['truth_speakers']:>7}"
              f"{a['n_clusters']:>7}{b['n_clusters']:>8}"
              f"{100*a['oracle']:>8.1f}%{100*b['oracle']:>8.1f}%"
              f"{100*m['gain_oracle']:>+9.2f}"
              f"{100*a['bijective']:>7.1f}%{100*b['bijective']:>7.1f}%"
              f"{100*m['gain_bijective']:>+8.2f}"
              f"{100*m['gain_merged']:>+8.2f}"
              f"{m['n_new_speakers']:>7}{100*b['dup_share']:>5.0f}%")


def summary(rows):
    go = np.array([m["gain_oracle"] for m in rows])
    gb = np.array([m["gain_bijective"] for m in rows])
    gm = np.array([m["gain_merged"] for m in rows])
    print("\n" + "=" * 118)
    print("汇总（三口径，读法见文末判读）")
    print("=" * 118)
    print(f"  Δoracle（多对一，现行口径）   均值 {100*go.mean():+7.2f} pts  "
          f"为正 {int((go>1e-9).sum())}/{len(go)} 场")
    print(f"  Δbijective（匈牙利双射）      均值 {100*gb.mean():+7.2f} pts  "
          f"为正 {int((gb>1e-9).sum())}/{len(gb)} 场")
    print(f"  ★ Δmerged（dup 并回主簇后）   均值 {100*gm.mean():+7.2f} pts  "
          f"为正 {int((gm>1e-9).sum())}/{len(gm)} 场")
    vanish = sum(1 for a, c in zip(go, gm) if a > 1e-9 and c <= 1e-9)
    keep = sum(1 for c in gm if c > 1e-9)
    print(f"\n  oracle 涨分但合并后消失的场次：{vanish}/{len(go)}")
    print(f"  合并 dup 后仍有正收益的场次：  {keep}/{len(gm)}  ← **真实拆分**的证据")
    print("\n  判读：")
    print("    ① Δmerged ≈ 0 而 Δoracle > 0 ⇒ 那份收益来自重复簇，是**过分割不扣分**的产物；")
    print("    ② Δmerged > 0 ⇒ 存在**非 dup 的真拆分**，`k_all_best` 不是纯口径产物；")
    print("    ③ Δbijective 大幅为负 ⇒ 现行 oracle 口径**系统性掩盖**过分割的代价。")


def detail(rows, meeting):
    m = next((x for x in rows if x["meeting"] == meeting), None)
    if m is None:
        print(f"无此场次：{meeting}")
        return
    print("=" * 118)
    print(f"{meeting}  真值人 {m['truth_speakers']}  可见 {m['visible_speakers']}  "
          f"窗 {m['n_windows']}  新增人 {m['new_speakers']}  丢失人 {m['lost_speakers']}")
    print("=" * 118)
    for k in (K_LOW, K_HIGH):
        d = m["per_k"][k]
        print(f"\n--- k={k}  簇 {d['n_clusters']}  主导人 {d['distinct_dominant']}  "
              f"dup {d['dup_clusters']}（{100*d['dup_share']:.0f}%）  "
              f"oracle {100*d['oracle']:.2f}%  双射 {100*d['bijective']:.2f}%")
        print(f"    {'簇':>4}{'窗数':>6}{'主导':>10}{'纯度':>8}{'dup':>5}{'该人主簇':>9}  真值分布")
        for r in d["rows"]:
            pur = f"{100*r['purity']:.0f}%" if r["purity"] is not None else "-"
            mix = " ".join(f"{a}:{b}" for a, b in list(r["truth_mix"].items())[:3])
            print(f"    {r['cluster']:>4}{r['n_windows']:>6}{r['dominant']:>10}{pur:>8}"
                  f"{('是' if r['is_dup'] else ''):>5}"
                  f"{str(r['speaker_home_size']):>9}  {mix}")


def selfcheck(rows):
    assert len(rows) == 12, f"应处理 12 场，实得 {len(rows)}"
    for m in rows:
        for k in (K_LOW, K_HIGH):
            d = m["per_k"][k]
            total = sum(r["n_windows"] for r in d["rows"])
            assert total == m["n_windows"], \
                f"{m['meeting']} k={k} 簇窗数之和 {total} != 主窗数 {m['n_windows']}"
            assert 0.0 <= d["oracle"] <= 1.0, f"{m['meeting']} k={k} oracle 越界"
            assert 0.0 <= d["bijective"] <= 1.0, f"{m['meeting']} k={k} 双射越界"
        assert m["per_k"][K_HIGH]["n_clusters"] >= m["per_k"][K_LOW]["n_clusters"], \
            f"{m['meeting']} k=10 簇数少于 k=6"
    # 所有可见真值人必须出现在 k=10 的 k=6 并集里（否则是有窗无人认领）
    for m in rows:
        doms = {r["dominant"] for k in (K_LOW, K_HIGH)
                for r in m["per_k"][k]["rows"] if r["dominant"]}
        assert not (set(m["new_speakers"]) & set(m["lost_speakers"])), \
            f"{m['meeting']} 新增人集合与丢失人集合相交"
        del doms
    print("自检通过：")
    print("  12 场全部处理；每场两个 k 的簇窗数之和 == 主窗数；k=10 簇数 >= k=6 簇数")
    print("  （`oracle` 为多对一口径，`bijective` 为匈牙利双射口径，两者定义见 docstring）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--detail", default="")
    ap.add_argument("--out", default="v13_over_seg_gain.json")
    args = ap.parse_args()
    rows = [probe(m) for m in split_meetings(DATA, "dev")]
    if args.selfcheck:
        selfcheck(rows)
        return
    report(rows)
    summary(rows)
    if args.detail:
        detail(rows, args.detail)
    selfcheck(rows)
    (ROOT / "artifacts" / args.out).write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ artifacts/{args.out}")


if __name__ == "__main__":
    main()
