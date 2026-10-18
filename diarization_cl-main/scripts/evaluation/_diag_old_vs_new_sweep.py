# -*- coding: utf-8 -*-
"""逐场逐 k 对比：污染版 sweep（标签 0 被过滤 + 短句被剔除 + 弃权移出分母）
vs 修正版（全 k 网格）。

要回答三件事：
1. 标签 0 过滤**实际影响了哪些场次/哪些 k**？（k=1 必受影响；k≥2 时带 0 号簇的也受影响）
2. 修复是否改变了 argmax 与排名？
3. 弃权 / 未匹配两个量在修正版里各是多少？
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts" / "v13_sentence_attrib_dev_sweep_BUGGY_label0.json"
NEW = ROOT / "artifacts" / "v13_sentence_attrib_dev_fullgrid_FIXED.json"

def main():
    old = json.loads(OLD.read_text(encoding="utf-8"))["reports"]
    new = json.loads(NEW.read_text(encoding="utf-8"))["reports"]

    # 真实路径选 k（dev 12 场）
    e2e = json.loads((ROOT / "artifacts" /
                      "v13_e2e_unlocked2_dev_mw2_objective.json").read_text(encoding="utf-8"))["rows"]
    real = {r["meeting"]: r["fixed"]["k"] for r in e2e}

    print("=" * 96)
    print("逐场逐 k：旧(污染) → 新(修正)。':=' = 逐位相同")
    print("=" * 96)
    tot_cells = tot_same = 0
    for m in sorted(set(old) & set(new)):
        # 旧的候选网格只跑了 1,3,5,7；新的跑了 1..12。逐格对比只能在交集上做。
        shared = [k for k in old[m] if k in new[m]]
        cells, diffs = [], []
        for k in sorted(shared, key=int):
            o = old[m][k]["overall"]["duration_weighted"]
            n = new[m][k]["overall"]["duration_weighted"]
            cells.append(f"{k}:{100*o:.0f}→{100*n:.0f}" if o != n else f"{k}:=")
            tot_cells += 1
            if o == n:
                tot_same += 1
            else:
                diffs.append((k, o, n))
        # **双报 argmax**，两种口径不能混称：
        #   common_argmax —— 只在**两者都测过的 k** 上取，是"同一候选集下的公平对比"
        #   full_argmax   —— 在**各自全集**上取，是"补全网格后最优 k 落在哪"
        # 旧版只跑了 1,3,5,7；新版跑了 1..12。只看 full 会把"旧版没测"误读成"算法变了"，
        # 只看 common 又会漏掉补全网格带来的变化。两个都报。
        shared = sorted(shared, key=int)
        o_common = max(shared, key=lambda k: old[m][k]["overall"]["duration_weighted"] or 0)
        n_common = max(shared, key=lambda k: new[m][k]["overall"]["duration_weighted"] or 0)
        o_full = max(old[m], key=lambda k: old[m][k]["overall"]["duration_weighted"] or 0)
        n_full = max(new[m], key=lambda k: new[m][k]["overall"]["duration_weighted"] or 0)
        extra = sorted(set(new[m]) - set(old[m]), key=int)
        note = f"  [旧版未测 {','.join(extra)}]" if extra else ""
        cflag = "" if o_common == n_common else f"  ★共同网格 {o_common}→{n_common}"
        fflag = "" if o_full == n_full else f"  全集 {o_full}→{n_full}"
        print(f"\n{m}  (真实 fix k={real.get(m,'?')}){cflag}{fflag}{note}")
        print("   " + "  ".join(cells))
        if diffs:
            print("   变化明细: " + ", ".join(f"k={k}: {100*o:.1f}%→{100*n:.1f}%"
                                              for k, o, n in diffs))

    print("\n" + "=" * 96)
    print(f"逐位相同的格子: {tot_same}/{tot_cells}（只比两者都测过的 k）")
    common = sorted(set(old) & set(new))


    def _argmax(report, m, keys=None):
        """keys=None → 该报告的全集；否则只在给定 k 集合上取。"""
        ks = list(report[m]) if keys is None else keys
        return max(ks, key=lambda k: report[m][k]["overall"]["duration_weighted"] or 0)


    if not common:
        # 空交集：没有可比的场次。`common[0]` 会 IndexError，必须显式退出。
        print("\n两个产物的场次集合无交集，无法对比。")
        print(f"  仅旧 {sorted(set(old))}")
        print(f"  仅新 {sorted(set(new))}")
        raise SystemExit(0)

    # 两个口径分开报，且**不要混称**
    shared_keys = sorted(set(old[common[0]]) & set(new[common[0]]), key=int)
    same_common = sum(1 for m in common
                      if _argmax(old, m, shared_keys) == _argmax(new, m, shared_keys))
    same_full = sum(1 for m in common if _argmax(old, m) == _argmax(new, m))
    print(f"argmax 一致(共同网格 {','.join(shared_keys)}): {same_common}/{len(common)}")
    print(f"argmax 一致(各自全集): {same_full}/{len(common)}")
    if set(old) - set(new) or set(new) - set(old):
        print(f"  场次集合不一致：仅旧 {sorted(set(old)-set(new))}，仅新 {sorted(set(new)-set(old))}")

    # 修正版：弃权 vs 未匹配（分开报）
    print("\n" + "=" * 96)
    print("修正版：弃权率 vs 未匹配率（项目负责人要求分开报）")
    print(f"{'会议':<20}{'k':>4}{'归属率':>9}{'弃权':>8}{'未匹配':>8}{'句数':>7}")
    for m in sorted(new):
        r = new[m]
        arg = max(r, key=lambda k: r[k]["overall"]["duration_weighted"] or 0)
        o = r[arg]["overall"]
        print(f"{m:<20}{arg:>4}{100*o['duration_weighted']:>8.1f}%"
              f"{100*o['abstain']:>7.1f}%{100*o['unmapped']:>7.1f}%{o['n']:>7}")


if __name__ == "__main__":
    main()
