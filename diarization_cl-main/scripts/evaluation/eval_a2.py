# -*- coding: utf-8 -*-
"""
A2 细粒度分离 评估（借用 dsh 线 purity/good 口径）
A2 输出: 段 [{start,end,spk}], 无文本 -> 用"时间范围 + 文本"双对齐到真值。
真值: ground_truth.json (轮级 {spk,text}, 无时间)
用法: python eval_a2.py [安静|嘈杂|all]   (默认 all)
输出: 每文件 主标签/纯度/good, 聚合表
"""
import json, os, re, sys
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT = os.path.join(HERE, "eval_dsh", "ground_truth.json")
A2_SUMMARY = os.path.join(HERE, "diar_result", "a2_all", "summary.json")

def norm(s):
    return re.sub(r"[\s,，。.！!？?、：:；;\"'“”‘’]", "", s)

def main(env_filter="all"):
    gt = json.load(open(GT, encoding="utf-8"))
    summary = json.load(open(A2_SUMMARY, encoding="utf-8"))
    # 建立 文件 -> 距离/环境/顾客 映射
    a2_by_file = {}
    for row in summary:
        a2_by_file[row["call"]] = (row["dist"], row["env"], row["cust"], "customer")
        a2_by_file[row["note"]] = (row["dist"], row["env"], row["cust"], "staff")

    lines = []
    agg = {}   # (env) -> {n_file, good, purity_sum}
    for g in gt:
        base = g["file"]
        if base not in a2_by_file:
            continue
        dist, env, cust, chan = a2_by_file[base]
        if env_filter != "all" and env != env_filter:
            continue
        # 取该文件的 A2 段
        row = None
        for s in summary:
            if s["call"] == base or s["note"] == base:
                row = s; break
        info = row["customer"] if chan == "customer" else row["staff"]
        segs = info["segs"]   # [{start,end,spk}]

        # 用文本相似度把真值轮次对齐到 A2 段（按时间最近 + 文本相似）
        # 简化: 真值轮次按顺序, A2 段也按 start 排序, 用"轮次文本"和"段(无文本)"无法直接匹配
        # => 借用他口径: 需要文本. A2 没文本, 所以降级为:
        #   "每轮文本 -> 该文件 ASR 分句(若有)匹配", 但我们没有分句文本.
        # 替代: 用 A2 段的"时间覆盖率"作为近似
        # 更实际: 我们保留 dsh 口径的"每真人 main_spk/purity"关键需求 =>
        #   用时间重叠把真值轮次(假设均匀分布时长)映射到段.
        # 真值無时间戳, 无法严格做; 采用近似:
        #   1) 取真值轮数 N, 假设各轮等时长
        #   2) 每个真值轮 -> 对应一个时间范围(按顺序)
        #   3) 落在哪个 A2 段 => 该真值轮 spk = 段 spk
        n_turns = g["n_turns"]
        segs_sorted = sorted(segs, key=lambda s: s["start"])
        if not segs_sorted:
            continue
        total_dur = max(s["end"] for s in segs_sorted)
        per_turn = total_dur / n_turns if n_turns else 0
        # 每个真值轮映射
        person_spks = {p: {} for p in g["speakers"]}
        matched = 0
        for i, t in enumerate(g["turns"]):
            t_mid = (i + 0.5) * per_turn
            # 找包含该时刻的 A2 段
            spk_here = None
            for s in segs_sorted:
                if s["start"] <= t_mid <= s["end"]:
                    spk_here = s["spk"]; break
            if spk_here is None:
                continue
            person_spks[t["spk"]][spk_here] = person_spks[t["spk"]].get(spk_here, 0) + 1
            matched += 1

        n_spks = len(set(s["spk"] for s in segs_sorted))
        line = f"{base:<26} [{dist}|{env}|{chan}] matched={matched}/{n_turns} n_spks={n_spks}"
        lines.append(line)
        pure = []
        for p in g["speakers"]:
            d = person_spks[p]
            main = max(d.items(), key=lambda kv: kv[1])[0] if d else "-"
            purity = (max(d.values()) / sum(d.values())) if sum(d.values()) else 0.0
            pure.append(purity)
            lines.append(f"    {p:<5} spk分布={d} 主标签={main} 纯度={purity:.2f}")
        # good: distinct + n_spks<=2 + 双方>=0.8
        mains = [person_spks[p].get(max(person_spks[p], key=person_spks[p].get), 0) for p in g["speakers"] if person_spks[p]]
        distinct = len(set(person_spks[p].get(max(person_spks[p], key=person_spks[p].get), None) for p in g["speakers"] if person_spks[p])) == len([p for p in g["speakers"] if person_spks[p]])
        good = distinct and n_spks <= 2 and all(p >= 0.8 for p in pure) and len(pure) == 2
        lines.append(f"    distinct={distinct} good={good}")
        key = env
        agg.setdefault(key, {"n":0,"good":0,"purity_sum":0,"files":[]})
        agg[key]["n"] += 1
        if good: agg[key]["good"] += 1
        agg[key]["purity_sum"] += (min(pure) if len(pure)==2 else 0)
        agg[key]["files"].append(base)

    # 聚合
    lines.append("\n===== 聚合(按环境) =====")
    for env, d in agg.items():
        lines.append(f"{env}: n={d['n']} good={d['good']} 平均min-purity={d['purity_sum']/max(d['n'],1):.3f}")
    report = "\n".join(lines)
    print(report, flush=True)

    out = os.path.join(HERE, "diar_result", "a2_all", f"eval_a2_{env_filter}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print("saved:", out)

if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(f)
