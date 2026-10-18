# -*- coding: utf-8 -*-
"""本地版 compare_seaco 打分: 对我 diar_result 的 c1 结果 vs 本地 ground_truth (文本对齐).
用法: python local_eval.py <结果目录>  (目录内含 *_diar.json)
输出逐文件 matched/gt、按距离汇总。"""
import json, os, re, sys, glob
from difflib import SequenceMatcher

BASE = r"c:\Users\EDY\Desktop\new"
HERE = BASE
GT = os.path.join(BASE, "eval_dsh", "ground_truth.json")


def norm(s):
    return re.sub(r"[\s,，。.！!？?、：:；;\"'“”‘’]", "", s or "")


def best_match(text, cands):
    t = norm(text)
    if not t:
        return None
    best, bi = 0.0, None
    for i, c in cands:
        r = SequenceMatcher(None, t, c).ratio()
        if r > best:
            best, bi = r, i
    return bi, best


def main(resdir):
    gt = json.load(open(GT, encoding="utf-8"))
    report = []
    grand = 0
    gtot = 0
    bydist = {}
    for g in gt:
        base = g["file"].rsplit(".", 1)[0]
        jp = os.path.join(resdir, base + "_diar.json")
        if not os.path.exists(jp):
            print("no result for", g["file"]); continue
        out = json.load(open(jp, encoding="utf-8"))
        si = out.get("sentence_info") or []
        if not si:
            print("empty si", g["file"]); continue
        cands = [(i, norm(s.get("text", ""))) for i, s in enumerate(si)]
        person_spks = {p: {} for p in g["speakers"]}
        matched = 0
        for t in g["turns"]:
            bi, r = best_match(t["text"], cands)
            if bi is None or r < 0.35:
                continue
            spk = str(si[bi].get("spk", "?"))
            person_spks[t["spk"]][spk] = person_spks[t["spk"]].get(spk, 0) + 1
            matched += 1
        n_spks = sorted(set(str(s.get("spk", "?")) for s in si))
        grand += matched; gtot += g["n_turns"]
        d = g["distance"]
        bydist.setdefault(d, [0, 0]); bydist[d][0] += matched; bydist[d][1] += g["n_turns"]
        # 每说话人的主标签纯度
        purity_txt = []
        for p in g["speakers"]:
            dd = person_spks[p]
            if dd:
                main = max(dd.items(), key=lambda kv: kv[1])[0]
                pur = max(dd.values()) / sum(dd.values())
                purity_txt.append(f"{p}->{main}:{pur:.2f}")
            else:
                purity_txt.append(f"{p}->-")
        print(f"{g['file']:<30} {d:<5} nspks={n_spks} matched={matched}/{g['n_turns']}  "
              f"{' | '.join(purity_txt)}")
        report.append({"file": g["file"], "dist": d, "matched": matched,
                       "gt_turns": g["n_turns"], "spks": n_spks})
    print(f"\n=== 合计 {grand}/{gtot} = {grand/gtot:.3f} ===")
    for d, (m, t) in sorted(bydist.items()):
        print(f"  {d}: {m}/{t} = {m/t:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "diar_result", "a2_v3_improved"))
