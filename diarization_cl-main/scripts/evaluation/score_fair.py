# -*- coding: utf-8 -*-
"""公平对比评分：c1_fair vs dsh_fair，两边都是 FRCRN 输入。
GT = gt_ts_diag(11文件)；另加 3 文件官方 gt_ts 口径。
"""
import json, os, sys
sys.path.insert(0, r"D:\Deepseek_herness\2026-08\diarization")
from eval_der import eval_one  # noqa: E402
from eval_gt import ENV  # noqa: E402

DSH = r"D:\Deepseek_herness\2026-08\diarization"
FAIR = r"C:\Users\EDY\Desktop\new\diar_result\fair"
COLLAR, TRIM = 0.5, True

DIAG = ["Call20260718210657", "Call20260718211031", "Call20260718211506",
        "Call20260718211745", "Call20260718211938", "Note20260803111605",
        "Note20260803111943", "Note20260803112220", "Note20260803112415",
        "Note20260803112655", "Note20260803112841"]


def load(b, src):
    name = b + "_frcrn_dsh" if src == "dsh" else b + "_c1"
    return json.load(open(os.path.join(FAIR, src, name + ".json"),
                          encoding="utf-8"))["sentence_info"]


def score_both(turns, b, cache):
    out = {}
    for src in ("c1", "dsh"):
        ns = load(b, src)
        n, m, f, e, mp = eval_one(turns, ns, COLLAR, TRIM)
        out[src] = (n, m, f, e, (m + f + e) / max(1, n))
    return out


def main():
    gts = json.load(open(os.path.join(DSH, "output", "gt_ts_diag.json"),
                         encoding="utf-8"))["files"]
    print(f"===== 公平对比 (两边FRCRN) collar={COLLAR}s+trim =====")
    print(f"{'file':<22} {'env':<4} {'dist':<5} | {'c1':>6} {'dsh':>6} | "
          f"c1(m/f/e)        dsh(m/f/e)")
    print("-" * 82)
    rows = {}
    for b in sorted(gts):
        turns = gts[b]["turns"]
        env = ENV.get(b, {}).get("env", "?"); dist = ENV.get(b, {}).get("distance", "?")
        r = score_both(turns, b, {})
        rows[b] = (env, dist, r)
        c = r["c1"]; d = r["dsh"]
        print(f"{b:<22} {env:<4} {dist:<5} | {c[4]:>5.1%} {d[4]:>5.1%} | "
              f"{c[1]}/{c[2]}/{c[3]:>3}        {d[1]}/{d[2]}/{d[3]:>3}")

    def agg(which):
        tot = {"c1": 0, "dsh": 0}; acc = {"c1": 0, "dsh": 0}
        for b, (env, dist, r) in rows.items():
            if which == "all" or which == "reliable" and env == "安静" \
               or which == "noisy" and env == "嘈杂":
                for s in ("c1", "dsh"):
                    tot[s] += r[s][0]; acc[s] += r[s][4] * r[s][0]
        return {s: acc[s] / max(1, tot[s]) for s in ("c1", "dsh")}, tot

    for label, which in (("全部11", "all"), ("安静5", "reliable"), ("嘈杂6", "noisy")):
        a, t = agg(which)
        print(f"\n[{label}]  c1 DER={a['c1']:.1%}   dsh DER={a['dsh']:.1%}  "
              f"(GT帧 c1={t['c1']})")


if __name__ == "__main__":
    main()
