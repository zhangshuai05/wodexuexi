# -*- coding: utf-8 -*-
"""候选簇准入审计：复核「小簇一票否决整个候选」是不是 K 塌陷的机制。

对治的代码是 `claude_a2_v11._candidate_clusters`：

    eligible = (min(sizes) >= MIN_CLUSTER_WINDOWS          # 4
                and min(sizes) / n >= MIN_CLUSTER_SHARE)   # 0.015
    score = _silhouette(E, labels) if eligible else -1.0

`min(sizes)` 把账算到**整个候选**头上；而 `fcluster(criterion="maxclust", t=k)`
**强制切出 k 个簇**，k 一超真值就必然产生小簇 ⇒ 好候选被整候选否决 ⇒ 退回 k=1。

审计口径：
- 「小簇」= 窗数 < max(MIN_CLUSTER_WINDOWS, ceil(SHARE*n))。注意 n 大时**占比门更严**
  （n=1415 ⇒ 阈值 22 窗），所以「1-5 窗碎簇」只是其中一类。
- 「可救的候选」= 被否、但去掉小簇后**仍有 >=2 个大簇**的候选。它给出修法的上界：
  这类候选在「先剔除/合并小簇再判合格」之后至少能进入竞争。

⚠️ 本脚本只做**结构性**统计，**不预测 DER**：产物里被否候选的 `score` 已写成 -1.0，
真实轮廓系数不在盘上。要拿效果必须真跑（见 run_oracle_k.py / run_der.py）。

用法:
    $env:PYTHONIOENCODING="utf-8"
    python scripts/diagnostics/audit_candidate_eligibility.py [oracle_k产物.json]
"""
import json
import math
import sys

MIN_W = 4
SHARE = 0.015
DEFAULT = "AliMeeting/results/oracle_k_alimeeting_full.json"


def audit(path):
    data = json.load(open(path, encoding="utf-8"))
    rows = data["rows"]
    tot = denied = only_share = only_win = both = 0
    rescue = []
    hit = 0
    print("%-14s %6s %6s %7s %10s" % ("meeting", "true_k", "auto_k", "n_win", "临界窗数"))
    for r in rows:
        n = r["n_windows"]
        thr = max(MIN_W, int(math.ceil(SHARE * n)))
        hit += int(r["true_k"] == r["auto_k"])
        print("%-14s %6d %6d %7d %10d"
              % (r["meeting"], r["true_k"], r["auto_k"], n, thr))
        for c in r["candidates"]:
            if c["k"] == 1:
                continue
            tot += 1
            if c["eligible"]:
                continue
            denied += 1
            sizes = c["sizes"]
            fw = [x for x in sizes if x < MIN_W]
            fs = [x for x in sizes if x < thr]
            if fs and not fw:
                only_share += 1
            elif fw and not fs:
                only_win += 1
            else:
                both += 1
            big = sorted((x for x in sizes if x >= thr), reverse=True)
            if len(big) >= 2:
                rescue.append((r["meeting"], c["k"], len(big), big))

    print()
    print("K 命中真值: %d/%d 场" % (hit, len(rows)))
    print("候选总数(k>=2) = %d ; 被否 = %d (%.0f%%)"
          % (tot, denied, 100.0 * denied / max(tot, 1)))
    print("    仅占比门(<%.3f*n) = %d" % (SHARE, only_share))
    print("    仅 4 窗门         = %d" % only_win)
    print("    两者都犯          = %d" % both)
    print()
    print("可救候选（被否但去掉小簇后仍有 >=2 个大簇）= %d 个" % len(rescue))
    for meeting, k, nbig, big in rescue[:24]:
        print("    %-14s k=%2d -> %d 个大簇 %s" % (meeting, k, nbig, big[:6]))
    return {"k_hit": hit, "n_meetings": len(rows), "denied": denied,
            "total": tot, "rescue": len(rescue)}


if __name__ == "__main__":
    audit(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
