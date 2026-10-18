# -*- coding: utf-8 -*-
"""DER 扩展：c1_final vs DSH a2ovl，在 gt_ts_diag 的 11 个带时间戳文件上。

- 两边均为 raw 输入（已实测：c1_final 与当前版 claude_a2 在 raw 上逐条一致；
  DSH 旧 output/a2ovl 与当前版 a2_ovl.run(raw) 逐条一致），前端平权。
- GT: output/gt_ts_diag.json（自动时间戳，quiet 文件对齐未验证，仅参考口径）。
- 复用 DSH eval_der.eval_one（0.1s 帧，GT说话人->系统标签多数票），
  口径与 _der_v3.log 一致（collar0.5+trim）。
- 输出按环境分组帧级合并 + 分文件行。
"""
import json, os, sys
import numpy as np

DSH = r"D:\Deepseek_herness\2026-08\diarization"
sys.path.insert(0, DSH)
from eval_der import eval_one  # noqa: E402
from eval_gt import ENV  # noqa: E402

C1 = r"C:\Users\EDY\Desktop\new\diar_result\c1_final"
DSH_OUT = os.path.join(DSH, "output", "a2ovl")

COLLAR, TRIM = 0.5, True


def load_c1(base):
    return json.load(open(os.path.join(C1, base + "_diar.json"),
                          encoding="utf-8"))["sentence_info"]


def load_dsh(base):
    return json.load(open(os.path.join(DSH_OUT, base + "_a2ovl.json"),
                          encoding="utf-8"))["sentence_info"]


def main():
    gts = json.load(open(os.path.join(DSH, "output", "gt_ts_diag.json"),
                         encoding="utf-8"))["files"]
    print(f"===== DER c1 vs DSH: {len(gts)} files, collar={COLLAR}s+trim "
          f"(gt_ts_diag 自动时间戳) =====")
    hdr = f"{'file':<22} {'env':<4} {'dist':<5} | {'c1':>6} {'dsh':>6} | " \
          f"c1(miss/fa/err)       dsh(miss/fa/err)"
    print(hdr)
    print("-" * 80)
    rows = []
    for base in sorted(gts):
        turns = gts[base]["turns"]
        env = ENV.get(base, {}).get("env", "?")
        dist = ENV.get(base, {}).get("distance", "?")
        rec = {"base": base, "env": env, "dist": dist}
        for src, loader in (("c1", load_c1), ("dsh", load_dsh)):
            ns = loader(base)
            n, miss, fa, err, spk_map = eval_one(turns, ns, COLLAR, TRIM)
            der = (miss + fa + err) / max(1, n)
            rec[src + "_der"] = der
            rec[src + "_m"], rec[src + "_f"], rec[src + "_e"] = miss, fa, err
            rec[src + "_n"] = n
        rows.append(rec)
        print(f"{base:<22} {env:<4} {dist:<5} | {rec['c1_der']:>5.1%} "
              f"{rec['dsh_der']:>5.1%} | "
              f"{rec['c1_m']}/{rec['c1_f']}/{rec['c1_e']:>4}      "
              f"{rec['dsh_m']}/{rec['dsh_f']}/{rec['dsh_e']:>4}")

    # 帧级合并：全量 + 分环境
    def agg(env_filter=None):
        res = {}
        for src in ("c1", "dsh"):
            tot = sum(r[src + "_n"] for r in rows
                      if env_filter is None or r["env"] == env_filter)
            der = sum(r[src + "_der"] * r[src + "_n"] for r in rows
                      if env_filter is None or r["env"] == env_filter) / max(1, tot)
            mm = sum(r[src + "_m"] for r in rows
                     if env_filter is None or r["env"] == env_filter) / max(1, tot)
            ff = sum(r[src + "_f"] for r in rows
                     if env_filter is None or r["env"] == env_filter) / max(1, tot)
            ee = sum(r[src + "_e"] for r in rows
                     if env_filter is None or r["env"] == env_filter) / max(1, tot)
            res[src] = (der, mm, ff, ee, tot)
        return res

    for label, filt in (("全量 11 文件", None),
                        ("嘈杂", "嘈杂"), ("安静", "安静")):
        a = agg(filt)
        print(f"\n=== {label}（帧级合并） ===")
        for src in ("c1", "dsh"):
            der, mm, ff, ee, tot = a[src]
            print(f"  {src:<4}: DER={der:.1%}  miss={mm:.1%} fa={ff:.1%} "
                  f"spk_err={ee:.1%}  (GT语音帧={tot})")


if __name__ == "__main__":
    main()
