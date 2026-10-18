# -*- coding: utf-8 -*-
"""DER 对比: c1_final vs DSH a2ovl, 在官方 gt_ts v3 的 3 个文件上。

复用 DSH 的 eval_der.eval_one(0.1s 帧, GT说话人->系统标签多数票映射)。
口径与 _der_v3.log 完全一致, 保证可与 DSH 已记录的 DER 直接对比。
"""
import json, os, sys
import numpy as np

DSH = r"D:\Deepseek_herness\2026-08\diarization"
sys.path.insert(0, DSH)
from eval_der import eval_one  # noqa: E402

FILES = ["Call20260718211842", "Call20260718211938",
         "Note20260803112220"]
C1 = r"C:\Users\EDY\Desktop\new\diar_result\c1_final"
DSH_OUT = os.path.join(DSH, "output", "a2ovl")


def load_c1(base):
    d = json.load(open(os.path.join(C1, base + "_diar.json"),
                       encoding="utf-8"))
    return d["sentence_info"]


def load_dsh(base):
    d = json.load(open(os.path.join(DSH_OUT, base + "_a2ovl.json"),
                       encoding="utf-8"))
    return d["sentence_info"]


def main():
    gts = json.load(open(os.path.join(DSH, "output", "gt_ts.json"),
                         encoding="utf-8"))
    rows = []
    for collar, trim, tag in [(0.0, False, "collar0"),
                              (0.5, False, "collar0.5"),
                              (0.5, True, "collar0.5+trim")]:
        print(f"\n===== {tag} =====")
        print(f"{'file':<22} {'src':<6} {'GT语音帧':>7} {'miss':>5} "
              f"{'fa':>5} {'spk_err':>7} {'DER':>7}")
        print("-" * 68)
        for base in FILES:
            turns = gts["files"][base]["turns"]
            for src, loader in (("c1", load_c1), ("dsh", load_dsh)):
                ns = loader(base)
                n_sp, miss, fa, err, spk_map = eval_one(turns, ns, collar, trim)
                der = (miss + fa + err) / max(1, n_sp)
                print(f"{base:<22} {src:<6} {n_sp:>7} {miss:>5} {fa:>5} "
                      f"{err:>7} {der:>6.1%}")
                rows.append((tag, base, src, n_sp, miss, fa, err, der))

    # 帧级合并汇总(collar0.5+trim), 与 _der_compare 的汇总口径一致
    print("\n===== 汇总 (collar0.5+trim, 帧级合并) =====")
    for src in ("c1", "dsh"):
        for tag in ("collar0.5+trim",):
            sub = [(n, m, f, e, d) for (t, b, s, n, m, f, e, d) in rows
                   if t == tag and s == src]
            tot = sum(x[0] for x in sub)
            der = sum(x[4] * x[0] for x in sub) / max(1, tot)
            miss = sum(x[1] for x in sub) / max(1, tot)
            fa = sum(x[2] for x in sub) / max(1, tot)
            err = sum(x[3] for x in sub) / max(1, tot)
            print(f"  {src:<6}: DER={der:.1%}  miss={miss:.1%} "
                  f"fa={fa:.1%} spk_err={err:.1%}  (GT语音帧={tot})")


if __name__ == "__main__":
    main()
