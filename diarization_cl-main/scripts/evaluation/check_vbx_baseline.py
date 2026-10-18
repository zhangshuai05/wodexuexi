# -*- coding: utf-8 -*-
"""核对 VBx spike 的双门基线，落一份可重跑的判据记录。

**为什么必须有这一步**：spec §10.4 规定「臂 0 复现 9.15% / 7.19%，不一致即停」。
复现不了就说明评测链路已经漂了，后面 VBx 的数字全部无意义。所以这条核对是
**独立的可拒绝关口**，不是顺手一算。

    $env:PYTHONIOENCODING="utf-8"
    python -u -B scripts/evaluation/check_vbx_baseline.py

## 输入与回落（控制器 2026-10-14 裁决，R1 追加）

正常输入是 Step 1/3 新跑出来的 `artifacts/_vbx_baseline_{gate}.json`。但 Step 5
要求删掉那两个中间件，于是"删完就再也重跑不了本关口"。⇒ 缺中间件时**回落**读
已发布产物 `v13_selector_stability_diagnosis_{gate}.json`。

⚠️ **回落必须留痕**：只读旧值而不标注，产物里的 `mean` 会**看起来像一次新复现**
（"静默替换"，本线一直在防的东西）。所以每门都带 `freshly_run`：
本次真重跑 = `true`，回落读已发布产物 = `false`，且 `source` 相应写明「本次未重跑」。

⚠️ **★ 的语义不变**：仍然**只在** `abs(mean - expected) > TOLERANCE` 时非零退出。
"缺少中间件"**不是**失败（能回落）；"基线不复现"**才是**失败。

⚠️ **`.json` 只读**：`v13_selector_stability_diagnosis_*.json` 是已发布产物，
本脚本**只读、绝不改写**；只有 `v13_vbx_baseline_gate_check.json` 被覆盖（预期）。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
EXPECTED = {"rttm": 0.0915, "vad": 0.0719}
SELECTOR = "生产 objective(λ=0.005)"
TOLERANCE = 0.0005


def read_mean(path):
    """只读一个产物的「生产 objective(λ=0.005)」行。**不写**。"""
    return json.loads(path.read_text(encoding="utf-8"))["selectors"][SELECTOR]["mean"]


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

    out = {}
    for gate, expected in EXPECTED.items():
        recheck = ARTIFACTS / f"_vbx_baseline_{gate}.json"
        published = ARTIFACTS / f"v13_selector_stability_diagnosis_{gate}.json"
        # 有中间件 ⇒ 用本次真复现的值；没有 ⇒ 回落读已发布产物，但**留下痕迹**。
        freshly_run = recheck.exists()
        mean = read_mean(recheck if freshly_run else published)
        # 同时读一次已发布的诊断产物：两者都记在这里，好让别人一眼看出
        # 「复现值」与「当年落盘值」是否同源，而不用去翻两份文件。
        published_mean = read_mean(published)
        out[gate] = {
            "mean": mean,
            "expected": expected,
            "published_artifact_mean": published_mean,
            "freshly_run": freshly_run,
            "source": (f"{published.name} 的「{SELECTOR}」行" if freshly_run else
                       f"{published.name}（回落：本次未重跑）"),
            "ok": abs(mean - expected) <= TOLERANCE,
        }
    out["tolerance"] = TOLERANCE
    out["ok"] = all(v["ok"] for k, v in out.items() if k in EXPECTED)
    (ARTIFACTS / "v13_vbx_baseline_gate_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not out["ok"]:
        # 非零退出：让"基线不复现"在脚本层面就是**失败**，而不是一行需要人去读的打印。
        raise SystemExit("★ 基线不复现 ⇒ 本 spike 必须停下（spec §10.4）")


if __name__ == "__main__":
    main()
