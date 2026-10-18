# -*- coding: utf-8 -*-
"""带标签全量评估: DSH 当前版(a2ovl) vs c1(v4)，23 文件同口径。

指标复用 compare_c1.score():
  文本      = 真值轮次文本相似度 >=0.35 命中任一输出句
  带标签    = 额外要求真值身份→输出标签一对一映射正确(取最优排列)
两者都是文本代理指标, 不是 DER。

DSH 输出含重叠段(无 spk, 而是 spks=[0,1]+overlap=True), 故带标签给两个口径:
  严格   = 重叠段记 "?" 永不命中(等同 DSH compare_seaco 的 .get("spk","?"))
  计重叠 = 重叠段视为"这些人都在说", 命中其一即算对
c1 无重叠段, 两口径下数值相同。
用法: python eval_labeled.py
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_c1 import BASE, score, labels_of

DSH = Path(r"D:\Deepseek_herness\2026-08\diarization\output\a2ovl")
SOURCES = {"DSH_cur": (DSH, "_a2ovl.json"),
           "c1_v4": (BASE / "diar_result/c1_final", "_diar.json")}
COLS = ["matched", "speaker_matched", "speaker_matched_ovl"]


def score_both(gt, sentences):
    strict = score(gt, sentences)
    return {"matched": strict["matched"],
            "speaker_matched": strict["speaker_matched"],
            "speaker_matched_ovl": score(gt, sentences, overlap_credit=True)["speaker_matched"],
            "gt": strict["gt"], "segments": strict["segments"],
            "overlap_segments": sum(1 for s in sentences if "spk" not in s and s.get("spks"))}


def selfcheck():
    gt = {"speakers": ["A", "B"], "n_turns": 2,
          "turns": [{"spk": "A", "text": "你好啊"}, {"spk": "B", "text": "我知道了"}]}
    plain = [{"start": 0, "end": 1, "spk": 0, "text": "你好啊"},
             {"start": 1, "end": 2, "spk": 1, "text": "我知道了"}]
    assert score(gt, plain)["speaker_matched"] == 2
    # 重叠段: 严格口径不给分, 计重叠口径给分
    ovl = [{"start": 0, "end": 1, "spks": [0, 1], "overlap": True, "text": "你好啊"},
           {"start": 1, "end": 2, "spk": 1, "text": "我知道了"}]
    assert score(gt, ovl)["matched"] == 2
    assert score(gt, ovl)["speaker_matched"] == 1
    assert score(gt, ovl, overlap_credit=True)["speaker_matched"] == 2
    assert labels_of({"spk": 0}, True) == {"0"}
    assert labels_of({"spks": [0, 1]}, False) == {"?"}


def main():
    selfcheck()
    gt = json.loads((BASE / "eval_dsh/ground_truth.json").read_text(encoding="utf-8"))
    rows, groups = [], {}
    for i, g in enumerate(gt):
        # 环境分组: 前12=嘈杂, 后11=安静 (已按 DSH eval_gt.ENV 程序化核对一致)
        group = ("嘈杂" if i < 12 else "安静") + " " + g["distance"]
        row = {"file": g["file"], "group": group}
        for name, (folder, suffix) in SOURCES.items():
            path = folder / (Path(g["file"]).stem + suffix)
            if not path.exists():
                raise FileNotFoundError(path)
            row[name] = score_both(g, json.loads(path.read_text(encoding="utf-8"))["sentence_info"])
        rows.append(row)
        acc = groups.setdefault(group, {n: [0, 0, 0, 0] for n in SOURCES})
        for name in SOURCES:
            s = row[name]
            acc[name][:] = [a + b for a, b in zip(acc[name], [s[c] for c in COLS] + [s["gt"]])]

    names = list(SOURCES)
    head = f"{'场景':<12}" + "".join(f"{n+'/文本':>13}{n+'/带标签':>14}{n+'/计重叠':>14}" for n in names)
    print(head)
    total = {n: [0, 0, 0, 0] for n in names}
    for group in sorted(groups, key=lambda k: (k.split()[0] != "嘈杂", k)):
        line = f"{group:<12}"
        for n in names:
            v = groups[group][n]
            total[n] = [a + b for a, b in zip(total[n], v)]
            line += "".join(f"{f'{v[j]}/{v[3]}':>13}" for j in range(3))
        print(line)
    print("-" * len(head))
    line = f"{'合计':<12}"
    for n in names:
        v = total[n]
        line += "".join(f"{f'{v[j]}/{v[3]}={v[j]/v[3]:.3f}':>15}" for j in range(3))
    print(line)
    out = BASE / "diar_result/labeled_eval_dsh_vs_c1.json"
    out.write_text(json.dumps(
        {"note": "文本相似度>=0.35; speaker_matched=一对一说话人标签映射后正确数; "
                 "_ovl 口径给 DSH 重叠段(spks=[0,1])算命中其一; 文本代理指标, 非 DER",
         "dsh_output_freshness": "output/a2ovl 生成于 09-07; 与今日 a2ovl_seaco_now 10 文件子集对比, "
                                 "分离标签/边界/gap2 逐字节一致, 得分完全相同(文本96 带标签83), "
                                 "唯一差异文件 112415 仅 ASR 分句不同、diag 一致",
         "sources": {n: str(p) for n, (p, _) in SOURCES.items()},
         "total": total, "columns": COLS + ["gt"], "groups": groups, "files": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告:", out)


if __name__ == "__main__":
    main()
