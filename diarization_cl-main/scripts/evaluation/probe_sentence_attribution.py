# -*- coding: utf-8 -*-
"""句子级说话人归属率：主指标的定义与干跑（零推理，只读缓存）。

## 定义（**这是主指标**）

**句子** = 某个说话人 tier 上的一个非空 `mark` 段（标注自己在停顿处切的发言单位）。
这一条是 `交接文档_0924.md` §6 第二步原始建议，**经实测确认正确**：

    L_R003S01C02 单 tier mark 515 段，相邻空档 19.8% <0.3s、43.6% <1.0s
    （006-M 的 127 个 mark 里 59 个空档 <0.3s）→ 这些 mark 是句子，
    空档是换气。按空档>0.5s 合并得 367 个话语段。

**归属** = 与"预测说话人时间轴"重叠最长的预测说话人。句子自带锚说话人（真值），
所以这不是循环——循环风险此前被用错了地方，见下。

## 相对文档草案的两处修正（都有实测理由）

### 修正 1：弃权判据换成"预测未覆盖"，不是"无说话人占≥50%"

文档原判据（"无任何说话人占该句 ≥50% → 弃权"）**结构上永不触发**：
句子的 `xmin/xmax` 取自锚的 `mark` 边界，锚对自己的句子覆盖**恒为 100.00%**
（50 场 22753 句实测最小值 100.00%，无一例外）。判据恒假，等于没写。

改判"**预测**未覆盖该句"后，实测弃权率：**test 5.7% / train_L 待测**，
且不偏向任何说话人。这是真实可报的量。

### 修正 2：句子**不要**用全 tier 并集去切（本探针初版犯过的错）

初版照"重叠区要成为独立单位"的直觉，把全 tier 非空 mark 的边界取并集切段，
`L_R003S01C02` 从 **515 段被切成 123 段（24%）**——等于按**别人的话音边界
切碎本说话人的句子**，弃权随之假报 25.2%，且 **88.7% 的 `005-M` 被弃权**。
正确单位是 515（或按空档合并的 367）。**重叠不需要靠改切段单位来处理**。

## 循环风险的正确位置

句子自带真值锚，所以"主项"确实接近恒真——但**不是因为指标错**：
只要预测时间轴与句子只部分重叠，归属就有真实失败模式。约束来自两侧：
预测试图覆盖整场（连静音一起标），而句子只覆盖某人的发言。

真正会退化的用法是**用句子自己的 tier 去查它该属于谁**（那是同义反复）。
本探针的锚只用来出真值，映射与打分都在预测侧。

## 复现 / 干跑

    python -u -B scripts/evaluation/probe_sentence_attribution.py --meeting L_R003S01C02
    python -u -B scripts/evaluation/probe_sentence_attribution.py --split dev --ks auto
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# 只插 `streaming_asr/`：探针用它拿 `claude_a2_v11`。
# **不要插仓库根**——那里有一份同名 `claude_a2.py`（DSH 测试台用），
# 插进去会遮住 `streaming_asr/claude_a2.py`，连带毁掉 test_service 的导入。
# `scripts/evaluation/` 不用插：它是本脚本自身目录，python 自动加。
# 回归测试：`streaming_asr/tests/test_evaluator_path_hygiene.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_asr"))

import claude_a2_v11 as v11  # noqa: E402

from probe_gap_decomp import cached_windows, gate_windows  # noqa: E402
from unlocked_selector import build_unlocked, pick_unlocked  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
VAD_CACHE = CACHE / "vad"
DATA = Path(r"D:\yinp\AISHELL-4\data\test")
WIN, HOP, SECONDS = 2.0, 1.0, 600.0
MIN_VAD = v11.MIN_VAD_FRACTION
FRAME = 0.01
MIN_SENTENCE = 0.5


# ───────────────────────── TextGrid ─────────────────────────

def read_textgrid(path):
    """标准 ooTextFile，每说话人一个 IntervalTier。返回 {tier: [(xmin,xmax,text)]}。

    仓库无 praatio/textgrid 依赖，格式够简单，手写 15 行即可（不为一次读取装包）。
    """
    tiers, name, cur = {}, None, None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("name = "):
            name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("intervals ["):
            cur = None
        elif line.startswith("xmin = "):
            cur = {"xmin": float(line.split("=", 1)[1])}
        elif line.startswith("xmax = ") and cur is not None:
            cur["xmax"] = float(line.split("=", 1)[1])
        elif line.startswith("text = ") and cur is not None:
            cur["text"] = line.split("=", 1)[1].strip().strip('"')
            tiers.setdefault(name, []).append((cur["xmin"], cur["xmax"], cur["text"]))
            cur = None
    return tiers


def sentences_from_textgrid(tiers, seconds=None):
    """句子 = 单 tier 的非空 mark 段（每段自带锚说话人 = 真值）。

    返回 [{start, end, speaker, text}]，按起始时间排序。
    """
    out = []
    for name, ivs in tiers.items():
        for s, e, text in ivs:
            if not text.strip():
                continue
            if seconds is not None and s >= seconds:
                continue
            end = min(e, seconds) if seconds is not None else e
            if end - s <= 0:
                continue
            out.append({"start": s, "end": end, "speaker": name, "text": text.strip()})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def annotate_overlap(sentences, tiers, rel_boundary=0.3):
    """给每句标 other_cov = 其他说话人覆盖该句的时长占比。

    只有当别人的 mark 覆盖超过 `rel_boundary` 比例时才算"重叠句"——
    标注在换气处切开时会有几毫秒的边界交叠，那不是真重叠。
    """
    for sent in sentences:
        s, e = sent["start"], sent["end"]
        span = e - s
        other = 0.0
        for name, ivs in tiers.items():
            if name == sent["speaker"]:
                continue
            for s2, e2, t2 in ivs:
                if not t2.strip() or s2 >= e or e2 <= s:
                    continue
                other += max(0.0, min(e, e2) - max(s, s2))
        sent["other_cov"] = min(1.0, other / span) if span else 0.0
        sent["overlapped"] = sent["other_cov"] > rel_boundary
    return sentences


# ───────────────────────── 预测 ─────────────────────────

def _name(raw):
    """簇号 → 预测名。`None` 与 `""` 表示"无预测"。

    ⚠️ **不要用 `if nm` 判空**：整数簇号 `0` 在 Python 里是 falsy，
    会把整个 0 号簇丢掉（k=1 时 `clusters_for` 返回全 0，于是全判成无覆盖）。
    必须显式判 `None` / `""`。
    """
    return "" if raw is None else str(raw)


def frame_map(starts, labels, durations, end, frame=FRAME):
    """把 (起, 名, 长) 铺成逐帧名数组。后写覆盖先写。"""
    n = int(end / frame) + 2
    grid = np.full(n, "", dtype=object)
    for t, name, dur in zip(starts, labels, durations):
        name = _name(name)
        if name == "":
            continue
        a = max(0, int(round(t / frame)))
        b = min(n, int(round((t + dur) / frame)))
        if b > a:
            grid[a:b] = name
    return grid


def segment_map(starts, labels, durations, end):
    """把预测铺成**段表**（不是逐帧）。句子多于窗时逐帧 O(句长/0.01) 太浪费。"""
    segs = [(float(t), float(t + d), _name(nm))
            for t, nm, d in zip(starts, labels, durations) if _name(nm) != ""]
    segs.sort(key=lambda x: x[0])
    return segs


def sentence_dict_predictions(sentences, segs):
    """每句取与之重叠最长的**预测**说话人。无任何重叠 → 弃权(None)。

    段表按起点排序，用二分定位候选段（每句只需扫 O(log n + 命中数)）。
    """
    import bisect
    starts = [s[0] for s in segs]
    pred = []
    for sent in sentences:
        a, b = sent["start"], sent["end"]
        hi = bisect.bisect_left(starts, b)
        lo = bisect.bisect_left(starts, a - 1.0)
        # 从 lo 往前退，覆盖"起点早于 a 但仍在延伸"的段
        j = lo
        while j > 0 and segs[j - 1][1] > a:
            j -= 1
        counts = Counter()
        for i in range(j, hi):
            s0, e0, nm = segs[i]
            cover = min(e0, b) - max(s0, a)
            if cover > 0:
                counts[nm] += cover
        pred.append(counts.most_common(1)[0][0] if counts else None)
    return np.asarray(pred, dtype=object)


# ───────────────────────── 指标 ─────────────────────────

def hungarian_map(pred, truth, weights):
    """句级混淆矩阵上求全局最优置换（**按 `weights` 最大化**）。

    `der.py` 用贪心；句级在重叠句上有歧义，全局最优更稳。scipy 已在依赖里。
    返回 {pred: truth}。

    ⚠️ **调用方的目标函数与 `weights` 必须一致，否则置换是另一口径的最优。**
    `attribution` 传的 `weights` 是**句时长**，所以它求的是"时长最优"置换，
    而主指标 `per_sentence` 是**按句计数**——两者的最优置换可以不同
    （一个长句能压过好几个短句）。实测（`probe_mapping_objective.py`, dev 12 场
    × k=1..12, 144 单元）：按句主指标仅 1/144 单元受影响（最大 2.66 pts，
    0/12 场改变选出的 k），时长加权辅表 24/144 单元受影响（最大 7.46 pts）。

    要用按句计数映射，传 `weights=[1]*n` 即可（本函数不需要改）：
    最大化计数与最大化时长是同一个 `linear_sum_assignment`，只换权重。
    """
    pairs = Counter()
    for p, t, w in zip(pred, truth, weights):
        # 与 `_name()` 同一条规则：**显式判 None**，不要用真值判空——
        # 整数标签 `0` 是 falsy，会被静默排除出混淆矩阵。
        if p is None or p == "" or not t:
            continue
        pairs[(str(p), str(t))] += w
    preds = sorted({p for p, _ in pairs})
    truths = sorted({t for _, t in pairs})
    if not preds or not truths:
        return {}
    pi = {p: i for i, p in enumerate(preds)}
    ti = {t: i for i, t in enumerate(truths)}
    cost = np.zeros((len(preds), len(truths)))
    for (p, t), w in pairs.items():
        cost[pi[p], ti[t]] = -w          # 最大化权重 → 取负做最小化
    rows, cols = linear_sum_assignment(cost)
    return {preds[r]: truths[c] for r, c in zip(rows, cols) if cost[r, c] < 0}


def attribution(sentences, pred, min_sentence=MIN_SENTENCE):
    """句子级归属率：时长加权为主表，按句平均为辅表；分有/无重叠两栏。

    三条记账规则（都由测试锁住）：

    * **弃权计入分母，算未答对。** 弃权率另单列，用来区分"全弃权"与"全答错"。
      早先版本把无覆盖句移出分母，导致"只答 1 句且答对"得 100%
      （k=2 碎簇只剩 1 窗时真实触发过）。
    * **短句留在总体里**，另单列 `short` 一栏。项目负责人明确：不能仅凭
      "噪声大"把短句从总体剔除——那会掩盖系统在短句上的真实失败。
    * **重叠句单列**：合并报会把信号稀释（无重叠句的主项近似恒真）。
    """
    if not sentences:
        return {"overall": _empty(), "overlap": _empty(), "clean": _empty(),
                "short": _empty(), "n_sentences_all": 0, "n_sentences_scored": 0,
                "n_dropped_short": 0, "n_wrong": 0, "n_abstain": 0,
                "n_unmapped": 0, "n_mapping_pairs": 0}

    truth = [s["speaker"] for s in sentences]
    p = list(pred)
    # 信任边界校验：长度不齐时给出可读错误，而不是让 `pred[j]` 在后面抛 IndexError。
    # （真实踩过：调试时给 4 句传了 3 个预测，报的是难以定位的
    #  `IndexError: list index out of range`，指向的是第 278 行而非调用处。）
    if len(p) != len(sentences):
        raise ValueError(
            f"pred 长度与 sentences 不一致：{len(p)} vs {len(sentences)}。"
            "每个句子必须有一个对应预测（无预测请显式给 None）。"
        )
    weights = [s["end"] - s["start"] for s in sentences]
    # 映射用**句时长**权重（= 时长最优置换），而主指标报 `per_sentence`（按句计数）。
    # 这是**有意的口径不一致**，量级已测：按句主指标仅 1/144 单元受影响、
    # 最大 2.66 pts、0/12 场改变选出的 k（详见 `hungarian_map` docstring 与
    # `probe_mapping_objective.py`）。改成按句计数映射只需传 `[1.0]*len(weights)`。
    mapping = hungarian_map(p, truth, weights)
    mapped = [mapping.get(str(x), "") for x in p]

    # 弃权与未匹配是**两个不同的量**，全文件只用这一份判定（项目负责人明确）。
    #   abstain  = 该句没有任何预测覆盖
    #   unmapped = 该句有预测，但那个标签没被匈牙利映射到任何名字
    #              （映射容量不足 / 簇分裂造成；**不是**弃权）
    #
    # ⚠️ 判据是**显式无预测**（None / ""），不是 falsy ——
    # 整数簇号 `0` 是 falsy，用 `not pred[j]` 会把它误算成弃权。
    # 这是同一条陷阱的第三处（另两处：`_name()`、`hungarian_map`）。
    def is_abstain(j):
        return pred[j] is None or pred[j] == ""

    def is_unmapped(j):
        return not is_abstain(j) and not mapped[j]

    def split(mask):
        keep = [j for j in range(len(sentences)) if mask(sentences[j])]
        n = len(keep)
        if not n:
            return _empty()
        w = sum(weights[j] for j in keep)
        hit_w = sum(weights[j] for j in keep if mapped[j] == truth[j])
        hit_n = sum(1 for j in keep if mapped[j] == truth[j])
        return {"n": n, "seconds": round(w, 1),
                "duration_weighted": round(hit_w / w, 4) if w else None,
                "per_sentence": round(hit_n / n, 4),
                "abstain": round(sum(1 for j in keep if is_abstain(j)) / n, 4),
                "unmapped": round(sum(1 for j in keep if is_unmapped(j)) / n, 4)}

    return {"overall": split(lambda s: True),
            "overlap": split(lambda s: s["overlapped"]),
            "clean": split(lambda s: not s["overlapped"]),
            "short": split(lambda s: s["end"] - s["start"] < min_sentence),
            "n_sentences_all": len(sentences),
            "n_sentences_scored": len(sentences),
            "n_dropped_short": 0,
            "n_wrong": int(sum(1 for j in range(len(sentences))
                               if mapped[j] and mapped[j] != truth[j])),
            "n_abstain": int(sum(1 for j in range(len(sentences)) if is_abstain(j))),
            "n_unmapped": int(sum(1 for j in range(len(sentences)) if is_unmapped(j))),
            # `n_mapping_pairs` = 匈牙利映射的配对数（= min(预测簇数, 真名数)）。
            # 旧名 `n_mapped_unused` 会被读成"未被使用的映射数"，与实际含义相反。
            "n_mapping_pairs": len(mapping)}


def _empty():
    return {"n": 0, "seconds": 0, "duration_weighted": None,
            "per_sentence": None, "abstain": None, "unmapped": None}


# ───────────────────────── 主流程 ─────────────────────────

def clusters_for(emb, k):
    from scipy.cluster.hierarchy import fcluster, linkage
    if k <= 1 or len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    tree = linkage(emb, method="average", metric="cosine")
    return v11._relabel_by_first_seen(fcluster(tree, t=k, criterion="maxclust"))


def run_meeting(meeting, ks, mode, seconds, data_root=DATA):
    tg_path = data_root / "TextGrid" / f"{meeting}.TextGrid"
    vad_file = VAD_CACHE / f"{meeting}.json"
    if not tg_path.exists() or not vad_file.exists():
        return None
    emb_all, times_all = cached_windows(str(CACHE), meeting, WIN, HOP)
    intervals = [(a, b) for a, b in
                 json.loads(vad_file.read_text(encoding="utf-8"))["intervals"]]
    emb, times = gate_windows(emb_all, times_all, intervals, WIN, MIN_VAD)
    if len(emb) < 2:
        return None
    tiers = read_textgrid(tg_path)
    sentences = annotate_overlap(sentences_from_textgrid(tiers, seconds), tiers)
    if not sentences:
        return None
    end = max(s["end"] for s in sentences)
    # `build_unlocked` 含 linkage + 逐 k 轮廓系数，开销不小；只有需要 "auto"
    # 时才构造（默认网格 1..8 不含 auto，构造了就白做）。
    cands = build_unlocked(emb) if "auto" in ks else None
    out = {}
    for k in ks:
        if k == "auto":
            chosen = pick_unlocked(cands, mode=mode)
            kk, labels, sizes = chosen["k"], chosen["labels"], chosen["sizes"]
        else:
            kk = k
            labels = clusters_for(emb, kk)
            sizes = [int((labels == l).sum()) for l in range(kk)]
        segs = segment_map(times, labels, np.full(len(times), WIN), end)
        pred = sentence_dict_predictions(sentences, segs)
        res = attribution(sentences, pred)
        res.update({"k": kk, "sizes": sizes, "n_windows": len(emb),
                    "n_pred_speakers": len(set(labels))})
        out[str(k)] = res
    return out


def fmt(x):
    return "-" if x is None else f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", default="L_R003S01C02")
    ap.add_argument("--split", help="dev / holdout")
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--mode", default="objective")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--out")
    args = ap.parse_args()

    ks = ["auto"] if args.ks == "auto" else [int(x) for x in args.ks.split(",")]
    if args.split:
        from v12_aishell4 import split_meetings
        meetings = split_meetings(DATA, args.split)
    else:
        meetings = [args.meeting]

    report = {}
    for meeting in meetings:
        res = run_meeting(meeting, ks, args.mode, args.seconds)
        if res is None:
            print(f"跳过 {meeting}（无缓存或无 TextGrid）")
            continue
        report[meeting] = res
        print(f"\n===== {meeting} =====")
        print(f"{'k':>4}{'簇大小':>24}{'主表(时长)':>11}{'辅表(按句)':>11}"
              f"{'重叠句':>9}{'无重叠':>9}{'弃权':>8}{'句数':>7}")
        for r in res.values():
            s = r["overall"]
            print(f"{r['k']:>4}{str(r['sizes'])[:22]:>24}"
                  f"{fmt(s['duration_weighted']):>11}{fmt(s['per_sentence']):>11}"
                  f"{fmt(r['overlap']['duration_weighted']):>9}"
                  f"{fmt(r['clean']['duration_weighted']):>9}"
                  f"{fmt(s['abstain']):>8}{s['n']:>7}")
        best = max(res.values(), key=lambda r: r["overall"]["duration_weighted"] or 0)
        worst = min(res.values(), key=lambda r: r["overall"]["duration_weighted"] or 0)
        print(f"  → 最好 k={best['k']} ({fmt(best['overall']['duration_weighted'])}) "
              f"最差 k={worst['k']} ({fmt(worst['overall']['duration_weighted'])})")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"schema": "v13-sentence-attribution-v2",
                        "win": WIN, "hop": HOP, "seconds": args.seconds,
                        "mode": args.mode, "unit": "single-tier-mark",
                        "reports": report}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
