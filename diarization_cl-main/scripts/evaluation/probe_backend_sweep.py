# -*- coding: utf-8 -*-
"""后端扫描：LDA 打分为何净负，以及怎样（如果能）修好。

上一轮把 LDA 的类修成 (身份组, 说话人) 之后净收益反而更差（dev −2.1、holdout −5.1），
而崩掉的场次（S_R004 组）恰好是训练里含**极小类**的那些：S_R004S02C01 的
002-F=2 窗、003-F=2 窗。`lda_fit` 只排除 <2 窗的类，于是 2 窗的类贡献了
接近奇异的判别方向。这是**实验的缺陷**，不是"后端没用"的证据。

同时在扫的轴：
  min_class_windows  —— 每个 LDA 类的纯窗下限（2 / 8 / 16 / 32）
  normalize          —— LDA 输出后是否 L2 归一化再聚类
                        False：欧氏距离 + average linkage（LDA 坐标的原生度量）
                        True ：余弦距离（与裸余弦基线同度量，只换了空间）
  训练池             —— 默认用**本 split 全部场次**（除目标场）拟合；`--meetings`
                        可收窄。**不含 holdout** —— 见 `build_pool` docstring。

零推理。

    python scripts/evaluation/probe_backend_sweep.py --split dev --seconds 600
"""
import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from v12_aishell4 import meeting_group, split_meetings  # noqa: E402
import probe_ab_dev as AB  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, cluster_labels, frame_accuracy, gate_windows, map_by_frames,
    rttm_speech, window_reference)
from probe_pair_oracle import Transform, lda_fit, pure_windows  # noqa: E402
from probe_sentence_attribution import (  # noqa: E402
    attribution, segment_map, sentence_dict_predictions)
from unlocked_selector import CAP, MIN_WIN, build_unlocked, pick_unlocked  # noqa: E402


def load(cache, meeting, window, hop):
    data = np.load(Path(cache) / meeting / f"w{window:g}_h{hop:g}.npz",
                   allow_pickle=False)
    return (data["emb"], np.asarray(data["times"], dtype=float),
            data["speakers"].astype(str), np.asarray(data["votes"], dtype=float))


DEFAULT_CACHE = r"D:\yinp\AISHELL-4\_v13_spike\cache"


def build_pool(meetings, cache, window, hop):
    """加载标注窗缓存，供 LDA 拟合。

    **只加载传入的场次** —— 由调用方决定范围。
    当前唯一调用方是 `main`，它传**本 split 的场次**。

    ⚠️ 曾经这里写的是 `set(dev + holdout)`，于是对 dev 的每一场，
    LDA 方向都从含 8 场 holdout 的池里学 —— 封存集为一个输出数字做了贡献，
    直接违反 spec §3.3。改成按 split 收口后池从 20 场降到 12 场。
    """
    pool = {}
    for meeting in meetings:
        try:
            pool[meeting] = load(cache, meeting, window, hop)
        except FileNotFoundError:
            continue
    return pool


def fit_backend(pool, target, min_class_windows, normalize):
    """用除目标场外的全部纯窗拟合 LDA，类 = (身份组, 说话人)。

    返回 `(fit, kept_names)` —— `kept_names` 是**保留类的名字清单**（已排序），
    不是个数。`classes` 轴的真实分辨率只有两点（见全局约束），
    只有类名清单能让产物自证用了哪批类。
    """
    Xs, ys = [], []
    for meeting, (emb, times, speakers, votes) in pool.items():
        if meeting == target:
            continue
        pe, _, ps = pure_windows(emb, times, speakers, votes)
        if len(pe):
            Xs.append(pe)
            ys.append(np.array([f"{meeting_group(meeting)}:{x}" for x in ps]))
    if not Xs:
        return None, []
    X, y = np.concatenate(Xs), np.concatenate(ys)
    counts = Counter(y.tolist())
    kept_names = sorted(c for c, n in counts.items() if n >= min_class_windows)
    if len(kept_names) < 2:
        return None, []
    fit = lda_fit(X, y, min_class_windows=min_class_windows)
    if fit is None:
        return None, kept_names
    return Transform(fit.mean, fit.weights, normalize=normalize), kept_names


def refine_labels(labels, E, times, rE, rtimes, enabled=True):
    """走 v12 的同一套两级细化，返回 (labels, times, level)。

    **两臂必须同构**：若 `none` 臂与 LDA 臂都在无细化下跑，结论是
    "后端在 AHC 那一层的差异"；而生产里细化按簇质心重新归属、
    会把簇边界错误吸掉一部分 —— **可能正好把 LDA 想修的那部分先修了**。
    不带细化的数字对"这条线有没有东西"没有回答力。

    ⚠️ `vectors` 必须传**聚类时用的那个空间**（LDA 输出），
    与 `real_path_labels` 内部取质心一致。
    """
    if not enabled:
        return labels, times, "none"
    return AB.real_path_labels(labels, E, times, rE, rtimes)


def variants_for(class_limits, norms, k_modes=("true",)):
    """枚举全部变体名。

    命名：`{none|c<类窗下限>}_{cos|euc}[_autok]`

    ⚠️ `_euc` 变体**不进 autok 臂**：`build_unlocked` 内部
    `linkage(E, method="average", metric="cosine")` 是写死的
    （`unlocked_selector.py:48`），而欧氏空间的 LDA 输出必须用欧氏距离。
    让它跑 autok 等于用余弦 silhouette 评价欧氏空间。
    给它加 metric 参数会动到生产共用实现 —— 不值得，跳过即可。
    """
    out = []
    for k_mode in k_modes:
        suffix = "_autok" if k_mode == "auto" else ""
        if k_mode == "auto":
            out.append("none" + suffix)
        elif k_mode == "true":
            out.append("none")
        for c in class_limits:
            for n in norms:
                if k_mode == "auto" and not n:
                    continue
                out.append(f"c{c}_{'cos' if n else 'euc'}{suffix}")
    # 去重保序
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def sentence_metrics(sentences, labels, times, win):
    """句级归属率，带 clean / overlap / overall 三桶，每桶**四类分开**。

    ⚠️ **不改 `AB.score`** —— 它只返回 `overall` 派生的 6 个键，而现有调用方
    （`probe_unlocked_sentence.run_meeting` 等）全是 `**AB.score(...)` 解包，
    给它加桶会静默改掉它们的产物 schema。这里重建 `score` 的取桶路径，
    段表构造与 `score` 逐行一致（`segment_map` → `sentence_dict_predictions`
    → `attribution`），只是不丢掉三桶。

    **弃权与错答必须分开报**：弃权是显式缺失（读者看得到），错答是静默错误
    （读者被误导而不自知）。合报一个 clean 数会让两者在不该互换的预算里互换。

    四类**恰好加满 1**（`attribution.is_unmapped` 的定义含 `not is_abstain`，
    两者互斥；而 `mapped == truth` 在 `mapped` 为空串时恒假）::

        per_sentence + wrong + unmapped + abstain == 1

    · `wrong`    —— 映射到了某个名字，但名字不是该句的真值人（**混人**）
    · `unmapped` —— 有预测标签，但那个标签没被匈牙利映射到任何名字
                    （映射容量不足 / 簇分裂 —— 与"混人"是两种病）
    · `abstain`  —— 该句没有任何预测覆盖（显式缺失）

    **产品验收的归属错误率 = `wrong + unmapped`**（对读者而言两者都是挂错人：
    一个是挂了别人，一个是挂了个不存在的名字），继续受原错答预算（≤8%）约束。
    拆开是为了**诊断**（混人 vs 过分割），不是为了让验收成绩变好 ——
    两个字段都在同一个预算里，拆它不会放松验收。

    早期版本把 `unmapped` 折进 `wrong`（`wrong = 1 - ps - abstain`），
    读的人分不出"混人"与"过分割"；只在 k=8 档量级可见（0.042–0.049），
    但那已占 8% 预算的近一半。
    """
    end = max((s["end"] for s in sentences), default=0.0)
    w = np.full(len(labels), win)
    segs = segment_map(times, labels, w, end)
    pred = sentence_dict_predictions(sentences, segs)
    a = attribution(sentences, pred)
    out = {}
    for bucket in ("overall", "clean", "overlap"):
        b = a[bucket]
        ps, ab, un = b["per_sentence"], b["abstain"], b["unmapped"]
        if ps is None or ab is None or un is None:
            wrong = attr_err = None
        else:
            wrong = 1.0 - ps - ab - un
            attr_err = wrong + un          # 产品口径：两个字段在同一预算里
        out[bucket] = {
            "per_sentence": ps,
            "wrong": wrong,
            "unmapped": un,
            "abstain": ab,
            "attribution_error": attr_err,
            "n": b["n"],
        }
    return out


def stub_row(name, k_mode, skip=None):
    """未出数的变体行。**键集与成功行逐位对齐** —— 产物是按行读的，
    "这个键没写" 与 "这一步没走" 在 JSON 里长得一样（`json.dumps` 省略
    None vs 整个键缺失，读的人分不出）。缺 `classes` 曾让下游必须写
    `.get("classes")` 才能不崩，而 `.get` 把"键不存在"和"值为 None"混成一个结果。
    """
    row = {"accuracy": None, "classes": None, "level": None,
           "n_final": None, "n_classes": None, "k_mode": k_mode,
           "k_used": None, "sentence": None}
    if skip is not None:
        row["skip"] = skip
    return row


def k_modes_arg(text):
    """解析 `--k-modes`，未知取值直接报错。

    不这样做的话 `--k-modes True` 会静默落进 `variants_for` 的 else 兜底：
    既不产 `none` 臂，行里的 `k_modes` 又写着 `True` —— 产物对自己撒谎。
    一个存在意义就是"记录这一跑用了哪种 K 模式"的开关，不能报错数。
    """
    modes = [piece for piece in text.split(",") if piece]
    unknown = sorted(set(modes) - {"true", "auto"})
    if unknown:
        raise argparse.ArgumentTypeError(f"未知 k-mode：{unknown}（只认 true / auto）")
    return modes


def main():
    # Windows 控制台默认 GBK，而本脚本的汇总里有 `⚠`/`⇒` 这类非 GBK 字符 ——
    # 一次 `print` 编不出去就抛 UnicodeEncodeError，**整场扫描白跑**
    # （实测：12 场全出数后死在最后一行判据上，产物都没落）。
    # `errors="replace"` 让不认识的字符退化成 `?` 而不是崩。
    # `ponytail:` 逐字符换 ASCII 是治症状；正解是仓库统一设 `PYTHONIOENCODING`
    # （61 个脚本同病），但那超出本 task 范围。
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):      # 非 tty / 老解释器：不致命
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="显式场次列表；缺省 = 本 split 全部")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=v12.WIN_S)
    ap.add_argument("--hop", type=float, default=v12.HOP_S)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--classes", default="2,8,16,32")
    ap.add_argument("--norm", default="euclid,cosine")
    ap.add_argument("--k-modes", default="true,auto", type=k_modes_arg,
                    help="true = 真值有效人数（诊断）；auto = pick_unlocked 自选（判据）")
    ap.add_argument("--refine", default="on", choices=("on", "off"),
                    help="是否走两级细化。off 仅用于诊断，产物会标 level=none")
    ap.add_argument("--output")
    args = ap.parse_args()

    class_limits = [int(x) for x in args.classes.split(",")]
    norms = [x == "cosine" for x in args.norm.split(",")]

    root = Path(args.root)
    # `AB.load_sentences` 读的是 `AB.DATA` 下的 TextGrid，而本脚本的 `--root`
    # 才是这一跑的数据根 —— 不接上就会**静默读另一个目录**的标注
    # （或在别的机器上直接 FileNotFoundError）。
    AB.DATA = root
    meetings = args.meetings or split_meetings(root, args.split)
    # ★ 池只含本 split —— 见 build_pool docstring 与 test_backend_sweep.py
    pool = build_pool(meetings, args.cache, args.window, args.hop)

    refine_cache = {}
    for meeting in meetings:
        try:
            refine_cache[meeting] = PE.cached_stage(meeting, "refine")
        except FileNotFoundError:
            continue
    missing_refine = sorted(set(meetings) - set(refine_cache))
    if missing_refine:
        # ⚠️ 不能只 `continue`：那会让这些场次**静默**从产物里消失，
        # 而"少了几场"看起来和"本来就这么几场"一模一样 —— 本仓库
        # 反复出现的"不报错只出错数"。§R7：失败场次必须显式记录。
        print(f"⚠️ 无 refine 缓存的场次（跳过）：{missing_refine}", flush=True)

    k_modes = args.k_modes
    variants = variants_for(class_limits, norms, k_modes)
    rows = []
    for meeting in meetings:
        if meeting not in refine_cache:
            # ⚠️ 不能 `continue`：那会让这些场次**静默**从产物里消失，而"少了几场"
            # 看起来和"本来就这么几场"一模一样（§R7）。这里落存根行 —— 与下面
            # "窗数不足"那条路径对称；此前只有它落行，两条失败路径处理不一致。
            row = {"meeting": meeting, "true_k": None, "n_windows": None,
                   "variants": {n: stub_row(n, "auto" if n.endswith("_autok") else "true",
                                            "无 refine 缓存")
                                for n in variants},
                   "k_modes": k_modes}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        rE, rtimes = refine_cache[meeting]
        reference = read_rttm(root / "TextGrid" / f"{meeting}.rttm")
        # `max()` 空序列会抛 ValueError —— 空 RTTM（或所有段都被上面的 `start < seconds`
        # 滤掉）会让整场崩掉，而崩溃的场次与"本来就没这场"在产物里分不开。
        # `--seconds > 0` 时这条不参与求值，所以只在真的读空标注时才生效。
        seconds = args.seconds if args.seconds > 0 else max(
            (s["end"] for s in reference), default=0.0)
        reference = [{"start": s["start"], "end": min(s["end"], seconds),
                      "speaker": s["speaker"]} for s in reference if s["start"] < seconds]
        by_speaker = {}
        for s in reference:
            by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0.0) + s["end"] - s["start"]
        grand = sum(by_speaker.values()) or 1.0
        effective = sorted(n for n, v in by_speaker.items() if v / grand >= args.share)

        emb_all, times_all = cached_windows(args.cache, meeting, args.window, args.hop)
        emb, times = gate_windows(emb_all, times_all,
                                  rttm_speech(reference, seconds=seconds),
                                  args.window, v11.MIN_VAD_FRACTION)
        # 每场**只读一次** TextGrid（`load_sentences` 解的是标注文件，不是缓存），
        # 放进变体循环里会让每个变体各读一遍 —— 纯浪费。
        sentences = AB.load_sentences(meeting)
        row = {"meeting": meeting, "true_k": len(effective),
               "n_windows": int(len(emb)), "variants": {}, "k_modes": k_modes}
        if len(emb) < 2 or len(effective) < 2:
            # ⚠️ 不能 `continue` 掉整行：跳过与"本来就不跑"在产物里必须可区分。
            # 该场窗数不足/有效人数不足，逐变体记存根，行仍然落盘。
            # 与上面"无 refine 缓存"那条同理（§R7：失败场次必须显式记录）。
            reason = "窗数不足" if len(emb) < 2 else "有效人数不足"
            for name in variants:
                row["variants"][name] = stub_row(
                    name, "auto" if name.endswith("_autok") else "true", reason)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue
        k = min(len(effective), len(emb))
        for name in variants:
            base_name = name[:-len("_autok")] if name.endswith("_autok") else name
            k_mode = "auto" if name.endswith("_autok") else "true"
            if base_name == "none":
                vectors, metric, kept = emb, "cosine", None
                rvectors = rE                      # 裸空间：两侧同为 192 维原始嵌入
            else:
                limit = int(base_name[1:base_name.index("_")])
                normalize = base_name.endswith("cos")
                fit, kept = fit_backend(pool, meeting, limit, normalize)
                if fit is None:
                    # `level`/`k_mode`/`k_used` 都必须在**每一行**上都在：产物按行读，
                    # 缺键会让"这一步没走"与"这段代码没写"分不开。键集由 `stub_row`
                    # 统一 —— 三处失败路径各写各的曾让 `classes` 只在一处存在。
                    row["variants"][name] = stub_row(name, k_mode, "未拟合")
                    continue
                vectors, metric = fit(emb), fit.metric
                # ⚠️ 细化窗必须过**同一个** `fit`。`_refine_labels` 里是
                # `rE @ anchors.T`，而 anchors 取自 `vectors` —— 若 rE 留在
                # 原始 192 维、vectors 已在 LDA 空间（维度 = 判别方向数），
                # 这条矩阵乘直接 `ValueError`，**每个 LDA 臂都会崩**。
                # 只 `none` 臂能跑（两侧同为原始嵌入），于是崩溃被伪装成
                # "后端臂无数据"。投影后两侧同空间，与生产同构。
                rvectors = fit(rE)
            if k_mode == "auto":
                chosen = pick_unlocked(build_unlocked(vectors, CAP, MIN_WIN), mode="objective")
                if chosen is None:
                    # 空候选：与生产一致地退化，但**显式记 None 而非静默出数**
                    row["variants"][name] = stub_row(name, k_mode, "空候选")
                    continue
                k_used, clusters = int(chosen["k"]), np.asarray(chosen["labels"])
            else:
                k_used, clusters = k, cluster_labels(vectors, k, metric=metric)
            fclusters, ftimes, level = refine_labels(
                clusters, vectors, times, rvectors, rtimes, args.refine == "on")
            # `AB._win` 只认 refined/2s-only；关掉细化时该用主窗（`v12.WIN_S`）
            fwin = v12.WIN_S if level == "none" else AB._win(level)
            named = map_by_frames(reference, ftimes, fclusters, np.full(len(ftimes), fwin))
            acc = frame_accuracy(reference, ftimes, named, np.full(len(ftimes), fwin))
            # 句级指标必须和 frame 指标一样**每条臂都出**（含 `none` 臂）：
            # 少一条臂就会让"这条臂没跑"与"这条臂在后端上更好"分不开。
            # `load_sentences` 对本场返回空时记 None（而不是崩），与存根行同构。
            sent = sentence_metrics(sentences, fclusters, ftimes, fwin) if sentences else None
            row["variants"][name] = {"accuracy": acc["accuracy"],
                                     "classes": kept,
                                     "level": level, "n_final": int(len(ftimes)),
                                     "n_classes": len(kept) if kept else 0,
                                     "k_mode": k_mode, "k_used": int(k_used),
                                     "sentence": sent}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print(f"\n{'会议':<16}{'K':>3}{'n':>5}" + "".join(f"{v:>11}" for v in variants))
    for row in rows:
        cells = []
        for v in variants:
            a = (row["variants"].get(v) or {}).get("accuracy")
            cells.append(f"{a * 100:10.1f}%" if a is not None else "         -")
        print(f"{row['meeting']:<16}{row['true_k']:>3}{row['n_windows']:>5}"
              + "".join(cells))

    # 基线 = 裸臂。`--k-modes auto` 时根本不存在 `none`（只有 `none_autok`），
    # 写死 `["none"]` 会 KeyError —— 新开的开关不能有值把它跑崩。
    base = {row["meeting"]: (row["variants"].get("none") or
                             row["variants"].get("none_autok") or {}).get("accuracy")
            for row in rows}
    print()
    for v in variants[1:]:
        deltas = []
        for row in rows:
            a = (row["variants"].get(v) or {}).get("accuracy")
            b = base[row["meeting"]]
            if a is not None and b is not None:
                deltas.append(a - b)
        if not deltas:
            continue
        deltas = np.array(deltas)
        wins = int((deltas > 0.01).sum())
        losses = int((deltas < -0.01).sum())
        print(f"  {v:>11}  净值 {deltas.mean() * 100:+6.1f} 点   "
              f"赢{wins}/亏{losses}/平{len(deltas) - wins - losses}")

    # 句级汇总：clean 桶的**四类分开**报。`1 - 归属率` 不是错答率
    # （clean 桶含弃权句且算未答对），硬凑成一个数会让四类互相顶替。
    # 判据看的数是**归属错误率 = 混人 + 未映射**（同一个预算），
    # 混人与未映射分列只为诊断，不为放松验收。
    print()
    print("=== 句级（clean 桶，四类分开）===")
    print(f"  {'变体':>12}  {'归属率':>7}  {'归属错误率':>10}  "
          f"{'= 混人 +':>9}  {'未映射':>7}  {'弃权':>7}")
    means = {}
    for v in variants:
        rows_v = [(row["variants"].get(v) or {}).get("sentence") for row in rows]
        # 无句子的场次 / 存根行 / 未拟合行都会是 None —— 逐场过滤，
        # 空集合直接跳过（不能让一个 None 把整张表打成 NaN 或抛 TypeError）。
        rows_v = [x for x in rows_v if x and x["clean"]["per_sentence"] is not None]

        def avg(key):
            vals = [x["clean"][key] for x in rows_v if x["clean"][key] is not None]
            return None if not vals else float(np.mean(vals))

        got = {k: avg(k) for k in ("per_sentence", "attribution_error",
                                   "wrong", "unmapped", "abstain")}
        means[v] = got
        if got["per_sentence"] is None:
            print(f"  {v:>12}  （无句级数据）")
            continue
        f = lambda x: "     -" if x is None else f"{x * 100:5.1f}%"  # noqa: E731
        print(f"  {v:>12}  {f(got['per_sentence'])}  {f(got['attribution_error'])}  "
              f"{f(got['wrong'])}  {f(got['unmapped'])}  {f(got['abstain'])}"
              f"   (n={len(rows_v)})")

    # ── 判据：`auto` 臂上 LDA 变体的归属错误率相对裸基线降了多少 ──
    # 只看 auto 臂 —— `true` 臂用了明令禁引的真值 K 假设，仅供诊断。
    # 基线是**同一场**的 `none_autok`（`--k-modes true` 时无此臂，退回 `none`）。
    print()
    print("=== 判据（auto 臂 · clean 桶归属错误率，相对裸基线）===")
    auto_variants = [v for v in variants if v.endswith("_autok") and v != "none_autok"]
    if not auto_variants:
        print("  （本次未跑 auto 臂 —— 判据需要 `--k-modes auto`）")
    per_meeting = {}
    for v in auto_variants:
        ds = []
        for row in rows:
            base = (row["variants"].get("none_autok") or row["variants"].get("none") or {})
            b = (base.get("sentence") or {}).get("clean", {}).get("attribution_error")
            t = ((row["variants"].get(v) or {}).get("sentence") or {})
            t = t.get("clean", {}).get("attribution_error")
            if b is None or t is None:
                continue
            ds.append(b - t)          # 正 = 改进
            per_meeting.setdefault(v, []).append((row["meeting"], b - t))
        if not ds:
            print(f"  {v:>12}  （无可用场次）")
            continue
        ds = np.array(ds)
        worst = min(per_meeting[v], key=lambda kv: kv[1])
        verdict = ("=> 支持继续投入" if ds.mean() >= 0.01
                   else "=> 暂停这套 LDA 配置（不足以关闭后端方向）")
        print(f"  {v:>12}  降 {ds.mean() * 100:+5.1f} 点   "
              f"赢{int((ds > 0.01).sum())}/亏{int((ds < -0.01).sum())}/平"
              f"{int((abs(ds) <= 0.01).sum())}   最差场 {worst[0]} {worst[1] * 100:+.1f}"
              f"   {verdict}")

    if args.output:
        # `gate` 必须显式标：本台子用 **RTTM 门**的 562 窗缓存拟合 LDA，
        # 而生产的 VAD 门是另一个窗集（418/422 重合，不是同一个）——
        # 不标的话下次有人会拿它跟生产数字直接比。
        doc = {
            "schema": "v13-backend-sweep-v2",
            "gate": "rttm",
            "split": args.split,
            "k_modes": k_modes,
            "classes": class_limits,
            "norm": args.norm,
            "refine": args.refine,
            "meetings": rows,
            "note": [
                "细化阈值 v12.SLIVER_MARGIN=0.18 在**原始 192 维余弦尺度**上标定，"
                "LDA 拉长判别方向后 margin 尺度整体放大 ⇒ LDA 臂系统性少吸碎段"
                "（实测吸碎段量 0.045–0.18 vs 裸臂 0.32–0.45）。"
                "**归一化治不好**：对 `_cos` 是空操作，对 `_euc` 也只抬到 ~0.13。"
                "**影响方向待验证**：碎段吸收净有益还是净有害尚无对照，"
                "故不能断言净值偏保守（也不能断言偏乐观）——此为已知混杂，"
                "读 `auto` 臂结果时须计入。见 ledger Ruling T2-4。",
                "句级四类：per_sentence + wrong + unmapped + abstain == 1。"
                "产品验收的**归属错误率 = wrong + unmapped**（两者同受 ≤8% 错答预算），"
                "拆开只为诊断（混人 vs 过分割），不为放松验收。",
                "判据只看 `auto` 臂（`true` 臂用真值 K 假设，仅诊断）。"
                "归属错误率降 ≥1 点 ⇒ 支持继续投入；<1 点 ⇒ 暂停这套 LDA 配置，"
                "**不足以关闭整个后端方向**（LDA 无收益推不出 VBx 无收益）。",
                "训练池只含本 split（不含 holdout）—— 见 build_pool docstring。",
                "【dev 12 场实测，schema v2 首跑】`auto` 臂上 LDA 变体的归属错误率"
                "相对裸基线只动 ≤0.4 点且方向不一致（c2 +0.1 最好，c8/c16/c32 反而"
                "−0.1/−0.3/−0.4）⇒ **低于 1 点门槛，暂停这套 LDA 配置**。"
                "`true` 臂同批场次上 c2_cos 降 2.9 点、c8/c16 各降 1.7 点"
                "（相对 `none` 的 21.0%），但**该臂固定的是经 5% 过滤的参考人数，"
                "不是最优划分**，且逐场赢5/亏4/平3、中位 0、单场摆动 ±40 点 —— "
                "故**不得**据此说'收益被选 K 吃掉'。"
                "配套的 K 网格扫描（`v13_k_grid_scan_dev_*.json`）显示：两个空间的"
                "最佳合格候选几乎相同（原始 4.257% vs c2_cos 4.196%），选择损失也"
                "几乎相同（4.893 / 4.882 点）⇒ **LDA 对可达上界几乎没有贡献，"
                "选择器的损失才是主要项**。⚠️ 该 4.9 点是**诊断上界**（逐场用真值"
                "挑最优 K），**不是可达数字**，尚无不读真值的规则能兑现它。"
                "**LDA 无收益推不出 VBx 无收益** —— 暂停的是这套 LDA 配置。",
            ],
        }
        Path(args.output).write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
