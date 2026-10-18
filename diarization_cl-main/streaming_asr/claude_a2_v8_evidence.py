# -*- coding: utf-8 -*-
"""v8 外生证据层：VAD 静音结构 + 重建误差。

设计约束（成因见 milestones/c1_v7_20260910/evidence/v7_三项致命缺陷评审）：
v7 的三个缺陷共享同一形状 —— 判据的参考量由被判决的对象自己产生。
   人数：段级/窗级各硬编码 fcluster(t=2)
   救援门：拿同模型欠切结果当基线（_rescue_gate 的 n_sw_win）
   ASR：逐段解码后无条件 reliable

工作日志 621-623 已实测：三个标准 k 判据在 CAM++ 特征空间全废（远端
within-person 与 between-person 余弦同量级，真实 2 人文件簇内相似度仅
0.48~0.56）。所以"更好的聚类内判据"这条路是死的，本模块提供的证据一律
不依赖 CAM++ 嵌入：

  pause_support   —— 一次说话人切换是否落在真实静音上（VAD，完全外生）
  recon_error     —— 相邻段质心是同一人时的重构残差（跨窗预测，非自比）
  fit_multi_k     —— 各 k 的簇内最大直径（判 k 能否自洽容纳）

纯函数，不加载模型，可直接单测。
"""
import numpy as np

EPS = 1e-9


def _norm(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / (np.linalg.norm(v) + EPS)


def pause_len(a, b, vsegs):
    """[a,b] 内未被 VAD 有声区覆盖的总时长（秒）。"""
    if b <= a:
        return 0.0
    covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in vsegs)
    return max(0.0, (b - a) - covered)


def boundary_support(segs, vsegs, min_pause=0.15, min_run=0.75):
    """每个内部边界的外生证据。

    segs: [(start, end, spk)]（按时间升序）
    返回每个边界一项：{t, pause, shorter, direct, runs}
      direct = pause >= min_pause 且两侧段都 >= min_run
               —— 有静音支撑、且两侧都够长，是"真切换"的强证据。
    这是与说话人模型完全独立的量：静音本来就在 VAD 里，不是聚类算出来的。
    """
    out = []
    for i in range(1, len(segs)):
        s0, e0, _ = segs[i - 1]
        s1, e1, _ = segs[i]
        pause = pause_len(e0, s1, vsegs) if s1 > e0 else 0.0
        shorter = min(e0 - s0, e1 - s1)
        out.append({"t": float(s1), "pause": float(pause),
                    "gap": float(max(0.0, s1 - e0)), "shorter": float(shorter),
                    "direct": bool(pause >= min_pause and shorter >= min_run)})
    return out


def run_stats(segs, vsegs, min_pause=0.15, min_run=0.75, short_dur=0.4):
    """段序列的结构统计：切换总数、无静音支撑的切换、碎段。

    unsupported 是替代 v7 `n_sw_win` 基线的新量：它不跟另一个假设比，
    只问"这次切换在 VAD 上有没有落地"。欠切方案切换少，但这个比值
    不会因此占优——它惩罚的是"切在没有静音的地方"，与切多切少无关。
    """
    sup = boundary_support(segs, vsegs, min_pause, min_run)
    n_sw = len(segs) - 1 if segs else 0
    unsup = sum(1 for b in sup if not b["direct"])
    shorts = [i for i, (s, e, _) in enumerate(segs) if e - s < short_dur]
    return {
        "n_seg": len(segs),
        "n_sw": max(0, n_sw),
        "n_unsupported": int(unsup),
        "unsupported_ratio": float(unsup / n_sw) if n_sw else 0.0,
        "n_short": len(shorts),
        "short_ratio": float(len(shorts) / len(segs)) if segs else 0.0,
        "median_dur": float(np.median([e - s for s, e, _ in segs])) if segs else 0.0,
        "boundaries": sup,
    }


def recon_error(segs, reps, times, win_s, window=(1, 1)):
    """相邻段质心能否互相预测 —— "它跟邻居是不是同一人"。

    与 _rescue_gate 的关键区别：这是跨段预测，不是同一批窗自己跟自己比。
    同一人的邻居能预测它（误差≈噪声底）；不同人预测不了（误差显著抬升）。

    强判据形式（window=(1,1)）：邻居质心的**中点**应当接近本段中心窗质心；
    距离显著大于邻居间距离时，说明中间夹着第三个人（三人以上的征兆）。
    返回每段残差（None 表示该段无法判定）。
    """
    n = len(reps) if reps is not None else 0
    if n < 3:
        return None
    R = np.stack([_norm(r) for r in reps])
    errs = []
    for i in range(n):
        a, b = i - window[0], i + window[1]
        if a < 0 or b >= n:
            errs.append(None)
            continue
        mid = _norm(R[a] + R[b])
        errs.append(float(np.clip(1.0 - float(R[i] @ mid), 0.0, 2.0)))
    return errs


def fit_multi_k(reps, kmax=4):
    """段级树上前 kmax 个间隙，以及各 k 的簇大小分布。

    gap_ratio_k = d[k+1]-d[k] 相对前一个间隙的倍数。v7 只用 gap 选链接方式，
    没用它判人数。这里返回各 k 的间隙**和各簇大小** —— 后者才是"第三个人
    是不是一个孤段"的直接证据（实测：真 3 人 k=3 得 3/3/3，真 2 人硬切 k=3
    得 5/4/1 的孤段）。
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    R = np.stack([_norm(r) for r in np.asarray(reps)])
    if len(R) < 2:
        return []
    Z = linkage(R, method="average", metric="cosine")
    d = Z[:, 2]
    out = []
    for k in range(2, min(kmax, len(R)) + 1):
        i = len(d) - k
        if not (0 <= i and i + 1 < len(d)):
            continue
        prev = float(d[i] - d[i - 1]) if i - 1 >= 0 else float(d[i])
        lab = fcluster(Z, t=k, criterion="maxclust")
        # bincount 会补齐到 max(label)+1；标签未必从 0 起，所以过滤掉空桶，
        # 否则真 3 人会得到 [3,3,3,0] 而被误读成"孤段"。
        sizes = sorted((s for s in np.bincount(np.asarray(lab)).tolist() if s),
                       reverse=True)
        out.append({"k": k, "gap": float(d[i + 1] - d[i]),
                    "prev_gap": prev,
                    "ratio": float((d[i + 1] - d[i]) / (prev + EPS)),
                    "sizes": [int(s) for s in sizes],
                    "min_size": int(min(sizes)),
                    "balance": float(min(sizes) / max(1, np.mean(sizes)))})
    return out


def count_gate(n_seg, seg_tree, vsegs, segs, med_wall=0.35,
               gap_single=0.02, min_total=6.0,
               gap_multi_abs=0.10, min_clus_size=3,
               gap_weak=0.05):
    """三态人数判定：1 / 2 / "uncertain"。

    判据组合（全部外生或跨段，不用簇内相似度——那个已实测失效）：
      1 人：段级树在 k=2 处几乎没有间隙；或整段是一坨语音（无声学换人空间）。
      2 人：k=2 处有明显间隙，且切换大多落在静音上。
      uncertain：证据自相矛盾或不足时，不声称 reliable。

    返回 (n_speakers, reasons)。
    """
    reasons = []
    if n_seg < 2:
        return 1, ["单一代表段"]
    wall = [max(0.0, e - s) for s, e, _ in segs] if segs else [1.0]
    med = float(np.median(wall)) if wall else 0.0
    total = float(sum(wall))
    g = {k["k"]: k for k in (seg_tree or [])}
    g2 = g.get(2)
    st = run_stats(segs, vsegs) if segs else {"unsupported_ratio": 1.0, "n_sw": 0}

    if g2 is None or g2["gap"] <= gap_single:
        reasons.append(f"k=2 间隙 {g2['gap']:.4f} ≤ {gap_single}"
                       if g2 else "段数不足以构造 k=2")
        # 段级树说"没有第二簇"时，无静音支撑的切换大多只是同一人的噪声分割，
        # 不构成"可能是两人"的反证（那会把长单人录音误判成 uncertain）。
        return 1, reasons

    if med < med_wall and total < min_total:
        reasons.append(f"段中位时长 {med:.2f}s < {med_wall}s 且总时长 {total:.1f}s，"
                       f"无声学换人空间")
        return 1, reasons

    # 3 人以上的证据必须**绝对**成立，不能只看 k=3 相对 k=2 的倍数。
    # 实测教训：k3/k2 大只说明"第三簇离得远"，段分布不均时这是常态。
    # 首版按倍数判，把 6 个真值 2 人文件全误判成 uncertain，其中包括
    # 全集合 DER 最好的 20260803112220（1.9%）—— 等于否定自己最好的输出。
    # 真正的判据是"第三簇本身是不是一个像样的簇"：真 3 人 k=3 得 3/3/3，
    # 真 2 人硬切 k=3 得 5/4/1（第三簇是孤段）。
    g3 = g.get(3)
    if g3 is not None and g3["gap"] >= gap_multi_abs:
        if g3["min_size"] >= min_clus_size:
            reasons.append(f"k=3 间隙 {g3['gap']:.4f} ≥ {gap_multi_abs} 且三簇"
                           f"大小 {g3['sizes']} 均 ≥ {min_clus_size}，可能 >2 人 "
                           f"—— 现有素材无真实 3 人录音，标 uncertain")
            return "uncertain", reasons
        reasons.append(f"k=3 间隙 {g3['gap']:.4f} 大但第三簇仅 "
                       f"{g3['min_size']} 段（{g3['sizes']}）= 孤段，判 2 人")

    # k=2 间隙偏小：两簇可分性弱，但仍无任何 3 人证据。这是"2 人但切不准"，
    # 不是"人数不确定"，报出来让调用方自己决定要不要信。
    if g2["gap"] < gap_weak:
        reasons.append(f"k=2 间隙 {g2['gap']:.4f} < {gap_weak}，两簇可分性弱"
                       f"（无 3 人证据，仍判 2 人但置信度低）")
        return 2, reasons

    reasons.append(f"k=2 间隙 {g2['gap']:.4f} > {gap_weak}，静音支撑切换 "
                   f"{1 - st['unsupported_ratio']:.0%}")
    return 2, reasons
