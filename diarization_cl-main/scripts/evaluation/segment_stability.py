# -*- coding: utf-8 -*-
"""发言段稳定性评分：给候选划分打分，**不读任何真值**。

## 研究问题

现有 `silhouette` 评价的是**主窗聚类本身**（簇内/簇间距离），
而产品质量取决于**细化后的句级归属**。两者之间隔着一整套两级细化。
一个直接建在**细化结果**上的信号，能不能更好地区分候选？

## 信号（只一个机制，两个侧面）

把 VAD 的**连续语音段**当作"一个人一次发言"的近似单元（部署时可得，
不需要真值）。好的划分应当让**同一个语音段内的预测身份一致**，
而且**轻微扰动段边界后仍然一致**。

对给定划分：

    purity_obs  = Σ_s n_s · (段 s 内主导标签的占比) / Σ_s n_s
    purity_null = 同法计算，但把标签向量**随机置换**后（保留簇大小与段大小，
                  只打散两者之间的关联）
    A           = (purity_obs − purity_null) / (1 − purity_null)

`A` 是**按机会校正**的占比（同 adjusted Rand / kappa 的构造），不是新发明的权重。

再对边界做轻微扰动（每个边界独立抖动 U(−δ, +δ)，δ=0.3 秒），
在 δ=0 与 P 次抖动上取 `A` 的均值作为**评分**，
取它们的极差作为**不稳定度**诊断。

## 三条退化，逐条堵住

* **K=1（一人全包）**：`purity_obs = purity_null = 1` ⇒ **A ≡ 0**，不会拿最高分。
  这是按机会校正自然给的，不是硬编码。
* **K=CAP（簇多）**：`A` 以 1 为上界，多切簇不能白拿分；把一个段切碎会同时
  降低 `purity_obs` 与 `purity_null`，净效应需实测（见产物）。
* **不读真值**：输入只有细化后的标签与时刻、窗长、VAD 段。
  不碰 RTTM 人名、TextGrid 句界、真值人数、匈牙利映射。

    from segment_stability import stability_score
"""
import numpy as np

JITTER = 0.3        # 边界扰动幅度（秒）。窗 0.6 秒、hop 0.15 秒 ⇒ 每个边界至多挪 2 窗
N_JITTER = 3        # 扰动次数（固定种子，可复现）
EPS = 1e-12


def segment_ids(times, window, segments, shift=0.0):
    """每个窗的中点在哪个语音段里（-1 = 不属于任何段）。

    段之间重叠时**先匹配者胜**（VAD 段本应互斥；真重叠了要能看出来，
    故同时返回重复匹配的窗数）。
    """
    mid = np.asarray(times, float) + window / 2.0
    ids = np.full(len(mid), -1, dtype=int)
    double = 0
    for j, (a, b) in enumerate(segments):
        hit = (mid >= a + shift) & (mid < b + shift)
        double += int((hit & (ids >= 0)).sum())
        ids[hit & (ids < 0)] = j
    return ids, double


def purity(ids, labels):
    """按窗数加权的"段内主导标签占比"。段内只有一种标签时 = 1。"""
    labels = np.asarray(labels)
    total = weighted = 0.0
    for j in np.unique(ids[ids >= 0]):
        sel = labels[ids == j]
        n = len(sel)
        if not n:
            continue
        total += n
        weighted += n * (np.bincount(sel).max() / n)
    return weighted / total if total else None


def adjusted_purity(ids, labels, n_perm=N_JITTER, rng=None):
    """按机会校正的段内一致性 `A`。

    `A = (obs − null) / (1 − null)`；`null` 由**随机置换标签**估计
    （保留簇大小与段大小，只打散关联）。退化情形（K=1 时 obs=null=1）返回 0.0。
    """
    obs = purity(ids, labels)
    if obs is None:
        return None, None, None
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.asarray(labels)
    nulls = [purity(ids, rng.permutation(labels)) for _ in range(n_perm)]
    null = float(np.mean([x for x in nulls if x is not None]))
    if null >= 1.0 - EPS:
        # K=1（或少数派小于置换噪声）落在这里：`(obs-null)/(1-null)` 是 0/0。
        # ⚠️ 红验证过：删掉这条保护，K=1 会得到 **NaN** —— 不是满分，是 NaN。
        # 注意 `(obs-null)` 是**有符号**的，所以 obs < null 时会给出**负分**，
        # 这是刻意的（"比随机还差"应当被惩罚），不是缺陷。
        return 0.0, obs, null
    return float((obs - null) / (1.0 - null)), obs, null


def stability_score(labels, times, window, segments, jitter=JITTER,
                    n_jitter=N_JITTER, seed=0):
    """对给定划分打分。返回 `(score, detail)`。

    `score` = `A` 在 {δ=0} ∪ {P 次边界抖动} 上的均值；
    `detail["instability"]` = 同一集合上 `A` 的极差（越小越稳）。
    """
    labels = np.asarray(labels)
    if len(labels) == 0 or not segments:
        return None, {"reason": "无窗或无语音段"}
    # ⚠️ 每次 `adjusted_purity` 都**重新播种**：空模型的置换会消耗随机数，
    # 若共用一条流，扰动之间的差异里就混着"这次抽到的置换不一样"——
    # 而 `instability` 要度量的只是**边界**敏感性。同一套置换下，
    # 差异才干净地归因给边界。
    rng = np.random.default_rng(seed)
    mid = np.asarray(times, float) + window / 2.0

    ids0, dbl0 = segment_ids(times, window, segments, 0.0)
    a0, obs0, null0 = adjusted_purity(ids0, labels, n_jitter,
                                      np.random.default_rng(seed))
    if a0 is None:
        return None, {"reason": "没有任何窗落在语音段内"}
    vals, dbl = [a0], dbl0

    for _ in range(n_jitter):
        shifts = rng.uniform(-jitter, jitter, size=len(segments))
        ids = np.full(len(mid), -1, dtype=int)
        for j, ((a, b), s) in enumerate(zip(segments, shifts)):
            hit = (mid >= a + s) & (mid < b + s)
            dbl = max(dbl, int((hit & (ids >= 0)).sum()))
            ids[hit & (ids < 0)] = j
        a, _, _ = adjusted_purity(ids, labels, n_jitter,
                                  np.random.default_rng(seed))
        if a is not None:
            vals.append(a)

    return float(np.mean(vals)), {
        "A_at_zero": a0, "purity_obs": obs0, "purity_null": null0,
        "A_by_perturbation": [round(float(v), 6) for v in vals],
        "instability": float(max(vals) - min(vals)),
        "n_windows_in_segments": int((ids0 >= 0).sum()),
        "n_windows_in_multiple_segments": int(dbl),
    }


def _toy(k_segments=3, per_segment=20, hop=0.15, window=0.6):
    """合成一条与真实配置同几何的输入：段边界落在两窗中点之间。"""
    times = np.arange(k_segments * per_segment) * hop
    mids = times + window / 2.0
    lo, hi = mids[0] - hop / 2, mids[-1] + hop / 2
    edges = np.linspace(lo, hi, k_segments + 1)
    return times, window, [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def _selfcheck():
    """非恒真自检：合成数据，已知正确答案。"""
    times, window, segs = _toy()

    # ① K=1 全包：必须恰好 0 分（不是最高分）
    s_one, d_one = stability_score(np.zeros(len(times), dtype=int), times, window, segs)
    assert s_one == 0.0, f"K=1 应得 0 分，实得 {s_one}"
    assert d_one["purity_obs"] == 1.0 and d_one["purity_null"] == 1.0

    # ② 段与标签一一对应（完美）：δ=0 纯度必须为 1。
    #    ⚠️ 均值**不到 1**：抖动把边界窗推去邻段，而它的标签是对的 ⇒ 边界上的
    #    标签质量本身要交一份"边界税"。这是这个信号的真实性质，不是实现 bug ——
    #    税随 K 增大（K 大 ⇒ 更多相邻段标签不同 ⇒ 更多边界窗会翻），
    #    与纯度项（偏好小 K）方向相反。产物里两个分量都报。
    perfect = np.repeat([0, 1, 2], 20)
    s_ok, d_ok = stability_score(perfect, times, window, segs)
    assert abs(d_ok["purity_obs"] - 1.0) < 1e-12, d_ok["purity_obs"]
    assert d_ok["A_at_zero"] == 1.0, d_ok["A_at_zero"]
    assert 0.9 < s_ok < 1.0, f"完美划分应接近但低于 1（边界税）：{s_ok}"
    # ③ 过分割：每段被切成两个标签 ⇒ 段纯度 0.5，得分显著低于完美，
    #    且**不得**因"簇多"而反超 K=1
    over = np.repeat([0, 3, 1, 4, 2, 5], 10)
    s_over, d_over = stability_score(over, times, window, segs)
    assert d_over["purity_obs"] == 0.5, d_over["purity_obs"]
    assert s_over < s_ok - 0.2, f"过分割应明显低于完美：{s_over} vs {s_ok}"
    assert s_over > s_one, f"过分割仍应高于 K=1：{s_over} vs {s_one}"

    # ④ 扰动的真实性质（**与最初的设计直觉相反，已实测**）：
    #    它不是"只惩罚错在边界的划分"。正确而锐利的边界被抖动推开后同样掉分，
    #    所以 `instability` 由**边界税**主导，不区分错在边界还是错在段中。
    #    能验证的只有这条不变量：扰动只会掉分，不会涨分。
    def one_wrong(offset):
        lab = perfect.copy()
        for j in range(3):
            lab[20 * j + offset] = (j + 1) % 3
        return lab
    s_edge, d_edge = stability_score(one_wrong(19), times, window, segs)
    s_mid, d_mid = stability_score(one_wrong(10), times, window, segs)
    assert d_edge["purity_obs"] == d_mid["purity_obs"] == 0.95, (
        d_edge["purity_obs"], d_mid["purity_obs"])
    for d in (d_ok, d_edge, d_mid):
        assert d["A_at_zero"] >= np.mean(d["A_by_perturbation"]) - 1e-12, (
            "扰动不应让 A 变好", d)

    # ⑥ 抖动幅度为 0 时，扰动必须退化成空操作（证明扰动本身没引入偏差）
    s_zero, d_zero = stability_score(perfect, times, window, segs, jitter=0.0)
    assert abs(s_zero - d_zero["A_at_zero"]) < 1e-12, (s_zero, d_zero["A_at_zero"])

    # ⑤ 空输入不崩
    assert stability_score(np.array([], dtype=int), times, window, segs)[0] is None
    assert stability_score(perfect, times, window, [])[0] is None
    print("selfcheck ✓  K=1→0／完美→纯度1且A0=1／过分割→降／扰动只掉分／"
          "抖动0→空操作／空输入→None")


if __name__ == "__main__":
    _selfcheck()
