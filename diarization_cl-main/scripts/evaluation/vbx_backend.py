# -*- coding: utf-8 -*-
"""VBx 后端：把 BUT 的 `VB_diarization` 接到本仓库的窗嵌入上。

**这里不实现算法。** 算法体是 `vbx_src/VB_diarization.py`（BUT 官方逐行复制），
本文件只负责：把窗嵌入变成它要的 (`fea`, `sm`, `iE`, `V`)，把它的输出变回
本仓库的硬标签约定，以及把初值定死以保证可复算。

## 三臂
- 臂 0（基线）：生产选择器，不走这里。
- 臂 1 `arm="vbx_ahc"`：用**生产选择器给的** AHC 标签做初值，`maxSpeakers = k0`。
- 臂 2 `arm="vbx_flat"`：**固定种子**的 Dirichlet 初值，`maxSpeakers = cap`。

## 实测行为（Task 5 探查，非推测）
一个**退化的**初值（例如 `labels_ahc` 全 0，即 k0=1）是 VB 的**不动点**：
`init_smoothing=5.0` 的 softmax 把单列 one-hot 变成全 1 的均匀后验，
而 VB 在该数据上 2 轮即收敛（`epsilon=1e-6`），后验仍接近均匀 ⇒ `argmax` 恒为 0。
**所以臂 1 的输出在某些初值下会等于输入的 AHC 标签** —— 这不是"臂 1 退化成臂 0"，
而是「k0=1 的初值没有信息可供 VB 拆分」。断言"臂 1 输出必须 ≠ 输入的 AHC 标签"
只在**非退化初值**（k0 ≥ 2）下成立。详见 task-5-report.md。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax  # noqa: E402

_VBX_SRC = Path(__file__).resolve().parent / "vbx_src"
if str(_VBX_SRC) not in sys.path:
    sys.path.insert(0, str(_VBX_SRC))

import VB_diarization as VD  # noqa: E402
from diarization_lib import AHC, cos_similarity, twoGMMcalib_lin  # noqa: E402,F401

SEED = 20261013

# spec / Global Constraints 定死的超参：判据臂固定 Fb=17，其余全部官方默认。
VBX_HYPER = {"Fa": 0.3, "loopProb": 0.99, "maxIters": 40, "epsilon": 1e-6,
             "init_smoothing": 5.0, "minDur": 1, "threshold": -0.015}
FB_GRID = [1, 5, 10, 17, 30, 60]
FB_CRITERION = 17


def engine_name():
    """产物要记：跑的是真 numexpr 还是我们的 numpy 垫片。

    `VB_diarization.py` 顶部是 `import _numexpr as ne`。垫片在位时 `ne.__name__`
    是 `"_numexpr"`（仓内文件名），被真 numexpr 顶掉时才是 `"numexpr"`。
    所以这里**不能**用 `import numexpr` 是否成功来判断。
    """
    return "numexpr" if VD.ne.__name__ == "numexpr" else "numpy"


def _relabel_by_first_seen(labels):
    """与 `v11._relabel_by_first_seen` 同约定：按首次出现的顺序编成 0..k-1。"""
    mapping, out = {}, np.empty(len(labels), dtype=int)
    for i, value in enumerate(labels):
        key = int(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        out[i] = mapping[key]
    return out


def ahc_init_labels(emb, threshold=None):
    """官方 `vbhmm.py` 的 AHC 初值：两高斯标定阈值 + 余弦相似度。

    只用于**臂 1 之外的诊断**。臂 1 用的是生产选择器给的标签（`labels_ahc`），
    这样"VB 能把生产的错吸掉多少"才和生产可比。

    ⚠️ `cos_similarity` 的**私有前置条件**：行必须是 L2 单位向量。
    它内部直接 `assert np.allclose(ones, np.sum(x**2, axis=1))`，
    传未归一化的行会**抛裸 AssertionError**，不是给你一个错的相似度。
    本函数不做隐式归一化 —— 归一化口径由调用方决定（生产侧是 CAM++ 嵌入的
    L2-norm），静默替你归一化会让"到底归一化没有"这个口径消失在黑盒里。

    ⚠️ 实测（Task 5，合成 300x192、3 个植入簇）：该阈值在**窗级**嵌入上会
    给出约 0.49，把几乎每个窗都判成单例簇。官方 `vbhmm.py` 用它是给
    **整段话语级** x-vector 聚类的，不是给滑窗帧用的。本函数只作诊断，
    **不要**拿它的输出去当臂 1 的初值。
    """
    threshold = VBX_HYPER["threshold"] if threshold is None else threshold
    scr = cos_similarity(np.asarray(emb, dtype=np.float64))
    thr, _ = twoGMMcalib_lin(scr.ravel())
    return AHC(scr, thr + threshold)


def _dirichlet_init(n_frames, max_speakers, seed):
    """臂 2 的初值：**固定种子**的 Dirichlet 随机后验。

    官方的 `VB_diarization` 在 `gamma is None` 时自己走 `np.random.gamma`，
    **没有种子** —— 那样臂 2 跑两次两个数，产物不可复算。所以由调用方给。

    独立命名（而不是内联在 `arm_labels` 里）是为了让测试能钉住"这一步真的发生了"：
    测试用替身记录 `VB_diarization` 收到的 `gamma`，再与本函数在**测试内独立**的
    调用结果逐位比对。内联的话就只能靠"跑两次比结果"，而那实测**红不了** ——
    传 `None` 时 VB 自己走全局 `np.random`，但收敛后 `argmax` 可能仍然一致。
    """
    rng = np.random.default_rng(seed)
    gamma = rng.gamma(1.0, size=(n_frames, max_speakers))
    return gamma / gamma.sum(axis=1, keepdims=True)


def arm_labels(emb, plda, labels_ahc, arm, fb, cap=10, seed=SEED):
    """跑一次 VBx，返回本仓库约定的硬标签（0..k-1，按首见顺序）。

    返回数组长度为 `len(emb)`，`dtype=int`，取值恰为 `0..k-1`（k = 实际出现的
    说话人数），可直接喂给下游的 `sizes` / 匈牙利映射。

    `gamma` **永远由本函数显式构造后传入**，绝不给 `VB_diarization` 传 `None`：
    那条路径内部用无种子的 `np.random.gamma`，臂 2 就不可复算了。

    ⚠️ 已知边界（实测，非推测）：`labels_ahc` 全为同一个值时 `k0 == 1`，
    `maxSpeakers=1` 的 VB 只能输出单一说话人 ⇒ 返回值等于输入标签。
    这不是缺陷，是"初值没给出可拆分的结构"。见模块 docstring。
    """
    emb = np.asarray(emb, dtype=np.float64)
    fea = plda.to_lda_space(emb)
    R = fea.shape[1]
    sV = plda.speaker_prior()

    if arm == "vbx_ahc":
        labels_ahc = np.asarray(labels_ahc).reshape(-1)
        k0 = len(set(int(v) for v in labels_ahc))
        assert labels_ahc.min() >= 0 and labels_ahc.max() + 1 == k0, (
            "AHC 标签必须是 0..k-1 的连续整数；稀疏下标会让 onehot 的列对不上")
        onehot = np.zeros((len(labels_ahc), k0))
        onehot[np.arange(len(labels_ahc)), labels_ahc] = 1.0
        gamma0 = softmax(onehot * VBX_HYPER["init_smoothing"], axis=1)
        max_speakers = k0
    elif arm == "vbx_flat":
        gamma0 = _dirichlet_init(len(fea), cap, seed)
        max_speakers = cap
    else:
        raise ValueError(f"未知臂：{arm!r}（只支持 vbx_ahc / vbx_flat）")

    gamma, _pi, _Li = VD.VB_diarization(
        fea, np.zeros(R), np.eye(R), np.diag(sV),
        pi=None, gamma=gamma0, maxSpeakers=max_speakers,
        maxIters=VBX_HYPER["maxIters"], epsilon=VBX_HYPER["epsilon"],
        loopProb=VBX_HYPER["loopProb"], Fa=VBX_HYPER["Fa"], Fb=fb,
        minDur=VBX_HYPER["minDur"])
    return _relabel_by_first_seen(gamma.argmax(axis=1))
