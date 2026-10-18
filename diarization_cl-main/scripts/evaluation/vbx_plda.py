# -*- coding: utf-8 -*-
"""两协方差 PLDA 的矩估计 + LDA 空间 —— 供 `VB_diarization` 当先验用。

为什么自训而不是用仓库里那份 ResNet101 的 `plda`：那份是 256 维 x-vector
空间的（`D:\\yinp\\AISHELL-4\\_v13_spike\\VBx\\models\\ResNet101_16kHz\\plda`），
对 192 维 CAM++ 无效。上次 spike 硬把 CAM++ 用**随机正交基**升维塞进去，
得到的 93.18% 是「失效变换 vs 什么都不做」，信息量为零。见 spec §1。

参数化与 `diarization_lib.kaldi_ivector_plda_scoring_dense` 在数学上等价：
那里从 (mu, tr, psi) 反推出 `wccn`（白化，使类内协方差 → I）与降序 `acvar`
（类间特征值），再用 `fea = (x − mu) · trᵀ[:, :R]`、`sV = sqrt(psi[:R])`。
本模块直接拟合出这两个量，不做无用的参数往返。
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as spl

DEFAULT_ALPHA = 0.05
DEFAULT_EPS = 1e-6
LDA_DIM_CAP = 32


@dataclass(frozen=True)
class Plda:
    mean: np.ndarray      # (D,)   全局均值
    wccn: np.ndarray      # (D,D)  白化：列 = Bc 在 Wc 度量下的特征向量，降序
    acvar: np.ndarray     # (D,)   类间特征值，降序、全正
    R: int                # LDA 空间维度
    n_classes: int
    n_windows: int
    alpha: float
    eps: float

    def to_lda_space(self, x):
        """(N,D) → (N,R)。与官方 `vbhmm.py` 的 `fea` 逐字同法。"""
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.wccn[:, :self.R]

    def speaker_prior(self):
        """eigenvoice 先验的 `sV`：`sqrt(acvar[:R])`（官方 `vbhmm.py`）。"""
        return np.sqrt(self.acvar[:self.R])

    def save(self, path):
        payload = {
            "schema": "v13-vbx-plda-v1",
            "mean": self.mean.tolist(),
            "wccn": self.wccn.tolist(),
            "acvar": self.acvar.tolist(),
            "R": int(self.R), "n_classes": int(self.n_classes),
            "n_windows": int(self.n_windows),
            "alpha": float(self.alpha), "eps": float(self.eps),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mean=np.array(d["mean"], dtype=np.float64),
                   wccn=np.array(d["wccn"], dtype=np.float64),
                   acvar=np.array(d["acvar"], dtype=np.float64),
                   R=int(d["R"]), n_classes=int(d["n_classes"]),
                   n_windows=int(d["n_windows"]),
                   alpha=float(d["alpha"]), eps=float(d["eps"]))


def _shrink(M, alpha, eps):
    """收缩到对角 + 单位阵脊。去掉了就可能奇异，而奇异仍会「跑通」。"""
    return (1.0 - alpha) * M + alpha * np.diag(np.diag(M)) + eps * np.eye(M.shape[0])


def _whiten_between(Wc, Bc):
    """解 Bc v = λ Wc v，返回 (wccn, acvar) —— 两者都按 λ 降序。

    numpy 的 `eigh` 返回**升序**。官方 `vbhmm.py` 用 `acvar[::-1]` /
    `wccn.T[::-1]` 反转；本函数返回的已经是降序版本，调用方不要再反转。
    """
    Wc = (Wc + Wc.T) / 2.0
    Bc = (Bc + Bc.T) / 2.0
    L = np.linalg.cholesky(Wc)                     # Wc = L Lᵀ
    Linv = spl.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    M = Linv @ Bc @ Linv.T                         # 对称
    acvar, S = np.linalg.eigh(M)                   # 升序
    acvar = acvar[::-1]
    S = S[:, ::-1]
    wccn = Linv.T @ S                              # 列 = 特征向量，降序
    # 数值上 acvar 可能出极小负数；先验要开根号 ⇒ 必须钳住并留下证据。
    acvar = np.maximum(acvar, 1e-12)
    # 特征向量的符号是任意的。**定死它**，否则换 BLAS 两次跑不出同一份产物。
    signs = np.sign(wccn[np.argmax(np.abs(wccn), axis=0),
                         np.arange(wccn.shape[1])])
    signs[signs == 0] = 1.0
    wccn = wccn * signs
    return wccn, acvar


def fit_plda(x, classes, alpha=DEFAULT_ALPHA, eps=DEFAULT_EPS, lda_dim=None):
    """矩估计两协方差 PLDA。

    `classes` 是**每个窗的类标识**，必须已经是 `f"{场次}:{标签}"` 的形态
    （spec §2.2）。本函数不替调用方拼 —— 拼错了在这里看不出来，
    只有类数会变小，而那正是要防的失败。
    """
    x = np.asarray(x, dtype=np.float64)
    classes = list(classes)
    if len(classes) != len(x):
        raise ValueError(f"classes 与 x 长度不符：{len(classes)} != {len(x)}")
    names = sorted(set(classes))
    if len(names) < 2:
        raise ValueError(
            f"至少需要 2 个类才能拟合 PLDA，实际 {len(names)} 个"
            f"（若这里是 1，先检查类标识是不是忘了带场次前缀）")

    mean = x.mean(axis=0)
    D = x.shape[1]
    Wc = np.zeros((D, D))
    Bc = np.zeros((D, D))
    within_n = 0
    for name in names:
        rows = x[[i for i, c in enumerate(classes) if c == name]]
        if len(rows) == 0:
            continue
        cm = rows.mean(axis=0)
        d = rows - cm
        Wc += d.T @ d
        within_n += len(rows)
        Bc += len(rows) * np.outer(cm - mean, cm - mean)
    Wc /= max(within_n, 1)
    Bc /= max(len(x), 1)

    wccn, acvar = _whiten_between(_shrink(Wc, alpha, eps), _shrink(Bc, alpha, eps))
    R = min(LDA_DIM_CAP, len(names) - 1) if lda_dim is None else int(lda_dim)
    if R < 1 or R > D:
        raise ValueError(f"lda_dim={R} 越界：必须在 1..{D} 之间")
    return Plda(mean=mean, wccn=wccn, acvar=acvar, R=R,
                n_classes=len(names), n_windows=len(x),
                alpha=float(alpha), eps=float(eps))
