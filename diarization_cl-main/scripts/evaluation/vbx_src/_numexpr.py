# -*- coding: utf-8 -*-
"""`numexpr` 的替身：只用 numpy 实现 `VB_diarization.py` 里的 numexpr 入口。

为什么不是 `pip install numexpr`：spec「不引入任何新依赖」。热路径
（`VB_diarization` → `forward_backward` 的 `:287`、`:290`）走的是官方**纯 numpy 的
`logsumexp`**，本垫片只服务 `logsumexp_ne` / `exp_ne` 这两个 numexpr 加速版。
`evaluate` 只认那两个版本实际发出的表达式模式，不是通用表达式求值器。
"""
import inspect

import numpy as np


def logsumexp(x, axis=0):
    """先减最大值再取 log。

    与官方 `VB_diarization.logsumexp` 的关系，如实说：**全有限值输入下结果一致**
    （见 `test_vbx_plda`）；**`±inf` / `nan` 输入下不保证同构**。
    已知差异只有一处：

    1. `errstate(invalid="ignore")` 的包裹范围：官方把 `xmax + np.log(...)` 也包在
       里面，本函数只包住 `np.sum(np.exp(...))`。

    （曾记过第二条"官方原地改写传入的 `x`"，**该说法有误、已删**：官方是
    `x = xmax + np.log(...)`，**重绑定局部名**，不修改入参；实测两边都对入参
    数组无副作用，故这不构成差异。）

    当前被测试钉住、且经验证成立的只有**全有限值**输入域（见
    `test_logsumexp_shim_matches_scipy_oracle`）。`±inf` 下两者的异同**未验证**，
    故这里不作声明。

    调用者说明（复审 MIN-1，保留原样）：本函数是 brief「Interfaces」点名要求的
    接口，但**成品内无任何调用者** —— 官方只用 `logsumexp_ne` / `exp_ne`，
    而垫片的 `evaluate` 走字面 numpy 路径、也不调它。它当前只被测试直接调用。
    """
    xmax = np.array(x).max(axis=axis)
    xmax_e = np.expand_dims(xmax, axis)
    with np.errstate(invalid="ignore"):
        total = np.sum(np.exp(x - xmax_e), axis=axis)
    out = xmax + np.log(total)
    infs = np.isinf(xmax)
    if np.ndim(out) > 0:
        out[infs] = xmax[infs]
    elif infs:
        out = xmax
    return out


def exp(x, out=None):
    return np.exp(x, out=out)


def evaluate(expr, local_dict=None, out=None, **kwargs):
    """`ne.evaluate` 的替身，**只**支持 `VB_diarization.py` 实际发出的那三个模式。

    刻意**不**实现通用表达式求值器：解析任意表达式是过度设计，而静默算错远比
    响亮报错危险。这里按表达式**字面**返回，三个模式各自独立：

    1. `"sum(exp(x - xmax_e), axis=<int>)"`（`logsumexp_ne` 第 236 行）→
       `np.sum(np.exp(x - xmax_e), axis=<int>)`，即**未取对数的和**。
       注意：不能改走本模块的 `logsumexp(x, axis)`。理由**不是**"它自带 `xmax`
       参数"（那是错的：它返回 `xmax + np.log(total)`，`+` 加的是 `xmax` 这个
       **向量**），而是**官方把这件事拆成了两行**：第 236 行只求和，第 237 行才补
       `xmax + log(...)`。若第一步就改调 `logsumexp`，它会先返回
       `xmax + log(Σexp(...))`，第二步再补一次 `xmax + log(·)` ⇒ **`xmax` 被加
       两次**（且多套一次 `log`）。
    2. `"xmax + log(x)"`（`logsumexp_ne` 第 237 行）→ `xmax + np.log(x)`，
       其中 `x` 是第 236 行的返回值、`xmax` 是该轴最大值。**不能**改走 `exp`：
       `exp(s)` 与 `xmax + log(s)` 相差一次对数/指数（实测轴 0 得 `[2.718,2.718,2.718]`
       而正确值是 `[0,1,2]`，轴 1 更会直接 `inf`）。
    3. `"exp(x)"`（`exp_ne` 第 247 行）→ `np.exp(x, out=out)`。官方那一行传的是 `out=None`。

    变量取值照 numexpr 的惯例：`ne.evaluate` 不接受数组参数，它从**调用者的局部
    作用域**里按名字取（只需 `x` 与 `xmax` 两个固定名字；`xmax_e` 由 `xmax` 现算，
    以免两处 `expand_dims` 的轴约定漂移）。其他表达式一律抛 `ValueError`。
    """
    ns = local_dict
    if ns is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        ns = dict(caller.f_locals) if caller is not None else {}
        del frame, caller
    x = ns.get("x")
    if x is None:
        raise ValueError(
            "vbx_src/_numexpr.evaluate 的替代物不做符号求值：表达式里的变量 x 必须"
            "能从调用方局部作用域取到（照 numexpr 惯例），收到 expr=%r" % (expr,)
        )
    if "axis=" in expr:
        axis = int(expr.rsplit("axis=", 1)[1].strip().rstrip(")"))
        xmax = ns.get("xmax")
        if xmax is None:      # 理论上到不了：官方先算 xmax 再发这条表达式
            xmax = np.array(x).max(axis=axis)
        xmax_e = np.expand_dims(xmax, axis=axis)
        with np.errstate(invalid="ignore"):
            return np.sum(np.exp(x - xmax_e), axis=axis)
    if "log(" in expr:        # "xmax + log(x)"
        return ns["xmax"] + np.log(x)
    if expr.strip().startswith("exp("):
        return exp(x, out=out)
    raise ValueError(
        "vbx_src/_numexpr.evaluate 只支持 VB_diarization.py 实际发出的三个模式"
        "（sum(exp(x - xmax_e), axis=<int>) / xmax + log(x) / exp(x)），"
        "收到: %r" % (expr,)
    )
