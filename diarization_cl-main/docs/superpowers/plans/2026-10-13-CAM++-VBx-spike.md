# CAM++ 上的真 VBx spike —— 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CAM++ 192 维嵌入上实现真正的 VBx（两协方差 PLDA + 贝叶斯 HMM），并在 dev 12 场上测它能否让句级归属错误率在 RTTM 与 VAD 两个门上各降 ≥ 1 点。

**Architecture:** 从 `train_L` 30 场建 RTTM 门窗缓存 → 用 `(场次, 标签)` 207 个类拟合两协方差 PLDA（矩估计，无新依赖）→ 得到 LDA 空间与 eigenvoice 先验 → 三臂（生产基线 / AHC 初始化 + VB / 平初始化 + VB）× 两门 × `Fb` 曲线，全部走**同一套**既有细化与句级指标 → 落一份可重算的产物并给出 `verdict`。

**Tech Stack:** Python 3.12 / numpy 1.26.4 / scipy 1.17.1 / tqdm 4.68.4 / pytest；复用 `funasr`+`torch` 的 CAM++ 与 FSMN VAD（仅建缓存时需要）。

**Spec:** `docs/superpowers/specs/2026-10-13-CAM++-VBx-spike-design.md`

---

## Global Constraints

以下每一条都是 spec 里的硬约束，**每个任务的要求都隐含包含本节**。取值逐字来自 spec。

- **不引入任何新依赖**。只用 numpy / scipy / tqdm / soundfile / funasr / torch / pytest。`kaldi_io`、`numexpr`、`h5py` 一律不装。
- **不改生产代码一行**。`streaming_asr/claude_a2_v11.py`、`claude_a2_v12.py`、`scripts/evaluation/unlocked_selector.py`、`web/` 全部只读。
- **不读 holdout**。任何脚本不得加载 holdout 场次的嵌入、不得对其评测、产物里不得有它的行。
- **不用 ResNet101 的 `plda` / `transform.h5`**（`D:\yinp\AISHELL-4\_v13_spike\VBx\models\`）。它们对 CAM++ 无效，正是上次失败的根因。
- **真值不得进入选 K**。VBx 的输入只有嵌入与超参；RTTM/TextGrid 只能进入评测与训练池的类标签。
- **§2.2 类 = `(场次, 标签)`**，实测 207 个类。**不是** 9 个裸标签。
- **判据（不可回改）**：RTTM 门 ≤ **0.0815**、VAD 门 ≤ **0.0619**（都相对生产选择器基线 9.150% / 7.190%）。**双门都要过**。
- **`Fb` 网格** `[1, 5, 10, 17, 30, 60]`，判据臂固定 `Fb=17`。`Fa=0.3`、`loopProb=0.99`、`maxIters=40`、`epsilon=1e-6`、`init_smoothing=5.0`、`threshold=-0.015`、`minDur=1` 全部固定为官方默认。
- **`SEED = 20261013`**，用于臂 2 的 Dirichlet 初值；必须写进产物。
- **失败不许静默**：任何场次拟合不足 / 抛异常 ⇒ 产物含带 `skip` 理由的存根行（坑 A6）。
- **`verdict` 取值域** `pass` / `falsified` / `invalid_design`；第三种必须附 `invalid_reason`。
- **不执行任何 git 写操作** —— 不 `add`、不 `commit`、不 `stash`。主检出压着 500+ 行与本线无关的未提交改动；spec §10.7 明令。散落在计划里的 `git commit` 步骤**一律跳过**，改为在任务报告里记录「已改动文件清单」。
- **测试红线**（坑 A5）：每条断言都要能回答「哪个合理回归会让它保持绿」，并在报告里附**真去改源码让它变红的命令与输出**。做不到就如实说做不到，不许写「已红验证」。
- **环境**（坑 W3/W5/W6）：数据缺失一律 `pytest.skip` 而非 error；pytest 必须在 `streaming_asr/` 下跑；带中文输出的命令必须 `PYTHONIOENCODING=utf-8`。
- **基线回归**：当前 `streaming_asr/` 全量测试 = **169 passed / 0 failed**（2026-10-13 实测）。任何任务结束后不得低于此数。

---

## 与 spec 的一处偏差（已决定，实施者照此做）

spec §4 写的是 `streaming_asr/vbx/` 并且 `VB_diarization.py` 用 `from . import _numexpr as ne`
**相对导入**。实测本仓库 `streaming_asr/`、`scripts/`、`scripts/evaluation/`
**都没有 `__init__.py`**（全是命名空间包），相对导入会直接失败。

⇒ **改为** `scripts/evaluation/vbx_src/`，与仓库既有惯例一致（`probe_stability_selector.py`
的注释自述：「不写 `__init__.py`…按仓库既有惯例」）。`vbx_backend.py` 用
`sys.path.insert` 把这个目录加进去后 `import VB_diarization`。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `scripts/evaluation/vbx_src/_numexpr.py` | **新建**。`logsumexp` / `exp` 包成 numpy，替掉 `numexpr` 依赖 |
| `scripts/evaluation/vbx_src/VB_diarization.py` | **新建**。BUT 官方算法体逐行复制，只改 2 行 import |
| `scripts/evaluation/vbx_src/__init__.py` | **不建**。与仓库其余目录一致，靠 `sys.path` |
| `scripts/evaluation/vbx_plda.py` | **新建**。两协方差 PLDA 矩估计 + LDA 空间 + `save`/`load` |
| `scripts/evaluation/vbx_backend.py` | **新建**。三臂的标签接口 + 确定性设置 + 引擎选择 |
| `scripts/evaluation/build_train_vbx_cache.py` | **新建**。建 `train_L` 的 `w2_h1.npz` |
| `scripts/evaluation/run_campplus_vbx.py` | **新建**。唯一入口：池 → 拟合 → 臂 × 门 → 评测 → 产物 |
| `streaming_asr/tests/test_vbx_plda.py` | **新建**。纯单元测试（任务 1/2/3 的断言） |
| `streaming_asr/tests/test_vbx_integration.py` | **新建**。需要数据/模型的集成断言（任务 5/6/7） |
| `artifacts/v13_campplus_vbx_dev.json` | **产物**。schema 见 spec §7 |
| `artifacts/v13_vbx_plda_trainL.json` | **产物**。PLDA 参数（含 `acvar` 全量，供复算）|
| `docs/HANDOFF_1013_后端对照spike.md`、`artifacts/v13_交付合并表.md` | **改**。§G 之后加一节；spec §4.3 被推翻的那句原地标注 |

---

## Task 1: 落 VBx 引擎与 `_numexpr` 垫片

**Files:**
- Create: `scripts/evaluation/vbx_src/_numexpr.py`
- Create: `scripts/evaluation/vbx_src/VB_diarization.py`
- Create: `streaming_asr/tests/test_vbx_plda.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `VB_diarization.VB_diarization(X, m, iE, V, pi=None, gamma=None, maxSpeakers=10, maxIters=10, epsilon=1e-4, loopProb=0.99, alphaQInit=1.0, ref=None, plot=False, minDur=1, Fa=1.0, Fb=1.0) -> (gamma, pi, Li)`
  - `VB_diarization.ne.logsumexp(x, axis=0)`、`VB_diarization.ne.exp(x, out=None)`

- [ ] **Step 1: 先写失败的测试**

创建 `streaming_asr/tests/test_vbx_plda.py`：

```python
# -*- coding: utf-8 -*-
"""锁住 VBx 引擎接入与本仓库的 PLDA 拟合。

每条断言都要能回答「哪个合理回归会让它保持绿」——答不出来就是没覆盖（坑 A5）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vbx_backend as VB  # noqa: E402


def test_numexpr_shim_is_ours_not_the_real_package():
    """哨兵：`VB_diarization.ne` 必须是我们的垫片。

    合理回归：有人 `pip install numexpr` 之后垫片被真包遮住 —— 那时
    `ne.__name__` 变成 `numexpr`，本断言红。这不是洁癖：真 numexpr 会把
    `logsumexp_ne` 变成 C 实现，两套实现在 `-inf` 输入上行为不同，
    而我们的产物要求可复算。
    """
    import VB_diarization as VD
    assert VD.ne.__name__ == "_numexpr"
    assert VD.ne.logsumexp.__module__ == "_numexpr"


def test_logsumexp_matches_the_plain_numpy_version():
    """垫片的 logsumexp 与 `VB_diarization.logsumexp` 逐位一致。

    合理回归：有人把垫片写成 `np.log(np.sum(np.exp(x)))` —— 溢出后
    大数输入返回 `inf`，与既有实现不符，断言红。
    """
    import VB_diarization as VD
    x = np.array([[0.0, 1.0, 2.0], [-1000.0, -1001.0, -999.0]])
    got = VD.ne.logsumexp(x, axis=1)
    want = VD.logsumexp(x, axis=1)
    assert np.allclose(got, want)
    assert np.isfinite(got).all()
```

- [ ] **Step 2: 跑测试确认失败**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'vbx_backend'`

- [ ] **Step 3: 写 `_numexpr.py`**

创建 `scripts/evaluation/vbx_src/_numexpr.py`：

```python
# -*- coding: utf-8 -*-
"""`numexpr` 的替身：只用 numpy 实现 `VB_diarization.py` 需要的那两个函数。

为什么不是 `pip install numexpr`：spec「不引入任何新依赖」。而且这两个函数在
`VB_diarization.py` 里只被 `logsumexp_ne` / `exp_ne` 用到，而真正跑在
`forward_backward`（`:287`、`:290`）里的是**纯 numpy 的 `logsumexp`** ——
垫片的语义只要与它一致就够了，见 `test_vbx_plda.test_logsumexp_matches_...`。
"""
import numpy as np


def logsumexp(x, axis=0):
    """与 `VB_diarization.logsumexp` 同法：先减最大值再取 log。"""
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
```

- [ ] **Step 4: 逐行复制官方 `VB_diarization.py` 并只改 2 行 import**

```powershell
Copy-Item "D:\yinp\AISHELL-4\sd\VBx\VB_diarization.py" `
          "C:\Users\EDY\Desktop\new\scripts\evaluation\vbx_src\VB_diarization.py"
```

然后在复制件里**只改这两行**（其余 292 行逐字不动，包括版权头）：

| 原行 | 改成 |
|---|---|
| `from kaldi_io import open_or_fd` | **整行删除**（`open_or_fd` 在文件内从未被调用）|
| `import numexpr as ne # the dependency on this modul can be avoided by replacing` | `from _numexpr import logsumexp as _shim_logsumexp  # noqa: F401  (垫片见 _numexpr.py)`<br>`import _numexpr as ne  # noqa: E402  替掉 numexpr 依赖` |

⚠️ 保留注释里原本跟着 `import numexpr` 的续行 `# logsumexp_ne and exp_ne with logsumexp and np.exp`
（它是注释，不影响语法），但**不要**动 `logsumexp_ne` / `exp_ne` 的函数体。

- [ ] **Step 5: 加 `vbx_backend.py` 的 import 垫（先只放 import，函数体在 Task 5 补）**

创建 `scripts/evaluation/vbx_backend.py`：

```python
# -*- coding: utf-8 -*-
"""VBx 后端：把 BUT 的 `VB_diarization` 接到本仓库的窗嵌入上。

**这里不实现算法。** 算法体是 `vbx_src/VB_diarization.py`（BUT 官方逐行复制），
本文件只负责：把窗嵌入变成它要的 (`fea`, `sm`, `iE`, `V`)，把它的输出变回
本仓库的硬标签约定，以及把初值定死以保证可复算。
"""
import sys
from pathlib import Path

import numpy as np

_VBX_SRC = Path(__file__).resolve().parent / "vbx_src"
if str(_VBX_SRC) not in sys.path:
    sys.path.insert(0, str(_VBX_SRC))

import VB_diarization as VD  # noqa: E402

SEED = 20261013
```

- [ ] **Step 6: 跑测试确认通过**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: `2 passed`

- [ ] **Step 7: 红验证（必须做，并把命令与输出抄进报告）**

1. 把 `_numexpr.py` 的 `logsumexp` 函数体改成一行 `return np.log(np.sum(np.exp(x), axis=axis))`
   → 第 2 条断言应红（大负数输入下 `xmax` 处理缺失 → `-inf` 或溢出）。
2. 把 `VB_diarization.py` 的 `import _numexpr as ne` 改成 `import numpy as ne`
   → 第 1 条断言应红（`ne.__name__ == "numpy"`）。
3. **两条都还原**，再跑一次确认回到 `2 passed`。

- [ ] **Step 8: 记录改动（不提交）**

在任务报告里列出新增文件清单；**不执行任何 git 命令**（Global Constraints）。

---

## Task 2: 两协方差 PLDA 拟合 + LDA 空间

**Files:**
- Create: `scripts/evaluation/vbx_plda.py`
- Modify: `streaming_asr/tests/test_vbx_plda.py`（追加断言）

**Interfaces:**
- Consumes: Task 1 无依赖（本任务纯数值）
- Produces:
  - `vbx_plda.Plda` —— frozen dataclass，字段 `mean: (D,)`、`wccn: (D,D)`、`acvar: (D,)` 降序、`R: int`、`n_classes: int`、`n_windows: int`、`alpha: float`、`eps: float`
  - `vbx_plda.fit_plda(x: (N,D) float64, classes: list[str], alpha: float, eps: float, lda_dim: int|None) -> Plda`
  - `vbx_plda.Plda.to_lda_space(x: (N,D)) -> (N,R)` —— `(x − mean) @ wccn[:, :R]`
  - `vbx_plda.Plda.speaker_prior() -> (R,)` —— `sqrt(acvar[:R])`
  - `vbx_plda.Plda.save(path)` / `vbx_plda.Plda.load(path)`
  - `vbx_plda.DEFAULT_ALPHA = 0.05`、`vbx_plda.DEFAULT_EPS = 1e-6`

- [ ] **Step 1: 写失败的测试**

追加到 `streaming_asr/tests/test_vbx_plda.py`：

```python
import vbx_plda as VP  # noqa: E402


def _synthetic(n_classes=12, per_class=8, dim=6, seed=0):
    """合成两协方差数据：类均值本身也服从一个高斯。"""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=dim)
    class_mu = rng.normal(scale=0.5, size=(n_classes, dim))
    x, y = [], []
    for c in range(n_classes):
        x.append(class_mu[c] + rng.normal(scale=0.15, size=(per_class, dim)))
        y.extend([f"m{c // 4}:{c}"] * per_class)
    return np.asarray(np.concatenate(x), dtype=np.float64), y


def test_classes_are_meeting_scoped_not_bare_labels():
    """★ spec §2.2：类 = (场次, 标签)。

    合理回归：有人把类退回成裸标签。构造两场、每场各 2 个同名标签 '001-M'
    —— 裸标签会得到 2 类，正确的 (场次,标签) 得到 4 类。
    """
    x = np.random.default_rng(1).normal(size=(16, 4))
    classes = (["A:001-M"] * 4 + ["A:002-M"] * 4
               + ["B:001-M"] * 4 + ["B:002-M"] * 4)
    plda = VP.fit_plda(x, classes, alpha=0.1, eps=1e-6, lda_dim=2)
    assert plda.n_classes == 4

    def collapse(cls):
        return cls.split(":", 1)[1]

    collapsed = VP.fit_plda(x, [collapse(c) for c in classes],
                            alpha=0.1, eps=1e-6, lda_dim=2)
    assert collapsed.n_classes == 2, "裸标签口径下应当只有 2 类"


def test_covariance_round_trip_and_acvar_positive():
    """`wccnᵀ · diag(acvar) · wccn` 必须还原类间协方差；acvar 全正。

    合理回归：有人把 `eigh` 的结果按**升序**留下（numpy 默认升序）而不是
    反转为降序 —— `acvar` 会变成升序、`sqrt` 后先验完全错位，还原断言红。
    """
    rng = np.random.default_rng(3)
    Wc = rng.normal(size=(5, 5))
    Wc = Wc @ Wc.T + 0.3 * np.eye(5)
    Bc = rng.normal(size=(5, 5))
    Bc = Bc @ Bc.T + 0.1 * np.eye(5)
    wccn, acvar = VP._whiten_between(Wc, Bc)
    assert (acvar > 0).all()
    assert np.all(np.diff(acvar) <= 1e-12), "acvar 必须降序"
    # 还原
    assert np.allclose(wccn.T @ np.diag(acvar) @ wccn, Bc, atol=1e-8)
    # 白化：wccnᵀ Wc wccn == I
    assert np.allclose(wccn.T @ Wc @ wccn, np.eye(5), atol=1e-8)


def test_fit_rejects_too_few_classes():
    """少于 2 个类时必须抛，不许返回一个看似可用的模型。

    合理回归：有人把 `if len(unique) < 2: raise` 改成 `return None`
    —— 上层会拿到 None 再崩在别处，失败原因丢失（坑 A6 同族）。
    """
    x = np.random.default_rng(4).normal(size=(8, 4))
    with pytest.raises(ValueError, match="至少需要 2 个类"):
        VP.fit_plda(x, ["only"] * 8, alpha=0.1, eps=1e-6, lda_dim=2)


def test_lda_dim_default_is_min_32_and_classes_minus_one():
    """R = min(32, 类数 − 1)——spec §2.6。

    合理回归：有人把 32 改成 128（照抄官方 ResNet101 的 `lda_dim`）——
    192 维、207 类下 128 维的先验没有依据，断言红。
    """
    x, y = _synthetic(n_classes=12, per_class=8, dim=40)
    plda = VP.fit_plda(x, y, alpha=0.05, eps=1e-6, lda_dim=None)
    assert plda.R == min(32, plda.n_classes - 1) == 11


def test_lda_space_width_matches_prior_length():
    """`to_lda_space` 的列数必须等于 `speaker_prior` 的长度。

    合理回归：只截断 fea 不截断 acvar（或反之）—— VBx 会在
    `V.dot(iE).dot(V.T)` 上维度不匹配，断言红。
    """
    x, y = _synthetic(n_classes=12, per_class=8, dim=40)
    plda = VP.fit_plda(x, y, alpha=0.05, eps=1e-6, lda_dim=8)
    fea = plda.to_lda_space(x)
    assert fea.shape == (len(x), 8)
    assert plda.speaker_prior().shape == (8,)


def test_save_load_is_value_exact(tmp_path):
    """保存再读回必须逐位相同 —— 产物要求 PLDA 可复算。

    合理回归：有人用 `json.dump` 直接写 numpy 数组（会抛），或写成
    浮点截断格式 —— 读回后 `np.array_equal` 红。
    """
    x, y = _synthetic(n_classes=8, per_class=6, dim=10)
    plda = VP.fit_plda(x, y, alpha=0.05, eps=1e-6, lda_dim=4)
    path = tmp_path / "plda.json"
    plda.save(path)
    back = VP.Plda.load(path)
    assert np.array_equal(back.mean, plda.mean)
    assert np.array_equal(back.wccn, plda.wccn)
    assert np.array_equal(back.acvar, plda.acvar)
    assert back.R == plda.R and back.n_classes == plda.n_classes
```

- [ ] **Step 2: 跑测试确认失败**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'vbx_plda'`

- [ ] **Step 3: 写 `vbx_plda.py`**

创建 `scripts/evaluation/vbx_plda.py`：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: `8 passed`

- [ ] **Step 5: 红验证**

1. 把 `_whiten_between` 里的 `acvar[::-1]` 去掉（留升序）、`wccn.T[::-1]` 改成 `Linv.T @ S`
   → `test_covariance_round_trip_and_acvar_positive` 的「降序」与还原断言应红。
2. 把 `_shrink` 改成 `return M` → 在 `test_fit_rejects_too_few_classes` 之外
   另跑一次 `test_covariance_round_trip...`（合成数据不奇异，可能仍绿）
   —— **若绿，如实记录"该断言覆盖不到正则化"，不许写成已覆盖**。
3. 把 `signs` 那三行删掉 → 记录是否变红（同 BLAS 下可能仍绿，
   **如实记录**）。为验证符号确实被定死，改用：
   手工把 `wccn[:, 0] *= -1` 后调 `_whiten_between` 的同参数复算，
   比较两次 `np.array_equal` —— 这一步必须真跑并抄输出。
4. 全部还原，确认回到 `8 passed`。

---

## Task 3: 建 `train_L` 的 RTTM 门窗缓存

**Files:**
- Create: `scripts/evaluation/build_train_vbx_cache.py`
- Consumes: `D:\yinp\AISHELL-4\_v13_spike\build_cache.py`（只读复用，不修改）

**Interfaces:**
- Consumes: `build_cache.extract_meeting(audio, rate, segs, win_s, hop_s, gate, spk_model, min_vad=0.5)`、`build_cache.read_rttm(path)`、`build_cache.vote_split(start, end, segs)`、`build_cache.rttm_speech(segs, skip=10.0)`
- Produces: `D:\yinp\AISHELL-4\_v13_spike\cache\<场次>\w2_h1.npz`，键 `emb (n,192) float32`（L2 归一化）/ `times (n,) float32` / `speakers (n,) <U*` / `votes (n,) float32`，与 test 侧**同键同名**
- Produces: `scripts/evaluation/build_train_vbx_cache.py` 的
  `train_meetings(root: Path) -> list[str]`、`build(meeting, root, out, window, hop, models) -> dict`

- [ ] **Step 1: 写失败测试（本任务的可测部分：场次发现 + 键名契约）**

追加到 `streaming_asr/tests/test_vbx_plda.py`：

```python
import build_train_vbx_cache as BTC  # noqa: E402


def test_train_meetings_found_by_date_prefix(tmp_path):
    """train_L 的场次发现不能走 `v12_aishell4.meeting_group`。

    合理回归：有人图省事复用 `split_meetings(root, "dev")` —— 它内部对每个
    stem 调 `meeting_group()`，而 train_L 的 id 是 `20200706_L_R001S01C01`
    形态，会抛 `ValueError: invalid AISHELL-4 meeting id`，断言红。
    """
    grid = tmp_path / "TextGrid"
    grid.mkdir()
    for name in ("20200706_L_R001S01C01", "20200707_L_R002S08C01"):
        (grid / f"{name}.rttm").write_text("", encoding="utf-8")
    (grid / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert BTC.train_meetings(tmp_path) == [
        "20200706_L_R001S01C01", "20200707_L_R002S08C01"]


def test_cache_keys_match_the_test_side_contract(tmp_path, monkeypatch):
    """产物键名必须与 test 侧的 `w2_h1.npz` 完全一致。

    合理回归：有人漏写 `votes` —— `probe_pair_oracle.pure_windows` 读
    `data["votes"]` 会 KeyError，品控门失效；这里用键集合断言挡住。
    """
    npz = tmp_path / "w2_h1.npz"
    np.savez_compressed(npz, emb=np.zeros((2, 192), np.float32),
                        times=np.array([0.0, 1.0], np.float32),
                        speakers=np.array(["001-M", "002-F"]),
                        votes=np.array([1.0, 1.0], np.float32))
    data = np.load(npz, allow_pickle=False)
    assert set(data.files) == {"emb", "times", "speakers", "votes"}
    assert data["emb"].shape[1] == 192
```

- [ ] **Step 2: 跑测试确认失败**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'build_train_vbx_cache'`

- [ ] **Step 3: 写 `build_train_vbx_cache.py`**

创建 `scripts/evaluation/build_train_vbx_cache.py`：

```python
# -*- coding: utf-8 -*-
"""建 `train_L` 的 RTTM 门窗缓存（`w2_h1.npz`），供 VBx 的 PLDA 训练池用。

**为什么必须新建**：`_v13_spike/cache/` 下只有 test 的 20 场有 `w2_h1.npz`；
train_L 30 场一份都没有，而它是本 spike 唯一的 PLDA 训练来源（spec §2.1）。

**为什么不重写提取循环**：`build_cache.extract_meeting` 是 test 侧那份缓存的
生成者，逐位复用才能保证「训练池的窗」与「评测的窗」同法（坑 A7）。
本脚本只做三件它没做的事：按日期前缀发现场次、指定输出目录、按场落盘。

    $env:PYTHONIOENCODING="utf-8"
    python -u -B scripts/evaluation/build_train_vbx_cache.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPIKE = Path(r"D:\yinp\AISHELL-4\_v13_spike")
if str(SPIKE) not in sys.path:
    sys.path.insert(0, str(SPIKE))

import build_cache as BC  # noqa: E402

DEFAULT_ROOT = Path(r"D:\yinp\AISHELL-4\data\train_L")
DEFAULT_OUT = SPIKE / "cache"
TRAIN_PREFIX = "2020"          # train_L 的 id 以日期开头；test 的是 `L_R003...`


def train_meetings(root):
    """发现 train_L 的场次。

    **不要**用 `v12_aishell4.split_meetings`：它内部对每个 stem 调
    `meeting_group()`，其正则只认 `L_R003S01C02` 形态，遇到
    `20200706_L_R001S01C01` 会抛 ValueError。
    """
    return sorted(p.stem for p in (Path(root) / "TextGrid").glob("*.rttm")
                  if p.stem.startswith(TRAIN_PREFIX))


def build(meeting, root, out, window, hop, models, seconds=0.0):
    import soundfile as sf

    wav = Path(root) / "wav" / f"{meeting}.flac"
    segs = BC.read_rttm(Path(root) / "TextGrid" / f"{meeting}.rttm")
    stop = int(16000 * seconds) if seconds else -1
    audio, rate = sf.read(wav, dtype="float32", start=0, stop=stop, always_2d=True)
    audio = audio[:, 0]
    emb, times, speakers = BC.extract_meeting(
        audio, rate, segs, window, hop, "rttm", models["spk"])
    target = Path(out) / meeting
    target.mkdir(parents=True, exist_ok=True)
    tag = f"w{window:g}_h{hop:g}"
    np.savez_compressed(
        target / f"{tag}.npz", emb=emb, times=times,
        speakers=np.asarray(speakers).astype(str),
        votes=np.asarray([BC.vote_split(t, t + window, segs) for t in times],
                         dtype=np.float32))
    return {"n": int(len(emb)), "dim": int(emb.shape[1]) if len(emb) else 0,
            "gt_speakers": sorted({s["speaker"] for s in segs}),
            "duration": round(len(audio) / rate, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = 整场")
    ap.add_argument("--meetings", nargs="*", default=None)
    args = ap.parse_args()

    from web_server import get_models

    root, out = Path(args.root), Path(args.out)
    meetings = args.meetings or train_meetings(root)
    if not meetings:
        raise SystemExit(f"没找到 train_L 场次（检查 --root：{root}）")
    print(f"[train-cache] {len(meetings)} 场，win={args.window} hop={args.hop} "
          f"gate=rttm → {out}", flush=True)

    models = get_models(lambda m: None, need_stream=False, need_offline=True)
    index_path = out / "train_index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8")).get("meetings", {})

    for i, meeting in enumerate(meetings, 1):
        t0 = time.perf_counter()
        try:
            entry = build(meeting, root, out, args.window, args.hop, models,
                          args.seconds)
        except FileNotFoundError as exc:
            # 静默丢场是最贵的坑（A6）：少了几场和本来就几场长得一模一样。
            print(f"  [{i}/{len(meetings)}] {meeting} SKIP: {exc}", flush=True)
            index[meeting] = {"skip": f"{type(exc).__name__}: {exc}"}
            continue
        index[meeting] = entry
        print(f"  [{i}/{len(meetings)}] {meeting} n={entry['n']} "
              f"({entry['duration']:.0f}s, {time.perf_counter() - t0:.1f}s)",
              flush=True)
        # 每场落一次：30 场 × 20 小时音频，中途断了不该丢已完成的场次。
        index_path.write_text(json.dumps(
            {"schema": "v13-train-vbx-cache-v1", "root": str(root),
             "gate": "rttm", "skip_seconds": BC.SKIP_SECONDS,
             "window": args.window, "hop": args.hop,
             "meetings": index}, ensure_ascii=False, indent=2), encoding="utf-8")

    skipped = [m for m, v in index.items() if "skip" in v]
    print(f"\n写入 {index_path}")
    if skipped:
        print(f"⚠ 跳过 {len(skipped)} 场：{skipped}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: `10 passed`

- [ ] **Step 5: 红验证**

把 `train_meetings` 改成 `return sorted(p.stem for p in (Path(root)/"TextGrid").glob("*.rttm"))`
（去掉前缀过滤）→ 第 1 条断言应红（`notes.txt` 不是 rttm，实际不会红 —— 所以
**再加一个 `.rttm` 但非 train_L 命名的文件**到测试的 tmp 目录里，例如
`L_R003S01C02.rttm`，那时断言才会红）。**按这个加强版测试做**：先在 Step 1
的测试里补上第三个文件：

```python
    for name in ("20200706_L_R001S01C01", "20200707_L_R002S08C01"):
        (grid / f"{name}.rttm").write_text("", encoding="utf-8")
    (grid / "L_R003S01C02.rttm").write_text("", encoding="utf-8")   # ← 必须加
```

- [ ] **Step 6: 真跑建缓存（长跑，用后台任务）**

先确认显存够（spec §9 风险 3：RTX 5060 8 GB，实测占用 7.8 GB）：

```powershell
nvidia-smi --query-gpu=memory.total,memory.used --format=csv
```

**先跑 1 场验证通路，再跑全部**：

```powershell
cd C:\Users\EDY\Desktop\new
$env:PYTHONIOENCODING="utf-8"
python -u -B scripts/evaluation/build_train_vbx_cache.py --meetings 20200706_L_R001S01C01
```

Expected: `n=100..400` 之间、`dim=192`、无异常。

若显存不足报 `CUDA out of memory`：**不要改生产代码**，改用
`CUDA_VISIBLE_DEVICES=""` 强制 CPU（慢但能跑），并在任务报告里记下这一点。

然后后台跑全部 30 场：

```powershell
python -u -B scripts/evaluation/build_train_vbx_cache.py
```

跑完后核对：

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import numpy as np,glob; fs=glob.glob(r'D:\yinp\AISHELL-4\_v13_spike\cache\2020*\w2_h1.npz'); print(len(fs)); d=np.load(fs[0]); print({k:d[k].shape for k in d.files})"
```

Expected: `30` 与 `{'emb': (n, 192), 'times': (n,), 'speakers': (n,), 'votes': (n,)}`

⚠️ **这一步**不要**在 test 侧跑**：`build_cache` 的 `main()` 会用 DEV/HOLDOUT 组
重建 test 缓存，而 holdout 不许碰。只跑本脚本（它只认 `2020` 前缀）。

---

## Task 4: 复现基线（臂 0）—— 不过就停

**Files:**
- 无新增。复用 `scripts/evaluation/probe_stability_selector.py`（只读）
- Create: `artifacts/v13_vbx_baseline_gate_check.json`（本任务的核对记录）

**Interfaces:**
- Consumes: `probe_stability_selector` 的 CLI（`--split dev --gate rttm|vad --cap 10 --seconds 600 --output ...`）
- Produces: `artifacts/v13_vbx_baseline_gate_check.json`，字段
  `{"rttm": {"mean": float, "expected": 0.0915, "source": str}, "vad": {...}, "ok": bool}`

> **为什么单列一个任务**：spec §10.4 要求「臂 0 复现 9.15% / 7.19%，不一致即停」。
> 复现不了就说明评测链路已经漂了，后面 VBx 的数字全部无意义。
> 这一条必须是**独立的可拒绝关口**，不能混进大脚本里。

- [ ] **Step 1: 跑 RTTM 门的基线**

```powershell
cd C:\Users\EDY\Desktop\new
$env:PYTHONIOENCODING="utf-8"
python -u -B scripts/evaluation/probe_stability_selector.py --split dev --gate rttm --cap 10 --seconds 600 --output artifacts\_vbx_baseline_rttm.json
```

- [ ] **Step 2: 读出「生产 objective(λ=0.005)」的选择损失并核对**

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import json;d=json.load(open(r'artifacts\_vbx_baseline_rttm.json',encoding='utf-8'));print(d['selectors']['生产 objective(λ=0.005)'])"
```

Expected: `{'mean': 0.0915, 'loss': 0.0489, 'wins': 0, 'losses': 7, 'ties': 5, 'worst': [...]}`
—— **字段名是 `mean`，不是 `clean_attribution_error`**（实测该产物的 selector 字段为
`['mean','loss','wins','losses','ties','worst']`）。`mean` 必须落在 `0.0915 ± 0.0005`。
**若不在这个区间 ⇒ 停**，先把偏差原因查清（可能是 `cap` / `share` / `seconds` 口径变了），
**不许**带着漂了的基线往下走。

- [ ] **Step 3: 跑 VAD 门的基线并核对**

```powershell
python -u -B scripts/evaluation/probe_stability_selector.py --split dev --gate vad --cap 10 --seconds 600 --output artifacts\_vbx_baseline_vad.json
python -c "import json;d=json.load(open(r'artifacts\_vbx_baseline_vad.json',encoding='utf-8'));print(d['selectors']['生产 objective(λ=0.005)'])"
```

Expected: `mean` = `0.0719`（±0.0005）。

- [ ] **Step 4: 落核对记录**

```powershell
$env:PYTHONIOENCODING="utf-8"
python - <<'PY'
import json
from pathlib import Path
base = Path("artifacts")
out = {}
for gate, expected in (("rttm", 0.0915), ("vad", 0.0719)):
    d = json.loads((base / f"_vbx_baseline_{gate}.json").read_text(encoding="utf-8"))
    mean = d["selectors"]["生产 objective(λ=0.005)"]["mean"]
    out[gate] = {"mean": mean, "expected": expected, "source":
                 f"v13_selector_stability_diagnosis_{gate}.json 的生产 objective 行",
                 "ok": abs(mean - expected) <= 0.0005}
out["ok"] = all(v["ok"] for k, v in out.items() if k != "ok")
(base / "v13_vbx_baseline_gate_check.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
```

Expected: 两个 `ok` 都是 `true`，顶层 `ok: true`。
**若 `ok: false` ⇒ 本 spike 停下**，把偏差写进交接文档，不要继续 Task 5。

- [ ] **Step 5: 清掉中间文件**

`artifacts/_vbx_baseline_rttm.json` 与 `_vbx_baseline_vad.json` 是核对用的临时件，
保留 `v13_vbx_baseline_gate_check.json` 即可。**不要删**已有的
`v13_selector_stability_diagnosis_*.json`。

---

## Task 5: 三臂的 VBx 标签接口

**Files:**
- Modify: `scripts/evaluation/vbx_backend.py`（补函数体）
- Modify: `streaming_asr/tests/test_vbx_plda.py`（追加断言）

**Interfaces:**
- Consumes: `vbx_plda.Plda`（Task 2）
- Produces:
  - `vbx_backend.VBX_HYPER: dict` —— `{"Fa": 0.3, "loopProb": 0.99, "maxIters": 40, "epsilon": 1e-6, "init_smoothing": 5.0, "minDur": 1, "threshold": -0.015}`
  - `vbx_backend.FB_GRID = [1, 5, 10, 17, 30, 60]`、`vbx_backend.FB_CRITERION = 17`
  - `vbx_backend.arm_labels(emb, plda, labels_ahc, arm, fb, cap=10, seed=SEED) -> np.ndarray[int]`
    - `arm="vbx_ahc"`：用 `labels_ahc` 做初值，`maxSpeakers = len(unique(labels_ahc))`
    - `arm="vbx_flat"`：用 `seed` 生成 Dirichlet 初值，`maxSpeakers = cap`
    - 返回长度 `len(emb)` 的整数硬标签（`gamma.argmax(1)`，已按首见顺序重编号）
  - `vbx_backend.engine_name() -> str` —— `"numpy"` 或 `"numexpr"`，写进产物

- [ ] **Step 1: 写失败的测试**

追加到 `streaming_asr/tests/test_vbx_plda.py`：

```python
def test_flat_arm_is_reproducible_and_ahc_arm_is_not_ahc():
    """臂 2 两次逐位相同；臂 1 的输出不许等于输入的 AHC 标签。

    合理回归 A：有人把臂 2 的 gamma 传成 None —— `VB_diarization` 内部走
    无种子的 `np.random.gamma`，两次跑出两个数，第一条断言红。
    合理回归 B：有人让臂 1 直接 `return labels_ahc`（省掉 VB 那一步）——
    臂 1 退化成臂 0，第二条断言红。
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 8))
    # 造一个「AHC 把两团合并了」的初值：真结构是 2 团，初值只给 1 个 label
    x[:30] += 3.0
    labels_ahc = np.zeros(60, dtype=int)
    plda = VP.fit_plda(x, [f"m{i//6}:s{i%6}" for i in range(60)],
                       alpha=0.05, eps=1e-6, lda_dim=4)

    a = VB.arm_labels(x, plda, labels_ahc, "vbx_flat", fb=17.0, cap=4)
    b = VB.arm_labels(x, plda, labels_ahc, "vbx_flat", fb=17.0, cap=4)
    assert np.array_equal(a, b), "臂 2 必须可复算（固定 SEED）"

    c = VB.arm_labels(x, plda, labels_ahc, "vbx_ahc", fb=17.0, cap=4)
    assert not np.array_equal(c, labels_ahc), "臂 1 不许把 AHC 标签原样吐出"


def test_labels_are_relabelled_by_first_seen():
    """硬标签必须是 `0..k-1` 连续整数 —— 与 `v11._relabel_by_first_seen` 同约定。

    合理回归：有人直接 `gamma.argmax(1)` 返回而不重编号，标签会是
    `maxSpeakers` 空间里的稀疏下标（如 {0,3,7}），下游 `sizes` / 匈牙利映射
    都会算错，断言红。
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 6))
    x[:20] += 4.0
    plda = VP.fit_plda(x, [f"m{i//5}:s{i%5}" for i in range(40)],
                       alpha=0.05, eps=1e-6, lda_dim=3)
    labels = VB.arm_labels(x, plda, np.zeros(40, dtype=int), "vbx_flat",
                           fb=17.0, cap=5)
    uniq = sorted(set(int(v) for v in labels))
    assert uniq == list(range(len(uniq))), f"标签不连续：{uniq}"
```

- [ ] **Step 2: 跑测试确认失败**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q -k "flat_arm or relabelled"
```

Expected: FAIL —— `AttributeError: module 'vbx_backend' has no attribute 'arm_labels'`

- [ ] **Step 3: 补 `vbx_backend.py` 的函数体**

把 `scripts/evaluation/vbx_backend.py` 补成（保留 Task 1 已有的头与 `SEED`）：

```python
from scipy.special import softmax  # noqa: E402

from diarization_lib import AHC, cos_similarity, twoGMMcalib_lin  # noqa: E402

VBX_HYPER = {"Fa": 0.3, "loopProb": 0.99, "maxIters": 40, "epsilon": 1e-6,
             "init_smoothing": 5.0, "minDur": 1, "threshold": -0.015}
FB_GRID = [1, 5, 10, 17, 30, 60]
FB_CRITERION = 17


def engine_name():
    """产物要记：跑的是真 numexpr 还是我们的 numpy 垫片。"""
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
    """
    threshold = VBX_HYPER["threshold"] if threshold is None else threshold
    scr = cos_similarity(np.asarray(emb, dtype=np.float64))
    thr, _ = twoGMMcalib_lin(scr.ravel())
    return AHC(scr, thr + threshold)


def _dirichlet_init(n_frames, max_speakers, seed):
    """臂 2 的初值：**固定种子**的 Dirichlet 随机后验。

    官方的 `VB_diarization` 在 `gamma is None` 时自己走 `np.random.gamma`，
    **没有种子** —— 那样臂 2 跑两次两个数，产物不可复算。所以由调用方给。
    """
    rng = np.random.default_rng(seed)
    gamma = rng.gamma(1.0, size=(n_frames, max_speakers))
    return gamma / gamma.sum(axis=1, keepdims=True)


def arm_labels(emb, plda, labels_ahc, arm, fb, cap=10, seed=SEED):
    """跑一次 VBx，返回本仓库约定的硬标签（0..k-1，按首见顺序）。"""
    emb = np.asarray(emb, dtype=np.float64)
    fea = plda.to_lda_space(emb)
    R = fea.shape[1]
    sV = plda.speaker_prior()

    if arm == "vbx_ahc":
        labels_ahc = np.asarray(labels_ahc).reshape(-1)
        k0 = len(set(int(v) for v in labels_ahc))
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
```

⚠️ `arm_labels` 里 `labels_ahc` 若是**已重编号过**的生产标签，`onehot` 的列数
必须与之匹配。生产侧 `build_unlocked` 的候选标签已经是 `0..k-1`（走
`v11._relabel_by_first_seen`），所以 `k0 == labels_ahc.max()+1`。**加一行断言
把这个前提钉住**（下面 Step 3b）。

- [ ] **Step 3b: 在 `arm_labels` 里加前提断言**

在 `if arm == "vbx_ahc":` 分支的 `k0 = ...` 之后插入：

```python
        assert labels_ahc.min() >= 0 and labels_ahc.max() + 1 == k0, (
            "AHC 标签必须是 0..k-1 的连续整数；稀疏下标会让 onehot 的列对不上")
```

- [ ] **Step 4: 跑测试确认通过**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py -q
```

Expected: `12 passed`

- [ ] **Step 5: 红验证**

1. 把 `_dirichlet_init(len(fea), cap, seed)` 改成 `None` → 第一条断言应红
   （两次不等）。（**若刚好两次相等**，说明该合成数据下 VB 收敛到同一解
   —— 那就把断言改成比较 `gamma0`：直接断言
   `not np.array_equal(_dirichlet_init(5, 3, SEED), _dirichlet_init(5, 3, None))`
   不可行（需要 seed），改为断言两次 `arm_labels(..., "vbx_flat", ...)`
   的 `skip` 计数……**若两条路都做不到，如实记录"该断言在此数据下红不了"**，
   并改测 `_dirichlet_init` 本身：`np.array_equal(_dirichlet_init(5,3,SEED),
   _dirichlet_init(5,3,SEED)) is True` 且 `_dirichlet_init(5,3,1)` 与
   `_dirichlet_init(5,3,2)` 不等 —— 后者是能红的。）
2. 把臂 1 分支改成 `return _relabel_by_first_seen(labels_ahc)` → 第二条断言应红。
3. 把 `_relabel_by_first_seen(gamma.argmax(axis=1))` 改成 `gamma.argmax(axis=1)`
   → `test_labels_are_relabelled_by_first_seen` 应红。
4. 全部还原，确认回到 `12 passed`。

---

## Task 6: VBx 运行器（唯一入口，出产物）

**Files:**
- Create: `scripts/evaluation/run_campplus_vbx.py`
- Create: `artifacts/v13_campplus_vbx_dev.json`（产物）
- Create: `artifacts/v13_vbx_plda_trainL.json`（PLDA 参数）

**Interfaces:**
- Consumes: `vbx_plda.fit_plda` / `Plda`（Task 2）· `vbx_backend.arm_labels` / `FB_GRID` / `FB_CRITERION` / `engine_name`（Task 5）· `build_train_vbx_cache.train_meetings`（Task 3）· `probe_backend_sweep.{refine_labels, sentence_metrics}` · `probe_gap_decomp.{cached_windows, gate_windows, rttm_speech}` · `probe_stability_selector.{vad_segments, raw_candidates}` · `probe_pair_oracle.pure_windows` · `probe_path_equiv.cached_stage` · `unlocked_selector.{CAP, MIN_WIN, pick_unlocked}` · `probe_ab_dev.{load_sentences, _win}`
- Produces:
  - `run_campplus_vbx.build_pool(cache, meetings, window, hop) -> (x, classes, stats)`
  - `run_campplus_vbx.meeting_arms(meeting, cache, root, gate, plda, fb_grid, cap, seconds, share) -> (rows, skip)`
  - `run_campplus_vbx.verdict(per_gate) -> dict`
  - CLI: `python -u -B scripts/evaluation/run_campplus_vbx.py --split dev --output artifacts/v13_campplus_vbx_dev.json`

- [ ] **Step 1: 写失败测试（verdict 与 skip 存根是纯函数，先测它们）**

创建 `streaming_asr/tests/test_vbx_integration.py`：

```python
# -*- coding: utf-8 -*-
"""VBx 运行器的可离线测部分：verdict 判定、skip 存根、holdout 拒绝。

真正跑模型的部分在 `test_vbx_integration.py::test_arm0_matches_published_baseline`
—— 它需要数据与缓存，缺了就 skip（坑 W3）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campplus_vbx as RV  # noqa: E402


def test_verdict_requires_both_gates():
    """双门都要过。只过一个 ⇒ falsified（spec §3.1）。

    合理回归：有人把判据写成 `any(...)`（两个门过一个就行）—— 第二个断言红。
    这正是 spec 明确否掉的那条（VAD 门 7.19% 已在 ≤8% 预算内，
    单门会让判据在实验开始前自我满足）。
    """
    rttm_pass = {"mean": 0.0800}     # ≤ 0.0815
    rttm_fail = {"mean": 0.0820}
    vad_pass = {"mean": 0.0600}      # ≤ 0.0619
    vad_fail = {"mean": 0.0620}

    assert RV.verdict({"rttm": rttm_pass, "vad": vad_pass})["verdict"] == "pass"
    assert RV.verdict({"rttm": rttm_fail, "vad": vad_pass})["verdict"] == "falsified"
    assert RV.verdict({"rttm": rttm_pass, "vad": vad_fail})["verdict"] == "falsified"


def test_verdict_reports_invalid_design_separately():
    """设计缺陷走 `invalid_design`，且必须带理由（spec §3.2、§7）。

    合理回归：有人把拟合失败也算成 `falsified` —— 那就重演了本线已犯过的错
    （实验缺陷被记成方向被否），第二条断言红。
    """
    out = RV.verdict({}, invalid_reason="train_L 缓存只有 2 场，PLDA 无法拟合")
    assert out["verdict"] == "invalid_design"
    assert out["invalid_reason"]
    with pytest.raises(ValueError):
        RV.verdict({}, invalid_reason=None)   # 没理由不许叫 invalid_design


def test_skip_stub_carries_a_reason():
    """丢场必须带理由（坑 A6）。合理回归：有人改成 `continue` —— 断言红。"""
    stub = RV.skip_stub("M_R003S04C01", "rttm", "ValueError: 类数不足")
    assert stub["meeting"] == "M_R003S04C01"
    assert stub["skip"] and "类数不足" in stub["skip"]
    assert "attribution_error" not in stub, "存根行不许冒充成测量行"


def test_holdout_split_is_refused():
    """`--split holdout` 必须被拒（spec §2.1）。

    合理回归：有人把 choices 放开成 dev/holdout —— 断言红。
    """
    with pytest.raises(SystemExit):
        RV.parse_args(["--split", "holdout"])
```

- [ ] **Step 2: 跑测试确认失败**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_integration.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'run_campplus_vbx'`

- [ ] **Step 3: 写 `run_campplus_vbx.py`**

创建 `scripts/evaluation/run_campplus_vbx.py`。**关键口径逐条都有注释**，照抄：

```python
# -*- coding: utf-8 -*-
"""CAM++ 上的真 VBx spike：三臂 × 两门 × Fb 曲线，出产物并给 verdict。

判据与全部约束见 `docs/superpowers/specs/2026-10-13-CAM++-VBx-spike-design.md`。
**判据不可回改。**

    $env:PYTHONIOENCODING="utf-8"
    python -u -B scripts/evaluation/run_campplus_vbx.py --split dev `
        --output artifacts/v13_campplus_vbx_dev.json
"""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402
from der import read_rttm  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402
import probe_ab_dev as AB  # noqa: E402
import probe_backend_sweep as BS  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
import vbx_backend as VB  # noqa: E402
import vbx_plda as VP  # noqa: E402
from probe_gap_decomp import (  # noqa: E402
    cached_windows, gate_windows, rttm_speech, window_reference)
from probe_pair_oracle import pure_windows  # noqa: E402
from probe_stability_selector import raw_candidates, vad_segments  # noqa: E402
from unlocked_selector import CAP, MIN_WIN, pick_unlocked  # noqa: E402

SPIKE_CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
TRAIN_INDEX = SPIKE_CACHE / "train_index.json"
PASS_LINE = {"rttm": 0.0815, "vad": 0.0619}
BASELINE = {"rttm": 0.0915, "vad": 0.0719}


def skip_stub(meeting, gate, reason):
    """带理由的存根行 —— 少了几场和本来就几场不能长得一样（坑 A6）。"""
    return {"meeting": meeting, "gate": gate, "skip": str(reason), "rows": []}


def verdict(per_gate, invalid_reason=None):
    """双门都要过（spec §3.1）。设计缺陷单列，不许算 falsified（§3.2）。"""
    if invalid_reason is not None:
        return {"verdict": "invalid_design", "invalid_reason": str(invalid_reason),
                "rttm_pass": None, "vad_pass": None}
    if not per_gate:
        raise ValueError("没有任何门的读数：要么是 invalid_design，要么是没跑完")
    flags = {gate: (row["mean"] <= PASS_LINE[gate]) for gate, row in per_gate.items()}
    return {"verdict": "pass" if all(flags.values()) else "falsified",
            "rttm_pass": flags.get("rttm"), "vad_pass": flags.get("vad"),
            "pass_line": PASS_LINE, "baseline": BASELINE}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev",), default="dev",
                    help="只允许 dev —— holdout 是封存集（spec §2.1）")
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--train-root", default=r"D:\yinp\AISHELL-4\data\train_L")
    ap.add_argument("--cache", default=str(SPIKE_CACHE))
    ap.add_argument("--gate", nargs="*", default=["rttm", "vad"])
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--share", type=float, default=0.05)
    ap.add_argument("--plda-shrink", type=float, default=VP.DEFAULT_ALPHA)
    ap.add_argument("--plda-eps", type=float, default=VP.DEFAULT_EPS)
    ap.add_argument("--lda-dim", type=int, default=None)
    ap.add_argument("--meetings", nargs="*", default=None)
    ap.add_argument("--output", default=str(ROOT / "artifacts" / "v13_campplus_vbx_dev.json"))
    ap.add_argument("--plda-output", default=str(ROOT / "artifacts" / "v13_vbx_plda_trainL.json"))
    return ap.parse_args(argv)


def build_pool(cache, meetings, window, hop):
    """PLDA 训练池：train_L 的纯窗，类 = (场次, 标签)（spec §2.2）。"""
    xs, classes = [], []
    per_meeting, skipped = {}, []
    for meeting in meetings:
        try:
            data = np.load(Path(cache) / meeting / f"w{window:g}_h{hop:g}.npz",
                           allow_pickle=False)
        except FileNotFoundError as exc:
            skipped.append(f"{meeting}: {exc}")
            continue
        emb = data["emb"]
        times = np.asarray(data["times"], dtype=float)
        speakers = data["speakers"].astype(str)
        votes = np.asarray(data["votes"], dtype=float)
        pe, _pt, ps = pure_windows(emb, times, speakers, votes)
        if len(pe) == 0:
            skipped.append(f"{meeting}: 无纯窗（votes >= 1.0）")
            continue
        xs.append(pe)
        classes.extend([f"{meeting}:{name}" for name in ps])
        per_meeting[meeting] = int(len(pe))
    if not xs:
        return None, None, {"skipped": skipped, "per_meeting": {}}
    x = np.concatenate(xs).astype(np.float64)
    return x, classes, {"skipped": skipped, "per_meeting": per_meeting,
                        "counts": dict(Counter(classes))}


def meeting_arms(meeting, cache, root, gate, plda, fb_grid, cap, seconds, share):
    """一场 × 一个门 × 全部臂与 Fb。返回 (rows, skip)。"""
    rE, rtimes = PE.cached_stage(meeting, "refine")
    reference = read_rttm(Path(root) / "TextGrid" / f"{meeting}.rttm")
    limit = seconds if seconds > 0 else max((s["end"] for s in reference), default=0.0)
    reference = [{"start": s["start"], "end": min(s["end"], limit),
                  "speaker": s["speaker"]} for s in reference if s["start"] < limit]
    by_speaker = {}
    for s in reference:
        by_speaker[s["speaker"]] = by_speaker.get(s["speaker"], 0.0) + s["end"] - s["start"]
    grand = sum(by_speaker.values()) or 1.0
    effective = sorted(n for n, v in by_speaker.items() if v / grand >= share)

    emb_all, times_all = cached_windows(Path(cache), meeting, v12.WIN_S, v12.HOP_S)
    speech = (rttm_speech(reference, seconds=limit) if gate == "rttm"
              else vad_segments(meeting, limit))
    emb, times = gate_windows(emb_all, times_all, speech, v12.WIN_S,
                              v11.MIN_VAD_FRACTION)
    sents = AB.load_sentences(meeting)
    if len(emb) < 2 or not sents:
        return [], "窗数或句数不足"

    rows = []

    def measure(tag, labels):
        fclusters, ftimes, level = BS.refine_labels(
            np.asarray(labels), emb, times, rE, rtimes, True)
        fwin = v12.WIN_S if level == "none" else AB._win(level)
        clean = BS.sentence_metrics(sents, fclusters, ftimes, fwin)["clean"]
        k = len(set(int(v) for v in fclusters))
        return {"meeting": meeting, "gate": gate, "arm": tag,
                "n_windows": int(len(emb)), "true_k_effective": len(effective),
                "k": int(k), "level": level,
                "per_sentence": clean["per_sentence"], "wrong": clean["wrong"],
                "unmapped": clean["unmapped"], "abstain": clean["abstain"],
                "attribution_error": clean["attribution_error"],
                "floor_pass": bool(clean["per_sentence"] + clean["abstain"] >= 0.80)}

    # ── 臂 0：生产选择器（复现基线，spec §10.4）──────────────────────────
    cands = raw_candidates(emb, cap)
    base = pick_unlocked(cands, mode="objective")
    rows.append(dict(measure("baseline", np.asarray(base["labels"])),
                     fb=None, k_baseline=int(base["k"]), k_vbx=None))

    # ── 臂 1 / 臂 2：VBx，每个 Fb 一行 ──────────────────────────────────
    for arm in ("vbx_ahc", "vbx_flat"):
        for fb in fb_grid:
            try:
                labels = VB.arm_labels(emb, plda, np.asarray(base["labels"]),
                                       arm, fb, cap=cap)
            except Exception as exc:      # noqa: BLE001 — 失败要留痕，不许静默
                rows.append({"meeting": meeting, "gate": gate, "arm": arm,
                             "fb": float(fb), "skip": f"{type(exc).__name__}: {exc}"})
                continue
            row = measure(arm, labels)
            row.update({"fb": float(fb), "k_baseline": int(base["k"]),
                        "k_vbx": int(row["k"])})
            rows.append(row)
    return rows, None


def _git_head():
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
        return head, dirty
    except Exception:                      # noqa: BLE001
        return None, None


def main(argv=None):
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

    # ── 池与 PLDA ──────────────────────────────────────────────────────
    if not TRAIN_INDEX.exists():
        raise SystemExit(f"缺少 {TRAIN_INDEX} —— 先跑 build_train_vbx_cache.py")
    train_meetings = sorted(json.loads(TRAIN_INDEX.read_text(encoding="utf-8"))["meetings"])
    x, classes, pool_info = build_pool(args.cache, train_meetings,
                                       v12.WIN_S, v12.HOP_S)
    print(f"[pool] {len(train_meetings)} 场请求，"
          f"{len(pool_info['per_meeting'])} 场有纯窗，"
          f"{len(set(classes)) if classes else 0} 个类，"
          f"{0 if x is None else len(x)} 个窗", flush=True)
    if pool_info["skipped"]:
        print(f"[pool] ⚠ 跳过 {len(pool_info['skipped'])} 场", flush=True)

    if x is None or len(set(classes)) < 2:
        payload = {"schema": "v13-campplus-vbx-dev-v1",
                   "criterion": verdict({}, invalid_reason=(
                       f"PLDA 训练池不可用：类数 "
                       f"{0 if not classes else len(set(classes))}，"
                       f"纯窗 {0 if x is None else len(x)}")),
                   "plda": {"pool": "train_L", "per_meeting": pool_info["per_meeting"],
                            "skipped": pool_info["skipped"]}}
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"[verdict] invalid_design → {args.output}")
        return

    t0 = time.perf_counter()
    plda = VP.fit_plda(x, classes, alpha=args.plda_shrink, eps=args.plda_eps,
                       lda_dim=args.lda_dim)
    plda.save(args.plda_output)
    print(f"[plda] {plda.n_classes} 类 / {plda.n_windows} 窗 / R={plda.R} "
          f"({time.perf_counter() - t0:.1f}s) acvar "
          f"{plda.acvar.min():.3g}..{plda.acvar.max():.3g}", flush=True)

    head, dirty = _git_head()
    meetings = args.meetings or split_meetings(Path(args.root), args.split)
    payload = {
        "schema": "v13-campplus-vbx-dev-v1",
        "generated_by": "scripts/evaluation/run_campplus_vbx.py",
        "code": {"head": head, "dirty": dirty},
        "truth_used": False,
        "engine": VB.engine_name(),
        "seed": VB.SEED,
        "plda": {"pool": "train_L", "n_meetings": len(pool_info["per_meeting"]),
                 "n_classes": plda.n_classes, "n_windows": plda.n_windows,
                 "R": plda.R, "alpha": plda.alpha, "eps": plda.eps,
                 "acvar_min": float(plda.acvar.min()),
                 "acvar_max": float(plda.acvar.max()),
                 "per_meeting": pool_info["per_meeting"],
                 "skipped": pool_info["skipped"]},
        "hyper": dict(VB.VBX_HYPER, caps=[args.cap], fb_grid=VB.FB_GRID,
                      fb_criterion=VB.FB_CRITERION),
        "gates": {}, "criterion": None, "skipped": [],
    }

    for gate in args.gate:
        rows, per_gate = [], {}
        for i, meeting in enumerate(meetings, 1):
            try:
                got, skip = meeting_arms(meeting, args.cache, args.root, gate,
                                         plda, VB.FB_GRID, args.cap,
                                         args.seconds, args.share)
            except FileNotFoundError as exc:
                got, skip = [], f"{type(exc).__name__}: {exc}"
            if skip:
                payload["skipped"].append(f"{gate}/{meeting}: {skip}")
                got = [skip_stub(meeting, gate, skip)]
            rows.extend(got)
            errs = [r["attribution_error"] for r in got
                    if r.get("arm") == "baseline" and "attribution_error" in r]
            print(f"  [{gate} {i}/{len(meetings)}] {meeting} "
                  f"baseline={'—' if not errs else f'{errs[0] * 100:.2f}%'}",
                  flush=True)

        def _mean(arm, fb):
            vals = [r["attribution_error"] for r in rows
                    if r.get("arm") == arm and r.get("fb") == fb
                    and "attribution_error" in r]
            return float(np.mean(vals)) if vals else None

        base_mean = _mean("baseline", None)
        by_fb = {arm: {str(fb): {"mean_attribution_error": _mean(arm, float(fb))}
                       for fb in VB.FB_GRID}
                 for arm in ("vbx_ahc", "vbx_flat")}
        crit = _mean("vbx_flat", float(VB.FB_CRITERION))
        per_gate = {"baseline_reproduced": {"mean_attribution_error": base_mean,
                                            "n_meetings": len(meetings)},
                    "arms": by_fb,
                    "criterion_arm": {"arm": "vbx_flat",
                                      "fb": VB.FB_CRITERION, "mean": crit},
                    "expected_baseline": BASELINE[gate]}
        if base_mean is not None and abs(base_mean - BASELINE[gate]) > 0.0005:
            payload["criterion"] = verdict({}, invalid_reason=(
                f"{gate} 门基线复现失败：跑出 {base_mean:.4f}，"
                f"published {BASELINE[gate]:.4f} —— 评测链路已漂，后面数字无意义"))
            print(f"[STOP] {gate} 门基线不复现 ⇒ invalid_design")
            payload["gates"][gate] = per_gate
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
            return
        payload["gates"][gate] = dict(per_gate, rows=rows)

    # 判据只认 vbx_flat @ Fb=17（spec §3.3）
    means = {gate: {"mean": payload["gates"][gate]["criterion_arm"]["mean"]}
             for gate in args.gate}
    if any(v["mean"] is None for v in means.values()):
        payload["criterion"] = verdict({}, invalid_reason="判据臂没有出数（见 skipped）")
    else:
        payload["criterion"] = verdict(means)

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"\n[verdict] {payload['criterion']}")
    print(f"写入 {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_integration.py -q
```

Expected: `4 passed`

- [ ] **Step 5: 红验证**

1. `verdict` 里 `all(flags.values())` 改成 `any(flags.values())` → 第 1 条断言应红。
2. `if invalid_reason is not None:` 改成 `if False:` → 第 2 条断言应红。
3. `skip_stub` 的 `"skip": str(reason)` 改成 `"reason": str(reason)` → 第 3 条断言应红。
4. `choices=("dev",)` 改成 `choices=("dev", "holdout")` → 第 4 条断言应红。
5. 全部还原，确认回到 `4 passed`。

---

## Task 7: 端到端跑一次并核对产物

**Files:**
- Modify: `streaming_asr/tests/test_vbx_integration.py`（追加一条需要数据的断言）
- Create: `artifacts/v13_campplus_vbx_dev.json`

**Interfaces:**
- Consumes: Task 3 的缓存、Task 4 的基线核对、Task 6 的运行器
- Produces: 产物文件本身 + 一条可重复的核对断言

- [ ] **Step 1: 先跑单场冒烟（不要一上来跑 12 场）**

```powershell
cd C:\Users\EDY\Desktop\new
$env:PYTHONIOENCODING="utf-8"
python -u -B scripts/evaluation/run_campplus_vbx.py --gate rttm --meetings L_R003S01C02 --output artifacts\_vbx_smoke.json
```

Expected: 打印 `[pool] …`、`[plda] … R=…`，然后该场 baseline 错误率 ≈ `30.87%`（±1 点）。
⚠️ 这个期望值取自 `v13_candidate_score_table_dev.json`（同 `cap=10` 口径），
**不要**拿 `v13_selector_stability_diagnosis_rttm.json` 的 `worst` 字段（那里是 25.93%）
来对 —— 两者口径不同，混用会把一个正常的复现判成失败。先读一次期望值：

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import json;d=json.load(open(r'artifacts\v13_candidate_score_table_dev.json',encoding='utf-8'));print([(r['meeting'],r['k'],round(r['attribution_error'],4)) for m in d['meetings'] for r in m['rows'] if m['meeting']=='L_R003S01C02'])"
```

**若 baseline 与它对不上（±1 点之外）** ⇒ 停，查口径（`--cap` / `--share` / `--seconds`）。

- [ ] **Step 2: 检查冒烟产物里臂 1 确实动过标签**

```powershell
$env:PYTHONIOENCODING="utf-8"
python -c "import json;d=json.load(open(r'artifacts\_vbx_smoke.json',encoding='utf-8'));rows=d['gates']['rttm']['rows'];[print(r.get('arm'),r.get('fb'),r.get('k'),round(r.get('attribution_error',-1),4)) for r in rows]"
```

Expected: 有 `baseline` 一行 + `vbx_ahc` / `vbx_flat` 各 6 个 Fb 共 13 行；
`vbx_ahc` 在 `fb=17` 的 `k` 与 baseline 的 `k` 可以相同或不同，
但 `attribution_error` **不应与 baseline 逐位相同**（若相同，说明 VB 没起作用
—— 记下来，这本身是个发现，但要先确认不是接线错误）。

- [ ] **Step 3: 全量跑（两门 × 12 场）**

```powershell
python -u -B scripts/evaluation/run_campplus_vbx.py --output artifacts/v13_campplus_vbx_dev.json
```

Expected: 结束后打印 `[verdict] {...}`，产物落盘。
预计分钟级；若超过 30 分钟用后台任务并记 job id。

- [ ] **Step 4: 追加数据依赖的核对断言**

追加到 `streaming_asr/tests/test_vbx_integration.py`：

```python
ARTIFACT = ROOT / "artifacts" / "v13_campplus_vbx_dev.json"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="产物未生成（缺数据/缓存）")
def test_artifact_shape_and_no_holdout_rows():
    """产物形状 + 不含 holdout（spec §2.1、§7）。

    合理回归：有人把 split 传错或忘了过滤 —— 出现 holdout 场次名即红。
    """
    import json
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert d["schema"] == "v13-campplus-vbx-dev-v1"
    assert d["truth_used"] is False
    assert set(d["gates"]) <= {"rttm", "vad"}
    assert d["criterion"]["verdict"] in {"pass", "falsified", "invalid_design"}
    holdout_groups = {"L_R004C01", "S_R003C01"}
    import re
    for gate, block in d["gates"].items():
        for row in block.get("rows", []):
            m = re.fullmatch(r"([LMS]_R\d{3})S\d{2}(C\d{2})", row["meeting"])
            assert m, row["meeting"]
            assert m.group(1) + m.group(2) not in holdout_groups, row["meeting"]


@pytest.mark.skipif(not ARTIFACT.exists(), reason="产物未生成（缺数据/缓存）")
def test_same_meeting_same_gate_has_identical_window_counts():
    """三臂在同一场同一门上窗数必须逐位相同（坑 A7：两臂不同构）。

    合理回归：有人给某一臂的 `gate_windows` 换了窗长或门限 —— 断言红。
    """
    import json
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    for gate, block in d["gates"].items():
        seen = {}
        for row in block.get("rows", []):
            if "n_windows" not in row:
                continue
            key = row["meeting"]
            seen.setdefault(key, set()).add(row["n_windows"])
        for meeting, counts in seen.items():
            assert len(counts) == 1, f"{gate}/{meeting} 窗数不一致：{counts}"
```

- [ ] **Step 5: 跑测试 + 全量回归**

```
cd C:\Users\EDY\Desktop\new\streaming_asr
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_vbx_plda.py tests/test_vbx_integration.py -q
python -m pytest -q
```

Expected: 前者全绿；后者 **≥ 169 passed / 0 failed**（Global Constraints 的基线）。

- [ ] **Step 6: 红验证**

把产物 JSON 手工改一行：给某场加一条 `"meeting": "L_R004S01C01"`（holdout 组）
→ `test_artifact_shape_and_no_holdout_rows` 应红。**改回**。
把某场的某一行 `n_windows` 加 1 → 第 2 条断言应红。**改回**。

- [ ] **Step 7: 清掉冒烟临时件**

删 `artifacts\_vbx_smoke.json`。**保留** `v13_campplus_vbx_dev.json` 与
`v13_vbx_plda_trainL.json`。

---

## Task 8: 交接文档与结论记录

**Files:**
- Modify: `docs/HANDOFF_1013_后端对照spike.md`（§7 之后新增一节；§8 禁引表增行）
- Modify: `artifacts/v13_交付合并表.md`（§G 之后新增 §H）
- Modify: `docs/superpowers/specs/2026-10-10-说话人识别目标规范-design.md`（§4.3 的三处更正）

**Interfaces:**
- Consumes: `artifacts/v13_campplus_vbx_dev.json`（Task 7）
- Produces: 三份文档的改动 + 一条「不得引用」新行

- [ ] **Step 1: 在合并表加 §H，把数字逐条对上产物**

在 `artifacts/v13_交付合并表.md` 的 §G.4 之后追加：

```markdown
## H. CAM++ 上的真 VBx（RTTM + VAD 双门 · 诊断口径 · **不作交付数字**）

> 产物：`v13_campplus_vbx_dev.json`（三臂 × 两门 × Fb 曲线）·
> `v13_vbx_plda_trainL.json`（PLDA 参数，可复算）。
> 设计见 `docs/superpowers/specs/2026-10-13-CAM++-VBx-spike-design.md`。
> 判据预注册：双门各降 ≥ 1 点（RTTM ≤ 8.15%、VAD ≤ 6.19%）。

| 臂 | RTTM 门 | VAD 门 | 距判据 |
|---|---:|---:|---|
| 0 生产基线（复现） | {填} | {填} | — |
| 1 VBx-AHC (`Fb=17`) | {填} | {填} | {填} |
| 2 VBx-flat (`Fb=17`) | {填} | {填} | {填} |
| 诊断上界（读真值） | 4.26% | 4.05% | — |

`verdict` = **{填}**。{一段话说清结论与它**不能**说明什么。}
```

⚠️ 表里的 `{填}` 必须**从产物里读出来填**，不许手打估算。填完跑一次
`python -c "import json;print(json.load(open(r'artifacts/v13_campplus_vbx_dev.json',encoding='utf-8'))['criterion'])"`
把 `verdict` 抄进去。

- [ ] **Step 2: 更正 spec §4.3 那三处被推翻的表述**

在 `docs/superpowers/specs/2026-10-10-说话人识别目标规范-design.md` 的 §4.3
表格**下方**插入（不改原表，保留可追溯性）：

```markdown
> ⚠️ **2026-10-13 更正**：本表第二行「AHC 初始化 + VB HMM 93.18%」**不是 VBx 的读数**。
> 核实 `_v13_spike/compare_backends.py:43-57`：该臂把 192 维 CAM++ 嵌入乘**固定随机
> 正交基**升到 256 维，再喂 ResNet101 的 `transform.h5` + `plda`（脚本自述
> "uncalibrated"）。所以 93.18% vs 93.06% 是「失效变换 vs 什么都不做」。
> 另：本表「只能精修，无法恢复被 AHC 合并掉的说话人」**与源码相反** ——
> `VB_diarization.py:34` 的 `maxSpeakers` 打开的是 maxSpeakers 个**空簇**，
> 先验允许休眠说话人复活；那句话描述的是旧代码**怎么用**它，不是它的能力上限。
> 结论以 `docs/superpowers/specs/2026-10-13-CAM++-VBx-spike-design.md` §1 与
> `v13_交付合并表.md` §H 为准。
```

- [ ] **Step 3: HANDOFF 加一节 + 禁引表加行**

在 `docs/HANDOFF_1013_后端对照spike.md` 的 §7 之后插入一节 `## 7b. CAM++ 上的真 VBx（2026-10-13）`，
内容包含：判据、三臂两门的数字、`verdict`、`Fb` 曲线要点、以及**结论的边界**
（若 `falsified`：按 spec §3.2 不得再解释，转 spec §4.4 的 (1) 对齐文献口径）。

同时在 §8「不得引用的表述」表里加一行：

| 不要引用 | 原因 |
|---|---|
| 「AHC 初始化 + VB HMM 93.18% ⇒ VBx 在 CAM++ 上不行」 | 那一次该臂的输入经过**随机正交投影**（`compare_backends.py:43-57`），PLDA 未标定 ⇒ 读数无效。真 VBx 的结论见 §7b |

- [ ] **Step 4: 核对文档里没有编造的数字**

```powershell
cd C:\Users\EDY\Desktop\new
$env:PYTHONIOENCODING="utf-8"
python -c "import json;d=json.load(open(r'artifacts/v13_campplus_vbx_dev.json',encoding='utf-8'));print(json.dumps({'criterion':d['criterion'],'rttm':{k:v['mean_attribution_error'] for k,v in d['gates']['rttm']['arms'].items()}},ensure_ascii=False,indent=1))"
```

把输出与 §H / §7b 里写的数字**逐个比对**。对不上的以产物为准并改文档。

- [ ] **Step 5: 收尾核对（不做 git 操作）**

```powershell
cd C:\Users\EDY\Desktop\new
git status --short
git diff --cached --stat      # 必须为空
```

记录「已改动文件清单」进任务报告。**不 add、不 commit**（Global Constraints）。

---

## Self-Review

**1. Spec 覆盖检查**

| spec 节 | 落在哪个任务 |
|---|---|
| §1 为什么没测过 | Task 8 Step 2（更正 spec §4.3）|
| §2.1 数据划分 | Task 6 `--split` 只允许 dev；Task 7 Step 4 holdout 断言 |
| §2.2 类 = (场次, 标签) | Task 2 `test_classes_are_meeting_scoped_not_bare_labels`；Task 6 `build_pool` |
| §2.3 两个门 | Task 4（基线核对）+ Task 6 `--gate` + Task 7 |
| §2.4 三臂 | Task 5（`arm_labels`）+ Task 6 `meeting_arms` |
| §2.5 训练池窗与纯度 | Task 3（建缓存）+ Task 6 `build_pool`（`pure_windows`）|
| §2.6 PLDA 与 LDA | Task 2 全部 |
| §2.7 `Fb` 曲线 | Task 5 `FB_GRID`/`FB_CRITERION`；Task 6 逐 Fb 落行 |
| §3.1 双门判据 | Task 6 `verdict` + `test_verdict_requires_both_gates` |
| §3.2 失败分类 | Task 6 `verdict(..., invalid_reason=...)` + Task 7 Step 3 的基线不复现即停 |
| §3.3 反过拟合 | Global Constraints（`alpha`/`eps`/`R` 不由 dev 调；判据只认 `Fb=17`）|
| §3.4 次要观察 | Task 6 逐场行含 `floor_pass` 与 `k_vbx` |
| §4 组件边界 | File Structure + Task 1 偏差说明 |
| §5 数据流 | Task 3 → Task 6 的顺序即数据流 |
| §6 八条测试 | Task 1（哨兵）· Task 2（1、5、6）· Task 3（键契约）· Task 5（2、3）· Task 7（4、7、8）|
| §7 产物 schema | Task 6 `payload` 构造 |
| §9 成本风险 | Task 3 Step 6（显存/CPU 兜底）|
| §10 完成定义 | Task 7 Step 5（回归）+ Task 8（文档）|

**缺口（已知并接受）**：spec §6 测试 8（某场抛异常时产物含存根行）在 Task 6 里
只由 `skip_stub` 的单元测试覆盖，**没有**跑一次「真让某场抛异常」的集成验证。
⇒ **在 Task 7 Step 6 补一条**：把 `--meetings` 指向一个缓存不存在的场次
（如 `--meetings NOSUCHMEETING`），确认产物里出现带 `skip` 理由的存根行、
且脚本不崩。这一步必须真跑。

**2. 占位符扫描**：Task 8 的 `{填}` 是**刻意**的 —— 它们的值只有跑完 Task 7
才知道，且步骤里给了取值命令。其余无 TBD/TODO。

**3. 类型一致性核对**

| 名字 | 定义处 | 使用处 | 一致？ |
|---|---|---|---|
| `Plda.R` | Task 2 `fit_plda` | Task 5 `to_lda_space`/`speaker_prior`、Task 6 打印 | ✅ |
| `arm_labels(emb, plda, labels_ahc, arm, fb, cap, seed)` | Task 5 | Task 6 `meeting_arms` | ✅ |
| `FB_GRID` / `FB_CRITERION` / `VBX_HYPER` | Task 5 | Task 6 `payload["hyper"]` | ✅ |
| `engine_name()` | Task 5 | Task 6 `payload["engine"]` | ✅ |
| `skip_stub(meeting, gate, reason)` | Task 6 | Task 6 main、Task 6 测试 | ✅ |
| `verdict(per_gate, invalid_reason=None)` | Task 6 | Task 6 main、Task 6 测试 | ✅ |
| `train_meetings(root)` | Task 3 | Task 6 经 `train_index.json`（不直接调）| ✅ |
| `build_pool(cache, meetings, window, hop)` | Task 6 | Task 6 main | ✅ |
| `raw_candidates` / `vad_segments` | `probe_stability_selector`（既有）| Task 6 | ✅ |
| `BS.refine_labels` / `BS.sentence_metrics` | `probe_backend_sweep`（既有）| Task 6 | ✅ |
