# -*- coding: utf-8 -*-
"""锁住 VBx 引擎接入与本仓库的 PLDA 拟合。

每条断言都要能回答「哪个合理回归会让它保持绿」——答不出来就是没覆盖（坑 A5）。

**参照物纪律**：数值对照一律用第三方 `scipy.special.logsumexp` 当 oracle，
不用测试内自己重写的 max-shift 公式 —— 那与被测垫片是同一段公式的拷贝，
公式层面若出错会**两份同时错**，等于自己给自己打分（复审 Important，已按 R7 改）。
"""
import json
import os
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp  # oracle，独立第三方实现
from scipy.special import softmax as scipy_softmax      # oracle，独立第三方实现

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 注意：这条 import 是**承重**的，不只是"先用一下 backend"。它执行
# `vbx_backend` 顶部的 sys.path 垫，把 `scripts/evaluation/vbx_src` 放进
# sys.path —— 下面测试 2/3 里的 `import VB_diarization` 全靠它。
# `# noqa: E402` **不豁免 F401**，所以一次"清理未用 import"会让两条测试
# 直接 ModuleNotFoundError。测试 1 里对 `VB` 显式断言，正是为了钉住这条接线。
import vbx_backend as VB  # noqa: E402


def test_numexpr_shim_is_ours_not_the_real_package():
    """哨兵：`VB_diarization.ne` 必须是我们的垫片，且 `vbx_backend` 接线必须成立。

    合理回归 A：有人 `pip install numexpr` 之后垫片被真包遮住 —— 那时
    `ne.__name__` 变成 `numexpr`、`ne.__file__` 变成 site-packages 里的路径，
    两条断言各自红。这不是洁癖：产物要求**可复算**，而 `ne` 换成任何第三方包
    都会让 `logsumexp_ne` / `exp_ne` 换实现，"同一输入得到同一结果"就不再由
    本仓掌控。（原先此处写"真 numexpr 两套实现在 `-inf` 输入上行为不同"——
    那是对一个**当前装不上的包**的**未经验证**的断言，已按复审 Minor 删除。）

    合理回归 B：有人"清理未用 import"删掉模块顶部的 `import vbx_backend as VB`
    —— 测试 1 的 `VB.VD is VD` 与 `VB.SEED` 两条断言会红（`NameError` 或找不到名），
    而不是等到测试 2/3 才以 `ModuleNotFoundError` 的形式炸出来。

    写法说明（修复轮 1）：两条判定 `name_ok` / `file_ok` **先全算完再断言**，
    且都**不触碰 `.logsumexp` 属性**。原先还有第三条 `fn_ok`
    （`VD.ne.logsumexp.__module__ == "_numexpr"`），已按复审裁决**删除**：
    它对"`VD.ne` 指向我们的垫片"这个性质是多余的一层，而且**恰恰是遮蔽源** ——
    真 numexpr / numpy 顶替时它会先抛 `AttributeError`，把上面两条的失败一起卷进
    `ExceptionGroup`。删掉后，顶替场景下 `name_ok` 直接给出一条指名道姓的
    `AssertionError`。拦截能力不减（`name_ok` + `file_ok` 已完整覆盖该性质）。
    """
    import VB_diarization as VD

    name_ok = VD.ne.__name__ == "_numexpr"
    file_ok = Path(VD.ne.__file__).name == "_numexpr.py"
    assert name_ok, "VD.ne.__name__ = %r，垫片被顶掉了" % (VD.ne.__name__,)
    assert file_ok, "VD.ne.__file__ = %r，不是仓内垫片" % (VD.ne.__file__,)

    # 接线钉死：`vbx_backend` 必须真的把官方引擎挂出来，且 SEED 写死可复算
    assert VB.VD is VD, "vbx_backend.VD 不是同一个 VB_diarization 模块"
    assert VB.SEED == 20261013, "vbx_backend.SEED = %r" % (VB.SEED,)
    assert callable(VD.VB_diarization)


def test_logsumexp_shim_matches_scipy_oracle():
    """垫片的 logsumexp 与第三方 oracle 一致（**全有限值**输入域）。

    合理回归：有人把垫片写成 `np.log(np.sum(np.exp(x), axis))`（去掉 max-shift）。
    核正后的机理（修复轮 1，原 brief 写的"溢出→inf"不成立）：float64 下
    `exp(-1000)` **下溢为 0**，于是 `log(0) = -inf`（伴随
    `divide by zero encountered in log` 警告），`got` 第 2 行成为 `-inf`，
    红在 `allclose` 那条断言上，`isfinite` 根本轮不到。

    为什么 oracle 是 scipy 而不是测试内自写的 max-shift 公式：后者与
    `_numexpr.logsumexp` 是同一段公式的拷贝，公式层面若错会两份同时错
    （复审 Important，R7 已改）。
    """
    import VB_diarization as VD
    x = np.array([[0.0, 1.0, 2.0], [-1000.0, -1001.0, -999.0]])
    got = VD.ne.logsumexp(x, axis=1)
    want = scipy_logsumexp(x, axis=1)
    assert np.allclose(got, want)
    assert np.isfinite(got).all()


def test_numexpr_ne_paths_are_alive_and_match_the_reference():
    """`logsumexp_ne` / `exp_ne` 必须**能用**且结果正确，不能只是"不抛异常"。

    合理回归 A：`_numexpr` 的 `evaluate` 被人删掉或改窄 —— 官方 `logsumexp_ne`
    （`VB_diarization.py:236-237`）与 `exp_ne`（`:247`）全靠 `ne.evaluate`，
    一删就抛 `AttributeError`，本断言红。
    合理回归 B（更隐蔽）：把 `"xmax + log(x)"` 那条错接到 `exp` 上
    （`exp(s)` vs `xmax + log(s)`），此时不抛异常但数值全错，
    由与 oracle 的 `allclose` 抓住。
    合理回归 C：`evaluate` 的第一条表达式改走垫片自己的 `logsumexp`
    —— 那会让 `xmax` 被加两次，同样由 oracle 抓住。

    输入含 `-inf`：这正是不减最大值就会下溢的区段，也是 `logsumexp_ne` 相对
    裸版唯一有价值的地方。

    输入取**非均匀值**（`3.0, 1.0, -2.0`）：这样 `log(Σexp(·))` 不是整数，
    "`xmax` 被加两次"之类的错法不可能碰巧对上。用 `[[0,1,2], ...]` 这种输入时
    `log(Σexp)` 恰好是整数，有些错法会**蒙对**，断言就漏了。

    ⚠️ **覆盖边界，如实声明**（复审 Minor，R7）：
    - 本测试只迭代 `axis` 的 **2 个取值**（0 与 1），不是"三条轴"（旧措辞有误，已改）。
    - 输入里**没有任何一条轴整列 `-inf`**，即 `xmax` 永远不会是 `-inf` ⇒
      官方 `logsumexp_ne` 的 inf 补写分支（`VB_diarization.py:238-242`）
      **从未被执行、未被本测试覆盖**。要覆盖它需要一条整列 `-inf` 的输入，
      但那样"处处有限"这条断言就不成立，二者不可兼得。
    - 官方 `exp_ne` 那一行传的是 `out=None`，故 `out` 参数传到垫片后恒为 `None`；
      垫片 `exp(x, out=...)` 的**非 None 分支未被覆盖**。
    """
    import VB_diarization as VD

    x = np.array([
        [3.0, 1.0, -2.0],
        [-1000.0, -1001.0, -999.0],
        [float("-inf"), 0.0, 1.0],
    ])

    for axis in (0, 1):
        got = VD.logsumexp_ne(x, axis=axis)
        ref = scipy_logsumexp(x, axis=axis)   # 独立第三方 oracle
        assert np.isfinite(got).all(), "axis=%d 出非有限值: %r" % (axis, got)
        assert np.allclose(got, ref), "axis=%d: ne=%r != oracle=%r" % (axis, got, ref)

    got_exp = VD.exp_ne(x)
    assert np.isfinite(got_exp).all()
    assert np.allclose(got_exp, np.exp(x))


# ---------------------------------------------------------------------------
# Task 2 追加区：两协方差 PLDA 拟合 + LDA 空间（`scripts/evaluation/vbx_plda.py`）
# 上面 3 条是 Task 1 的产出，断言一字未动；自本条以下全部为追加。
# ---------------------------------------------------------------------------
import vbx_plda as VP  # noqa: E402


def _synthetic(n_classes=12, per_class=8, dim=6, seed=0):
    """合成两协方差数据：类均值本身也服从一个高斯。

    `n_classes < 4` 时 `m{c // 4}` 会把所有类折叠成同一个名字（全成 `m0:*`），
    退化成单类 —— `fit_plda` 会对一个**合法调用**抛"至少需要 2 个类"，
    把调用方的口径错误伪装成 `fit_plda` 的失败。这里显式挡住。
    """
    assert n_classes >= 4, "m{c//4} 的分组需要 ≥4 类"
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
    """`_whiten_between` 解 `Bc v = λ Wc v`：故 `wccnᵀ Bc wccn == diag(acvar)`、
    `wccnᵀ Wc wccn == I`，二者合起来等价于还原 `Bc`；且 `acvar` 全正、降序。

    ⚠️ 契约澄清（控制器裁决，实测确认；原措辞"还原类间协方差"不准确，已改）：
    本函数拿到的是**已收缩**矩阵 —— `fit_plda` 写的是
    `_whiten_between(_shrink(Wc, …), _shrink(Bc, …))`，收缩发生在**调用方**；
    `_whiten_between` **自身不收缩**。因此"`wccnᵀ · diag(acvar) · wccn` 还原类间
    协方差"这条只在**未收缩**入参下成立 —— 而且如上所述，那条式子本身方向就是错的
    （详见下面标了 ⚠️ 的那行代码注释）。本测试传的正是**未收缩**的 `Wc`/`Bc`
    （不是 `_shrink(Wc)`），所以断言测的是本函数在未收缩入参下的真实契约。

    合理回归：有人把 `eigh` 的结果按**升序**留下（numpy 默认升序）而不是
    反转为降序 —— `acvar` 变成升序、`sqrt` 后先验完全错位，"降序"断言红。
    ⚠️ 如实声明：**"两条反转一起去掉"这一种改法不会让还原断言红**（还原式只依赖
    `acvar` 与 `wccn` 的**配对**，不依赖排序方向）；让还原断言真红的是"只反转
    `acvar` 不反转 `S`"这种半反转改法，报告里有真命令与真输出。
    """
    rng = np.random.default_rng(3)
    Wc = rng.normal(size=(5, 5))
    Wc = Wc @ Wc.T + 0.3 * np.eye(5)
    Bc = rng.normal(size=(5, 5))
    Bc = Bc @ Bc.T + 0.1 * np.eye(5)
    wccn, acvar = VP._whiten_between(Wc, Bc)
    assert (acvar > 0).all()
    assert np.all(np.diff(acvar) <= 1e-12), "acvar 必须降序"
    # 「还原类间协方差」的正确方向。
    # ⚠️ 与简报的偏离（控制器裁决第 2 项，已如实上报）：简报这一行写的是
    # `wccn.T @ np.diag(acvar) @ wccn == Bc`，实测**必红**（残差 1.3e+01），
    # 且与下面那条白化断言在数学上互斥 —— Bc = wccnᵀ·diag(acvar)·wccn 要求
    # wccn 正交（实测 wccnᵀwccn 与 I 的残差 2.19），而白化要求
    # wccnᵀ Wc wccn = I，二者同时成立只能是 Wc = I。
    # 真契约是控制器裁决书里写下的那句：wccnᵀ Bc wccn == diag(acvar)
    # （实测残差 3.3e-15，等价表述 wccn⁻ᵀ diag(acvar) wccn⁻¹ == Bc 亦成立）。
    assert np.allclose(wccn.T @ Bc @ wccn, np.diag(acvar), atol=1e-8), \
        "wccnᵀ Bc wccn 应等于 diag(acvar)"
    # 白化：wccnᵀ Wc wccn == I
    assert np.allclose(wccn.T @ Wc @ wccn, np.eye(5), atol=1e-8)


def test_shrink_regularizes_singular_input_to_cholesky_decomposable():
    """`_shrink` 必须把**奇异**输入变成可 Cholesky 分解的矩阵。

    为什么必须单独钉这条：`_shrink` 的危险恰在于**把它删掉之后一切都还"跑通"**
    —— `fit_plda` 之外没有任何路径碰它，而 `fit_plda` 的合成数据是良态的，
    改成 `return M` 后 `_whiten_between` 里的 `cholesky(Wc)` 仍会成功、旧断言仍绿。
    所以简报原定的红验证 2（"把 `_shrink` 改成 `return M` 看旧断言变红"）
    在现有断言下**不成立**（详见报告，已实测），本测试把它补上。

    合理回归 A：有人把 `_shrink` 写成 `return M`（或以"没必要"为由删掉调用）
    —— `np.linalg.cholesky` 在秩 1 的 `np.ones((3,3))` 上抛 `LinAlgError`，断言红。
    合理回归 B：有人改成 `M + 一个大脊`（也能过 cholesky），非对角断言红。

    ⚠️ 覆盖边界（如实声明）：本条只覆盖"奇异入参仍可分解"+"非对角确实被收缩"，
    **不**覆盖 `_whiten_between` 直接拿到奇异 `Wc` 时的整体行为。
    """
    M = np.ones((3, 3))               # 秩 1 ⇒ 奇异，裸 cholesky 必抛 LinAlgError
    S = VP._shrink(M, 0.05, 1e-6)
    assert np.allclose(S, S.T), "收缩后必须仍是对称矩阵"
    L = np.linalg.cholesky(S)         # `_shrink` 改成 `return M` 时红在这一行
    assert np.allclose(L @ L.T, S), "cholesky 因子没有还原出 S"
    # 收缩的语义：非对角整体乘 (1 - alpha)。这条挡的是"加个大脊"式的假实现 ——
    # 那种实现同样能过 cholesky，却把类间/类内结构整个带偏。
    off_diag = ~np.eye(3, dtype=bool)
    assert np.allclose(S[off_diag], (1.0 - 0.05) * M[off_diag]), \
        "收缩必须作用在非对角上（(1-alpha) 倍），不是简单加脊"


def test_fit_rejects_too_few_classes():
    """少于 2 个类时必须抛，不许返回一个看似可用的模型。

    合理回归：有人把 `if len(unique) < 2: raise` 改成 `return None`
    —— 上层会拿到 None 再崩在别处，失败原因丢失（坑 A6 同族）。
    """
    x = np.random.default_rng(4).normal(size=(8, 4))
    with pytest.raises(ValueError, match="至少需要 2 个类"):
        VP.fit_plda(x, ["only"] * 8, alpha=0.1, eps=1e-6, lda_dim=2)


def test_lda_dim_default_is_min_32_and_classes_minus_one():
    """`R = min(LDA_DIM_CAP, 类数 − 1)` 的 **`类数−1` 分支**（12 类 ⇒ min(32, 11) = 11）。

    ⚠️ 覆盖边界（R2 更正 —— 原 docstring 写的"把 32 改成 128 ⇒ 断言红"是**假的**）：
    本条 fixture 只有 12 类，`min(32, 11)` 恒等于 `类数−1` ⇒ **对 cap 完全不敏感**。
    复审实测：`LDA_DIM_CAP` 改成 128 / 改成 11 / 把 cap 整个删掉，本条三种改法**全绿**。
    cap 分支改由 `test_lda_dim_cap_32_binds_at_production_scale`（40 类）专钉；
    spec §2.6 在**生产规模（207 类 ⇒ R = 32）**下生效的正是那一支。

    合理回归（本分支真正承重的那一类）：有人把 `类数 − 1` 写成 `类数`
    （`R = min(LDA_DIM_CAP, 类数)`）—— 12 类下 `R` 变 12，断言红（复审实测红在本行）。
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

    MIN-2（R2 追加）：`n_windows` / `alpha` / `eps` 也必须钉住。复审实测：
    `load` 把 `alpha`/`eps` 读成 `0.0` 时原测试仍全绿 —— 而 α/ε 是 spec §3.3
    「**不得按 dev 结果调**」的留痕量，静默丢失没人会发现。
    这里刻意用**非默认**的 `alpha=0.2, eps=1e-4`（模块默认是 0.05 / 1e-6），
    这样"load 忽略文件里的值、回落到模块默认常量"这种改法也会红。
    """
    x, y = _synthetic(n_classes=8, per_class=6, dim=10)
    plda = VP.fit_plda(x, y, alpha=0.2, eps=1e-4, lda_dim=4)
    path = tmp_path / "plda.json"
    plda.save(path)
    back = VP.Plda.load(path)
    assert np.array_equal(back.mean, plda.mean)
    assert np.array_equal(back.wccn, plda.wccn)
    assert np.array_equal(back.acvar, plda.acvar)
    assert back.R == plda.R and back.n_classes == plda.n_classes
    assert back.n_windows == plda.n_windows, "n_windows 是要写进产物的记账量，不许丢"
    assert back.alpha == plda.alpha, "alpha 是「不得按 dev 结果调」的留痕量（spec §3.3）"
    assert back.eps == plda.eps, "eps 同上"


# ---------------------------------------------------------------------------
# R1 追加（控制器修复轮 1 裁决）：符号规范化的定义性后果。
# 背景：简报 Step 1 的「还原」断言是**控制器的转写笔误**（把 `wccnᵀ Bc wccn`
# 写成了 `wccnᵀ diag(acvar) wccn`），控制器已裁决按其在裁决中写下的**文字契约**
# `wccnᵀ Bc wccn = diag(acvar)` 执行。详见 task-2-report.md §4.1 与 ## R1。
# ---------------------------------------------------------------------------


def _lead_entries(wccn):
    """每列绝对值最大的那个元素 —— 符号规范化的判别位置。"""
    return wccn[np.argmax(np.abs(wccn), axis=0), np.arange(wccn.shape[1])]


def test_wccn_is_sign_canonicalized_and_solver_sign_invariant(monkeypatch):
    """`wccn` 必须被钉成规范代表：每列绝对值最大的元素为正，且输出与 eigensolver
    的符号选择**逐位无关**。

    合理回归 A：有人删掉 `signs` 那三行（"符号本来就无所谓，都是同一组特征向量"）
    —— 本测试把 `np.linalg.eigh` 换成**每列取反**的替身（等价于换 BLAS / 换 driver
    之后 LAPACK 选到另一组符号）。出厂实现两次输出必须逐位相同；删掉 `signs` 后
    第二次输出整列翻号 ⇒ 下面 `lead > 0` 与 `array_equal` 两条同时红。
    合理回归 B：有人把符号判定的轴取错 / 改成 `argmin` —— `lead > 0` 直接红。

    为什么测的是 canonicalization 的**定义性后果**，而不是"两次跑结果相同"：
    同 BLAS 下 `eigh` 本来就稳定，形如
    `np.array_equal(_whiten_between(W, B), _whiten_between(W, B))` 的断言
    **删掉 `signs` 也不会红**（实测：无符号块时 10 passed 全绿）。这里钉的是
    "符号被钉死 ⇒ 输出对 solver 的任意符号不变"，前者只是后者的平凡特例。

    ⚠️ 判别性由**构造**保证，不看运气（控制器点名的陷阱）：本测试**不**依赖
    "本机 LAPACK 恰好给出负号"这一观察 —— 换个 BLAS 若恰好全给正号，
    `lead > 0` 那两条就会静默变成空气断言。这里由测试**强制**替身返回反号，
    于是第三条 `array_equal(wccn_flipped, wccn)` 是**必红**的：没有符号修正时，
    把 solver 的符号取反必然把输出整列取反，而特征向量非零 ⇒ 不可能等于自身。
    因此"删掉 `signs` 必红"跨环境成立（旁证见报告 §4.3：本机无符号块实现对这组
    seed=3 输入的 5 列里有 3 列给出负号，`lead` 从 `[1,1,1,1,1]` 变成
    `[1,-1,-1,-1,1]`）。

    ⚠️ 覆盖边界（如实声明）：本条钉的是"规范代表与 solver 符号无关"这一性质，
    **不**覆盖 `signs == 0` 那条分支（特征向量非零时进不去）。
    """
    rng = np.random.default_rng(3)     # 与 test_covariance_round_trip... 同一组确定性输入
    Wc = rng.normal(size=(5, 5))
    Wc = Wc @ Wc.T + 0.3 * np.eye(5)
    Bc = rng.normal(size=(5, 5))
    Bc = Bc @ Bc.T + 0.1 * np.eye(5)

    wccn, acvar = VP._whiten_between(Wc, Bc)
    assert (_lead_entries(wccn) > 0).all(), \
        "每列绝对值最大的元素必须为正（规范代表）"

    real_eigh = np.linalg.eigh

    def eigh_with_flipped_signs(M, *args, **kwargs):
        """同一组特征向量、每列取反 —— 模拟 LAPACK 选了另一组符号。"""
        w, V = real_eigh(M, *args, **kwargs)
        return w, -V

    monkeypatch.setattr(np.linalg, "eigh", eigh_with_flipped_signs)
    wccn_flipped, acvar_flipped = VP._whiten_between(Wc, Bc)

    assert (_lead_entries(wccn_flipped) > 0).all(), \
        "solver 换了符号之后，规范代表仍必须为正"
    assert np.array_equal(wccn_flipped, wccn), \
        "输出是规范代表 ⇒ 必须与 eigensolver 的符号选择逐位无关"
    assert np.array_equal(acvar_flipped, acvar), "符号选择不该影响特征值"


# ---------------------------------------------------------------------------
# R2 追加（复审轮 2 裁决）：IMP-1 / IMP-2 / MIN-1。
# ⚠️ 这三条补的是**计划造成的覆盖空洞（plan-mandated）**，不是实现缺陷：复审用
# 15 个变异实测证明 —— `LDA_DIM_CAP 32→128 / →11 / 整个删掉 cap`、
# `删掉调用点上的 _shrink(...)`、`DEFAULT_ALPHA+EPS→0`、`wccn[:, -R:]`、`acvar[-R:]`、
# `不减均值`、`load 读 alpha/eps 成 0.0` 这些改法在 R2 之前的 11 条测试下**全部绿**。
# 模块 `scripts/evaluation/vbx_plda.py` 本轮**行为零改动**（只加测试）。
# ---------------------------------------------------------------------------


def test_lda_dim_cap_32_binds_at_production_scale():
    """`R = min(LDA_DIM_CAP, 类数 − 1)` 的 **cap 分支** —— 生产规模（207 类 ⇒ R = 32）走的正是这支。

    与 `test_lda_dim_default_is_min_32_and_classes_minus_one` 的分工：那条只有 12 类
    （`min(32, 11) = 11`）⇒ 永远走 `类数−1` 分支、对 cap 完全不敏感。本条用 40 类
    （`min(32, 39) = 32`）专钉 cap。

    合理回归 A：`LDA_DIM_CAP` 改成 128（照抄官方 ResNet101 的 `lda_dim`）⇒ `R` 变 39，`R == 32` 红。
    合理回归 B：把 cap 整个删掉（退化成 `R = 类数 − 1` = 39）⇒ 同样红。

    断言分三层（不是只写 `R == 32`）：
    1. `R == 32`：cap 真的生效；
    2. `R == min(VP.LDA_DIM_CAP, 类数 − 1)`：与实现常量挂钩，cap 改值即红；
    3. `R < 类数 − 1`：cap 是**有效约束**，不是碰巧等于 `类数 − 1`（"删掉 cap" 专杀）。
    """
    x, y = _synthetic(n_classes=40, per_class=4, dim=64)
    plda = VP.fit_plda(x, y, alpha=0.05, eps=1e-6, lda_dim=None)
    assert plda.n_classes == 40
    assert plda.R == 32, "40 类下必须走到 cap：min(32, 39) = 32"
    assert plda.R == min(VP.LDA_DIM_CAP, plda.n_classes - 1)
    assert plda.R < plda.n_classes - 1, "cap 必须是有效约束（否则等于把 cap 删掉）"


def test_fit_plda_regularizes_singular_within_class_scatter():
    """`fit_plda` 的**唯一调用点**必须真的做正则化 —— spec §6 第 5 条点名的那条红验证。

    为什么必须钉在 `fit_plda` 这一层：单独测 `_shrink`（见
    `test_shrink_regularizes_singular_input_to_cholesky_decomposable`）只能抓
    "`_shrink` 自身被改坏"；而**把调用点上的 `_shrink(...)` 删掉、或把
    `DEFAULT_ALPHA`/`DEFAULT_EPS` 改成 0**，R2 之前的 11 条测试**全绿**（复审实测）。

    fixture：每类的所有窗**完全相同** ⇒ 类内散布 `Wc = 0`（奇异）。
    刻意用**模块默认** `alpha`/`eps`（不显式传参 —— 显式传参就绕过了默认常量，
    `DEFAULT_EPS → 0` 那种变异会漏掉）。

    合理回归 A：`_whiten_between(_shrink(Wc,…), _shrink(Bc,…))` 改成
    `_whiten_between(Wc, Bc)` ⇒ `cholesky(0)` 抛
    `LinAlgError: Matrix is not positive definite`，断言红。
    合理回归 B：`DEFAULT_EPS = 1e-6` 改成 `0.0`（正则化的单位阵脊消失）⇒ 同样红。
    合理回归 C（**本条盖不住，如实声明**）：`alpha`/`eps` 被改成别的**非零**量级 ⇒
    `acvar` 仍有限且全正，本条照旧绿；"数值被改小/改偏"由 MIN-1 那条数值断言管。
    """
    rng = np.random.default_rng(11)
    class_mu = rng.normal(size=(3, 5))
    x = np.repeat(class_mu, 4, axis=0)             # 每类 4 个窗逐行相同 ⇒ Wc = 0
    classes = [f"s{c}:{c}" for c in range(3) for _ in range(4)]
    plda = VP.fit_plda(x, classes, lda_dim=3)      # 走模块默认 alpha/eps
    assert np.isfinite(plda.acvar).all(), "正则化缺失会让 acvar 出 nan/inf"
    assert (plda.acvar > 0).all(), "eigenvoice 先验要开根号 ⇒ acvar 必须全正"
    assert np.isfinite(plda.to_lda_space(x)).all(), "fea 必须有限"


def test_lda_space_is_mean_centered_and_matches_speaker_prior():
    """`to_lda_space` / `speaker_prior` 的**数值**（不只是形状）—— Task 5 吃的就是这两行。

    复审实测：`wccn[:, -R:]`（取尾列）、`acvar[-R:]`（取尾段）、**不减均值**
    三种改法在原测试下全绿（原来只有 shape 断言）。

    两条断言都刻意与实现**不同构**（不是把公式抄一遍）：
    1. `fea` 的列均值 ≈ 0 —— 阈 1e-9。**本 fixture 实测**：出厂 `1.13e-15`，
       "不减均值"下 `3.65`；
    2. `fea` 空间的**类间散布对角** ≈ `speaker_prior()**2`（= `acvar[:R]`），
       相对误差阈 0.35。**本 fixture 实测**：出厂 `0.0349`；`wccn[:, -R:]` 下
       `1.0000`；`acvar[-R:]` 下 `429`。阈值 0.35 对出厂留了约 10 倍余量，
       同时又能接住上面两种偏移（收缩本身造成约 3.5% 的固有偏差，故阈值不能再收紧）。
       （复审给的是同一分支的另组数字 9.7e-16 / 4.76 / 0.036 / 1.0 / 556 —— 量级一致；
       差异只因 `_synthetic` 会先抽一个**没被使用**的 `mu`，换了 fixture 就换了 RNG 流。
       本 docstring 只写**本 fixture**实测值。）

    ⚠️ **不要**顺手断言该散布的**非对角** ≈ 0：本 fixture 出厂的非对角最大元是
    `0.685`（收缩使 `acvar` 与 `wccn` 不完全自洽），那样写会**假红** —— 复审专门点出的陷阱。
    """
    x, y = _synthetic(n_classes=12, per_class=8, dim=40)
    plda = VP.fit_plda(x, y, alpha=0.05, eps=1e-6, lda_dim=8)
    fea = plda.to_lda_space(x)
    assert fea.shape == (len(x), 8)

    mean_abs = np.abs(fea.mean(axis=0)).max()
    assert np.allclose(fea.mean(axis=0), 0.0, atol=1e-9), \
        "fea 必须已减全局均值（实测 max|列均值| = %.3g，阈 1e-9）" % (mean_abs,)

    # fea 空间类间散布，估计量与 `fit_plda` 内部同一口径：Σ_c n_c (μ_c−μ)(μ_c−μ)ᵀ / N
    mu = fea.mean(axis=0)
    scatter = np.zeros((8, 8))
    for name in sorted(set(y)):
        rows = fea[[i for i, c in enumerate(y) if c == name]]
        d = rows.mean(axis=0) - mu
        scatter += len(rows) * np.outer(d, d)
    scatter /= len(x)

    prior_var = plda.speaker_prior() ** 2          # = acvar[:R]
    rel = np.abs(np.diag(scatter) - prior_var) / prior_var
    assert (rel < 0.35).all(), \
        "fea 空间类间散布对角应 ≈ speaker_prior²（最大相对误差 %.3g）" % (rel.max(),)


# ---------------------------------------------------------------------------
# Task 3 追加区：建 train_L 的 RTTM 门窗缓存（`scripts/evaluation/build_train_vbx_cache.py`）。
# 上面 14 条（Task 1 的 3 条 + Task 2 的 11 条）断言一字未动；自本条以下全部为追加。
# ---------------------------------------------------------------------------
import build_train_vbx_cache as BTC  # noqa: E402

# R1/R2：缓存目录在**仓库外**（`D:\yinp\AISHELL-4\_v13_spike\cache\train_L`），
# CI / 别的机器上没有。用环境变量指路，红验证才能把它指到装着"故意写坏 npz"的
# 临时目录；`VBX_TEST_CACHE_OUT` 是 `validate_out()` 要拒绝的那个 test 侧树。
CACHE_OUT = Path(os.environ.get("VBX_TRAIN_CACHE_OUT",
                                r"D:\yinp\AISHELL-4\_v13_spike\cache\train_L"))
TEST_CACHE_OUT = Path(os.environ.get(
    "VBX_TEST_CACHE_OUT", r"D:\yinp\AISHELL-4\_v13_spike\cache\test"))


@pytest.fixture
def tmp_spike(tmp_path, monkeypatch):
    """把 `BTC.SPIKE` 指到 tmp：合法的 `out` 必须落在**缓存根之内**。

    R1 之后 `validate_out()` 会拒掉缓存根之外的 out（含 `tmp_path/"out"` 这种
    方便写法），所以凡是要走完 `build()` 守卫、进入下游读盘的测试，
    都得先有一个"合法的 out"。返回 `(spike, out)`，`out` 已是合法的 train_L 子目录。
    """
    spike = tmp_path / "spike"
    monkeypatch.setattr(BTC, "SPIKE", spike)
    return spike, spike / "cache" / BTC.TRAIN_SUBDIR


def test_train_meetings_found_by_date_prefix(tmp_path):
    """train_L 的场次发现不能走 `v12_aishell4.meeting_group`。

    合理回归：有人图省事复用 `split_meetings(root, "dev")` —— 它内部对每个
    stem 调 `meeting_group()`，而 train_L 的 id 是 `20200706_L_R001S01C01`
    形态，会抛 `ValueError: invalid AISHELL-4 meeting id`，断言红。

    Task 3 追加的第三个文件（brief Step 5 指定）：`L_R003S01C02.rttm` 是**合法
    AISHELL-4 id 但不是 train_L 场次**（它是 test/dev 组的场次）。没有它，
    "去掉前缀过滤"这个变异**不会红**（`notes.txt` 不是 `.rttm`，本来就被 glob
    挡掉），红验证就成了空气 —— 有它，去过滤的变异必然多出这一行，断言红。
    """
    grid = tmp_path / "TextGrid"
    grid.mkdir()
    for name in ("20200706_L_R001S01C01", "20200707_L_R002S08C01"):
        (grid / f"{name}.rttm").write_text("", encoding="utf-8")
    (grid / "L_R003S01C02.rttm").write_text("", encoding="utf-8")   # ← 必须加
    (grid / "notes.txt").write_text("ignore me", encoding="utf-8")
    assert BTC.train_meetings(tmp_path) == [
        "20200706_L_R001S01C01", "20200707_L_R002S08C01"]


def test_cache_keys_match_the_test_side_contract(tmp_path, monkeypatch):
    """产物键名必须与 test 侧的 `w2_h1.npz` 完全一致。

    合理回归：有人漏写 `votes` —— `probe_pair_oracle.pure_windows` 读
    `data["votes"]` 会 KeyError，品控门失效；这里用键集合断言挡住。

    ⚠️ 覆盖边界（如实声明）：本条**自带** fixture，不调 `BTC.build`，故严格说它
    钉的是「契约本身」而非「产物」。真产物的键由 Step 6 的实测核对承担
    （单场试水实读四键，见 task-3-report.md）。之所以不在这里调 `build`：
    那需要真模型 + 真音频，属数据/模型依赖，按本线红线应 `pytest.skip` 而非
    在单测里硬跑 —— 那会把"契约"测试变成"环境"测试。
    """
    npz = tmp_path / "w2_h1.npz"
    np.savez_compressed(npz, emb=np.zeros((2, 192), np.float32),
                        times=np.array([0.0, 1.0], np.float32),
                        speakers=np.array(["001-M", "002-F"]),
                        votes=np.array([1.0, 1.0], np.float32))
    data = np.load(npz, allow_pickle=False)
    assert set(data.files) == {"emb", "times", "speakers", "votes"}
    assert data["emb"].shape[1] == 192


def test_build_hard_rejects_any_non_train_split(tmp_spike, tmp_path):
    """★ 控制器追加的硬守卫（用户 2026-10-13 裁决）：`build` 只认 train_L 场次名。

    R1 适配：本测试改用 `tmp_spike` fixture 拿一个**合法** out（`tmp_spike` 是
    `validate_out()` 允许的路径）。原先用 `Path("__no_such_out__")` —— R1 之后
    `validate_out()` 会先一步拒掉它，于是"合法 id 放行到读盘"这条反向对照
    会收到 `NotTrainSplit` 而不是 `FileNotFoundError`，**测试自身的判别性会被
    另一个守卫遮蔽**。改路径不改变任何一条断言的语义。

    合理回归 A：有人把 `NotTrainSplit` 的 `startswith(TRAIN_PREFIX)` 去掉
    （"调用方自己会传对"）—— 那时 `build("L_R004S01C01", …)` 会继续往下走去
    读 holdout 的 flac/rttm 并**把缓存写进 holdout 目录**，`pytest.raises` 红。
    这不是洁癖：spec §2.1 规定 `L_R004*` / `S_R003*` 是封存集，而缓存一旦落盘，
    "读没读过"就再也无法自证。

    判别性由**构造**保证，不看环境：守卫是 `build` 的**第一条语句**（在
    `import soundfile` 之前），本测试传的三个都是明显不存在的假路径 —— 守卫在位
    就必然抛 `NotTrainSplit`，**碰不到任何真实音频、也不会落盘**（末尾断言假输出
    目录没被创建）；守卫被拿掉则必然 `FileNotFoundError`（假 flac）
    ⇒ `pytest.raises(NotTrainSplit)` 直接红，且照样读不到数据。

    合理回归 B：有人把守卫放宽 / 写反（例如 `if meeting.startswith(...)` 漏了
    `not`）—— 第三个 `pytest.raises(FileNotFoundError)` 钉住**合法 train_L id
    必须放行**：它得走到读盘那一步才对。守卫写反时这里会收到 `NotTrainSplit`，
    断言红。两条互为反向对照，"守卫永远抛"和"守卫永不抛"都盖得住。

    （`L_R003S01C02` 与上面那条测试用的是同一个 id：它是**合法 AISHELL-4 场次、
    但不是 train_L** —— 只测 `L_R004*`/`S_R003*` 会漏掉"守卫其实是拿
    `meeting_group` 之类去判的"这种实现。）
    """
    fake_root = tmp_path / "no_such_root"       # 绝对路径：R2 的包含性断言要 resolve 它
    _spike, out = tmp_spike
    for meeting in ("L_R004S01C01", "S_R003S01C02", "L_R003S01C02", "L_R004S02C01"):
        with pytest.raises(BTC.NotTrainSplit):
            BTC.build(meeting, fake_root, out, 2.0, 1.0, {"spk": None})
    assert not out.exists(), "守卫必须在落盘之前抛出，不许留下半成品目录"

    # 反向对照：合法 train_L 前缀必须**放行**（进而红在下游的读盘上）
    with pytest.raises(FileNotFoundError):
        BTC.build("20200706_L_R001S01C01", fake_root, out, 2.0, 1.0,
                  {"spk": None})


# 缓存目录在**仓库外**（`D:\yinp\AISHELL-4\_v13_spike\cache\train_L`），CI / 别的机器上没有。
# 用环境变量指路，红验证才能把它指到一个装着"故意写坏的 npz"的临时目录上。
# （`CACHE_OUT` / `TEST_CACHE_OUT` 常量已在上面 Task 3 的 import 区定义。）


def test_every_cached_meeting_decodes_and_matches_the_index():
    """`train_index.json` 里的每条记录都必须与**真能解码出来**的 npz 对得上。

    为什么必须是"完整解码"而不是只查 `files`：`.npz` 是 zip，**中央目录在文件尾**。
    实测（本仓真跑，见 task-3-report.md 的矩阵）：
    - 把 npz 截断到 99.9% / 99% / 90% / 50% / 10% ⇒ `np.load` **连 `.files` 都读不出**
      （`BadZipFile: File is not a zip file`）；
    - 但**只翻数据区 1 个字节**（zip 结构完好）⇒ `.files` 正常、键集合完全正确，
      **只有把数组读出来才会发现 `BadZipFile: Bad CRC-32 for file 'emb.npy'`**。
    所以"只查键存在性"的核对会漏掉后一类损坏，本条把 `z[k][...]` 全读出来当闸门。

    合理回归 A：`build()` 写成"先 `index[m] = entry` 再落盘 npz"或落盘被打断 ——
    文件坏了而 index 声称成功；`np.load` 在 `BadZipFile` 上直接 error，本断言红。
    合理回归 B：index 与实际产物**错位**（比如中途改了 window/hop 再续跑，
    旧记录留着新的 n）—— `长 == record["n"]` 红。
    合理回归 C：`dim` 口径漂了（换模型 / 换 L2 归一化层）—— `192` 那条红。

    ⚠️ 缺数据一律 `pytest.skip`（坑 W3：数据/模型缺失不许变成 error）。
    但**"index 在、文件却不在"不 skip，硬红** —— 那正是本测试要抓的错位，
    把它 skip 掉等于把最贵的一类失败（A6 静默丢场）伪装成"环境不具备"。

    ⚠️ 覆盖边界（如实声明）：本条只核对 **train_L 侧**（`2020*`）产物；
    它**不**碰 test / holdout 目录，也不核对 votes 的语义（只核长度）。
    """
    index_path = CACHE_OUT / "train_index.json"
    if not index_path.exists():
        pytest.skip(f"train_L 缓存 index 不存在（未建缓存）：{index_path}")
    meetings = json.loads(index_path.read_text(encoding="utf-8")).get("meetings", {})
    if not meetings:
        pytest.fail(f"index 存在但没有任何场次记录：{index_path}")

    checked, skipped = [], []
    for meeting, record in sorted(meetings.items()):
        if "skip" in record:                       # 带 skip 理由的存根行：不算产物
            skipped.append(meeting)
            continue
        # C1 修复（复审实证"第 3 层未修"）：index **键本身**也必须过白名单。
        # 原先这里只写 `assert meeting.startswith("2020")` —— 键
        # `"2020/../L_R004S01C01"` 能通过它，随后 `CACHE_OUT / meeting` 里的 `..`
        # 在 `exists()` / `np.load` 时被折叠，命中**非 `2020*`** 的目录：
        # 四键真读出、n/dim 全对，直到末尾 `assert on_disk` 才红 —— 即"靠别的断言兜"。
        # ⇒ 改成**用归一化后的返回值**（不是把它当个断言看），并让"index 被污染"
        #   以 `pytest.fail` 的清晰形式呈现，而不是抛一个看不懂的 `NotTrainSplit`。
        try:
            meeting = BTC.validate_meeting(meeting)
        except BTC.NotTrainSplit as exc:
            # `from None` 很关键：否则 pytest 会把原始 `NotTrainSplit` 当作
            # exception context 一并打印，失败信息里出现两行 `E ...`，
            # 读的人要自己分辨哪条才是真正的断言失败原因。
            raise pytest.fail.Exception(
                f"index 键未过白名单：{meeting!r} ⇒ {exc}") from None
        npz = CACHE_OUT / meeting / "w2_h1.npz"
        assert npz.exists(), f"index 声称 {meeting} 已建，但 {npz} 不存在"
        with np.load(npz, allow_pickle=False) as data:
            # 四个键**全部真读出来** ⇒ 触发真实解码 + CRC 校验
            files = set(data.files)
            emb = data["emb"][...]
            times = data["times"][...]
            speakers = data["speakers"][...]
            votes = data["votes"][...]
        assert files == {"emb", "times", "speakers", "votes"}, \
            f"{meeting}: 键集合 = {sorted(files)}"
        assert len(times) == len(emb) == len(speakers) == len(votes) == record["n"], (
            f"{meeting}: index 记 n={record['n']}，实际 "
            f"emb={len(emb)} times={len(times)} speakers={len(speakers)} "
            f"votes={len(votes)}")
        assert emb.shape[1] == record["dim"] == 192, \
            f"{meeting}: dim = {emb.shape[1]}，index 记 {record['dim']}"
        # ★ 升级版闸门（控制器 R3 追加）：用**实测出来的** duration，而不是记账值。
        # `seconds` 是**外部传入的参数**，`duration` 是 `build()` 从**实际读到的音频**
        # 算出来的结果（`len(audio)/rate`）—— **两者不同源，所以能互相证伪**。
        # 只断言 `seconds == 0.0` 时，"用 --seconds 60 建 + 手工把 seconds 改成 0.0"
        # 这种组合抓不到（断言与被断言对象同源 ⇒ 无法证伪；本线已踩过三次同类：
        # Task 1 的 IMP-1 参照物同族、Task 2 的 `_shrink` 零覆盖、R2 的 `:165-174` 零覆盖）。
        assert record["duration"] > 600, (
            f"{meeting}: duration={record['duration']:.1f}s ⇒ 这批产物不是整场建的"
            "（可能 --seconds 截断过）")
        checked.append(meeting)

    assert checked, (f"index 里没有一场可核对的产物（全部 skip？{skipped}）："
                     f"{index_path}")
    assert not skipped, f"有场次带 skip 存根行（不许静默丢场）：{skipped}"

    # I1 修复（复审实测的洞）：`--seconds` 必须入账，且全量缓存必须是 `0.0`。
    # 没有这条时：npz 只留 26 窗、index 记 n=26 ⇒ 本条测试**照样 passed**，
    # 于是"只跑了前 60 秒的局部缓存"与"整场缓存"在产物层面无法区分 ——
    # 而报告 §8 又把 `--seconds 60` 写成推荐的烟测命令、与全量共用 `--out`。
    meta = json.loads(index_path.read_text(encoding="utf-8"))
    assert meta.get("seconds") == 0.0, (
        f"index 的 seconds = {meta.get('seconds')!r}（期望 0.0 = 整场）。"
        "非 0 说明这是局部缓存，不能当 train_L 全量池用。")

    # I2 修复（复审实测的洞）：**孤儿 npz** 必须被抓。
    # 反向遍历产物目录：每个 w2_h1.npz 都必须出现在 index 里。
    # 这正是"np.savez 非原子写被打断"留下的形态 —— `main()` 是 `build()` 返回
    # **之后**才写 index，所以残缺产物可以存在而 index 里没有它。
    # 没有这条时：index 里一个正常场次 + 一个不在 index 里的截断 npz ⇒ 照样 passed。
    on_disk = {p.parent.name for p in CACHE_OUT.glob("2020*/w2_h1.npz")}
    assert on_disk, f"产物目录里一个 w2_h1.npz 都没有（{CACHE_OUT}）—— 检查 --out 指对了没"
    orphans = sorted(on_disk - set(meetings))
    assert not orphans, (
        f"这些场次的 npz 在磁盘上、却不在 index 里（孤儿产物 / 疑似写盘被打断）："
        f"{orphans}")
    unlisted_only = sorted(set(meetings) - on_disk - {m for m in skipped})
    assert not unlisted_only, f"index 声称已建但磁盘没有：{unlisted_only}"
# ---------------------------------------------------------------------------
# R3 追加（复审者，2026-10-14，控制器裁决）：路径穿越绕过硬守卫。
# ⚠️ 本条在**当前未修复**的实现上**故意红** —— 修复（把两层守卫从
# "未归一化字符串的 startswith" 换成 "归一化后仍以 2020 开头、且不含
# `..` / 路径分隔符"）由控制器统一裁决，不在本轮改源码。
#
# 为什么必须补（复审者实测，非推理）：`build()` 的守卫只看 `str(meeting)` 的
# 前缀，而它真正打开的是 `Path(root) / "wav" / f"{meeting}.flac"` ——
# **校验对象 ≠ 使用对象**：`"2020/../L_R004S01C01"` 通过前缀检查，
# `..` 直到打开文件时才被折叠。把该名字与 `--root D:\yinp\AISHELL-4\data\test`
# 一起用时，实测：守卫放行 → 读了 holdout 的 flac/rttm → 把
# `_v13_spike\cache\L_R004S01C01\w2_h1.npz` **就地覆盖**（755 B / 0 窗口，
# mtime 2026-10-14 10:43:59，原 ~400 KB）。`main()` 的整轮前置守卫用的是
# **同一个** `startswith` 字符串检查，同样放行该 argv。
# 结论一句话：**只校验未归一化的字符串，使用的却是归一化后的路径。**
# ---------------------------------------------------------------------------
HOSTILE_MEETINGS = (
    "2020/../L_R004S01C01",            # 前向斜杠，归一化后落到 holdout 场次
    "2020\\..\\S_R003S01C01",            # Windows 分隔符，同样落到 holdout
    "2020/../../x",                    # 归一化后逃出 out 目录
    "2020/sub/../../L_R004S01C01",     # 多一层中转
    "2020/../20200706_L_R001S01C01",   # 归一化成一个**合法** train_L id：写盘目标
                                       # 与 index 键会指向不同场次 ⇒ 同样必须拒
    "2020/./L_R004S01C01",             # 未逃逸，但含路径分隔符 ⇒ 同样必须拒
)

LEGAL_MEETINGS = ("20200706_L_R001S01C01", "20200999_L_R099S99C99")


def test_guard_rejects_path_traversal_meeting_names(tmp_spike, tmp_path):
    """★ 守卫必须按**归一化后的名字**判，不能只判未归一化的字符串。

    契约（控制器 2026-10-14 裁决）：归一化后仍必须是**单个**路径分量、
    且整体匹配 `BTC.TRAIN_MEETING_RE` 白名单（R2 强化：逐特征拒 `..` 会漏
    `"2020/./L_R004S01C01"` 这类）。

    合理回归 A（本条要抓的）：守卫留在 `str(meeting).startswith(...)` 上
    —— `"2020/../L_R004S01C01"` 通过检查，`build()` 继续去读
    `root/wav/L_R004S01C01.flac`（holdout）并写 `out/L_R004S01C01/w2_h1.npz`。
    本断言红在 `pytest.raises(NotTrainSplit)`（实际抛的是下游的
    `FileNotFoundError`，临时目录里当然没有那份 holdout 数据）。
    合理回归 B：有人"修"成 `if not Path(meeting).name.startswith(...)` 或
    无条件拒 —— 反向对照那两条（合法 id 必须**放行**到读盘）会红。

    ⚠️ 判别性由**构造**保证：全部用 `tmp_path` 下的假 root/out，任何分支都碰不到
    真音频、也不会落盘；末尾 `assert not out.exists()` 钉住"守卫在落盘之前抛"。

    R1 适配（两处，均为**让本测试恢复判别性**、不改任何断言语义）：
    1. 改用 `tmp_spike` fixture 拿**合法** out —— 原先的 `tmp_path/"out"` 会被
       R1 新增的 `validate_out()` 先一步拒掉，那时 `pytest.raises(NotTrainSplit)`
       变成**真空通过**（红在另一个守卫上，测不出场次名守卫是否还在）；
    2. 名单形状的预检改用**白名单**口径（原 `os.path.normpath` +
       "含分隔符/含.."的逐特征口径，对 `"2020/./L_R004S01C01"` 会算作**没有理由**，
       从而把一条合法条目误判成"不该在名单里"）。
    """
    _spike, out = tmp_spike

    # 名单自身的形状：每条都必须有"为什么必须拒"的硬理由（防止名单退化成空气）
    for name in HOSTILE_MEETINGS:
        unified = name.replace("\\", "/")
        parts = PurePosixPath(unified).parts
        reasons = []
        if "/" in name or "\\" in name:
            reasons.append("含路径分隔符")
        if ".." in name:
            reasons.append("含 ..")
        if len(parts) != 1:
            reasons.append(f"归一化后是 {len(parts)} 个路径分量")
        if not BTC.TRAIN_MEETING_RE.match(parts[0] if len(parts) == 1 else name):
            reasons.append("不符合场次名白名单格式")
        assert reasons, f"{name!r} 既不逃逸也不违反白名单，不该出现在 HOSTILE 名单里"

    for name in HOSTILE_MEETINGS:
        with pytest.raises(BTC.NotTrainSplit):
            BTC.build(name, tmp_path, out, 2.0, 1.0, {"spk": None})
    assert not out.exists(), "守卫必须在落盘之前抛出，不许留下半成品目录"

    # 反向对照：合法 train_L id 必须放行（红在下游的读盘上），否则"一律拒"也能过
    for name in LEGAL_MEETINGS:
        with pytest.raises(FileNotFoundError):
            BTC.build(name, tmp_path, out, 2.0, 1.0, {"spk": None})


# ---------------------------------------------------------------------------
# R2 追加（控制器 2026-10-14）：守卫的**纯函数**层 + `main()` 整轮前置守卫。
# 背景：复审把三层守卫**全部**实测穿透（含 `main()` 层与它自己那条测试）。
# 修法两条：(1) 白名单正则整体匹配，而不是逐特征拒；(2) `out` 根做包含性断言
# ——"`out=任意路径` 照样能写出去"这条通路此前完全没人管。
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "2020/../L_R004S01C01",            # 落到 holdout
    "2020\\..\\S_R003S01C01",           # Windows 分隔符同样落到 holdout
    "2020/../20200706_L_R001S01C01",   # 归一化成一个合法 id ⇒ 写盘目标与 index 键不一致
    "../20200706_L_R001S01C01",        # 相对逃逸
    "/abs/20200706_L_R001S01C01",      # 绝对路径
    "20200706_L_R001S01C01/../../x",   # 尾随逃逸
    "2020/./L_R004S01C01",             # 未逃逸但含分隔符（逐特征口径会漏这条）
    "2020/../../x",
    "2020/sub/../../L_R004S01C01",
])
def test_validate_meeting_rejects_path_traversal(name):
    """`validate_meeting` 必须拒绝一切"不是单个场次名"的输入（纯函数层）。

    ⚠️ 判别性边界（实测，如实声明；初稿的措辞是**错的**，已改）：
    本参数化有 **9 条**，但**不是 9 条都对本条要抓的缺陷敏感**。
    实测（见 task-3-report.md 的 `## R2`）：把 `validate_meeting` **整体退回**
    事故当时的旧守卫 `str(meeting).startswith(TRAIN_PREFIX)` ⇒
    **`7 failed, 2 passed`** —— 红的正是 7 条以 `2020` 开头、从而骗过旧前缀检查的
    条目；另 2 条（`"../20200706_L_R001S01C01"`、`"/abs/20200706_L_R001S01C01"`）
    不以 `2020` 开头，**旧守卫本来就会拒**，所以它们对这次回归**不敏感**，
    留着是为了钉住"绝对路径/相对路径也必须拒"这个独立性质。
    另注：单独删掉归一化那几行（保留白名单正则）时本条**全绿** ——
    因为白名单本身也能拦住这些名字（纵深防御，不是单点）。
    ⇒ 所以本条的"合理回归 A"应当表述为：**退回旧守卫**（而不是"删掉归一化"）。

    合理回归 A（复审实证过的穿透）：退回旧前缀守卫 ⇒ 上列 7 条红。
    合理回归 B：只按 POSIX 解析（不先把 `\\` 当分隔符）⇒
    `"2020\\..\\S_R003S01C01"` 被当成单个文件名而通过，该参数红。
    合理回归 C：白名单正则被删掉 ⇒ `"2020"` 之类仍会通过（由
    `test_validate_meeting_rejects_non_meeting_shape` 专钉）。

    纯函数调用：**不碰文件系统**、不 build、不加载模型。
    """
    with pytest.raises(BTC.NotTrainSplit):
        BTC.validate_meeting(name)


@pytest.mark.parametrize("name", [
    "20200706_L_R001S01C01",
    "20200709_L_R002S08C01",
    "20200999_L_R099S99C99",
])
def test_validate_meeting_returns_the_normalized_name(name):
    """合法 id 必须**原样返回**（`build()` 后面只许用这个返回值拼路径）。

    合理回归：有人把返回改成 `Path(meeting).name` 或 `normpath(meeting)` ——
    对合法输入这两者恰好相等，故本断言**挡不住**那种改法（如实声明）；
    它真正钉的是"返回值 == 归一化名"这个契约本身，以及下面这条：
    返回值必须**不是** `None`（有人忘了 `return`）—— 那时 `build()` 会用
    `"None.flac"` 去读盘，属静默错场。
    （M5 断言卫生：原先还有一行 `assert isinstance(got, str)`，它被
    `got == name` 蕴含、永远不可能单独红 —— 已删。）
    """
    got = BTC.validate_meeting(name)
    assert got == name, f"归一化后应原样返回，实际 {got!r}"


@pytest.mark.parametrize("name", [
    "2020",                            # 只有前缀，不是场次名
    "2020\x00x",                       # NUL：下游 Path 操作会抛 ValueError 而非干净的 NotTrainSplit
    "20200706_L_R001S01C01 ",          # 尾随空格
    "20200706_L_R001S01C01.py",
    "20200706_L_R001S01C01.rttm",
    "20200706_l_R001S01C01",           # 小写 l
    "20200_L_R001S01C01",              # 位数不对
    "",
    ".",
    "..",
])
def test_validate_meeting_rejects_non_meeting_shape(name):
    """白名单正则层：**看着像但不是场次名**的输入也必须被拒。

    这一层是 R2 加的。没有它时：`"2020"` 会通过（下游 `FileNotFoundError` ⇒
    被 `main()` 记成 skip 存根行而**静默丢场**，坑 A6）；
    `"2020\\x00x"` 会让 `Path` 抛 `ValueError`（**不是** `FileNotFoundError`，
    于是既不被 skip 兜住、也不被守卫拦住）。

    合理回归：`validate_meeting` 里删掉 `TRAIN_MEETING_RE.match(name)` 那三行
    ⇒ 前两条红。
    """
    with pytest.raises(BTC.NotTrainSplit):
        BTC.validate_meeting(name)


def test_validate_out_rejects_paths_outside_the_cache_root(tmp_path, monkeypatch):
    """`out` 根必须有**包含性断言** —— 这是复审点名的"最宽的洞"。

    没有它时：`--out <任意路径>` 照样把 train_L 缓存写出去，
    `--out <test 侧根>` 更是直接搅坏已被 Task 4 依赖的那份产物。
    要求：解析后必须落在缓存根之内、且不得落在 test 侧缓存树里。

    合理回归：删掉 `validate_out` 里的前缀／包含性判断（或把它改成只判
    `out` 非空）⇒ 下面三条"必须拒"的断言红。

    用 `tmp_path` 造一棵假的缓存树，**不碰真缓存**。
    """
    spike = tmp_path / "spike"
    monkeypatch.setattr(BTC, "SPIKE", spike)
    cache = spike / "cache"

    # 合法：缓存根之内的 train_L 子目录（可以还不存在 —— mkdir 会建）
    assert BTC.validate_out(cache / "train_L") == (cache / "train_L").resolve()
    # 合法：更深的子目录也允许（只要在根之内）
    assert BTC.validate_out(cache / "train_L" / "sub") == \
        (cache / "train_L" / "sub").resolve()

    # 逐条断言"**哪条规则**在开火"，而不是"抛了就完事"。
    # ⚠️ 为什么不能只断言 `pytest.raises(BTC.NotTrainSplit)`：实测
    # `cache` / `cache/test` 这类路径**两条规则都可能拒**，只看异常类型的话，
    # 把第 2 条（test 侧）删掉、第 1 条（包含性）照样兜住 ⇒ 断言**恒真**。
    # 这正是本线反复踩的"同源/不判别"老毛病，故这里分开钉。
    for bad, why, token in (
        (cache, "缓存根本身 ⇒ 包含性规则", "不在指定缓存根"),
        (cache / "test", "test 侧缓存根 ⇒ test 侧规则", "test 侧缓存树"),
        (cache / "test" / "probe", "test 侧子目录 ⇒ test 侧规则", "test 侧缓存树"),
        (spike / "elsewhere", "缓存根之外 ⇒ 包含性规则", "不在指定缓存根"),
        (tmp_path / "totally_elsewhere", "完全在外 ⇒ 包含性规则", "不在指定缓存根"),
    ):
        with pytest.raises(BTC.NotTrainSplit) as exc:
            BTC.validate_out(bad)
        assert token in str(exc.value), (
            f"（{why}）期望开火的是 {token!r} 那条规则，实际信息：{exc.value}")
    # 包含性断言的**穿越**场景：写在缓存根之内、但用 `..` 逃出去
    with pytest.raises(BTC.NotTrainSplit):
        BTC.validate_out(cache / "train_L" / ".." / ".." / "x")


def test_assert_inside_rejects_paths_that_escape_the_root(tmp_path):
    """`assert_inside(path, root, label)` 必须拒掉逃出根的路径（M3 补覆盖）。

    为什么必须补：R2 把这段包含性断言**内联**在 `build()` 里，复审把它改成
    `if False:` 之后 **28 条相关测试逐字相同、全绿** —— 即那条"最宽的洞"的补丁
    **零覆盖**。R3 把它抽成纯函数，这里直接钉住。

    合理回归（红验证已真跑）：把函数体里的判断去掉（改成 `return Path(name)`）
    ⇒ 本测试红在下面 `pytest.raises(BTC.NotTrainSplit)` 的
    `DID NOT RAISE NotTrainSplit` 上。
    （如实表述：**本测试未参数化**，多个逃逸形态用一个 `for` 循环覆盖；
    上一版 docstring 写成"本参数化红"，措辞与实现不符，已改。）

    顺带钉住 M2 的修法：`path` 与 `root` **两边口径一致**。
    原先只对 `path` 做 `os.path.normpath`、基准用 `Path(root).resolve()`，
    传**相对** root 时会误报"逃出了允许的根"（fail-closed 但信息误导）。
    """
    root = tmp_path / "root"
    (root / "wav").mkdir(parents=True)

    # 合法：根之内（含不存在的子路径）
    assert BTC.assert_inside(root / "wav" / "a.flac", root, "wav") == \
        (root / "wav" / "a.flac")
    assert BTC.assert_inside(root / "wav" / "sub" / "a.flac", root, "wav") == \
        (root / "wav" / "sub" / "a.flac")

    # 非法：用 `..` 逃出根
    for escaped, why in (
        (root / "wav" / ".." / ".." / "outside.flac", "上跳两级"),
        (root / ".." / "outside.flac", "上跳一级"),
        (tmp_path / "totally_elsewhere.flac", "完全在外"),
    ):
        with pytest.raises(BTC.NotTrainSplit):
            BTC.assert_inside(escaped, root, "wav")

    # M2：**相对** root 也必须两边同口径（不该因为形式不同就误报逃逸）
    assert BTC.assert_inside(root / "wav" / "a.flac", str(root), "wav") == \
        (root / "wav" / "a.flac")


def test_main_guard_rejects_traversal_before_loading_models(tmp_path, monkeypatch):
    """★ `main()` 的**整轮前置守卫**必须拦住穿越 argv，且**在加载模型之前**。

    为什么这条能便宜地测（纠正控制器先前的记账依据）：`web_server` 只在
    `main()` 内部 import，用 `sys.modules` 注入一个**替身**即可观察"模型加载
    有没有发生" —— 实测约 0.4 秒，**不加载任何模型**。控制器先前据此裁定
    "不补测"的依据（要加载模型 65 s+）是错的。

    合理回归 A（复审实测）：删掉 `main()` 里那段非法场次前置检查 ⇒ 替身
    `get_models` **被调用** ⇒ `visits` 非空，本断言红；且不会被
    `pytest.raises(SystemExit)` 掩盖（那时根本不会抛 SystemExit）。
    合理回归 B：**两个变异都已真跑**（R3，原文见 task-3-report.md `## R3`）：
    (a) 删掉 `out = validate_out(args.out)` 那一行 ⇒ 第二条子用例红（`DID NOT RAISE`）；
    (b) 把该行挪到 `get_models()` **之后** ⇒ 同样红（替身被调用 ⇒ `visits` 非空）。

    ⚠️ **本条的价值来自一次被抓的错**：I3 之前的版本里，第二条子用例的 root 是
    **空 TextGrid**，`main()` 在"没找到场次"就退出了，**根本走不到 out 校验** ⇒
    上面 (a)(b) 两个变异当时**都照样 passed**，即那半段是**恒真**的；
    而且当时写的 `pytest.raises(SystemExit)` 与真实行为（`NotTrainSplit`）连
    **异常类型都不一致**，只是靠提前退出才碰巧通过。
    复审是**靠本文档里那句没验过的"合理回归 B ⇒ 红"**发现它名不副实的 ——
    ⇒ **docstring 里写"某变异会红"之前必须真的跑一遍。**

    全部在 `tmp_path` 与替身模块上跑：**不碰真缓存、不加载模型**。
    """
    import types

    visits = []
    fake_ws = types.ModuleType("web_server")

    def fake_get_models(*a, **k):                  # pragma: no cover - 不该被调用
        visits.append((a, k))
        raise AssertionError("守卫失效：模型被加载了")

    fake_ws.get_models = fake_get_models
    monkeypatch.setitem(sys.modules, "web_server", fake_ws)

    # 造一个"合法但查不到场次"的 root，然后从 **argv** 传穿越型场次名 ——
    # 这样守卫的输入正是 argv 里那个未归一化的字符串（不能造一个文件名里带
    # 斜杠的 .rttm：Windows 根本建不出这种文件，而"建不出来"本身也会让测试失真）。
    root = tmp_path / "train_root"
    (root / "TextGrid").mkdir(parents=True)

    spike = tmp_path / "spike"
    monkeypatch.setattr(BTC, "SPIKE", spike)
    monkeypatch.setattr(sys, "argv", [
        "build_train_vbx_cache.py", "--root", str(root),
        "--out", str(spike / "cache" / "train_L"),
        "--meetings", "2020/../L_R004S01C01"])

    with pytest.raises(SystemExit) as exc:
        BTC.main()
    assert "拒绝执行" in str(exc.value), f"期望被前置守卫拒，实际：{exc.value!r}"
    assert visits == [], "守卫必须在加载模型之前退出，但 get_models 被调用了"

    # 第二条：输出根越界（test 侧）也必须在加载模型之前被拒。
    # ⚠️ I3 修复（复审实证本段此前**恒真**）：原先这里 root 的 TextGrid 是**空的**
    # ⇒ `main()` 在"没找到场次"那条 `SystemExit` 就退出了，**根本走不到 out 校验**
    # ⇒ 删掉 `out = validate_out(args.out)` 整行、或把它挪到 `get_models()` 之后，
    #    这条都**照样 passed**（复审两个变异实测均 `1 passed`）。
    # 现在放一个**合法** rttm 让它真走到 out 校验；并同时修正异常类型预期：
    # 修前 `validate_out` 抛的是 `NotTrainSplit`(ValueError)，与这里写的
    # `pytest.raises(SystemExit)` **类型都不一致**（当时只因提前退出才碰巧通过）。
    (root / "TextGrid" / "20200706_L_R001S01C01.rttm").write_text(
        "", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "build_train_vbx_cache.py", "--root", str(root),
        "--out", str(spike / "cache" / "test")])
    with pytest.raises(SystemExit) as exc2:
        BTC.main()
    msg2 = str(exc2.value)
    # ⚠️ 钉住**规则 2 独有的文案**，不要用宽泛的 `"test" in msg2`：
    # 复审实测（变异 `i3_msg2_no_name`）—— `tmp_path` 的名字
    # （形如 `test_main_guard_rejects_traver0`）与路径本身都含 "test"，
    # 于是把规则 2 的文案改成**不点名 test 侧**之后，`"test" in msg2` **仍然通过**
    # ⇒ 那条断言的判别性为零。这里改钉规则 2 独有的那句话。
    assert "落在 test 侧缓存树" in msg2, \
        f"期望开火的是『落在 test 侧缓存树』那条规则，实际：{msg2!r}"
    assert "拒绝" in msg2, f"期望是干净的 CLI 拒绝，实际：{msg2!r}"
    assert visits == [], "输出根被拒时也不许已经加载过模型"
    assert not (spike / "cache" / "test").exists(), "不许在 test 侧留下任何目录"


# ===========================================================================
# Task 5：三臂的 VBx 标签接口
#
# ⚠️ 本节全部用**合成数据**（`np.random.default_rng` 造的向量 + 现造的 `Plda`）：
# 不读 `_v13_spike/cache/`、不读音频、不加载任何模型。
#
# ⚠️ brief 的 `test_flat_arm_is_reproducible_and_ahc_arm_is_not_ahc` 里第二条
# 断言（`not np.array_equal(c, labels_ahc)`）**实测不可能绿且修不掉**，判据
# 已按 A5 换掉。完整证据见 task-5-report.md §缺陷 1，要点：
#   - 初值 `np.zeros(60)` ⇒ `k0 == 1` ⇒ `maxSpeakers=1` ⇒ VB 只能输出单一
#     说话人 ⇒ 输出恒等于输入。（这不是"臂 1 退化成臂 0"，是"没给可拆分的结构"。）
#   - 换成 k0=2 的非退化初值后，该断言要么恒真要么恒假：初值恰好等于 VB 的解
#     （含整块互换重编号后相等）⇒ 红；初值与解不同 ⇒ 绿。它测的是"初值对不对"，
#     而它宣称要抓的变异（臂 1 提前 `return labels_ahc`）在**两侧都照样通过**
#     ⇒ 判别性为零。
#   - brief 提的 fallback 也不覆盖接线：`_dirichlet_init` 同 seed 两次相等是
#     **恒真**；"不同 seed 不等"只证明 seed 有效果，变异"把 seed 写死成 1、
#     忽略入参"实测**照样绿**。
#   ⇒ 改为 spy `VB.VD.VB_diarization`，钉"调用次数 + 收到的 gamma"。
#     已实测：臂 1 提前 return ⇒ 调用 0 次 ⇒ 必红。
# ===========================================================================

_VBX_HYPER_KEYS = {"Fa", "loopProb", "maxIters", "epsilon",
                   "init_smoothing", "minDur", "threshold"}


def _synth_emb(n_per_group=30, dim=8, shift=3.0, seed=7):
    """两团合成嵌入：前一半整体平移 `shift`。返回 (emb, 真实 2 团标签)。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2 * n_per_group, dim))
    x[:n_per_group] += shift
    truth = np.zeros(2 * n_per_group, dtype=int)
    truth[n_per_group:] = 1
    return x, truth


def _synth_plda(x, lda_dim, per_group=6, groups=2):
    """现造一个 `Plda`：类 = `f"m{i//per_group}:s{i%per_group}"`（带场次前缀）。"""
    classes = [f"m{i // per_group}:s{i % per_group}" for i in range(len(x))]
    assert len(set(classes)) == (len(x) + per_group - 1) // per_group * groups \
        or len(set(classes)) >= 2
    return VP.fit_plda(x, classes, alpha=0.05, eps=1e-6, lda_dim=lda_dim)


def _capture_vb(monkeypatch):
    """把 `VB.VD.VB_diarization` 换成记录调用的替身，返回承载记录的列表。

    注意补丁打在 `VB.VD`（模块对象）上而不是 `VB` 上 —— `arm_labels` 走的是
    `VD.VB_diarization` 这个属性查找，两者是同一个模块对象。
    """
    calls = []
    real = VB.VD.VB_diarization

    def spy(fea, m, iE, V, pi=None, gamma=None, **kw):
        calls.append({"fea": fea, "m": m, "iE": iE, "V": V,
                      "pi": pi, "gamma": gamma, "kw": kw})
        return real(fea, m, iE, V, pi=pi, gamma=gamma, **kw)

    monkeypatch.setattr(VB.VD, "VB_diarization", spy)
    return calls


def _vbx_stub(sentinel_gamma, monkeypatch):
    """把 `VB_diarization` 换成"只回放给定 gamma"的替身，用来构造稀疏/倒序 argmax。

    ⚠️⚠️ **参数名不能叫 `gamma`**（本函数第一版就踩了这个坑，实测）：
    替身自己的形参也叫 `gamma`，会在**替身内部**遮蔽闭包变量 ——
    于是 `return np.array(gamma)` 拿到的是 `arm_labels` 传进来的初值、**不是哨兵**，
    替身看起来"完全没生效"（表现为：真实 VB 的结果、或 Dirichlet 初值的 argmax）。
    这里刻意把闭包变量改名成 `sentinel_gamma`，让遮蔽不可能发生。

    签名其余部分必须与官方逐字一致（`pi` 是第 5 个**关键字**参数）：`arm_labels`
    用 `pi=None` 调它，形参名字对不上会让替身自己 TypeError。
    """
    def stub(fea, m, iE, V, pi=None, gamma=None, **kw):
        return np.array(sentinel_gamma), \
            np.ones(sentinel_gamma.shape[1]) / sentinel_gamma.shape[1], [0.0]

    monkeypatch.setattr(VB.VD, "VB_diarization", stub)


def _sentinel_gamma(raw_argmax, n_frames, n_cols):
    """造一个 `argmax == raw_argmax` 的**离散 one-hot** gamma 哨兵。

    ⚠️ 必须逐行写、**不能**用 `gamma[np.arange(n), raw_argmax] = 0.9` 那种花式索引：
    若 `raw_argmax` 恰好长度等于行数（60）而 gamma 是 `(60, 3)`，numpy 会把
    `np.arange(60)` 与 `raw_argmax` 一起当成**展平后的**下标对，散到完全不同的
    位置上去；而 `gamma[np.arange(60), raw_argmax]` 里的 `raw_argmax` 若写成了
    `(60,3)` 的形状（例如 `np.array([...]*10)` 被 `np.full((60,3))` 广播）还会
    静默变成三列一起赋值。两种都只会让你得到一个"看起来没生效"的桩子。
    这里用显式的 `np.eye` 行取值，杜绝这类踩空。
    """
    raw_argmax = np.asarray(raw_argmax, dtype=int)
    assert raw_argmax.shape == (n_frames,), f"raw_argmax 形状 {raw_argmax.shape}"
    assert raw_argmax.min() >= 0 and raw_argmax.max() < n_cols, "raw_argmax 越界"
    return np.eye(n_cols)[raw_argmax]


def test_diarization_lib_is_vendored_and_reachable():
    """`AHC` / `cos_similarity` / `twoGMMcalib_lin` 必须来自**仓内** vbxs_src。

    合理回归：有人删掉 `vbx_backend` 的 sys.path 垫 → 这三个名字 import 不到，
    臂 1 的 AHC 初值路径整个失效。断言它们来自仓内目录而不是别处。

    （顺带钉住：安装成第三方包的 `diarization_lib` 会让 `parent.name` 变成
    `site-packages` 而不是 `vbx_src`，第一条路径断言红。）
    """
    import diarization_lib as DL
    assert Path(DL.__file__).name == "diarization_lib.py"
    assert Path(DL.__file__).resolve().parent.name == "vbx_src"
    assert all(callable(getattr(DL, n)) for n in
               ("AHC", "cos_similarity", "twoGMMcalib_lin"))


def test_vbx_hyper_and_fb_grid_are_pinned_to_the_spec():
    """§4.2/§4.3 定死的超参网格必须逐值成立，且 `engine_name()` 如实标注引擎。

    合理回归 A：有人"顺手调参"（例如 `Fa` 改成 1.0、`maxIters` 改成 10、
    `threshold` 改成 0）—— 判据臂的复算基线就漂了，逐值断言红。

    合理回归 B：有人把 `Fb` 网格多加/少加一格、或把判据臂从 17 挪走 ⇒ 第二组断言红。

    合理回归 C：有人把 `engine_name()` 写成 `return "numexpr"`（直接写死，
    或改成 `import numexpr` 是否成功来判断）—— 本机 numexpr **没装**，
    写死或探测"包在不在"都会给出与真实 `VD.ne` 不符的答案，第三条断言红。
    这里比的不是字面量，而是"返回值必须与 `VD.ne.__name__` 的实际取值一致"，
    所以将来真装了 numexpr、`VD.ne` 被顶掉时它同样红。
    """
    assert VB.VBX_HYPER == {"Fa": 0.3, "loopProb": 0.99, "maxIters": 40,
                            "epsilon": 1e-6, "init_smoothing": 5.0,
                            "minDur": 1, "threshold": -0.015}
    assert set(VB.VBX_HYPER) == _VBX_HYPER_KEYS
    assert VB.FB_GRID == [1, 5, 10, 17, 30, 60]
    assert VB.FB_CRITERION == 17
    assert VB.FB_CRITERION in VB.FB_GRID, "判据臂必须在网格里"
    assert VB.SEED == 20261013

    import VB_diarization as VD
    expected = "numexpr" if VD.ne.__name__ == "numexpr" else "numpy"
    assert VB.engine_name() == expected, (
        "engine_name()=%r 与 VD.ne.__name__=%r 不一致" % (VB.engine_name(), VD.ne.__name__))
    assert VB.engine_name() in ("numpy", "numexpr"), "产物里的 engine 字段取值域只有这两个"


def test_flat_arm_gamma_is_caller_supplied_with_the_fixed_seed(monkeypatch):
    """臂 2 的 `gamma` 必须**由调用方按 SEED 造好再传**，绝不给 `VB_diarization` 传 `None`。

    为什么这条必须用 spy 而不是"跑两次比结果"（实测，见 task-5-report.md §R3）：
    同一份数据上把 `gamma` 换成 `None` 后 VB 仍**可能**收敛到同一个 `argmax`
    —— 60x8 那份数据 6/6 轮都红（差 6~47 帧），但 40x6 那份只有 3/6 轮红。
    即"跑两次"是**看运气才红**的断言，不能承重。

    ⚠️⚠️ **不要把本测试的数据换成更小的 fixture（例如 brief 测试 B 那份 40x6）**：
    本测试的判别性**依赖 fixture 规模**。已实测：60x8（`_synth_emb` 的默认值）
    上"`gamma` 传 `None`"6/6 轮都改变 `argmax`；40x6 上只有 3/6 轮改变。
    换句话说，把数据改小会让 spy 契约里的"`gamma is not None`"那条断言
    （以及若有人退回"跑两次比结果"的写法）**变成看运气才红**。
    另外 `gamma0.shape == (60, 4)` 与 `default_rng(SEED).gamma(size=(60, 4))`
    也把 `len(x) == 60` 写死了 —— 改 fixture 必须同步改这两处。

    合理回归 A（承重）：`gamma0 = None` / 删掉 `_dirichlet_init` 调用 / 把参数
    错传成 `None` ⇒ `calls[0]["gamma"] is None` ⇒ 红。

    合理回归 B：`_dirichlet_init` 把入参 seed 丢掉（例如写死成常量 1）⇒ 抓到的
    gamma 与**测试内独立算出的** `default_rng(SEED)` 抽样不等 ⇒ 红。这条已实测：
    `default_rng(SEED)` 与 `default_rng(1)` 的抽样不等，且"跑两次比结果"**抓不到**
    这个变异（两次仍相等）—— 只有逐位比对参照物才抓得到。

    合理回归 C：`_dirichlet_init` 忘记按行归一化（返回未归一化的 raw gamma）、
    或把 `cap` 与 `R` 弄混（形状错）、或把 `len(fea)` 写成 `len(emb)`
    ⇒ 形状/数值断言红。

    参照物纪律：期望值用**测试内独立的** `np.random.default_rng(SEED).gamma(...)`
    当场算，**不调用** `VB._dirichlet_init`、也不用任何 `VB.*` 的公式 ——
    否则就是拿被测实现给自己打分。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)
    labels_ahc = np.zeros(len(x), dtype=int)

    calls = _capture_vb(monkeypatch)
    # ⚠️ `cap` 刻意取 5 而 `plda.R == 4`（`lda_dim=4`）：两者相等的话
    # "maxSpeakers 误用 R" 这个变异会**绿**（已实测 M16 存活过一轮）。
    out = VB.arm_labels(x, plda, labels_ahc, "vbx_flat", fb=17.0, cap=5)

    assert plda.R != 5, "本测试要求 cap != R，否则 maxSpeakers 的两条路径不可区分"
    assert len(calls) == 1, f"臂 2 必须恰好调用一次 VB_diarization，实际 {len(calls)} 次"
    gamma0 = calls[0]["gamma"]
    assert gamma0 is not None, (
        "臂 2 给 VB_diarization 传了 gamma=None：内部会走无种子的 "
        "np.random.gamma，产物不可复算")
    assert gamma0.shape == (60, 5), f"gamma 形状 {gamma0.shape} != (60, 5)"

    raw = np.random.default_rng(VB.SEED).gamma(1.0, size=(60, 5))
    expected = raw / raw.sum(axis=1, keepdims=True)
    assert np.array_equal(gamma0, expected), (
        "gamma 不等于测试内独立算出的 default_rng(SEED) Dirichlet 初值")
    assert np.allclose(gamma0.sum(axis=1), 1.0), "gamma 没有按行归一化"
    assert np.all(gamma0 > 0), "gamma 必须全正"

    assert calls[0]["kw"]["maxSpeakers"] == 5, (
        "臂 2 的 maxSpeakers 必须是 cap（=5），实际 %r —— "
        "写成 R 会让搜索空间与 cap 脱钩" % (calls[0]["kw"]["maxSpeakers"],))
    assert calls[0]["kw"]["Fb"] == 17.0, "判据臂的 Fb 必须是 17"
    assert calls[0]["kw"]["maxIters"] == VB.VBX_HYPER["maxIters"]
    assert calls[0]["kw"]["epsilon"] == VB.VBX_HYPER["epsilon"]
    assert calls[0]["kw"]["loopProb"] == VB.VBX_HYPER["loopProb"]
    assert calls[0]["kw"]["Fa"] == VB.VBX_HYPER["Fa"]
    assert calls[0]["kw"]["minDur"] == VB.VBX_HYPER["minDur"]

    assert len(out) == len(x)
    assert np.issubdtype(out.dtype, np.integer), f"硬标签必须是整数，实际 {out.dtype}"


def test_ahc_arm_feeds_the_labels_in_and_does_not_short_circuit(monkeypatch):
    """臂 1 必须**真的把 AHC 标签做成初值喂给 VB**，不许提前把标签原样返回。

    ⚠️ 为什么判据是 spy 而不是 brief 原文的 `not np.array_equal(c, labels_ahc)`：
    后者实测**判别性为零**（详见本节抬头注释与 task-5-report.md §缺陷 1）。
    spy 版本对"臂 1 退化成臂 0"这个变异**恒红**，不依赖初值恰好落在哪一侧。

    ⚠️ **本测试的初值刻意用 brief 原样的 `np.zeros(60)`（k0=1）**，而全 0 是
    VB 的**不动点**（`softmax(onehot*5.0)` 把单列 one-hot 变成均匀后验，2 轮收敛，
    `argmax` 恒 0）⇒ 输出恒等于输入。因此本测试**不拿"输出 ≠ 输入"当判据**，
    只用 spy 契约。**下一个人请不要因为"输出等于输入看着像 bug"而把初值改掉**：
    换成 k0≥2 的初值会让这条断言变成"测某个特定初值下的数值结果"，既脆弱，
    又仍然抓不到"提前 return"（那时输出恰好也可能不等于输入）。

    合理回归 A（这条断言的靶心）：臂 1 分支被改成
    `return _relabel_by_first_seen(labels_ahc)`（省掉 VB 那一步，臂 1 退化成臂 0）
    ⇒ `calls == []` ⇒ 红。已实测。

    合理回归 B：`maxSpeakers` 不再由 `len(unique(labels_ahc))` 决定（例如写死 10、
    或误用 `cap`）⇒ `kw["maxSpeakers"] != k0` ⇒ 红。

    合理回归 C：初值不再是 one-hot（例如把 `gamma0` 传成 `None`、或传成均匀矩阵、
    或忘了乘 `init_smoothing` 再 softmax）⇒ `gamma is None` / 行最大值 ≈ 1/k0
    / 不构成 one-hot 三条断言各自红。注意 `init_smoothing=5.0` 下 one-hot 行是
    `softmax([5,0]) = [0.9933, 0.0067]`，**不是**严格 one-hot，所以这里钉的是
    「行最大值 > 0.9」而不是「等于 1」。

    合理回归 D：`fea` 不是 LDA 空间（例如直接把 `emb` 喂进 VB）⇒ `fea.shape[1] != R` ⇒ 红。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)
    labels_ahc = np.zeros(len(x), dtype=int)   # k0 = 1，与 brief 原文一致

    calls = _capture_vb(monkeypatch)
    out = VB.arm_labels(x, plda, labels_ahc, "vbx_ahc", fb=17.0, cap=4)

    assert len(calls) == 1, (
        "臂 1 没有把标签喂给 VB（调用 0 次）或多调了 —— "
        "0 次就是『臂 1 退化成臂 0』那个变异")
    call = calls[0]
    gamma0 = call["gamma"]
    assert gamma0 is not None, "臂 1 也不许给 VB_diarization 传 gamma=None"
    assert call["fea"].shape == (60, plda.R), (
        f"喂给 VB 的 fea 不是 LDA 空间：{call['fea'].shape} != (60, {plda.R})")
    assert call["kw"]["maxSpeakers"] == 1, (
        "maxSpeakers 必须等于 len(unique(labels_ahc))，实际 %r" % (call["kw"]["maxSpeakers"],))
    assert gamma0.shape == (60, 1)
    assert np.allclose(gamma0.sum(axis=1), 1.0), "初值必须是后验（行和为 1）"
    assert float(gamma0.max(axis=1).min()) > 0.9, (
        "one-hot × init_smoothing 再 softmax 后每行应接近 one-hot")
    assert call["pi"] is None, (
        "臂 1 传了 pi=%r：VB 会用 len(pi) 覆盖 maxSpeakers，"
        "臂 1 的搜索空间就跟生产候选的 K 脱钩了" % (call["pi"],))

    assert np.array_equal(out, labels_ahc), (
        "k0=1 时 VB 只能输出单一说话人 —— 这是**已知边界**，不是缺陷；"
        "本条断言把它钉住，另一侧由 test_ahc_arm_maxspeakers_follows_the_label_count 覆盖")


def test_ahc_arm_maxspeakers_follows_the_label_count(monkeypatch):
    """臂 1 的 `maxSpeakers` 必须跟着**标签里实际出现的类数**走，不是 `cap`、不是常数。

    合理回归：有人把 `k0` 写死成 `cap`、或把 `maxSpeakers=cap` 从臂 2 复制到臂 1
    —— 那 `maxSpeakers != labels_ahc.max() + 1`，臂 1 的搜索空间就跟生产候选的 K
    脱钩，"同一个候选划分上 VB 能吸掉多少错"这个问法直接失效。断言红。

    反向也钉住：这里 `cap=4 != k0=3`，所以"臂 1 误用 cap"必然被抓到；
    若两者相等这条就退化了 —— 故刻意让它们不等。

    ⚠️ 本测试里的 `gamma0` 逐元素断言是**补的一课**（实测 M9 变异存活过一轮）：
    `k0 == 1` 时整行只有一个元素、softmax 恒为 1，所以
    `test_ahc_arm_feeds_the_labels_in_and_does_not_short_circuit` 里那条
    "行最大值 > 0.9" **抓不到**"把 `init_smoothing` 乘成 0（均匀初值）"这个变异。
    换成 `k0 = 3` 并逐元素比对（期望值用**第三方** `scipy.special.softmax`
    当场算，不抄被测公式）之后，改平滑系数 / 去掉 softmax / 用均匀矩阵都必红。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)
    labels_ahc = np.array([0] * 20 + [1] * 20 + [2] * 20)

    calls = _capture_vb(monkeypatch)
    out = VB.arm_labels(x, plda, labels_ahc, "vbx_ahc", fb=17.0, cap=4)

    assert len(calls) == 1
    assert calls[0]["kw"]["maxSpeakers"] == 3, (
        "maxSpeakers 必须 = len(unique(labels_ahc)) = 3，实际 %r"
        % (calls[0]["kw"]["maxSpeakers"],))
    gamma0 = calls[0]["gamma"]
    assert gamma0.shape == (60, 3)
    assert len(out) == 60

    # 逐元素钉住初值：one-hot(label) × init_smoothing 再 softmax。
    # 期望值用第三方 `scipy.special.softmax` 独立算，避免与被测同源。
    onehot = np.zeros((60, 3))
    onehot[np.arange(60), labels_ahc] = 1.0
    assert np.array_equal(onehot.sum(axis=1), np.ones(60)), "自检：one-hot 造对了"
    expected = scipy_softmax(onehot * VB.VBX_HYPER["init_smoothing"], axis=1)
    assert np.array_equal(gamma0, expected), (
        "臂 1 的初值不是 one-hot × init_smoothing 的 softmax：\n"
        "got 首行 %r\nexp 首行 %r" % (gamma0[0].tolist(), expected[0].tolist()))
    assert not np.allclose(gamma0, np.full((60, 3), 1.0 / 3)), (
        "初值是均匀后验 ⇒ init_smoothing 没起作用（或 onehot 被抹平了）")


def test_ahc_arm_rejects_sparse_labels(monkeypatch):
    """`labels_ahc` 必须是 `0..k-1` 的**连续**整数；稀疏下标必须响亮报错。

    合理回归：有人把前提断言删掉（"省一行"）—— 那时 `onehot` 的列数 k0 与实际
    最大下标对不上，`onehot[np.arange(n), labels]` 要么广播出错、要么**静默**
    少建一列，臂 1 的初值就悄悄错位。断言红。

    为什么必须有这条：生产侧 `build_unlocked` 的候选标签走
    `v11._relabel_by_first_seen`，本来就是 0..k-1；这个前提**当前由上游保证**，
    一旦上游口径变了，这里是唯一会开火的地方。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)
    sparse = np.array([0] * 20 + [3] * 20 + [7] * 20)   # 有空洞

    _capture_vb(monkeypatch)
    with pytest.raises(AssertionError) as exc:
        VB.arm_labels(x, plda, sparse, "vbx_ahc", fb=17.0, cap=4)
    assert "0..k-1" in str(exc.value), f"期望前提断言开火，实际：{exc.value!r}"


def test_unknown_arm_is_rejected(monkeypatch):
    """只认 `vbx_ahc` / `vbx_flat` 两臂；写错臂名必须抛 `ValueError`，不许静默走默认。

    合理回归：有人把 `else: raise ValueError` 改成 `else: gamma0 = ...`（"顺手加个
    兜底"）⇒ 臂名拼错时静默产出一份**看着正常**的标签，产物里的 arm 字段却与实际
    算法不符 —— 那是最难查的一类错。断言红。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)
    _capture_vb(monkeypatch)
    with pytest.raises(ValueError) as exc:
        VB.arm_labels(x, plda, np.zeros(len(x), dtype=int), "vbx_AHC", fb=17.0, cap=4)
    assert "vbx_AHC" in str(exc.value), f"错误信息要点名臂名，实际：{exc.value!r}"


def test_labels_are_relabelled_by_first_seen(monkeypatch):
    """硬标签必须是 `0..k-1`（按**首见顺序**），不是 `maxSpeakers` 空间里的稀疏下标。

    合理回归 A（靶心）：有人把 `_relabel_by_first_seen(gamma.argmax(axis=1))`
    改成裸 `gamma.argmax(axis=1)` —— **但仅在 gamma 的 argmax 本来就稀疏/乱序时
    才可见**。真实 VB 输出在多数合成数据上恰好已经是密的（实测），所以这里不靠
    真实 VB：`_vbx_stub` 回放一个 argmax = `[2,2,0,2,1,1,...]` 的 gamma，
    期望输出是 `[0,0,1,0,2,2,...]`。裸 argmax 会吐出含 `2` 的标签，第一条断言红。

    合理回归 B：有人把"首见顺序"改成别的稠密化规则（例如按簇大小排序）⇒ 逐位
    期望值红。这里钉的是逐位结果，不是"集合连续"这种弱性质。

    合理回归 C：`arm_labels` 返回的 dtype 变成 float（例如直接返回 `gamma.argmax`
    被后续运算污染）⇒ 第三条断言红。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)

    # 造 gamma：第 i 行的 argmax 恰好是 argmax_spec[i]
    argmax_spec = np.array([2, 2, 0, 2, 1, 1] * 10)
    gamma = _sentinel_gamma(argmax_spec, n_frames=60, n_cols=3)
    assert np.array_equal(gamma.argmax(axis=1), argmax_spec)   # 自检：桩子造对了

    _vbx_stub(gamma, monkeypatch)
    labels = VB.arm_labels(x, plda, np.zeros(60, dtype=int), "vbx_flat", fb=17.0, cap=5)

    expected = np.array([0, 0, 1, 0, 2, 2] * 10)
    assert np.array_equal(labels, expected), (
        "不是按首见顺序重编号：got %r ..." % (labels[:6].tolist(),))
    uniq = sorted(set(int(v) for v in labels))
    assert uniq == list(range(len(uniq))), f"标签不连续：{uniq}"
    assert np.issubdtype(labels.dtype, np.integer), f"dtype={labels.dtype}"


def test_dirichlet_init_is_shape_safe_and_seed_sensitive():
    """`_dirichlet_init` 的形状/归一化/seed 敏感性 —— **种子敏感**那半边是承重的。

    合理回归 A：`size=(n_frames, max_speakers)` 写成 `(max_speakers, n_frames)`
    或漏了 `max_speakers` ⇒ 形状断言红。

    合理回归 B：忘了 `/ gamma.sum(axis=1, keepdims=True)` ⇒ 行和断言红；
    写成 `/ gamma.sum(axis=0, keepdims=True)`（按列归一化）⇒ 同样红。

    合理回归 C：seed 被丢掉（写死常量、或改用全局 `np.random.gamma`）⇒
    `_dirichlet_init(20, 3, 1)` 与 `_dirichlet_init(20, 3, 2)` **相等** ⇒ 红。
    ⚠️ 这一条**只**证明 seed 有效果，**不**证明 `arm_labels` 用了它 ——
    后者由 `test_flat_arm_gamma_is_caller_supplied_with_the_fixed_seed` 钉住。
    两条都要有：少任何一条，都有一个变异能溜过去（已实测：seed 被写死时
    `arm_labels` 跑两次仍逐位相同）。
    """
    a = VB._dirichlet_init(20, 3, VB.SEED)
    assert a.shape == (20, 3), f"shape={a.shape}"
    assert np.allclose(a.sum(axis=1), 1.0), "行和必须是 1（Dirichlet 后验）"
    assert np.all(a > 0), "Dirichlet(1) 抽样必须全正"

    reverse = VB._dirichlet_init(3, 7, VB.SEED)
    assert reverse.shape == (3, 7), (
        f"shape={reverse.shape} —— n_frames 与 max_speakers 疑似写反")
    assert not np.array_equal(a, VB._dirichlet_init(20, 3, 2)), (
        "换 seed 抽样不变 ⇒ seed 没被用上（写死常量或走了全局 np.random）")
    # 归一化轴：按列归一化（axis=0）时行和不会是 1，且全 1 列不满足 >0 之外无约束
    assert not np.allclose(reverse.sum(axis=0), 1.0) or reverse.shape[0] == reverse.shape[1], (
        "疑似把归一化写成了 axis=0")


def test_arm_labels_result_comes_from_vb_not_from_a_shortcut(monkeypatch):
    """返回值必须**来自 `VB_diarization` 的输出** —— 抓"调了但把返回值丢掉"那种短路。

    ⚠️ 这条是复审（控制器裁决）点名补的，理由很硬：把臂改成提前 `return labels_ahc`
    有**两种形态**，`test_ahc_arm_feeds_the_labels_in_and_does_not_short_circuit`
    的 spy 契约只抓得到 (a)：

      (a) 在 `arm_labels` **开头**提前 return ⇒ spy 一次都没被调用 ⇒ 已覆盖；
      (b) **调用了** `VB_diarization`、`gamma` 参数也完全正确，但**把它的返回值
          丢掉**、仍然 `return labels_ahc` ⇒ 调用次数/入参全部正常 ⇒ **(a) 的契约
          抓不到**。

    做法：让替身回放一个 `argmax` 稀疏且乱序的哨兵 `gamma`
    （`argmax == [2,2,0,2,1,1,...]`）。那么
      - 正常实现 ⇒ `_relabel_by_first_seen` 把它变成 `[0,0,1,0,2,2,...]`；
      - 形态 (b) ⇒ 返回的是 `labels_ahc`（全 0），与期望的前 6 位就不同 ⇒ **红**；
      - 裸 `gamma.argmax`（不做重编号）⇒ 含 2，与期望不同 ⇒ 同样红。
    期望值**逐位**写死在测试里，不是从任何 `VB.*` 公式推出来的。

    合理回归：任何"VB 的结果没被用上"的短路（提前 return、丢返回值、返回入参）
    都红。
    """
    x, _truth = _synth_emb()
    plda = _synth_plda(x, lda_dim=4)

    argmax_spec = np.array([2, 2, 0, 2, 1, 1] * 10)
    sentinel = _sentinel_gamma(argmax_spec, n_frames=60, n_cols=3)
    assert np.array_equal(sentinel.argmax(axis=1), argmax_spec)   # 自检：哨兵造对了

    _vbx_stub(sentinel, monkeypatch)
    labels_ahc = np.zeros(60, dtype=int)      # 全 0 初值：形态 (b) 会原样吐回它
    labels = VB.arm_labels(x, plda, labels_ahc, "vbx_flat", fb=17.0, cap=3)

    expected = np.array([0, 0, 1, 0, 2, 2] * 10)
    assert not np.array_equal(labels, labels_ahc), (
        "返回值等于入参 labels_ahc ⇒ VB_diarization 的结果被丢掉了（形态 b）")
    assert np.array_equal(labels, expected), (
        "返回值不是由 gamma 的 argmax 按首见顺序推出的：got %r 期望 %r"
        % (labels[:6].tolist(), expected[:6].tolist()))
    assert np.array_equal(labels, VB._relabel_by_first_seen(sentinel.argmax(axis=1))), (
        "与 gamma.argmax 的首见重编号不一致")
