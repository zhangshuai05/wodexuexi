# -*- coding: utf-8 -*-
"""锁住 `probe_backend_sweep` 的四处口径修正（2026-10-10）。

这四处都不报错、只出错数 —— 与本仓库反复出现的同一类陷阱
（见 `test_probe_silent_failures.py` 的四个先例）：

1. 训练池含 holdout —— 从封存集里学，直接违反 spec §3.3
2. k 钉在真值有效人数 —— 用了被禁假设，把最贵的一环（选 K）整个摘掉
3. 跳过两级细化 —— 与生产不同构，测的只是链条中间一层
4. 只有 frame accuracy、无句级指标 —— 而验收指标是句级错答率
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(ROOT / "scripts" / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import probe_backend_sweep as BS  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

DATA = Path(r"D:\yinp\AISHELL-4\data\test")


def test_pool_excludes_holdout_when_split_is_dev():
    """池必须只含本 split 的场次 —— 否则封存集为一个输出数字做了贡献。"""
    # 数据/缓存是机器本地的绝对路径（D:\yinp\…），新克隆、CI、同事机器上不存在。
    # pytest.ini 的 testpaths=tests 让本文件进每一次全量跑，所以缺数据时必须
    # skip 而不是 error —— 否则整个包套件在别的机器上全红。
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:\\yinp\\…）")
    dev = split_meetings(DATA, "dev")
    holdout = split_meetings(DATA, "holdout")
    assert dev and holdout, "前置：两个 split 都应非空"
    assert not (set(dev) & set(holdout)), "前置：两者身份组已隔离"

    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    # 前置：缓存缺失/改名时池会是空的，两条断言随即恒真 —— 那正是本仓库
    # 反复出现的"不报错只出错数"，所以先钉住池非空。
    assert pool, f"前置：池非空（缓存 {BS.DEFAULT_CACHE} 读数是否正常？）"
    leaked = sorted(set(pool) & set(holdout))
    assert leaked == [], f"训练池泄漏了封存集场次：{leaked}"
    assert set(pool) <= set(dev), f"池含非本 split 场次：{sorted(set(pool) - set(dev))}"


def test_refinement_is_applied_and_level_recorded():
    """变体结果必须记录细化层级 —— 否则无法区分'后端在 AHC 层的差异'
    与'整条链路的差异'，而生产里 AHC 只是中间一环。"""
    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    if not pool:
        import pytest
        pytest.skip("无标注窗缓存")

    import numpy as np
    import probe_ab_dev as AB
    import probe_path_equiv as PE

    meeting = sorted(pool)[0]
    emb, times, speakers, votes = pool[meeting]
    rE, rtimes = PE.cached_stage(meeting, "refine")
    assert len(rE) > 0, "前置：refine 缓存应非空"

    labels = np.zeros(len(emb), dtype=int)          # k=1，最稳的输入
    fl, ft, level = AB.real_path_labels(labels, emb, times, rE, rtimes)
    assert level in ("refined", "2s-only")
    # k<2 时必须返回原 times（`real_path_labels` docstring 记录的坑：
    # 用 arange 重建网格会让 600s 参考的后半段全被判成无覆盖，弃权假报 37%）
    if level == "2s-only":
        assert np.allclose(ft, times), "k<2 时不得重建时间网格"


def test_refine_labels_off_and_on_are_wired():
    """`refine_labels` 是本任务新增的接线点，且 `--refine off` 必须不崩。

    旧测试只碰 `probe_ab_dev.real_path_labels`（本任务未改），k=1 时它走
    早退分支、两个断言恒真 —— 对本任务新增的代码零覆盖。
    本测试走两段：先单元层验两个分支，再真跑一次 `main` 验行里的 `level`。
    """
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:\\yinp\\…）")
    import numpy as np
    import probe_ab_dev as AB
    import probe_path_equiv as PE

    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    assert pool, "前置：标注窗缓存应非空"
    meeting = sorted(pool)[0]
    emb, times, _, _ = pool[meeting]
    rE, rtimes = PE.cached_stage(meeting, "refine")

    labels = np.zeros(len(emb), dtype=int)

    # off：原样返回 + level="none"（不能调 AB._win，它不认 "none"）
    off_l, off_t, off_level = BS.refine_labels(labels, emb, times, rE, rtimes, False)
    assert off_level == "none"
    assert off_t is times and off_l is labels, "off 分支必须原样返回，不得重建网格"

    # on：走 real_path_labels，level 只能是这两个
    _, _, on_level = BS.refine_labels(labels, emb, times, rE, rtimes, True)
    assert on_level in ("refined", "2s-only")

    # 细化窗与锚点必须在**同一空间** —— 这是本任务唯一会崩的地方，也是
    # brief 原文里唯一真正错的地方（原文直接传 rE，而 anchors 取自 LDA 空间）。
    #
    # ⚠️ 判据不能写成"`fit` 输出的列数与输入无关"。那是**恒真**的：`Transform`
    # 是 `(X - mean) @ weights`，输出宽度只由 `weights` 定（实测 14 == 14），
    # 在 brief 原文下也照样通过。唯一会红的形式是"目标调用不抛"。
    from probe_gap_decomp import cluster_labels
    fit, _ = BS.fit_backend(pool, meeting, 8, True)
    assert fit is not None, "前置：本场应能拟合"
    vectors, rvectors = fit(emb), fit(rE)
    assert rvectors.shape[1] == vectors.shape[1], (
        f"细化窗在错误空间（{rvectors.shape[1]} 列 vs 聚类空间 {vectors.shape[1]} 列）")
    labels_k = cluster_labels(vectors, 3, metric=fit.metric)
    # brief 原文下这里是 `rE @ anchors.T`：ValueError: matmul ... size 14 is
    # different from 192 —— 会 raise，不会 assert。
    fclusters, ftimes, level = BS.refine_labels(
        labels_k, vectors, times, rvectors, rtimes, True)
    assert level in ("refined", "2s-only"), "投影后的细化窗必须能被消化"
    assert len(ftimes) == len(rvectors), "细化网格应与细化窗一一对应"

    # ── 端到端：真跑一次 `main`，把同一件事钉在整条变体循环上 ──
    # 上面几段都只调函数；"level 落进产物行"这个验收点只有穿过 `main`
    # 才能验到，`fit is None` 存根行带不带 `level` 也只在这里暴露。
    import json
    import tempfile

    # 两场：目标场之外还剩一场可拟合 ⇒ LDA 臂真的会跑（单场时池为空、
    # 只走存根，就绕过了要测的那条路）。
    two = sorted(pool)[:2]
    assert len(two) == 2, "前置：至少两场"

    def _run(seconds, out_path):
        argv = sys.argv
        # `--k-modes` 显式传：本任务就是加这条轴，靠默认值会让"flag 没接上"
        # 与"flag 接上了"看起来一样。
        sys.argv = ["probe_backend_sweep.py", "--meetings", *two,
                    "--seconds", str(seconds), "--classes", "8", "--norm", "cosine",
                    "--k-modes", "true,auto", "--output", str(out_path)]
        try:
            BS.main()          # brief 原文下：这里抛 ValueError，测试红
        finally:
            sys.argv = argv
        return json.loads(out_path.read_text(encoding="utf-8"))["meetings"]

    with tempfile.TemporaryDirectory() as tmp:
        rows = _run(120, Path(tmp) / "sweep.json")
        # 第二轮：窗数极少的场次走"窗数不足"跳过路径。原来那条路直接
        # `continue` 丢整行，产物里"跳过了"与"本来就没这场"长得一模一样。
        skip_rows = _run(10, Path(tmp) / "skip.json")

        # ── 间谍臂：证明 `auto` 真的是**经生产选择器**选的 k ──
        # 上面那些断言只看产物。一个"名字/标签都对、但 auto 分支实际走
        # 欧氏 K 选择"的变异能全绿通过（已实测）—— 因为产物里看不出
        # 选择器有没有被调用。这里包一层，把选择器**返回的 k** 记下来，
        # 再和产物里记的 `k_used` 逐值对上：只数调用次数不够（true 臂
        # 若也调一次，次数照样相等），必须绑定"选择器吐出来的数"。
        returned = []
        real_build, real_pick = BS.build_unlocked, BS.pick_unlocked

        def spy_build(E, cap=BS.CAP, min_win=BS.MIN_WIN):
            return real_build(E, cap, min_win)

        def spy_pick(cands, mode="objective"):
            assert mode == "objective", f"生产默认 mode 应为 objective，收到 {mode}"
            chosen = real_pick(cands, mode=mode)
            returned.append(None if chosen is None else int(chosen["k"]))
            return chosen

        BS.build_unlocked, BS.pick_unlocked = spy_build, spy_pick
        try:
            spy_rows = _run(120, Path(tmp) / "spy.json")
        finally:
            BS.build_unlocked, BS.pick_unlocked = real_build, real_pick

    assert returned, "auto 臂根本没调用生产选择器（pick_unlocked 一次没进）"
    # 产物里每个出了数的 `_autok` 变体，其 k_used 必须来自上面的选择器返回。
    product = [v["k_used"] for row in spy_rows for name, v in row["variants"].items()
               if name.endswith("_autok") and v["accuracy"] is not None]
    assert product, "auto 臂没有产出任何可用变体行"
    assert sorted(product) == sorted(returned), (
        f"产物记的 k_used {sorted(product)} 与生产选择器实际返回的 {sorted(returned)} "
        "对不上 —— auto 臂没走选择器（或 true 臂也在调它）")

    assert len(rows) == 2, f"两场都应出行：{len(rows)}"
    assert len(skip_rows) == 2, f"跳过路径也必须出行（不得丢整行）：{len(skip_rows)}"

    for label, run_rows in (("正常", rows), ("跳过", skip_rows)):
        for row in run_rows:
            assert row["variants"], f"{label}/{row['meeting']} 无变体行"
            assert "k_modes" in row, f"{label}/{row['meeting']} 缺 k_modes"
            assert not [n for n in row["variants"] if n.endswith("_euc_autok")], \
                f"{label}/{row['meeting']} 欧氏变体混进了自选 k 臂"
            for name, v in row["variants"].items():
                assert "level" in v, f"{label}/{row['meeting']}/{name} 缺 level 键"
                assert "k_mode" in v, f"{label}/{row['meeting']}/{name} 缺 k_mode 键"
                assert "k_used" in v, f"{label}/{row['meeting']}/{name} 缺 k_used 键"
                # 名字与 k_mode 必须一致 —— 钉住"名字枚举对了但臂接错了"。
                assert v["k_mode"] == ("auto" if name.endswith("_autok") else "true"), \
                    f"{label}/{row['meeting']}/{name} k_mode 与变体名不符：{v['k_mode']}"
                if v["accuracy"] is not None:
                    assert v["level"] in ("refined", "2s-only", "none"), v["level"]
                    assert v["k_mode"] == "auto" or v["k_used"] is not None, name
    # 跳过轮的每一行都必须是存根：没有数、但**有**键。
    for row in skip_rows:
        for name, v in row["variants"].items():
            assert v["accuracy"] is None, f"{row['meeting']}/{name} 窗数不足却出了数"
            assert v["k_used"] is None, f"{row['meeting']}/{name} 未拟合却有 k_used"


def test_euclid_variants_skip_the_auto_k_arm():
    """`build_unlocked` 内部写死余弦（unlocked_selector.py:48）。

    若让欧氏空间的 LDA 输出经它选 k，就是拿余弦 silhouette 去评价欧氏空间
    —— 第四个"换了东西没隔离变量"。所以 _euc 变体必须跳过自选 k 臂，
    而不是悄悄用一个不成立的度量。
    """
    variants = BS.variants_for([8], [True, False], k_modes=["true", "auto"])
    assert "c8_cos_autok" in variants
    assert "c8_euc_autok" not in variants, "欧氏变体不得进入自选 k 臂"
    assert "c8_euc" in variants, "欧氏变体本身仍要跑（真值 k 臂）"
    assert "none_autok" in variants, "裸基线也要有自选 k 臂"


def test_sentence_metrics_are_reported_for_every_variant():
    """验收指标是句级归属率，而 frame accuracy 对"多出来的簇"记账方式不同：
    frame 走贪心映射、句级走匈牙利双射，两者不能互相换算。

    必须报**两个数**：错答（静默错误，读者被误导而不自知）与弃权（显式缺失，
    读者看得到）。`1 - per_sentence` **不是**错答率 —— clean 桶含弃权句且
    算未答对，混在里面的那一份必须单列出来。
    """
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip("AISHELL-4 数据/缓存不在本机（绝对路径 D:/yinp/…）")
    import json
    import tempfile

    import probe_ab_dev as AB
    import probe_path_equiv as PE
    import claude_a2_v12 as v12
    from probe_gap_decomp import cluster_labels

    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    assert pool, "前置：标注窗缓存应非空"

    # ── 单元层：与 `AB.score` 走同一条取桶路径（段表 → 句预测 → attribution）──
    # ⚠️ 输入必须是**真跑出来的** fclusters/ftimes/fwin，不能拿
    # `zeros(len(sentences))` + `arange` 合成网格充数：合成网格只验证桶的搬运，
    # 对"接的是不是真实窗网格"零覆盖 —— 而后者才是本任务的全部内容。
    meeting = sorted(pool)[0]
    emb, times, _, _ = pool[meeting]
    rE, rtimes = PE.cached_stage(meeting, "refine")
    fit, _ = BS.fit_backend(pool, meeting, 8, True)
    assert fit is not None, "前置：本场应能拟合（否则 LDA 臂只出行存根，测不到）"
    vectors, rvectors = fit(emb), fit(rE)
    fclusters, ftimes, level = BS.refine_labels(
        cluster_labels(vectors, 3, metric=fit.metric), vectors, times,
        rvectors, rtimes, True)
    assert level in ("refined", "2s-only")
    fwin = v12.WIN_S if level == "none" else AB._win(level)

    sentences = AB.load_sentences(meeting)
    assert sentences, "前置：本场应有句子（TextGrid）"

    # 这一条是"重建 score 取桶路径是否走样"的唯一判据。注意它**只**在
    # 真输入上有效：`attribution` 的分桶/映射都与 segments 的形状有关。
    s = BS.sentence_metrics(sentences, fclusters, ftimes, fwin)
    base = AB.score(sentences, fclusters, ftimes, fwin)
    assert set(s) == {"overall", "clean", "overlap"}, sorted(s)
    assert s["overall"]["per_sentence"] == base["per_sentence"], (
        "sentence_metrics 的 overall 必须与 AB.score 同口径（重建路径走样了）")
    assert s["overall"]["n"] == base["n"]
    assert s["overall"]["abstain"] == base["abstain"]

    for bucket, b in s.items():
        assert set(b) == {"per_sentence", "wrong", "unmapped", "abstain",
                          "attribution_error", "n"}, (bucket, sorted(b))
        if b["per_sentence"] is None:
            assert b["abstain"] is None and b["wrong"] is None, bucket
            assert b["unmapped"] is None and b["attribution_error"] is None, bucket
            continue
        # 四类**恰好加满 1**。这条把"`unmapped` 被折进 `wrong`"的回归钉死：
        # 折回去（`wrong = 1 - ps - ab`）会让和式多出 `unmapped` 那一份。
        assert abs(b["per_sentence"] + b["wrong"] + b["unmapped"] + b["abstain"] - 1.0) < 1e-9, \
            f"{bucket} 四类之和不为 1：{b}"
        assert b["abstain"] >= 0.0 and b["wrong"] >= -1e-9 and b["unmapped"] >= -1e-9, (bucket, b)
        # 产品口径 = 混人 + 未映射，且**恰好等于** `1 - 答对 - 弃权` ——
        # 拆字段不得改变对外那个数（拆开只加诊断分辨率，不放松验收）。
        assert abs(b["attribution_error"] - (1.0 - b["per_sentence"] - b["abstain"])) < 1e-9, \
            f"{bucket} 归属错误率被拆字段改变了：{b}"
        assert b["attribution_error"] >= b["wrong"] - 1e-9, (bucket, b)

    clean = s["clean"]
    assert clean["n"] > 0, "前置：本场应有 clean 桶句子"
    if clean["wrong"] > 0:
        # 混人 >0 时 `1 - per_sentence` 与错答率**必须**不同 —— 这条把
        # "四类分开" 与 "一个数" 钉开（少了 `- abstain - unmapped` 就在这里红）。
        assert abs(clean["wrong"] - (1.0 - clean["per_sentence"])) > 1e-9, clean
    assert clean["n"] + s["overlap"]["n"] == s["overall"]["n"], (
        "clean/overlap 必须恰好把 overall 切开")

    # ── 端到端：`sentence` 键必须落进**每一行**，且没有哪条臂豁免 ──
    # 只单测 `sentence_metrics` 是不够的：把变体行里那个 `"sentence"` 整段删掉，
    # 上面全部断言照样全绿。所以这里真跑一次 `main`。
    # 另外把 `AB.DATA` 换成一个不存在的路径再跑：`main` 若没设
    # `AB.DATA = root`，`load_sentences` 就会抛 FileNotFoundError —— 这条
    # 同时钉住 brief 的"`AB.DATA` 必须指向 `--root`"。
    two = sorted(pool)[:2]
    assert len(two) == 2, "前置：至少两场（单场时池为空、LDA 臂只出行存根）"
    saved_data = AB.DATA

    def _run(seconds, out_path):
        argv = sys.argv
        sys.argv = ["probe_backend_sweep.py", "--root", str(DATA),
                    "--meetings", *two, "--seconds", str(seconds),
                    "--classes", "8", "--norm", "cosine",
                    "--k-modes", "true,auto", "--output", str(out_path)]
        try:
            BS.main()          # AB.DATA 未接上 --root 时：这里抛 FileNotFoundError
        finally:
            sys.argv = argv
        return json.loads(out_path.read_text(encoding="utf-8"))["meetings"]

    AB.DATA = Path(r"D:\__no_such_data_root__")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rows = _run(120, Path(tmp) / "sent.json")
            skip_rows = _run(10, Path(tmp) / "skip.json")
    finally:
        AB.DATA = saved_data

    assert len(rows) == 2, f"两场都应出行：{len(rows)}"
    arm_seen = {}
    for label, run_rows in (("正常", rows), ("跳过", skip_rows)):
        for row in run_rows:
            assert row["variants"], f"{label}/{row['meeting']} 无变体行"
            for name, v in row["variants"].items():
                assert "sentence" in v, f"{label}/{row['meeting']}/{name} 缺 sentence 键"
                if v["accuracy"] is None:
                    # 存根行：不许出数，但键必须在（缺键会让"这步没走"与
                    # "这段代码没写"分不开 —— 与 `level` 同一条理由）。
                    assert v["sentence"] is None, \
                        f"{label}/{row['meeting']}/{name} 未出 frame 数却有句级数"
                    continue
                sn = v["sentence"]
                assert sn is not None, f"{label}/{row['meeting']}/{name} 有 frame 数却无句级数"
                assert set(sn) == {"overall", "clean", "overlap"}, (name, sorted(sn))
                for bucket, b in sn.items():
                    assert set(b) == {"per_sentence", "wrong", "unmapped", "abstain",
                                      "attribution_error", "n"}, (name, bucket)
                    if b["per_sentence"] is None:
                        continue
                    # 产物侧也验四类加满 1 —— 单元层对了但落盘掉了 `unmapped`
                    # 这类回归，只有穿过 `main` 才看得见。
                    assert abs(b["per_sentence"] + b["wrong"] + b["unmapped"]
                               + b["abstain"] - 1.0) < 1e-9, (name, bucket, b)
                arm_seen.setdefault(name, 0)
                arm_seen[name] += 1

    # `none` 臂与 LDA 臂都必须拿到句级数 —— 同一件事，没有哪条臂豁免。
    assert arm_seen.get("none"), f"裸臂没出句级数：{sorted(arm_seen)}"
    assert any(n.startswith("c8_") for n in arm_seen), \
        f"LDA 臂没出句级数（存根行不算）：{sorted(arm_seen)}"
    # 跳过轮的每一行都必须是存根（顺带钉住汇总打印不因 sentence=None 而崩）。
    for row in skip_rows:
        for name, v in row["variants"].items():
            assert v["accuracy"] is None, f"{row['meeting']}/{name} 窗数不足却出了数"
            assert v["sentence"] is None, f"{row['meeting']}/{name} 未跑却有句级数"


def test_stub_rows_share_the_success_schema():
    """失败行与成功行必须是**同一种形状**（键集只差 `skip`）。

    产物是按行读的：`{"classes": null}` 与 `{"classes" 不存在}` 在 `json.loads`
    之后分不开 —— 缺键会让下游必须写 `.get("classes")`，而 `.get` 把
    "键不存在"和"值为 None"混成一个结果。此前 `classes` 只在成功路径上有，
    三条失败路径各写各的字典，改一处忘两处正是这类 schema 漂移的成因。

    红验证：把 `stub_row` 里的 `"classes": None` 删掉即红。
    """
    skip_keys = {"skip"}
    stub_name, stub = "c8_cos", BS.stub_row("c8_cos", "true")
    assert "skip" not in stub, "无 skip 理由时不应凭空多出 skip 键"
    assert BS.stub_row("c8_cos", "auto", "x")["skip"] == "x"

    # 成功行的键集：与 `probe_backend_sweep.main` 里那条赋值逐字对齐。
    success_keys = {"accuracy", "classes", "level", "n_final", "n_classes",
                    "k_mode", "k_used", "sentence"}
    assert set(stub) == success_keys, (
        f"存根行键集与成功行不一致：多 {sorted(set(stub) - success_keys)}，"
        f"少 {sorted(success_keys - set(stub))}")

    # 真产物上核一遍：跳过轮（--seconds 10 ⇒ 窗数不足）每行都必须是这种形状。
    if not DATA.is_dir() or not Path(BS.DEFAULT_CACHE).is_dir():
        pytest.skip(r"AISHELL-4 数据/缓存不在本机（绝对路径 D:\yinp\…）")
    import json
    import tempfile

    dev = split_meetings(DATA, "dev")
    pool = BS.build_pool(dev, BS.DEFAULT_CACHE, 2.0, 1.0)
    assert pool, "前置：标注窗缓存应非空"
    two = sorted(pool)[:2]
    assert len(two) == 2, "前置：至少两场"
    argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = ["probe_backend_sweep.py", "--meetings", *two, "--seconds", "10",
                    "--classes", "8", "--norm", "cosine", "--k-modes", "true,auto",
                    "--output", str(Path(tmp) / "stub.json")]
        try:
            BS.main()
        finally:
            sys.argv = argv
        rows = json.loads((Path(tmp) / "stub.json").read_text(encoding="utf-8"))["meetings"]
    assert rows, "跳过轮也必须落行（不得静默丢场次）"
    for row in rows:
        for name, v in row["variants"].items():
            assert v["accuracy"] is None, f"{row['meeting']}/{name} 窗数不足却出了数"
            assert "skip" in v, f"{row['meeting']}/{name} 跳过却没写理由（与成功行分不开）"
            assert set(v) == success_keys | {"skip"}, \
                f"{row['meeting']}/{name} 键集 {sorted(set(v))} ≠ 成功行 + skip"
