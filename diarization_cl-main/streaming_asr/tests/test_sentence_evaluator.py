# -*- coding: utf-8 -*-
"""评价器校验（纯函数，不读缓存、不加载模型）。

## 为什么先做这个

`probe_sentence_attribution.py` 已经产出过"12/12 场分离""r=+0.89""1.5 倍放大"
这类结论。**若尺子本身坏了，这些数全是假的**，跑 70 分钟实验只是浪费。
所以最便宜、最高价值的检验是确认尺子没坏。

## 四条必查（项目负责人指定）

1. **标签重编号不变性** —— 把预测簇号整体置换，分数必须逐位不变。
   匈牙利映射若真在做全局最优，这条自动成立；不成立说明映射有 bug。
2. **完美预测得满分** —— 构造一个"每句都归给自己锚说话人"的预测。
   ⚠️ **注意**：这条**只在非重叠句上要求满分**。重叠句上没有任何单标签
   预测能同时满足两个说话人（`read_textgrid` 每句只给一个锚），
   所以整体满分**不是恒真期望**，是非重叠部分的期望。
3. **全漏预测得零分** —— 预测完全无覆盖 → 弃权 100%，且**分母为 0**。
   必须报"未定义"，不能悄悄报 0（那会与"预测全错"混为一谈）。
4. **一个簇覆盖多人 → 合理的多数类基线** —— k=2 把两个真说话人并进一个簇，
   正确率应≈被并两人的时长占比，**不是 100% 也不是 0%**。
   这条锁住"并人必须扣分"。

    from tests.test_sentence_evaluator import *   # pytest 直接收
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from probe_sentence_attribution import attribution  # noqa: E402


def sent(start, end, speaker, overlapped=False):
    return {"start": start, "end": end, "speaker": speaker, "text": "x",
            "other_cov": 0.4 if overlapped else 0.0, "overlapped": overlapped}


def test_relabelling_invariance():
    """预测标签整体置换，分数逐位不变。"""
    sentences = [sent(0, 2, "A"), sent(2, 4, "B"), sent(4, 6, "A"), sent(6, 8, "C")]
    base = attribution(sentences, ["p", "q", "p", "r"])
    swapped = attribution(sentences, ["q", "p", "q", "r"])   # p/q 对调
    assert base["overall"]["duration_weighted"] == swapped["overall"]["duration_weighted"]
    assert base["overall"]["per_sentence"] == swapped["overall"]["per_sentence"]


def test_perfect_non_overlapping_scores_one():
    """非重叠句全对 → 满分。"""
    sentences = [sent(0, 2, "A"), sent(2, 4, "B"), sent(4, 6, "C")]
    r = attribution(sentences, ["A", "B", "C"])
    assert r["overall"]["duration_weighted"] == 1.0
    assert r["overall"]["per_sentence"] == 1.0
    assert r["overall"]["abstain"] == 0.0


def test_unmapped_rate_is_distinct_from_abstain_rate():
    """"有预测但标签未被匈牙利映射使用"与"无预测覆盖"是两个不同的量，必须分开报。

    项目负责人明确：混在一起会扭曲指标解释。
      * 弃权（abstain）   = 该句**没有任何预测**覆盖
      * 未匹配（unmapped）= 该句**有预测**，但那个标签没被匈牙利映射到任何名字
        （映射容量不足 / 簇分裂造成；**不是**弃权）

    构造：真值只有 2 个说话人（A、B），预测却有 4 个簇号（p/q/r/s）。
    匈牙利最多映射 2 个 → 剩下 2 个必然未匹配。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "A"), sent(4, 6, "B"), sent(6, 8, "B")]
    r = attribution(sentences, ["p", "q", "r", "s"])["overall"]
    assert r["abstain"] == 0.0, "所有句都有预测，弃权应为 0"
    assert r["n"] == 4
    assert r["unmapped"] > 0.0, "未匹配的标签必须单列（此处应为 0.5）"
    assert abs(r["unmapped"] - 0.5) < 1e-9


def test_fully_mapped_prediction_has_zero_unmapped():
    """每个预测标签都能映射到名字时，未匹配率必须为 0。"""
    sentences = [sent(0, 2, "A"), sent(2, 4, "B")]
    r = attribution(sentences, ["p", "q"])["overall"]
    assert r["unmapped"] == 0.0
    assert r["abstain"] == 0.0


def test_abstain_and_unmapped_are_not_conflated():
    """一弃权 + 一未匹配：两个数必须分别体现，不能合并。

    构造要**保证**未匹配：真值只有 2 个说话人，预测有 3 个簇号，
    匈牙利最多映射 2 个 → 必有一个未匹配。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "A"), sent(4, 6, "B"), sent(6, 8, "B")]
    r = attribution(sentences, [None, "p", "q", "s"])["overall"]
    assert r["abstain"] == 0.25, "第 1 句无预测 → 弃权 0.25"
    assert r["unmapped"] == 0.25, "3 个预测名 vs 2 个真名 → 恰有一个未匹配"
    assert abs(r["abstain"] + r["unmapped"] - 0.5) < 1e-9


def test_unmapped_sentences_stay_in_denominator_as_wrong():
    """未匹配的句子**留在分母里算未答对**，不得被排除（§R1 口径）。

    排除它们会重新奖励"多切簇、赌一些簇能撞上"——切得越细越容易有簇未被映射，
    若不扣分，碎簇就变成免费的。

    ⚠️ 不猜具体数值：匈牙利做**全局最大化**，未被映射的簇会被指到"残留名字"，
    得分取决于它撞上谁（本构造里 `s` 撞上 A，故 3/4 而非 2/4）。
    本测试只锁**性质**：分母是全部句子；未匹配句不在分子里。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "A"), sent(4, 6, "B"), sent(6, 8, "B")]
    r = attribution(sentences, ["p", "q", "s", "s"])["overall"]
    assert r["n"] == 4, "全部句子都要在分母里"
    assert r["unmapped"] > 0.0, "本构造必须产生未匹配"
    # 关键性质：至少有一句被扣分（正确率严格小于 1），且不是靠弃权扣的
    assert r["abstain"] == 0.0, "每句都有预测，不该有弃权"
    assert r["per_sentence"] < 1.0, f"未匹配句未被扣分：{r}"


def test_pred_length_mismatch_raises_readable_error():
    """`pred` 与 `sentences` 长度不一致 → 可读的 ValueError，不是 IndexError。

    回归：调试时给 4 句传了 3 个预测，报的是难以定位的
    `IndexError: list index out of range`，指向函数内部而非调用处。
    """
    import pytest as _pytest
    sentences = [sent(0, 2, "A"), sent(2, 4, "B")]
    with _pytest.raises(ValueError, match="长度"):
        attribution(sentences, ["p"])


def test_no_coverage_scores_zero_and_abstain_one():
    """预测全无覆盖 → 弃权 1.0，正确率 0.0。

    ⚠️ 语义更正：正确率报 **0.0**（弃权 = 未答对），**不是 None**。
    早先版本把无覆盖句移出分母并报 None，那会让"只答 1 句且答对"得 100%
    （真实 bug，见 test_abstaining_on_almost_everything_cannot_score_high）。
    弃权率单列，用来区分"全弃权"与"全答错"这两种不同的失败。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "B")]
    r = attribution(sentences, [None, None])
    assert r["overall"]["abstain"] == 1.0
    assert r["overall"]["duration_weighted"] == 0.0
    assert r["overall"]["per_sentence"] == 0.0


def test_merged_cluster_gives_majority_class_baseline():
    """k=2 把两个真说话人并进一个簇 → 正确率≈被并两人占多数者的占比。

    A 占 50s、B 占 30s 被并成一个簇，C 单独一个簇。
    → 该簇只能被映射到一个名字，最多救回 A（50s），B 全错。
    期望：加权正确率 = 70/100，按句 = 2/3。**不是 100% 也不是 0%**。
    """
    sentences = [sent(0, 50, "A"), sent(50, 80, "B"), sent(80, 100, "C")]
    r = attribution(sentences, ["merged", "merged", "c"])
    # merged→A 映射后：A 对(50s)，B 错，C 对(20s) → 70/100
    assert abs(r["overall"]["duration_weighted"] - 0.70) < 1e-9
    # 按句：A 对、B 错、C 对 → 2/3（round 到 4 位小数）
    assert abs(r["overall"]["per_sentence"] - 2 / 3) < 1e-3


def test_abstaining_on_almost_everything_cannot_score_high():
    """只覆盖 1 句、对其余全部弃权 → **绝不能报 100%**。

    这是真实 bug 的回归：k=2 时碎簇只剩 1 窗，102 句里 101 句无覆盖，
    旧实现把无覆盖句排除出分母，于是 1/1 = 100%。
    "几乎不输出"必须是低分，不能是满分。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "B"), sent(4, 6, "C"), sent(6, 8, "D")]
    r = attribution(sentences, ["A", None, None, None])
    assert r["overall"]["abstain"] == 0.75
    # 弃权按"未答对"计入分母 → 1/4
    assert abs(r["overall"]["per_sentence"] - 0.25) < 1e-9
    assert abs(r["overall"]["duration_weighted"] - 0.25) < 1e-9


def test_zero_label_scores_identically_to_renamed_all_one_label():
    """全 0 标签的一簇预测，必须与改名为全 1 的同一预测得**完全相同**的分数。

    回归：`segment_map` / `frame_map` 里写的是 `if nm`，
    而整数 `0` 在 Python 里是 falsy —— 于是 0 号簇被整个丢掉，
    k=1（`clusters_for` 返回全 0）或 k≥2 时带 0 号簇的预测全被判成"无覆盖"。
    这条用"改名不变性"把它锁死：簇号是任意的，0 号不该有特殊待遇。
    """
    from probe_sentence_attribution import (segment_map, sentence_dict_predictions)

    sentences = [sent(0, 2, "A"), sent(2, 4, "B"), sent(4, 6, "A")]
    # 同一份预测，簇号整体 +1：[[0,0,0]] vs [[1,1,1]]
    zero = sentence_dict_predictions(sentences, segment_map([0.0], [0], [6.0], 6.0))
    one = sentence_dict_predictions(sentences, segment_map([0.0], [1], [6.0], 6.0))
    assert list(zero) == ["0", "0", "0"], f"0 号标签未穿透: {list(zero)}"
    assert list(one) == ["1", "1", "1"]

    # 名字不同没关系；**分数必须相同**（簇号是任意的）
    r0 = attribution(sentences, zero)["overall"]
    r1 = attribution(sentences, one)["overall"]
    assert r0 == r1, f"0 号簇被区别对待：{r0} vs {r1}"
    assert r0["abstain"] == 0.0, "全 0 标签不得被判成无覆盖"


def test_segment_map_keeps_zero_label():
    """`segment_map` 直接测：标签 0 必须产出段（不是空表）。"""
    from probe_sentence_attribution import segment_map
    assert segment_map([0.0], [0], [2.0], 2.0) == [(0.0, 2.0, "0")]
    assert segment_map([0.0], [1], [2.0], 2.0) == [(0.0, 2.0, "1")]


def test_frame_map_keeps_zero_label():
    """`frame_map` 同病同测：标签 0 必须铺进逐帧数组。"""
    from probe_sentence_attribution import frame_map
    grid = frame_map([0.0], [0], [1.0], 1.0)
    assert (grid == "0").sum() > 0, "0 号标签没有铺进网格"


def test_empty_label_still_dropped():
    """显式的空标签（None / ""）仍应被丢弃——修 0 号不能把空标签也放进来。"""
    from probe_sentence_attribution import segment_map
    assert segment_map([0.0, 1.0], [None, ""], [1.0, 1.0], 2.0) == []


def test_hungarian_map_accepts_integer_zero_label():
    """`hungarian_map` 直接收整数标签时，0 号簇也必须参与映射。

    回归：内部写的是 `if p and t`，整数 `0` 是 falsy → 0 号簇被整个排除出
    混淆矩阵，映射不到名字。`segment_map` 会把标签转成字符串所以主链路碰不到，
    但 `hungarian_map` 是公开函数，调用方直接传 `clusters_for()` 的 numpy 数组
    就会踩中。与 `_name()` 的警告是同一类陷阱。
    """
    from probe_sentence_attribution import hungarian_map
    pred = [0, 0, 1, 1]          # 整数标签，0 号簇
    truth = ["A", "A", "B", "B"]
    weights = [2.0, 2.0, 2.0, 2.0]
    m = hungarian_map(pred, truth, weights)
    assert "0" in m, f"0 号簇未被映射：{m}"
    assert m["0"] == "A" and m["1"] == "B"


def test_empty_result_has_same_schema_as_nonempty():
    """空输入的返回 schema 必须与非空一致，否则调用方 KeyError。

    回归：空路径少了 `n_abstain`。
    """
    empty = attribution([], [])
    nonempty = attribution([sent(0, 2, "A")], ["p"])
    assert set(empty) == set(nonempty), (
        f"schema 不一致：空缺 {set(nonempty) - set(empty)}，"
        f"非空缺 {set(empty) - set(nonempty)}"
    )
    for group in ("overall", "overlap", "clean", "short"):
        assert set(empty[group]) == set(nonempty[group]), f"{group} 栏 schema 不一致"


def test_top_level_abstain_count_matches_group_counts():
    """顶层 `n_abstain` 必须与 `overall.abstain × n` 一致（同一份定义）。

    回归：顶层用 `if not x` 数 `mapped` 为空，把"未匹配"也计入弃权，
    而 `split()` 内部把弃权与未匹配分开算 —— 同一份数据两处给出不同的弃权数。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "A"), sent(4, 6, "B"), sent(6, 8, "B")]
    r = attribution(sentences, [None, "p", "q", "s"])   # 1 弃权 + 1 未匹配
    o = r["overall"]
    assert o["abstain"] == 0.25
    assert o["unmapped"] == 0.25
    assert r["n_abstain"] == round(o["abstain"] * o["n"]), (
        f"顶层 n_abstain={r['n_abstain']} 与 overall.abstain×n={o['abstain']*o['n']} 不一致"
    )
    assert r["n_abstain"] == 1, "弃权应只数真弃权（无预测），不含未匹配"


def test_integer_zero_labels_survive_full_evaluation_chain():
    """整数 0 号簇走**完整链**：clusters_for 风格的全 0 数组 → 段表 → 归属率。

    这是"整数 0"陷阱的端到端回归。`segment_map` 与 `hungarian_map`
    各自都有过 `if nm` / `if p and t` 的 falsy 过滤，单测各锁一次；
    这条锁**串起来**的结果：k=1（全 0 簇）不得被当成"全弃权"。
    """
    from probe_sentence_attribution import segment_map, sentence_dict_predictions
    import numpy as np

    sentences = [sent(0, 2, "A"), sent(2, 4, "A"), sent(4, 6, "B")]
    labels = np.zeros(3, dtype=int)          # clusters_for(k=1) 的真实形态
    segs = segment_map([0.0, 2.0, 4.0], labels, [2.0, 2.0, 2.0], 6.0)
    assert len(segs) == 3, f"全 0 簇被丢弃：{segs}"
    pred = sentence_dict_predictions(sentences, segs)
    r = attribution(sentences, pred)["overall"]
    # 一个簇映射到一个名字 → 最多救回占多数的那位；但**绝不能**全是弃权
    assert r["abstain"] == 0.0, f"全 0 簇被判成弃权：{r}"


def test_pick_best_k_survives_json_roundtrip_keys():
    """JSON 往返后 `sizes` 的键变字符串，选 k 不得静默丢掉碎簇惩罚。

    真实场景：sweep 结果写进 artifacts/*.json 再读回来，
    `{k: sizes}` 的 k 从 int 变 str，`sizes.get(k, [])` 会全部落空，
    碎簇数恒为 0 —— 惩罚静默失效（不是崩溃，是静默）。
    """
    from sentence_k_rule import pick_best_k
    acc_int = {6: 0.90, 9: 0.90}
    sizes_int = {6: [200, 150, 120, 100, 80, 60],
                 9: [200, 150, 120, 100, 80, 60, 3, 2, 2]}
    # 模拟 json.dumps/loads 往返
    import json
    acc_str = {str(k): v for k, v in acc_int.items()}
    sizes_str = {str(k): v for k, v in sizes_int.items()}
    a = pick_best_k(acc_int, sizes_int)
    b = pick_best_k(acc_str, sizes_str)
    assert a == 6, "整数键下应选健康簇 k=6"
    assert b == a, f"JSON 往返改变了选 k：{a} vs {b}（碎簇惩罚被静默丢弃）"


def test_attribution_accepts_integer_zero_without_counting_it_as_abstain():
    """直接给 `attribution` 整数标签时，0 号簇不得被算成弃权。

    完整流水线（`segment_map` → `sentence_dict_predictions`）会把标签转成字符串，
    所以主链路碰不到；但 `attribution` 是公开函数，且 `is_abstain` 写的是
    `not pred[j]` —— 整数 `0` 是 falsy，会被误判成"无预测"。
    与 `_name()` / `hungarian_map` 是**同一个陷阱的第三处**。
    """
    sentences = [sent(0, 2, "A"), sent(2, 4, "B")]
    r = attribution(sentences, [0, 1])["overall"]
    assert r["abstain"] == 0.0, f"整数 0 被算成弃权：{r}"
    assert r["per_sentence"] == 1.0


def test_is_abstain_treats_none_and_empty_string_only():
    """`is_abstain` 的判据是"显式无预测"，不是"falsy"。"""
    from probe_sentence_attribution import attribution as _att
    # 混合：None 与 "" 是弃权；整数 0 不是
    r = _att([sent(0, 2, "A"), sent(2, 4, "B"), sent(4, 6, "C")], [None, "", 0])["overall"]
    # 实现把比率 round 到 4 位，容差要与之匹配（1e-6 是假失败）
    assert abs(r["abstain"] - 2 / 3) < 1e-3, f"弃权应只数 None 与空串：{r}"


def test_abstain_rate_is_reported_separately_from_accuracy():
    """弃权率单列，但**不**把弃权句移出正确率分母。"""

    """弃权率单列，但**不**把弃权句移出正确率分母。"""
    sentences = [sent(0, 2, "A"), sent(2, 4, "B")]
    r = attribution(sentences, ["A", None])
    assert r["overall"]["abstain"] == 0.5
    assert abs(r["overall"]["per_sentence"] - 0.5) < 1e-9   # 1 对 / 2 句


def test_sub_min_sentence_is_scored_but_listed_separately():
    """<0.5s 的句子仍计入 overall（项目负责人要求），并单列 short 一栏。"""
    sentences = [sent(0, 5, "A"), sent(5, 5.2, "B")]
    r = attribution(sentences, ["A", "B"])
    assert r["overall"]["n"] == 2, "短句必须计入 overall"
    assert "short" in r, "必须单列短句一栏"
    assert r["short"]["n"] == 1
    assert r["n_sentences_scored"] == 2


def test_overall_uses_all_sentences_regardless_of_length():
    """`attribution` 不按长度剔除：总体就是全部句子。"""
    sentences = [sent(0, 5, "A"), sent(5, 5.1, "B"), sent(5.1, 10, "A")]
    r = attribution(sentences, ["A", "B", "A"])
    assert r["overall"]["n"] == 3
    assert r["short"]["n"] == 1
