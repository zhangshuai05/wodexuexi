# -*- coding: utf-8 -*-
"""句子级归属率当 K 判据时的**选 k** 规则（唯一实现，纯函数，无推理依赖）。

## 为什么需要它

`probe_sentence_attribution.py` 直接取归属率 argmax 时，实测系统性偏高
（dev 12 场里 9 场偏高；代理均值 5.7 vs 真实路径 3.2）。机制不是阈值，
是**代理缺碎簇惩罚**：

  代理跑在 2.0s 粗窗、**没有两级细化**。粗窗下把簇切细几乎总能提高句子归属率
  （每个句子只需匹配上某个碎片的多数票），碎簇零成本。而真实路径的细化阶段
  会用碎簇当锚点重判边界，反而把大片区域判给错的锚点——§4.1f 实测
  `L_R004S01C01` k=8 时细化把 22.5% 推到 19.7%，两腿差 9.1 点。

`largest_eligible` 当年是同一个死法（离线第一、真实路径半数回退，§4.1e）。

## 规则

在归属率上减掉碎簇惩罚，再取最大：

    分数(k) = 句子归属率(k) − FRAGMENT_PENALTY × 碎簇数(k)

碎簇 = 成员数 < `MIN_HEALTHY` 的簇。平局取**小 k**。
这样"碎簇要付代价"与"真优势要能赢"同时成立（两条都由测试锁住）。

    from sentence_k_rule import pick_best_k
"""
FRAGMENT_PENALTY = 0.01   # 每个碎簇的扣分。见下方"标定"说明。
MIN_HEALTHY = 4           # 与 v11.MIN_CLUSTER_WINDOWS 同口径


def pick_best_k(accuracy, sizes, penalty=FRAGMENT_PENALTY, min_healthy=MIN_HEALTHY):
    """在 {k: 归属率} 上选 k，碎簇按 `penalty` 逐个扣分。平局取小 k。

    **键可以是 int 也可以是 str**：sweep 结果写进 `artifacts/*.json` 再读回来，
    键会变成字符串（JSON 的键只能是字符串）。不归一化的话
    `sizes.get(k)` 全部落空（碎簇数恒 0，惩罚静默失效），
    而 `-k` 会对字符串取负直接 `TypeError`。

    `sizes` 缺某个 k 时按"无碎簇"处理（不崩）。空输入返回 None。
    """
    if not accuracy:
        return None
    norm_sizes = {int(k): v for k, v in (sizes or {}).items()}
    scored = {}
    for key, value in accuracy.items():
        if value is None:
            continue
        k = int(key)
        tiny = sum(1 for s in norm_sizes.get(k, []) if s < min_healthy)
        scored[k] = value - penalty * tiny
    if not scored:
        return None
    # 平局取小 k：max 先按分数、再按 -k
    return max(scored, key=lambda k: (scored[k], -k))


# ponytail: FRAGMENT_PENALTY 是**未标定**的初值。它必须用真实路径回评来定
# （见 probe_sentence_vs_realpath.py 的思路：拿 dev 已知的 3 场改善做校准）。
# 校准不过就不能用它选判据——这是 §4.1e 的教训。
if __name__ == "__main__":
    # 自检：与测试同口径的两个关键边界
    assert pick_best_k({6: 0.90, 9: 0.90}, {6: [200, 150, 120, 100, 80, 60],
                                            9: [200, 150, 120, 100, 80, 60, 3, 2, 2]}) == 6
    assert pick_best_k({3: 0.35, 6: 0.97}, {3: [300, 200, 100],
                                            6: [200, 150, 120, 60, 5, 2]}) == 6
    print("sentence_k_rule 自检通过")
