# HANDOFF —— 大簇拆分收益诊断（2026-09-30 会话）

> **给一个完全没有上下文的新对话。** 先读 `交接文档_0930.md` → `092902` → `0929` → `0928`
> 建立背景，再读本文件。本文件只覆盖 **2026-09-30 之后新增的那一项工作**
> （`probe_large_cluster_split.py`），不推翻前几份的任何结论。
>
> 一句话：**按上一份交接 §10 的建议，做完了"大簇拆分收益"这个指标，
> 结论是这个指标失败了——它区分不了"拆对人"和"过分割涨分"。已按验收标准 7 停在原地，
> 没有动生产代码。**

---

## 0. 三十秒版本

| 项 | 状态 |
|---|---|
| 任务 | 设计一个**不依赖真值**的"大簇拆分收益"指标，用来判断某个更大的 k 是不是真的拆对了一个人 |
| 产出 | `scripts/evaluation/probe_large_cluster_split.py`（新增）+ `artifacts/v13_large_cluster_split.json`（新增） |
| 生产代码 | **一行未改**（`git diff HEAD -- streaming_asr/` 为空） |
| 测试 | 154 passed；新脚本 selfcheck 7 项全过 |
| 结论 | **失败**。指标与 oracle 几乎正交（r = +0.041），精确率 0.35 |
| 下一步 | **先证伪"选大 k 就涨分"这个退化信号**，不要做候选选择，不要进生产 A/B |

---

## 1. 我们在做什么

项目：`C:\Users\EDY\Desktop\new`，AISHELL-4 场景的说话人分离 / 识别。
生产入口是 **v12**（`streaming_asr/claude_a2_v12.py`），聚类组件来自 **v11**（`claude_a2_v11.py`）。

前一份交接（`交接文档_0930.md`）把问题定位成：

- 映射层损失 = 0（可从嫌疑名单划掉）；
- 合格候选集内，现行选择器与 oracle 10/12 场一致 → 选择器不是主要问题；
- **症结在候选有效性**：全局更优的 k 经常在**候选准入**阶段被排除；
- 12 场里 k_all_best 与生产 k 不同的 **11 场**（见 §4 实测表）；
- 三道锁 A/B/C 是**串联**的，单独放宽门限无效（`probe_lock_diag.py` 已锁死）；
- 收益来自**大簇拆分**，不是小簇（小簇 clean 贡献 +0.67，大簇 +19.93）。

`交接文档_0930.md` §10 给的下一步建议是：

> 对 11/12 场的全优 k，查**从 `k_prod` 到 `k_all_best` 大簇具体拆出了什么**——
> 哪些真值人原本合在一起、拆开后各自纯度多少。

**本轮就是把这句话做成可运行的脚本。**

---

## 2. 已经完成了什么

### 2.1 关键结论先行：**指标失败**

新脚本的核心产出是 `gain`（拆分收益）：

```
gain = 父簇内平均到质心余弦距离 − 子簇加权平均
```

**它区分不了有效拆分和过分割。** 三条独立证据：

| 证据 | 数值 | 含义 |
|---|---|---|
| `gain` vs 该层 **Δoracle** | Pearson **r = +0.041** | 几乎正交 |
| `gain` vs 该层 **绝对 oracle** | Pearson **r = −0.049** | 略微负相关 |
| 四项判据的精确率 | **0.35**（达峰 11 次 / 明显低于峰值 20 次） | 三分之二是假阳性 |

### 2.2 消融实验：规则的"胜利"来自选大 k，不来自 gain

这是本轮最重要的实验。规则定义：取"该层新增拆分通过四项判据（`sizes_ok` + `gain_ok` +
`sep_ok` + 非 `scratchy`）"的**最大** k。逐场与 oracle 峰值比较平均 regret：

| 规则 | 平均 regret |
|---|---|
| 现行生产 | **+12.99 pts** |
| 恒选 k=1（等价于 C 锁回退） | +40.55 |
| 恒选 k=2 | +23.03 |
| 恒选 k=4 | +10.35 |
| 恒选 k=6 | +0.97 |
| **恒选 k=10（上限）** | **+0.00** |
| 四项判据规则 | +2.92 |
| ↳ **去掉 `gain>0` 这一条** | **+2.92（完全没变）** |
| ↳ 只留 `parent_size>=4` | +0.09 |
| ↳ 完全不加判据（取最大 level_k） | +0.09 |

**读法**：`gain` 换成任何东西都不影响结果；规则唯一的作用是**剔掉 k=1 和 k=2**。
真正的信号是"k 越大越好"。

⚠️ **但这不等于"大 k 一定对"**——见 §3 的口径陷阱。

### 2.3 逐场实测表（可直接引用）

oracle 峰值 k **不是**普遍很大：12 场的峰值 k 是 `[7,6,7,6,5,10,8,6,5,8,6,5]`，
其中只有 **1 场** 峰值是 10（`M_R003S02C01`），**5 场** 峰值 ≥ 7。
但恒选 k=10 的平均 regret 仍是 **0.00** —— 原因见 §3。

| 会议 | 主窗 | 生产k | 峰值k | 生产regret | 四项判据规则k | 规则regret |
|---|---|---|---|---|---|---|
| L_R003S01C02 | 422 | 1 | 7 | +50.98 | 9 | +0.00 |
| L_R003S02C02 | 281 | 4 | 6 | +2.35 | 7 | +0.00 |
| L_R003S03C02 | 387 | 3 | 7 | +25.30 | 5 | **+22.89** |
| L_R003S04C02 | 485 | 4 | 6 | +0.84 | 9 | +0.00 |
| M_R003S01C01 | 367 | 2 | 5 | +2.41 | 8 | +0.00 |
| M_R003S02C01 | 373 | 2 | 10 | +5.61 | 8 | +2.24 |
| M_R003S04C01 | 556 | 2 | 8 | +10.68 | 6 | +6.31 |
| M_R003S05C01 | 408 | 2 | 6 | +49.12 | 8 | +0.00 |
| S_R004S01C01 | 513 | 5 | 5 | +0.00 | 6 | +0.00 |
| S_R004S02C01 | 578 | 2 | 8 | +2.73 | 3 | +1.82 |
| S_R004S03C01 | 570 | 2 | 6 | +3.51 | 4 | +1.76 |
| S_R004S04C01 | 585 | 2 | 5 | +2.38 | 8 | +0.00 |

规则胜 **11** / 平 **1** / 亏 **0**（唯一亏的是 `L_R003S03C02`，+22.89 pts 没救回来）。

**k_all_best 层的拆分类别**（96 个）：**orphan 50 / real 34 / fragmented 6 / overlap_born 6**。
"全通过四项判据"只有 **2/12 场**。

### 2.4 脚本本身做了什么

`probe_large_cluster_split.py` 遍历 AHC 层次树的**每一个内部合并节点**，
把它当成一次"父簇 → 两子簇"的拆分，输出：

| 字段 | 含义 |
|---|---|
| `gain` / `gain_ratio` | 拆分收益（绝对值 / 相对父簇散度） |
| `sep` | 两子簇质心的余弦距离 |
| `flip_ratio` | 时间穿插比 = 实际翻转次数 / 超几何随机期望 |
| `parent_mass_share` | 父簇窗在全体窗里的相似度占比 |
| `sizes_ok` / `gain_ok` / `sep_ok` / `scratchy` / `pure` | 四项判据 + 合取 |
| `level_k` | **该节点在哪个 `maxclust` k 层是一次可见拆分**（用 `fcluster` 直接对账得出，不是推的） |

跑法（全部零模型，普通终端）：

```bash
cd "C:/Users/EDY/Desktop/new"
PYTHONIOENCODING=utf-8 python -u -B scripts/evaluation/probe_large_cluster_split.py --selfcheck
PYTHONIOENCODING=utf-8 python -u -B scripts/evaluation/probe_large_cluster_split.py
PYTHONIOENCODING=utf-8 python -u -B scripts/evaluation/probe_large_cluster_split.py --detail M_R003S05C01
```

⚠️ **必须带 `PYTHONIOENCODING=utf-8`**，否则 Windows 控制台是 GBK，中文报告全是乱码。

---

## 3. 当前卡在哪

### 卡点 1：`k_all_best` 是不是"蹭分"，还没证伪

恒选 k=10 的 regret 恰好是 **0.00**，但 12 场里只有 1 场峰值 k 真的是 10。
这两件事**只能同时成立**，如果 **oracle 准确率在大 k 上单调不减**（先升，到峰值后走平）。

也就是说：句级 oracle 这个指标在过分割时**不会掉**。一个说话人被切成 5 个簇，
匈牙利/独立取最大仍能把每个簇映射到那个人，句级归属率不降。

⇒ **`k_all_best` 可能根本不是"最优 k"，而只是"过分割不扣分的那个 k"。**
若是这样，上一份交接 §3 的"症结在候选有效性"这个定位需要**重新评估**——
因为被合格集挡在外面的"全局最佳 k"，可能不是一次真实的两人分离，而是评分口径的产物。

### 卡点 2：`gain` 为什么失效，还没查清

已知 `gain` 与变化**不是**线性的。没查的是：
- 子簇纯度高的拆分 vs 纯度低的拆分，`gain` 分布是否可分？
- 同一个父簇在不同 k 层反复拆（`L_R003S01C02` 的 421→300→230→125 链）时，
  哪一层对应真值上的真实换人？

### 卡点 3：`L_R003S03C02` 这个反例

规则给它选 k=5（62.65%），峰值在 k=7（85.54%），**亏了 22.89 pts**。
这一场里 k=6 和 k=7 的拆分都被判成 `orphan`（`78→[76,2]`、`4→[2,2]`），
但 oracle 在 k=6 跳了 **+21.69 pts**。**说明 orphan 类里也可能混着真实拆分。**
还没查清这两个 2 窗簇到底是什么。

---

## 4. 下一步计划（按优先级）

### 第一优先：证伪或证实"过分割涨分"

**这是阻塞项，做完它才谈得上其他。**

对 12 场，取 k=6 与 k=10 两个 partition，**逐簇**看 k=10 多出来的那些簇：

1. 每个新增簇对应哪个真值人？
2. 该真值人在 k=6 里是否已经有主簇（即新增簇是不是它的**重复副本**）？
3. 若是副本 → 说明 k=10 的"涨分"是过分割，`oracle` 口径需要先修；
4. 若确实对应 k=6 里合在一起的第 7/8 个真值人 → 说明大 k 是真收益，回到候选有效性主线。

`probe_small_cluster_audit.py` 里已有 `dup` / `short` / `overlap` 的性质分类可直接复用，
不用另写。

### 第二优先（仅当上面证明大 k 是真收益）：找不依赖真值的判据

本轮已证 `gain` 不行。剩下没试过的方向：

- **簇的绝对纯度代理**：簇内到质心的距离分布（`scatter`）而不是跨 k 的 `gain` 差分；
- **时间连续性**：`flip_ratio` 本轮信号很弱（`real` 类里也有 `flip_ratio` 0.76 的），
  但没试过按**簇**（而非按层）算；
- **两个子簇各自的最小尺寸**：本轮 `min_child_size > 2` 只是硬门槛，没当连续量用。

### 明确不要做的

- ❌ 不要改动 `_candidate_clusters` / `_select_candidate` / `_silhouette`
- ❌ 不要删 `MIN_CLUSTER_WINDOWS`
- ❌ 不要把 `gain` 当判据接进生产
- ❌ 不要用"恒选大 k"当方案
- ❌ 不要跑门限热力图（上一份交接已否决）
- ❌ 不要在 holdout 上验证任何东西，除非 dev 12 场先有可信结论

---

## 5. 踩过的坑（**绝对不要再踩**）

### 坑 1：`k-1 → k` 那一层永远只拆掉小簇（本轮最大的坑）

**症状**：第一版直接比较相邻两层 k 的 partition，12 场拆出来的全是 `[n-1, 1]` 或
`[n-2, 2]`，全是 orphan，看起来"大簇拆分根本不存在"。

**原因**：`maxclust` 的 k-1 → k **只拆掉树里最后一次合并**，那次合并永远是把一个
小簇粘上去。

**正确做法**：遍历层次树的**所有内部节点**，再用 `fcluster` 判断该节点在**哪一层**
partition 上是一次可见拆分（`level_k`）。本轮脚本的 `mark_level_splits()` 就是干这个的。

**教训**：`k` 层候选和"树的内部节点"不是一一对应。不要用"相邻两层相减"来定位拆分。

### 坑 2：`window_reference` 看不到重叠

**症状**：真值"重叠占比"这一列**恒为 0**，12 场 96 个拆分全是 0%。

**原因**：`probe_gap_decomp.window_reference` 只返回每窗的**多数说话人**，
输出里永远不含重叠信息。

**正确做法**：用 `probe_small_cluster_audit.truth_flags(ref, times, win)`
——它按帧投票，返回 `(boundary, overlap)` 两个布尔数组。

### 坑 3：`score_oracle` 返回的是完整结构，不是 `overall`

`score_oracle(sents, pred)` 返回的是 `attribution` 的全量 dict，
取值要写 `score_oracle(...)["overall"]["per_sentence"]`，
直接写 `["per_sentence"]` 会 `KeyError: 'per_sentence'`，写 `["clean"]` 会 `KeyError: 'clean'`。

### 坑 4：模块级 `times` 会遮蔽函数参数

`probe_path_equiv` 里有模块级的 `times` 之类的名字，函数里再用同名参数会静默出错
（报 `IndexError: boolean index did not match`，维度对不上）。本轮的解法是参数改名 `starts`。

### 坑 5：numpy 运算符优先级

```python
float(group @ centroid) / norm / len(group)   # ❌ TypeError: only length-1 arrays...
float(np.mean(group @ centroid)) / norm       # ✅
```

`float()` 要包在 `np.mean()` **外面**。

### 坑 6：Windows 控制台编码

所有中文报告都要 `PYTHONIOENCODING=utf-8`，否则输出全乱码（用 `tail` 看会以为脚本坏了）。

### 坑 7：`git add -A` 会带进别人的文件

`git status` 里长期有**不属于本任务**的三类文件，**逐个精确暂存**：

```
 M docs/工作日志.md              ← 另一条线（DSH/V5）的改动，不是我们的
?? Audio_音频位置.txt            ← 非本项目文件，来源不明
?? artifacts/_lock_diag_stdout.txt  ← 上一轮的调试中转，可删
```

### 坑 8：不要把汇总列当逐场事实用

**这是上一份交接 §6 里作者自己犯过两次的错**（`#5` 和 `#6` 是同一类）。
本轮同样适用：§2.3 的"规则胜 11/1/0"是**汇总**，
`L_R003S03C02` 一场就亏了 22.89 pts。**读汇总列必须逐行核。**

---

## 6. 不得引用的表述（本轮新增）

| 不要引用 | 原因 |
|---|---|
| "大簇拆分收益指标可用" | 与 oracle 相关性 r=+0.041，精确率 0.35 |
| "判据规则可以替代现行选择器" | 它的收益全部来自"选大 k"，去掉 `gain>0` 结果不变 |
| "k=10 是 dev 12 场的最优 k" | 只有 1 场峰值是 10；0.00 的 regret 是"过分割不扣分"的产物 |
| "`k_all_best` 代表一次真实的两人分离" | **尚未证伪"过分割涨分"**，见 §3 卡点 1 |
| "gain 越大拆分越好" | 同层内都不成立（精确率 0.35） |
| "orphan 类一定不是真实拆分" | `L_R003S03C02` 的 k=6 判为 orphan 却涨了 +21.69 pts |

**仍然有效**（前几份的）：
`K 应等于真值人数`、`份额门是根因`、`A 是普适判据`、`rescue 已端到端验证`、
`12/12 行为保持`、`touched_cap = 收敛到 cap`、`映射层损失为 0`、`小簇不是冗余副本`。

---

## 7. 仓库状态快照（本会话结束时）

> **2026-10-08 补记**：本节下列文件**已全部提交并推送到 `origin`**
> （本文件、`docs/交接文档_0930.md`、9 个 `probe_*.py`、8 个 `artifacts/v13_*.json`
> 与 `artifacts/过分割涨分证伪.html`）。唯一仍未跟踪的是 `artifacts/_lock_diag_stdout.txt`
> （本地调试中转）。下面的「未提交、未推送」描述的是本会话结束时的历史状态。

```
HEAD = f8efc64（本地 main），落后 origin/local-work 8 个提交
分支：只有 main，未新建、未提交、未推送
```

新增未跟踪文件（**本会话产出**）：

```
?? scripts/evaluation/probe_large_cluster_split.py    ← 本轮新增
?? artifacts/v13_large_cluster_split.json             ← 本轮新增（711KB）
```

新增未跟踪文件（**本会话之前就在**，来自 2026-09-30 那一轮）：

```
?? scripts/evaluation/probe_{loss_decomp,lock_diag,gate_diag,
                             small_cluster_audit,overlap_aware,
                             small_cluster_clean_attrib}.py
?? artifacts/v13_{loss_decomp,lock_diag,small_cluster_audit,
                  overlap_aware_candidate_diag,small_cluster_clean_attrib}.json
?? 交接文档_0930.md
```

**未提交、未推送。** 若要提交，参考 `交接文档_0930.md` §9 的精确暂存清单。

---

## 8. 环境备忘

| 项 | 值 |
|---|---|
| Python | 3.12.10 |
| 探针缓存 | `D:\yinp\AISHELL-4\_v13_spike\cache\probe\<场次>\{main,refine,rescue}.npz` |
| 真值 | `D:\yinp\AISHELL-4\data\test\TextGrid\<场次>.rttm` |
| 场次划分 | `v12_aishell4.split_meetings(DATA, "dev")` → 12 场，顺序固定 |
| 主窗 | 2.0 秒 / hop 1.0 秒，600 秒截断 |
| 常用导入 | `probe_path_equiv.cached_stage(m, "main")` → `(emb, times)`；`PE.load_sentences(m)` |
| 真值指标 | `probe_mapping_objective.score_oracle`（上界）；`probe_sentence_attribution.attribution`（匈牙利映射） |

---

**本文件里的"我" = 2026-09-30 之后那个会话（负责 `probe_large_cluster_split.py`）。**

§2.2 的消融表是本轮最有价值的东西——**它证明了规则赢在选大 k，而不是赢在拆分判据**。
§3 的卡点 1 是本轮**没有解决**的问题，也是下一轮的第一优先：
**在大 k 涨分被证伪或证实之前，任何候选选择方案都不要动生产代码。**
