# c1_v13_ab_eval 里程碑：dev 评估快照（**不是部署版本**）

归档日期：2026-09-24
定位：**dev 评估快照**——句子级主指标的实现与修复、A/B 两个 K 候选的对照结果。
**这不是可部署的 v13 算法版本**，也**不是**一次方法上的成功归档。

> **一句话**：句子级说话人归属率**已实现可用**（主指标），
> 但本轮两个 K 候选（AHC 距离阈值 / 谱间隙）**均未形成有效 K 判据**；
> 当前可验证的收益只有**样本内 oracle K 上限 13.6 点**，
> 该数字受 K 网格、cap、真实路径与 dev 样本内选择限制，**不能视为方法收益**。

---

## 1. 本归档支持哪些结论

| 结论 | 证据 |
|---|---|
| A 在当前阈值范围与 cap 下**没有形成有效 K 判据**（退化为按阈值间接选 K，部分落入接近最大 K 的平台行为） | `artifacts/v13_ab_dev_report_20260924.md` §2 |
| B 在当前图构造与参数下**退化为近似恒取 K=1**（10/12 场） | 同上 §3 |
| **13.6 点**是 dev 12 场、当前 K 网格与真实路径下的**样本内 oracle 上限增益** | 同上 §4 |
| 不应继续堆叠相对间隙 / 随机图零模型等复杂判据 | 修订规格 §R2；报告 §6 |
| 句子级归属率是**有效的监督评价指标** | 报告 §1；`tests/` 54 passed |

**本归档不支持的结论**（引用时务必不要越界）：

- ❌ **A 类方法原则上不可用** —— 只针对本轮实现、cap 与阈值范围。
- ❌ **谱方法一概不可用** —— 只针对本轮的图构造与参数。
- ❌ **13.6 点可迁移 / 可实现 / 是理论上限** —— 它是样本内 oracle。
- ❌ **可部署的 K 判据** —— 尚未存在。
- ❌ **句子级归属率可用于部署选 K** —— 它**读目标会议真值**，
  定义上不能用于未知会议推理。

---

## 2. 目录内容

```
artifacts/                     评估结果与报告（含失效说明）
  v13_ab_dev_report_20260924.md        ★ 正式报告（逐场表、全部证据、限制）
  v13_ab_dev_spec_rev1_20260924.md     ★ 修订规格（冻结项；取代旧版 §1.1/§1.4/§3）
  v13_ab_dev_spec_20260924.md          初版规格（保留作历史，**请读 rev1**）
  v13_ab_dev_results.json              A/B 对照原始结果（12 场，0 失败）
  v13_sentence_attrib_README.md        句子级产物的失效 / 有效说明
  v13_sentence_attrib_dev_fullgrid_FIXED.json   修正评价器后的全 k 网格（有效）
  v13_sentence_attrib_dev_sweep_BUGGY_label0.json  旧产物（**失效**，保留作证据）

scripts/evaluation/
  probe_sentence_attribution.py    ★ 句子级归属率主指标（唯一实现）
  k_candidates.py                  ★ A/B 两个 K 候选（纯函数）
  probe_ab_dev.py                  ★ A/B 真实路径对照
  analyze_ab_dev.py                    出表：全局阈值选择 + 平台期 + 逐场主表
  discriminate_ab.py                   判别：成绩是否只由 k 决定
  probe_ab_cost.py                     分阶段成本实测（C1/C2/C3）
  probe_sentence_vs_realpath.py        选的 k 是否与真实路径一致
  probe_sentence_proxy_gain.py         增益相关性入口（含 n<3 只报方向的纪律）
  sentence_k_rule.py                   选 k 规则；**FRAGMENT_PENALTY 未标定、未采用**
  _diag_old_vs_new_sweep.py            污染版 vs 修正版逐格对比
  unlocked_selector.py / probe_gap_decomp.py / probe_selector_locks.py
                                       被上面脚本复用的既有模块（三道锁修复与窗缓存工具）

streaming_asr/
  claude_a2_v11.py / claude_a2_v12.py / der.py / v12_aishell4.py / claude_a2.py
  pytest.ini
  tests/  5 个测试文件（句子级评价器、选 k 规则、K 候选、路径卫生）
```

**未归档**：示例音频、声学模型权重、`_v13_spike/cache/`（数 GB 嵌入缓存）、
`__pycache__`、一次性诊断脚本（已移至工作区 `scripts/evaluation/archive/`）。

---

## 3. 复现

### 3.1 零推理（只读缓存，几十秒）

```bash
# 句子级指标（主指标），单场 k 网格
python -u -B scripts/evaluation/probe_sentence_attribution.py --meeting L_R003S01C02 --ks 1,3,5,7

# A/B 判据的判别（只读已有结果，零推理）
python -u -B scripts/evaluation/discriminate_ab.py
python -u -B scripts/evaluation/analyze_ab_dev.py --results artifacts/v13_ab_dev_results.json
```

### 3.2 需要加载模型

```bash
# 分阶段成本实测（约 2 分钟）
python -u -B scripts/evaluation/probe_ab_cost.py --meeting L_R003S01C02

# A/B 正式对照（dev 12 场，实测约 18 分钟）
python -u -B scripts/evaluation/probe_ab_dev.py --split dev --out artifacts/v13_ab_dev_results.json
```

### 3.3 测试

```bash
cd streaming_asr && python -B -m pytest -q      # 54 passed（归档快照内）
```

必须在 `streaming_asr/` 里跑：那里的 `pytest.ini` 设了 `testpaths=tests`、`pythonpath=.`。
在工作区根跑会一并收集 `milestones/` 下的同名测试，报 import file mismatch。

---

## 4. 环境与依赖

| 项 | 值 |
|---|---|
| Python | 3.12（`C:\Users\EDY\AppData\Local\Programs\Python\Python312`） |
| 第三方 | `numpy`、`scipy`（`cluster.hierarchy` + `optimize.linear_sum_assignment`）、`soundfile` |
| 声学模型 | CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`）、FSMN VAD（**本机魔改版**：不显式传 `max_end_silence_time` 则 `speech_noise_thres` 失效） |
| 数据 | `D:\yinp\AISHELL-4\data\test`（dev 12 场，`wav/*.flac` + `TextGrid/*.TextGrid`） |
| 嵌入缓存 | `D:\yinp\AISHELL-4\_v13_spike\cache`（`w2_h1.npz`，**只覆盖 test 的 20 场**） |
| VAD 区间缓存 | `...\cache\vad\*.json`（20 场全） |

**数据与缓存不在归档内**：路径写死在脚本里（`CACHE` / `DATA`），
换机器需改这三个常量或做符号链接。

**成本实测（本机）**：C1 模型加载 6.2s；C2 每场 91.5s（其中密窗嵌入 67.5s）；
C3 换 K 的增量 ≈0（`_refine_windows` **不依赖 K**，密窗嵌入每场只算一次、全部候选共用）。

---

## 5. 归档时的状态（测试与版本）

| 项 | 值 |
|---|---|
| 工作区测试 | **131 passed**（含归档内不含的 v10/v11/v12/服务等测试） |
| 归档内测试 | **54 passed** |
| 归档内脚本独立运行 | ✅ 已验证（从归档目录跑 `probe_sentence_attribution.py` 得到与工作区一致的 94.0% / 33.2%） |
| 本归档**未**包含 | holdout（8 场）、train_L（30 场）的任何新实验 |

**归档内 54 passed 只证明回归检查通过，不证明任何选 K 方法有效。**

---

## 6. 已知未解决 / 下一步

**下一步的重点不是再找通用 K 判据**，而是**分析高收益场次的机制**（前 2 场贡献 60.7%、
前 5 场贡献 93.5% 的样本内 oracle 增益）：

1. `M_R003S05C01` 与 `L_R003S01C02` 为什么 base K 很低（2 / 1），而 oracle K 分别到 7 / 10；
2. 最优 K 高于真实说话人数时，收益来自身份区分，还是来自更细的换人边界；
3. 归属率提升发生在长窗聚类、密窗细化、还是最终段合并；
4. 高收益场次与低收益场次在 VAD 覆盖、说话人占比、Embedding 簇纯度、重叠比例上的差别。

**一个待优先验证的假设**（⚠️ 状态：**假设，未验证**。截至 2026-09-29 仍未被检验，
引用时不得写成结论）：

> **K 可能同时承担"说话人身份数量"与"时间边界锚点数量"两个角色。**
> 若最优 K 经常大于真实说话人数，继续寻找更好的"说话人数 K 判据"可能方向不对；
> 更合理的后续范式可能是**把身份数与边界锚点数分开**。
>
> **推论（同样是假设）**：不能把 RTTM 真值人数直接当作应选 K 的目标值。
> 2026-09-29 实测 dev 12 场 base 系统性低于真值人数（11/12 场，平均差 3 人），
> 但那是"base 偏低"的证据，**不是**"应选 K 等于真值人数"的证据。

**未做但已立项**：A 的阈值网格在 0.6→0.7 之间跨度过大（k 从中位 9 掉到 4），可加密；
`train_L` 的样本外验证（未开始）；`交接文档_0924.md` §6 第三步的全量报告。

---

## 7. 相关文档（工作区，不在本归档内）

- `交接文档_0924.md` §11：本轮的完整记录（含撤回的结论清单与方法论增补）
- `artifacts/v13_ab_dev_report_20260924.md`：正式报告（已复制进本归档）
- `milestones/c1_v12_20260922/`：上一份冻结快照（v12，本归档的起点）
