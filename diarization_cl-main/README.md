# diarization_cl

单 mic 长会议系统的**说话人分离 + 说话人识别**（会议内匿名说话人归属，不是声纹库实名识别）。
目标场景：安静、轮流发言、36–40 分钟长音频。

本仓库同时承载**代码主线**与**历史里程碑归档**（见文末「里程碑归档」）。

---

## 快速上手

```bash
pip install -r requirements.txt

# 1) 数据与模型不在仓库内，先准备（见「外部依赖」）

# 2) 零推理先验口径（只读缓存，几十秒）
python -u -B scripts/evaluation/probe_fullref_truncation.py --skip-v12

# 3) 测试（必须在 streaming_asr/ 里跑：那里有 pytest.ini 设 testpaths/pythonpath；
#    在仓库根跑会把 milestones/ 下的同名测试一起收集，报 import file mismatch）
cd streaming_asr && python -B -m pytest -q     # 154 passed
```

> **测试数口径**：`154 passed` 是 2026-09-29 交接文档记录的基准数。本会话在 DSH 沙箱内
> 复跑实测为 **148 passed + 6 errors**，6 个 error 全部是 `tmp_path` 夹具 setup 阶段的
> `PermissionError`（沙箱拒绝写 `%LOCALAPPDATA%\Temp\pytest-of-EDY` 与 `.pytest-tmp*`），
> 与代码无关；在普通终端下应恢复到 154 passed。**若你复跑得到的不是 154，先查这一点。**

**入口链**：生产分离入口是 `streaming_asr/claude_a2_v12.py` 的 `diarize()`，它内部调用
`claude_a2_v11.py` 的聚类组件。长窗（2.0s）定身份、短窗（0.6s）只找换人边界。

---

## 当前状态（2026-09-29）

### 版本分工

| 版本 | 角色 |
|---|---|
| v7–v10 | 早期测试台、ASR 与双人基线 |
| v11 | 多人聚类实验泳道：候选 K 生成、聚类、候选筛选 |
| **v12** | **当前生产分离流程**：2.0s 长窗定身份 + 0.6s 短窗细化边界 |
| v13 | **评估阶段**（评估工具、证据链、bug 修复），**不是**已冻结的新算法 |

`v12` 是生产入口，`v11` 提供聚类组件。**K 选择不是独立的 v13 算法**——v13 回答的是
"生产路径与探针口径是否一致""修复的代价是多少""A/B 规则是否真的改善了真实路径"。

### 已确认

- **K 上限静默截断 bug 已修复**（`977fc6b`）。根因：`_candidate_clusters(E, max_speakers=MAX_SPEAKERS)`
  的默认值在 **`def` 执行时**就绑成 4，`_v11_window_params` 改模块变量**改不动它**，
  于是候选恒为 k=1..4 而意图是 1..10。修法是 `diarize` 两处调用点显式传参，`v11` 零改动。
- **修复代价已量化**：12 场里 **1 场改善 / 11 场无变化**。`S_R004S01C01` 的 k 由 4→5，
  按句主指标 78.7% → **90.7%（+12.0 pts）**，其余 11 场 k、段表、归属率逐位相同。
- **测试 154 passed**，工作区证据链可复算。

### 仍是待验证假设（**不许当结论引用**）

- **"K 应等于真值人数"** —— 未证明。K 可能同时承担"说话人身份数量"与"时间边界锚点数量"
  两个角色；本仓库只用 RTTM 人数说明"base 偏低"，**不主张** base k 应等于真值人数。
- **"份额门 `MIN_CLUSTER_SHARE` 是欠分割根因"** —— 只是**候选准入层面**的观察。
  改这道门能否修好欠分割未验证；且过分割代价与簇纯度都未度量。
- **A/B 选 K 规则的作用** —— 在 dev 12 场、当前探针口径内，A 表现为**条件性兜底**：
  base k≤3 的 9 场中 7 提升 / 2 持平 / 0 下降；base k≥4 的 3 场全部下降。
  这**不能推出** A 是普适 K 判据，也**不能**当作方法收益。A **未进入生产**。
- **rescue 分支** —— "替换逻辑的**控制流**"已由合成回归覆盖（`test_rescue_replacement.py`）；
  **真实会议上的替换仍未验证**（dev 12 场无一场真的替换过）。**不得**写成"救援机制已端到端验证"。

### 不得引用的表述

| 不要引用 | 原因 |
|---|---|
| "12/12 行为保持" = 修复无代价 | **同义反复**。探针 base 修正后与生产同源，本来就该一致 |
| "9/9"、"一律变好" | 实为 7 提升 + 2 持平 |
| "base 已分对" | `k≥4` 只说明"分得较多"，未对真值验证 |
| `L_K_all` 是当前系统可获得的收益 | 那是**全候选强制 oracle 上界**，越过合格门限 |
| `touched_cap=true` 代表选择器收敛到 cap | 只说明候选**生成**到了 cap=10 |

---

## 文档入口（按可靠性排序）

| 文件 | 内容 |
|---|---|
| `docs/HANDOFF.md` | **最新**。unlocked 的主指标收口（+8.8，非样本外）+ 残差归因翻转（「剩下是聚类可分性」不成立） |
| `docs/交接文档_1008_过分割涨分证伪.md` | 过分割涨分**一半成立**（4 真 / 1 假 / 7 零收益）+ 换尺子重跑，最优 k 从 10 → 6 |
| `docs/交接文档_0930b_大簇拆分.md` | 大簇拆分收益**指标失败**（gain 与 oracle 近乎正交，精确率 0.35） |
| `docs/交接文档_0930.md` | 零模型诊断链：三道锁 / 小簇性质 / 候选有效性 |
| `docs/交接文档_092902.md` | OCR 复审第二轮：3 处静默失真 + 8 条 medium，含"明确不修的"及理由 |
| `docs/交接文档_0929.md` | K 上限修复的代价量化、A 的定位改写、六处被数据推翻的判断 |
| `docs/交接文档_0928.md` | K 上限 bug 的**发现过程**（含路径等价检查方法论） |
| `docs/交接文档_0924.md` | 句子级主指标的定义草案与三步计划 |
| `artifacts/v13_交付合并表.md` | 交付数字 + 证据索引（含物理边界与未验证项） |
| `artifacts/过分割涨分证伪.html` | **本轮结论的分析页**（浏览器直接打开） |
| `artifacts/v13_会话发现_20260923.md` | 该会话实测记录 |
| `docs/环境清单.md` | 绝对路径与本机未装的备选模型 |

> 交接文档（`交接文档_09*.md` / `HANDOFF.md`）**自 2026-10-08 起统一在 `docs/`**，
> 不再放仓库根。根目录只留 `README.md`。

**新对话开场**：先读 `docs/HANDOFF.md`，再按 `交接文档_1008_过分割涨分证伪` →
`0930b_大簇拆分` → `0930` → `092902` → `0929` → `0928` 上溯，**先复现再采信**：

```bash
cd streaming_asr && python -B -m pytest -q                              # 154 passed
python -B -m pytest tests/test_k_cap_dispatch.py -q                     # K 上限修复的回归
python -B -m pytest tests/test_rescue_replacement.py -q                 # rescue 替换控制流
cd .. && python -u -B scripts/evaluation/probe_step_a_baseline.py       # 四口径证据链（零模型）
PYTHONIOENCODING=utf-8 python -u -B \
    scripts/evaluation/probe_unlocked_sentence.py --min-wins 2,3,4 --selfcheck   # 主指标，秒级
```

> ⚠️ 跑**真实路径**（`probe_e2e_unlocked2.py`）是 **21 分钟/场**，别拿它做扫描。
> 句级口径走缓存即可（`probe_path_equiv` 已验证等价）。
>
> ⚠️ **建缓存时退出码 1 不代表失败**：门限自检的参照
> （`v13_ab_dev_results.json`）**只覆盖 dev 12 场**，所以 train_L 与 holdout 的场次
> 全部会报"不在参照里"，最后 `SystemExit(1)`——**此时 npz 与 manifest 已经写完**。
> 这两批缓存的门限自检**是空的**（`--check-only` 复算的是同一批 VAD 区间，
> 只验确定性、不验 VAD 本身）。要补真正的门限验证，得先按同口径跑
> `probe_ab_dev` 生成覆盖该 cohort 的参照。

---

## 外部依赖（不入库，需自行准备）

### 数据

| 项 | 路径 | 说明 |
|---|---|---|
| AISHELL-4 test | `D:\yinp\AISHELL-4\data\test\` | 20 场、8 通道、12.7 小时；本项目**只读 ch0** |
| 嵌入缓存 | `D:\yinp\AISHELL-4\_v13_spike\cache\` | 2.0s 窗 CAM++ 嵌入，探针的零推理前提 |
| 生产门缓存 | `...\_v13_spike\cache\probe\<场次>\{main,refine,rescue}.npz` | `build_probe_cache.py` 生成 |
| VAD 区间缓存 | `...\_v13_spike\cache\vad\` | `cache_vad_intervals.py` 生成，只需一次 |

数据下载：OpenSLR SLR111 <https://www.openslr.org/111/>。

⚠️ **生产门缓存与 RTTM 门缓存窗集不同、不可混用**（见 `docs/交接文档_0928.md`）。
`build_probe_cache.py` **无参数 = 重建既有全部场次**（约 35 分钟）；只重建三场须显式传 `--meetings`。

### 模型（走 ModelScope 缓存，不入库）

- **CAM++ 声纹**：`iic/speech_campplus_sv_zh-cn_16k-common`
- **FSMN VAD**：本机用的是**调过的本地副本**（`speech_noise_thres=0.7`），
  共享缓存是 0.6 且会把语音并成连续段。路径见 `streaming_asr/web_server.py` 的 `VAD_LOCAL`。
  ⚠ 本机 funasr 是魔改版：**不显式传 `max_end_silence_time` 时 `speech_noise_thres` 失效**。

### 环境

两个 conda 环境：`funasr`（主链路）、`3d-speaker`（CAM++）。

---

## 目录

| 目录/文件 | 内容 |
|---|---|
| `streaming_asr/` | **算法主线**：v7–v12 泳道、`der.py`（DER 评分器）、`web_server.py`（8735 测试台）、`tests/` |
| `scripts/evaluation/` | **评估、对照与验收脚本**（当前主战场，含 `unlocked_selector.py`、`build_probe_cache.py`、`probe_*.py`） |
| `scripts/batch/`、`scripts/diagnostics/`、`scripts/audio/` | 批量推理、一次性诊断、音频处理 |
| `artifacts/` | 实测记录、交付表与 JSON 产物 |
| `docs/` | **交接文档**（`交接文档_09*.md` / `HANDOFF.md`）、音频位置与复现入口、历史计划、工作日志、环境清单 |
| `milestones/` | 版本归档（**只读，勿改**） |
| `archive/` | 退役产物：`legacy_versions/`（c1 旧版 py）、`legacy_docs/`、`big_output/`（大体积 gitignore 产物） |
| `dsh_line/` | DeepSeek 科研线的交接与脚本（**只读，勿改**） |
| `eval_dsh/` | DSH 对比用的真值 `ground_truth.json`（**活引用**，多处脚本读它） |
| `not_mind/` | 与本项目无关（**只读，勿改**） |

**根目录只留 `README.md` 一份文档。** 其余平铺文档已收进 `docs/`（2026-10-08）。

根目录的 `.py` 分两类，**不要合并**：

| 文件 | 身份 |
|---|---|
| `claude_a2.py` | **c1 v4**（双人核心）。`run.py` 与 `cv_ss_test/*` 从**仓库根**加载它 |
| `a2_v2.py` | c1 v4 的分段实现。`scripts/batch/` + `scripts/diagnostics/` 的 33 个脚本从**仓库根**加载它 |
| `streaming_asr/claude_a2.py` | **同名但不同物**：v7–v12 泳道分发器，8735 测试台走这个 |
| `denoise_frcrn.py` / `env_detect.py` | 根目录这份是 `run.py` 用的；`streaming_asr/` 里另有一份**更新的**（含 `_chunked_infer` 分块推理） |

---

## 已知边界（写报告必须声明）

- **单通道天花板**：说话人重叠时单标签输出必然错一部分；本机无 TS-SEP。
  分离式提纯已否证（ClearVoice MossFormer2_SS 在轮替对话上退化成对半拆）。
- **参考里多是小说话人**：53 个"短发言者"时长占比 <5%，中位仅 2.0 秒；
  按"参考全人数"当已知 K 会造出假收益。
- **`der.py` 口径**：union 分母（md-eval 口径）；**并人落在 `confusion`，不在 FA**。
- **DER 的两条口径差 69 个点**：参考是否截断到分析窗口。
  引用任何上界数字时必须标明（`probe_fullref_truncation.py` 有对照表）。
- **数据口径**：dev 12 场 / 前 600 秒 / 按句平均主指标、弃权计入分母、失败场次显式记录。
  **holdout 8 场与 train_L 未按本轮新口径作迁移结论**。

---

# 里程碑归档

本仓库根目录与各归档目录保存历史里程碑，各目录自带 README 与 `SHA256.json` 完整性清单。
**归档目录只读，不要修改。**

| 目录 | 日期 | 方法标识 | 内容 |
| --- | --- | --- | --- |
| （根目录代码） | 2026-09-08 | `claude-a2-v4-c1` | c1 v4：双人分离 + 逐段 ASR。含 `run.py`（服务器入口）、`denoise_frcrn.py`、`env_detect.py`、`vendor_pkgs/`、`request.txt`、`verify.py`、`SHA256.json` |
| `evidence/` | 2026-09-08 | — | 该里程碑的全量评估与网页接口结果 |
| `v12_meeting/` | 2026-09-21 | `claude-a2-v12-meeting` | v12：两级窗多说话人分离（2.0s 定身份 + 0.6s 找边界）+ 自动人数。**状态 `experimental`**，未替换 v7/v4 基线 |
| `milestones/c1_v13_ab_eval_20260924/` | 2026-09-24 | — | v13 A/B 评估快照 |
| `milestones/c1_v13_ab_eval_rev1_20260929/` | 2026-09-29 | — | v13 评估修订快照（含修复前产物） |

`v12_meeting/` 相对 v7 在 5 个三人文件上加权轮次准确率 +0.073，但**自动人数在长会议上
仍是首要缺口** —— 详见 `v12_meeting/README.md` 的「已知缺口」。

> 归档数字是**当时口径**的产物，不要与当前口径的数字混用。当前结论一律以本节之前的
> 正文与 `交接文档_09*.md` 为准。
