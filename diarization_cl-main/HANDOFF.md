# HANDOFF — 离线说话人识别支线

> 写给**完全没有上下文**的接任者。读完这一份就能接手。
> 最后更新：2026-10-18（会话结束前）。

---

## 0. 30 秒速览

- **本仓库**：`D:\Deepseek\diarization_cl-main`（**不是 git 仓库**，见 §6 坑 4），
  只负责**离线、会后的说话人识别 / 回填**。`D:\Deepseek\Meeting_main` 是下游全栈仓库，
  **默认只读**；2026-10-18 用户明确授权做过一次同步（见 §1.2 / §3.11）。
- **当前状态（2026-10-18 会话结束）**：
  - 上游封板：5 条 `HANDOFF_1015 §10.2` 缺陷已修；`backfill_speakers` 默认
    `normalize_peak=0.85` 弱音提拉；`with_segments=True` 返回句内
    `segments/sub_turns`；密集窗 K 救援**默认关闭**
    （`DENSE_RESCUE_ENABLED=False`）。
  - 8 场新基线 **54.03% / +细化 61.01%**（旧 54.05 / 60.44）。
  - 物业段 `smoke_clip_180s.wav`（`b8f0…`）自然 **2 人**；餐饮段
    `session_audio.wav`/`sweep_*.wav`（`b2d1…`）**3 人 / 11 切换**。
  - 下游 `Meeting_main` 已同步封板指纹并接入 `sub_turns`；`L_R003S01C02`
    新 CER **35.87%**（Sub 10.31 / Del 22.54 / Ins 3.02），比旧 v1 35.30%
    **+0.57 pt**——这是当前**唯一未决的取舍**（§4.1）。

---

## 1. 职责边界（先看这条，能省你半天）

### 1.1 这个仓库负责什么

`D:\Deepseek\diarization_cl-main` **只负责离线说话人识别**。具体是四个文件（都在 `.zhang/`）：

| 文件 | 职责 | 权限 |
|---|---|---|
| `.zhang/offline_diarizer.py` | **唯一交付接口** `backfill_speakers()` | 可改（改前必须快照，见 §6 坑 11） |
| `.zhang/streaming_asr_engine.py` | 流式 VAD + Paraformer，产出句子与 `session_audio.wav` | 可改，但切句参数已冻结 |
| `.zhang/streaming_llm_polish.py` | FIFO 异步润色队列 | 可改 |
| `.zhang/offline_e2e.py` | 四阶段端到端自检器（**不是产品路径**） | 可改 |

### 1.2 下游 `Meeting_main`：默认只读，2026-10-18 有一次用户授权的同步

`D:\Deepseek\Meeting_main` 是另一条工作线。**默认不要改**：
`run_offline.py`、`stream_service.py`、`session_store.py`、`stream_llm_polish.py`、
`diar_web.html` 等仍按契约 `HANDOFF_OFFLINE_DIAR_AGENT.md` §1.2 处理。

**但 2026-10-18 用户明确授权并完成了以下同步**（覆盖当时“只读”约束，仅此一次）：

- `bootstrap.py::BASELINE` 刷新为上游封板指纹；
- `diar_backfill.py` 启用 `with_segments=True` 并把 `sub_turns` 写入 `diarization.json`；
- `turns.py` 新增 `split_utterance_by_sub_turns()`；
- `session_store.py` 在 `build_session_json` 中按 `sub_turns` 展开单人子段；
- `offline_asr.py` 禁止跨说话人拼段；
- 新增/更新对应单测；`Meeting_main/HANDOFF.md` §3.3 有完整记录。

**后续若还要改 `Meeting_main`，必须先拿到用户对具体文件的明确授权**；不要默认沿用这次授权。

> ⚠️ 用户原话（2026-10-18 同步前）：「上游 `diarization_cl-main/.zhang/` 已完成全部缺陷修复与收口封板，请立即在 `Meeting_main` 中执行以下同步与精度升级」——这是本次授权，不是长期规则。

---

## 2. 我们在做什么

### 2.1 产品目标

会议录音 → 流式转写（草稿上屏）→ 句尾异步 LLM 纠错固化 → **会后整场音频做全局说话人回填**。

我们负责最后一环和支撑它的流式底座。

### 2.2 回填接口（唯一契约）

```python
from offline_diarizer import backfill_speakers

mapping = backfill_speakers(
    wav_path,                    # 整场 16 kHz 单声道音频
    utterances,                  # [{"utt_id": "utt_0001", "start_ms": 0, "end_ms": 600}, ...]
    models=models,               # web_server.get_models(...) 的返回，含 "vad"/"spk"
    mode="auto",                 # auto | class | vad
    normalize_peak=0.85,         # 2026-10-18 起默认：只提拉弱音；0=关闭
    with_segments=False,         # True 时返回 {"mapping","segments","sub_turns"}
)
# 默认 -> {"utt_0001": "Speaker_1", ...}
```

`FULLSTACK_STREAMING_SPEC.md:89` 的旧签名仍兼容（新增参数都有默认值）。
`with_segments=True` 的富返回结构见 `HANDOFF_OFFLINE_DIAR_AGENT.md` §3.1；
`sub_turns` 只含非主说话人且持续 >=1.0s 的句内片段。

### 2.3 内部链路

```
fsmn-vad 切语音段
  → OSD(pyannote/segmentation-3.0 ONNX) 给出局部说话人区间，按 class 切 1.5s/0.75s chunk
  → CAM++ 提取 192 维嵌入
  → 谱聚类（k-NN 稀疏化 + eigengap 自动定 K）
  → 0.6s 密集短窗细化（长窗定身份、短窗找边界）
  → 句级主导投票（无短窗覆盖的句子退回长窗 fallback）
```

关键模块来自 `AliMeeting/run_3dspeaker_pipeline.py` 和 `claude_a2_v11.py` / `claude_a2_v12.py`。**不要重写，只复用。**

---

## 3. 已经完成了什么（带可核对证据）

### 3.1 8 场基线逐位复现 ✅

`AliMeeting/results/accept_8mtg_20261017.json` —— 44 个数字与 `docs/HANDOFF_1015_AliMeeting困境六收口.md` §1 **完全一致**：

| 指标 | 值 |
|---|---|
| 谱聚类 8 场均 | 54.0% |
| +细化 8 场均 | 60.44% |
| 细化增益 | +6.4 / +6.8 |

> ⚠️ 这是 **1015 冻结基线**（`--normalize-peak 0`、救援关闭前的历史口径）。
> 2026-10-18 解冻后的产品默认新基线见 §3.10：**54.03% / 61.01%**。

运行命令见 `docs/HANDOFF_1015_AliMeeting困境六收口.md` §9。

### 3.2 两种评测口径都测过，且**不可混用**

| 口径 | 谱聚类 | +细化 |
|---|---|---|
| TextGrid 句子口径 | 53.0% | 53.1% |
| VAD 句子口径 | **88.0%** | **82.9%** |

产品走的是 VAD 句子口径（`eval_multilabel.py` 的 `vad_nopost` 臂）。**报数时必须说清楚是哪把尺子。**

### 3.3 流式切句参数已拍板锁定 ✅

`max_end_silence_time = 1000ms` + `asr_reset_mode = "per_utt"`。

1000ms 复测：`vad_nopost` 88.0 → **89.4**（+1.4，在事先写死的 ±3 容忍内），TextGrid 臂不变。

四档扫描（450/650/800/1000）× 两种 cache 策略已完成，结论已归档。**不要再扫第二次。**

### 3.4 R8008_M8013 覆盖率之谜已收口 ✅

51.8% 的细化覆盖率根因是**输入电平**（peak 0.2635，8 场最低），不是算法缺陷。
`--normalize-peak 0.85` 后覆盖 24% → 51%，细化 50.8% → **54.5%（+3.7）**；
高电平对照 `R8007_M8010` **逐位不变**。

**默认值保持 0（关闭）** —— 开了就是另一套口径，与 §1 的基线数字不可混引。

### 3.5 端到端闭环跑通 ✅

`.zhang/offline_e2e.py` 四阶段（流式 → T+0 转写 → 回填 → 合成）：

```
33 句锁定 / 220 次草稿更新
说话人回填：33 句 → 2 人
总耗时 171.5s → output/sessions/s_smoke/session.json
```

### 3.6 8736 实机闭环验证 ✅（隔壁 Agent 的服务）

297 个事件、0 错误；`asr_final 32` = `llm_confirmed 32`；mapping 覆盖全部 utt_id；时间戳单调；最后一句在 180s 音频的 179.98s 正常 flush。

### 3.7 模块同名冲突已解决 ✅

`.zhang/run_offline.py` 与 `Meeting_main/run_offline.py` 同名会互相遮蔽 → 已改名 **`.zhang/offline_e2e.py`**。
全仓库确认无其他地方引用旧名。（`streaming_llm_polish.py` 与 `stream_llm_polish.py` 名字本就不同，从未冲突。）

### 3.8 代码审查（2026-10-18，`open-code-review` 两轮）

20 条发现，无 critical。**13 条已修（均为异常/边界路径），5 条记录不修**（会移动冻结基线）。
完整清单见 `docs/HANDOFF_1015_AliMeeting困境六收口.md` **§10**。

> ⚠️ 第一轮 OCR **只审了 `AliMeeting/` 的 2 个文件**，`.zhang/` 那 4 个（真正的交付物）因 LLM 超时静默失败；第二轮单独补审才拿到。**跑 OCR 后一定要看 summary 里的 "N of M selected item(s) failed"。**

### 3.10 2026-10-18 解冻整改（用户正式批准）

用户明确解冻 1015 基线：「做真实产品，不是博物馆里的静态数字」。本轮完成：

1. **`HANDOFF_1015` §10.2 的 5 条缺陷全部修复**：
   - `turns_multilabel()` / `sentence_pred_sets()`：全零 `share` 不再把簇 0 当预测
     （旧逻辑会让映射到簇 0 的说话人白送正确句，`acc` 虚高而 `coverage` 不涨）；
   - `run_3dspeaker_pipeline.py` 汇总：`refined=None` 不再按 `0.0` 计入均值；
   - `eval_multilabel.score_arm()`：`extras` 去掉 `set(list(P)[1:])` 的迭代序依赖；
   - `eval_multilabel.vad_sentences()`：TypeError 时先只丢 `dynamic_silence` 重试，
     再退回默认参数。
2. **生产回填默认弱音提拉**：`backfill_speakers(..., normalize_peak=0.85)`，
   **只提拉、不衰减**（`peak >= 0.85` 原样通过）；`normalize_peak=0` 可复现旧基线。
3. **句内切分**：`with_segments=True` 返回 `segments` / `sub_turns`；
   `sub_turns` 只含非主说话人且持续 >= 1.0s 的稳定片段，供下游切 Turn。
4. **密集窗救援：实现后按用户拍板默认关闭**（`DENSE_RESCUE_ENABLED=False`）。
   - 打开时：物业段 34 句 → 3 人（22/8/4），e2e `EXIT=0`、196.7s；仅长窗 K=2→3 触发。
   - **关闭后（最终默认）**：物业段回到自然 **2 人**；餐饮段（`b2d1…`）长窗本身
     就是 **3 人 / 11 切换**，不依赖救援。
   - 过切证据：R8001_M8004 长窗 K=4、密集窗 K=5 曾让细化 47.85→41.9，故救援即使
     打开也限制为 K=2→3。
   - **重要更正**：`smoke_clip_180s.wav`（物业，hash `b8f0…`）与
     `session_audio.wav`/`sweep_*.wav`（餐饮，hash `b2d1…`）是两段不同录音；
     旧/新 diarizer 在物业段都是 2 人，四档扫描的 3/11 属于餐饮段。见 §6 坑 16。
5. **清理**：临时 `.git` 已删除；改动前快照在 `milestones/diar_fix_20261018/`
   （含 6 个代码文件 + 3 份文档 + `SHA256_before.txt`）。
6. **新增测试**：`streaming_asr/tests/test_offline_diarizer_fixes.py`（6 条纯函数测试，
   `pytest` 全绿）。
7. **`eval_multilabel` 8 场复核**（`--vad-silence 1000`，产物
   `AliMeeting/results/ml_8mtg_fix_20261018.json`）：
   `vad_nopost` 89.42%、`vad_post` 82.63%、`tg_nopost` 52.99%、`tg_post` 53.05%、
   `tg_post_truek` 53.08%。与旧基线相比仅 `tg_post`/`tg_post_truek` **−0.02**，
   其余逐位一致，**无回归**。

**新基线（8 场，`--arm class --post --normalize-peak 0.85`）**：
产物 `AliMeeting/results/verify_fix_20261018_final.json`（两轮合并：R8007_M8010 /
R8008_M8013 来自收紧救援前的同一轮，二者 `dense_k <= 长窗 K`，未触发救援，数值不受影响）。

| meeting | 真值K | 长窗K | 密集K | 谱聚类 | +细化 | 旧+细化 | Δ细化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R8007_M8010 | 4 | 5 | 4 | 26.16% | 31.58% | 31.58% | 0.00 |
| R8008_M8013 | 3 | 3 | 3 | 50.14% | **54.51%** | 50.82% | **+3.69** |
| R8001_M8004 | 4 | 4 | 5* | 37.58% | 48.51% | 47.85% | +0.66 |
| R8003_M8001 | 4 | 4 | 4 | 41.99% | 50.89% | 50.89% | 0.00 |
| R8007_M8011 | 4 | 4 | 4 | 38.15% | 51.30% | 51.30% | 0.00 |
| R8009_M8018 | 2 | 2 | 2 | 73.38% | 79.10% | 79.10% | 0.00 |
| R8009_M8019 | 2 | 2 | 2 | 83.94% | 87.07% | 86.86% | +0.21 |
| R8009_M8020 | 2 | 2 | 2 | 80.89% | 85.13% | 85.13% | 0.00 |
| **平均** | | | | **54.03%** | **61.01%** | **60.44%** | **+0.57** |

`*` R8001 的密集窗估到 K=5，但触发条件已收紧为「仅长窗 K=2 时救援」，
故实际仍用长窗 K=4；K=5 曾把该场细化打到 41.9%，收紧后恢复 48.5%。

- 细化增益均值：**6.98 pt**（旧 6.39）；
- 归一化增益集中在低电平场：R8008 +3.69、R8001 +0.66、R8009_M8019 +0.21；
  高电平/未归一化场与旧基线**逐位一致**。

---

### 3.11 2026-10-18 下游 `Meeting_main` 同步与 `L_R003S01C02` 复核

用户明确授权（仅此一次）后，下游做了：

- `bootstrap.BASELINE` 刷新为上游封板指纹：
  `streaming_asr_engine.py` `5152424c04f4e572`/15831；
  `offline_diarizer.py` `eb454a8bf70d2081`/15061。
- `diar_backfill` 调 `backfill_speakers(..., with_segments=True)`，`diarization.json`
  新增可选 `sub_turns`；`session_store.build_session_json` 按 sub_turns 把长句展开成
  `utt_xxx#N` 单人子段；`turns.build_turns` 因此按子段说话人断开；
  `offline_asr._group_utterances` 禁止跨说话人拼段。
- 单测：**156 passed, 3 deselected**。

`L_R003S01C02`（39 分钟，AISHELL-4，真值 6 人）复核：

- 451 句 → **7 人**（K 多 1）；28 句 / 44 个 `sub_turns`；展开后 **530 句 / 182 turns**；
- 离线二遍 530 条 `final_text`，286 组、296 段、536 次 AGC，0 失败；
- CER（标准 Levenshtein，含口癖）：

| 版本 | CER | Sub | Del | Ins |
|---|---:|---:|---:|---:|
| 旧 v1（无 diarization） | 35.30% | 9.68% | 22.89% | 2.73% |
| **新 sub_turns** | **35.87%** | **10.31%** | **22.54%** | **3.02%** |

- 去口癖：旧 34.29% / 新 34.84%。
- 结论：sub_turns 让句内换人进入 turn 边界，但本场 CER 退化 **+0.57 pt**。
  未决取舍与下一步见 §4.1 / §5。

---

## 4. 当前卡在哪

### 4.1 🟡 唯一未决：`sub_turns` 用在 turns 还是也用于离线 ASR

- `L_R003S01C02` 新路径 CER **35.87%**（Sub 10.31 / Del 22.54 / Ins 3.02），
  旧 v1（无 diarization）35.30%（9.68 / 22.89 / 2.73）；去口癖 34.84 vs 34.29。
- `sub_turns` 让句内换人进入 turn 边界（530 句 / 182 turns），但 CER +0.57。
- 两种选择（**等用户拍板，不要擅自选**）：
  1. **turn 边界优先**：保留当前实现（同说话人才拼段 + 子段独立重转写）；
  2. **CER 优先**：`sub_turns` 只用于 turns，离线二遍恢复旧的 gap 拼段。
- 另一个可能因素：上游新链路在 AISHELL-4 上 **K=7**（真值 6）；先确认 K 是否
  应该压回 6，再做下一轮对比。**这属于下游验证，不是 AliMeeting 调参。**

### 4.2 ✅ 已完成（不要重做）

- 8 场新基线（§3.10）：**54.03 / 61.01**；
- 5 条 §10.2 缺陷、默认归一化、`segments/sub_turns`、救援默认关闭；
- 冒烟/最终 e2e：物业 2 人（`s_final_20261018`）；
- 餐饮段 3 人 / 11 切换；
- `Meeting_main` bootstrap 指纹同步 + `sub_turns` 接线（§3.11）；
- 临时 `.git` 删除；两份带日期快照（`milestones/diar_fix_20261018/`、
  `milestones/diar_final_20261018/`）。

### 4.3 🟢 已知但不修

- `Meeting_main` 的 `on_confirmed` 异常防护 / events 写入锁是对方代码，未确认。
- `vad_end_ms` 的 +250ms 钳位在测试片段上仍产生 1 处小偏差（可观测性低）。
- `AliMeeting` 的 5 条数字相关缺陷已于 2026-10-18 修复（§3.10），不再有挂起项。

---

## 5. 下一步计划（按顺序）

1. **等用户拍板 §4.1 的取舍**（turn 边界 vs CER）。
2. 若选 CER 优先：把 `session_store` 的 `sub_turns` 展开只用于 `build_turns`，
   离线二遍改回旧 gap 拼段；重跑 `L_R003S01C02` 的 `offline_asr + cer_eval`。
3. 若怀疑 K=7：单独验证 AISHELL-4 的 K（不在 AliMeeting 评测线上调参）；
   任何上游算法改动都要先拿用户授权并重跑 8 场基线。
4. 可选：用完整 `run_offline.py --wav L_R003S01C02.flac`（含流式 + diarization +
   离线二遍）复跑一次，确认 in-process 路径与手工三步一致；本轮只验证了
   `backfill → offline_asr → cer_eval` 的后半段。
5. **不做**：新的 AliMeeting 消融、为了 CER 调提示词、混用 LCS/Levenshtein 口径。

---

## 6. 踩过的坑（绝对不要踩）

### 🔴 红线

**坑 1：两把尺子不能混用。**
TextGrid 句子口径（53.0/53.1%）与 VAD 句子口径（88.0/82.9%）是**两个不同的任务定义**，不是同一个数的两种算法。产品走 VAD 口径。混引 = 造假。

**坑 2：不要动 `Meeting_main`（除非拿到用户对具体文件的明确授权）。**
那是另一条工作线，契约 `HANDOFF_OFFLINE_DIAR_AGENT.md` §1.2 默认只读。
2026-10-18 用户明确授权过一次同步（§1.2 / §3.11）；**那次授权只覆盖当时列出的文件，不自动延续**。
改 `.zhang/` 之后仍需通知对方刷指纹。

**坑 3：不要开新的 AliMeeting 消融实验。**
用户原话：「千万别答应它去跑那个拆分实验！」「我们不再在 AliMeeting 评测支线上做任何调参或发散，严防偏离产品主线！」
评测/诊断线已于 2026-10-17 封板。**任何"顺手做个对比实验"都是越界。**

**坑 4：这个仓库不是 git 仓库。**
没有版本控制。改文件前**必须**把原文件复制到带日期的目录。
临时 `.git` 已于 2026-10-18 删除。本轮快照：
`milestones/diar_fix_20261018/`（改前）、`milestones/diar_final_20261018/`（封板）。

**坑 5：默认值变更必须拿到用户明确批准。**
2026-10-18 用户批准：`backfill_speakers` 默认 `normalize_peak=0.85`（只提拉）、
`with_segments=False` 默认返回扁平结构；评测脚本 `--normalize-peak` 默认仍 0。
除此之外不要再“顺手”改默认值；1015 基线已不是产品默认（§3.10）。

### 🟠 实测教训

**坑 6：`fsmn-vad` 的 `max_end_silence_time` 必须显式传，否则阈值完全无效。**
`web_server.get_models()` 构造 VAD 时没带这个参数，必须在**调用期**补：
```python
vad.generate(input=..., max_end_silence_time=1000, ...)
```
这个 funasr 构建改过，不传就是模型默认值。`streaming_asr_engine._step_chunk` 里有完整的 `TypeError` 兜底写法，照抄。

**坑 7：`def f(x=MODULE_CONST)` 在定义时绑定。**
想通过改模块常量（`CHUNK_S`）影响默认值，必须写成 `x=None` 再在函数体里取。这个坑在 `class_chunks` 和 `_candidate_clusters` 上各踩过一次。

**坑 8：Python 必须用 funasr 环境的绝对路径。**
未 `conda init`，直接敲 `python` 会缺 `soundfile` 等依赖：
```
C:\Users\31817\miniconda3\envs\funasr\python.exe
```

**坑 9：中文日志必须设 `PYTHONIOENCODING=utf-8`。**
否则 GBK 乱码，`print` 里带 emoji 还会直接抛 `UnicodeEncodeError`。

**坑 10：`numexpr` 绝对不能装。**
`requirements.txt` 列了它，但装了会让哨兵测试变红。

**坑 11：`| tail` 会吞掉退出码。**
跑长任务一律：
```bash
cmd > log 2>&1; echo "EXIT=$?" >> log
```
再看 `tail log`。

**坑 12：OCR 的 `--output` 必须用，不要靠 stdout。**
大 diff 会截断，前面的审查意见全丢。而且**必须检查 summary 里的失败文件数** —— 本轮第一轮就静默丢了 4 个文件。

**坑 13：`ffmpeg` 缺失时必须以 NDJSON 报致命错误。**
`stream_service.py` 里有现成的处理范例，别自己发明。（该文件只读，照抄写法即可。）

**坑 14：回填失败必须退回单说话人，不能丢转写产物。**
`offline_e2e.py` 和 `diar_backfill.py` 都做了 `try/except` 兜底。改这块时不要"fail fast"到把 T+0 的转写也一起炸掉。

**坑 15：OSD 模型是社区导出，未必等于当年产生 60.44% 的那份。**
如果数字对不上，**先排除 OSD 导出差异，再谈算法差异**。路径 `D:\yinp\pyannote_seg3\model.onnx`（已有 `DIAR_OSD_ONNX` 环境变量覆盖）。

**坑 16：`smoke_clip_180s.wav`（物业会议）与 `session_audio.wav`/`sweep_*.wav`（餐饮剧本）是两段不同录音。**

- `smoke_clip_180s.wav`（hash `b8f0…`）：物业管理会议。旧 `s_smoke`（2026-10-17，
  旧 `OverlapAwareDiarizer`）与 `s_ocrfix`（2026-10-18，新链路）**都是 2 人**；
  e2e 的「34 句 → 2 人」说的是这一段，**不存在 3→2 回归**。
- `session_audio.wav` / `sweep_*.wav`（hash `b2d1…`）：餐饮行业剧本。用户记忆中的
  「四档扫描 3 人 / 11 切换」来自这一段，**不是 smoke_clip**。
- 餐饮段文本与 R8008_M8013 的 TextGrid 逐句吻合，但样本级相关只有 **0.007**、
  整场能量包络相关 **0.14**；它也只是**同剧本的另一场录制**，不能用 R8008 的
  RTTM 打分。2026-10-18 已据此纠正一次跨音频误判。

**坑 17：密集窗 K 救援默认关闭，不要随手打开。**
`DENSE_RESCUE_ENABLED=False`。打开时只在长窗 K=2→3 触发；R8001_M8004 的
K=4→5 实测让细化 47.85→41.9。物业段（`b8f0…`）自然 2 人，餐饮段（`b2d1…`）
长窗本身就是 3 人 / 11 切换；不需要救援。

**坑 18：`sub_turns` 子段 id 是 `utt_xxx#N`，`final_text` 必须用子段 id。**
`diarization.json` 的 `sub_turns` 只含非主说话人且 >=1.0s 的片段；
`session_store` 展开后，`offline_asr` 逐子段写 `final_text`。不要用父 `utt_id`
回填子段，否则整段文本会堆到第一段。

**坑 19：不同说话人的相邻子段绝不拼组。**
`offline_asr._group_utterances` 已禁止跨说话人拼段（含 short-group bridge）。
去掉这个限制会把句内换人重新吞回去，`sub_turns` 白做。

**坑 20：不要并行跑两个重模型管线。**
2026-10-18 实测：8 场跑批 + e2e 并行时，e2e 在加载 Paraformer 时 OOM
（`DefaultCPUAllocator`）。串行跑；先确认没有 Agent B 的 `offline_asr.py`
等大进程。

**坑 21：AISHELL-4 的 K=7 不是 6。**
新上游链路在 `L_R003S01C02` 上输出 7 人（TextGrid 真值 6 人）。不要把旧交接里
“K=6、归属 96.2%”当作当前读数；也不要把新 CER 35.87% 与旧 LCS 口径
（48.62/46.86/52.11/57.85）混引——旧口径已作废。

---

## 7. 环境与路径速查

| 项 | 值 |
|---|---|
| Python | `C:\Users\31817\miniconda3\envs\funasr\python.exe` |
| 编码 | `PYTHONIOENCODING=utf-8` |
| 语料 | `D:\yinp\AliMeeting\` |
| OSD 模型 | `D:\yinp\pyannote_seg3\model.onnx`（或 `$DIAR_OSD_ONNX`） |
| 冒烟音频 | `D:\Deepseek\yinp\smoke_clip_180s.wav`（物业段，2 人） |
| 餐饮段音频 | `D:\Deepseek\yinp\session_audio.wav` / `sweep_*.wav`（3 人 / 11 切换） |
| 基线结果 | `AliMeeting/results/accept_8mtg_20261017.json`（旧 1015） |
| 新基线结果 | `AliMeeting/results/verify_fix_20261018_final.json`、`ml_8mtg_fix_20261018.json` |
| 下游仓库 | `D:\Deepseek\Meeting_main`（默认只读；2026-10-18 授权同步过） |
| AISHELL-4 语料 | `D:\yinp\AISHELL-4\data\test\`（`wav/L_R003S01C02.flac`、`TextGrid/`） |
| ch0 新 session | `D:\Deepseek\Meeting_main\output\sessions\s_ch0_offline_subturns\` |
| 验收环境清单 | `docs/环境清单_新机器_20261017.md` |

---

## 8. 文件地图

```
D:\Deepseek\diarization_cl-main\
├── HANDOFF.md                        ← 本文件
├── HANDOFF_OFFLINE_DIAR_AGENT.md     ← 契约（红线在这，必读）
├── .zhang/
│   ├── offline_diarizer.py           ← 唯一交付接口 backfill_speakers()
│   ├── streaming_asr_engine.py       ← 流式底座
│   ├── streaming_llm_polish.py       ← 异步润色
│   └── offline_e2e.py                ← 四阶段自检器（非产品路径）
├── AliMeeting/
│   ├── run_3dspeaker_pipeline.py     ← 核心算法链（可复用，不再开实验）
│   ├── eval_multilabel.py            ← 评测器
│   └── results/                      ← 所有 JSON 产物
├── docs/
│   ├── HANDOFF_1015_AliMeeting困境六收口.md   ← 基线结论 + §10 审查记录
│   └── 环境清单_新机器_20261017.md
├── scripts/diagnostics/
│   ├── diag_smoke_k.py / diag_smoke_k2.py / diag_smoke_dense.py
│   │     ← 2026-10-18 冒烟 K 诊断（电平/OSD/eigengap/密集窗几何）
├── streaming_asr/tests/test_offline_diarizer_fixes.py
│        ← 新增纯函数测试（密集窗 K / 归一化 / sub_turns / 全零 share）
├── milestones/diar_fix_20261018/      ← 本次改动前的带日期快照 + SHA256
└── output/
    ├── sessions/                     ← 端到端产物
    └── ocr_round2.txt / ocr_round3.txt  ← 两轮审查原始输出

D:\Deepseek\Meeting_main\             ← 默认只读；2026-10-18 授权同步过
├── bootstrap.py                      ← BASELINE 已刷封板指纹
├── diar_backfill.py                  ← with_segments=True + sub_turns
├── turns.py                          ← split_utterance_by_sub_turns
├── session_store.py                  ← build_session_json 展开单人子段
├── offline_asr.py                    ← 禁止跨说话人拼段
├── cer_eval.py                       ← 标准 Levenshtein + Sub/Del/Ins
└── output/sessions/s_ch0_offline_subturns/  ← L_R003S01C02 复核
```

---

## 9. 接手后的第一件事

本支线已于 **2026-10-18 封板**（§10），**不要再开任何 AliMeeting 实验**。
先读 **§4.1**：唯一未决的是 `sub_turns` 用于 turns 还是也用于离线 ASR，用户尚未拍板。

若只是复核：

- 8 场新基线：`AliMeeting/results/verify_fix_20261018_final.json`
- eval 复核：`AliMeeting/results/ml_8mtg_fix_20261018.json`
- 物业段：`output/final_property_check.log`（**2 人**）、`output/sessions/s_final_20261018/`
- 餐饮段：`output/restaurant_rescue.log`（**3 人 / 11 切换**）
- 下游：`D:\Deepseek\Meeting_main\output\sessions\s_ch0_offline_subturns\`，
  `L_R003S01C02` 新 CER **35.87%**（Sub 10.31 / Del 22.54 / Ins 3.02），
  旧 v1 35.30%；`Meeting_main/HANDOFF.md` §3.3。

---

## 10. 封板（2026-10-18）

- **密集窗 K 救援默认关闭**：`DENSE_RESCUE_ENABLED=False`
  （`AliMeeting/run_3dspeaker_pipeline.py`）。物业段回到自然 2 人；
  餐饮段长窗本身就是 3 人/11 切换，不依赖救援。
- **最终指纹**（`Meeting_main/bootstrap.py::BASELINE` 已于 2026-10-18 刷新为下列值）：
  - `.zhang/offline_diarizer.py`：`eb454a8bf70d2081`，**15061 bytes**
  - `.zhang/streaming_asr_engine.py`：`5152424c04f4e572`，15831 bytes
- **最终快照**：`milestones/diar_final_20261018/`（含 `SHA256_final.txt`）。
- **下游同步（2026-10-18 用户授权）**：
  - `diar_backfill` 启用 `with_segments=True`，`sub_turns` 写入 `diarization.json`；
  - `turns`/`session_store` 按 `sub_turns` 展开单人子段；`offline_asr` 禁止跨说话人拼段；
  - `L_R003S01C02` 新 CER **35.87%**（Sub 10.31 / Del 22.54 / Ins 3.02），
    旧 v1 35.30%（+0.57）；取舍待 §4.1 拍板。
- **验收口径**：
  - 餐饮段 `session_audio.wav`/`sweep_*.wav`（hash `b2d1…`）：**3 人 / 11 切换**；
  - 物业段 `smoke_clip_180s.wav`（hash `b8f0…`）：自然 **2 人**；
  - 8 场 AliMeeting：谱聚类 **54.03%** / +细化 **61.01%**（旧 54.05 / 60.44）。
- 支线正式封板，**后续不再开启任何调参或消融实验**。
