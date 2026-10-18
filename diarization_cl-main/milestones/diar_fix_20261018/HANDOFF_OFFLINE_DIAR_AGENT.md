# 离线说话人识别子系统 —— 收口封板与接口契约规范

> **文档编号**：HANDOFF-OFFLINE-DIAR-20261017
> **放置位置**：`D:\Deepseek\diarization_cl-main\HANDOFF_OFFLINE_DIAR_AGENT.md`
> **适用对象**：离线说话人识别 / 声纹回填专职 Agent（Agent A）
> **配套文档**：`D:\Deepseek\Meeting_main\FULLSTACK_STREAMING_SPEC.md`（Agent B）
> **核对基准**：2026-10-17 仓库实况
> **阅读约定**：本文所有路径、函数签名、行号均已按当前仓库核对；与原交接稿不一致的地方集中在 §7「校订记录」。标注 **【核对注记】** 的句子，是仓库文档中找不到直接出处、引用前必须再核实的断言。

---

## 1. 角色定位与边界

你是本项目的 **【离线说话人识别与声纹回填专职 Agent】**。

全栈流式框架（流式 ASR、DeepSeek 句尾异步润色、Web 服务端与前端 UI）由另一个全栈 Agent（Agent B）负责，已并行推进。你的职责仅限于以下三件事：

1. **冻结评测资产**：维护并收口 `AliMeeting/` 的评测文档与基线产物，**正式停止一切新的评测消融与调参实验**。
2. **维护算法入口**：维护 `.zhang/offline_diarizer.py`，保证 `backfill_speakers()` 的签名、返回值和默认行为稳定。
3. **校验回填契约**：校验 `Meeting_main/diar_backfill.py` 对上层 session 契约的转换是否正确（该文件属于 Meeting_main 薄适配层，跨仓库改动需与 Agent B 协同）。

### 1.1 你负责的文件（以实际路径为准）

| 文件 | 实际路径 | 你的权限 |
|---|---|---|
| 离线回填算法 | `D:\Deepseek\diarization_cl-main\.zhang\offline_diarizer.py` | **可改**（算法 / 签名）。改后必须通知 Agent B，并刷新 `Meeting_main\bootstrap.py` 的上游指纹 |
| 回填 CLI / 契约适配 | `D:\Deepseek\Meeting_main\diar_backfill.py` | **只校验**契约与回填行为；跨仓库改动先与 Agent B 协同 |
| AliMeeting 评测资产 | `D:\Deepseek\diarization_cl-main\AliMeeting\` | 只维护文档、冻结基线；**不再新增消融实验** |
| 困境六收口记录 | `D:\Deepseek\diarization_cl-main\docs\HANDOFF_1015_AliMeeting困境六收口.md` | 只读引用，不重写历史结论 |

> **版本控制风险**：`D:\Deepseek\diarization_cl-main` **不是 git 仓库**（已核实无 `.git`）。修改 `offline_diarizer.py` 前，先把当前文件复制到带日期的快照目录（例如 `milestones/` 下新建 `offline_diarizer_YYYYMMDD.py`），并记录 SHA256。否则上游一变，无法回溯。

> **指纹联动**：`Meeting_main\bootstrap.py:24-28` 固定了 `streaming_asr_engine.py` 与 `offline_diarizer.py` 的 SHA256 前 16 位与字节数。你改动 `offline_diarizer.py` 后，`run_offline.py` 会打印上游漂移告警。**这是预期行为，不要为了消警而随手改 BASELINE**；先跑通验收，再由 Agent B 更新基线。

### 1.2 严禁越界（红线）

- ❌ 不修改 `Meeting_main` 的 `run_offline.py`、`session_store.py`、`stream_llm_polish.py`、`stream_service.py`、前端页面。
- ❌ 不修改 `D:\Deepseek\diarization_cl-main\streaming_asr\web_server.py` 与 `diar_web.html`；它们是上游只读资产。
- ❌ 不重新设计流式切句参数；`1000ms + per_utt` 已拍板（见 §2.4）。
- ❌ 不引入新的聚类后端（SortFormer / VBx / PLDA）或新的评测口径。
- ❌ 不改默认行为来“顺手提升”某个数字；任何默认值变更都视为破坏 1015 基线。

---

## 2. 已确认的技术结论（正式封板）

### 2.1 多人高重叠会议主线

**定型链路**（`AliMeeting/run_3dspeaker_pipeline.py`）：

```
class 区间切分
  → CAM++ 嵌入
  → 1.5s 谱聚类定身份（k-NN 图 + eigengap 定 K）
  → 0.6s / 0.15s 短窗细化找边界
  → 未覆盖句退回长窗投票（fallback=(主窗, 主窗标签)）
```

**已核实战果**：

| 结论 | 数值 | 出处 |
|---|---|---|
| R8008_M8013 fallback 修复 | **−10.2 → +0.3** | `docs/HANDOFF_1015_AliMeeting困境六收口.md:11,137` |
| 8 场细化均值 | **59.05% → 60.44%**（+1.39） | 同上 |
| `final8_refined_fb.json` 8 场 refined 均值 | **60.44%**（逐行复算一致） | `AliMeeting/results/final8_refined_fb.json` |
| 自选 K 命中真值 K | **7/8 场**（唯一偏差：R8007_M8010 真值 4、自选 5） | 同上，逐场 `true_k` / `k_auto` |
| R8008_M8013 细化覆盖率 | **51.8%**，8 场中唯一覆盖率崩掉的场 | `HANDOFF_1015:123` |

**R8007_M8010 K 塌陷问题**：在当前 `final8_refined_fb.json` 中，自选 `k_auto=5`、AHC `k_ahc=2`，**不再出现 K=1 的整场塌陷**。引用时请写清是“谱聚类自选 K 路径”，不要把它与 AHC 路径混为一谈。

**电平问题**：原交接稿称 `R8008_M8013` 的 `peak=0.2635, rms=0.003657`。仓库文档只能直接佐证：

- `docs/环境清单_新机器_20261017.md:261` 记录 8 场 ch0 的**峰值区间下界为 0.2635**（不是单场专属值）；
- `HANDOFF_1015:123` 记录该场细化覆盖率 **51.8%**。

**【核对注记】** 单场 `rms=0.003657` 在仓库文档中未找到直接出处。对外引用前，请先用 `prepare_ch0.py` 的输出或音频实测重新核对；不要把“全局峰值下界”误写成“R8008 单场峰值”。

### 2.2 `--normalize-peak` 的真实位置

`--normalize-peak` 只存在于 **评测脚本** `AliMeeting/run_3dspeaker_pipeline.py:431`（`default=0.0`，默认关闭），用于诊断 VAD 对绝对电平的敏感性。

**它不在生产回填接口里**：

- `.zhang/offline_diarizer.py::backfill_speakers()` **没有** `normalize_peak` 参数；
- `Meeting_main/diar_backfill.py` 的 CLI **没有** `--normalize-peak`；
- 因此 **不要**在交接契约里承诺 `--normalize-peak` 可用。

如果确实要把归一化接入生产回填，必须单开任务、加可选参数、默认关闭，并在 8 场上证明默认路径数字逐位不变。**在完成该任务前，任何“已落库到回填接口”的说法都是不成立的。**

### 2.3 双人对话支线（备选能力，未接入主线）

- 代码位置：`D:\Deepseek\diarization_cl-main\claude_a2.py`（仓库根）。
- 关键函数：`a2_segments`（约 `:261`）、`_dense_rescue`（约 `:97`）、`_stabilize_labels`（约 `:64`）、`_absorb_slivers`（约 `:182`）。
- 现状：`offline_diarizer.py` **只调用** `claude_a2_v11` / `claude_a2_v12` 的细化函数（`_vad_segs`、`_refine_windows`、`_refine_labels`），**没有调用 `a2_segments`**。
- 结论：双人支线保留为已知 2 人对话场景的备选能力；v0 主线默认单臂走多人新链路。**不要**在本轮把它强行接进 `backfill_speakers()`。

### 2.4 1000ms VAD 口径备案

流式切句已拍板 **`max_end_silence_time=1000ms` + `asr_reset_mode="per_utt"`**（`.zhang/streaming_asr_engine.py:63-65`）。

1000ms 口径的离线评测产物已生成：`AliMeeting/results/ml_8mtg_sil1000_20261017.json`。按 8 场逐行复算的均值如下：

| 臂 | mean `acc_single` | mean `coverage` |
|---|---:|---:|
| `vad_nopost` | **89.42%** | 98.00% |
| `vad_post` | **82.63%** | 99.86% |
| `tg_nopost` | 52.99% | 88.20% |
| `tg_post` | 53.06% | 99.85% |

> ⚠️ **引用口径**：这些是“换句子定义后的读数”，与 TextGrid 口径的 53.1% **不可互换**，也不得作为系统准确率对外报数。`HANDOFF_1015` §7 已明令禁止“VAD 句子口径 82.9% = 系统准确率”这类表述。此处只作**内部备案**，不再据此改算法。

### 2.5 明确停止的事项

- ❌ 不做“只归一化 VAD vs 归一化 CAM++ 嵌入”的拆分实验。
- ❌ 不做 T1b 双臂路由的 `eval_multilabel.py` 改造（该项属于全栈方案的测量任务，不属于离线封板范围）。
- ❌ 不碰 SortFormer、VBx / PLDA 旧支线。
- ❌ 不再新增任何调参轮次；现有 8 场产物即封板基线。

---

## 3. 对外交付的唯一契约

Agent B 会在会议流式转写结束时生成 session 目录（默认 `output/sessions/s_<timestamp>/`），其中你需要的输入只有两个文件：

1. **`session_audio.wav`**：16 kHz、16 bit、单声道完整会议录音。
2. **`utterances.jsonl`**：流式阶段按 `1000ms + per_utt` 锁死物理时间边界的句子列表，每行：

```json
{"utt_id": "utt_0001", "start_ms": 600, "end_ms": 3420, "raw_text": "那我们这个餐饮市场的前景真"}
```

### 3.1 交付物 1：`backfill_speakers()` 的真实签名

文件：`D:\Deepseek\diarization_cl-main\.zhang\offline_diarizer.py`

```python
def backfill_speakers(
    wav_path: str,
    utterances: List[Dict],
    models: Optional[Dict] = None,
    mode: str = "auto",
) -> Dict[str, str]:
    """整场音频 + 流式阶段锁死的句子 -> {utt_id: "Speaker_N"}。

    utterances: [{"utt_id": "utt_0001", "start_ms": 1200, "end_ms": 4500}, ...]
                start_ms / end_ms 必须是流式阶段按采样点计数锁死的绝对值。
    models:     web_server.get_models() 的返回（含 "vad" / "spk"）。None 时自取。
    mode:       "auto"（有 OSD 用 class，否则退化 vad）| "class" | "vad"
    """
```

**真实返回**：

```python
{
    "utt_0001": "Speaker_1",
    "utt_0002": "Speaker_2",
}
```

- 返回的是 **字符串标签** `"Speaker_{p+1}"`，其中 `p` 是 0-based 簇标签。
- 上游**不返回** `n_speakers`、`elapsed_sec`、`mapping`、`segments` 这些字段。
- 如果 Agent B 需要上述统计信息，必须由 Agent A **新增**一个函数（例如 `backfill_speakers_ex()`）或在不破坏旧签名的前提下扩展返回类型；**不能指望现有函数返回**。

**模型注入**：

```python
models = get_models(need_stream=False, need_offline=False)  # 含 "vad" / "spk"
mapping = backfill_speakers("session_audio.wav", utterances, models=models, mode="auto")
```

**mode 行为**：

| mode | 行为 | 缺失 OSD 时 |
|---|---|---|
| `auto` | 有 OSD 用 class 区间切窗，否则退化纯 VAD 窗 | 静默退化 |
| `class` | 强制 class 区间切窗 | 抛 `FileNotFoundError` |
| `vad` | 只用 VAD 窗 | 正常 |

OSD 路径：`D:\yinp\pyannote_seg3\model.onnx`（`offline_diarizer.py:30`）。缺它时 `mode="auto"` 的读数**不能**与 60.44% 基线比较（见 `docs/环境清单_新机器_20261017.md:286-292`）。

### 3.2 交付物 2：独立 CLI 的真实用法

文件：`D:\Deepseek\Meeting_main\diar_backfill.py`

```powershell
# 在 D:\Deepseek\Meeting_main 下运行
python diar_backfill.py --session output/sessions/s_20261017_1432
```

**真实参数**（`diar_backfill.py:119-124`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--session` | 必填 | session 目录，如 `output/sessions/s_xxx` |
| `--mode` | `auto` | `auto` / `class` / `vad` |
| `--force` | 关 | 忽略已存在的 `diarization.json`，强制重算 |

**没有** `--normalize-peak`。

**行为**：

1. `bootstrap.install()` 挂载上游路径；
2. `get_models(need_stream=False, need_offline=False)` 加载 VAD + CAM++（约 90 秒冷启动）；
3. 读取 `session_audio.wav` 与 `utterances.jsonl`（按 `utt_id` 去重；损坏行跳过；若一行有效的都没有，回退到 `transcript.json`）；
4. 调用 `backfill_speakers()`；
5. 用 `to_contract()` 把 `"Speaker_{p+1}"` 一次性换算成 `{"spk": p, "spk_label": "speaker_p"}`；
6. 写 `diarization.json`，并调用 `SessionStore.build_session_json()` 重建 `session.json`。

**已存在 `diarization.json` 且未加 `--force`**：直接复用，不重复计算。

### 3.3 输出文件形状（以 Meeting_main 契约为准）

`diarization.json`：

```json
{
  "utt_0001": {"spk": 0, "spk_label": "speaker_0"},
  "utt_0002": {"spk": 1, "spk_label": "speaker_1"}
}
```

`session.json` 中的单句（合并后的最终态）：

```json
{
  "utt_id": "utt_0001",
  "start_ms": 600,
  "end_ms": 3420,
  "raw_text": "那我们这个餐饮市场的前景真",
  "polished_text": "那我们这个餐饮市场的前景真不错。",
  "corrected": true,
  "bypassed": false,
  "error": null,
  "spk": 0,
  "spk_label": "speaker_0",
  "overlap_spk": null
}
```

> `spk` 是 **0-based 整数**；`spk_label` 恒为 `speaker_{spk}`（全小写）。`Speaker_{p+1}` → `spk=p` 的 off-by-one **只在 `diar_backfill.to_contract()` 换算一次**，前端和上层不得再减 1。

---

## 4. 验收清单

在 `D:\Deepseek\Meeting_main` 下执行：

```powershell
$env:PYTHONIOENCODING="utf-8"

# 1. 快速单测（不跑真实音频）
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest

# 2. 180 秒真实音频端到端（不启用 LLM，先验证转写 + 回填）
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav `
    --session-dir output/sessions/s_smoke_180 `
    --fast --no-llm

# 3. 独立重跑回填
C:\Users\31817\miniconda3\envs\funasr\python.exe diar_backfill.py `
    --session output/sessions/s_smoke_180 --force
```

**通过标准**：

- [ ] `output/sessions/s_smoke_180/session.json` 的 `status` 为 `diarized`；
- [ ] 每个 `utterances[]` 元素同时具备 `raw_text`、`polished_text`（`--no-llm` 时回退为 `raw_text`）、`spk`、`spk_label`；
- [ ] `spk_label` 全小写、无 `Speaker_1` 混入；
- [ ] `diarization.json` 的 key 与 `utterances.jsonl` 的 `utt_id` 一一对应；
- [ ] 日志中无 `ImportError`、无 `KeyError`、无 K 塌陷告警；
- [ ] `pytest` 快速套件全绿。

**算法自检（可选，不产生对外数字）**：

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe .zhang\offline_diarizer.py R8008_M8013
```

期望：`K = 3（未塌陷）`，并打印三名说话人的句数分布。

---

## 5. 交接声明

1. **职责隔离**：Agent A 对 `offline_diarizer.py` 的算法与签名负责；Agent B 对 `Meeting_main` 的服务层、落盘、前端负责。`diar_backfill.py` 是两者之间的薄适配层，任何跨仓库改动先沟通。
2. **接口冻结**：`backfill_speakers(wav_path, utterances, models=None, mode="auto") -> Dict[str, str]` 是当前唯一稳定入口。要改签名，必须同时提供旧签名兼容层，并通知 Agent B 重新跑 `pytest` + 180 秒冒烟。
3. **上游只读**：Agent B 依赖 `bootstrap.py` 的 `sys.path` 挂载；Agent A 不要移动 `.zhang/`、`AliMeeting/`、`streaming_asr/` 的目录结构。
4. **失败可恢复**：`diar_backfill.py --session <dir>` 必须能在崩溃后独立重跑，不依赖内存状态。

---

## 6. 待核实 / 未决（交接时必须保留）

| # | 事项 | 现状 | 处理建议 |
|---|---|---|---|
| 1 | “44 个指标数字逐位可复现” | **【核对注记】** 在 `docs/` 中未找到“44 个”这一出处的直接文字；可核实的基线是 `final8_refined_fb.json` 的 8 场 refined 均值 60.44% 与逐场读数 | 封板清单改为指向具体产物文件与字段，不再使用未注明出处的“44 个数字”表述 |
| 2 | R8008_M8013 单场 `peak / rms` | 只核实了 8 场峰值下界 0.2635 与覆盖率 51.8%；单场 `rms=0.003657` 无文档出处 | 引用前用 `prepare_ch0.py` 输出或音频实测复核 |
| 3 | `normalize_peak` 进生产回填 | **未实现**；只存在于评测脚本 `run_3dspeaker_pipeline.py:431` | 若要做，单开任务；默认关闭且必须证明默认路径数字不变 |
| 4 | 双人支线接入 | `claude_a2.a2_segments` 未接入 `offline_diarizer.py` | 保持备选；不在本轮强行接线 |
| 5 | OSD 导出出处 | `onnx-community` 社区导出未必等于当年产生 60.44% 的那份 | 复现偏差时先排除 OSD 差异，再谈算法差异（`环境清单_新机器_20261017.md:286-292`） |

---

## 7. 校订记录

本文件相对原交接稿的事实修正，逐条留痕。

| # | 原稿表述 | 仓库实况 | 处置 |
|---|---|---|---|
| 1 | 你维护 `.zhang/offline_diarizer.py` 与独立 CLI `diar_backfill.py` | `offline_diarizer.py` 在 `diarization_cl-main\.zhang\`；`diar_backfill.py` 在 `Meeting_main\`，依赖 `bootstrap` / `session_store` | §1.1 明确两者分属不同仓库；Agent A 只校验适配层 |
| 2 | `backfill_speakers(..., normalize_peak=0.0)`，返回 `{n_speakers, elapsed_sec, mapping, segments}` | 真实签名为 `(wav_path, utterances, models=None, mode="auto")`，返回 `{utt_id: "Speaker_{p+1}"}` | §3.1 改为真实签名与返回值；富结构需新增函数 |
| 3 | CLI 支持 `--normalize-peak 0.9` | 真实参数只有 `--session` / `--mode` / `--force` | §3.2 修正；`normalize_peak` 仅在评测脚本 |
| 4 | `diarization.json` 含 `n_speakers / mapping / segments` | 实际是扁平的 `{utt_id: {spk, spk_label}}` | §3.3 改为实际形状 |
| 5 | R8008 `peak=0.2635, rms=0.003657` | 只能核实 8 场峰值下界 0.2635 与覆盖率 51.8% | §2.1、§6 标为待核实，不当作既成事实 |
| 6 | “44 个指标数字逐位可复现” | 仓库未找到“44 个”直接出处；60.44% 与 7/8 K 命中已逐位核实 | §6 要求改指向具体产物文件 |
| 7 | `--normalize-peak` 已“落库”到回填链路 | 只存在于评测脚本，生产回填未暴露 | §2.2 明确纠正 |

---

## 8. 一句话总结

**你的历史使命是守住 `offline_diarizer.py` 的稳定签名与 1015 基线，而不是继续刷 AliMeeting 的数字。** 上层全栈如何编排、如何落盘、如何渲染，交给 Agent B；你只需要保证：给定 `session_audio.wav` + `utterances.jsonl`，`backfill_speakers()` 能稳定返回 `{utt_id: "Speaker_N"}`，并且 `diar_backfill.py --session` 能在崩溃后独立重跑。
