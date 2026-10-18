# 真实麦克风基线评测设计

> 状态：已确认（2026-10-18）
> 项目：`D:\Deepseek\Meeting_main`
> 目标：用真实房间、3–6 人、Intel 阵列，测一次当前流水线的 CER/Sub/Del/Ins，
> 判断 AISHELL-4 的 35% 是否代表产品真实水平。

## 1. 背景

2026-10-18 的新口径评测（标准 Levenshtein + 精确 Sub/Del/Ins）显示：

- AISHELL-4 `L_R003S01C02`：在线 raw 36.90%，离线 v1（AGC+4–15s 长段）35.30%。
- Del 22.89%；VAD 覆盖 <50% 的区间贡献 35.4% 的 Del。
- 参考 49.33% 的字落在跨说话人重叠区。

但 AISHELL-4 是 6 人远场阵列语料，重叠率极高。真实产品是单支远场麦、普通会议室；
先测真实场景，避免继续优化错误的测试集。

## 2. 范围与非目标

**范围**

- 10 分钟真实会议录音（已完成）的 16k 预处理。
- 2 分钟窗口的人工修正真值与 CER 评测。
- `cer_eval.py` 增加 `--ref-txt`、`--window`、`--dump-draft` 三个能力。

**非目标**

- 不改流式/离线 ASR 模型与算法。
- 不做整场人工标注（只标 2 分钟）。
- 不把 Intel 阵列结果当成“单支远场麦”产品结论；它是当前硬件的上界。
- 不实现重叠感知 ASR。

## 3. 录音与预处理

**已完成**

- 设备：Intel 内置麦克风阵列（Windows 音频端点，DirectShow 名称
  `麦克风阵列 (适用于数字麦克风的英特尔® 智音技术)`）。
- 原始文件：`output/real_meeting/intel_meeting_10min_48k.wav`
  - 48 kHz / 2 ch / PCM_16 / 599.89 s；实测 ch0 与 ch1 完全相同。
- 16k 文件：`output/real_meeting/intel_meeting_10min_16k.wav`
  - ffmpeg `-ac 2 -ar 16000`，保留 2 声道，不下混；`run_offline` 取 ch0。
- 电平：整体 RMS -29.8 dBFS，语音中位 -39.2 dBFS，活跃帧 67.7%；
  削波仅 0.0072%（集中在最后 30 秒的一次短点击）。

**约束**

- 前端/流水线仍然只允许 16 kHz 输入。
- 不修改 `session_audio.wav`；它是采样点时钟的唯一真相。

## 4. 评测窗口与真值

**窗口**

- 默认 `60–180 s`（会议开始后 1–3 分钟）。
- 若该窗口语音过少或没有多人重叠，允许改到含 2 人以上的相邻窗口，
  但必须在报告中写明实际窗口。
- **窗口粒度**：窗口评测与草稿都按 **utterances** 逐句过滤（中点），不使用 `turns`。
  真实 10 分钟会议在 `--no-diar` 下只有 3 个 turn，`turn_0001` 跨 42.7–552.0s；
  按中点过滤 60–180s 会得到 0 个 turn。`turns` 只用于整场展示。

**真值流程：draft-and-correct**

1. 用当前流水线跑完整 10 分钟，得到 `session.json.turns`（无说话人时用 `utterances`）。
2. `--dump-draft` 导出窗口内的带时间戳草稿。
3. 人工听录音，逐句修正草稿文本；保留时间戳与说话人标签。
4. 对 30 秒做一次盲听复核，记录“照抄草稿”的偏差（定性即可）。

**参考文件格式**

UTF-8 文本，一行一个片段：

```text
# 以 # 开头的行为注释
[01:00.0-01:04.2] speaker_1: 大家好，今天我们把这个问题过一下。
[01:04.2-01:08.0] speaker_2: 我先说一下背景。
```

规则：

- `[start-end]` 可省略；省略时整行按纯文本处理。
- `speaker:` 可省略；省略时该行没有说话人标签。
- 时间格式支持 `mm:ss.s`、`hh:mm:ss.s`、纯毫秒整数。
- 有标签时，文本取 `]` 之后第一个 `:` 或 `：` 后的内容；否则整行都是文本。
- 空行忽略。
- 口癖（呃/嗯/额/哦）按实际听到的逐字写入；报告同时给出“含口癖”与
  “去口癖”两个 CER，主口径为含口癖（与 AISHELL-4 一致）。

## 5. 工具设计

### 5.1 `cer_eval.py --ref-txt`

新增参数：

```text
--ref-txt PATH       使用纯文本参考，不读 TextGrid
--window START-END   只评该时间窗（支持 mm:ss / 毫秒）
--dump-draft PATH    从 session.json 导出该窗口的草稿后退出（不评测）
```

行为：

- `--ref-txt` 与 TextGrid 二选一；只有实际评测时两者都缺失才报错。
  `--dump-draft` 模式不需要参考输入，导出后直接退出。
- `--window` 缺省为全片；参考片段与假设片段都只保留**中点落在窗口内**的项。
  中点定义为 `(start_ms + end_ms) // 2`。
- **使用 `--window` 时，参考行必须全部带时间戳**；纯文本参考（无时间戳）
  只支持全片评测，否则窗口过滤无从判断。
- 假设文本：**窗口模式**用 `utterances[]`（按中点过滤）逐句拼接，
  主口径为 `final_fallback`（`final_text` 优先，缺失回退 `polished_text`/`raw_text`）；
  `final` 保留 strict 诊断口径（只统计真正有 `final_text` 的句子）；
  **全片模式**保持 `turns` 优先（与现有 TextGrid 行为一致）。
- 参考文本拼接顺序：所有非注释行都带时间戳时按 start 排序；
  否则保持文件行序（此时不能用 `--window`）。
- 指标复用现有 `edit_counts` + `EditCounts`；输出 CER / Sub / Del / Ins。
- 若参考片段带时间戳，额外计算 VAD 覆盖率；若带 speaker 标签，额外计算
  跨说话人重叠率（同 tier 不重叠、不同 tier 时间相交）。
- `--json` 输出与现有结构兼容，新增 `reference_source: "txt"` 字段。

### 5.2 `--dump-draft`

```text
cer_eval.py --session DIR --window 60000-180000 --dump-draft draft.txt
```

- 读取 `session.json`；**优先 `utterances[]`**（逐句，适合人工修正），
  没有 utterances 时才回退 `turns[]`。
- 每行输出 `[mm:ss.s-mm:ss.s] speaker_N: text`；无说话人时写 `speaker_?`。
- 导出时去掉汉字之间的空格（SEACO 输出为空格分隔 token）；CER 归一化本来就去空白，
  这一步只影响草稿可读性。
- 导出的文本是**草稿**，不是真值；人工修正后才用于 `--ref-txt`。
- 该模式不加载任何模型，不写 session 文件。

### 5.3 数据流

```text
intel_meeting_10min_16k.wav
        │ run_offline.py --fast --no-llm --no-diar
        ▼
output/sessions/s_real_intel/session.json
        │ cer_eval.py --dump-draft --window 60000-180000
        ▼
draft.txt ──人工修正──► ref.txt
        │ cer_eval.py --ref-txt ref.txt --window 60000-180000
        ▼
CER / Sub / Del / Ins + 覆盖率/重叠归因
```

## 6. 指标与报告

主表：

| 指标 | 含义 |
|---|---|
| CER | (Sub+Del+Ins)/参考字数 |
| Sub / Del / Ins | 各自比率与计数 |
| 参考字数 / 假设字数 | 归一化后长度 |
| 口癖 CER | 去口癖后的 CER，仅作参考 |

若参考带时间戳/说话人：

- VAD 未覆盖参考字数与比例。
- 跨说话人重叠参考字数与比例。
- 按说话人的 Sub/Del/Ins。

报告必须注明：

- 这是 2 分钟样本，不是整场结论。
- Intel 阵列是波束形成后的单路输出，是当前硬件上界。
- 人工修正可能带有锚定偏差。

## 7. 测试计划

新增 `tests/test_cer_ref_txt.py`（不加载模型）：

1. `[mm:ss.s-mm:ss.s] speaker: text` 解析正确。
2. 无时间戳/无说话人的纯文本解析正确。
3. `--window` 中点过滤正确（边界值 0.5 项）。
4. **turn 跨窗口时，窗口内 `turns=0`，但 `final_fallback`（utterance 级）仍有正确 CER；
   `final` 是 strict 诊断口径。**
5. `--dump-draft` 优先 utterances，导出行数与时间格式正确，且去掉汉字间空格。
6. 合成参考/假设的 CER、Sub、Del、Ins 与手算一致。
7. TextGrid 路径保持现有行为（回归测试）。

现有 131 个快速测试必须继续通过。

## 8. 验收标准

- `cer_eval.py --ref-txt ... --window ...` 对 2 分钟窗口输出 CER/Sub/Del/Ins；
  **主口径是 `versions.final_fallback`（utterance 级，final_text 优先、缺失回退）**，
  `turns` 在窗口模式下可能为空；`versions.final` 是 strict 诊断口径。
- 真实会议基线数字写入 `HANDOFF.md`，并注明窗口、设备、参考来源。
- 不修改 TextGrid 路径的现有读数。
- `--dump-draft` 生成的草稿可直接人工修正，无需额外脚本。

## 9. 风险与限制

- **锚定偏差**：修正者可能漏掉 ASR 错误；用 30 秒盲听复核。
- **样本偏差**：2 分钟可能不代表整场；报告注明。
- **设备偏差**：Intel 阵列不是单支麦；结论不能直接外推到产品。
- **边界近似**：窗口按中点过滤，边界句可能被排除；选择窗口时尽量落在停顿处。
- **口癖口径**：含口癖 CER 与 AISHELL-4 一致；去口癖 CER 仅作参考，不得混用。
- **说话人标签**：`--no-diar` 首跑没有 speaker 标签；人工修正时可手填，
  或后续重跑 diarization。

## 10. 文件清单

- `cer_eval.py`：新增 `--ref-txt` / `--window` / `--dump-draft`。
- `tests/test_cer_ref_txt.py`：新增单元测试。
- `output/real_meeting/intel_meeting_10min_16k.wav`：真实会议 16k 输入。
- `output/sessions/s_real_intel/`：流水线输出。
- `output/real_meeting/ref_60_180.txt`：人工修正后的 2 分钟真值。
- `HANDOFF.md`：新增真实基线一节。
