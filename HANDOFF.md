# HANDOFF — 会议流式转写（Meeting_main）

> 写给一个完全没有上下文的新对话。先读这一份，再读 `README.md`、`STREAMING_PLAN.md`、
> `.superpowers/sdd/progress.md`。最后更新：2026-10-18。

## 0. 一句话现状

用户对效果不满意，这是对的：**最终稿一直由在线流式 ASR 产生，LLM 纠错只值 1.4 个百分点**。
已完成“会后离线二遍 ASR + 多声道取 ch0”改造，并跑通 ch0 端到端：
**raw 52.11% → 最终稿 46.86%（去 `sil`）**。仍远未到商业级。
不要回到“继续调 LLM 提示词”这条路上。

> **2026-10-18 评测口径更正（必读）**：上面的 52.11% / 46.86% 和 §3.2 的旧数字都是
> **旧口径**（`difflib` LCS indel，把一次替换记成 2 次编辑），系统性高估 10–15 点。
> 标准 Levenshtein 重算（同一份 ch0、同一批 451 句、参考已去 `<sil>`）：
> 在线 raw **36.90%**、离线 v1（AGC+4–15s 长段）**35.30%**、+LLM strict **35.24%**。
> **离线二遍只赢 1.6 点，LLM 再赢 0.06 点（噪声）**；增益几乎全部来自长段拼接，
> AGC 无收益。详见 §3.2 更正块。

## 1. 我们在做什么

在 `D:\Deepseek\Meeting_main` 做一套会议流式转写 + 会后定稿系统：

- 直播：流式 VAD + Paraformer-Streaming 出草稿；句尾 DeepSeek 保守纠错。
- 会后：说话人回填（复用上游 `diarization_cl-main` 的 CAM++/谱聚类）。
- **最终稿（新）**：会后用本地离线 SEACO Paraformer 按 VAD 句子重转写，作为最终文本。
- 产出 `output/sessions/s_*/` 目录；CLI + 127.0.0.1:8736 回放 HTTP 服务。

硬约束：

- `diarization_cl-main` **只读**，另一个 AI 正在实时改它，绝对不要修改、不要提交它的改动、
  不要杀它的进程。
- 不得写死 API key；`DEEPSEEK_API_KEY` 从进程环境或 Windows 用户级注册表读。
- 绑定只允许 `127.0.0.1:8736`。
- 用户已选：**最终稿只能本地**（成本原因）；**单支远场麦**；下一步=离线二遍 + ch0。

## 2. 用户的工作规则（必须遵守）

1. 先检查有没有错误前提、逻辑跳跃或信息缺失。
2. 不要迎合我，要独立判断。
3. 涉及数字、人物和结论时，尽量核实来源。
4. 如果我说得不对，直接指出来，并说明依据和风险。
5. 主动提醒我忽略的变量、成本和偏差。

## 3. 已经完成什么

### 3.1 代码

| 文件 | 作用 |
|---|---|
| `bootstrap.py` | 唯一上游耦合点；把 `diarization_cl-main` 挂到 `sys.path`，强制本目录优先，记录上游指纹基线 |
| `turns.py` | `build_turns`（最终 turn 合并，8s 阈值）、`StreamTurnTracker`（直播 5s 分组）；文本优先级 **final_text > polished_text > raw_text** |
| `stream_llm_polish.py` | 直播句尾异步 DeepSeek 纠错；单消费者 FIFO；守卫 1.5/0.7（2026-10-18 **移除数字集合守卫**）；strict 档非标点字数一致；只把高置信度结果作上文；`MAX_CONTEXT_CHARS=800`；当前句不得重复进“上文” |
| `final_polish.py` | 会后按 speaker turn 的 LLM 精修；守卫 **1.35/0.65**（2026-10-18 由 1.2/0.85 放宽）；strict 档字数一致；输出带 `start_ms/end_ms`；`max_tokens=4096`；abort-aware |
| `offline_asr.py` | **新增**：会后离线二遍 ASR；**AGC**（0.002<rms<0.04→0.05，peak 0.95）+ 相邻短句拼 **4–15s 长段** + 可选 `hotword` + 独立重跑 CLI；两侧 pad 80ms、失败/空段回退直播稿 |
| `run_offline.py` | 端到端 CLI；`no_final_asr` / `offline_asr_fn`；多声道允许并取 ch0；会后顺序：回填 → 收尾润色 → 离线二遍 → LLM 精修 |
| `session_store.py` | 六文件契约 + `final_text.json` + `turns_polished.json`；原子写、合并锁；`turns_polished` 校验 turn 边界 |
| `diar_backfill.py` | 单臂回填；`Speaker_{p+1}` → `spk=p`、`spk_label=speaker_p`，只换算一次 |
| `stream_service.py` | 8736 NDJSON 回放；上传转码改为 `pan=mono|c0=c0`（**不下混**）；单实例绑定 |
| `meeting_bridge.js` / `diar_web.html` | 前端桥接与逐句样式；最终渲染走 `speaker_diarized.turns` |
| `cer_eval.py` | **2026-10-18 重写**：标准 Levenshtein + 精确 Sub/Del/Ins（rapidfuzz）；TN 去 `<sil>`/统一数字；VAD 覆盖/跨说话人重叠/说话人归因；`--legacy` 仅作历史对照 |

测试：**131 passed, 3 deselected**（`pytest -q`，在 `Meeting_main` 下用 funasr 解释器）。
新增覆盖：`tests/test_offline_asr.py`（AGC/长段/文本回填）、`final_text` 合并、
run_offline 的离线二遍成功/失败/关闭、多声道取 ch0、strict 档字数守卫、脏上文隔离。

### 3.2 关键实测数据（2026-10-18 重新标定）

> ⚠️ **下表旧数字全部作废（LCS indel 口径）。** 旧口径把一次替换记成 2 次编辑，
> 系统性高估 10–15 点。标准 Levenshtein 重算（同一份 ch0、同一批 451 句、
> 参考已去 `<sil>`）：
>
> | 版本 | CER | Sub | Del | Ins |
> |---|---:|---:|---:|---:|
> | 在线流式 raw | 36.90% | 16.25% | 16.58% | 4.07% |
> | 离线二遍 v0（无 AGC/无长段） | 36.49% | 10.98% | 22.69% | 2.82% |
> | 离线二遍 v1（AGC + 4–15s 长段） | **35.30%** | 9.68% | 22.89% | 2.73% |
> | v1 + 会后 LLM strict（15/17 接受） | 35.24% | 9.61% | 22.90% | 2.73% |
> | 仅长段拼接（无 AGC） | 35.22% | 9.69% | 22.83% | 2.70% |
>
> - 离线二遍相对流式 raw 只赢 **1.6 点**，不是旧口径下的 5–11 点；
>   增益几乎全部来自 **4–15s 长段拼接**，AGC 在本数据上无收益（35.22 vs 35.30）。
> - 离线二遍是“用 Del 换 Sub”：Sub 16.25%→9.68%，Del 16.58%→22.89%，
>   净收益很小。最终稿规范化长度 6929 字，比 raw 7592 少 663 字。
> - Del 归因（精确 editops 映射）：VAD 0% 覆盖区间只有 97 字 / 88 Del（占全部 Del 4.4%）；
>   但覆盖率 <50% 的区间有 1383 字 / 704 Del（35.4%）。覆盖率梯度：
>   0%→CER 95.9%，0–25%→79.5%，25–50%→58.0%，50–75%→37.3%，
>   75–100%→22.4%，100%→34.1%。**VAD 漏检是重要因素，但不是 Del 的唯一主因**；
>   即使 100% 覆盖区间仍有 20.0% Del。
> - 跨说话人重叠占参考 **49.33%**；重叠区 CER 38.80% vs 非重叠 31.89%。
> - 参考口癖（呃/嗯/额/哦）245 字（2.82%），ASR 不转写就会被算成 Del。
> - 说话人差异极大：004-M 16.10%，006-M 43.00%，003-F 68.88%（Del 53.78%）。
>
> 旧表保留在下方，仅供追溯，**不得再对外引用**。

39 分钟 AISHELL-4 `L_R003S01C02`；TextGrid 6 个说话人；参考已去 `sil`。

| 版本 | CER | 备注 |
|---|---:|---|
| 在线流式 raw（8 声道下混） | 57.85% | 含 `sil` 时是 59.36% |
| 在线 + A + B 全部 LLM 纠错 | 57.91% | LLM 基本没有收益 |
| 离线 SEACO 逐句（同一份下混音频、同一批 VAD 句子） | **50.69%** | 149 秒跑完 39 分钟 |
| 在线流式 raw（ch0，端到端 39 分钟） | 52.11% | 451 句；比下混 raw 好 5.7 个点 |
| **离线二遍最终稿（ch0，端到端）** | **46.86%** | 451/451 句有 `final_text`；整场 1323 秒 |
| 在线单句（ch0，24 句真值句） | 57.7% | 同一段音频 |
| 离线 SEACO 单句（ch0） | **42.1%** | 离线赢 16/24 句 |
| ch0 vs 下混（2 分钟窗口，流式） | 51.1% vs 62.2%；77.5% vs 83.9% | 下混代价 6–11 个点 |
| FRCRN 降噪后（ch0） | 流式 63.0%（原 51.1%）；SEACO 66.0%（原 45.1%） | 降噪反而更差 |
| 说话人归属 | 96.2%（442/453 句可判） | K=6 与真值 6 人一致；不要重做 diarization |

其他：

- 逐 2 分钟 CER 在 35%–84% 之间；最差窗口与低 RMS 相关（1080–1200s RMS 0.0081、CER 84%）。
- FRCRN 在 MX450 2GB 上 **120 秒音频要 467 秒**，不可用于产品。
- 离线二遍 453 句耗时 148 秒；完全塞得进现有会后等待。
- 会话音频与 8 声道 downmix 相关系数 0.999、与 ch0 0.972 ⇒ 旧会话确实是下混。

### 3.3 2026-10-18 句内换人（sub_turns）接线与 ch0 复核

- `bootstrap.BASELINE` 已刷新到上游封板指纹：
  `streaming_asr_engine.py` `5152424c04f4e572`/15831；
  `offline_diarizer.py` `eb454a8bf70d2081`/15061。
- `diar_backfill` 调 `backfill_speakers(..., with_segments=True)`，`diarization.json`
  新增可选 `sub_turns`；`session_store.build_session_json` 按 sub_turns 把长句展开成
  `utt_xxx#N` 单人子段（文本按时长比例切，离线二遍逐子段覆盖 `final_text`）；
  `turns.build_turns` 因此按子段说话人断开；`offline_asr._group_utterances` 禁止
  跨说话人拼段。
- 单测：**156 passed, 3 deselected**。
- `L_R003S01C02` 复核（`output/sessions/s_ch0_offline_subturns`）：
  - 451 句 → **7 人**（真值 6 人，K 多 1）；28 句 / 44 个 `sub_turns`；
  - 展开后 **530 句 / 182 turns / 530 条 final_text**；离线二遍 286 组、296 段、
    536 次 AGC，0 失败；
  - CER（标准 Levenshtein，含口癖）：

    | 版本 | CER | Sub | Del | Ins |
    |---|---:|---:|---:|---:|
    | 旧 v1（无 diarization） | 35.30% | 9.68% | 22.89% | 2.73% |
    | **新 sub_turns** | **35.87%** | **10.31%** | **22.54%** | **3.02%** |

    - 去口癖：旧 34.29% / 新 34.84%。
  - 结论：sub_turns 让句内换人进入 turn 边界，但本场 CER 退化 **+0.57 pt**
    （Sub/Ins 增加、Del 减少）；K=7 多分一人和“同说话人才拼段”都可能是原因。
    若 CER 优先，下一步应把 sub_turns 只用于 turns、离线二遍保持旧拼段，
    或先让 AISHELL-4 的 K 回到 6（需另行验证，本轮不做）。

## 4. 当前卡在哪

1. **精度离商业级还很远**：ch0 + 离线二遍的全片 CER 新口径 **35.30%**（旧口径 46.86%），
   商业转写通常要 <10%–15%。单支远场麦 + 本地模型可能到不了。
   Del 占 22.89%；VAD 覆盖不足（<50% 覆盖区间贡献 35.4% 的 Del）与跨说话人重叠
   （占参考 49.33%）是主要结构性因素。
2. **本地模型有天花板**：单支远场麦 + 2GB 显存 + 16GB RAM；Fun-ASR-Nano 是上游选定的离线
   后端（2.15GB 模型，含 Qwen3-0.6B），本机未下载、未验证；上游期望路径
   `C:\Users\EDY\.cache\modelscope\...` 在本机不存在。
3. **FRCRN 不是解药**：实测更差且极慢，默认关闭。
4. **B（会后 LLM 精修）还没在离线新基座上重测**：现在它会作用在 `final_text` 生成的 turns 上，
   增益未知；如果不明显，应该默认关闭，避免额外 API 成本与引入新错。
5. **上游正在漂移**：2026-10-18 启动时检测到
   `streaming_asr_engine.py` e69c0ee6 14658B（基线 f65dd78a 14087B）、
   `offline_diarizer.py` 4336c583 6407B（基线 8fd519e6 6357B）。结果可能不可复现；
   不要改上游，必要时只更新本项目 `bootstrap.py` 的基线指纹并记录。
6. **前端最终稿展示尚未人工验收**：`run()` 返回后服务读 `session.json` 发
   `speaker_diarized.turns`，理论上已包含 `final_text`；需要打开页面确认。

## 5. 下一步计划（按优先级）

**P0（已完成，口径已更正）**：ch0 端到端跑通。

- `output/sessions/s_ch0_offline/`：451 句，451 条 `final_text`；
  旧口径 raw 52.11% / 最终稿 46.86%；**新口径 raw 36.90% / v1 35.30%**（见 §3.2）；
  整场 1323 秒（含流式 + 离线二遍）。
- 复现：`run_offline.py --wav L_R003S01C02.flac --fast --no-llm --no-diar`。
- 注意 `cer_eval.py` 的 `turns` 行含 `sil`（实测 48.62%）；对外统一用去 `sil` 的 46.86%，
  并在汇报里注明口径。
- 下一步直接做 P1；重跑请换新 session 目录，避免继承旧派生文件。

**P1**：把最终稿接到前端。

- 页面会收到 `progress.stage="final_asr"`（文案“最终稿离线转写中…”）；确认进度条不误导。
- `speaker_diarized.turns` 应显示 `final_text`；人工对比直播稿与最终稿的差异。
- 明确区分“直播草稿”和“最终稿”，不要让用户把草稿当最终结果。

**P2**：在离线新基座上重测 B（`final_polish.py`）。

- 对同一场跑 `--no-final-polish` 与默认，比较 CER；没有稳定收益就默认关闭 B。
- LLM 只在最终稿上做同音字/标点，不要重写。
- **2026-10-18 已重测**：strict 档 15/17 接受，CER 35.30→35.24（**-0.06 pt，噪声**）；
  正常档只接受 3/17，CER 无变化。建议默认关闭 B，除非换更强的模型/提示词。

**P3**：评估 Fun-ASR-Nano（本地，2.15GB）。

- 下载前先确认 RAM/显存与推理速度；建议只在会后加载，先卸载流式模型（上游有 `unload_models()`）。
- 只测“逐句离线”路径，不要同时加载 VAD+CAM+++Streaming+Nano。
- 如果 CPU 上 39 分钟音频要跑几小时，直接放弃，继续用 SEACO。

**P4**：真实麦克风验证。

- 用目标单支远场麦录 10 分钟真实会议，人工标注 2 分钟，算 CER。
- AISHELL-4 是远场压力测试，绝对值不能直接搬；但“最终稿必须离线二遍”结论与麦克风无关。

**P5**：商业目标决策。

- 若目标真的是 CER 10%–15%：单支远场麦 + 本机本地模型基本达不到，需要更好拾音
  （近场/阵列）或云端/服务器 ASR，二者至少选一个。不要用提示词硬扛。

## 6. 绝对不要踩的坑

1. **不要修改 `D:\Deepseek\diarization_cl-main`**。另一个 AI 在实时改；只读。不要提交它的
   未提交改动，不要杀它的进程（曾见 PID 8964；8736 上的服务也可能是旧的）。
2. **不要用 `python`（3.13）跑本项目**：没有 `soundfile`。用
   `C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12）。
3. **不要在 `D:\Deepseek` 根目录跑 pytest**：会把上游测试一起收集，报 23 个 collection
   error。必须 `cd D:\Deepseek\Meeting_main` 再跑。
4. **不要把多声道下混成单声道**。上传转码已改成 `pan=mono|c0=c0`；`_validate_wav` 允许
   多声道但只取 ch0。`-ac 1` 会让 CER 差 6–11 个点。
5. **不要启用 FRCRN 作为默认降噪**：本数据上更差（51.1→63.0），而且 2GB GPU 上极慢。
6. **不要把直播流式稿当最终稿**。最终稿来自 `final_text.json` / `offline_asr.py`。
7. **不要用带 `sil` 的 CER 做对比，也不要混用 LCS 与 Levenshtein 口径**。
   `cer_eval.py` 2026-10-18 起默认剥离 `<sil>`/`<%>` 并改用标准 Levenshtein；
   旧读数（48.62% / 46.86% / 52.11% / 57.85%）都是 LCS indel 口径，已作废。
   也不要把开头 23 秒的报号（“零零六/零幺三…”）当正文来评判（可用
   `--skip-start-ms 23000`）。
8. **不要把当前句重复放进 LLM 的“上文”**。`turn_confirmed_text` 末尾就是当前句；已修：
   `_polish_one` 会剥掉 raw 后缀。修之前 corrected 从 254 掉到 162，CER 反而略差。
9. **`StreamingSession` 的第一个参数是输出 `session_audio.wav` 路径，不是输入音频**。
   PCM 通过 `feed_pcm_chunk` 喂。搞错会报 “Format not recognised”。
10. **重跑回填会改变 turn 边界**。`turns_polished.json` 带 `start_ms/end_ms`，边界不符会被
    忽略；`final_text.json` 按 `utt_id` 存，天然不怕 turn 重排。不要绕过这个校验。
11. **`turns_polished.json` / `final_text.json` 都是可选派生文件**。`raw_text` 与
    `polished.json` 必须始终保留，便于 A/B 回溯；不要原地覆盖 raw。
12. **8736 服务重启才会加载新代码**。`allow_reuse_address=False`；若端口被旧进程占用，
    先确认 PID 再处理，不要盲目杀进程。
13. **不要假设 Fun-ASR-Nano 可用**。`get_models(need_offline=True)` 会先试
    `C:\Users\EDY\...`，失败后回退 SEACO；本机没有 Nano，看到回退提示是正常的。
14. **不要同时加载全部模型去跑 Nano**。VAD+CAM+++Nano+Streaming 在 16GB/2GB 机器上很危险；
    会后只加载离线模型，必要时先卸载流式模型。
15. **单元测试不要让 `run()` 加载离线模型**：现有测试传 `no_final_asr=True`；新测试用
    `offline_asr_fn` 注入假函数。否则每个测试都会试图加载 SEACO，变慢且不稳定。
16. **后台命令的工作目录会变**：本会话曾因 cwd 变成 `D:\Deepseek` 导致日志重定向失败。
    后台脚本一律用绝对路径写日志/输出。
17. **不要引用 `vad_nopost 88.0% / vad_post 82.9%` 作为系统准确率**。`STREAMING_PLAN.md`
    §八 校订 1 已明确禁止；那是换句子口径的读数，不可互换，且调参污染。
18. **不要把 LLM 当作主要提升手段**。旧口径 A+B 只值 1.45 个点；新口径下
    会后 B strict 只值 **0.06 个点**（35.30→35.24，15/17 接受），在噪声内。
    ASR 模型/输入/重叠才是主因。
19. **不要为了“更流式”去改在线 chunk_size 当作最终稿**。直播模型就是在线 Paraformer
    （`chunk_size=[0,10,5]`，600ms 步长）；最终稿必须离线二遍。用户说“试试 paraformer
    流式模型”时，要先说明这一点。
20. **API key 不得写死**。`llm_correct._get_key()` 从进程环境或用户注册表读；测试不要打真实
    DeepSeek 接口，注入 `call=` / `polisher_factory` / `offline_asr_fn`。

## 7. 文件契约（最终稿相关）

`output/sessions/s_*/` 六个核心文件：
`session_audio.wav`、`utterances.jsonl`、`transcript.json`、`polished.json`、
`diarization.json`、`session.json`、`events.ndjson`。

可选派生文件：

```json
// final_text.json：离线二遍最终稿（键是 utt_id）
{"utt_0001": {"text": "最终稿文本", "start_ms": 0, "end_ms": 1200}}

// turns_polished.json：会后 LLM 精修（键是 turn_id；边界必须对得上）
{"turn_0001": {"text": "精修文本", "start_ms": 0, "end_ms": 12000}}
```

`session.json.utterances[]` 同时保留 `raw_text`、`polished_text`、`final_text`；
`turns[].text` 优先级是 `final_text > polished_text > raw_text`。
`--no-final-asr` 关闭离线二遍；`--no-final-polish` 关闭会后 LLM 精修；`--no-llm` 同时关闭
直播纠错和会后精修，但 **不关闭离线二遍 ASR**。

## 8. 常用命令

```powershell
# 测试（必须从项目根目录，用 funasr 解释器）
cd D:\Deepseek\Meeting_main
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest -q

# ch0 端到端（最新架构；离线二遍会生成 final_text.json）
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
  --wav D:\yinp\AISHELL-4\data\test\wav\L_R003S01C02.flac `
  --session-dir output/sessions/s_ch0_offline --fast --no-llm --no-diar

# CER（新口径：Levenshtein + Sub/Del/Ins + 归因；--legacy 打印旧 LCS）
C:\Users\31817\miniconda3\envs\funasr\python.exe cer_eval.py `
  output/sessions/s_ch0_offline `
  D:\yinp\AISHELL-4\data\test\TextGrid\L_R003S01C02.TextGrid --legacy

# 只重跑离线二遍（不重跑流式，A/B 用）
C:\Users\31817\miniconda3\envs\funasr\python.exe offline_asr.py `
  --session output/sessions/s_ch0_offline `
  --output output/sessions/s_ch0_offline/final_text.v1.json --no-rebuild

# 消融：--no-agc（仅长段）/ --no-group（仅 AGC）
# 测 CER 时 LLM 用严格档：run_offline.py ... --strict-polish

# 启动/重启回放服务（会加载模型；先用 netstat 确认 8736 的 PID）
C:\Users\31817\miniconda3\envs\funasr\python.exe stream_service.py
```

## 9. 当前正在跑 / 待确认

- ch0 端到端已完成：`output/sessions/s_ch0_offline/`，451 句、451 条 `final_text`；
  旧口径 raw 52.11% / 最终稿 46.86%；**新口径 raw 36.90% / v1 35.30%**；
  总耗时 1323 秒（`--no-llm --no-diar`）。日志 `output/ch0_offline_run.log`。
- 2026-10-18 重跑了离线二遍并保留基线：`final_text.v0.json`（旧基线）、
  `final_text.v1.json`（AGC+长段）、`final_text.group_only.json`（仅长段）；
  `session.json` 当前是 v1。评测目录：`_eval_v0` / `_eval_v1` / `_eval_group_only` /
  `_eval_llm_normal` / `_eval_llm_strict`。
- `cer_eval.py` 2026-10-18 起默认剥离 `<sil>` 并用标准 Levenshtein；
  `--legacy` 才能复现 48.62/46.86 这类旧口径。
- 8736 上可能仍有旧代码进程在监听；新前端行为需要重启服务后再验收。

## 10. 建议阅读顺序

1. 本文件。
2. `README.md`（运行方式、参数、契约）。
3. `STREAMING_PLAN.md`（架构与已校订的事实，尤其 §八 禁止引用项）。
4. `.superpowers/sdd/progress.md`（完整实验台账：CAB、A 的重复当前句 bug、B 的边界校验）。
5. 上游 `D:\Deepseek\diarization_cl-main\streaming_asr\V10_DUAL_TRACK_PLAN.md`
   （上游自己的结论：在线双轨尚未验证、旧 E1 预览失败、最终稿应走离线二遍）。
