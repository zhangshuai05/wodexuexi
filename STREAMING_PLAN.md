# 实时会议流式转写与全局说话人回填总体方案

> **状态**：架构说明书（校订版）
> **落盘日期**：2026-10-17
> **项目根**：`D:\Deepseek\Meeting_main`（新项目，只读依赖 `D:\Deepseek\diarization_cl-main`）
> **校订说明**：本文对原方案做了 6 处事实校订，全部集中记录在文末 §八「校订记录」。
> 凡标注 ⚠️ 的地方都是**会导致返工或错误对外报数**的坑，请先读那一节。

---

## 一、方案设计基石

### 基石 1：为什么「流式切句 + 会后回填」能避开重叠语音坍缩

> ⚠️ **本节已重写。** 原方案引用「VAD 句子口径 88.0% / 82.9%，比 TextGrid 口径 53.1% 高 30 个点，
> 所以流式 VAD 切句天然跑在高准确率口径上」作为设计基石。
> 这**正是数据来源文档明令禁止引用的推论**——见 §八 校订 1。此处保留事实、去掉因果。

**事实**（`AliMeeting/results/ml_8meetings.json`，8 场均值，
经 `docs/HANDOFF_1015_AliMeeting困境六收口.md:164-179` 记录）：

| 臂 | acc_single | coverage | det2 | second_ok | pair_exact | false2 |
|---|---|---|---|---|---|---|
| tg_nopost | 53.0% | 88.2% | — | — | — | — |
| tg_post | 53.1% | 99.9% | 42.9% | 71.4% | 19.2% | 13.3% |
| **vad_nopost** | **88.0%** | 97.0% | — | — | — | — |
| **vad_post** | **82.9%** | 99.9% | 52.7% | 69.8% | 30.3% | 6.7% |
| tg_post_truek | 53.1% | 99.9% | 43.2% | 71.3% | 19.4% | 13.1% |

**同一份 diarization 输出、同一套映射口径，只换句子迭代器，读数差 30 个点。**

**已知**：换句子迭代器 = 30 点。真值 K 与自选 K 几乎无差（53.1% vs 53.1%）⇒ 这条线上 K 不是瓶颈。

**未知**：**原因未查。** 来源文档 `HANDOFF_1015` 把这条列为 §8「未决」第 1 条，
措辞是「**原因未查，这是当前最该追问的事实**」，优先级最高。

**来源文档 §7「⚠️ 不得引用」原话**：

> | "VAD 句子口径 82.9% = 系统准确率" | 那是换了句子定义的读数，与 TextGrid 口径的 53.1% 不可互换 |

**不对外报数的三个理由**（**互相独立**，任一成立即足以不报）：

1. **不可互换** —— 换了句子迭代器就是换了评分任务的难度：分母（测试集）变了，
   不是系统变强了。VAD 句子通常更长、声学上更纯，归属判定本身更容易；这是最可疑的
   解释方向，但**尚未验证**（见 §九 未决 1）。53.1% 与 82.9% 之间**不存在**可换算关系。
2. **调参污染** —— 这 8 场同时用于定位与调参，且「现状」基线是脚本硬编码常量
   （`HANDOFF_1015` §7 末行：「8 场 +6.8 是样本外收益」列为**不得引用**）。
   **即使两个口径完全一样，这 8 场的均值也是乐观有偏的。**
   → 这条与第 1 条独立，所以「不报」**不是**「等泛化验证之后就能报」，
   而是**在这 8 场上永远报不了**。要报数必须换 cohort 或扩到 Eval_Ali 其余场，
   并把「现状」重跑一遍（`HANDOFF_1015` §8 第 3 条）。
3. **成因未查** —— `HANDOFF_1015` §8 把「VAD vs TextGrid 差 30 点」列为**未决第 1 条**，
   原话「**原因未查，这是当前最该追问的事实**」。成因未查清之前，
   「我们跑在高准确率口径上」这句话连**是不是真的**都不知道。

> ⚠️ **最该防的是手滑，不是误解。** 表里最高的读数是 **`vad_nopost` 的 88.0%**，
> 它比 `vad_post` 的 82.9% 还高 —— 因为 `vad_post` 为了"造出"多标签，付了单标签准确率
> **−5.1 点**的代价（`HANDOFF_1015` §4 读法：「VAD 88.0 → 82.9」）。
> 也就是说，**想报漂亮数时最顺手的那个数，恰恰是没接 OSD + 匈牙利后处理的那条臂**。
> 写汇报、做对比表时，**88.0% 与 82.9% 一并不许引用**。

**所以本方案选流式 VAD 切句，理由换成工程性的**（不是「口径分高」）：

1. **采样点时钟零漂移，且必须与离线回填的迭代器同源** —— 这是本方案唯一的硬约束。
   会后回填要求每句有与音频零漂移的物理 `[start_ms, end_ms]`，流式 VAD 按采样点计数
   天然提供（见 §三 锚定铁律 1）。若在线用一种句子定义、离线用另一种，
   时间戳就得跨口径换算，那是凭空引入一个误差源。
2. **TextGrid / 人工句子口径在在线场景根本不可得** —— 它需要离线人工标注。
   拿一个在线拿不到的迭代器当设计依据，本身就不成立。
3. **复用已在 VAD 口径上验证过的既有链路** —— `offline_diarizer.py` 的多人支
   在 8 场上全量不亏损（`HANDOFF_1015` 已收口），这是**可以引用**的依据。

### 基石 2：复用现有双轨声纹资产，不重造轮子

| 轨 | 适用 | 复用对象 | 状态 |
|---|---|---|---|
| 多人会议轨 | ≥3 人或未知人数 | `AliMeeting/run_3dspeaker_pipeline.py`：class 区间切分 + 1.5s 谱聚类定身份 + 0.6s/0.15s 短窗细化找边界 + 困境六 fallback（未覆盖句退回长窗投票） | ✅ 已收口，8 场全量不亏损 |
| 双人对话轨 | 2 人 | `claude_a2.py`（**仓库根**）：`a2_segments` (`:261`) + `_dense_rescue` (`:97`) + `_stabilize_labels` (`:64`) + `_absorb_slivers` (`:182`) | ⚠️ 代码在，但**尚未接进 `offline_diarizer.py`**——见 §八 校订 4 |

### 基石 3：100% 复用现有 Web 与模型基建

- 模型池 `_models` + `get_models()` / `preload_models()`（`streaming_asr/web_server.py`）
- 显存释放 `unload_models()` / 预加载 `/api/preload`
- 流式推送协议：标准库 `ThreadingHTTPServer` + `application/x-ndjson` + `Transfer-Encoding: chunked`
- **不引入 Flask / FastAPI / uvicorn 等重型框架**

---

## 二、前置阶段搭建清单（Stage 0）

### 2.1 环境与本地模型

| 项 | 值 |
|---|---|
| Python | `C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12.15） |
| 编码 | 跑脚本前设 `$env:PYTHONIOENCODING="utf-8"`，否则 Windows 控制台 GBK 读中文 docstring 乱码 |
| WMI 补丁 | 保留 `platform._wmi_query = _no_wmi`，防 funasr 导入时 WMI 查询挂起 |
| API Key | `DEEPSEEK_API_KEY`（进程环境变量，或 Windows 用户级注册表 `HKEY_CURRENT_USER\Environment`；`llm_correct._get_key()` 两处都会读） |

**四大模型路径**（统一由 `streaming_asr/web_server.py:26-32` 的常量托管，均已实测就位）：

| 键 | 路径 | 备注 |
|---|---|---|
| `vad` | `<repo>\iic--speech_fsmn_vad_zh-cn-16k-common-pytorch\snapshots\master` | **项目本地调优副本**，`speech_noise_thres=0.7`。默认 0.6 会把多轮对话粘连成一段（5 turns → 3）。有 `test_config_consistency.py` 断言 |
| `spk` | `C:\Users\31817\.cache\modelscope\models\iic--speech_campplus_sv_zh-cn_16k-common\snapshots\master` | CAM++ 声纹嵌入 |
| `stream_asr` | `C:\Users\31817\.cache\modelscope\models\iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online\snapshots\v2.0.4` | 流式 Paraformer |
| `asr` | Fun-ASR-Nano → 失败回退 SEACO Paraformer | 离线高精度，兜底/对比用 |

### 2.2 目录与模块挂载拓扑（**已按实际路径校订**）

⚠️ 原方案把 `web_server.py` / `diar_web.html` / `llm_correct.py` 画在仓库根，
**实际都在 `streaming_asr/` 下**。按原图施工会全部找不到文件。

```
D:\Deepseek\
├── diarization_cl-main\                        ← 只读依赖，本方案不改它
│   ├── claude_a2.py                            # [已有·根] 双人轨：a2_segments + dense_rescue
│   ├── AliMeeting\
│   │   ├── run_3dspeaker_pipeline.py           # [已有] 多人轨：谱聚类 + 0.6s 细化 + fallback
│   │   ├── run_der.py                          # [已有] load_models()
│   │   └── common.py                           # [已有] resolve_wav 等路径常量
│   ├── streaming_asr\
│   │   ├── web_server.py                       # [已有] 端口 8735，模型池，NDJSON 路由
│   │   ├── diar_web.html                       # [已有] 实验台前端
│   │   ├── llm_correct.py                      # [已有] 会后**全文级说话人**微调 + 姓名推断
│   │   └── claude_a2_v11.py / claude_a2_v12.py # [已有] 离线链路依赖的 _vad_segs / _refine_*
│   └── .zhang\                                 # [已有] 流式套件（未跟踪目录）
│       ├── streaming_asr_engine.py             # [已有✅] Task 2：流式 VAD+ASR 状态机 + 双写 WAV
│       └── offline_diarizer.py                 # [已有✅] Task 1（多人支）：backfill_speakers
│
└── Meeting_main\                               ← 本项目（新代码落这里）
    ├── STREAMING_PLAN.md                       # 本文
    ├── bootstrap.py                            # sys.path 挂载 diarization_cl-main（唯一耦合点）
    ├── stream_llm_polish.py                    # [新] Task 3：句尾 FIFO 异步 DeepSeek 固化
    ├── stream_service.py                       # [新] Task 4：/api/stream_meeting 路由 + 会话编排
    ├── diar_backfill.py                        # [新] Task 1b：双臂 + 选优器（见 §6.1）
    ├── ws_transport.py                         # [新] 麦克风通道：手写 RFC6455 WebSocket（标准库）
    ├── tests\
    └── docs\superpowers\specs\
```

**耦合方式**：`bootstrap.py` 一处 `sys.path.insert`，指向 `diarization_cl-main`
及其 `AliMeeting\`、`streaming_asr\`、`scripts\evaluation\` 三个子目录
（与 `run_3dspeaker_pipeline.py:34-37` 自身的 bootstrap 同源）。**不复制任何算法代码。**

---

## 三、全局唯一数据契约（Utterance 单句生命周期）

```jsonc
{
  "utt_id": "utt_0001",
  "start_ms": 1200,               // 严格由采样点计数 (samples * 1000 // 16000) 算出
  "end_ms": 4850,                 // VAD 句尾断句时永久锁死；partial 态为 null
  "status": "diarized",           // partial -> asr_final -> llm_confirmed -> diarized
  "raw_text": "我们先把思露理顺",   // 流式 Paraformer 原始输出（前端斜体）
  "polished_text": "我们先把思路理顺。", // DeepSeek 上下文纠错后（前端正体）
  "spk": 0,                       // 会后回填的说话人**簇标签 p**，0-based
  "spk_label": "speaker_0",       // 人类可读标签 = speaker_{p}
  "overlap_spk": null             // 预留字段；v1 **不实现**，见 §八 校订 5
}
```

### 锚定铁律

1. **物理采样点时钟**：严禁用 `time.time()` 计算 `start_ms` / `end_ms`。
   必须用 `累计送入采样点数 * 1000 // 16000`，确保与落盘的 `session_audio.wav` **零漂移**。
   → 已由 `StreamingSession.processed_ms` 实现。
2. **时间壳子不可变**：VAD 判定句尾停顿后 `[start_ms, end_ms]` 立即封口锁死；
   后续 DeepSeek 无论怎么增删改写句内文字，**均不改变物理时间边界**。
3. **⚠️ `spk` 的 off-by-one 陷阱**：`offline_diarizer.backfill_speakers` 返回的是
   **字符串** `"Speaker_{p+1}"`（`offline_diarizer.py:116`，`p` 是簇标签）。
   契约里 `spk` 是 **0-based 整数 `p`**，`spk_label` 是 `"speaker_{p}"`。
   服务层在 `to_contract` 中**只换算一次**（`p = int(label.split("_")[1]) - 1`），
   并且**同时**下发 `spk` 与 `spk_label`，避免前端自己减 1 减错。

---

## 四、三阶段流水线

```
[前端麦克风 / WAV 回放 (16kHz PCM Chunks)]
       │
       ├──► 旁路1：实时追加写 session_audio.wav（每块更新 WAV Header）
       │
       └──► 旁路2：【阶段 1：流式 VAD + 流式 Paraformer（600ms 步长）】
                     ├─► 说话进行中：持续抛 asr_partial ──► 前端【灰色斜体，实时跳动】
                     └─► 检测到句尾停顿（1000ms）或满 12s 换气：（2026-10-17 拍板：1000ms，由上游 .zhang/streaming_asr_engine.py 的默认值变更同步而来；本项目显式 pin）
                           1. 对 ASR 执行局部 Flush (is_final=True) 逼出尾字，清空 asr_cache
                           2. 锁死 utt_k 的 [start_ms, end_ms]，抛 asr_final
                           3. 主线程 0 等待，开启 utt_{k+1} 继续输出下一句斜体
                                  │
                                  ▼（异步压入后台 FIFO 队列）
                    【阶段 2：单消费者 FIFO 异步 DeepSeek 润色器】
                     ├─► 短句短路：有效字数 ≤ 2（"嗯"、"好的"），0ms 直接补句号转正体
                     └─► 常规句子：携带前 3 句【已固化正体】作只读语境，
                         调 deepseek-chat 修同音错字 + 加标点
                           └──► 抛 llm_confirmed ──► 前端原地切为【黑色正体】
                                  │
       ===========================╪=========================== （点击"结束会议"）
                                  ▼
                    【阶段 3：会后全局离线说话人回填】
                     1. Flush 最后一句未完结草稿，等 FIFO 清空，关闭 session_audio.wav
                     2. 调 offline_diarizer.backfill_speakers(session_audio.wav, utterances,
                        models, mode="auto")
                        - 多人支：1.5s 谱聚类 + 0.6s 细化 + 未覆盖句退回长窗投票（fallback）
                        - 双人支：claude_a2.a2_segments（含密集窗救援与碎段吸收）  ← 待补全
                     3. 用 utt_id 的 [start_ms, end_ms] 与声纹段做时间交集投票
                     4. 抛 speaker_diarized ──► 前端批量挂载【Speaker_X 标签】
```

### 阶段 2 的关键约束

- **单消费者**：`queue.Queue` + 一个后台线程。并发多句会让 DeepSeek 的上下文互相打架，
  且容易撞 API 限流。
- **主线程零阻塞**：入队即返回。润色慢不能拖慢转写。
- **失败降级（fail-soft）**：API 报错 / 超时 / 返回非法 JSON 时，
  **把 `polished_text` 置为 `raw_text` 并照常抛 `llm_confirmed`**，
  事件里带 `"corrected": false` 和 `"error": "..."`。
  绝不能让一句润色失败卡死整条队列。
- **上下文只读**：前 3 句正体仅作语境，**不参与改写**。

### 4.1 ⚠️ 阶段 3 的耗时预算（**原方案完全缺失**）

原方案对阶段 3 只写了一句「调 `backfill_speakers`」：**没有耗时预算、没有进度设计、
没有"先出部分结果"的方案**。而这是产品级问题，不是实现细节。

**实测（本机，运行中）** —— `AliMeeting/results/accept_8mtg_20261017.json`
（PID 5196 此刻仍在跑，`--arm class`，8 场，02:36:25 启动）：

| 场次 | `elapsed_s` | 折合 |
|---|---|---|
| R8008_M8013 | 352.0 | 5.9 分钟 |
| R8007_M8010 | 672.1 | 11.2 分钟 |
| R8001_M8004 | 709.2 | 11.8 分钟 |
| R8003_M8001 | 1104.2 | **18.4 分钟** |

已跑 4 场累计 **47.3 分钟**；按均值 ≈709 s/场外推，**8 场全量约 95 分钟**。

⚠️ **引用口径**：该文件是**运行中**产物，每完成一场就被重写
（`run_3dspeaker_pipeline.py:540-545`）。引用必须带抓取时间戳，否则是移动靶。

⚠️ **不要把它的 `elapsed_s` 与 `final8_refined_fb.json` 比较后归因为"代码路径不同"。**
两者是**同一条路径**：config 块与 row schema 相同，且逐场数值**完全相同**
（R8007_M8010 `auto acc 0.26164267569856053` / `refined 0.31583403895004236` / `k_ahc 2`；
R8008_M8013 `refined 0.5081967213114754` / `coverage 0.51775956284153`）。
但 `final8_refined_fb.json` 的 `elapsed_s` 是 **337.7 / 128.0 / 171.6**，
比 accept 低 **2.0–4.1 倍**。**相同代码产生相同输出，不可能因代码路径差 2–4 倍。**

⇒ **真正的第二因子是机器**：`final8_refined_fb.json` 出自旧机器，accept 出自本机（MX450 2GB）。
**本机比旧机器慢 2–4 倍**，这是硬约束。同一现象也解释了
`环境清单_新机器_20261017.md` §8.1 记的「文档旧记 35 分钟/12 场偏乐观」。

**产品后果**：用户点「结束会议」后要等 **6–18 分钟**才看到说话人标签。
方案的卖点是"实时生长"，**这个尾巴与前半段体验是断裂的**。

**因此阶段 3 必须补三件原方案没有的东西**：

1. **耗时预算进接口** —— `progress` 事件要带**百分比或已处理进度**，
   不能只有一条无信息量的 `msg`
2. **允许分批下发** —— `speaker_diarized` 按音频时间序**分批**推，
   让用户先看到前几分钟的标签，而不是空等一个整场原子事件
3. **标注"本机"口径** —— MX450 2GB 是当前基线，换机器必须重测

---

## 五、前后端流式通信协议（NDJSON 事件规格）

> **这是交付给前端 AI 的接口契约，必须冻结。** 后端只保证事件形状与顺序，
> 渲染逻辑由前端负责（CSS 类名建议见 §5.2，仅作参考，非强制）。

### 5.1 事件规格（7 类）

```jsonc
{"t": "asr_partial",   "utt_id": "utt_0001", "start_ms": 600,  "current_ms": 1800,
                       "text": "大家早上好今天我们把思露"}

{"t": "asr_final",     "utt_id": "utt_0001", "start_ms": 600,  "end_ms": 2950,
                       "raw_text": "大家早上好今天我们把思露整理一下"}

{"t": "llm_confirmed", "utt_id": "utt_0001", "start_ms": 600,  "end_ms": 2950,
                       "raw_text": "大家早上好今天我们把思露整理一下",
                       "polished_text": "大家早上好，今天我们把思路整理一下。",
                       "corrected": true, "error": null}

{"t": "progress",      "msg": "会议录音已结束，正在执行全局声纹聚类与说话人回填…"}

{"t": "speaker_diarized", "n_clusters": 3,
                          "mapping": {"utt_0001": {"spk": 0, "spk_label": "speaker_0", "overlap_spk": null},
                                      "utt_0002": {"spk": 1, "spk_label": "speaker_1", "overlap_spk": null}}}

{"t": "error",         "msg": "...", "fatal": false}

{"t": "finish"}
```

**顺序保证**：`asr_partial*` → `asr_final` →（异步）`llm_confirmed`。
`llm_confirmed` **一定晚于**同 `utt_id` 的 `asr_final`，但不保证与其它 `utt_id` 的相对顺序。
`speaker_diarized` 一定在 `finish` 之前、且在阶段 3 完成后。

### 5.2 前端渲染逻辑（参考实现，交前端 AI）

维护一个按 `utt_id` 索引的列表容器：每行 = `[说话人徽章占位] + [时间戳] + [文本 span]`。

| 收到事件 | 渲染动作 |
|---|---|
| `asr_partial` | 若 `#row-<utt_id>` 不存在则追加新行；文本设为 `text`，类名 `.draft-italic`（`font-style: italic; color: #888;`） |
| `asr_final` | 更新时间戳为 `[start_ms - end_ms]`，文本更新为 `raw_text`，右侧显示"润色中…"呼吸点 |
| `llm_confirmed` | 定位 `#row-<utt_id>`，文本平滑替换为 `polished_text`，类名切 `.confirmed-normal`（`font-style: normal; color: #111; font-weight: 500;`）；`corrected=false` 时徽章标灰以示未纠错 |
| `speaker_diarized` | 遍历 `mapping`，为每行徽章填 `spk_label` 并按 `spk` 取色 |
| `error` | `fatal=true` 时终止会话并提示；否则仅日志 |

### 5.3 ⚠️ 传输层：两条通道，不是一条

原方案写「直接复用 `_emit()` 的 NDJSON Chunked 通道」——**这对回放通道成立，对麦克风通道不成立**。
NDJSON-over-POST 是**单向**的（一个 HTTP 响应体往下流），无法承载"客户端持续往上推 PCM"。

| 通道 | 传输 | 复用程度 |
|---|---|---|
| **WAV 回放**（仿真调试 + 确定性回归测试） | `POST /api/stream_meeting` → `application/x-ndjson` chunked | ✅ 完全复刻 `web_server.py:253-263` 的 `_send_chunk` 模式 |
| **真实麦克风** | **WebSocket**（RFC6455，标准库手写，约 120 行：握手 + 帧编解码） | ⚠️ 仓库现有**零** WebSocket 代码，需新写 |

两条通道**共用同一个 `StreamingSession` 实例接口**（`feed_pcm_chunk`），
业务代码零分支。`ws_transport.py` 只负责帧收发，不碰状态机。

---

## 六、任务执行路线图（**已按现状校订**）

原方案列 4 个 Task，其中 **Task 1 大部分与 Task 2 已存在**。校订后的实际待办：

| # | 模块 | 状态 | 输入 → 输出契约 | 验收 |
|---|---|---|---|---|
| **T0** | `bootstrap.py` | 🆕 新写 | — → `sys.path` 挂载 | `import run_3dspeaker_pipeline` / `import claude_a2` / `from web_server import get_models` 全部成功 |
| **T1** | `.zhang/offline_diarizer.py` | ✅ **已存在**，多人支已通 | `(wav_path, utterances, models, mode)` → `{utt_id: "Speaker_{p+1}"}` | 跑 `python .zhang/offline_diarizer.py R8008_M8013` 不塌陷（现有自检入口） |
| **T1b** | `Meeting_main/diar_backfill.py` | 🆕 **双臂 + 选优器** | 同 T1，但同时跑 `run_3dspeaker_pipeline`（谱聚类）与 `claude_a2.a2_segments` 两臂，由选优器裁决 | 选优器在 AliMeeting 8 场（有 `reference_rttm` 真值）上命中率 **≥ 6/8**；不达标则退化为"只用谱聚类"。详见 §6.1 |
| **T2** | `.zhang/streaming_asr_engine.py` | ✅ **已存在且可跑** | `feed_pcm_chunk(bytes, is_final)` → `asr_partial` / `asr_final` 事件 | 已有 `__main__`：WAV 按 600ms 喂入能刷出草稿与锁定句，落盘 WAV 能直接喂 T1 |
| **T3** | `Meeting_main/stream_llm_polish.py` | 🆕 **核心新工作** | `submit(asr_final_event)` → 回调抛 `llm_confirmed` | 主线程推流不卡顿；断句后 0.5–1s 内转正体；**API 挂了也不能卡队列** |
| **T4** | `Meeting_main/stream_service.py` + `ws_transport.py` | 🆕 新写 | **把已跑通的自检尾部从 `print` 换成 NDJSON 事件回调**并走服务层。（**不是**"串联 T1b→T2→T3" —— T2→T1 **已经通了**，见 `.zhang/streaming_asr_engine.py:319-322`） | 页面完整呈现三态；且**阶段 3 必须有进度反馈**（见 §4.1） |
| **T5** | 会后姓名推断 | ✅ 已有，接线即可 | `POST /api/llm` → `llm_correct.correct` | 由前端 AI 决定是否接 |

**开工前置（两个小项，但第一个会让最常用的调试动作第一次就失败）**：

1. **`--no-diar` 兜底** —— `.zhang/streaming_asr_engine.py` 的
   `from offline_diarizer import backfill_speakers` 在 `__main__` 里是**无条件**的，
   所以"只跑引擎看转写、不做回填"这个最常用的调试动作会直接 `ImportError`（不是被跳过）。
   给 `--no-diar` 或 try/except 兜一下。
2. **`git init`** —— `Meeting_main` 与 `diarization_cl-main` **都不是 git 仓库**，
   即整个方案目前**没有任何版本控制**。本方案已经派了 T0–T5 的编号，
   正是需要 commit 粒度的场景；且 `bootstrap.py` 的 sys.path 依赖上游，
   上游一变"同源"就断了，没有版本控制无法回溯。

**两个文件本身不需要改造**：`.zhang/` 那两个的接口
（`feed_pcm_chunk` / `completed_utterances` / `backfill_speakers(wav, utts, models, mode)`）
与 §三 契约严丝合缝，T4 直接调即可。要动的只有 §6.1.4 的三段拆分（放在 `Meeting_main` 里做）
与 `.zhang/offline_diarizer.py:11` 那句**过时的 docstring**
（它写「`.zhang/streaming_asr_engine.py` 已不 import 本模块」，
而 `:319` 就是 import —— 字面已被反证，建议改写成流水线阶段说明）。

**注意分工**：`llm_correct.py` 是**会后全文级说话人微调 + 姓名推断**，跑在阶段 3 之后；
`stream_llm_polish.py` 是**句尾单句文本纠错**，跑在阶段 1.5。**两者不是一回事，不要合并。**

### 6.1 T1b 双臂选优器（已拍板：两条都跑，自动选优 —— 但**测量先行**）

> **本节经 2026-10-17 一轮对抗式核实后重写。** 原稿的「选优器命中率 ≥ 6/8 才上线」门槛
> **已撤销**：它会在目标人群上轻松通过而什么也没证明（6.1.1）。取而代之的是
> 「先测量、后决策」，且测量本身有两个前置障碍（6.1.2 / 6.1.3）。

#### 6.1.1 为什么撤销 6/8 门槛

| 问题 | 依据（均已核实） |
|---|---|
| **目标人群上两臂本来就是平的** | `diag_2spk_cluster_compare.json`：3 场 2 人会议，谱聚类 vs AHC 差 **+0.30 / 0.00 / −0.46 pt**。选优器在 2 人档挑谁都对，因为它俩都对 |
| **门槛会被噪声穿过** | 6/8 = 75%。**纯抛硬币**在 8 场上拿到 ≥6 的概率约 **14.5%**。样本太小，"命中率"这个统计量的噪声有门槛一半那么宽 |
| **标定人群错位** | 8 场里只有 3 场是 2 人档。用 4 人场的分歧去标定 2 人场用的选优器，是"在错误的总体上调参"——比"用魔法打败魔法"更隐蔽 |

#### 6.1.2 前置 A：真值裁判先自证自洽（**别照直觉跑 `--validate`**）

⚠️ `textgrid_to_rttm.py` 两个模式**都不校验 AliMeeting 的每场真值**：

| 命令 | 实际校验对象 | 为什么不够 |
|---|---|---|
| `--validate` | **AISHELL-4 的 20 场**转换规则（`textgrid_to_rttm.py:66-90`，glob 的是 `AISHELL_TG/*.rttm`） | 换了语料。AliMeeting 每场 true_k 有错，它也照样打印 20/20 |
| `--corpus alimeeting` | 每场**打印** `N 人`（`:107-108`），唯一**断言**是 `len(speakers) == 25`（`:112`，**语料级总数**） | 每场错但总数仍为 25 时，它照样打印「一致」 |

**正确做法**：跑 `--corpus alimeeting`，**抓下 8 行「N 人」**，与 `ml_8meetings.json` 的
`true_k` 逐场对账。

⚠️ **重跑前给 `reference_rttm/` 做校验和** —— `build_alimeeting()` 第 103 行是**重写**该目录，
而它是全部 `results/*.json` 与 `predictions/*.rttm` 的打分依据。否则「重生成是幂等的」无法证明。

**已排除的一个假警报**：`embedding_separability.json` 对 R8001_M8004 / R8007_M8011 记的
`true_k=3` 与真值的 4 **不冲突** —— 那个字段**不是真值说话人数**。生成器
`scripts/diagnostics/diagnose_embedding_separability.py:222` 是 `true_k = len(by_speaker)`，
而 `by_speaker` 只统计「有段存活于 `clean_segments`（丢弃与它人重叠 >0.05s 的段，`:56-69`）
且含完整 2.0s 窗（`:72-86`）」的说话人。成因已**逐位复现四行**：R8001_M8004 的 `N_SPK8015`
有 154 段但最长非重叠段仅 **0.47s**；R8007_M8011 的 `N_SPK8069` 有 201 段但最长非重叠段
**1.70s** —— 都低于 2.0s 窗，被**静默丢弃**。其余全部产物（`oracle_k_alimeeting_full`、
`ml_8meetings`、`final8_*`、`arms_*`、`speaker_time_balance`、`refine_anchor_quality`）一致说 4。
→ **待办**：给该字段改名（如 `n_speakers_clean_2s`）或标注不可当真值用。这是**静默丢弃**，不是数据冲突。

#### 6.1.3 前置 B：`eval_multilabel.py` **没有 arm 参数**（新发现的成本项）

产出 `acc_single` 的唯一脚本是 `AliMeeting/eval_multilabel.py`（`score_arm` 在 `:217`，
句数 `n` 在 `:186`）。但它的 CLI 只有 `--meetings / --out / --second-thresh / --min-abs /
--min-rel / --judge-pair-exact / --judge-false2` —— **没有 arm 或预测文件参数，臂是硬编码的**。

⇒ **"两臂都在 8 场上跑一遍"不能直接执行**，必须先改 `eval_multilabel.py` 或另写打分器。
**原方案与 review 都没算这笔。**

#### 6.1.4 前置 C：两臂必须共享嵌入，否则"省时间"是假的

`backfill_speakers` 把 `embed_chunks` 藏在函数体内（`.zhang/offline_diarizer.py:92`），
第二臂**拿不到已算好的嵌入**。CAM++ 嵌入是瓶颈，不拆分的双臂全跑 = 嵌入算两遍。

⇒ **T1b 前置**：把回填拆成 `windows → embeddings → cluster(path)` 三段。
**在 `Meeting_main/diar_backfill.py` 里做**（自带拆好的三段），
`.zhang/offline_diarizer.py` **一行不动**，保持 §2.2「只读依赖」成立。

#### 6.1.5 三步测量（**T1b 的唯一验收；门槛待测完再定**）

- **第 0 步**：6.1.2 的真值对账（几小时，不加载模型，不接任何线）
- **第 0' 步**：6.1.3 给打分器加 arm 参数（**新成本**）
- **第 1 步：oracle 上界测量**（不接服务线）。每场 4 个数：

| 数 | 含义 |
|---|---|
| 谱聚类臂真值 `acc_single` | 主臂 |
| `a2_segments` 臂真值 `acc_single` | 备臂 |
| **oracle 上界 = best-of-two** | 选优器价值的**上界** |
| 两臂句级一致率 | 仅作短路条件（见 6.1.6） |

⚠️ **`acc` 口径必须写死。** 仓库里有两个互不相同的 `acc`：
`ml_8meetings.json` 的 `acc_single`（**句级 top-1**，`eval_multilabel.py:217`）
与 `accept_8mtg_20261017.json` / `diag_2spk_cluster_compare.json` 的 `acc`（**段级/窗口级**，
2 人档只有 73–84%）。**本方案一律认句级 `acc_single`** —— 产品交付粒度就是句级。

- **第 2 步：决策（两段式，不是单一门槛）**

```
oracle − 谱聚类 < 2× 噪声带  ⇒  确定砍掉选优器，T1b 不做（两臂本就是一回事）
oracle − 谱聚类 ≥ 2× 噪声带  ⇒  必要非充分：还须证明选优器能兑现其中 ≥50% 的 gap
```

⚠️ **`oracle gap 大 ⇒ 选优器有价值` 是逻辑跳跃，必须拦住。**
oracle 是**上帝视角**，参考无关的选优器**没有任何判据保证命中** —— 6.1.6 里三个候选判据
两个不可用、一个只能当短路，正是这个问题的体现。**oracle gap 是上界，不是期望值。**

**噪声带的定义（可实测，不用估计）**：`web_server.py:23-25` 自己写着「模型有小的 GPU
逐次运行差异，故同一输入必须复用首次结果」；`docs/环境清单_新机器_20261017.md` §8.1 实测过
同一输入跨机器嵌入值有浮点级差异、**足以让边界场次的 k 选择翻转**（`L_R003S03C02`，mw2 下 k 6≠3）。
⇒ **同一条臂、同一台机器跑两次，`acc_single` 之差即噪声带下界**；跨机器差异是上界。
⚠️ 这条同时意味着**"两臂差 1–2 点"很可能整个落在噪声里** —— 是对 6.1.1 的加强。

**若第 2 步判定要做选优器**，验收用「**不劣化**」约束而非命中率：选优器挑中的臂，
真值 `acc_single` **不低于谱聚类臂超过 1pt**，且 8 场里**没有一次挑中明显更差的那条**（差 > 2pt）。

#### 6.1.6 判据取舍（经核实）

| 判据 | 结论 | 依据 |
|---|---|---|
| embedding 可分性 margin | ❌ **不可复用** | `embedding_separability.json` 的 config 自述用的是**真值段内窗** + 留一最近质心、丢弃重叠段。它的 `margin_median` / `auc` / `loo_accuracy` **全都需要真值窗**。选优器手上没有真值，产物一个也搬不过来 |
| 两臂句级一致率 | ⚠️ **只能当短路，不能排名** | 一致率高**不等于对** —— 两臂共享同一套 VAD 与同一套嵌入，可能**一起错**。别让它承担"证明确实更好"的职责 |
| 覆盖率 `cover` | ⚠️ **参考，方向存疑** | fallback 是**设计上的**兜底（`.zhang/offline_diarizer.py:98`），不是缺陷。且 `refined.coverage` 有低到 **0.5178**（`accept_8mtg_20261017.json` R8008_M8013）而 acc 反而略升的例子。"cover 高 = 更好"仓库数据不支撑 |

**一致率的定义（"一致率 > 95%" 这种说法不能用）**：标签是**簇编号**，直接比会因**置换**全错。
必须先做句级混淆矩阵的**匈牙利匹配**，再算**按人的宏平均召回**。
且判停条件不能是固定条数 —— "一致率 > 95%" 在 40 句上只能证明一致率 > 88%。
**要报置信下界，不是点估计。**

**短路口径修正**：短路省掉的是第二臂的**聚类 + 细化**（贵的那部分），
不是"完整打分"（打分本身是零成本的按句求均值）。

```
跑谱聚类臂（主臂）
   │
   ├─ 与 a2_segments 臂的 K 相同？ ── 否 ──► 选优器裁决（K 不同才是真分歧）
   └─ 是 ──► 抽样比对句级标签（匈牙利匹配后按人宏平均）
                ├─ 一致率下界 > 阈值 ──► 采主臂，不跑第二臂的聚类+细化
                └─ 否则 ──────────────► 选优器裁决
```

### 6.2 T3 纠错契约（已拍板：保守尺度）

**允许**：修同音字 / 误识字、加标点、断句标点。
**禁止**：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语。

**Prompt 硬约束**（写进 `stream_llm_polish.py` 的模板）：

```
你是 ASR 转写纠错器。只做两件事：改同音/误识别的字，补标点。
严格禁止：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
不确定的地方保持原样——宁可少改，不可改错。
输出严格 JSON，不要解释，不要 markdown 代码块：
{"polished": "纠错后的整句"}
```

**三道守卫**（任何一道不过就**丢弃改写、保留 `raw_text`**，并置 `corrected=false`）：

| 守卫 | 阈值（初值，须实测标定） | 防的是什么 |
|---|---|---|
| 长度增幅 | > 50% | LLM 顺着上下文"补"出用户没说的一整句 |
| 长度缩减 | > 30% | LLM 把半句当成冗余删掉 |
| 数字集合 | 改写后数字集合 ≠ 原文数字集合 | "三点"变"4点" —— 数字错误在会议场景代价最高 |

**上下文用法**：前 3 句**已固化正体**只作只读语境传入，**不参与改写**；
prompt 里明确标注这是"上文语境，仅用于判断同音字，不要续写、不要回应"。

> ⚠️ 保守尺度的已知代价：`"大家早上好今天我们把思露整理一下"` → `"大家早上好，今天我们把思路整理一下。"`
> 只修了「思露→思路」和标点。**口语重复（"就是就是"）与语气词会原样保留**。
> 这是刻意的取舍——会议记录的**事实保真**优先于**阅读流畅**。
> 若日后要放宽，必须先在真实会议样本上量化"改写引入事实错误"的比率。

---

## 七、双模输入设计

| 模式 | 触发 | 服务端行为 |
|---|---|---|
| **仿真调试（WAV 回放）** | 上传 WAV 或给本地路径 | `soundfile` 校验 16kHz 单声道（非 16k 用 ffmpeg 转）→ 按 300ms 切片喂 `StreamingSession`，**按墙钟节流**（喂 300ms 音频睡 ~300ms）以还原真实观感；调试可加 `--fast` 全速 |
| **真实麦克风** | 浏览器 `getUserMedia` → `AudioWorklet` | 采 16kHz PCM → WebSocket 送 `feed_pcm_chunk` |

**为什么先做回放**：回放通道 100% 确定性（同输入 → 同事件序列），能进 pytest；
麦克风通道只能手测。两者共用状态机，所以**先回放不是为了省事，是为了有回归网**。

> 节流粒度：引擎内部按 9600 采样点（600ms）攒批才 step，
> 所以喂 300ms 只影响"上屏的平滑度"，不影响 `[start_ms, end_ms]` 的锁定值。

---

## 八、校订记录

原方案共 6 处与仓库现状或数据来源不符，逐条列出。**已在上文正文改掉，此处留痕。**

| # | 原文 | 事实 | 处置 |
|---|---|---|---|
| **1** | 「同一份结果 TextGrid 口径 53.1%、VAD 口径 88.0%/82.9%，**相差 30 个百分点**…我们的流式架构正是通过 VAD 切句，因此会后回填天然运行在 82.9%～88.0% 的高准确率 VAD 句子口径上」 | 数字真实（`ml_8meetings.json`，记于 `HANDOFF_1015:164-179`）。但同文档 **§7「⚠️ 不得引用」** 明列：「"VAD 句子口径 82.9% = 系统准确率" —— 那是换了句子定义的读数，与 TextGrid 口径的 53.1% **不可互换**」。且 §8 把「差 30 点」列为**未决第 1 条，原因未查**。原文为该未决问题**编造了一个成因**（"因为 VAD 按真实声学停顿切句"） | 基石 1 **整节重写**：保留事实表格与来源，删去因果断言，加「不可对外报数」告警，并把选 VAD 切句的理由换成工程性理由（时间边界零漂移 / 在线场景 TextGrid 不可得 / 避免口径转换风险） |
| **2** | 目录树把 `web_server.py`、`diar_web.html`、`llm_correct.py` 画在**仓库根** | 三者**实际都在 `streaming_asr/`**。根目录只有 `claude_a2.py`（这个原文是对的） | 拓扑图按实际路径重画，逐文件标注 `[已有]` / `[新]` |
| **3** | 「确认 `preload_models(need_stream=True, need_offline=True)` 能正常加载」 | `preload_models` 定义在 `web_server.py:65`，签名是 **`preload_models(progress=None)`**，**不接受** `need_stream`/`need_offline`（它内部硬编码 `get_models(progress, need_stream=True, need_offline=True)`）。照原文调用会 `TypeError` | §2.1 只列 `get_models()` / `preload_models()`（无参形式）；提示直接 `POST /api/preload` |
| **4** | Task 1 描述：`offline_diarizer.py`「多人调 `run_3dspeaker_pipeline`，**双人调 `claude_a2.a2_segments`**」 | `offline_diarizer.py` **只有多人支**，全文无 `a2_segments` 调用；其 docstring 明确说自研重叠感知路径已删除。`claude_a2.py`（根）的 `a2_segments` 确实存在（`:261`），但**从未接进回填层** | Task 1 拆成 T1（已存在）+ **T1b（`diar_backfill.py`）**。T1b 的最终形态经拍板定为「双臂 + 选优器」而非按人数分派，见 §6.1 与 §9.1 |
| **5** | 契约含 `"overlap_spk": null`「可选：若该句内检测到明显第二人抢话」 | 现链路 `turns_to_speakers(windows, labels, turns, fallback)` 返回 `(pred, cover)`——**每句仅单标签**，无重叠信息。多标签能力在 `eval_multilabel.py`（评测用），不在生产回填路径上 | 字段**保留但标为 v1 不实现**（预留位，不保证有值）。若确需重叠归属，按 `ml_8meetings.json` 的 `vad_post` 臂另立任务（注意该臂单标签准确率 **−5.1 点**，是笔交易） |
| **6** | 「前 **2** 句正体作上下文」（首轮口述）vs「前 **3** 句」（本文档） | 两处不一致 | 取本文档的 **3 句**为一等公民，并在 §4 写明 |

**另外三处不算错、但值得写死的**：

- **`spk` 的 off-by-one**：现有回填返回 `"Speaker_{p+1}"` 字符串，而契约要 0-based `spk`。
  已在 §三 锚定铁律 3 写死换算与"两者同时下发"的约定。
- **`llm_correct.py` ≠ `stream_llm_polish.py`**：前者是会后全文级**说话人**微调 + 姓名推断，
  后者是句尾单句**文本**纠错。已在 §六 显式区分，避免被合并。
- **`_emit()` 通道只适用于回放**：麦克风需要 WebSocket，仓库现有零实现。已在 §5.3 写死双通道。

---

## 九、决策台账

### 9.1 已拍板

| # | 事项 | 决策 | 落点 |
|---|---|---|---|
| 1 | T1b 双臂 | **两条都跑，自动选优** —— 但**测量先行**：先做真值对账 + oracle 上界测量，再决定选优器是否值得存在。原「命中率 ≥ 6/8」门槛**已撤销** | §6.1 |
| 2 | T3 纠错尺度 | **保守：只改错字 + 标点**，不改词序、不增删内容 | §6.2 |

### 9.2 仍未决

| # | 事项 | 影响 | 归属 |
|---|---|---|---|
| 1 | **VAD 句子口径比 TextGrid 高 30 点的成因**（`HANDOFF_1015` §8 第 1 条） | 决定多标签口径该用哪个句子定义；也决定本方案能否对外报准确率。**这是研究任务，不是实现任务**，别混进 T1–T5 | 研究 |
| 2 | **噪声带到底多大** —— 按 §6.1.5 的方法实测（同臂同机跑两次） | 直接决定 T1b 做不做：它决定"两臂差 1–2 点"是否有意义 | 测量（第 1 步顺带产出） |
| 3 | **`eval_multilabel.py` 加 arm 参数** —— 现在臂是硬编码的，第 1 步测量**跑不起来** | T1b 的硬前置，**新成本** | 实现（第 0' 步） |
| 4 | **域偏移复核** —— 全部读数来自 AliMeeting（远场阵列 + 高重叠 + 朗读式），产品是近场麦克风 + 真实会议 | 决定 AliMeeting 上的测量能否迁移。§6.2 已要求 LLM 守卫在**真实录音**上标定，**同一要求也适用于选优器与 oracle 测量**（不能对一处严格、对另一处放宽） | 待规划 |
| 5 | **会后等待 6–18 分钟的产品处置** —— 分批下发？上云？还是接受？ | 见 §4.1。这是方案里最大的未处置体验断点 | 待决策 |
| 6 | **回放墙钟节流**：是否需要 `--fast` 全速模式 | 影响 T4 接口。默认给，代价极小 | 实现 |
| 7 | **麦克风通道实时性上限** —— 流式 Paraformer 跑 `device="cpu"`（`web_server.py:197`），本机 MX450 2GB | T4 完成后**必须实测**，这是"商业级"的最低门槛 | 实现后实测 |

### 9.3 已撤销 / 已排除（留痕，别再捡回来）

| 事项 | 结论 | 依据 |
|---|---|---|
| T1b「命中率 ≥ 6/8 才上线」 | **撤销** —— 会在目标人群上轻松通过而什么也没证明 | §6.1.1 |
| 「两文件真值冲突」（`embedding_separability` 的 `true_k=3` vs 真值 4） | **排除** —— 同名字段在量不同的东西（"有干净 ≥2s 段的说话人数"），成因已逐位复现四行 | §6.1.2 |
| 「两文件耗时差 10 倍 = 代码路径不同」 | **推翻** —— 同一条路径、逐场数值相同；真因子是**机器**（本机比旧机慢 2–4 倍） | §4.1 |
| 「Task 1 / Task 2 从未串起来」 | **推翻（我在上一轮的误报）** —— `:319-322` 已接线 | §六 |
| embedding 可分性 margin 当判据 | **不可复用** —— 它需要真值窗 | §6.1.6 |
| `cover` 当排名判据 | **降级为参考** —— 仓库数据不支撑"cover 高 = 更好" | §6.1.6 |

---

## 十、本轮核实的证据基础

§6.1 / §4.1 / §9.3 的结论来自 2026-10-17 一轮对抗式核实：8 条事实断言各配一名
独立反驳者（**8 条数字全部核对无误**，其中 2 条结论被反驳者推翻），
外加 2 项可行性核查（句级打分器接口、两臂接口鸿沟）。
**凡本文件标"已核实"的，均指这一轮。**
