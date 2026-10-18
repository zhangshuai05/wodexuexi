# 实时长时会议系统 —— 全栈架构与开发规格说明书

> **文档编号**：FULLSTACK-STREAMING-20261017
> **放置位置**：`D:\Deepseek\Meeting_main\FULLSTACK_STREAMING_SPEC.md`
> **适用对象**：全栈流式框架 + 前后端总装 Agent（Agent B）
> **上游算法交接**：`D:\Deepseek\diarization_cl-main\HANDOFF_OFFLINE_DIAR_AGENT.md`（Agent A）
> **先读**：`D:\Deepseek\Meeting_main\README.md`（v0 已实现能力）、`D:\Deepseek\Meeting_main\STREAMING_PLAN.md`（总体方案与校订记录）
> **核对基准**：2026-10-17 仓库实况
> **阅读约定**：本文与原规格稿不一致的地方集中在 §9「校订记录」。**§0 是必读的现状盘点——v0 CLI 已经落地，不要重写。**

---

## 0. 当前仓库现状（先读，避免重复造轮子）

### 0.1 已经完成的部分（不要重写）

`D:\Deepseek\Meeting_main` 是一个**已初始化 git 的仓库**，当前提交链包含 v0 的完整 CLI 闭环：

```text
edb5e5a fix: close final-review gaps (locking, shutdown, label validation, drift guard, CJK numerals)
cff875b docs: sync event contract and plan with implementation (bypassed field, pinned end-silence)
497f4a5 test: real-audio smoke coverage and v0 README
6b17fe4 fix: pin streaming end-silence to 1000ms so upstream default changes cannot drift behavior
dae9702 fix: keep local modules ahead of upstream and harden shared writes
```

已落地文件：

| 文件 | 状态 | 作用 |
|---|---|---|
| `bootstrap.py` | ✅ | 唯一的上下游耦合点：`sys.path` 挂载 + 上游指纹校验 |
| `run_offline.py` | ✅ | WAV → 流式三态 → 异步润色 → 回填 → session 目录 |
| `session_store.py` | ✅ | 六文件契约、原子写、并发锁、`build_session_json()` |
| `stream_llm_polish.py` | ✅ | 句尾单消费者 FIFO 异步 DeepSeek 润色 |
| `diar_backfill.py` | ✅ | 单臂回填适配层 + 独立 CLI |
| `tests/` | ✅ | `test_bootstrap` / `test_diar_backfill` / `test_llm_polish` / `test_run_offline` / `test_session_store` / `test_smoke_real` |

`README.md` 记录的实测结果（本机）：

| 场景 | 墙钟 | 结果 |
|---|---:|---|
| slow 转写冒烟（前 30 秒音频） | 32.2 s | PASS |
| 完整 180 秒 `--fast --no-llm` | 3 m 06.5 s | `status=diarized`，33 句，2 人 |
| 完整 180 秒 `--fast`（含润色） | 2 m 59.8 s | `status=diarized`，33 句，2 人，16 句 `corrected=true` |

### 0.2 尚未完成的部分（你的主战场）

按 `STREAMING_PLAN.md` 的任务编号，剩余工作集中在 **T4 / T5**：

| # | 交付物 | 说明 |
|---|---|---|
| **T4** | `Meeting_main/stream_service.py` | HTTP/NDJSON 回放服务 + session 编排；不修改上游 `web_server.py` |
| **T4** | `Meeting_main/ws_transport.py` | 标准库手写 RFC6455 WebSocket，承载浏览器麦克风 PCM |
| **T4** | `Meeting_main/diar_web.html` | 三态实时渲染前端；可从上游 `diar_web.html` **只读参考**后重写 |
| **T5** | 会后全文级姓名推断 | 调用 `llm_correct.correct`，由前端决定是否接 |
| — | 浏览器端到端验收 | 回放通道 + 麦克风通道各跑一次 |

### 0.3 上游只读

`D:\Deepseek\diarization_cl-main` 是**只读依赖**：

- 它**不是 git 仓库**（无 `.git`），改坏了无法 `git checkout`；
- `bootstrap.py` 只负责把它的 `.zhang`、`AliMeeting`、`streaming_asr`、`scripts/evaluation` 挂进 `sys.path`；
- **不要修改** `streaming_asr/web_server.py`、`streaming_asr/diar_web.html`、`streaming_asr/llm_correct.py`、`.zhang/*`；
- 需要扩展时，在 `Meeting_main` 里新增模块，通过 import 复用上游函数。

---

## 1. 项目目标与你的职责

### 1.1 产品交互范式

**「流式草稿实时上屏（灰色斜体） + 句尾异步 DeepSeek 纠错固化（原地变黑色正体） + 结束会议后离线全局说话人回填（挂载 Speaker 标签）」**

### 1.2 你的职责

你包揽除离线声纹算法之外的全部全栈工程：

1. 流式音频双写与 VAD+ASR 切句引擎的**服务化封装**（引擎本体已存在，不要重写）；
2. 句尾 FIFO 异步 DeepSeek 纠错固化器的**服务化接入**（润色器本体已存在，不要重写）；
3. Session 目录分阶段落盘与 CLI 端到端闭环（v0 已通，保持不回归）；
4. HTTP/WebSocket 流式服务端 + 前端三态实时渲染；
5. 历史 session 的列表/加载；
6. 浏览器端到端验收与自动化测试。

### 1.3 你不需要做的事

- ❌ 不写底层离线声纹聚类算法；
- ❌ 不改 `offline_diarizer.py` 的 `backfill_speakers()` 签名；
- ❌ 不重新调 VAD / ASR 参数；
- ❌ 不引入 Flask / FastAPI / uvicorn 等重型框架（上游与 v0 都是标准库）；
- ❌ 不新增第三方 Python 依赖（沿用 `funasr` 环境）。

### 1.4 与 Agent A 的接口

Agent A 提供的唯一入口：

```python
# D:\Deepseek\diarization_cl-main\.zhang\offline_diarizer.py
def backfill_speakers(
    wav_path: str,
    utterances: List[Dict],
    models: Optional[Dict] = None,
    mode: str = "auto",
) -> Dict[str, str]:
    """返回 {utt_id: "Speaker_{p+1}"}，p 是 0-based 簇标签。"""
```

**注意**：上游返回的是字符串 `"Speaker_{p+1}"`，不是 `{"spk": p, ...}`。`Meeting_main/diar_backfill.py::to_contract()` 负责把它换算成契约形状；**不要在服务层或前端再做一次减 1**。

---

## 2. 环境、模型与防踩坑必读

### 2.1 运行环境

| 项 | 值 |
|---|---|
| 操作系统 | Windows 11 |
| Python | `C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12.15） |
| 编码 | 所有文件读写、子进程输出强制 UTF-8；运行前设 `$env:PYTHONIOENCODING="utf-8"` |
| 依赖 | 沿用 funasr 环境（funasr / soundfile / numpy / torch 等），**不新增第三方依赖** |
| 音频 | 16 kHz、16 bit、单声道 WAV；非 16k 先用 `ffmpeg` 转 |

```powershell
$env:PYTHONIOENCODING="utf-8"
```

### 2.2 Windows Python 3.12 WMI 卡死补丁（极其关键）

在导入 `funasr` 之前，必须屏蔽 `platform._wmi_query`，否则进程可能无限挂起。

**真实位置**（原稿写的 `web_server.py:58-68` 已过时）：

| 文件 | 行号 | 说明 |
|---|---|---|
| `D:\Deepseek\diarization_cl-main\streaming_asr\web_server.py` | `75-88` | `_no_wmi()` + `_get_auto_model()` 内打补丁 |
| `D:\Deepseek\diarization_cl-main\.zhang\streaming_asr_engine.py` | `13-18` | 模块导入期直接打补丁 |

标准写法：

```python
import sys, platform

def _no_wmi(*args):
    raise OSError("WMI unavailable in this service")

if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
    platform._wmi_query = _no_wmi

from funasr import AutoModel  # 必须在补丁之后
```

如果你的新模块先 import 了 `funasr`，补丁就失效了。**把补丁放在文件最顶部、`from funasr import ...` 之前。**

### 2.3 本机已就位的模型路径（绝对路径）

统一通过 `web_server.get_models()` 托管在模块级单例 `_models` 中，**严禁重复实例化**（冷启动约 90 秒）。

| 键 | 绝对路径 | 备注 |
|---|---|---|
| `vad` | `D:\Deepseek\diarization_cl-main\iic--speech_fsmn_vad_zh-cn-16k-common-pytorch\snapshots\master` | 项目本地调优副本；`config.yaml:32` 为 `speech_noise_thres: 0.7`。**不要用默认 ModelScope 缓存的 0.6 版本**（会把多轮对话粘连） |
| `spk` | `C:\Users\31817\.cache\modelscope\models\iic--speech_campplus_sv_zh-cn_16k-common\snapshots\master` | CAM++ 声纹嵌入 |
| `stream_asr` | `C:\Users\31817\.cache\modelscope\models\iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online\snapshots\v2.0.4` | 流式 Paraformer，`device="cpu"` |
| `asr` | 上游 `web_server.py` 管理：Fun-ASR-Nano → 失败回退 SEACO | 仅 `/api/compare` 等离线对比需要，流式会议服务用不到 |

**`get_models()` 的真实签名**（`streaming_asr/web_server.py:165`）：

```python
def get_models(progress=None, need_stream=False, need_offline=True):
    ...
    return dict(_models)
```

调用示例：

```python
# 流式会议：只要 vad / spk / stream_asr，不加载离线 ASR（省约 30 秒）
models = get_models(need_stream=True, need_offline=False)
```

**`preload_models()` 的真实签名**（`web_server.py:65`）：

```python
def preload_models(progress=None):
    """内部固定 get_models(progress, need_stream=True, need_offline=True)。"""
```

它**不接受** `need_stream` / `need_offline` 参数；不要写 `preload_models(need_stream=True)`，会 `TypeError`。

> ⚠️ **EDY 遗留路径（真实存在的坑）**：
> `streaming_asr/web_server.py:18` 仍然写着
> `os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")`；
> `.zhang/streaming_asr_engine.py:25` 也把 `C:\Users\EDY\.cache\modelscope` 当默认值。
> 本机不存在 `C:\Users\EDY`。当前 v0 之所以能跑，是因为 `run_offline.py` 把 `models` 显式注入给了 `StreamingSession`，没有走引擎的自动下载回退。
> **你的新服务必须显式设置**：
> ```powershell
> $env:MODELSCOPE_CACHE="C:\Users\31817\.cache\modelscope"
> ```
> 或者沿用 `get_models()` 注入模型。**不要**擅自修改上游默认值；如果确要修，先和用户确认。

### 2.4 DeepSeek API Key

复用上游 `llm_correct._get_key()`（`streaming_asr/llm_correct.py:67-82`）：

1. 先读进程环境变量 `DEEPSEEK_API_KEY`；
2. 为空则用 `winreg` 读 `HKEY_CURRENT_USER\Environment` 下的同名值；
3. 两处都没有则抛 `RuntimeError`。

**缺少 Key 不是致命错误**：润色器会 fail-soft，把 `polished_text` 回退为 `raw_text`，`status` 仍可到 `diarized`。

### 2.5 冒烟测试音频

```text
D:\Deepseek\yinp\smoke_clip_180s.wav
```

已核实存在：180 秒、3 人对话。`tests/test_smoke_real.py` 在文件缺失时会自动 skip。

### 2.6 上下游耦合

`bootstrap.py` 是唯一耦合点：

```python
REPO = Path(r"D:\Deepseek\diarization_cl-main")
_SUBPATHS = (REPO / ".zhang", REPO / "AliMeeting",
             REPO / "streaming_asr", REPO / "scripts" / "evaluation")
```

- 本项目目录始终被强制放在 `sys.path` 第一位，避免被上游同名模块遮蔽；
- `upstream_fingerprint()` 会校验 `.zhang/streaming_asr_engine.py` 与 `.zhang/offline_diarizer.py` 的 SHA256 前 16 位；
- 上游文件一变，`run_offline.py` 会打印漂移告警。**这是保护，不是 bug。**

---

## 3. 全局核心数据契约

### 3.1 单句对象（Utterance）生命周期

全系统围绕唯一 `utt_id`（如 `"utt_0001"`）做三阶段状态演进：

```json
{
  "utt_id": "utt_0001",
  "start_ms": 600,
  "end_ms": 3420,
  "status": "diarized",
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

状态序列：

```text
partial → asr_final → llm_confirmed → diarized
```

> **两条铁律**：
>
> 1. **物理采样点时钟**：`start_ms` / `end_ms` 必须由 `累计送入采样点数 * 1000 // 16000` 计算，**严禁使用 `time.time()`**。`StreamingSession.processed_ms` 已实现该口径。
> 2. **时间壳子不可变**：VAD 在句尾触发 `asr_final` 后，`[start_ms, end_ms]` 永久锁死。DeepSeek 润色只替换文本，绝不改变时间边界。

### 3.2 Session 目录文件契约（无数据库，文件级物理隔离）

每场会议在 `output/sessions/s_YYYYMMDD_HHMMSS/` 下生成以下文件：

```text
output/sessions/s_20261017_1432/
├── session_audio.wav       # 流式边收边写（每块更新 WAV header），结束时 close
├── utterances.jsonl        # 每个 asr_final 实时 append 一行
├── events.ndjson           # 所有流式事件 append，供前端回放与调试
├── transcript.json         # 流式结束瞬间（T+0）立即生成，status="transcript"
├── polished.json           # 润色线程完成后写入；--no-llm 时不存在
├── diarization.json        # 离线声纹回填完成后写入；--no-diar 时不存在
└── session.json            # 最终合并产物，status="diarized"
```

**所有 JSON 由 `session_store.SessionStore` 写入**：

- `*.json` 走「临时文件 + `os.replace`」原子替换；
- `*.jsonl` / `*.ndjson` 逐行追加，进程内加锁；
- 不要绕过 `SessionStore` 直接 `open(..., "w")` 写这些文件。

#### `utterances.jsonl`

```json
{"utt_id": "utt_0001", "start_ms": 600, "end_ms": 3420, "raw_text": "那我们这个餐饮市场的前景真"}
```

#### `transcript.json`

```json
{
  "schema_version": 1,
  "session_id": "s_smoke_180",
  "status": "transcript",
  "audio": "output\\sessions\\s_smoke_180\\session_audio.wav",
  "started_at": "2026-10-17T14:44:42+08:00",
  "ended_at": "2026-10-17T14:46:04+08:00",
  "duration_ms": 180000,
  "utterances": [
    {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 600, "raw_text": "喂你"}
  ]
}
```

#### `polished.json`

```json
{
  "utt_0001": {
    "polished_text": "喂你。",
    "corrected": false,
    "bypassed": true,
    "error": null
  }
}
```

#### `diarization.json`

```json
{
  "utt_0001": {"spk": 0, "spk_label": "speaker_0"},
  "utt_0002": {"spk": 1, "spk_label": "speaker_1"}
}
```

- `spk` 是 **0-based 整数**簇标签；
- `spk_label` 恒为 `speaker_{spk}`（全小写）；
- 上游返回的 `"Speaker_{p+1}"` 只在 `diar_backfill.to_contract()` 换算一次。

#### `session.json`

```json
{
  "schema_version": 1,
  "session_id": "s_smoke_180",
  "status": "diarized",
  "audio": "output\\sessions\\s_smoke_180\\session_audio.wav",
  "started_at": "2026-10-17T14:44:42+08:00",
  "ended_at": "2026-10-17T14:46:04+08:00",
  "duration_ms": 180000,
  "utterances": [
    {
      "utt_id": "utt_0001",
      "start_ms": 0,
      "end_ms": 600,
      "raw_text": "喂你",
      "polished_text": "喂你。",
      "corrected": false,
      "bypassed": true,
      "error": null,
      "spk": 0,
      "spk_label": "speaker_0",
      "overlap_spk": null
    }
  ]
}
```

- `status`：有 `diarization.json` 时为 `diarized`，否则为 `transcript`；润色不改变 `status`；
- `polished_text` 缺省回退为 `raw_text`；`corrected` / `bypassed` 缺省为 `false`；
- `spk` / `spk_label` 未回填时为 `null`；
- `overlap_spk` 是 v1 预留字段，v0/v1 均恒为 `null`。

### 3.3 NDJSON 事件协议（冻结）

后端只保证事件形状与顺序；渲染逻辑由前端负责。

```jsonc
// 1. 说话进行中
{"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 600, "current_ms": 1800,
 "text": "大家早上好今天我们把思露"}

// 2. 句尾封口
{"t": "asr_final", "utt_id": "utt_0001", "start_ms": 600, "end_ms": 2950,
 "raw_text": "大家早上好今天我们把思露整理一下"}

// 3. 润色固化
{"t": "llm_confirmed", "utt_id": "utt_0001", "start_ms": 600, "end_ms": 2950,
 "raw_text": "大家早上好今天我们把思露整理一下",
 "polished_text": "大家早上好，今天我们把思路整理一下。",
 "corrected": true, "bypassed": false, "error": null}

// 4. 回填进度（必须带可量化进度，不能只有一句 msg）
{"t": "progress", "stage": "diarization", "done": 12, "total": 33,
 "msg": "正在进行全局声纹聚类与说话人回填…"}

// 5. 回填结果（支持分批）
{"t": "speaker_diarized", "n_clusters": 3,
 "mapping": {"utt_0001": {"spk": 0, "spk_label": "speaker_0", "overlap_spk": null},
             "utt_0002": {"spk": 1, "spk_label": "speaker_1", "overlap_spk": null}},
 "batch": {"done": 33, "total": 33}}

// 6. 错误
{"t": "error", "msg": "...", "fatal": false}

// 7. 结束
{"t": "finish"}
```

**顺序保证**：

- 同句：`asr_partial*` → `asr_final` →（异步）`llm_confirmed`；
- `llm_confirmed` **一定晚于**同 `utt_id` 的 `asr_final`，但不保证与其它 `utt_id` 的相对顺序；
- `speaker_diarized` 一定在 `finish` 之前、且在阶段 3 完成后；
- `progress` 可以在 `speaker_diarized` 前后多次出现。

> ⚠️ **回填长尾**：本机实测，8 场 AliMeeting 的单场回填约 **6–18 分钟**（`STREAMING_PLAN.md` §4.1）。产品上不能只发一条无信息量的 `progress` 然后空等；**必须**推送可量化进度，并允许按音频时间序分批下发 `speaker_diarized`。

---

## 4. 四大模块实现规格

### 模块 1：流式 VAD + ASR 引擎（已存在，不要重写）

文件：`D:\Deepseek\diarization_cl-main\.zhang\streaming_asr_engine.py`

**类**：`StreamingSession`

**构造参数**（`:58-68`）：

```python
StreamingSession(
    session_wav_path: str,
    asr_model: Optional[AutoModel] = None,
    vad_model: Optional[AutoModel] = None,
    max_end_silence_time: int = 1000,
    max_utt_duration_ms: int = 12000,
    asr_reset_mode: str = "per_utt",
    reset_silence_ms: int = 2000,
    device: str = "cpu",
)
```

**已锁定的黄金参数（严禁更改）**：

| 参数 | 值 | 位置 |
|---|---:|---|
| `max_end_silence_time` | **1000 ms** | `:63` |
| `max_utt_duration_ms` | **12000 ms** | `:64` |
| `asr_reset_mode` | **`per_utt`**（每句 flush 尾字并清 `asr_cache`） | `:65` |
| 流式步长 | **600 ms / 9600 采样点** | `:83-84` |
| `chunk_size_asr` | `[0, 10, 5]` | `:80` |
| `encoder_chunk_look_back` | `4` | `:81` |
| `decoder_chunk_look_back` | `1` | `:82` |

**对外 API**：

```python
events = session.feed_pcm_chunk(pcm_bytes: bytes, is_final: bool = False) -> list[dict]
session.processed_ms          # int，累计采样点换算的毫秒
session.completed_utterances  # list[dict]，已锁死的句子
```

**行为**：

- 每次 `feed_pcm_chunk` 会把 PCM 追加写入 `session_audio.wav`，并调用 `_patchheader()` 实时更新 WAV 头；
- 音频按 9600 采样点攒批后才走一次 VAD + ASR；
- `is_final=True` 时，把不足一批的尾音也送入，最后 `close_wav()`；
- 触发 `asr_final` 时，`per_utt` 模式会先用 1600 个零采样点 `is_final=True` 逼出尾字，再清空 `asr_cache`，保留 `vad_cache`；
- 返回事件只有 `asr_partial` / `asr_final` 两类。

> ⚠️ `.zhang/streaming_asr_engine.py` 的 `__main__` 在 `:333` **无条件** `from offline_diarizer import backfill_speakers`。如果你只想独立跑引擎、不做回填，直接运行 `__main__` 会 `ImportError`。服务层 import `StreamingSession` 不受影响；需要无回填自检时，自己写一个薄入口或加 try/except。

### 模块 2：句尾异步 DeepSeek 润色固化器（已存在，不要重写）

文件：`D:\Deepseek\Meeting_main\stream_llm_polish.py`

**类**：`Polisher`

**构造参数**（`:91-98`）：

```python
Polisher(
    on_result: Callable[[Dict], None],
    *,
    call: Optional[Callable[[str], str]] = None,
    key: Optional[str] = None,
    timeout: float = 60,
)
```

**对外 API**：

```python
polisher.submit(ev)             # 非阻塞；ev 至少含 utt_id / raw_text / start_ms / end_ms
polisher.results()              # {utt_id: {polished_text, corrected, bypassed, error}}
polisher.close(timeout=None)    # 投递哨兵、join 线程、返回全量结果
```

**为什么必须单线程 FIFO**：第 k 句修正后的正体 `polished_text`，要作为第 k+1 句纠错的只读上下文；并发会让上下文乱序。

**短碎词短路**（当前实现）：

- 去掉标点与空白后有效字数 **≤ 2**：不发起网络请求，直接补句号，`bypassed=True`；
- **没有**内置“嗯 / 好的 / 收到”之类的附和词集合。若产品需要，建议在服务层或 `is_short()` 里扩展，并补单测；不要声称当前已支持。

**System Prompt 约束**：

- 模型 `deepseek-chat`，`temperature=0.1`，`max_tokens=512`；
- 只修同音/误识别字、补标点；
- 禁止改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语；
- 只输出严格 JSON：`{"polished": "..."}`；
- 前 3 句已固化正体作为只读语境。

**三道守卫**（`:53-66`）：

| 守卫 | 阈值 | 失败处理 |
|---|---|---|
| 长度增幅 | `> 1.5 ×` 原文 | 丢弃改写，保留 `raw_text` |
| 长度缩减 | `< 0.7 ×` 原文 | 丢弃改写，保留 `raw_text` |
| 数字集合 | 改写后数字/中文数字集合与原文不同 | 丢弃改写，保留 `raw_text` |

**异常兜底**：

- 缺 Key、断网、超时、返回非法 JSON、回调自身抛异常：**都不得杀死消费者线程**；
- 单句失败时 `polished_text = raw_text`，并带 `error` 字段；
- 默认 `timeout=60s`（原稿写的 8.0s 是旧值）。流式服务若希望更快失败，可以在构造时传 `timeout=8~15`，但要接受“慢网时更多句子回退原文”的代价。

**输出事件**：

```json
{"t": "llm_confirmed", "utt_id": "...", "start_ms": 600, "end_ms": 2950,
 "raw_text": "...", "polished_text": "...", "corrected": true,
 "bypassed": false, "error": null}
```

> 上游 `.zhang/streaming_llm_polish.py` 是一个更旧的变体（`StreamingPolisher`，`submit` / `close` 无参）。**以 `Meeting_main/stream_llm_polish.py::Polisher` 为准**，不要同时维护两套。

### 模块 3：结束会议后的并发编排与 CLI（v0 已实现，保持不回归）

文件：`D:\Deepseek\Meeting_main\run_offline.py`

**真实时序**（`:147-162`）：

1. 送完最后一块音频（`is_final=True`），逼出最后一句 `asr_final`，关闭 `session_audio.wav`；
2. **立即（T+0）** 写 `transcript.json`（`status="transcript"`），并 `build_session_json()`；
3. 若未 `--no-diar`：调用 `diar_backfill.run_backfill()`，**同步**执行回填；与此同时，润色线程仍在后台消费队列（两者并发）；
4. 回填完成后写 `diarization.json`，`build_session_json()`；
5. 若启用润色：`polisher.close()` 等队列排空，写 `polished.json`，最后 `build_session_json()`。

**最终合并顺序不变量**（`:156-160` 的注释，不要重排）：

```text
backfill → polisher.close → write_polished → build_session_json
```

**CLI**：

```powershell
python run_offline.py --wav D:/Deepseek/yinp/smoke_clip_180s.wav --session-dir output/sessions/s_smoke --fast
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--wav` | 必填 | 16 kHz 单声道 WAV |
| `--session-dir` | 自动生成 | 指定 session 输出目录 |
| `--session-root` | `output/sessions` | 未指定 `--session-dir` 时的根目录 |
| `--chunk-ms` | `600` | 每次送流的音频块长 |
| `--max-seconds` | `0`（全量） | 只处理前 N 秒 |
| `--fast` | 关 | 全速推流，不按墙钟节流 |
| `--no-llm` | 关 | 跳过 DeepSeek 润色（不生成 `polished.json`） |
| `--no-diar` | 关 | 跳过回填（`status` 停留 `transcript`） |

**成功摘要示例**：

```json
{
  "session_dir": "output\\sessions\\s_smoke_180",
  "status": "diarized",
  "utterances": 33,
  "speakers": ["speaker_0", "speaker_1"],
  "duration_ms": 180000,
  "elapsed_s": 183.6
}
```

### 模块 4：Web 服务端与前端实时交互（你的主战场）

#### 4.4.1 架构决策：新服务放在 `Meeting_main`，不修改上游

上游 `D:\Deepseek\diarization_cl-main\streaming_asr\web_server.py` 是 8735 测试台，包含 `/api/compare`、`/api/llm`、`/api/preload`、`/api/unload` 等老路由。

**本规格的决策**：

1. **不修改** `web_server.py` / `diar_web.html`（上游只读，且无 git）；
2. 在 `Meeting_main` 新建 `stream_service.py`：
   - 用标准库 `ThreadingHTTPServer` + `application/x-ndjson` + chunked；
   - 复用上游 `get_models()`、`llm_correct.correct()`；
   - 复用 `SessionStore`、`StreamingSession`、`Polisher`、`diar_backfill`；
3. 新服务默认监听 `127.0.0.1:8736`；**旧 8735 测试台原样保留**；
4. 新前端 `Meeting_main/diar_web.html` 从上游 `diar_web.html` **只读参考**后重写，不直接改上游文件。

> **唯一需要拍板的架构分叉**：如果产品要求“必须占用 8735、且老路由 100% 同端口兼容”，那需要把老路由合并进新服务或代理到旧服务。这是产品决策，**不要擅自修改上游来实现**。默认方案是双端口共存：8735 老测试台、8736 会议服务。

#### 4.4.2 回放通道（WAV 仿真流式会议）

路由建议：

```text
POST /api/stream_meeting?name=<上传文件名>&fast=1
Content-Type: multipart/form-data
```

流程：

1. 接收上传文件，限制大小（建议 ≤ 500 MB），文件名只取 basename，禁止路径分隔符；
2. 用 `ffmpeg -y -i <src> -ar 16000 -ac 1 -c:a pcm_s16le <dst>` 转 16k 单声道；
3. 创建 `SessionStore` 目录（`new_session_dir`）；
4. 加载 `get_models(need_stream=True, need_offline=False)`（模块级单例，不要每次请求重新加载）；
5. 创建 `StreamingSession(session_audio.wav, asr_model=models["stream_asr"], vad_model=models["vad"], max_end_silence_time=1000)`；
6. 按 600 ms 切片喂流：
   - 每个 `asr_final`：`store.append_utterance()`、`store.append_event()`、`polisher.submit()`；
   - 每个 `asr_partial`：`store.append_event()` + `_emit()`；
   - `llm_confirmed`：`store.append_event()` + `store.write_polished()` + `_emit()`；
   - 默认按音频墙钟节流（喂 600 ms 睡约 600 ms）；`fast=1` 时全速；
7. 送完后写 `transcript.json`，启动回填；
8. 回填期间推送 `progress`；完成后写 `diarization.json`，推送 `speaker_diarized`（可分批）；
9. 等润色队列清空，写 `polished.json`，`build_session_json()`；
10. 推送 `finish`，关闭 NDJSON 响应。

**错误处理**：

- 任何阶段失败：推送 `{"t":"error","msg":"...","fatal":true}`，清理临时文件，关闭 WAV；
- 客户端断开：`_emit()` 捕获 `OSError` 并置 `_dead`，不要让它杀死会话线程；
- 单个会话异常不得影响其它会话（每个请求独立 `StreamingSession` / `SessionStore`）。

#### 4.4.3 麦克风通道（真实实时）

**主方案：WebSocket**（标准库手写 RFC6455，约 120 行）：

```text
WS /api/stream_live
```

- 浏览器 `getUserMedia` → `AudioWorklet` → 16 kHz int16 PCM 二进制帧；
- 服务端握手、帧解码、`feed_pcm_chunk()`；
- 客户端发送文本帧 `{"t":"stop"}` 或关闭连接时，执行与回放通道相同的收尾流水线；
- 只监听 `127.0.0.1`，校验 `Origin`；
- 帧大小上限建议 1 MB，超限直接断开。

**备选方案**（仅当浏览器/代理不支持 WS 时）：

```text
POST /api/stream_live/start
POST /api/stream_live/chunk
POST /api/stream_live/stop
```

用 `session_id` 关联服务端会话。该方案是“伪流式”（每个 chunk 一个 HTTP 请求），延迟与开销都更高，**不作为默认**。

> ⚠️ NDJSON-over-POST 是**单向**的（一个 HTTP 响应体往下流），无法承载“客户端持续往上推 PCM”。回放通道可以用 NDJSON；麦克风通道必须用 WebSocket 或分块 POST。

#### 4.4.4 Session 历史接口

```text
GET /api/sessions                 -> [{session_id, status, started_at, duration_ms, n_utterances, speakers}]
GET /api/sessions/<session_id>    -> session.json 内容
```

- 只允许访问 `output/sessions/` 下的目录；
- 对 `session_id` 做白名单校验（只允许 `[A-Za-z0-9_-]`），防止路径穿越；
- 不存在时返回 404 JSON，不要抛 HTML 堆栈。

#### 4.4.5 前端三态渲染

新建 `Meeting_main/diar_web.html`（可从上游 `streaming_asr/diar_web.html` 借鉴样式与 `SPK_COLORS`，但不要修改上游文件）。

维护一个按 `utt_id` 索引的列表容器，每行：

```text
[说话人徽章占位] [时间戳] [文本 span]
```

| 收到事件 | 渲染动作 |
|---|---|
| `asr_partial` | 若 `#row-<utt_id>` 不存在则追加新行；文本设为 `text`，类名 `.draft-italic`；徽章显示 `[...]` |
| `asr_final` | 更新时间戳为 `[start_ms - end_ms]`，文本更新为 `raw_text`；右侧显示“润色中…”呼吸点 |
| `llm_confirmed` | 定位 `#row-<utt_id>`，文本平滑替换为 `polished_text`，类名切 `.confirmed-normal`；`corrected=false` 时给徽章加“未纠错”提示 |
| `progress` | 顶部进度条显示 `done/total` 或百分比；不要只显示一句无信息量的文字 |
| `speaker_diarized` | 遍历 `mapping`，为每行徽章填 `spk_label` 并按 `spk` 取色；支持分批到达 |
| `error` | `fatal=true` 时终止会话并提示；否则仅记日志 |
| `finish` | 隐藏进度条，启用“加载历史 / 全文姓名推断”等会后操作 |

**三态样式建议**：

```css
.draft-italic {
  font-style: italic;
  color: #718096;
}
.confirmed-normal {
  font-style: normal;
  color: #1a202c;
  font-weight: 500;
}
.spk-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  color: #fff;
}
```

**可选视图开关**：

- “合并连续同说话人短句”：把同一个 `spk` 连续多句在视觉上聚合成一个自然段，减少短句碎片感；底层数据不变，只是渲染分组。

**会后附加能力**：

- 保留“调用 DeepSeek 全文后校正”按钮：对 `session.json` 调 `llm_correct.correct()`，做全文级说话人姓名推断；
- 注意：`llm_correct.py` 是**会后全文级说话人微调 + 姓名推断**，与 `stream_llm_polish.py` 的**句尾单句文本纠错**不是一回事，不要合并。

#### 4.4.6 并发与资源

- `get_models()` 内部有 `_model_lock`，但**只保护加载**；推理本身不要长时间持有全局锁；
- 每个请求一个 `StreamingSession`；模型对象共享（只读）；
- `SessionStore` 已处理同进程写锁；不要绕过它；
- 回填是 CPU/GPU 重任务，建议限制同时只跑一个回填（`threading.Lock`），否则多场会议并发会把机器打满；
- 服务端只绑定 `127.0.0.1`，不要 `0.0.0.0`。

---

## 5. 开发与交付顺序（按此推进）

### Step 0：读现状，确认不重写

- 读 `README.md`、`STREAMING_PLAN.md`、`run_offline.py`、`session_store.py`、`stream_llm_polish.py`、`diar_backfill.py`；
- 跑一遍快速测试，确认基线是绿的。

### Step 1：验证 v0 CLI（回归网）

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav `
    --session-dir output/sessions/s_smoke_180 --fast --no-llm
```

确认 `session.json` 同时具备 `raw_text`、`spk`、`spk_label`。

### Step 2：实现回放服务（`stream_service.py`）

- 先只做 `POST /api/stream_meeting` + NDJSON；
- 用 180 秒音频跑通事件序列；
- 写 pytest：事件顺序、`utt_id` 一致性、`session.json` 落盘、`spk_label` 小写；
- 回填阶段必须推送 `progress`。

### Step 3：前端回放三态

- 新建 `Meeting_main/diar_web.html`；
- 接通 `asr_partial` / `asr_final` / `llm_confirmed` / `speaker_diarized` / `finish`；
- 浏览器实测：斜体草稿 → 正体固化 → Speaker 徽章；
- 加“合并连续同说话人短句”开关。

### Step 4：麦克风通道（`ws_transport.py`）

- 标准库手写 RFC6455 握手 + 帧编解码；
- 只支持 16 kHz int16 单声道二进制帧；
- 浏览器 `getUserMedia` + `AudioWorklet` 实测；
- 断线、停止、异常收尾都要走同一套 `finish` 流水线。

### Step 5：历史 session + 会后姓名推断

- `GET /api/sessions`、`GET /api/sessions/<id>`；
- “加载历史”按钮；
- “全文姓名推断”按钮，调 `llm_correct.correct()`。

### Step 6：端到端验收

- 回放通道：上传 `smoke_clip_180s.wav`，观察三态；
- 麦克风通道：说 30–60 秒，观察实时草稿与句尾固化；
- 结束后观察进度条、分批说话人标签、最终 `session.json`；
- 刷新页面加载历史 session。

---

## 6. 验收标准（Definition of Done）

### 6.1 CLI 回归

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_smoke_real.py -m slow -k transcript -v -s
```

- [ ] 快速套件全绿；
- [ ] slow 转写冒烟 PASS；
- [ ] 180 秒全量 `--fast --no-llm` 产出 `status=diarized`，句数与人数的量级与 README 基线一致（33 句 / 2 人，允许模型随机浮动）。

### 6.2 回放通道

- [ ] `POST /api/stream_meeting` 返回 `application/x-ndjson`；
- [ ] 事件顺序满足 §3.3；
- [ ] `asr_partial` 的 `current_ms` 单调不减；
- [ ] `asr_final` 的 `[start_ms, end_ms]` 与 `session_audio.wav` 采样点时钟一致；
- [ ] `speaker_diarized` 的每个 `utt_id` 都能在 `utterances.jsonl` 找到；
- [ ] `session.json` 的 `spk_label` 全小写；
- [ ] 回填期间有可量化 `progress`；
- [ ] 客户端断开不会让服务端崩溃。

### 6.3 麦克风通道

- [ ] WebSocket 握手成功，浏览器可推 PCM；
- [ ] 草稿延迟可接受（本机 CPU 流式 Paraformer，需实测并记录）；
- [ ] 停止后走完回填与 `finish`；
- [ ] 刷新页面后能加载刚结束的 session。

### 6.4 前端

- [ ] 草稿灰色斜体、固化后深色正体；
- [ ] 鼠标悬停可查看 `raw_text`；
- [ ] 说话人徽章按 `spk` 取色，颜色稳定；
- [ ] “合并连续同说话人短句”开关有效；
- [ ] 结束后进度条、说话人标签、最终文本都正确；
- [ ] 历史 session 可加载。

---

## 7. 已知坑与硬约束

| # | 坑 | 正确做法 |
|---|---|---|
| 1 | 用 `time.time()` 算时间戳 | 只用 `session.processed_ms`（采样点计数） |
| 2 | WMI 挂起 | 在 `from funasr import ...` 之前打 `platform._wmi_query` 补丁 |
| 3 | `C:\Users\EDY` 遗留路径 | 显式设 `MODELSCOPE_CACHE=C:\Users\31817\.cache\modelscope`，或注入 `get_models()` 返回的模型 |
| 4 | 中文乱码 | `$env:PYTHONIOENCODING="utf-8"`；文件读写统一 `encoding="utf-8"` |
| 5 | 并发写 JSON 互相覆盖 | 只用 `SessionStore`；它已做临时文件 + `os.replace` + 进程内锁 |
| 6 | 回填 6–18 分钟，前端空等 | 必须推 `progress`，并支持分批 `speaker_diarized` |
| 7 | NDJSON 当成双向通道 | NDJSON 只能服务端→客户端；麦克风必须 WebSocket / 分块 POST |
| 8 | `spk` off-by-one | 上游 `"Speaker_{p+1}"` 只在 `to_contract()` 减 1；前端不再减 |
| 9 | 把 `llm_correct` 与 `stream_llm_polish` 合并 | 两者职责不同：会后全文姓名推断 vs 句尾单句文本纠错 |
| 10 | 以为回填支持 `--normalize-peak` | 不支持；该参数只在评测脚本 |
| 11 | 修改上游 `web_server.py` | 上游只读且无 git；新服务放 `Meeting_main` |
| 12 | 重复实例化模型 | 用 `get_models()` 单例；冷启动约 90 秒 |
| 13 | `preload_models(need_stream=True)` | 该函数不接受该参数；直接 `POST /api/preload` 或调无参形式 |
| 14 | 新增 Flask/FastAPI | 不引入；沿用标准库 |
| 15 | 上传文件名路径穿越 | 只取 basename，白名单校验，限制大小 |
| 16 | 服务绑定 `0.0.0.0` | 只绑 `127.0.0.1` |

---

## 8. 非目标 / 红线

- ❌ 不做多标签重叠归属（`overlap_spk` 恒为 `null`，v1 不实现）；
- ❌ 不做云端上传或远程推理；
- ❌ 不做数据库；session 目录即数据层；
- ❌ 不修改离线声纹算法与 `backfill_speakers()` 签名；
- ❌ 不重新调 `1000ms + per_utt`、`max_utt_duration_ms=12000`、600ms 步长；
- ❌ 不引入新第三方依赖；
- ❌ 不修改 `diarization_cl-main` 下任何文件；
- ❌ 不把 AliMeeting 的 VAD 口径准确率当作产品准确率对外报数。

---

## 9. 校订记录

本文件相对原全栈规格稿的事实修正，逐条留痕。

| # | 原稿表述 | 仓库实况 | 处置 |
|---|---|---|---|
| 1 | Conda 环境与 CAM++ 路径为 `C:\Users\EDY\...` | 本机用户目录是 `C:\Users\31817`；对应 Python 与 CAM++ 路径均已核实存在 | §2.1 / §2.3 全部改为 31817；并新增 EDY 遗留环境变量告警 |
| 2 | WMI 补丁在 `web_server.py` 第 58–68 行 | 实际在 `web_server.py:75-88`；引擎在 `streaming_asr_engine.py:13-18` | §2.2 修正行号 |
| 3 | `web_server.py` / `diar_web.html` / `llm_correct.py` 在项目根 | 实际都在 `D:\Deepseek\diarization_cl-main\streaming_asr\` | §0.3 / §4.4.1 明确位置与只读边界 |
| 4 | `backfill_speakers(...)` 返回 `{n_speakers, elapsed_sec, mapping, segments}` | 真实返回 `{utt_id: "Speaker_{p+1}"}`；契约换算在 `diar_backfill.to_contract()` | §1.4 / §3.2 改为真实接口 |
| 5 | `diar_backfill.py` 支持 `--normalize-peak` | 真实参数为 `--session` / `--mode` / `--force` | §7 明确不支持 |
| 6 | `diarization.json` 含 `n_speakers / mapping / segments` | 实际是扁平 `{utt_id: {spk, spk_label}}` | §3.2 改为实际形状 |
| 7 | `Polisher` 的接口是 `submit_utterance` / `on_confirmed`，超时 8.0s | 实际是 `submit(ev)` / `on_result`，默认 `timeout=60`；短句短路只有“≤2 字”，没有附和词集合 | §模块 2 修正；附和词集合标为可选增强 |
| 8 | 需要从零编写 `run_offline.py` 与 `build_session_json()` | v0 已实现并通过 180 秒冒烟；`SessionStore.build_session_json()` 已存在 | §0 改为“现状盘点 + 不要重写”；剩余工作聚焦 T4 |
| 9 | 在 `web_server.py` / `diar_web.html` 上直接改造 | 上游只读且无 git；`bootstrap.py` 的设计就是只读依赖 | §4.4.1 改为在 `Meeting_main` 新建 `stream_service.py` / `ws_transport.py` / `diar_web.html`；双端口共存 |
| 10 | 麦克风用 `POST /api/stream_live/chunk` | NDJSON-over-POST 是单向的；真实麦克风需要双向通道 | §4.4.3 改为 WebSocket 主方案，分块 POST 仅备选 |
| 11 | `preload_models(need_stream=True, need_offline=True)` | 真实签名是 `preload_models(progress=None)`，内部固定加载两项 | §2.3 修正 |
| 12 | `get_models(need_stream=True, need_offline=False)` 可直接调用 | 真实签名是 `get_models(progress=None, need_stream=False, need_offline=True)`；关键字调用可行，但默认会加载离线 ASR | §2.3 写明完整签名与调用方式 |
| 13 | `polished.json` 只有 `{polished_text, bypassed}` | 实际含 `polished_text` / `corrected` / `bypassed` / `error` | §3.2 补齐 |
| 14 | 只发一条 `progress` 文字即可 | 本机回填 6–18 分钟，产品上必须可量化进度 + 分批下发 | §3.3 / §4.4.2 强化 |

---

## 10. 一句话总结

**v0 CLI 已经跑通，你不要重写它；你的主战场是把 `StreamingSession` + `Polisher` + `backfill_speakers` 包成 8736 的流式服务，再配一个三态前端。** 回放通道用 NDJSON，麦克风通道用 WebSocket；落盘只走 `SessionStore`；回填期间必须给进度；上游 `diarization_cl-main` 一行都不要改。
