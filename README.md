# Meeting_main v0

离线会议转写 CLI：把 16 kHz 单声道 WAV 走完「流式草稿 → 句尾固化 → 异步润色 → 会后单臂说话人回填」，
产出 `output/sessions/s_*/` 六文件契约。v0 只做 WAV 回放，不启 HTTP、不做实时麦克风输入。

算法复用 `D:\Deepseek\diarization_cl-main`（**只读**，不复制、不修改）；`bootstrap.py` 是唯一耦合点，
它只把上游路径挂进 `sys.path`，并保证本仓库模块始终排在最前。

## 环境

| 项 | 值 |
|---|---|
| Python | `C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12.15） |
| 输入 | 16 kHz 单声道 WAV；其他采样率直接报错退出 |
| 依赖 | 沿用 funasr 环境（funasr / soundfile / numpy 等），**不新增第三方依赖** |
| 模型 | VAD / CAM++ / Paraformer-Streaming 均为本地模型，路径由上游 `streaming_asr/web_server.py` 常量托管 |

### 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `PYTHONIOENCODING` | 建议 | 设为 `utf-8`，否则 Windows 控制台按 GBK 解析中文输出会乱码 |
| `DEEPSEEK_API_KEY` | 可选 | 仅异步润色需要。进程环境变量或 Windows 用户级环境变量（`HKEY_CURRENT_USER\Environment`）任一处配置即可（`llm_correct._get_key()` 两处都会读）。缺失或调用失败时 `status` 仍为 `diarized`，只是该句润色字段带 `error`（fail-soft） |

```powershell
$env:PYTHONIOENCODING="utf-8"
$env:DEEPSEEK_API_KEY="sk-..."   # 可选；不配则跳过润色效果，仅保留 raw_text
```

## 运行

### 实时节奏（按音频墙钟节流）

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav
```

不传 `--fast` 时，送流速度与音频播放速度一致：180 秒音频至少跑 180 秒（模型加载与回填另计）。

### 全速（`--fast`，推荐离线批处理）

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav `
    --session-dir output/sessions/s_smoke_180 `
    --fast --no-llm
```

去掉 `--no-llm` 即启用 DeepSeek 润色（需要 `DEEPSEEK_API_KEY`）。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--wav` | 必填 | 16 kHz 单声道 WAV |
| `--session-dir` | 自动生成 | 指定 session 输出目录 |
| `--session-root` | `output/sessions` | 未指定 `--session-dir` 时的根目录，自动建 `s_YYYYMMDD_HHMMSS` |
| `--chunk-ms` | `600` | 每次送流的音频块长（毫秒） |
| `--max-seconds` | `0`（全量） | 只处理前 N 秒，调试用 |
| `--fast` | 关 | 全速推流，不按墙钟节流 |
| `--no-llm` | 关 | 跳过 DeepSeek 润色（不生成 `polished.json`） |
| `--no-diar` | 关 | 跳过会后说话人回填（`status` 停留 `transcript`，不生成 `diarization.json`） |

成功时打印 JSON 摘要（实际输出示例）：

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

## Session 六文件契约

六个契约文件之外，`session_audio.wav` 是流式边收边写的音频旁路（由上游 `StreamingSession` 负责，
`SessionStore` 不做写入）：16 kHz 单声道 PCM_16，与输入等长——使用 `--max-seconds` 时以**截断后**的
送流长度为等；它是采样点时钟的唯一真相。

`--no-llm` 模式下 `polished.json` 不会创建：此时实际契约是「五个文件 + `session_audio.wav`」，
而不是六个文件。

所有 JSON 均由 `session_store.SessionStore` 写入：`*.json` 走「临时文件 + `os.replace`」原子替换，
`*.jsonl` / `*.ndjson` 逐行追加（进程内加锁）。`session.json` 是 `transcript.json` + 可选
`polished.json` + 可选 `diarization.json` 的幂等合并结果，可被多次重写。

| 文件 | 何时生成 | 形状 |
|---|---|---|
| `utterances.jsonl` | 每句 `asr_final` 追加一行 | `{"utt_id":"utt_0001","start_ms":0,"end_ms":600,"raw_text":"喂你"}` |
| `transcript.json` | 转写结束（T+0 全量） | 见下 |
| `polished.json` | 每句润色完成即原子重写；`--no-llm` 时不存在 | 见下 |
| `diarization.json` | 回填完成；`--no-diar` 时不存在 | 见下 |
| `session.json` | 每次合并（幂等） | 见下 |
| `events.ndjson` | 所有流式事件逐行追加 | 见下 |

### `transcript.json`

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

### `polished.json`

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

### `diarization.json`

```json
{
  "utt_0001": {"spk": 0, "spk_label": "speaker_0"},
  "utt_0002": {"spk": 1, "spk_label": "speaker_1"}
}
```

`spk` 是 **0-based 整数**簇标签；`spk_label` 恒为 `speaker_{spk}`（全小写）。
上游返回的 `Speaker_{p+1}` 只在 `diar_backfill.to_contract()` 里换算一次。

### `session.json`

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

- `status`：有 `diarization.json` 时为 `diarized`，否则为 `transcript`；润色不改变 `status`。
- `polished_text` 缺省回退为 `raw_text`；`corrected` / `bypassed` 缺省为 `false`。
- `spk` / `spk_label` 在未回填时为 `null`。
- `overlap_spk` 为 v1 预留字段，v0 恒为 `null`。
- `start_ms` / `end_ms` **只由累计采样点数**（`samples * 1000 // 16000`）算出，不用墙钟。

### `events.ndjson`

每行一个事件对象，实际出现的三类：

```json
{"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 0, "current_ms": 1200, "text": "喂"}
{"t": "asr_final", "utt_id": "utt_0001", "start_ms": 0, "end_ms": 600, "raw_text": "喂你"}
{"t": "llm_confirmed", "utt_id": "utt_0001", "start_ms": 0, "end_ms": 600, "raw_text": "喂你", "polished_text": "喂你。", "corrected": false, "bypassed": true, "error": null}
```

顺序保证：同句 `asr_final` 早于 `llm_confirmed`；跨句的 `llm_confirmed` 按对应 `asr_final`
的提交顺序处理（单消费者 FIFO）。只保证上述两条，与其他事件类型的交错顺序不做保证。

## 说话人回填独立重跑

回填崩溃或想换 `--mode` 时，可单独重跑（只读 `utterances.jsonl`，写 `diarization.json` 并重建 `session.json`）：

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe diar_backfill.py --session output/sessions/s_smoke_180
```

可选参数：`--mode {auto,class,vad}`（默认 `auto`）、`--force`（忽略已存在的 `diarization.json`）。
不加 `--force` 时，若 `diarization.json` 已存在则直接复用、不重复计算。

## 测试

```powershell
# 快速套件（pytest.ini 默认 -m "not slow"）
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest

# slow 冒烟（真实音频；只有转写那条）
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_smoke_real.py -m slow -k transcript -v -s

# slow 冒烟（两条都跑，含完整回填，耗时更长）
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_smoke_real.py -m slow -v -s
```

slow 测试依赖 `D:\Deepseek\yinp\smoke_clip_180s.wav`；文件缺失时整条测试自动 skip。

## 实测耗时

本机基线：CPU 跑流式 Paraformer，MX450 2 GB；测量时另有上游任务同时占用 CPU/GPU。
数值为本机量级参考，换机器需重测。

| 场景 | 命令 | 墙钟 | 内部 `elapsed_s` | 结果 |
|---|---|---|---|---|
| slow 转写冒烟（前 30 秒音频） | `pytest ... -k transcript` | **32.2 s** | 29.31 s（pytest 计时） | PASS |
| 完整 180 秒（`--fast --no-llm`） | `run_offline.py ... --fast --no-llm` | **3 m 06.5 s** | 183.6 | `status=diarized`，33 句，2 人 |
| 完整 180 秒（`--fast`，含润色） | `run_offline.py ... --fast` | **2 m 59.8 s** | 177.1 | `status=diarized`，33 句，2 人，16 句 `corrected=true` |
