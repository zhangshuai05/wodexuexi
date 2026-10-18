# wodexuexi — 会议转写与说话人识别 monorepo

本仓库是 `Meeting_main` 产品线与 `diarization_cl-main` 离线说话人识别支线的**总代码仓库**。

## 项目概览

- **直播**：流式 VAD + Paraformer-Streaming 出草稿；句尾 DeepSeek 保守纠错。
- **会后**：CAM++ / 谱聚类说话人回填（复用 `diarization_cl-main`）。
- **最终稿**：会后本地 SEACO Paraformer 按句重转写；支持 4–15s 长段拼接与 RMS AGC。
- **评测**：标准 Levenshtein CER + Sub/Del/Ins；支持 TextGrid 或
  `--ref-txt --window` 的 2 分钟真实窗口。
- **服务**：`127.0.0.1:8736` NDJSON 回放 + 前端 `diar_web.html`。

## 快速开始

```powershell
cd Meeting_main
$env:PYTHONIOENCODING="utf-8"

# 全套快速测试
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest -q

# 回放服务（会加载模型）
C:\Users\31817\miniconda3\envs\funasr\python.exe stream_service.py
```

当前基线：**179 passed, 3 deselected**（2026-10-19）。

## 目录

- `Meeting_main/` — 会议流式转写 + 会后定稿（CLI、8736 回放服务、前端、测试）
- `diarization_cl-main/` — 离线说话人回填、流式底座、AliMeeting 评测
  （2026-10-19 已同步长窗主投票、句间迟滞与句内切分门控；见 `diarization_cl-main/VENDORED_UPSTREAM.md`）

## 环境

| 项 | 值 |
|---|---|
| Python | `C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12） |
| 编码 | `$env:PYTHONIOENCODING="utf-8"` |
| AISHELL-4 | `D:\yinp\AISHELL-4\data\test\` |
| AliMeeting | `D:\yinp\AliMeeting\` |
| ModelScope 缓存 | `C:\Users\31817\.cache\modelscope\` |
| OSD ONNX | `D:\yinp\pyannote_seg3\model.onnx` |
| VAD 调优副本 | 已随仓库带入 `diarization_cl-main/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/` |

`DEEPSEEK_API_KEY` 从进程环境或 Windows 用户级注册表读取；**不要提交密钥**。
音频、模型缓存、会话输出和实验产物不进入 git。

## 主要文档

- `Meeting_main/README.md` — 运行方式、参数、session 六文件契约
- `Meeting_main/HANDOFF.md` — 当前状态、实测数据、坑位
- `Meeting_main/STREAMING_PLAN.md` — 架构与已校订事实
- `Meeting_main/docs/superpowers/specs/2026-10-18-real-mic-baseline-design.md` — 真实麦克风基线设计
- `diarization_cl-main/HANDOFF.md` — 上游说话人支线交接
- `diarization_cl-main/HANDOFF_OFFLINE_DIAR_AGENT.md` — 上游契约红线

## 上游快照

`diarization_cl-main/` 当前同步快照的核心指纹（LF 规范化后）：

- `.zhang/offline_diarizer.py`：`efed74f75b4f770e`，25333 bytes
- `.zhang/streaming_asr_engine.py`：`5152424c04f4e572`，15831 bytes

`Meeting_main/bootstrap.py` 按「monorepo 同级 → 项目内 vendored → 旧绝对路径」解析上游。
重新同步步骤与排除目录见 `diarization_cl-main/VENDORED_UPSTREAM.md`。

## 不进入 git

`output/`、`*.wav`/`*.flac`/`*.mp3`、模型缓存、`.env`、`not_mind/`、`artifacts/`、
`archive/` 等实验/数据集目录。本地原始工作区 `D:\Deepseek\Meeting_main` 与
`D:\Deepseek\diarization_cl-main` 仍保留，作为迁移前的参考。
