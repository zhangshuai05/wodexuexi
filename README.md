# wodexuexi — 会议转写与说话人识别 monorepo

本仓库是 `Meeting_main` 产品线与 `diarization_cl-main` 离线说话人识别支线的**总代码仓库**。

## 目录

- `Meeting_main/` — 会议流式转写 + 会后定稿（CLI、8736 回放服务、前端、测试）
- `diarization_cl-main/` — 离线说话人回填、流式底座、AliMeeting 评测
  （2026-10-18 封板快照，默认只读；见 `diarization_cl-main/VENDORED_UPSTREAM.md`）

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

## 测试

```powershell
cd Meeting_main
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest -q
```

当前基线：**159 passed, 3 deselected**（2026-10-18）。

## 上游快照

`diarization_cl-main/` 是 2026-10-18 封板快照，核心指纹：

- `.zhang/offline_diarizer.py`：`eb454a8bf70d2081`，15061 bytes
- `.zhang/streaming_asr_engine.py`：`5152424c04f4e572`，15831 bytes

`Meeting_main/bootstrap.py` 按「monorepo 同级 → 项目内 vendored → 旧绝对路径」解析上游。
重新同步步骤与排除目录见 `diarization_cl-main/VENDORED_UPSTREAM.md`。

## 不进入 git

`output/`、`*.wav`/`*.flac`/`*.mp3`、模型缓存、`.env`、`not_mind/`、`artifacts/`、
`archive/` 等实验/数据集目录。本地原始工作区 `D:\Deepseek\Meeting_main` 与
`D:\Deepseek\diarization_cl-main` 仍保留，作为迁移前的参考。
