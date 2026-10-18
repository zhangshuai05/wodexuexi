# Vendored upstream snapshot — diarization_cl-main

- **Source**: `D:\Deepseek\diarization_cl-main`（上游不是 git 仓库）
- **Snapshot date**: 2026-10-18（上游封板版本）
- **Status**: read-only vendor snapshot；不要在本目录直接改上游算法。

## Core fingerprints

| 文件 | SHA256 前 16 位 | bytes |
|---|---|---|
| `.zhang/offline_diarizer.py` | `eb454a8bf70d2081` | 15061 |
| `.zhang/streaming_asr_engine.py` | `5152424c04f4e572` | 15831 |

与 `Meeting_main/bootstrap.py::BASELINE` 一致；若不一致，`upstream_fingerprint()` 会告警。

## Included

- `.zhang/`（离线回填接口、流式底座、异步润色、e2e）
- `AliMeeting/`（核心算法链、评测器、results）
- `streaming_asr/`（web_server、llm_correct、diar_web.html、funasr_backend、tests）
- `docs/`、`scripts/`
- `milestones/diar_fix_20261018/`、`milestones/diar_final_20261018/` 及其他里程碑
- 本地 VAD 调优副本 `iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/`（model.pt 约 1.6MB）
- 根目录文档/配置（README、HANDOFF、HANDOFF_OFFLINE_DIAR_AGENT、DESIGN、PRODUCT、
  requirements、models.example.json、SHA256.json 等）

## Excluded

`not_mind/`（图片/资料）、`artifacts/`（实验产物）、`output/`（会话/日志）、
`archive/`、`web/`、`v12_meeting/`、`dsh_line/`、`eval_dsh/`、`evidence/`、
`vendor_pkgs/`、`.claude/`、`.impeccable/`、`.vscode/`、`__pycache__/`、`*.log`。

## Re-sync

1. 在上游源目录确认封板指纹；若基线变化，先更新 `Meeting_main/bootstrap.py::BASELINE`。
2. 按上面的 Included/Excluded 列表重新复制（不要复制 `output/`、音频、模型缓存）。
3. VAD 模型被上游 `.gitignore` 的 `iic--*/`/`*.pt` 忽略，需要 `git add -f`。
4. 跑 `Meeting_main` 的 `pytest -q` 与 `bootstrap.check_paths()`。
5. 更新本文件的日期与指纹。
