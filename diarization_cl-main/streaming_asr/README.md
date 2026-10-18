# Claude 说话人分离测试台

这是 `http://127.0.0.1:8734/` 页面与服务的独立副本，监听
`http://127.0.0.1:8735/`。页面保留原有布局、上传队列、音频播放、进度和结果
展示，提供 Claude v7、v8、v9、v10 E1 和 v11 多人聚类实验，不包含 DSH 泳道、快照和重分配。

v10 当前先对整段音频做连续 Paraformer-Streaming 因果回放，再用冻结 v7 生成离线
对照结果，避免把流式草稿误当最终稿。它尚非双轨在线会议服务；查看最终体验需点击
一次“LLM 校正”。

v11 是独立的离线多人实验泳道：在同一套 CAM++ 密集窗上比较 1--4 人聚类候选，
并把 ASR 输入限制在 8 秒以内，避免错误标签吞并整段对话。它不替换冻结 v7；
当前三人拼接实验的完整记录见 `V11_EXPERIMENT_20260917.md`。

v12 正在使用 AISHELL-4 真实 5--7 人长会议做离线模型选型。第一阶段确定 CAM++ 2 秒窗
优于现有短窗方案，ERes2NetV2 未在身份隔离留出组胜出；详见 `V12_EXPERIMENT.md`。
v12 尚未加入 8735 方法列表。

运行前可选择自动、强制或关闭 FRCRN 降噪。每个结果都可单独调用 DeepSeek
进行 LLM 校正；校正失败时保留原始结果。

项目不导入 DSH 仓库。VAD、CAM++、Fun-ASR-Nano 和 SEACO 回退模型从
`C:\Users\EDY\.cache\modelscope\models` 只读加载。

启动：

```powershell
.\start.ps1
```

验证：

```powershell
pytest -q
```

v10 E1 Paraformer-Streaming 因果回放探针（首次运行会下载流式模型）：

```powershell
python v10_asr_probe.py <16k-mono.wav>
```

输出保存为 `output/v10_asr_probe.json`。`--fast` 只用于功能调试，延迟验收必须使用默认
的 1 倍速回放。端点 Nano 和 `fa-zh` 对齐是较慢的独立阶段，分别用
`--endpoint-nano` 和 `--align` 显式开启。
