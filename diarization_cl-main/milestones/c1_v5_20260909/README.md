# c1 v5 里程碑：2026-09-09

用户确认可用的 c1 说话人分离/识别版本，方法标识 `claude-a2-v5-c1`。本目录是独立归档；v4 与工作区、DSH 测试台的历史文件均保留。

## v5 变更

- 段级聚类出的身份质心在短窗边界细化时保持固定，不再被混合短窗拖移。
- 仅在已有身份锚的细化路径消除低置信度的单窗标签翻转，保留高置信度短答。
- 段级回退与密集窗救援都按相邻窗口中心放置边界，保证输出时间段单调且不重叠，并保留真实 VAD 空洞。
- ASR 文本不参与说话人分离、身份聚类或本轮评估；保留文本字段仅为现有测试台接口兼容。

正式 `gt_ts` 三文件（collar 0.5s + trim）DER 从 5.690% 降至 5.345%，其中 `spk_err` 从 4.655% 降至 4.310%。可信安静五文件 DER 从 7.806% 降至 7.708%。嘈杂档自动 GT 不可靠，未用于 v5 结论。

## 内容

- `claude_a2.py`：已部署测试台的完整 c1 v5 算法和兼容适配层。
- `a2_v2.py`：同算法的离线研究核心。
- `test_c1_refinement.py`、`compare_c1.py`、`local_eval.py`：无需加载语音模型的算法自检及其最小依赖。
- `denoise_frcrn.py`、`env_detect.py`、`vendor_pkgs/`：保留与原测试台一致的输入预处理。
- `run.py`：独立运行入口，配置模型后可复现转码、环境检测、前处理与分离链路。
- `verify.py`、`SHA256.json`：归档完整性校验；有模型和音频时可做 v5 输出契约检查。
- `evidence/`：v4 历史接口结果及本轮 v5 分离评估和归档记录。

不包含模型权重、音频、Python 环境或 FFmpeg 二进制。模型路径通过 `models.example.json` 提供；安装依赖、系统要求及常用运行方式见 `request.txt`。

## 校验与运行

```bash
python verify.py
python test_c1_refinement.py
python run.py /data/input.wav --models-config models.example.json -o output/result.json
```

`run.py` 的 `--denoise auto|on|off` 行为与测试台保持一致。若另行提供 `Note20260803112329.wav` 与 `Note20260803111855.wav`，可执行：

```bash
python verify.py --models-config models.example.json --audio-dir /data/reference_audio
```

该检查确认 v5 方法标识、双说话人输出、时间边界合法且无重叠；它不把文本转写作为算法质量指标。
