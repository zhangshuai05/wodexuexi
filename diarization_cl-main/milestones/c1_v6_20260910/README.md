# c1 v6 里程碑：2026-09-10

c1 说话人分离/识别版本，方法标识 `claude-a2-v6-c1`。本目录是独立归档；v4、v5 与工作区、DSH 测试台的历史文件均保留。

## v6 变更

- 短窗细化路径增加**低置信碎段吸收**：只吸收「时长 < 0.75s 且逐窗边距绝对值均值 < 0.18」的标签 run，并入边距更高的邻居。
- 两个阈值参数化（`min_dur` / `weak_margin`），真短答（本批 0.9~2.6s、边距 0.19~0.41）不受影响。
- 其余与 v5 相同（稳定身份锚、短窗换人边界、单调无重叠时间段、保留真实 VAD 空洞）。
- ASR 文本不参与说话人分离、身份聚类或本轮评估；保留文本字段仅为现有测试台接口兼容。

正式 `gt_ts` 三文件（collar 0.5s + trim）DER 从 v5 的 5.345% 降至 **4.828%**，碎段由 5 处减到 1 处，无文件回退。逐项对比见 `evidence/v6_评估_20260910.md`。嘈杂档自动 GT 不可靠，未用于 v6 结论。

## 内容

- `claude_a2.py`：已部署测试台的完整 c1 v6 算法和兼容适配层。
- `a2_v2.py`：同算法的离线研究核心。
- `test_c1_refinement.py`、`compare_c1.py`、`local_eval.py`：无需加载语音模型的算法自检及其最小依赖。
- `denoise_frcrn.py`、`env_detect.py`、`vendor_pkgs/`：保留与原测试台一致的输入预处理。
- `run.py`：独立运行入口，配置模型后可复现转码、环境检测、前处理与分离链路。
- `verify.py`、`SHA256.json`：归档完整性校验；有模型和音频时可做 v6 输出契约检查。
- `evidence/`：历次接口结果及本轮 v6 分离评估和归档记录。

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

该检查确认 v6 方法标识、双说话人输出、时间边界合法且无重叠；它不把文本转写作为算法质量指标。
