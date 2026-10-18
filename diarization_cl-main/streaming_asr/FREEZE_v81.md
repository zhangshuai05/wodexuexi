# v8.1 冻结清单（2026-09-17）

v7 的权威算法基线是 `../milestones/c1_v7_20260910/claude_a2.py`，SHA256 为
`c8c37e5d8cca62868f6eca21a6456c10d7a0e040b77cf0164e4e57693ea05aa5`。
`tests/test_v7_frozen_core.py` 会校验归档哈希及当前 v7 的核心辅助函数；后续版本必须使用
独立模块，不在 v7 核心上继续叠加算法改动。

本文件是 PLAN_0917.md 第一步"冻结输入，让对照可重复"的落盘产物。
只记录**可校验的哈希与事实**；不含任何未复现的测量数字。

## 1. 代码与配置

| 文件 | sha256[:16] | 行数 |
|---|---|---|
| `claude_a2_v7_v8.py` | `3526abbad2ea515e` | 821 |
| `claude_a2_v8_contract.py` | `718467e95eb674d7` | 130 |
| `claude_a2_v8_evidence.py` | `8232de7bc79ab1ae` | 202 |
| `claude_a2.py`（泳道分发） | `07ad66806712f3b1` | 43 |
| `web_server.py` | `b0f753568d307a39` | 452 |
| `denoise_frcrn.py` | `79f7300d815614bd` | 119 |
| `env_detect.py` | `6290d584883f55d2` | 95 |

方法名：`claude-a2-v7-c1`。**这是"含 verdict 修订的基线"**——与历史同名结果
不可直接比较（历史 v7 的 `verdict` 是无条件字面量 `reliable`）。PLAN §32 要求的
"避免继续用同一方法名混装历史结果"，实际做法：本清单的哈希即版本标识，凡引用
本哈希的结果都带 verdict 修订；不带哈希的历史结果按旧行为理解。

## 2. 模型权重

| 模型 | 文件 | sha256[:16] |
|---|---|---|
| CAM++ 声纹 | `campplus_cn_common.bin` | `3388cf5fd3493c9a` |
| CAM++ 声纹 | `config.yaml` | `17342041bd5b22f6` |
| FSMN VAD | `model.pt` | `b3be75be477f0780` |
| FSMN VAD | `config.yaml` | `86c77f4324827afe` |
| FSMN VAD | `am.mvn` | `6820fef9687708c4` |

路径：CAM++ 在 `C:\Users\EDY\.cache\modelscope\models\iic--speech_campplus_sv_zh-cn_16k-common\snapshots\master`；
VAD 走仓库内调优副本（`web_server.VAD_LOCAL`，`speech_noise_thres: 0.7`，见
`tests/test_config_consistency.py`），**不是** modelscope 缓存里那份。

## 3. 已确认的不确定性（PLAN §12/§35 的前提）

**FRCRN 在 GPU 上逐次输出不同。** 实测同一输入两次降噪：
`max|Δ| = 3.05e-05`（相对 `3.08e-05`）。

这个量级在后续离散判决上会被放大：

| 条件 | 同一份音频连跑 3 次的分段数 |
|---|---|
| `denoise=auto`（走 FRCRN） | **23 / 23 / 54** |
| `denoise=off` | **31 / 31 / 31** |

结论：
1. **不降噪时是确定的**（3/3 相同），所以"固定特征重跑聚类"这一层可以做干净对照。
2. **只要经过 FRCRN，同一输入的两次运行不可直接比较**。任何以 v7 全链路
   （含降噪）为基线的 A/B，都必须固定降噪输出，不能用"跑两遍"当对照。
3. 本清单建立之前我用过一次"含 FRCRN 的两遍对照"，其结论已作废并删除
   （见 `claude_a2_v7_v8.py` 细化解说处的记录）。

## 4. 复现方式

```powershell
# 服务（必须用系统 Python 3.12，不是 conda funasr 环境）
C:\Users\EDY\AppData\Local\Programs\Python\Python312\python.exe -u web_server.py --host 127.0.0.1 --port 8735

# 单测
python -m pytest -q          # 37 passed

# 固定输入重跑（第一步的分层重复）
#   denoise=off  → 期望逐次相同
#   denoise=auto → 期望可能不同，记录首次分歧位置
```

## 5. 尚未完成（PLAN 第一步的其余部分）

- 冻结的 FRCRN 音频尚未落盘留档（目前每次请求现算，所以发散）。
- VAD 区间 / 长窗 / 密集窗嵌入尚未落盘成缓存（缓存键需含：输入哈希、上述模型
  哈希、采样率、`WIN_S`/`HOP_S`）。
- 分层重复五次的实验脚本尚未写。
- 评分入口（SciPy 矩形最优一一匹配、未匹配计错、不截断）尚未建。
