# Claude v12 Meeting 里程碑：2 秒窗会议说话人识别

归档日期：2026-09-22  
方法标识：`claude-a2-v12-meeting`  
定位：实验版、可复现的多人长会议说话人分离泳道。

## 一句话

v12 用 **2.0 秒 CAM++ 长窗确定说话人身份和聚类人数**，再用 **0.6 秒密集窗定位换人边界**。短窗只按长窗质心分配标签，不重新聚类，避免身份漂移。

## 目录内容

- `claude_a2.py`：v7–v12 泳道分发器，v12 入口是 `run_claude_lane(..., "v12")`。
- `claude_a2_v12.py`：v12 主体。
- `claude_a2_v11.py`：VAD 分段、窗口提取、AHC 候选、silhouette 选 K、质心和 ASR 切分基础。
- `claude_a2_v7_v8.py`：复用 v7 的碎段稳定化与吸收逻辑。
- `claude_a2_v8_contract.py`：逐段 ASR 可疑性契约。
- `web_server.py`：本地 8735 测试台服务。
- `v12_aishell4.py`：AISHELL-4 的 embedding / EER / 聚类离线实验工具。
- `tests/test_v12_aishell4.py`：不加载声学模型的口径与聚类纯函数测试。
- `tests/test_config_consistency.py`：确认 `milestones/` 级别的调参 VAD 使用 `speech_noise_thres=0.7`。
- `SHA256.json` / `VAD_SHA256.json`：代码归档与配套 VAD 文件的哈希清单。

归档配套的调参 FSMN-VAD 配置与权重位于本目录的父级 `milestones/iic--speech_fsmn_vad_zh-cn-16k-common-pytorch/`，与原工作区的相对路径一致；没有归档示例音频、其他声学模型、上传缓存、`output/` 和 `__pycache__`。

## 代码执行链

```text
web_server.py
  -> claude_a2.py
    -> claude_a2_v12.run_claude_lane()
      -> v11._vad_segs()
      -> v11._extract_windows(2.0s / 1.0s)
      -> v11._candidate_clusters()
      -> v11._select_candidate()       # silhouette + 小 K 惩罚
      -> _refine_windows(0.6s / 0.15s)
      -> _refine_labels()               # 最近长窗质心，不重新聚类
      -> _segment_windows()
      -> v11._split_for_asr(max 8s)
      -> segment_contract()
      -> sentence_info + diag
```

## 关键实现

### 1. 运行时改写 v11 窗口参数

v11 的 `WIN_S`、`HOP_S`、`MAX_SPEAKERS` 是模块级常量。v12 用 `_v11_window_params` 临时切换到 2.0/1.0 秒，并用线程锁保护，服务结束后恢复原值。这样没有复制一套 v11 聚类代码。

### 2. 长窗定身份

`v11._extract_windows()` 得到归一化 CAM++ embedding。`v11._candidate_clusters()` 使用 average-linkage + cosine AHC 构造 K=1..10 候选；`v11._select_candidate()` 过滤过小簇，再以 silhouette 减少量 K 惩罚选候选。

这里的 K 是“最适合当前 embedding 切分的候选”，不是保证等于真实人数。

### 3. 短窗找边界

`_refine_windows()` 提取 0.6/0.15 秒短窗。`_refine_labels()` 将每个短窗与长窗聚类质心做余弦相似度，分配给最近质心，然后复用 `_stabilize_labels()` 和 `_absorb_slivers()` 清理孤立翻转和弱碎段。

短窗不调用 AHC，不创建新身份；这是 v12 最重要的结构约束。

### 4. 两条兜底

- 长窗选出至少 2 人但短窗不足或塌成单标签：退回长窗分段。
- 长窗选成 1 人：才用短窗重新做一次候选聚类，只有确实分出更多人时才接受救援。

当前缺口是：短窗细化虽然保留了至少两个标签，也可能比长窗结果更差；v12 还没有完整的结果质量回退守卫。

### 5. ASR 与诊断

每个说话人段最多切成 8 秒输入 ASR；`segment_contract()` 给每段添加 `flag`、字符速率、有声时长和可疑原因。返回的 `diag` 记录实际窗口、候选 K、场景、最大 VAD 间隙、碎段数量和是否走细化/救援路径。

## 实验结论

- AISHELL-4 开发集 2.0 秒 CAM++ 纯单人窗 EER：8.97%；1.2 秒：14.58%；0.6 秒：26.37%。
- 2.0 秒上 ERes2NetV2 与 CAM++ 基本持平，但 CAM++ 推理约快 2.28 倍，因此保留 CAM++。
- 已知人数的纯窗聚类准确率约为 dev 93.12%、holdout 92.77%。
- 录音卡代理集上 v12 加权轮次指标 0.586，v7 为 0.591；不能宣传为分离质量提升。
- v12 不是已验收生产算法；它的价值是建立长会议、多人数、自动 K 的实验基线。

## 运行

在本目录、模型缓存已经准备好的环境中：

```powershell
python -m pytest -q
python -u web_server.py --host 127.0.0.1 --port 8735
```

服务使用外部 ModelScope/FunASR 模型缓存加载 CAM++、ASR 等模型；调参 FSMN-VAD 已随里程碑保留。

## 研究边界

后续 v13 不应直接把“换 VBx 后端”当成修复方案：固定 CAM++ 的 spike 显示，VB HMM 继承 AHC 的初始簇，不能凭空恢复被并掉的人；官方 VBx 权重也与 CAM++ 的 192 维 embedding 不兼容。下一步应先固化 DER 口径，再研究如何选择更好的聚类切分，而不是单纯追求预测人数等于真值。
