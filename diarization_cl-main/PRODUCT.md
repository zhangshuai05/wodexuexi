# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

沿用仓库现有做法：原生 HTML/CSS/JS + 一个 Python `http.server`（与 `streaming_asr/web_server.py` 8735、`streaming_asr/bench/bench_server.py` 8736 同构）。不引入 node/构建链。新服务端口 `127.0.0.1:8737`。

## Users

主用户是算法开发者本人；小范围内部使用（约 ≤5 人，含看效果的同事）。
使用情形：本地单机、一次看一场真实会议、反复拧参数并立刻看结果，不需要登录、不需要公网、不需要多人并发。

## Product Purpose

把"选 K"这条决策链从数字变成**可看、可拧、可对照**的界面：
在 AISHELL-4 的真实会议上并排比较不同选 K 配置，同时看到**真值对照**、**句级四类分解**（对/弃权/未映射/错答）与**真 DER**，拧动旋钮后立刻更新。
成功＝调试一轮算法不再需要"改脚本 → 等 21 分钟 → 看打印"，而是秒级看到同一场的多条泳道差异。

## Positioning

它是**项目内部的诊断台**，不是产品页：只服务"这场会议某条配置到底行不行"这一个问题。
它独有的是把三件本来分离的东西合到一处：真值对照（RTTM/TextGrid）、生产的段表口径（DER 与生产同一份代码）、以及秒级的参数扫描（λ / min_win / cap / mode / 细化开关）。

## Operating Context

- 数据：AISHELL-4 的**窗缓存**（50 场 = dev 12 + holdout 8 + train_L 30），每场含主窗/细化/救援三阶段嵌入；dev 12 有真值参照与真模型产物。
- 真值：`TextGrid/*.rttm`（帧级 DER）与 `TextGrid/*.TextGrid`（句级归属率）。
- 缺缓存或需要新音频时，回退到真模型路径（`v12.diarize`），成本分钟级——不是主路径。
- 使用节奏：挑一场 → 看多泳道 → 拧旋钮 → 对比 → 换一场。会话式，不需要保存历史。
- 术语沿用仓库：泳道（lane）、选 K、细化（refine）、救援（rescue）、句级归属率、双射/多对一（bijective/oracle）、重复簇（dup）。

## Capabilities and Constraints

**必须有的**
- 场次选择：50 场，标出 cohort 与"有真值参照"（dev 12）；显示真值人数、时长。
- 多泳道并排：`base`（生产选 K）、`unlocked`（评估侧实现）、`oracle 上界`（诊断口径，读真值）；以及每条泳道的 `merged`（消重复簇）读数。
- 旋钮：λ（0–0.03）、`min_win`（2/3/4）、`cap`（4–10）、`mode`（objective/largest_eligible）、细化开/关。
- 指标：句级归属率（overall / **clean** / overlap 三栏）、弃权、未映射、错答、选中的 k、真 K、可见 K、`refined`/`2s-only`；以及 **DER**（含 miss/FA/confusion 分解）。
- 时间轴叠加真值段与各泳道预测段 + 音频播放与点击定位。

**硬约束**
- **DER 必须来自生产同一份代码**：缓存路径的段表要用 `v12._segment_windows` 与 k<2 救援分支复刻，并用已存真模型产物做**12 场 × 2 臂对拍**；对不上就不许把缓存 DER 当结论（页面上要显式标出对拍状态）。
- 不动：`streaming_asr/diar_web.html` 与 `web_server.py`（8735，另一条线在用）、`not_mind/shiyan/fe/`、`milestones/` `archive/` `backups/` 里的前端副本。
- `streaming_asr/bench/`（8736）的功能被 `web/` 吸收；旧目录先保留，确认无功能缺失后再删除（走回收站）。
- 不引入新依赖（除已有的 numpy/scipy/soundfile 等）。

**未定**
- 是否/何时把 8735 并入 `web/`（等 `web/` 稳定后再定）。
- 上传任意新音频走真模型（第二轮）。

## Evidence on Hand

- 窗缓存：`D:\yinp\AISHELL-4\_v13_spike\cache\probe\`（50 场三阶段 npz + `manifest.json`）。
- 真模型对照靶子：`artifacts/v13_e2e_unlocked2_dev_mw2_objective.json`（dev 12 场 base/fixed 两臂逐场 DER 与 miss/FA/confusion）。
- 探针与口径实现：`scripts/evaluation/{probe_ab_dev,probe_path_equiv,probe_sentence_attribution,unlocked_selector,probe_oracle_k_curve}.py`。
- 生产路径：`streaming_asr/claude_a2_v12.py`（`diarize` / `_segment_windows` / 救援分支）、`claude_a2_v11.py`（候选与选 K 常数，含 `MULTI_K_PENALTY = 0.005`）。
- **不得编造**：任何未跑过对拍的 DER、任何非本仓库来源的会议数据、"上线级"结论。

## Product Principles

1. **数字必须可追到出处**：每个读数都标口径（缓存/真模型、overall/clean/overlap）、口径版本与对拍状态。
2. **秒级优先**：能用缓存算的绝不跑模型；跑模型是显式的、带成本的例外。
3. **真值只用来评分与诊断，绝不进入选 K 的输入**（`oracle` 泳道必须在界面上标明"读真值"）。
4. **一屏回答一个问题**：这场、这几条配置、差在哪 —— 不做历史、不做对比存档。
5. **不美化结论**：旋钮调出来的好数字与生产口径不同时必须当场区分（如 `unlocked` 尚未进生产）。

## Accessibility & Inclusion

无产品特定的无障碍要求（内部工具、单人使用）；但保留键盘可用与足够对比度，便于阅读密集数字。
