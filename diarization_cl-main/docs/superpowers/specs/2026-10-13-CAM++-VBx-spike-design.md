# CAM++ 上的真 VBx spike —— 设计（2026-10-13）

> 本文是 `docs/superpowers/specs/2026-10-10-说话人识别目标规范-design.md` **§4.4 的执行设计**。
> 判据口径由本项目负责人于制定本文件时逐条裁决（§3）。**判据一经写定不得回改**；
> 改了这份 spike 就作废，只能重开一份新的。

---

## 0. 一句话

固定 CAM++ 192 维嵌入，用**真正的 VBx**（PLDA + 贝叶斯 HMM，从零 EM）替换
「AHC + `silhouette − λ(k−2)`」，在 dev 12 场上测句级归属错误率能否**两个门都降 ≥ 1 点**。

---

## 1. 为什么"还没测过"

目标规范 §4.3 把 2026-09-22 那次 spike 记成「AHC 初始化 + VB HMM 93.18% ⇒ VBx 在
CAM++ 上不行」。**那个数字是无效的**，本次设计前核实：

| 事实 | 位置 |
|---|---|
| 该臂的输入是 `192 维 CAM++` 乘**固定随机正交基**升到 256 维 | `D:\yinp\AISHELL-4\_v13_spike\compare_backends.py:43-57`（`project_to_vbx_space`，注释自述 "uncalibrated"）|
| 之后走的是 ResNet101 的 `transform.h5` + `plda` | 同文件 `:150-151` |
| 所以 93.18% vs 93.06% 是「失效变换 vs 什么都不做」 | —— 信息量为零 |

并且流传的机制解释**与源码相反**：`VB_diarization.py:34` 的 `maxSpeakers` 打开的是
`maxSpeakers` 个**空簇**，先验加上不切换的 HMM 允许休眠说话人复活 ——
「只能精修、不能恢复被 AHC 合并的说话人」描述的是**旧代码实际怎么用**（初始化钉死），
不是 VBx 的能力上限。本次的**臂 2** 专门验这一条。

⇒ 本轮结论：**「VBx 在 CAM++ 上有没有用」至今未知**，值得测，且这是 spec §4.5 的第 2 步。

---

## 2. 实验设计

### 2.1 数据划分（不可变）

| 用途 | 场次 | 场数 |
|---|---|---|
| **PLDA 训练池** | `train_L` 30 场（`20200706_L_R001S01C01` …）| 30 |
| **评测** | dev 12 场（`v12_aishell4.split_meetings(root, "dev")`）| 12 |
| **禁止读** | holdout 8 场 | 0 |

**dev 的嵌入不参与任何拟合**。这是与已失败的 LDA sweep 最大的口径差别：
那次只有 dev 12 场 / 16 个类，PLDA 必然拟合不足。

### 2.2 ★ 类 = `(场次, 标签)`，不是裸标签

`train_L` 的说话人标签只有 9 个 —— `001-M 002-M 003-M 004-M 005-F 006-F 007-M 007-F`
（外加 `005-F` 缺 3 场、`004-M` 缺 10 场、`007-*` 缺 14–16 场），
且**每个标签在全部 30 场里都出现**。

⇒ 裸标签建类 = 把 30 个不同会议里的不同人当成同一个人，类内方差被抹平、
类间方差被夸大，PLDA 直接废。

**规定：类标识 = `(场次, 标签)`**，实测 207 个类。此条必须有红验证测试（§6）。

### 2.3 门（两个，分开报，不混用）

| 门 | 窗口 | dev 基线（生产选择器） | 诊断上界 |
|---|---|---:|---:|
| **RTTM** | `cache/<场>/w2_h1.npz`（562 窗）+ `rttm_speech(skip=10)` | **9.15%** | 4.26% |
| **VAD** | `cache/probe/<场>/main.npz`（422 窗）+ manifest 记的真 FSMN 区间 | **7.19%** | 4.05% |

⚠️ `probe_gap_decomp.rttm_speech` 的 `skip` **默认 10.0 秒**，而 eval 侧现有全部调用点
都吃默认值。**两侧都必须用默认值**，显式写 `skip=10.0` 以防有人改默认。

⚠️ `w2_h1.npz` 建库时已按 `rttm_speech` 门控过（`build_cache.extract_meeting`），
再套一次 `gate_windows` 是幂等的 —— 与 `probe_backend_sweep` 同构。

### 2.4 三臂（全部零嵌入成本）

| 臂 | 决策规则 | 回答什么 |
|---|---|---|
| **0 基线** | `build_unlocked(emb, CAP=10, MIN_WIN)` + `pick_unlocked`（生产 objective）+ `refine_labels` | 先复现 9.15% / 7.19%。**复现不了就停**，后面的数字没有意义 |
| **1 VBx-AHC** | `gamma` = `softmax(onehot(该场选中候选的 AHC 标签) × 5)`，`pi=None`，`maxSpeakers` = 该候选 K | 同一个候选划分上只换打分与归属：VB 能否把 AHC 的错吸掉 |
| **2 VBx-flat** | `gamma` = **自己用固定种子生成**的 Dirichlet 初值（`np.random.default_rng(SEED).gamma(1.0, size=(T, CAP))` 后归一化），`pi=None`，`maxSpeakers = CAP = 10` | **spec §4.4 要的那一臂**：能否**恢复被 AHC 合并掉的说话人** |

⚠️ **臂 2 的 `gamma` 必须由调用方给，不能传 `None`。** `VB_diarization` 内部在
`gamma is None` 时走 `np.random.gamma(...)`（无种子），那样臂 2 不可复现，
§6 测试 3 就无从验证。**`SEED` 常量写进脚本并在产物里记录。**

**三臂共用同一份细化与句级指标**（`BS.refine_labels` / `BS.sentence_metrics`），
臂间唯一差别是喂给细化的标签来源 —— 这是坑 A7（两臂不同构）的防线。

### 2.5 训练池的窗与纯度口径

与 eval 侧**同法**取纯窗，避免另一处不同构：

```
train_L 场次
  → w2_h1 窗（2.0s / 1.0s，全体时间点）
  → rttm_speech(gts, skip=10.0) 门控          ← 与 eval 侧同一个 skip
  → speakers = PG.window_reference(gts, times, 2.0)   ← 逐窗多数说话人（按帧计票）
  → votes    = 多数说话人的时长占比 / 该窗 RTTM 语音时长
  → pure_windows(emb, times, speakers, votes)         ← 复用 probe_pair_oracle
  → 类 = (场次, speakers)
```

`votes` 的定义与 `_v13_spike/build_cache.py:vote_split` 逐字相同（该口径是
`w2_h1.npz` 里 `votes` 字段的来源）。`PURE_VOTES` 用 `probe_pair_oracle` 的既有值。

train_L 的 `w2_h1.npz` **不存在，本轮构建**（30 场 × 约 22–37 分钟音频）。
复用 `_v13_spike/build_cache.py` 的提取循环、**不重写**，产物落到 `_v13_spike/cache/`，
与 test 侧同目录同命名。

### 2.6 PLDA 与 LDA 变换（自训，不用 ResNet101 的资产）

官方 `vbhmm.py` 在 `lda_dim=128` 上截断，但 CAM++ 只有 192 维、训练类只有 207 个，
128 维的上限没有依据。**取 `R = min(32, 类数 − 1)` 并在产物里记录实际值**。

按 `diarization_lib.kaldi_ivector_plda_scoring_dense` 的方式反推，只需要两个量：

```
均值 μ
W⁻¹B 的特征分解  →  wccn（白化，类内协方差 → I）
                    acvar（类间特征值，降序）

fea   = (x − μ) @ wccn[:, :R]                    ← LDA 空间，类内协方差 ≈ I
sV    = sqrt(acvar[:R])                          ← eigenvoice 先验
sm    = 0,  iE = I,  V = diag(sV)                ← VB_diarization 的输入
```

**两协方差 PLDA 的矩估计**（numpy/scipy，无新依赖）：

```
类内散布  Wc = Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ / N
类间散布  Bc = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ / N
正则化    收缩到对角：Wc ← (1−α)·Wc + α·diag(Wc) + ε·I
```

`α`、`ε`、`R` 三个量由 `--plda-shrink` / `--plda-eps` / `--lda-dim` 给出，
**不得按 dev 结果调**（见 §3 反过拟合条款）。`Fb` 是唯一允许扫的参数（§3）。

### 2.7 `Fb` 曲线（唯一允许扫的参数）

官方默认 `Fb=17`（`run_vbx_baseline.py:110`）。`Fb` 直接控制最终说话人数，是 VBx
最敏感的超参。**扫一条曲线全部报出**，同时**固定 `Fb=17` 作为判据臂**：

```
Fb ∈ {1, 5, 10, 17, 30, 60}
```

其余超参固定为官方默认：`Fa=0.3`、`loopProb=0.99`、`maxIters=40`、`epsilon=1e-6`、
`threshold=-0.015`（仅臂 0/1 的 AHC 用）、`init_smoothing=5.0`、`minDur=1`。

---

## 3. 预注册判据（不可回改）

### 3.1 主判据 —— 双门都要 ≥ 1 点

`Fb=17` 的 **VBx-flat（臂 2）** 相对**生产选择器基线**：

| 门 | 基线 | 通过线 |
|---|---:|---:|
| RTTM | 9.150% | **≤ 8.150%** |
| VAD | 7.190% | **≤ 6.190%** |

**两个门都必须通过才算成立。** 只要一个门不到，结论记为**证伪**。

理由：只卡 RTTM 门拿不到生产口径的结论；只卡 VAD 门则 7.19% 已在 spec §2.2 的
≤8% 预算内，"已达标"会让判据在实验开始前就自我满足。双门同时要求是唯一能
把「VBx 有用」与「什么都没做」分开，又不放弃生产口径的严格线。

### 3.2 允许的失败原因 / 不允许的失败原因

预注册，防止事后解释：

| 结论 | 允许？ | 条件 |
|---|---|---|
| 实验设计缺陷（泄漏 / 两臂不同构 / 静默丢场 / 输入未标定）| ✅ | 修好重跑，旧数字作废并标注 |
| PLDA 拟合不足（类数 < 2、`Wc` 奇异、`acvar ≤ 0`）| ✅ | 必须**落盘失败原因**（存根行），不得静默跳过 |
| VBx 跑通、双门未达 ≥ 1 点 | ❌ 不许再解释 | **这就是证伪**，转 spec §4.4 的 (1)：对齐文献口径 |

⚠️ 本线已有一次同类事故：LDA sweep 因 `SLIVER_MARGIN` 尺度混淆与训练池缺陷收场，
却被记成"方向被否"。**本次禁止重演** —— 所以 `Fb` 扫、存根行、两臂同构三件事
都是**判据的一部分**，不是可选优化。

### 3.3 反过拟合条款

1. **dev 只用于评测**，不得用于选 `α` / `ε` / `R` / `maxSpeakers`。
2. `Fb` 允许扫，但**判据只认 `Fb=17`**；曲线全部报出，**不得事后挑最好的 Fb 当结论**。
3. **禁止读 holdout**（spec §3.3）。
4. 上界对照（4.26% / 4.05%）是**读真值的诊断数字**，只作参照，**不是可达目标**。

### 3.4 次要观察（不作判据，只记录）

- **逐场底线**（spec §2.4：任一场 clean ≥ 80%）。现最差 `M_R003S04C01` = 72.2%。
- VBx 自己估出的说话人数（`sum(pi > 0)`）vs 真值有效人数 —— **只作诊断**，
  spec §6 已定"不追求预测人数等于真值"。

---

## 4. 组件与边界

```
streaming_asr/vbx/
  VB_diarization.py      官方 BUT 算法体逐行复制
  _numexpr.py            兼容层：logsumexp / exp 包成 numpy（见下）
  plda.py                两协方差 PLDA 矩估计 + LDA 变换拟合（自写）
scripts/evaluation/
  run_campplus_vbx.py    唯一入口：建池 → 拟合 → 三臂 × 门 → 评测 → 落产物
  build_train_campplus_cache.py
                         建 train_L 的 w2_h1.npz（复用 build_cache 的提取循环）
streaming_asr/tests/
  test_vbx_plda.py       红验证过的单元测试
```

**`VB_diarization.py` 的改动清单（只有 import 一行，算法体零改动）**

已核实：该文件用到 `kaldi_io` 的只有 `from kaldi_io import open_or_fd` 这一行，
而 `open_or_fd` 在文件内**从未被调用**；`numexpr` 只被两个 helper
（`logsumexp_ne` / `exp_ne`）使用，`forward_backward` 实际调的是**纯 numpy 的
`logsumexp`**（`:287`、`:290`）。

| 改动 | 内容 |
|---|---|
| `from kaldi_io import open_or_fd` | **删除**（死 import）|
| `import numexpr as ne` | 改成 `from . import _numexpr as ne`；`_numexpr` 里 `logsumexp`/`exp` 直接调 numpy |

`tqdm` 已装（4.68.4），不改。`scipy`/`numpy` 已有。⇒ **不引入任何新依赖。**

**边界（硬约束）**

| 项 | 规定 |
|---|---|
| 生产代码 | **一行不改**。`claude_a2_v11/v12`、`unlocked_selector`、`web/` 全部只读 |
| 新依赖 | **不引入**。只用 numpy / scipy / tqdm / soundfile / funasr / torch |
| holdout | **不读**（不加载其嵌入、不跑其评测、产物里不含其行）|
| `transform.h5` / `plda`（ResNet101）| **不用**。它们对 CAM++ 无效，正是上次失败的根因 |
| 既有产物 | 只读。新产物一律新文件名 |
| 真值进入选 K | **禁止**。VBx 的输入只有嵌入与超参，不接触 RTTM/TextGrid |

**允许复用的既有工具**（复用即防不同构，不得重写）：
`probe_gap_decomp.{rttm_speech, cached_windows, gate_windows, window_reference}`、
`probe_pair_oracle.{pure_windows, l2norm}`、
`probe_backend_sweep.{refine_labels, sentence_metrics}`、
`unlocked_selector.{build_unlocked, pick_unlocked, CAP, MIN_WIN}`、
`v12_aishell4.split_meetings`、`der.read_rttm`。

---

## 5. 数据流

```
                    ┌─ train_L 30 场 ─────────────────────────────┐
                    │ flac + rttm → 2.0s/1.0s 窗 + CAM++ 192 维   │
                    │ → rttm_speech 门 → 多数说话人 + votes        │
                    │ → pure_windows → 类 = (场次, 标签) 207 个    │
                    └───────────────┬─────────────────────────────┘
                                    ▼
                        PLDA 矩估计 → μ / wccn / acvar
                                    │  （拟合一次，三臂共用）
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
   RTTM 门 dev 12 场                                     VAD 门 dev 12 场
        │                                                       │
        ├─ 臂 0: AHC + pick_unlocked ──────────┐                 │
        ├─ 臂 1: AHC 标签 → γ₀ → VB            ├→ refine_labels │
        └─ 臂 2: flat γ → VB ─────────────────┘   → sentence_metrics
                                                       │
                                            逐场四类 + 门级均值
                                                       ▼
                                    artifacts/v13_campplus_vbx_dev.json
```

---

## 6. 测试策略（每条断言必须红验证）

> 本仓库的既有纪律：**每写一条断言就问「哪个合理回归能让它保持绿」，答不出来
> 就是没覆盖**；验证方法是真的改源码让它变红，把命令与输出写进报告。（坑 A5）

| # | 断言 | 防的是什么回归 | 红验证怎么做 |
|---|---|---|---|
| 1 | 训练类的数量 = Σ 每场标签数（207），**不是**裸标签并集（9）| 有人把类退回成裸标签（§2.2）| 把 `meeting` 从类标识里删掉 → 计数变 9 → 断言红 |
| 2 | `VB_diarization` 返回的硬标签 ≠ 输入的 AHC 标签，且前者是后者的函数 | 有人把 `labels_ahc` 直接当输出（臂 1 退化成臂 0）| 让函数直接 `return labels_ahc` → 断言红 |
| 3 | 同一个 `Fb` 下 `run_vbx` 两次结果**逐位相同**（`gamma` 由固定 `SEED` 生成，§2.4）| 随机初始化导致跑两次两个数 | 改成直接传 `gamma=None` → 两次不等 → 断言红 |
| 4 | 三臂在同一场同一门上窗数**逐位相同** | 某臂悄悄换了窗集（坑 A7）| 给某臂的 `gate_windows` 传 `win=2.0` 改成别的 → 断言红 |
| 5 | PLDA 的 `acvar` 全为正、`Wc` 可逆 | 正则化去掉后奇异仍"跑通" | 把 `α` 设 0、`ε` 设 0 → 断言红 |
| 6 | `fea.shape[1] == R`，且 `len(sV) == R` | LDA 截断与先验维度脱节 | 只截断 `fea` 不截断 `acvar` → 断言红 |
| 7 | 产物里**没有** holdout 场次行 | 有人把 split 参数传错 | 把 `--split holdout` 传进去 → 断言红（脚本应拒绝）|
| 8 | 某场 VBx 抛异常时产物含**带 `skip` 理由的存根行** | 静默丢场（坑 A6）| 让某场必定抛 → 无存根行 → 断言红 |

数据/模型缺失一律 `pytest.skip` 而非 error（坑 W3），`streaming_asr/` 下跑
（坑 W5），中文输出带 `PYTHONIOENCODING=utf-8`（坑 W6）。

---

## 7. 产物 schema

`artifacts/v13_campplus_vbx_dev.json`

```json
{
  "schema": "v13-campplus-vbx-dev-v1",
  "generated_by": "scripts/evaluation/run_campplus_vbx.py",
  "code": {"head": "<git rev-parse --short HEAD>", "dirty": true},
  "truth_used": false,
  "plda": {"pool": "train_L", "n_meetings": 30, "n_classes": 207,
           "n_windows": 0, "R": 0, "alpha": 0.0, "eps": 0.0,
           "acvar_min": 0.0, "acvar_max": 0.0},
  "hyper": {"Fa": 0.3, "loopProb": 0.99, "maxIters": 40, "epsilon": 1e-06,
            "init_smoothing": 5.0, "threshold": -0.015, "caps": [10],
            "fb_grid": [1, 5, 10, 17, 30, 60], "fb_criterion": 17},
  "gates": {
    "rttm": {
      "baseline_reproduced": {"mean_attribution_error": 0.0, "n_meetings": 12},
      "arms": {
        "vbx_ahc":   {"by_fb": {"17": {"mean_attribution_error": 0.0}},
                      "meetings": []},
        "vbx_flat":  {"by_fb": {"17": {"mean_attribution_error": 0.0}},
                      "meetings": []}
      },
      "ceiling_diagnostic": 0.0426
    },
    "vad": {}
  },
  "criterion": {
    "rttm_pass_line": 0.0815, "vad_pass_line": 0.0619,
    "rttm_pass": null, "vad_pass": null, "verdict": null
  },
  "skipped": []
}
```

**逐场行的字段**（与 `v13_candidate_score_table_dev.json` 对齐，便于并列）：
`meeting` / `gate` / `arm` / `fb` / `k_vbx`（VBx 估出的人数）/ `k_baseline`（生产选的 K）/
`true_k_effective` / `n_windows` / `per_sentence` / `wrong` / `unmapped` / `abstain` /
`attribution_error` / `floor_pass`（clean ≥ 80%）。

`verdict` 取值域：`"pass"` / `"falsified"` / `"invalid_design"`（后者必须附
`invalid_reason`，且只在 §3.2 允许的两类里）。

---

## 8. 明确不做

| 不做 | 理由 |
|---|---|
| 改生产选择器 / 接 VBx 进生产 | 本轮是 spike，判据过了才立项（spec §4.4）|
| 调 `λ` / `min_win` / `CAP` | spec §4.6：已定稿 |
| 重新抽嵌入（不重叠窗）| 基线用的是同一批窗嵌入，重抽会**买不到可比性**；重叠窗是合理近似（§2.3）|
| 读 holdout | §2.1 |
| 用 ResNet101 的 `plda` / `transform.h5` | 对 CAM++ 无效（§1）|
| 换 SAD / 分段 | spec §4.6 |
| 追 train_L 的交付数字 | spec §2.7：只作诊断 |

---

## 9. 成本与风险

| 项 | 估计 |
|---|---|
| 建 train_L 的 `w2_h1.npz` | 30 场 × 22–37 分钟音频；GPU 是 RTX 5060 8 GB（当前占用 7.8 GB）⇒ **需先确认可用显存**，否则退化为 CPU 长跑 |
| PLDA 拟合 | 秒级（207 类 × 数千窗）|
| VB 迭代 | 每场 × 每个 `Fb` × `maxIters=40`；562 帧 × 10 说话人量级 ⇒ 秒级 |
| 总评测 | 12 场 × 2 门 × 2 臂 × 6 个 `Fb` ≈ 288 次 VB ⇒ 分钟级 |

**主要风险（按严重度）**

1. **PLDA 欠拟合**：207 类、192 维，类均窗数不均（长会议多、短会议少）。
   缓解：收缩正则 + **落盘 `acvar` 分布**，让"是否拟合充分"可判而非可猜。
2. **train_L 域偏移**：只有 L 房间、全 C01、有效 K 中位 5 vs dev 3。
   这**不是**设计缺陷而是已知代价（与 spec §4.3c 的训练池收口同性质），
   必须在产物 `note` 里写明，且不得据此把阴性结果解释成"换域会更好"。
3. **显存不足**：建缓存可能跑不动。缓解：减小 batch / 分段缓存 / 断点续建。

---

## 10. 完成定义

以下全部为真，本 spike 才算做完：

1. `pytest streaming_asr/tests/test_vbx_plda.py` 全绿，**每条断言都有红验证记录**；
2. `streaming_asr/` 全量回归 **≥ 169 passed / 0 failed**（当前基线 169，2026-10-13 实测）；
3. `artifacts/v13_campplus_vbx_dev.json` 落盘，含 §7 全部字段、两门、三臂、`Fb` 曲线；
4. **臂 0 复现 9.15% / 7.19%**（不要求逐位相同，要求同口径末位一致 —— 不一致即停）；
5. `criterion.verdict` 有确定取值（`pass` / `falsified` / `invalid_design`）；
6. 交接文档更新：§G 之后新增一节，并把 spec §4.3 那句被推翻的表述原地标注；
7. 不提交、不暂存（除非负责人另行指示）。

---

**本设计由项目负责人逐条裁决（2026-10-13）：**
训练池 = **train_L 30 场**（dev 不参与拟合，holdout 不读）·
判据 = **双门（RTTM ≤8.15% 且 VAD ≤6.19%）** ·
`Fb` = **扫曲线全部报出，判据臂固定 17** ·
前端 = **复用现有重叠窗缓存，不重抽嵌入**。
