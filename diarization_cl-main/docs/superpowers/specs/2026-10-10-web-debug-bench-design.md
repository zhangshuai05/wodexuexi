# AISHELL-4 分离调试台（`web/`）——设计

日期：2026-10-10　路径：architectural（brainstorming）／new surface extension（impeccable）
前置文档：`PRODUCT.md`（产品事实）、本文件（实现设计）、surface brief（方向契约，见 §9）

## 1. 目标与成功判据

把"选 K"决策链做成**本地单页调试台**：在一场 AISHELL-4 真实会议上并排比较 4 条泳道，**秒级**看到真值对照、句级四类分解与**真 DER**，拧旋钮即时更新。

成功（第一轮完成定义）：

1. `verify_der.py` 通过：dev 12 场 × `base`/`fixed` 两臂与 `artifacts/v13_e2e_unlocked2_dev_mw2_objective.json` 逐场对上（判定见 §6.3）。
2. 同一场给出句级 + DER：`oracle` 关闭时 **< 5 s**（2 条泳道计算 + merged 列），开启时 < 90 s。
3. `base` 钉住状态下改旋钮，base 读数**逐位不变**。
4. 不改 `streaming_asr/`，`154 passed` 不回归。
5. `web/README.md` 写明口径、缓存覆盖（50 场，其中 38 场无门限参照）、DER 对拍状态与日期。

## 2. 范围

**做**：`web/` 一个新目录；缓存驱动的 4 泳道对比；旋钮；真值对照；句级 + DER;音频播放定位。

**不做（第一轮）**：上传任意音频走真模型（第二轮，接口留位）、8735 合并、任何历史/存档/多用户/鉴权、前端框架与构建链、动效与装饰。

**不动**：`streaming_asr/diar_web.html` 与 `web_server.py`（8735）、`not_mind/shiyan/fe/`、`milestones/` `archive/` `backups/` 里的前端副本、`streaming_asr/bench/`（功能搬完并确认无缺失后再走回收站删除）。

## 3. 目录与文件

```
web/
├── README.md            跑法 / 口径 / 已知限制 / DER 对拍状态
├── server.py            8737：静态文件 + /api/*（ndjson 流式）
├── index.html           单页骨架
├── app.js               原生 JS：渲染、旋钮、播放头、日志
├── style.css            暗底 + 等宽数字；无动效/渐变/装饰
└── backend/
    ├── __init__.py
    ├── cache_path.py    ★ 缓存嵌入 → 生产同一份段表（含 k<2 救援）
    ├── lanes.py         4 条泳道 + 旋钮 → labels / segments / level
    ├── metrics.py       句级四类、clean·overlap 分桶、DER、merged 复算
    └── verify_der.py    ★ 门禁：dev 12 场 × 2 臂对拍
```

`server.py` 复用 8736 的做法：`ThreadingHTTPServer`、`allow_reuse_address = False`、ndjson 分块流、只绑 `127.0.0.1`、静态文件 `Cache-Control: no-store`。

## 4. 后端

### 4.1 `cache_path.py`——缓存 → 生产段表（**本任务的核心风险**）

现状：探针链（`probe_ab_dev.real_path_labels` + `probe_sentence_attribution.segment_map`）与生产 `v12.diarize` 有**两处已知差异**（`probe_path_equiv.py` docstring 记录）：

1. `k < 2` 时探针**不救援**，生产会重提 0.6 s 密窗、重跑候选、分出多人就**整体替换**；
2. 探针按"窗起点 + 窗长"平铺，生产用 `v12._segment_windows`（边界落在相邻窗重叠中点 + `hop/2`）——**段级几乎 100% 不同**，只有句级恰好一致，所以**探针路径不能算 DER**（坑 3）。

本模块把这两处**换成生产同一份代码**：

```
load cache(main) → _candidate_clusters/lanes 选 K → 
  k>=2:  rlabels = v12._refine_labels(cache(refine))
  k==1:  救援分支：v11._extract_windows 等价物改用 cache(rescue) → _candidate_clusters → 替换
→ centroids/margins（余弦 margin，与 diarize 同式）
→ v12._segment_windows(times, labels, margins, hop_s)
```

**不变式**：本模块只读缓存 + 调生产函数，**不复制**生产的逻辑分支。若某段逻辑必须复制（如 `_v11_window_params` 的窗口常量），用同一常量来源并在 README 记名。

**待实现时确认的一处**：已存真模型产物用的是**哪个段表口径**（`diarize` 的 `segments` 过滤前 vs `run_claude_lane` 过滤后），对拍以此为准（见 §6.3）。

### 4.2 `metrics.py`

- 句级：直接调 `probe_sentence_attribution.attribution`（复用，不重写）→ overall / clean / overlap 三栏 + 弃权 / 未映射 / 错答（`n_wrong / n`，精确计数）。
- DER：调 `der.der(reference, predicted, collar=0.25)`，返回 der 与 miss/FA/confusion 分解。
- `merged`：对当前选中的 k 调 `probe_oracle_k_curve.merge_dups`（先 `classify_dup`），并回主簇后**重走 §4.1 的段表**再算一次句级与 DER ⇒ 得到"消重后"列。

### 4.3 `lanes.py`——3 条泳道 + 1 个附加列

| 泳道 | 选 K | 说明 |
|---|---|---|
| `base` | `v11._candidate_clusters` + `v11._select_candidate`，常量取**生产值** | 钉住时不受旋钮影响 |
| `unlocked` | `unlocked_selector.build_unlocked(cap, min_win)` + `pick_unlocked(mode)` | 受全部旋钮影响 |
| `oracle` | k=2..10 网格，取 `bijective` 峰值（**读真值**） | 默认关，≈1 分钟/场；关时时间轴不画这条 |

`merged`（消重后读数）是**每条泳道的列**（§4.2），不占一条独立泳道——省一次细化，读数照样能看出"高分是不是靠多切同一人垫的"。时间轴上不叠 merged 段。

旋钮：`lam`(0–0.03) · `min_win`(2/3/4) · `cap`(4–10) · `mode`(objective/largest_eligible) · `refine`(开/关) · `oracle`(开/关) · `pin_base`(默认开)。

`lam` 通过临时改 `v11.MULTI_K_PENALTY` 生效（与 `probe_unlocked_sentence.candidates_for` 同一手法），**用完必还原**；`pin_base` 为真时 base 分支不受 `lam` 影响。

### 4.4 `verify_der.py`——门禁

对 dev 12 场跑两条臂（`base` = 生产常量；`fixed` = unlocked + `min_win=2` + 生产 λ），与 `v13_e2e_unlocked2_dev_mw2_objective.json` 的逐场 DER 比对，写 `artifacts/web_der_parity.json`。

**判定**：逐场 `|ours - ref| < 0.05` 个百分点记 pass，同时记录是否完全相等；24 项全 pass ⇒ `status="pass"`，否则 `status="fail"` 并列出未通过场次。**不允许**把 `status="fail"` 的缓存 DER 当结论展示。

```json
{"schema":"web-der-parity-v1","checked_at":"...","threshold_pts":0.05,
 "rows":[{"meeting":"...","arm":"base","ours":0.2354,"ref":0.2354,"delta_pts":0.0,"pass":true},...],
 "summary":{"n":24,"pass":24,"max_delta_pts":0.0,"status":"pass"}}
```

**页面必须读这个文件**：`status != "pass"` 时 DER 列显示 `—（未对拍）`，顶部状态条红字。

## 5. API 契约

| 端点 | 参数 | 返回 |
|---|---|---|
| `GET /api/meetings` | — | `{meetings:[{id,cohort,has_ref,gt_speakers,duration,n_windows}]}`（`has_ref` = 该场在 `v13_ab_dev_results.json` 参照里） |
| `GET /api/run` | `meeting, lam, min_win, cap, mode, refine, oracle, pin_base` | ndjson：每完成一条泳道推 `{"t":"lane","lane":"base","metrics":{...},"segments":[...]}`；结束推 `{"t":"done","parity":{...}}`；错误推 `{"t":"error",...}` |
| `GET /api/truth` | `meeting` | `{reference:[{start,end,speaker}], gt_k, has_ref}` —— 时间轴的真值轨与映射表的分母（**补于 Ruling 12**：§6.1 要求真值轨与映射表，原 §5 契约漏了这一路） |
| `GET /api/audio` | `meeting, seconds` | `audio/wav`（ffmpeg `pan=mono|c0=c0`，缓存到 `web/clips/`） |
| `POST /api/verify` | — | ndjson 进度；结束推 parity summary |

⚠️ **Ruling 12（2026-10-10 补）**：`lane` 事件必须带 `pairs`（映射表行：`pred`/`ref`/`overlap`/`share`/`pred_seconds`，实现照 `streaming_asr/bench/bench_server.py` 的 `pairs()`），且页面可选 `GET /api/truth?meeting=` 取真值轨；键名集合见 `contract.LANE_EVENT_KEYS = LANE_PAYLOAD_KEYS + ("pairs",)`。原因：§6.1 要求真值轨与映射表，而本节最初漏了这一路数据来源。

错误处理：`meeting` 必须命中 `meetings()` 白名单；`seconds` 钳制到 `[30, 3600]`；任何异常推 `{"t":"error"}` 并保留已完成泳道（**不写终止块之后的数据**——8736 踩过这个坑）。

## 6. 前端

### 6.1 布局（一屏）

```
状态条：DER 对拍 12/12 ✓ ｜ base 已钉住（生产口径） ｜ 旋钮非生产值 ⚠
左 260px：50 场列表（cohort 徽标 / 有真值参照 / 真值人数 / 时长）
旋钮行：λ ──●── 0.005  min_win[2] cap[10] mode[objective] 细化☑ oracle☐
指标表：行 = 3 泳道；列 = k · 真K/可见K · overall·clean·overlap · 弃权·未映射·错答 · merged · DER(miss/FA/conf) · level
时间轴：真值 1 条 + 泳道（base/unlocked 必出，oracle 开启时多一条），同一把尺子；播放头同步音频；点击定位
右 320px：映射表（重叠秒/纯度/簇时长）+ 进度日志
```

### 6.2 视觉（继承 8736 的世界）

暗底（`#0e1116` 系）、面板 `#11161e`、等宽数字（`ui-monospace`）、1px 细线分隔；颜色**只**用于区分说话人（真值一套、预测一套，色彩取自现有 `colorFor` 的调色板）；无动效、无渐变、无阴影装饰；`oracle` 泳道在表头与时间轴上标注"读真值"。

### 6.3 DER 的显示规则

判定在 §4.4。页面只读 `artifacts/web_der_parity.json`：`status == "pass"` 才渲染 DER 数字，并在状态条显示对拍日期与 `12/12`；否则 DER 列显示 `—（未对拍）`、状态条红字、并在日志里列出未通过场次。

## 7. 验收与测试

- `web/backend/selfcheck`（`python -m backend.verify_der --selfcheck`）：不依赖网络的合成用例——① 句级四类加总 = 1；② `merged` 在合成重复簇上把分数拉回；③ 空候选不抛异常。
- 对拍（§6.3）作为集成验收。
- `streaming_asr` 的 `154 passed` 必须仍通过（本设计不改它）。

## 8. 风险与回退

| 风险 | 回退 |
|---|---|
| 缓存段表对不上（`_segment_windows` / 救援分支差异定位超时） | DER 改走"真模型重跑"按钮（20 分钟/场），其余功能照做；对拍脚本保留，标 `fail` |
| `oracle` 泳道太慢 | 降级为 k 当前值 ±2 的局部网格（信息少但 < 15 s） |
| `TextGrid` 真值在其余 38 场无门限参照 | 界面标"无参照"，README 写清；不声称这 38 场经过验证 |
| 8736 功能搬漏 | 搬迁前逐项对照 8736 的 5 项能力清单（§7 验收），确认后再删旧目录 |

## 9. 交付前必须存在

- `PRODUCT.md`（已写）
- 本 spec
- surface brief（方向契约：THESIS / OWN-WORLD / STORY / FIRST VIEWPORT / FORM / FINISH）——通过 `impeccable surface-brief write` 记录
- `.impeccable/config.json` 的 `buildPath = "code"`（已写）
