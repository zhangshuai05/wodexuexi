# AISHELL-4 分离调试台（`web/`）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 一个本地单页调试台：在一场 AISHELL-4 会议上并排比较 3 条选 K 泳道（+merged 列），秒级给出真值对照、句级四类分解与**经对拍验证的 DER**。

**Architecture:** 缓存驱动，不跑模型。`web/backend/cache_path.py` 把缓存嵌入喂进**生产同一份**段表代码（`v12._segment_windows` + `k<2` 救援），`metrics.py` 复用 `probe_sentence_attribution` 与 `der.py`，`lanes.py` 给 3 条泳道与旋钮，`verify_der.py` 用已存真模型产物做 12 场 × 2 臂对拍门禁；`server.py`（8737）只做校验与编排，页面是原生 HTML/CSS/JS。

**Tech Stack:** Python 3.12 标准库（`http.server`、`threading`、`json`、`subprocess`）+ 已有 numpy/scipy/soundfile；前端原生 HTML/CSS/JS，零构建、零依赖。

**Spec:** `docs/superpowers/specs/2026-10-10-web-debug-bench-design.md`（执行者两篇都要读；本计划从 spec 论证，spec 随行）

## Global Constraints

- **不改 `streaming_asr/`**（8735 与 `154 passed` 都不许动）；不新增第三方依赖。
- 端口 `127.0.0.1:8737`，只绑本地；`allow_reuse_address = False`。
- 对拍阈值：逐场 `|ours - ref| < 0.05` 个百分点；未通过不许显示 DER。
- 性能：`oracle` 关时 `/api/run` < 5 s；开时 < 90 s。
- 页面：暗底、等宽数字、**无动效/无渐变/阴影装饰**；颜色只区分说话人；`oracle` 必须标"读真值"。
- `λ` 生效方式=临时改 `v11.MULTI_K_PENALTY`，**必须还原**（`try/finally`）。
- 提交只 `git add <明确路径>`，**禁止 `git add -A`/`git add .`**（本项目踩过：索引里已有他线暂存文件）。
- 删除文件/目录走回收站：`[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($p,'OnlyErrorDialogs','SendToRecycleBin')`。
- 测试在 `streaming_asr/` 下跑的那条命令不得改动；本计划的测试在仓库根跑 `python -m pytest web/tests -q`。

## Review Focus

1. 缓存缺场次或缺某个阶段 npz → 必须显式报错，不得静默跳过或回退空结果（T1）。
2. `/api/run` 中途抛异常 → 只推 `{"t":"error"}`，不得在终止块之后继续写（T5）。
3. `meeting=../../x`、`seconds=inf/nan/负/超大` → 拒绝或钳制（T5、T7）。
4. 钉住 base 时改旋钮 → base 逐位不变；解钉后 base 跟随（T4、T5）。
5. `web_der_parity.json` 缺失/`status!=pass` → 页面不得显示 DER 数字（T3、T6）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `web/backend/__init__.py` | 空包标记（1 行） |
| `web/backend/cache_path.py` | 缓存嵌入 → 生产段表（含 k<2 救援）；**只读缓存 + 调生产函数** |
| `web/backend/metrics.py` | 句级四类/分桶、DER、merged 复算 |
| `web/backend/lanes.py` | 3 条泳道 + 旋钮参数化（λ/min_win/cap/mode/refine/pin_base） |
| `web/backend/contract.py` | 泳道 payload 的字段名单（页面契约的唯一来源，T2 建、T3/T6 用） |
| `web/backend/verify_der.py` | 对拍门禁 → `artifacts/web_der_parity.json` |
| `web/server.py` | 8737：静态文件 + 4 个 API + ndjson |
| `web/index.html` / `app.js` / `style.css` | 页面（暗底、表格、时间轴、旋钮、播放头） |
| `web/tests/*.py` | 各模块测试（pytest，根目录跑） |
| `web/README.md` | 跑法、口径、限制、对拍状态 |

依赖顺序：T1 → T2 → T4 → T3 → T5 → T6 → T7 → T8 → T9。

---

### Task 1: 缓存 → 生产段表（`cache_path.py`）

**Files:** Create `web/backend/__init__.py`、`web/backend/cache_path.py`、`web/tests/test_cache_path.py`

**Interfaces:**
- Consumes: 生产 `streaming_asr/claude_a2_v12.py` 的 `_segment_windows(times, labels, margins, hop_s)`、`REFINE_HOP_S`、`WIN_S`、`HOP_S`、`MAX_SPEAKERS`、`_refine_labels`；`claude_a2_v11.py` 的 `_candidate_clusters(E, max_speakers)`、`_select_candidate(cands)`、`_centroids(E, labels)`；`probe_path_equiv.cached_stage(meeting, stage)` 的缓存读取。
- Produces: `segments_for(meeting, labels=None, level=None) -> dict`，返回 `{"segments":[{"start","end","spk","margin"}], "level":"refined"|"2s-only"|"rescued", "times","labels","margins","k"}`；`rescue_labels(meeting) -> (labels, times) | None`；`cached_meeting(meeting) -> dict`（缺缓存时 `raise FileNotFoundError`，消息含缺失路径）；`_segments_from(times, labels, margins, hop_s) -> list[dict]`（T3 与测试都用）。

- [ ] **Step 1: 写失败测试**（`web/tests/test_cache_path.py`）
```python
def test_missing_cache_raises_with_path(tmp_path, monkeypatch):
    import cache_path
    monkeypatch.setattr(cache_path.PE, "PROBE_CACHE", tmp_path)
    with pytest.raises(FileNotFoundError) as e:
        cache_path.cached_meeting("NOT_A_MEETING")
    assert "NOT_A_MEETING" in str(e.value)

def test_segment_boundaries_match_production_formula():
    # 窗起点 [0.0, 1.0, 2.0]，hop=1.0，标签 [0,0,1]；按生产 _segment_windows 的边界公式
    # 段起点=相邻窗重叠中点+hop/2，段末=末窗起点+hop
    import cache_path
    out = cache_path._segments_from(np.array([0.,1.,2.]), np.array([0,0,1]), np.zeros(3), 1.0)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 2.0), (2.0, 3.0)]
```
- [ ] **Step 2: 跑测试，确认失败**（`python -m pytest web/tests/test_cache_path.py -q`，期望 `ModuleNotFoundError`）
- [ ] **Step 3: 实现**：`_segments_from` 直接 `return v12._segment_windows(times, labels, margins, hop_s)` 并把返回值转 dict（**不重写公式**）；`cached_meeting` 读三阶段 npz，任一缺失抛 `FileNotFoundError`；`segments_for` 走 spec §4.1 的流程（k≥2→`_refine_labels`；k==1→`rescue_labels`，注意生产在 `not refined and k==1` 时整体替换）。
- [ ] **Step 4: 跑测试，确认通过**
- [ ] **Step 5: Commit**（`git add web/backend/__init__.py web/backend/cache_path.py web/tests/test_cache_path.py`）

### Task 2: 指标（`metrics.py`）

**Files:** Create `web/backend/contract.py`、`web/backend/metrics.py`、`web/tests/test_metrics.py`

**Interfaces:**
- Consumes: `probe_sentence_attribution.attribution(sentences, pred)`、`.segment_map`、`.sentence_dict_predictions`；`der.der(reference, predicted, collar=0.25)`；`probe_oracle_k_curve.classify_dup/merge_dups`。
- Produces: `sentence_metrics(sentences, segments, level) -> dict`（键 = `contract.LANE_PAYLOAD_KEYS` 的指标部分）；`der_metrics(reference, segments) -> dict`（`der/miss_rate/false_alarm_rate/confusion_rate`）；`with_merged(meeting, segments, ...) -> dict`（同结构）。

- [ ] **Step 1: 写失败测试**
```python
def test_four_buckets_sum_to_one():
    # 造 4 句：一正确、一弃权、一未映射、一错答
    ...
    m = metrics.sentence_metrics(sents, segs, "2s-only")
    assert abs(m["correct"] + m["abstain"] + m["unmapped"] + m["wrong"] - 1.0) < 1e-9

def test_der_zero_when_prediction_equals_reference():
    assert metrics.der_metrics(ref, ref)["der"] == 0.0
```
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：句级**复用** `attribution`（不重写四类规则）；`wrong` 用 `n_wrong / n`；clean/overlap 取自 `attribution` 的同名桶；DER 传 `collar=0.25`。
- [ ] **Step 4: 跑测试，确认通过**
- [ ] **Step 5: Commit**

### Task 3: 泳道与旋钮（`lanes.py` + `contract.py`）

**Files:** Create `web/backend/lanes.py`、`web/backend/contract.py`、`web/tests/test_lanes.py`

**Interfaces:**
- Consumes: T1 的 `segments_for`/`cached_meeting`；T2 的 `sentence_metrics`/`der_metrics`/`with_merged`（**`with_merged(meeting, payload)` 第二参是 payload，不是 segments**）与 `contract.LANE_PAYLOAD_KEYS`/`TRUTH_DERIVED`；`unlocked_selector.build_unlocked/pick_unlocked`。
- Produces: `run_lanes(meeting, knobs: dict) -> Iterator[dict]`（每完成一条泳道 yield 一条 payload，键 = `contract.LANE_PAYLOAD_KEYS`）；`KNOB_DEFAULTS = {"lam": 0.005, "min_win": 2, "cap": 10, "mode": "objective", "refine": True, "oracle": False, "pin_base": True}`；`contract.LANE_PAYLOAD_KEYS: tuple[str, ...]`。

- [ ] **Step 1: 写失败测试**
```python
def test_pin_base_ignores_lam(monkeypatch):
    a = list(lanes.run_lanes(M, {**lanes.KNOB_DEFAULTS, "pin_base": True,  "lam": 0.005}))[0]
    b = list(lanes.run_lanes(M, {**lanes.KNOB_DEFAULTS, "pin_base": True,  "lam": 0.03}))[0]
    assert a["metrics"] == b["metrics"]          # 逐位相等

def test_lam_restored_after_run():
    import claude_a2_v11 as v11
    before = v11.MULTI_K_PENALTY
    list(lanes.run_lanes(M, {**lanes.KNOB_DEFAULTS, "lam": 0.03}))
    assert v11.MULTI_K_PENALTY == before

def test_unlocked_moves_with_lam_when_unpinned():
    ...
```
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：`run_lanes` 顺序 = base → unlocked →（可选）oracle；`lam` 用 `try/finally` 改 `v11.MULTI_K_PENALTY`；`pin_base` 真时 base 分支先存还原点再单独用生产值；`oracle` 复用 `probe_oracle_k_curve` 的网格逻辑但只 yield 峰值。
- [ ] **Step 4: 跑测试，确认通过**
- [ ] **Step 5: Commit**

### Task 4: DER 对拍门禁（`verify_der.py`）

**Files:** Create `web/backend/verify_der.py`、`web/tests/test_verify_der.py`

**Interfaces:**
- Consumes: T1/T3。
- Produces: `parity_report(ref_path=None, out_path=None) -> dict`（写 `artifacts/web_der_parity.json`，schema 见 spec §4.4）；`parity_status() -> dict`（读文件；缺失或 `status!="pass"` 时返回 `{"status":"missing"|"fail", ...}`）。

- [ ] **Step 1: 写失败测试**（用临时目录造假参照与假结果，断言阈值判定与 `status` 取值）
```python
def test_threshold_and_status(tmp_path):
    rep = verify_der.compare(fake_rows(delta_pts=[0.0, 0.04, 0.05]), threshold_pts=0.05)
    assert rep["summary"]["status"] == "fail"      # 0.05 不算通过（严格小于）
def test_missing_file_reports_missing(tmp_path):
    assert verify_der.parity_status(tmp_path / "nope.json")["status"] == "missing"
```
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：`compare(rows, threshold_pts)` 纯函数；`parity_report` 跑 dev 12 场 × 2 臂（`base` = 生产常量、`fixed` = unlocked+min_win=2+生产 λ），**先确认参照产物用的段表口径**（`diarize` 的 `segments` 过滤前 / `run_claude_lane` 过滤后），并把这个口径写进报告字段 `segment_convention`。
- [ ] **Step 4: 跑测试通过**，再跑一次真实对拍（分钟级）：`python -m web.backend.verify_der`，期望 `status == "pass"`；不通过就把差异逐场列出（这是 spec §8 的回退触发点）。
- [ ] **Step 5: Commit**

### Task 5: 服务（`server.py`）

**Files:** Create `web/server.py`、`web/tests/test_server.py`

**Interfaces:**
- Consumes: T3 `run_lanes`、T4 `parity_status`、T1 `cached_meeting`。
- Produces: HTTP 端点 `/`, `/api/meetings`, `/api/run`, `/api/audio`, `/api/verify`（契约见 spec §5）。

- [ ] **Step 1: 写失败测试**（起真实 server 线程，`http.client` 打）
```python
def test_meeting_whitelist(server):
    assert status(server, "/api/run?meeting=../x") == 400
def test_seconds_clamped(server):
    assert json_of(server, "/api/audio?meeting=<合法>&seconds=1e9")["clamped_seconds"] == 3600
def test_error_path_emits_error_then_terminator(server, monkeypatch): ...
```
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：白名单用 `/api/meetings` 的 id 集合；`seconds` 钳到 `[30,3600]` 且拒绝 `nan/inf`；ndjson 用 chunked；**响应结束即置位 `_closed`，`_emit` 见到就静默返回**（8736 的坑：终止块之后仍写 socket）；异常推 `{"t":"error"}`。
- [ ] **Step 4: 跑测试通过**；再手起一次 `python web/server.py`，浏览器打开确认页面可加载。
- [ ] **Step 5: Commit**

### Task 6: 页面（`index.html` / `app.js` / `style.css`）

**Files:** Create `web/index.html`、`web/app.js`、`web/style.css`、`web/tests/test_page_contract.py`

**Interfaces:**
- Consumes: T5 的 API；`contract.LANE_PAYLOAD_KEYS`。
- Produces: 页面读取的字段名与后端一致（测试锁死）。

- [ ] **Step 1: 写失败测试**（读 `web/app.js` 里的 `LANE_PAYLOAD_KEYS` 字面量，与 `backend/contract.py` 比对；再断言 `status!="pass"` 时 app.js 不渲染 DER 数字的那段逻辑存在，用 `parity` 关键字与 `—（未对拍）` 字面量锚定）
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：按 spec §6.1 布局（**`merged` 列与 `oracle` 列都必须带"读真值"徽标**，判定读 `contract.TRUTH_DERIVED`，不许写死在页面里）（状态条 / 左 260 场次 / 旋钮行 / 指标表 3 行 / 时间轴真值+N 泳道 / 右 320 映射表+日志）；时间轴复用 8736 的"两列网格 + 绝对定位块 + 点击定位"做法；播放头 `left` 用**轨道自己的 rect** 换算（8736 有 72px 错位的老 bug）。
- [ ] **Step 4: 跑测试通过 + 手测清单**：① 选场次出结果 ② 拧 λ 看 unlocked 变、base 不变 ③ 解钉后 base 变 ④ oracle 勾选有进度 ⑤ 对拍状态条正确 ⑥ 音频点击定位准。
- [ ] **Step 5: Commit**

### Task 7: 音频切片（`/api/audio`）

**Files:** Modify `web/server.py`；Create `web/tests/test_audio.py`

**Interfaces:** Produces `clip_path(meeting, seconds) -> Path`（缓存到 `web/clips/`；`*.wav` 在 .gitignore 内）。

- [ ] **Step 1: 写失败测试**（`clip_path` 产物用 `soundfile.info` 断言 `channels==1`、`samplerate==16000`、`duration≈seconds±0.2`）
- [ ] **Step 2: 跑测试，确认失败**
- [ ] **Step 3: 实现**：`ffmpeg -y -t <seconds> -i <DATA>/wav/<m>.flac -af pan=mono|c0=c0 -ar 16000 <tmp>` → `os.replace`（复用 8736 的 `c0` 取法：**不是** `-ac 1`）；失败清理 `.tmp.wav`。
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit**

### Task 8: README 与旧目录处置

**Files:** Create `web/README.md`

- [ ] **Step 1: 写 README**：跑法（`python web/server.py` → 8737）、口径（缓存/真模型、overall/clean/overlap）、限制（50 场，其中 38 场无门限参照）、对拍状态与日期、旋钮语义（pin_base）、`oracle` 读真值。
- [ ] **Step 2: 8736 功能对照清单**（逐项打勾：场次列表 / 音频播放定位 / 真值预测时间轴 / 映射表 / 进度流）
- [ ] **Step 3: 对照通过后**，问用户是否删除 `streaming_asr/bench/`（**必须用户点头**，走回收站）
- [ ] **Step 4: Commit**（只 add `web/README.md`）

### Task 9: 完成定义核验（spec §1 的 5 条）

- [ ] **Step 1:** `python -m pytest web/tests -q` 全绿
- [ ] **Step 2:** `cd streaming_asr && python -m pytest -q tests ../test_c1_refinement.py` → `154 passed`
- [ ] **Step 3:** 计时实测：oracle 关 `/api/run` < 5 s；开 < 90 s
- [ ] **Step 4:** `oracle_k` 对拍状态为 `pass`，页面 DER 数字可见；把 `parity` 的日期写进 README
- [ ] **Step 5:** impeccable finish review：`impeccable.cmd detect --json web/index.html web/app.js web/style.css` 跑一次，修掉它报的真问题；记录 DESIGN.md 是否需要（extension 不重写，除非用户要求）
- [ ] **Step 6:** Commit

---

## Self-Review

- **Spec 覆盖**：spec §3→T1/T5/T6；§4.1→T1；§4.2→T2；§4.3→T3；§4.4→T4；§5→T5/T7；§6→T6；§7→T9；§8 风险→T1/T3 的报错与回退；§9 交付物→T8/T9。无遗漏。
- **步骤扫描**：每步一个动作 + 一个可检查结果；代码只给签名与断言，不给函数体（除生产公式这类被 spec 固定的）。
- **类型一致**：`segments_for` / `run_lanes` / `parity_status` / `clip_path` / `LANE_PAYLOAD_KEYS` 在 T1–T7 间名称一致。
- **Review Focus**：5 条各挂了测试（T1×1、T5×2、T4×1、T3+T6×1）。
- **篇幅**：本计划 ≈ spec 的 1.5 倍，未抄实现体。
