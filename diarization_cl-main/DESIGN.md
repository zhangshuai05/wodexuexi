---
name: AISHELL-4 分离调试台
description: 暗底、1px 细线、等宽数字的本地四泳道调试台（8737）
colors:
  bg: "#0e1116"
  panel: "#11161e"
  panel-2: "#151b25"
  rule: "#232b36"
  rule-soft: "#1c232d"
  fg: "#e7ecf2"
  fg-mute: "#9aa5b4"
  fg-faint: "#7d8798"
  ok: "#4ade80"
  ok-edge: "#24463a"
  warn: "#fbbf24"
  warn-edge: "#4a3d1e"
  bad: "#fb7185"
  bad-edge: "#4a2029"
  speaker-truth-0: "#5eead4"
  speaker-truth-1: "#fbbf24"
  speaker-truth-2: "#a78bfa"
  speaker-truth-3: "#4ade80"
  speaker-truth-4: "#fb7185"
  speaker-truth-5: "#60a5fa"
  speaker-truth-6: "#f97316"
  speaker-truth-7: "#e879f9"
  speaker-pred-0: "#38bdf8"
  speaker-pred-1: "#f472b6"
  speaker-pred-2: "#a3e635"
  speaker-pred-3: "#c084fc"
  speaker-pred-4: "#fb923c"
  speaker-pred-5: "#22d3ee"
  speaker-pred-6: "#facc15"
  speaker-pred-7: "#94a3b8"
  run-bg: "#12312e"
  run-border: "#2f6b62"
  run-fg: "#c9fff5"
typography:
  title:
    fontSize: "14px"
    fontWeight: 600
    letterSpacing: "-0.01em"
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Microsoft YaHei, sans-serif"
    fontSize: "13px"
    lineHeight: 1.5
  mono:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, SF Mono, monospace"
    fontSize: "12px"
    lineHeight: 1.5
  micro:
    fontFamily: "ui-monospace, Cascadia Mono, Consolas, SF Mono, monospace"
    fontSize: "11px"
    lineHeight: 1.5
  section-title:
    fontSize: "11px"
    fontWeight: 600
    letterSpacing: "0.08em"
    lineHeight: 1.5
rounded:
  default: "3px"
spacing:
  xxs: "2px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "10px"
  xl: "12px"
  xxl: "16px"
  xxxl: "32px"
components:
  badge:
    textColor: "{colors.fg-mute}"
    rounded: "{rounded.default}"
    padding: "4px 6px"
    typography: "{typography.micro}"
  run-button:
    backgroundColor: "{colors.run-bg}"
    textColor: "{colors.run-fg}"
    rounded: "{rounded.default}"
    padding: "7px 18px"
  metrics-table-wrap:
    backgroundColor: "{colors.panel}"
    padding: "8px"
  track:
    backgroundColor: "{colors.panel-2}"
    height: "16px"
---

# DESIGN.md —— 继承的世界

本文件只记录 `web/` 调试台**已经在用**的视觉规则，供后续维护者照抄；不发明新规则。
每一行右侧注明取值来源（`web/style.css` 行号 / `web/app.js` 函数名），改代码请同步改这里。

## Overview

一块暗底、密排、无装饰的本地调试台（spec §2「不做动效与装饰」）：信息密度优先，颜色**只**用于区分说话人，
DER 受对拍门禁约束（不 pass 就显示 `—（未对拍）`）。页面是单页原生 JS，无框架、无构建链、无外部资源。
（来源：`docs/superpowers/specs/2026-10-10-web-debug-bench-design.md` §2、§6.3）

## Colors

- 底色/面板/细线五件套：`--bg #0e1116`、`--panel #11161e`、`--panel-2 #151b25`、
  `--rule #232b36`、`--rule-soft #1c232d`。（来源：`web/style.css` `:root`）
- 前景三级：`--fg #e7ecf2`（正文/播放头）、`--fg-mute #9aa5b4`（次级）、`--fg-faint #7d8798`（标签/小字）。
  （来源：`web/style.css` `:root`）
- 状态色只有三个：`--ok #4ade80`、`--warn #fbbf24`、`--bad #fb7185`；徽标描边用同色系的暗调
  `#24463a` / `#4a3d1e` / `#4a2029`（`ok-edge` / `warn-edge` / `bad-edge`）。（来源：`web/style.css` `.badge.*`）
- 说话人配色**两套八槽**：真值 `--s0..--s7`、预测 `--p0..--p7`（取值见上方 frontmatter）。
  取槽规则（不是哈希）：真值按 `/api/truth` 段表里**首次出现顺序**取 `--s*`；
  预测按簇号 `#N`（`pairs[].pred` / `segment.spk`）取 `--pN % 8`；两边都取不到时兜底 `#94a3b8`。
  （来源：`web/app.js` `TRUTH_COLORS` / `truthSpeakers` / `truthColor` / `PREDICTION_COLORS` / `predictionColor`）
- 颜色不用于装饰：段块颜色只表示说话人身份，徽标颜色只表示状态。（来源：`web/style.css` 文件头注释）

## Typography

- 正文 `13px/1.5` 系统 sans（`--sans`）；所有数字/ID/时间码/日志走等宽 `--mono`
  `ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace` + `font-variant-numeric: tabular-nums`
  （对齐不跳动）。（来源：`web/style.css` `body` / `:root`）
- 字号栈：标题 14px、正文/表格/旋钮/场次 ID 12px、功能性小字与时间码 **11px 下限**
  （`th`、`td small`、`.m-meta`、`.meta`、`.badge`、`.lane-label`、`.tick-label`、`.note`、`.log`）；
  小节标题 `h2` = 11px 大写 + `letter-spacing: .08em`。行高：正文/表格 1.5，`.meta` 1.4，
  `.badge` 1，`.note` / `.log` 1.6。（来源：`web/style.css` `header h1` / `h2` / 各选择器）
- 时间码格式 `m:ss`（四舍五入、分补齐），轴刻度步长按总时长取 10/30/60/120 s。
  （来源：`web/app.js` `fmtTime` / `drawRuler`）

## Layout

- 骨架：`body` 两行网格（状态条 + 主区，`height: 100vh` 不整页滚动）；`main` 三列
  `260px minmax(0,1fr) 320px`，≤1100px 收成 `220px / 260px`；左/中/右各自滚动。
  （来源：`web/style.css` `body` / `main` / `@media`）
- 间距节奏（8 的倍数为主）：`2/4/6/8/10/12/16/32`；面板内边距 12px，状态条 `10px 16px`，
  舞台 `12px 16px 32px`，块与块之间 16px。（来源：`web/style.css` `.pane-*` / `.stage` / `.block`）
- 轨道几何 = **两列网格 + 百分比绝对定位**：每一行 `.row` 是
  `grid-template-columns: var(--gutter)=108px minmax(0,1fr)`、`gap: 0 8px`，左列是泳道名（右对齐）；
  段块 `.blk` 在 `position:relative` 的 `.track` 内用 `left = start/total`、`width = (end-start)/total`
  的百分数（`toFixed(4)`，最小 0.001 s）定位。（来源：`web/style.css` `.row` / `.track` / `.blk`；
  `web/app.js` `percent`）
- 轨道高度：标尺 18px（下沿 1px 细线），段轨 16px + 上下 2px 外边距；播放头 1px 白线、上下各内缩 8px。
  （来源：`web/style.css` `.track.ruler` / `.track` / `.playhead`）

## Elevation & Depth

- **没有 elevation**：全部深度靠 1px 细线（`--rule` / `--rule-soft`）+ 三档面板底色（`--bg` / `--panel` / `--panel-2`）。
  禁止 `box-shadow` / `text-shadow`。（来源：`web/style.css` 文件头注释；`web/tests/test_page_contract.py`
  `test_layout_columns_and_no_decorations`）

## Shapes

- 圆角只有一种 `3px`（徽标、场次项、下拉、按钮）。（来源：`web/style.css` `.badge` / `.meeting` / `select` / `button#run`）
- 段块是纯矩形色块、`min-width: 2px`；不做圆角、描边、透明度特效。（来源：`web/style.css` `.blk`）

## Components

- 状态条徽标 `.badge`：`--panel` 底 + 1px 状态色描边 + 11px 等宽；`#parity-badge` 在
  `parity.status !== "pass"` 时显示「DER 对拍未通过」，DER 数字一律 `—（未对拍）`。
  （来源：`web/style.css` `.badge`；`web/app.js` `renderStatusBar` / `derDisplay`）
- 运行按钮 `button#run`：`#12312e` 底 / `#2f6b62` 描边 / `#c9fff5` 字；禁用时退回 `--panel-2` + `--fg-faint`。
  （来源：`web/style.css` `button#run`）
- 指标表：`.tablewrap` 1px 边框 + 8px 内衬 + 内部滚动；行分隔只用 `--rule-soft` 1px 下划线；
  选中行 `--panel-2` 底 + 1px 描边。（来源：`web/style.css` `.tablewrap` / `th, td` / `tbody tr.selected`）
- 时间轴：从下往上依次是标尺行（`秒` 标签 + `.track.ruler`）、真值轨、各泳道轨；真值块用 `--s*`，
  泳道块用 `--p*`。（来源：`web/app.js` `renderTimeline`）

## Do's and Don'ts

- Don't：加动效 —— 禁止 `animation` / `transition` / `@keyframes`。（来源：`web/tests/test_page_contract.py`
  `test_layout_columns_and_no_decorations`）
- Don't：渐变/阴影/装饰 —— 禁止 `gradient` / `box-shadow` / `text-shadow`。（来源：同上）
- Don't：外部资源或 `@import` / `url(`；页面只引本地 `style.css` 与 `app.js`。（来源：同上）
- Do：功能性文字 ≥ 11px（表头、时间码、日志、标签都在此列）；正文 13px、块标题 12px。
  （来源：`web/style.css` 各选择器；impeccable detect 的 undersized-ui-text 规则）
- Do：数字一律等宽 + `tabular-nums`，时间码用 `m:ss`。（来源：`web/style.css`；`web/app.js` `fmtTime`）
- Do：新增颜色前先问「它是不是在区分说话人或状态」；不是就不要加。（来源：`web/style.css` 文件头注释）
