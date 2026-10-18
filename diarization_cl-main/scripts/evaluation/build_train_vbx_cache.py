# -*- coding: utf-8 -*-
"""建 `train_L` 的 RTTM 门窗缓存（`w2_h1.npz`），供 VBx 的 PLDA 训练池用。

**为什么必须新建**：`_v13_spike/cache/` 下只有 test 的 20 场有 `w2_h1.npz`；
train_L 30 场一份都没有，而它是本 spike 唯一的 PLDA 训练来源（spec §2.1）。

**为什么不重写提取循环**：`build_cache.extract_meeting` 是 test 侧那份缓存的
生成者，逐位复用才能保证「训练池的窗」与「评测的窗」同法（坑 A7）。
本脚本只做三件它没做的事：按日期前缀发现场次、指定输出目录、按场落盘。

    $env:PYTHONIOENCODING="utf-8"
    python -u -B scripts/evaluation/build_train_vbx_cache.py

**R1 修复（守卫设计缺陷，spec §2.1；2026-10-14 由复审探针实证）**：
原守卫写成 `str(meeting).startswith("2020")` —— 它校验的是**未归一化的字符串**，
而下游用 `Path(root) / "wav" / f"{meeting}.flac"` 去读文件，`..` 要到**打开文件时**
才被折叠。于是 `meeting="2020/../L_R004S01C01"` 能通过守卫、却读到 holdout 的
flac/rttm，并把缓存写进 holdout 目录（**实测发生**）。
⇒ 修法不是"再加几条黑名单"，而是让**校验对象与使用对象是同一个值**：
先归一化/解析，再校验，然后**只用返回值**去拼所有路径。
见 `validate_meeting()` 与 `validate_out()`。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPIKE = Path(r"D:\yinp\AISHELL-4\_v13_spike")
if str(SPIKE) not in sys.path:
    sys.path.insert(0, str(SPIKE))

import build_cache as BC  # noqa: E402

DEFAULT_ROOT = Path(r"D:\yinp\AISHELL-4\data\train_L")
# R1：train_L 的缓存必须写在**专门目录**里，不能与 test 侧 20 场同处一层
# （同处一层时，"写错到哪个场次"与"写错到哪个 split"都无法从路径上看出来）。
TRAIN_SUBDIR = "train_L"
DEFAULT_OUT = SPIKE / "cache" / TRAIN_SUBDIR
TRAIN_PREFIX = "2020"          # train_L 的 id 以日期开头；test 的是 `L_R003...`
# R2：**白名单**格式。逐特征"拒 `..`、拒分隔符"会漏（`"2020/./L_R004S01C01"`
# 不含 `..`、归一化也不逃逸）；白名单是"只有长这样才放行"。
TRAIN_MEETING_RE = re.compile(r"^2020\d{4}_[A-Z]_R\d{3}S\d{2}C\d{2}$")


class NotTrainSplit(ValueError):
    """试图建非 train_L 场次的缓存 / 把缓存写到不该写的地方 —— 硬拒。

    存在的理由：spec §2.1 规定 holdout（`L_R004*` / `S_R003*`）是封存集、
    **不许读**。靠"我不会传错 --meetings"不是保障；一个 argv 就能把缓存写进
    holdout 目录，而缓存一旦落盘，"读没读过"就再也无法自证。
    """


def validate_meeting(meeting):
    """返回**归一化后**的合法 train_L 场次名；不合法一律 raise `NotTrainSplit`。

    **这是 R1/R2 的核心**。原实现是 `str(meeting).startswith(TRAIN_PREFIX)`：
    它只看输入的**前 4 个字符**，而 `build()` 真正拿去拼路径的是同一个未归一化的
    字符串 —— `"2020/../L_R004S01C01"` 因此能以 `2020` 开头、却在打开文件时被
    折叠成 holdout 的路径（**实证发生过**，见模块 docstring）。
    修法不是"再加几条黑名单"，而是：**校验的对象必须就是被使用的那个值** ——
    先归一化成"单个路径分量"，再对这个分量做判断，最后**返回它**。

    **R2 强化（复审建议，2026-10-14）**：判据从"逐特征拒 `..` / 分隔符"升级为
    **整体白名单正则** `TRAIN_MEETING_RE`。逐特征拒已被证明会漏
    （`"2020/./L_R004S01C01"` 不含 `..`、归一化也不逃逸，但仍是路径而非场次名），
    而白名单是"只有长得像场次名的才放行"，漏面小得多。

    归一化口径（故意严格）：
    - 先把 `\\` 与 `/` **都当分隔符**：`"2020\\..\\S_R003S01C01"` 若只按 POSIX
      解析会被当成**单个文件名**，从而绕过下面 `name != text` 的检查；
    - 用 `PurePosixPath(...).parts` 判定：**必须恰好 1 个 part**，
      于是 `.` / `..` / 绝对路径 / 任何分隔符都自然被拒；
    - 再要求：非空、无分隔符、无 `..`、不以点开头、以 `TRAIN_PREFIX` 开头、
      **整体匹配 `TRAIN_MEETING_RE`**（挡住 `"2020"`、`"2020\x00x"`、
      带尾随空格或扩展名等"看着像但不是场次名"的输入 —— 这类会让下游抛
      `ValueError`/`OSError` 而不是干净的 `NotTrainSplit`）。

    纯函数、不碰文件系统 ⇒ **可单测**（原守卫做不到：它必须先跑到真路径上）。
    """
    text = str(meeting)
    unified = text.replace("\\", "/")
    parts = PurePosixPath(unified).parts
    if len(parts) != 1:
        raise NotTrainSplit(
            f"非法场次名 {text!r}：必须是不含任何路径分隔符的单个场次名"
            f"（归一化后解析出 {len(parts)} 个路径分量）。"
            "本脚本只允许建 train_L 缓存 —— holdout 是封存集，spec §2.1 禁止触碰。")
    name = parts[0]
    if (not name or name != text or "/" in name or "\\" in name
            or ".." in name or name.startswith(".")):
        raise NotTrainSplit(
            f"非法场次名 {text!r}：归一化后为 {name!r}，仍不是干净的单个场次名。")
    if not name.startswith(TRAIN_PREFIX):
        raise NotTrainSplit(
            f"{name} 不是 train_L 场次（只认 {TRAIN_PREFIX!r} 前缀）。"
            "本脚本只允许建 train_L 缓存 —— holdout 是封存集，spec §2.1 禁止触碰。")
    # R2：白名单格式 —— 上面那些特征检查全部通过 *也不等于* 它是个场次名。
    if not TRAIN_MEETING_RE.match(name):
        raise NotTrainSplit(
            f"非法场次名 {text!r}：不符合 train_L 场次名的白名单格式 "
            f"{TRAIN_MEETING_RE.pattern}（期望形如 20200706_L_R001S01C01）。")
    return name


def validate_out(out, spike_cache=None):
    """返回**解析后**的合法输出根；越界一律 raise `NotTrainSplit`。

    R1 新增、R2 强化：原先只校验 meeting 名，`--out` 指到哪儿就写到哪儿 ——
    `--out <test 侧根>` 或 `--out <任意路径>` 都能写出去。
    要求（全部对 `resolve()` **之后**的值判断，即"解析后再断言包含性"）：
    - 必须**位于缓存根之内**（默认 `SPIKE/cache`）—— 挡住 `out=/tmp/x`；
    - **不得落在 test 侧缓存树内**（`SPIKE/cache/test`）—— 挡住把训练缓存
      写进 test 侧、把已被 Task 4 依赖的那份产物搅坏。

    不存在时不出错：`Path.resolve()` 对不存在的路径同样做归一化，
    下游 `mkdir(parents=True)` 会创建。返回的是**解析后的 Path**，调用方必须用它。
    """
    spike_cache = Path(spike_cache) if spike_cache is not None else SPIKE / "cache"
    root = Path(out).resolve()
    base = spike_cache.resolve()
    test_cache = (spike_cache / "test").resolve()
    if root == base or base not in root.parents:
        raise NotTrainSplit(
            f"输出目录 {root} 不在指定缓存根 {base} 之内。"
            f"train_L 的缓存必须写在缓存根之内（建议 {TRAIN_SUBDIR!r} 子目录）。")
    if root == test_cache or test_cache in root.parents:
        raise NotTrainSplit(
            f"输出目录 {root} 落在 test 侧缓存树 {test_cache} 里 —— 拒绝。"
            "test 侧 20 场的产物已被下游依赖，训练缓存不许写进去。")
    return root


def train_meetings(root):
    """发现 train_L 的场次。

    **不要**用 `v12_aishell4.split_meetings`：它内部对每个 stem 调
    `meeting_group()`，其正则只认 `L_R003S01C02` 形态，遇到
    `20200706_L_R001S01C01` 会抛 ValueError。
    """
    return sorted(p.stem for p in (Path(root) / "TextGrid").glob("*.rttm")
                  if p.stem.startswith(TRAIN_PREFIX))


def assert_inside(path, root, label):
    """断言 `path` 归一化后仍在 `root` 之内 —— 否则 raise `NotTrainSplit`。

    R2 把这段逻辑**内联**在 `build()` 里，结果复审实测它**零覆盖**：
    把 `if not path.is_relative_to(inside)` 改成 `if False:` 之后，
    28 条相关测试**逐字相同**全绿。R3 因此把它抽成**可单测的纯函数**，
    交给 `test_assert_inside_rejects_paths_that_escape_the_root` 直接钉住。

    ⚠️ 两边都先 `resolve()` 再比（M2 修复）：原实现只对 `path` 做
    `os.path.normpath`、基准却用 `Path(root).resolve()`，传入**相对** `--root`
    时会因两边口径不同而误报"逃出了允许的根"（fail-closed，但报错信息误导）。
    现在两边同一口径。返回归一化后的 Path，调用方应当用它。
    """
    name = os.path.normpath(str(path))
    base = Path(root).resolve()
    if not Path(name).resolve().is_relative_to(base):
        raise NotTrainSplit(
            f"{label} 路径 {name} 逃出了允许的根 {base} —— 拒绝执行。")
    return Path(name)


def build(meeting, root, out, window, hop, models, seconds=0.0):
    # ★ 控制器追加的硬守卫（用户 2026-10-13 裁决）+ R1/R2/R3 修复（复审实证的穿透）。
    # 关键：**先归一化再校验，并且只用返回值拼路径** —— 校验对象 == 使用对象。
    name = validate_meeting(meeting)
    out = validate_out(out)
    import soundfile as sf

    # R2/R3：读侧/写侧的**包含性断言**（复审点名的"最宽的洞"）。
    # 名字已经过白名单，但这里再对**最终会被打开/写入的路径**断言一次。
    wav = assert_inside(Path(root) / "wav" / f"{name}.flac", root, "wav")
    rttm = assert_inside(Path(root) / "TextGrid" / f"{name}.rttm", root, "rttm")
    target = assert_inside(Path(out) / name, out, "out")
    segs = BC.read_rttm(rttm)
    stop = int(16000 * seconds) if seconds else -1
    audio, rate = sf.read(wav, dtype="float32", start=0, stop=stop, always_2d=True)
    audio = audio[:, 0]
    emb, times, speakers = BC.extract_meeting(
        audio, rate, segs, window, hop, "rttm", models["spk"])
    target.mkdir(parents=True, exist_ok=True)
    tag = f"w{window:g}_h{hop:g}"
    np.savez_compressed(
        target / f"{tag}.npz", emb=emb, times=times,
        speakers=np.asarray(speakers).astype(str),
        votes=np.asarray([BC.vote_split(t, t + window, segs) for t in times],
                         dtype=np.float32))
    return {"n": int(len(emb)), "dim": int(emb.shape[1]) if len(emb) else 0,
            "gt_speakers": sorted({s["speaker"] for s in segs}),
            "duration": round(len(audio) / rate, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = 整场")
    ap.add_argument("--meetings", nargs="*", default=None)
    args = ap.parse_args()

    from web_server import get_models

    root = Path(args.root).resolve()      # M2：立刻归一化，让包含性断言两边同一口径
    meetings = args.meetings or train_meetings(root)
    # ★ 控制器追加：显式传入的场次也走同一守卫，且**整轮开始前**就退出 ——
    #   不要先写了几场再在中间报错（那样缓存与 index 会留下半成品）。
    # R1：改用 validate_meeting ⇒ `2020/../L_R004...` 这类绕过输入在这里也被拦；
    #     且这一步排在 `get_models()` **之前**，模型都不会被加载。
    bad = []
    for m in meetings:
        try:
            validate_meeting(m)
        except NotTrainSplit as exc:
            bad.append(str(exc))
    if bad:
        raise SystemExit(
            f"★ 拒绝执行：--meetings 里有 {len(bad)} 个非法场次：{bad[:5]}"
            f"{' …' if len(bad) > 5 else ''}。本脚本只建 train_L 缓存。")
    if not meetings:
        raise SystemExit(f"没找到 train_L 场次（检查 --root：{root}）")
    # R1：输出根同样要在**任何落盘之前**校验，并且用返回值。
    # R3（I3 修复）：`validate_out` 抛的是 `NotTrainSplit`(ValueError) —— CLI 工具
    # 不该把 traceback 冒给用户，也不该与上面那条场次守卫的体验不一致（那条是
    # `SystemExit`）。这里统一包成**干净的 CLI 退出**。
    # ⚠️ 变异不对称的成因（复审实测，**勿写反** —— 本注释上一版就把因果写反了）：
    # 本函数把"校验"与"报告"拆成**两句**，所以"删掉校验段"的变异会让**存活的
    # print** 引用未赋值的 `out` ⇒ 红在 `NameError`。那是"删掉赋值语句"的必然
    # 产物，**不是**本设计要防的东西。
    # 反过来，若写成一句 `print(f"… → {validate_out(...)}")`，删掉它之后 `out` 的
    # 下一次使用是更下面的 `index_path = out / "train_index.json"`，而那一句排在
    # `get_models()` **之后** ⇒ tripwire 会先开火（红在真判据上）。
    # ⇒ **真正的设计原则只有两条**：① 校验必须用 `out`；② 校验必须排在
    #   `get_models()` **之前**。上面的 `NameError` 与这两条无关。
    # 显式传 `SPIKE / "cache"`：与 `validate_out` 的默认一致，但把它写出来，
    # 让"基准是谁"在调用点可见。
    try:
        out = validate_out(args.out, SPIKE / "cache")
    except NotTrainSplit as exc:
        raise SystemExit(f"★ 拒绝执行：{exc}")
    print(f"[train-cache] {len(meetings)} 场，win={args.window} hop={args.hop} "
          f"gate=rttm → {out}", flush=True)

    models = get_models(lambda m: None, need_stream=False, need_offline=True)
    index_path = out / "train_index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8")).get("meetings", {})

    for i, meeting in enumerate(meetings, 1):
        t0 = time.perf_counter()
        try:
            entry = build(meeting, root, out, args.window, args.hop, models,
                          args.seconds)
        except FileNotFoundError as exc:
            # 静默丢场是最贵的坑（A6）：少了几场和本来就几场长得一模一样。
            print(f"  [{i}/{len(meetings)}] {meeting} SKIP: {exc}", flush=True)
            index[meeting] = {"skip": f"{type(exc).__name__}: {exc}"}
            continue
        index[meeting] = entry
        print(f"  [{i}/{len(meetings)}] {meeting} n={entry['n']} "
              f"({entry['duration']:.0f}s, {time.perf_counter() - t0:.1f}s)",
              flush=True)
        # 每场落一次：30 场 × 20 小时音频，中途断了不该丢已完成的场次。
        # I1 修复（复审）：**必须把 `--seconds` 入账**。否则"只跑了前 60 秒的
        # 局部缓存"与"整场缓存"在 index 里长得一模一样（都只有 skip_seconds），
        # 核对测试无法区分它们 —— 而 §8 又把 `--seconds 60` 写成推荐烟测命令。
        index_path.write_text(json.dumps(
            {"schema": "v13-train-vbx-cache-v1", "root": str(root),
             "gate": "rttm", "skip_seconds": BC.SKIP_SECONDS,
             "seconds": args.seconds,
             "window": args.window, "hop": args.hop,
             "meetings": index}, ensure_ascii=False, indent=2), encoding="utf-8")

    skipped = [m for m, v in index.items() if "skip" in v]
    print(f"\n写入 {index_path}")
    if skipped:
        print(f"⚠ 跳过 {len(skipped)} 场：{skipped}")


if __name__ == "__main__":
    main()
