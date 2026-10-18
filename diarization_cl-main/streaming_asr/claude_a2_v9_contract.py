# -*- coding: utf-8 -*-
"""ASR 幻觉/失败契约（v7 缺陷 3，v9 修复第 1 项）。

v7 的 run_claude_lane 逐段解码后直接取 text，异常吞成 ""，顶层无条件
verdict="reliable"。DeepSeek 观察到的"文本幻觉多"在接口里没有任何表达方式。

v8 把 reliable 从常量变成了算出来的量，但契约本身是假的（2026-09-15 复核）：
  1. too_short 是"覆盖缺口"，被排除在 SUSPECT 之外且不参与任何降级分支，
     于是"没有任何语音段""20 段全部未解码"都返回 reliable。
  2. MIN_CHARS_PER_SEC 定义了却从未使用：100 秒有声只识别出一个字也是 ok。
  3. 顶层 `if not flags` 只加了一条 reason 字符串，返回值仍是 reliable。
v9 的修法：覆盖缺口单独走一条降级通道（按**文件级字符覆盖率**判，不按段数
判——短答文件不该因为句子多就变 uncertain），并把速率下限接进该通道。

判据全部可用纯文本+时间算（不依赖模型内部分数，FunASR/SEACO 接口不保证给）：
  空文本            —— ASR 无输出
  字符速率           —— 中文正常 2~8 字/秒；超上限意味着解码器在自行补全
  4-gram 重复        —— 同一段内自我复读是典型的 hallucination loop
  有声覆盖           —— 文本量与该段 VAD 有声时长不匹配（极少字 + 长有声）
"""
import re

CJK = re.compile(r"[一-鿿]")
WORDISH = re.compile(r"[A-Za-z0-9]+")
PUNCT = re.compile(r"[\s，。！？、；：,.!?;:~…—\-“”‘’（）()\[\]{}]")

MAX_CHARS_PER_SEC = 11.0
MIN_CHARS_PER_SEC = 0.8
NGRAM = 4
NGRAM_REPEAT_MAX = 3


def _cjk_len(text):
    return len(CJK.findall(text)) + len(WORDISH.findall(text))


def _ngram_repeat(text, n=NGRAM, min_reps=NGRAM_REPEAT_MAX):
    """最长**连续**重复片段，返回 (重复次数, 片段)。

    必须连续：中文口语里"谢谢谢谢"是自然的，而 hallucination loop 是
    "我们我们我们"这种同一片段原地复读（中间无其它内容）。
    实测把非连续计数打开会在真实输出上误报（"谢谢谢谢谢谢。"）。
    """
    s = PUNCT.sub("", text)
    if len(s) < n * min_reps:
        return 0, ""
    best_cnt, best_frag = 0, ""
    i = 0
    while i <= len(s) - n:
        g = s[i:i + n]
        c = 1
        j = i + n
        while s[j:j + n] == g:
            c += 1
            j += n
        if c > best_cnt:
            best_cnt, best_frag = c, g
        i = j if c > 1 else i + 1
    return best_cnt, best_frag


def segment_contract(text, start, end, vsegs, min_dur=0.3):
    """单段契约。返回 {flag, chars, cps, repeat, speech_sec, reasons}。

    flag 分两类，混在一起会让"短段没解码"冒充"ASR 幻觉"：
      覆盖缺口（不是幻觉，是没读到）：empty / too_short
      文本可疑（正是 DeepSeek 观察到的幻觉）：too_fast / repeated / no_speech
      ok

    注意这里**不判**速率过低（chars/speech < MIN_CHARS_PER_SEC）：中文短答
    "对。""好。"本身就只有一两个字，段级判低速率会把它们全部标成可疑。
    低速率只在文件级作为覆盖判据使用（见 file_coverage）。
    """
    text = (text or "").strip()
    dur = max(0.0, float(end) - float(start))
    speech = sum(max(0.0, min(end, e) - max(start, s)) for s, e in vsegs)
    chars = _cjk_len(text)
    # 速率分母用 VAD 有声时长，不用段墙钟：短答的段里含静音，
    # 用墙钟会把"对。"这种正常短答算成超速（实测 3.33/s 对 0.3s 段）。
    denom = max(0.1, min(dur, speech)) if speech > 0 else dur
    cps = chars / denom if denom > 0 else 0.0
    reps, frag = _ngram_repeat(text)
    reasons = []

    if not text:
        if dur < min_dur:
            reasons.append(f"段长 {dur:.2f}s < {min_dur}s，ASR 未调用（非幻觉）")
            flag = "too_short"
        else:
            reasons.append("ASR 无输出")
            flag = "empty"
    elif dur < min_dur:
        reasons.append(f"段长 {dur:.2f}s < {min_dur}s，短答，不独立解码")
        flag = "too_short"
    elif speech > 0 and speech < 0.3 * dur:
        reasons.append(f"VAD 有声覆盖仅 {speech:.2f}s/{dur:.2f}s")
        flag = "no_speech"
    elif cps > MAX_CHARS_PER_SEC:
        reasons.append(f"字符速率 {cps:.1f}/s > {MAX_CHARS_PER_SEC}（分母=VAD 有声 "
                       f"{speech:.2f}s），超出语速物理上限")
        flag = "too_fast"
    elif reps >= NGRAM_REPEAT_MAX and reps * NGRAM >= 12:
        reasons.append(f"片段「{frag}」连续重复 {reps} 次，疑似 hallucination loop")
        flag = "repeated"
    else:
        flag = "ok"
    return {"flag": flag, "chars": chars, "cps": round(cps, 2),
            "repeat": int(reps), "speech_sec": round(speech, 2),
            "reasons": reasons}


SUSPECT = ("empty", "no_speech", "too_fast", "repeated")
COVERAGE_GAP = ("too_short",)


def file_coverage(chars, wall_sec, speech_sec, min_cps=MIN_CHARS_PER_SEC):
    """文件级字符覆盖率。v8 定义了 MIN_CHARS_PER_SEC 却从未使用。

    分母用 VAD 有声总时长（不是文件墙钟）：整份录音里大段无人说话时，
    用墙钟会把正常文件算成覆盖率不足。有声时长为 0 或缺失时退回墙钟。
    """
    denom = float(speech_sec) if speech_sec and speech_sec > 0 else float(wall_sec)
    if denom <= 0:
        return 0.0
    return float(chars) / denom


def _verdict_from(flags, n_speakers, coverage=None, min_cps=MIN_CHARS_PER_SEC):
    """文件级三态。v7 这里是字面常量 "reliable"。

    三态由三类证据合成：
      文本可疑（SUSPECT 任一）        → uncertain
      人数不确定                      → uncertain
      字符覆盖率 < min_cps（含无语音） → uncertain  ← v9 新增的通道
    coverage 为 None 时跳过覆盖率判据（调用方拿不到时长时保持 v8 行为）。

    coverage 形如 {"chars":…, "wall_sec":…, "speech_sec":…}；无语音段时
    chars=0 也会被这条通道抓住，不再需要单独的特例分支。
    """
    reasons = []
    if n_speakers == "uncertain":
        reasons.append("说话人数不确定")
    elif isinstance(n_speakers, int) and n_speakers > 2:
        reasons.append(f"检测到 {n_speakers} 人（无真实 3 人素材验收）")
    bad = sorted(set(f for f in flags if f in SUSPECT))
    n_bad = sum(1 for f in flags if f in SUSPECT)
    n_gap = sum(1 for f in flags if f in COVERAGE_GAP)
    if n_bad:
        reasons.append(f"{n_bad} 段文本可疑（{'/'.join(bad)}）")

    cov = None
    if coverage is not None:
        cov = file_coverage(coverage.get("chars", 0), coverage.get("wall_sec", 0.0),
                            coverage.get("speech_sec", 0.0), min_cps)
    if not flags:
        reasons.append("无语音段")
    if n_gap:
        reasons.append(f"{n_gap} 段短于 ASR 门限，未解码")
    if cov is not None:
        reasons.append(f"字符覆盖率 {cov:.2f} 字/秒（下限 {min_cps}）"
                       if coverage.get("speech_sec") else
                       f"字符覆盖率 {cov:.2f} 字/秒（无声有段，按下限 {min_cps}）")

    if bad or n_speakers == "uncertain":
        return "uncertain", reasons
    if not flags:
        return "uncertain", reasons
    # 覆盖缺口的兜底：一份文件里**没有任何一段**真正解码出文本，无论它是
    # 全 too_short 还是全 empty，都不能声称 reliable。v8 在这里回落到
    # reliable，是"20 段全部未解码 → reliable"那条复现的直接成因。
    # 注意只在"零个 ok"时触发：有 ok 段的文件即使夹带短段也不降级。
    if COVERAGE_GAP and not any(f == "ok" for f in flags):
        return "uncertain", reasons
    if cov is not None and cov < min_cps:
        return "uncertain", reasons
    return "reliable", reasons
