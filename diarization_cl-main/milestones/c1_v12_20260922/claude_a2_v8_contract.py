# -*- coding: utf-8 -*-
"""ASR 幻觉/失败契约（v7 缺陷 3）。

v7 的 run_claude_lane 逐段解码后直接取 text，异常吞成 ""，顶层无条件
verdict="reliable"。DeepSeek 观察到的"文本幻觉多"在接口里没有任何表达方式。

本模块只做一件事：把"这段文本可不可信"从常量变成一个算出来的量。
不改 ASR 与说话人边界的耦合（那是 v8 的结构变更），因此可以在冻结的 v7
上直接跑，零回归风险。

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


def _verdict_from(flags, n_speakers):
    """文件级三态。v7 这里是字面常量 "reliable"。

    只有**文本可疑**（幻觉类）和人数不确定才降级。短段未解码是覆盖缺口，
    单独计数但不把整份判定打成 uncertain —— 否则任何短答文件都不可用。
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
    if n_gap:
        reasons.append(f"{n_gap} 段短于 ASR 门限，未解码")
    if not flags:
        reasons.append("无语音段")
    if n_bad or n_speakers == "uncertain":
        return "uncertain", reasons
    return "reliable", reasons
