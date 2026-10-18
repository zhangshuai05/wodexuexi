# -*- coding: utf-8 -*-
"""LLM 后校正 harness（免微调）：把分离结果喂给 DeepSeek，
修正已有说话人标签之间的归属/切分，并可选绑定姓名。
- API Key：进程环境变量 DEEPSEEK_API_KEY → 缺省时读 Windows 用户级环境变量（注册表）。
- 断网/缺 key 时干净报错，不破坏原结果。
用法: python llm_correct.py output/a2ovl/<name>_a2ovl.json [--outdir output/llm]
输出: <outdir>/<name>_llm.json（与 a2ovl 同构 + "names" 字段）
"""
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"

PROMPT_TMPL = """你是一名会议转写后处理专家。下面是一段多人会议/对话的说话人分离结果。
现有说话人标签只有：{speaker_ids}。系统可能把已有说话人并进同一段、错标短句或把标签整体翻转。

请修正归属与切分，输出严格 JSON（只输出 JSON，不要解释，不要 markdown 代码块）：
{{"sentence_info": [{{"start": 秒, "end": 秒, "spk": 现有标签之一, "text": "文本"}}], "names": {{"标签": "可选名称"}}}}

规则：
- 只能使用上面已有的 speaker 标签；不能新增、删除或合并 speaker 标签；
- start/end 为秒（数字）；只能使用输入中已有的时间边界，不能猜测新的声学边界；
- 相邻同标签条目可合并（end 取后者，text 用空格拼接）；
- "names" 只在你有把握时给出，否则省略该字段；
- 不要新增或删除内容，只修正归属与切分；文本保持原文（错别字不改）。

输入：
{transcript}"""


def _speaker_ids(sentence_info):
    ids = set()
    for sentence in sentence_info:
        if "spk" in sentence:
            try:
                ids.add(int(sentence["spk"]))
            except (TypeError, ValueError):
                pass
        for speaker in sentence.get("spks", []):
            try:
                ids.add(int(speaker))
            except (TypeError, ValueError):
                pass
    return tuple(sorted(speaker for speaker in ids if speaker >= 0))


def build_prompt(sentence_info):
    lines = []
    for s in sentence_info:
        if not (s.get("text") or "").strip():
            continue
        if s.get("overlap"):
            spk = "speaker_" + "+".join(str(x) for x in s.get("spks", []))
        else:
            spk = f"speaker_{s.get('spk')}"
        lines.append(f"[{s['start']:.1f}-{s['end']:.1f}] {spk}: {s['text']}")
    speaker_ids = _speaker_ids(sentence_info)
    if not speaker_ids:
        raise ValueError("结果中没有可供 LLM 校正的 speaker 标签")
    return PROMPT_TMPL.format(
        speaker_ids="、".join(f"speaker_{speaker}" for speaker in speaker_ids),
        transcript="\n".join(lines),
    )


def _get_key(key=None):
    """取 API Key：进程环境 → Windows 用户级环境变量（注册表）兜底。

    服务进程的环境变量是"启动那一刻"的快照：若 key 是终端打开之后才配好的，
    进程里就没有它（LLM 校正报"缺少 DEEPSEEK_API_KEY"的实际原因），
    所以这里再读一次用户级的当前值。
    """
    if key:
        return key
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")[0] or None
    except (ImportError, OSError):
        return None


def call_llm(prompt, key=None, timeout=300):
    key = _get_key(key)
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（进程环境或 Windows 用户级环境变量任一处配置即可）")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8000,
    }).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    r = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def extract_json(text):
    """从 LLM 回复里提取严格 JSON 对象（容忍代码块围栏与前后废话）。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


def normalize(obj, allowed_speakers=(0, 1)):
    allowed = {int(speaker) for speaker in allowed_speakers}
    out = []
    for s in obj.get("sentence_info", []):
        try:
            st, en = float(s["start"]), float(s["end"])
            text = str(s.get("text", "")).strip()
            if en <= st or not text:
                continue
            if "spk" in s:
                spk = int(s["spk"])
                if spk not in allowed:
                    continue
                out.append({"start": round(st, 2), "end": round(en, 2),
                            "spk": spk, "text": text, "overlap": False})
            elif s.get("spks"):
                speakers = [int(speaker) for speaker in s["spks"]]
                if not speakers or any(speaker not in allowed for speaker in speakers):
                    continue
                out.append({"start": round(st, 2), "end": round(en, 2),
                            "spks": speakers,
                            "text": text, "overlap": True})
        except (TypeError, ValueError, KeyError):
            continue
    out.sort(key=lambda x: x["start"])
    return out


def correct(result, key=None, timeout=300):
    """result: a2ovl/fusion 结果 dict → 返回 (corrected_sentence_info, names, llm_raw)。"""
    prompt = build_prompt(result.get("sentence_info", []))
    raw = call_llm(prompt, key=key, timeout=timeout)
    obj = extract_json(raw)
    if obj is None:
        raise RuntimeError("LLM 输出无法解析为 JSON")
    ns = normalize(obj, _speaker_ids(result.get("sentence_info", [])))
    if not ns:
        raise RuntimeError("LLM 输出解析后为空")
    names = obj.get("names") if isinstance(obj.get("names"), dict) else None
    return ns, names, raw


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    outdir = os.path.join(HERE, "output", "llm")
    for a in sys.argv[1:]:
        if a.startswith("--outdir="):
            outdir = a.split("=", 1)[1]
    if not args:
        print(__doc__)
        sys.exit(0)
    for p in args:
        base = os.path.basename(p).rsplit(".", 1)[0]
        base = re.sub(r"_(a2ovl|fusion)$", "", base)
        d = json.load(open(p, encoding="utf-8"))
        try:
            ns, names, raw = correct(d)
        except Exception as e:
            print(f"[LLM FAILED] {base}: {type(e).__name__}: {e}", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        out = {"key": base, "strategy": "llm",
               "names": names, "sentence_info": ns}
        with open(os.path.join(outdir, base + "_llm.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"{base}: {len(ns)} entries (LLM 校正完成)", flush=True)


if __name__ == "__main__":
    main()
