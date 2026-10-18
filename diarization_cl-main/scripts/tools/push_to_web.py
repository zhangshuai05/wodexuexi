# -*- coding: utf-8 -*-
"""把 A2 (Claude 侧) 的分离结果推送到 DSH 测试台 /api/external。

用法: push_to_web.py <slot> <src_dir>
  slot     = c1 | c2
  src_dir  = 含 {file}_diar.json 的目录 (sentence_info: start/end/spk/text)

只读 DSH 目录,只写本机文件。文件名与网页上传的 wav 逐字一致(含扩展名、大小写)。
"""
import json, os, sys, urllib.parse, urllib.request

BASE = r"c:\Users\EDY\Desktop\new"
AUDIO = os.path.join(BASE, "Audio")  # Device116=Call*, Device118=Note*
SERVER = "http://127.0.0.1:8734/api/external"

# 10 个安静文件的 wav 名 (与 DSH Audio 目录逐字一致)
FILES = [
    "Call20260718211423.wav",
    "Call20260718211506.wav",
    "Call20260718211745.wav",
    "Call20260718211842.wav",
    "Call20260718211938.wav",
    "Note20260803112329.wav",
    "Note20260803112415.wav",
    "Note20260803112655.wav",
    "Note20260803112752.wav",
    "Note20260803112841.wav",
]


def push(slot, name, result):
    url = f"{SERVER}?slot={slot}&name={urllib.parse.quote(name)}"
    data = json.dumps(result, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    slot = sys.argv[1]
    src_dir = sys.argv[2]
    ok, fail = 0, []
    for name in FILES:
        stem = name.rsplit(".", 1)[0]
        jpath = os.path.join(src_dir, stem + "_diar.json")
        if not os.path.exists(jpath):
            fail.append((name, "no result file")); continue
        result = json.load(open(jpath, encoding="utf-8"))
        # 时间轴校验: end > start, spk 为 int
        for s in result.get("sentence_info", []):
            assert s["end"] >= s["start"], f"{name}: end<start"
            assert isinstance(s["spk"], int), f"{name}: spk not int"
        result.setdefault("method", f"claude-{slot}")
        try:
            rec = push(slot, name, result)
            print(f"OK  {slot} {name}: entries={rec.get('entries')} saved_at={rec.get('saved_at')}")
            ok += 1
        except Exception as e:
            fail.append((name, str(e)))
            print(f"ERR {slot} {name}: {e}", flush=True)
    print(f"\n{slot}: {ok}/{len(FILES)} pushed; fails={fail}")


if __name__ == "__main__":
    main()
