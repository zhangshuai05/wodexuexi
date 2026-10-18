# -*- coding: utf-8 -*-
"""23 语料过 8735 的 v7，落盘完整 diag —— 给 v8.1 的零回归验收用。"""
import json, os, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
DST = r"C:\Users\EDY\Desktop\new\streaming_asr\output\_v81_23_abl"
os.makedirs(DST, exist_ok=True)
AUD = r"D:\Deepseek_herness\2026-08\diarization\Audio"
FILES = [
    ("Device116", "Call20260718210228.wav"), ("Device116", "Call20260718210454.wav"),
    ("Device116", "Call20260718210657.wav"), ("Device116", "Call20260718210836.wav"),
    ("Device116", "Call20260718210948.wav"), ("Device116", "Call20260718211031.wav"),
    ("Device118", "Note20260803111143.wav"), ("Device118", "Note20260803111436.wav"),
    ("Device118", "Note20260803111605.wav"), ("Device118", "Note20260803111741.wav"),
    ("Device118", "Note20260803111855.wav"), ("Device118", "Note20260803111943.wav"),
    ("Device116", "Call20260718211423.wav"), ("Device116", "Call20260718211506.wav"),
    ("Device116", "Call20260718211745.wav"), ("Device116", "Call20260718211842.wav"),
    ("Device116", "Call20260718211938.wav"), ("Device118", "Note20260803112220.wav"),
    ("Device118", "Note20260803112329.wav"), ("Device118", "Note20260803112415.wav"),
    ("Device118", "Note20260803112655.wav"), ("Device118", "Note20260803112752.wav"),
    ("Device118", "Note20260803112841.wav"),
]
for i, (dev, fn) in enumerate(FILES, 1):
    base = fn.rsplit(".", 1)[0]
    dst = os.path.join(DST, base + ".json")
    if os.path.exists(dst):
        print(f"[{i:2d}/23] {base} 已存在，跳过", flush=True); continue
    data = open(os.path.join(AUD, dev, fn), "rb").read()
    url = ("http://127.0.0.1:8735/api/compare?lanes=v7&denoise=auto&name=" + fn)
    req = urllib.request.Request(url, data=data, method="POST")
    t0 = time.time(); res = None
    with urllib.request.urlopen(req, timeout=1800) as fh:
        for line in fh:
            try: d = json.loads(line)
            except Exception: continue
            if d.get("t") == "done": res = d["result"]
            elif d.get("t") == "error": print("  ERR", d); break
    if res is None:
        print(f"[{i:2d}/23] {base} 无结果", flush=True); continue
    json.dump(res, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    dg = res["diag"]
    print(f"[{i:2d}/23] {base:26s} seg={dg['n_segments']:3d} clu={dg['n_clusters']} "
          f"verdict={res['verdict']:9s} {time.time()-t0:5.1f}s", flush=True)
print("完成 ->", DST)
