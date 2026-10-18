import glob, json, os, sys, time
sys.path.insert(0, os.getcwd())
from web_server import get_models
from claude_a2 import run_claude_lane
from v11_eval import evaluate

models = get_models(lambda m: None, need_stream=False, need_offline=True)
base = r"C:\Users\EDY\Desktop\new\Audio\Merged3Spk"
root = os.path.join(base, "_old_variants")
rows = []
for gtf in sorted(glob.glob(os.path.join(root, "*.gt.json"))):
    gt = json.load(open(gtf, encoding="utf-8"))
    wav = gtf.replace(".gt.json", ".wav")
    if not os.path.exists(wav):
        print("skip (no wav):", os.path.basename(gtf), flush=True)
        continue
    entry = {"file": os.path.basename(gt["name"]),
             "turns": len([t for t in gt["turns"] if t["end"] > t["start"]]),
             "gt_spk": len({t["spk"] for t in gt["turns"]})}
    for lane in ["v7", "v12"]:
        t0 = time.time()
        try:
            r = run_claude_lane(wav, models, lane, progress=None)
            rep = evaluate(gt, r)
            entry[lane] = {"acc": rep["scopes"]["all"]["turns"]["accuracy"],
                           "k": r.get("diag", {}).get("n_speakers"),
                           "elapsed": round(time.time() - t0, 1),
                           "path": r.get("diag", {}).get("cluster_level", ""),
                           "segs": len(r["sentence_info"])}
        except Exception as exc:
            entry[lane] = {"acc": None, "error": f"{type(exc).__name__}: {exc}"}
    rows.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)

print("\n=== SUMMARY ===")
scored = [r for r in rows if r.get("v7", {}).get("acc") is not None and r.get("v12", {}).get("acc") is not None]
if scored:
    tw = sum(r["turns"] for r in scored)
    for lane in ["v7", "v12"]:
        w = sum(r[lane]["acc"] * r["turns"] for r in scored) / tw
        print("%s: weighted turn_acc=%.3f  total_elapsed=%.1fs" % (lane, w, sum(r[lane]["elapsed"] for r in scored)))
json.dump(rows, open(os.path.join("output", "_audit_v7_v12_variants.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
