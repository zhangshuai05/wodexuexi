"""Smoke test the deployed local comparison endpoint."""
import json
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_c1 import BASE, score


def main():
    gt = {g["file"]: g for g in json.loads((BASE / "eval_dsh/ground_truth.json").read_text(encoding="utf-8"))}
    cases = [("Note20260803112329.wav", "off", "cur,snap,c1"),
             ("Note20260803112752.wav", "off", "c1"),
             ("Note20260803111855.wav", "auto", "c1")]
    for name, denoise, lanes in cases:
        wav = next((BASE / "Audio").glob("*/" + name))
        url = "http://127.0.0.1:8734/api/compare?" + urlencode({"name": name, "lanes": lanes, "denoise": denoise})
        request = Request(url, data=wav.read_bytes(), headers={"Content-Type": "audio/wav"})
        results = {}
        finished = False
        with urlopen(request, timeout=240) as response:
            for line in response:
                event = json.loads(line)
                if event["t"] == "error":
                    raise RuntimeError(event)
                if event["t"] == "progress":
                    print(name, event.get("lane", "all"), event.get("msg", ""), flush=True)
                if event["t"] == "done":
                    results[event["lane"]] = event["result"]
                if event["t"] == "finish":
                    finished = True
        assert finished and set(results) == set(lanes.split(",")), results.keys()
        assert results["c1"]["method"] == "claude-a2-v4-c1", results["c1"].get("method")
        assert results["c1"]["sentence_info"]
        path = BASE / "diar_result" / ("c1_api_" + wav.stem + ".json")
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        for lane, result in results.items():
            print("LIVE", name, lane, score(gt[name], result["sentence_info"]), flush=True)
    print("PASS: deployed c1 v4, raw and auto-denoised inputs, all three lanes", flush=True)


if __name__ == "__main__":
    main()
