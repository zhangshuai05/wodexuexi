"""Summarize saved comparisons without loading speech models."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from compare_c1 import BASE, score


def main():
    gt = json.loads((BASE / "eval_dsh/ground_truth.json").read_text(encoding="utf-8"))
    versions = ["web_old", "offline_old", "final"]
    report = {"metric_note": "Text similarity >=0.35; speaker_matched uses one-to-one speaker label mapping. This is a text proxy, not DER. Ground-truth turns can match the same output segment.",
              "raw": [], "enhanced": [], "groups": {}}
    for i, truth in enumerate(gt):
        group = ("noisy" if i < 12 else "quiet") + " " + truth["distance"]
        for suffix, category in [("", "raw"), ("_enhanced", "enhanced")]:
            row = {"file": truth["file"], "group": group}
            for version in versions:
                path = BASE / "diar_result" / ("c1_" + version + suffix) / (Path(truth["file"]).stem + "_diar.json")
                if path.exists():
                    sentences = json.loads(path.read_text(encoding="utf-8"))["sentence_info"]
                    row[version] = score(truth, sentences)
            if len(row) != len(versions) + 2:
                continue
            final = BASE / "diar_result" / ("c1_final" + suffix) / path.name
            embedded = BASE / "diar_result" / ("c1_embedded" + suffix) / path.name
            assert json.loads(final.read_text(encoding="utf-8"))["sentence_info"] == json.loads(embedded.read_text(encoding="utf-8"))["sentence_info"], truth["file"]
            report[category].append(row)
            if category == "raw":
                report["groups"].setdefault(group, {v: [0, 0, 0] for v in versions})
                for version in versions:
                    s = row[version]
                    counts = [s["matched"], s["speaker_matched"], s["gt"]]
                    report["groups"][group][version] = [a + b for a, b in zip(report["groups"][group][version], counts)]
    path = BASE / "diar_result/c1_comparison_20260908.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("matched/speaker_matched/gt; parity verified for all saved cases")
    for group, values in report["groups"].items():
        print(group, values)
    print("report:", path)


if __name__ == "__main__":
    main()
