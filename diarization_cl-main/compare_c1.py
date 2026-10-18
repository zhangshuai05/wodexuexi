"""Reproducible c1 experiments; cached model calls, unchanged source audio."""
import argparse
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from local_eval import norm, best_match

BASE = Path(__file__).resolve().parent
CACHE = BASE / "diar_result" / "c1_experiment_cache"
MODEL_ROOT = Path(r"C:\Users\EDY\.cache\modelscope\models")


UNKNOWN = "?"  # 无单一身份判断的输出句(DSH 重叠段在严格口径下)


def labels_of(sentence, overlap_credit):
    """一句输出对应的说话人标签集合。
    DSH 的重叠段没有 spk，而是 spks=[0,1] + overlap=True。
    overlap_credit=True 时重叠段算“这几个人都在说”，真值身份命中其中之一即算对；
    False 时记 UNKNOWN，且 UNKNOWN 不进入标签池——否则它会被当成一个额外的
    说话人标签参与排列，白送一个真人的分。"""
    if "spk" in sentence:
        return {str(sentence["spk"])}
    if overlap_credit and sentence.get("spks"):
        return {str(k) for k in sentence["spks"]}
    return {UNKNOWN}


def score(gt, sentences, overlap_credit=False):
    cands = [(i, norm(s.get("text", ""))) for i, s in enumerate(sentences)]
    pairs = []
    for turn in gt["turns"]:
        match = best_match(turn["text"], cands)
        if match and match[0] is not None and match[1] >= 0.35:
            pairs.append((turn["spk"], labels_of(sentences[match[0]], overlap_credit)))
    labels = sorted({l for s in sentences for l in labels_of(s, overlap_credit)} - {UNKNOWN})
    labels += ["missing"] * max(0, len(gt["speakers"]) - len(labels))
    correct = max((sum(dict(zip(gt["speakers"], perm))[p] in s for p, s in pairs)
                   for perm in itertools.permutations(labels, len(gt["speakers"]))), default=0)
    return {"matched": len(pairs), "speaker_matched": correct, "gt": gt["n_turns"],
            "segments": len(sentences),
            "longest": round(max((s["end"] - s["start"] for s in sentences), default=0), 2)}


class CachedModel:
    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        self.folder = CACHE / kind
        self.folder.mkdir(parents=True, exist_ok=True)

    def generate(self, input, **kwargs):
        import torch
        data = np.ascontiguousarray(input)
        key = hashlib.sha256(str(data.dtype).encode() + data.tobytes()).hexdigest()
        path = self.folder / (key + (".npy" if self.kind == "spk" else ".json"))
        if path.exists():
            if self.kind == "spk":
                return [{"spk_embedding": torch.from_numpy(np.load(path))}]
            return json.loads(path.read_text(encoding="utf-8"))
        result = self.model.generate(input=input, **kwargs)
        if self.kind == "spk":
            np.save(path, result[0]["spk_embedding"].detach().cpu().numpy())
        else:
            path.write_text(json.dumps([{"text": result[0].get("text", "")}]), encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--variants", nargs="+", default=["web_old", "offline_old", "short"])
    parser.add_argument("--audio-dir", type=Path)
    args = parser.parse_args()
    from funasr import AutoModel
    old = importlib.import_module("archive.legacy_versions.a2_v2_before_20260908")
    new = importlib.import_module("a2_v2")
    vad = AutoModel(model=old.VAD, disable_update=True, disable_pbar=True)
    spk = CachedModel(AutoModel(model=str(MODEL_ROOT / "iic--speech_campplus_sv_zh-cn_16k-common" / "snapshots/master"),
                               disable_update=True, disable_pbar=True), "spk")
    asr = CachedModel(AutoModel(model=str(MODEL_ROOT / "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" / "snapshots/master"),
                               disable_update=True, disable_pbar=True, vad_model=None), "asr")
    variants = {
        "web_old": (old.a2_segments_v2, {}),
        "offline_old": (old.a2_segments_v2, {"merge": False}),
        "short": (old.a2_segments_v2, {"merge": False, "win_s": 0.6, "hop_s": 0.15}),
        "final": (new.a2_segments_v2, {"merge": False}),
        "embedded": (importlib.import_module("claude_a2").a2_segments, {"merge": False}),
    }
    gt = json.loads((BASE / "eval_dsh/ground_truth.json").read_text(encoding="utf-8"))
    totals = {v: [0, 0, 0] for v in args.variants}
    for g in gt:
        if args.files and not any(f in g["file"] for f in args.files):
            continue
        wav = next((BASE / "Audio").glob("*/" + g["file"]))
        if args.audio_dir:
            wav = args.audio_dir / g["file"]
            if not wav.exists():
                continue
        x, sr = sf.read(wav)
        if x.ndim > 1:
            x = x[:, 0]
        for variant in args.variants:
            folder = BASE / "diar_result" / ("c1_" + variant + ("_enhanced" if args.audio_dir else ""))
            folder.mkdir(exist_ok=True)
            output = folder / (wav.stem + "_diar.json")
            start = time.monotonic()
            fn, kwargs = variants[variant]
            source = Path(importlib.import_module(fn.__module__).__file__).read_bytes()
            signature = hashlib.sha256(source + wav.read_bytes() + json.dumps(kwargs, sort_keys=True).encode()).hexdigest()
            saved = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
            if saved.get("signature") == signature:
                sentences = saved["sentence_info"]
            else:
                segments = fn(str(wav), spk, vad, **kwargs)
                sentences = []
                for seg in segments:
                    chunk = x[int(seg["start"] * sr):int(seg["end"] * sr)]
                    text = asr.generate(input=chunk)[0].get("text", "") if len(chunk) >= sr * 0.3 else ""
                    sentences.append({**seg, "text": text})
                output.write_text(json.dumps({"signature": signature, "sentence_info": sentences}, ensure_ascii=False, indent=2), encoding="utf-8")
            result = score(g, sentences)
            totals[variant] = [a + b for a, b in zip(totals[variant], [result["matched"], result["speaker_matched"], result["gt"]])]
            print(g["file"], variant, result, "seconds=", round(time.monotonic() - start, 1), flush=True)
    print("TOTAL matched/speaker_matched/gt:", totals, flush=True)


if __name__ == "__main__":
    main()
