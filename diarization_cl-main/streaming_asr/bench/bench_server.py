# -*- coding: utf-8 -*-
"""AISHELL-4 分离测试台：选场次 → 跑 v12 → 与 RTTM 真值并排看。

与 8735（`web_server.py`）的区别：那个跑的是业务上传音频 + v7~v12 全泳道 + ASR；
这个只跑 AISHELL-4 上已有的场次，**只做分离**（不加载转写模型），
并且**带真值对照**——DER、映射表、两条时间轴叠在同一把尺子上。

跑法：python bench_server.py   →  http://127.0.0.1:8736/
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ASR = HERE.parent
ROOT = ASR.parent
sys.path.insert(0, str(ASR))
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("AISHELL4_TEST", r"D:\yinp\AISHELL-4\data\test"))
CLIP_DIR = ASR / "output" / "bench_clips"
PORT = 8736
DEFAULT_SECONDS = 600.0
MAX_SECONDS = 3600.0
COLLAR = 0.25

_model_lock = threading.Lock()
_clip_lock = threading.Lock()


def _scalar(obj):
    """numpy 标量 → 原生类型；漏一个就会整条流 500。"""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(payload):
    return json.dumps(payload, ensure_ascii=False, default=_scalar)


def meetings():
    """AISHELL-4 test 的 20 场，带真值人数与时长（时长走 flac header，不解码）。"""
    import soundfile as sf

    from v12_aishell4 import DEV_GROUPS, HOLDOUT_GROUPS, meeting_group
    from der import read_rttm

    out = []
    for rttm in sorted((DATA / "TextGrid").glob("*.rttm")):
        meeting = rttm.stem
        group = meeting_group(meeting)
        split = ("dev" if group in DEV_GROUPS else
                 "holdout" if group in HOLDOUT_GROUPS else "其他")
        info = sf.info(DATA / "wav" / f"{meeting}.flac")
        out.append({
            "id": meeting,
            "gt_speakers": len({s["speaker"] for s in read_rttm(rttm)}),
            "duration": round(info.frames / info.samplerate, 1),
            "split": split,
        })
    return out


def clip_path(meeting, seconds):
    """给浏览器播的音频：和算法读的**同一条通道**（ch0→单声道），缓存一次。"""
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    target = CLIP_DIR / f"{meeting}_{seconds:g}.wav"
    if target.exists():
        return target
    with _clip_lock:
        if target.exists():
            return target
        temporary = target.with_suffix(".tmp.wav")
        subprocess.run(
            # pan=c0=c0 取第 0 通道，不是 -ac 1（8 通道下混到单声道是另一条信号）。
            ["ffmpeg", "-y", "-t", f"{seconds:g}", "-i", str(DATA / "wav" / f"{meeting}.flac"),
             "-af", "pan=mono|c0=c0", "-ar", "16000", str(temporary)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temporary, target)
    return target


def pairs(predicted, reference):
    """每个预测簇最像哪个真值人：按区间重叠秒数取最大。UI 的配色和映射表都用它。"""
    out = []
    for label in sorted({s["speaker"] for s in predicted}):
        p = [s for s in predicted if s["speaker"] == label]
        p_len = sum(s["end"] - s["start"] for s in p)
        best, best_overlap = None, 0.0
        for ref in {s["speaker"] for s in reference}:
            overlap = sum(max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
                          for a in p for b in reference if b["speaker"] == ref)
            if overlap > best_overlap:
                best, best_overlap = ref, overlap
        out.append({"pred": label, "ref": best, "overlap": round(best_overlap, 1),
                    "pred_seconds": round(p_len, 1),
                    "share": round(best_overlap / p_len, 3) if p_len else 0.0})
    return out


def get_models(progress):
    """VAD + CAM++ 上 GPU，流式转写上 CPU（两条腿互不抢显存）。首次约 30~60 秒。"""
    with _model_lock:
        from web_server import get_models as _get

        return _get(progress, need_stream=True, need_offline=False)


def transcribe(meeting, seconds, models, emit, progress):
    """边跑边上屏的流式转写（FunASR paraformer-streaming，CPU，不回放等待）。

    与分离结果**不做对齐**：它只是让等待期间有东西看，不代表谁说了哪句。
    """
    from v10_asr_probe import stream_replay
    from v12_aishell4 import read_channel0

    audio = read_channel0(DATA / "wav" / f"{meeting}.flac", 0.0, seconds)
    progress(f"流式转写 {len(audio) / 16000:.0f} 秒音频（CPU，逐块上屏）…")
    result = stream_replay(
        models["stream_asr"], audio, 16000, realtime=False,
        on_text=lambda text, ms: emit({"t": "asr", "ms": ms, "text": text}))
    emit({"t": "asr_done", "text": result["text"],
          "rtf": result["inference_rtf"], "seconds": result["wall_seconds"]})


def run(meeting, seconds, progress, emit):
    import claude_a2_v12 as v12
    from der import der, read_rttm

    reference_all = read_rttm(DATA / "TextGrid" / f"{meeting}.rttm")
    reference = [{"start": s["start"], "end": min(s["end"], seconds), "speaker": s["speaker"]}
                 for s in reference_all if s["start"] < seconds]

    progress("加载模型（VAD + CAM++ 上 GPU，流式转写上 CPU；首次约 30~60 秒）…")
    models = get_models(progress)

    # 转写先开工：分离在 GPU 上跑，转写在 CPU 上跑，两条腿并行；用户在这个空档里看字。
    worker = threading.Thread(target=_safe_transcribe,
                              args=(meeting, seconds, models, emit, progress), daemon=True)
    worker.start()

    progress(f"读取 {meeting}.flac 前 {seconds:g} 秒（8 通道取 ch0）…")
    started = time.time()
    result = v12.diarize(str(DATA / "wav" / f"{meeting}.flac"), models,
                         progress=progress, max_seconds=seconds)
    elapsed = time.time() - started

    if result["insufficient"]:
        worker.join()
        return {"meeting": meeting, "seconds": seconds, "elapsed": round(elapsed, 1),
                "level": "insufficient", "k": 0, "gt_k": len({s["speaker"] for s in reference}),
                "n_windows": result["n_windows"], "segments": [], "reference": reference,
                "pairs": [], "score": None,
                "note": "窗数不足，无法做多人聚类"}

    predicted = [{"start": s["start"], "end": s["end"], "speaker": f"#{int(s['spk'])}"}
                 for s in result["segments"]]
    score = der(reference, predicted, collar=COLLAR)
    progress(f"真值对照：DER {score['der'] * 100:.2f}%（collar {COLLAR} 秒）")

    payload = {
        "meeting": meeting, "seconds": seconds, "elapsed": round(elapsed, 1),
        "level": ("refined" if result["refined"] else
                  "rescued" if result["rescued"] else "2s-only"),
        "k": result["k"], "gt_k": len({s["speaker"] for s in reference}),
        "n_windows": result["n_windows"],
        "window_seconds": result["window_seconds"], "hop_seconds": result["hop_seconds"],
        "segments": predicted, "reference": reference,
        "pairs": pairs(predicted, reference),
        "score": score,
    }
    emit({"t": "done", "result": payload})   # 分离结果先上屏，转写继续往下流
    worker.join()
    return payload


def _safe_transcribe(meeting, seconds, models, emit, progress):
    """转写失败不能把分离结果一起带走。"""
    try:
        transcribe(meeting, seconds, models, emit, progress)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        emit({"t": "asr_error", "msg": f"流式转写失败：{type(exc).__name__}: {exc}"})

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    _lock = threading.Lock()   # 分离与转写两条线程共用一个 socket

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._file(HERE / "index.html", "text/html; charset=utf-8")
            elif parsed.path == "/api/meetings":
                self._json({"meetings": meetings()})
            elif parsed.path == "/api/audio":
                self._file(clip_path(query["meeting"][0], float(query["seconds"][0])),
                           "audio/wav")
            else:
                self.send_error(404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        try:
            meeting = query["meeting"][0]
            seconds = min(MAX_SECONDS, max(30.0, float(query["seconds"][0])))
        except (KeyError, ValueError) as exc:
            self._json({"error": f"参数错误：{exc}"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._emit({"t": "progress", "msg": f"场次 {meeting}，前 {seconds:g} 秒"})
            run(meeting, seconds,
                lambda msg: self._emit({"t": "progress", "msg": msg}),
                self._emit)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._emit({"t": "error", "msg": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass

    def _emit(self, payload):
        data = (_dumps(payload) + "\n").encode("utf-8")
        try:
            with self._lock:
                self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
                self.wfile.flush()
        except OSError:
            raise ConnectionAbortedError("client gone")

    def _file(self, path, content_type):
        body = Path(path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=200):
        body = _dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class Server(ThreadingHTTPServer):
    # Windows 上 SO_REUSEADDR 会让第二个实例**静默**绑上同一端口，两个进程分裂流量：
    # 页面来自新进程、/api/run 落到旧进程（旧代码），表现成"功能莫名不生效"。
    # 宁可启动失败，也不撕裂。
    allow_reuse_address = False

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def selfcheck():
    """Ponytail 自检：不加载模型，只查真值/配对/切片这三条离线路径。"""
    from der import read_rttm

    rows = meetings()
    assert len(rows) == 20, f"期望 20 场，得到 {len(rows)}"
    assert all(set(r) == {"id", "gt_speakers", "duration", "split"} for r in rows)
    assert sum(r["split"] == "dev" for r in rows) == 12
    assert sum(r["split"] == "holdout" for r in rows) == 8
    assert all(r["duration"] > 600 for r in rows)

    reference = read_rttm(DATA / "TextGrid" / "L_R003S01C02.rttm")
    # 预测故意 = 真值，pairs 必须把每个簇配到它自己，share = 1.0
    predicted = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
                 for s in reference if s["end"] <= 300]
    table = pairs(predicted, reference)
    assert {row["pred"] for row in table} == {s["speaker"] for s in predicted}
    assert all(row["pred"] == row["ref"] and row["share"] == 1.0 for row in table), table
    # 只按重叠配对：没跟预测重叠过的真值人不能出现在表里
    assert {row["ref"] for row in table} <= {s["speaker"] for s in predicted}
    # 预测里故意砍掉一个人，那个真值人不得再被任何簇匹配到
    dropped = next(iter({s["speaker"] for s in predicted}))
    pruned = pairs([s for s in predicted if s["speaker"] != dropped], reference)
    assert dropped not in {row["ref"] for row in pruned}, pruned
    from der import der

    assert der(predicted, predicted, collar=COLLAR)["der"] == 0.0

    clip = clip_path("L_R003S01C02", 30.0)
    import soundfile as sf

    info = sf.info(clip)
    assert info.channels == 1 and info.samplerate == 16000, info
    assert abs(info.duration - 30.0) < 0.2, info.duration
    print(f"selfcheck ok：{len(rows)} 场 · 配对 {len(table)} 簇 · 切片 {info.duration:.1f}s {info.samplerate}Hz")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        print(f"AISHELL-4 分离测试台: http://127.0.0.1:{PORT}/", flush=True)
        print(f"数据目录 {DATA}（模型在首次运行时加载）", flush=True)
        try:
            Server(("127.0.0.1", PORT), Handler).serve_forever()
        except OSError as exc:
            print(f"[已退出] 端口 {PORT} 被占用（可能上面还有一个实例）：{exc}", flush=True)
            sys.exit(1)
