# -*- coding: utf-8 -*-
"""8734 测试台的 Claude-only 副本，监听 http://127.0.0.1:8735/."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

PORT = 8735
UPLOAD = os.path.join(HERE, "output", "web_uploads")
DENOISE_CACHE = os.path.join(HERE, "output", "denoise_cache")
# Bump when FRCRN weights or denoise_frcrn's signal path changes.  The model
# has small GPU run-to-run differences, so an input must reuse its first result.
DENOISE_CACHE_VERSION = "frcrn-chunk15-overlap0.5-v1"
MODEL_ROOT = r"C:\Users\EDY\.cache\modelscope\models"
# VAD must be the tuned local copy (speech_noise_thres=0.7); the shared cache is 0.6 and
# merges this audio into one continuous segment (5 turns -> 3, not comparable with 8734).
VAD_LOCAL = os.path.join(os.path.dirname(HERE), "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")
SPK_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_campplus_sv_zh-cn_16k-common", "snapshots", "master")
SEACO_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "snapshots", "master")
STREAM_ASR_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "snapshots", "v2.0.4")
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(DENOISE_CACHE, exist_ok=True)


def _scalar(obj):
    """numpy scalar -> native python type.

    Lane results may carry numpy scalars (bool_/float32/int64...); a single one
    used to kill the whole stream ("Object of type bool_ is not JSON
    serializable"). Guarding here covers every current and future lane.
    """
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(payload):
    return json.dumps(payload, ensure_ascii=False, default=_scalar)

_models = {}
_model_lock = threading.Lock()
_denoise_lock = threading.Lock()
# 2026-09-17：页面上"中止本次"用。协作式取消——服务端在**泳道边界**检查（不改 Claude 侧算法
# 模块，所以做不到阶段内中断）；不挂起线程（挂起 CUDA 推理会死锁）。
_cancel = threading.Event()


def request_cancel():
    _cancel.set()
    return {"ok": True, "note": "已请求中止：当前泳道跑完后停止"}


def preload_models(progress=None):
    """预加载模型（页面「预加载模型」按钮）：省掉下一次运行的约 90 秒等待。"""
    started = time.time()
    get_models(progress, need_stream=True, need_offline=True)
    return {"ok": True, "seconds": round(time.time() - started, 1),
            "models": sorted(_models.keys()), "vram": vram_stats().get("vram")}

_auto_model_cls = None


def _no_wmi(*args):
    # Local Python 3.12 WMI calls hang; platform already handles OSError
    # by falling back to Windows APIs. Do not change the system installation.
    raise OSError("WMI unavailable in this service")


def _get_auto_model():
    global _auto_model_cls
    if _auto_model_cls is None:
        if sys.platform == "win32" and hasattr(platform, "_wmi_query"):
            platform._wmi_query = _no_wmi
        from funasr import AutoModel

        _auto_model_cls = AutoModel
    return _auto_model_cls


def vram_stats():
    """当前进程的显存/模型状态（torch 没加载过就不 import，免得为了看一眼显存先等几秒）。"""
    out = {"models": sorted(_models.keys()), "torch_loaded": "torch" in sys.modules}
    if "torch" not in sys.modules:
        out["vram"] = None
        return out
    try:
        import torch

        if not torch.cuda.is_available():
            out["vram"] = None
            return out
        free, total = torch.cuda.mem_get_info()
        out["vram"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20, 1),
            "device_free_mb": round(free / 2 ** 20, 1),
            "device_total_mb": round(total / 2 ** 20, 1),
        }
    except Exception as exc:  # pragma: no cover - 诊断用
        out["vram"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def unload_models():
    """手动释放：丢掉模型引用 + gc + empty_cache。下次请求会自动重新加载（约 90 秒）。

    模型是模块级常驻（`_models`），浏览器刷新/关页面都不会释放；本机 8734（DSH 台）
    与 8735（本台）各持一份时会抢显存。释放由用户点页面按钮触发，
    不做空闲自动卸载——重新加载要 ~90s，值不值得由用户决定。"""
    import gc

    before = vram_stats()
    with _model_lock:
        count = len(_models)
        _models.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    return {"ok": True, "unloaded": count, "before": before, "after": vram_stats()}


def select_lanes(query):
    """Resolve requested lanes in the fixed display order."""
    from claude_a2 import visible_versions

    versions = visible_versions()
    raw = query.get("lanes", [""])[0]
    requested = [value.strip() for value in raw.split(",") if value.strip()]
    if not requested:
        requested = list(versions)
    unknown = [key for key in requested if key not in versions]
    if unknown:
        raise ValueError(f"不支持的方法：{','.join(unknown)}")
    selected = set(requested)
    return [(key, value["label"]) for key, value in versions.items() if key in selected]


def select_denoise_mode(query):
    mode = query.get("denoise", ["auto"])[0]
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"denoise 只支持 auto|on|off（收到 {mode}）")
    return mode


def get_models(progress=None, need_stream=False, need_offline=True):
    def report(message):
        print(f"[models] {message}", flush=True)
        if progress:
            progress(message)

    report("正在准备模型…")
    with _model_lock:
        if "vad" not in _models or "spk" not in _models:
            report("正在导入 FunASR…")
            AutoModel = _get_auto_model()

            report("正在加载 VAD + CAM++ 声纹模型（首次约 30~60 秒）…")
            options = {"disable_update": True, "disable_pbar": True}
            _models["vad"] = AutoModel(model=VAD_LOCAL, **options)
            _models["spk"] = AutoModel(model=SPK_LOCAL, **options)
        if need_offline and "asr" not in _models:
            AutoModel = _get_auto_model()
            from funasr_backend import load_funasr, wrap_funasr

            try:
                report("正在加载 Fun-ASR-Nano 转写模型（约 30 秒）…")
                _models["asr"] = wrap_funasr(load_funasr())
            except Exception as exc:
                report(f"Fun-ASR-Nano 加载失败（{type(exc).__name__}），回退 SEACO …")
                _models["asr"] = AutoModel(model=SEACO_LOCAL, disable_update=True, disable_pbar=True)
        if need_stream and "stream_asr" not in _models:
            AutoModel = _get_auto_model()

            report("正在加载 Paraformer-Streaming（CPU）…")
            if not os.path.isdir(STREAM_ASR_LOCAL):
                raise FileNotFoundError(f"缺少 Paraformer-Streaming: {STREAM_ASR_LOCAL}")
            _models["stream_asr"] = AutoModel(
                model=STREAM_ASR_LOCAL,
                device="cpu",
                disable_update=True,
                disable_pbar=True,
            )
        return dict(_models)


def normalize_audio(source, destination):
    subprocess.run(
        ["ffmpeg", "-y", "-i", source, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", destination],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_denoise(source, denoise_func, cache_dir=DENOISE_CACHE):
    """Freeze the first FRCRN result for a normalized input, then reuse it."""
    source_hash = _file_sha256(source)
    key = hashlib.sha256(f"{DENOISE_CACHE_VERSION}:{source_hash}".encode("ascii")).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, key + ".wav")
    if os.path.isfile(target):
        return target, True, source_hash
    with _denoise_lock:
        if os.path.isfile(target):
            return target, True, source_hash
        temporary = target + "." + uuid.uuid4().hex + ".tmp.wav"
        try:
            denoise_func(source, temporary)
            os.replace(temporary, target)
        finally:
            try:
                os.remove(temporary)
            except OSError:
                pass
    return target, False, source_hash


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_chunk(self, line):
        data = (line + "\n").encode("utf-8")
        try:
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _emit(self, payload):
        if not getattr(self, "_dead", False):
            self._send_chunk(_dumps(payload))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/lanes":
            from claude_a2 import visible_versions

            lanes = [{"key": key, "label": value["label"], "kind": "claude"}
                     for key, value in visible_versions().items()]
            self._json({"lanes": lanes}, 200)
            return
        if parsed.path == "/api/gpu":
            # 手动释放显存用：看当前进程持有多少显存 / 模型是否已加载
            self._json(vram_stats(), 200)
            return
        if parsed.path in ("/", "/index.html"):
            self._serve_file(os.path.join(HERE, "diar_web.html"), "text/html; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/unload":
            # 手动释放显存（页面"释放显存"按钮）；幂等，没加载过也返回 ok
            self._json(unload_models(), 200)
            return
        if parsed.path == "/api/preload":
            # 预加载模型（约 90 秒）——省掉下一次运行的等待
            self._json(preload_models(), 200)
            return
        if parsed.path == "/api/cancel":
            # 协作式中止：当前泳道结束后停止（不挂起线程）
            self._json(request_cancel(), 200)
            return
        if parsed.path == "/api/llm":
            self._handle_llm()
            return
        if parsed.path == "/api/compare":
            self._handle_compare(parsed)
            return
        self.send_error(404)

    def _handle_compare(self, parsed):
        query = parse_qs(parsed.query)
        try:
            lanes = select_lanes(query)
            denoise_mode = select_denoise_mode(query)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
            return

        name = query.get("name", ["upload.wav"])[0]
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        if not data:
            self._json({"error": "empty upload"}, 400)
            return

        stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        extension = os.path.splitext(name)[1].lower() or ".bin"
        raw = os.path.join(UPLOAD, stamp + extension)
        wav = os.path.join(UPLOAD, stamp + "_16k.wav")
        cleanup_paths = [raw, wav]
        with open(raw, "wb") as file:
            file.write(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def progress(lane=None, message=None):
            if message is None:
                message, lane = lane, None
            self._emit({"t": "progress", "lane": lane, "msg": message})

        _cancel.clear()          # 每次新请求重置中止标志
        try:
            progress("音频转码中（ffmpeg → 16kHz 单声道）…")
            normalize_audio(raw, wav)
            models = get_models(
                progress,
                need_stream=any(lane == "v10" for lane, _ in lanes),
                need_offline=True,
            )
            from env_detect import detect_env

            env_diag = detect_env(wav, models["vad"])
            progress(f"环境检测：噪声底 {env_diag['noise_db']} dB → "
                     f"判定『{env_diag['verdict']}』…")
            should_denoise = denoise_mode == "on" or (
                denoise_mode == "auto" and env_diag["verdict"] == "嘈杂"
            )
            wav_effective = wav
            denoised = False
            denoise_cache_hit = False
            denoise_input_sha256 = None
            if should_denoise:
                try:
                    from denoise_frcrn import denoise_file

                    progress("执行 FRCRN 降噪前置…")
                    wav_effective, denoise_cache_hit, denoise_input_sha256 = cached_denoise(
                        wav, denoise_file)
                    denoised = True
                    progress("复用已冻结 FRCRN 结果" if denoise_cache_hit else "降噪完成并已冻结")
                except Exception as exc:
                    progress(f"降噪失败，回退原始音频（{type(exc).__name__}）")
                    wav_effective = wav
            from claude_a2 import run_claude_lane

            for lane, label in lanes:
                if _cancel.is_set():
                    progress("已中止：停止后续泳道")
                    self._emit({"t": "cancelled", "lane": lane,
                                "msg": "用户中止（当前泳道未跑）"})
                    break
                progress(lane, f"{label}：推理中…")
                started = time.time()
                try:
                    result = run_claude_lane(
                        wav_effective,
                        models,
                        lane,
                        progress=lambda message, key=lane: progress(key, message),
                    )
                    result["file"] = name
                    result["elapsed"] = round(time.time() - started, 1)
                    result["env"] = env_diag
                    result["denoise_mode"] = denoise_mode
                    result["denoised"] = denoised
                    result["denoise_cache_hit"] = denoise_cache_hit
                    result["denoise_input_sha256"] = denoise_input_sha256
                    self._emit({"t": "done", "lane": lane, "result": result})
                except Exception as exc:
                    self._emit({"t": "error", "lane": lane,
                                "msg": f"{type(exc).__name__}: {exc}"})
            self._emit({"t": "finish"})
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._emit({"t": "error", "msg": f"{type(exc).__name__}: {exc}"})
        finally:
            for path in cleanup_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass

    def _handle_llm(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            result = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self._json({"error": "bad body"}, 400)
            return
        if "+llm" in str(result.get("method", "")):
            self._json({"error": "该结果已经做过 LLM 校正；请从原始结果重新开始。"}, 409)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self._emit({"t": "progress", "msg": "调用 DeepSeek 进行后校正…"})
            from llm_correct import correct

            sentences, names, _ = correct(result)
            output = {key: value for key, value in result.items()
                      if key != "sentence_info"}
            output["sentence_info"] = sentences
            output["method"] = str(result.get("method", "claude-a2")) + "+llm"
            output["names"] = names
            self._emit({"t": "done", "result": output})
        except Exception as exc:
            self._emit({"t": "error", "msg": f"LLM: {type(exc).__name__}: {exc}"})
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                pass

    def _serve_file(self, path, content_type):
        with open(path, "rb") as file:
            body = file.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class QuietThreadingServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve():
    print("正在主线程预导入 FunASR…", flush=True)
    _get_auto_model()
    print(f"Claude 说话人分离测试台: http://127.0.0.1:{PORT}/", flush=True)
    print("模型在首次请求时加载，进度会实时推送到页面。", flush=True)
    QuietThreadingServer.allow_reuse_address = False
    try:
        QuietThreadingServer(("127.0.0.1", PORT), Handler).serve_forever()
    except OSError as exc:
        print(f"[已退出] 端口 {PORT} 被占用：{exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    serve()
