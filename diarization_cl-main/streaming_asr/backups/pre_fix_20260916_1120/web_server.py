# -*- coding: utf-8 -*-
"""8734 测试台的 Claude-only 副本，监听 http://127.0.0.1:8735/."""

import json
import os
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
MODEL_ROOT = r"C:\Users\EDY\.cache\modelscope\models"
VAD_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")
SPK_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_campplus_sv_zh-cn_16k-common", "snapshots", "master")
SEACO_LOCAL = os.path.join(MODEL_ROOT, "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "snapshots", "master")
os.makedirs(UPLOAD, exist_ok=True)

_models = {}
_model_lock = threading.Lock()


def select_lanes(query):
    """Resolve requested lanes in the fixed v7/v8/v9 display order."""
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


def get_models(progress=None):
    def report(message):
        if progress:
            progress(message)

    with _model_lock:
        if "vad" not in _models or "spk" not in _models:
            from funasr import AutoModel

            report("正在加载 VAD + CAM++ 声纹模型（首次约 30~60 秒）…")
            options = {"disable_update": True, "disable_pbar": True}
            _models["vad"] = AutoModel(model=VAD_LOCAL, **options)
            _models["spk"] = AutoModel(model=SPK_LOCAL, **options)
        if "asr" not in _models:
            from funasr import AutoModel
            from funasr_backend import load_funasr, wrap_funasr

            try:
                report("正在加载 Fun-ASR-Nano 转写模型（约 30 秒）…")
                _models["asr"] = wrap_funasr(load_funasr())
            except Exception as exc:
                report(f"Fun-ASR-Nano 加载失败（{type(exc).__name__}），回退 SEACO …")
                _models["asr"] = AutoModel(model=SEACO_LOCAL, disable_update=True, disable_pbar=True)
    return _models


def normalize_audio(source, destination):
    subprocess.run(
        ["ffmpeg", "-y", "-i", source, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", destination],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return destination


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
            self._send_chunk(json.dumps(payload, ensure_ascii=False))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/lanes":
            from claude_a2 import visible_versions

            lanes = [{"key": key, "label": value["label"], "kind": "claude"}
                     for key, value in visible_versions().items()]
            self._json({"lanes": lanes}, 200)
            return
        if parsed.path in ("/", "/index.html"):
            self._serve_file(os.path.join(HERE, "diar_web.html"), "text/html; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
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

        try:
            progress("音频转码中（ffmpeg → 16kHz 单声道）…")
            normalize_audio(raw, wav)
            models = get_models(progress)
            from env_detect import detect_env

            env_diag = detect_env(wav, models["vad"])
            progress(f"环境检测：噪声底 {env_diag['noise_db']} dB → "
                     f"判定『{env_diag['verdict']}』…")
            should_denoise = denoise_mode == "on" or (
                denoise_mode == "auto" and env_diag["verdict"] == "嘈杂"
            )
            wav_effective = wav
            denoised = False
            if should_denoise:
                progress("执行 FRCRN 降噪前置…")
                try:
                    from denoise_frcrn import denoise_file

                    wav_effective = os.path.join(UPLOAD, stamp + "_frcrn.wav")
                    cleanup_paths.append(wav_effective)
                    denoise_file(wav, wav_effective)
                    denoised = True
                    progress("降噪完成")
                except Exception as exc:
                    progress(f"降噪失败，回退原始音频（{type(exc).__name__}）")
                    wav_effective = wav
            from claude_a2 import run_claude_lane

            for lane, label in lanes:
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


if __name__ == "__main__":
    print(f"Claude 说话人分离测试台: http://127.0.0.1:{PORT}/", flush=True)
    print("模型在首次请求时加载，进度会实时推送到页面。", flush=True)
    QuietThreadingServer.allow_reuse_address = False
    try:
        QuietThreadingServer(("127.0.0.1", PORT), Handler).serve_forever()
    except OSError as exc:
        print(f"[已退出] 端口 {PORT} 被占用：{exc}", flush=True)
        sys.exit(1)
