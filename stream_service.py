# -*- coding: utf-8 -*-
"""会议回放通道 HTTP 服务：GET 页面/健康检查，POST WAV -> NDJSON 流式事件。

路由（规格 §4.4.2）：
    GET  /                   -> diar_web.html（响应中注入音频模拟桥接脚本）
    GET  /meeting_bridge.js  -> application/javascript; charset=utf-8
    GET  /api/health         -> {"ok": true, "busy": <bool>}
    POST /api/stream_meeting?name=<文件名>&fast=1
                             -> application/x-ndjson; charset=utf-8 + chunked

只监听 127.0.0.1。所有 session JSON 写入都经由 SessionStore；本模块只做
HTTP 外壳、上传解码、ffmpeg 转码与事件转发，不重复实现流水线编排。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import time
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import bootstrap

bootstrap.install()

from run_offline import run  # noqa: E402
from session_store import SessionStore, new_session_dir  # noqa: E402

SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = 8736
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
SESSION_ROOT = Path("output/sessions")
INDEX_FILE = Path(__file__).resolve().parent / "diar_web.html"
BRIDGE_FILE = Path(__file__).resolve().parent / "meeting_bridge.js"
BRIDGE_MARKER = b"meeting_bridge.js"
BRIDGE_TAG = b'<script src="/meeting_bridge.js"></script>\n'

# 同一时刻只允许一场会议；第二个请求直接 409。
_SESSION_LOCK = threading.Lock()
# Reset 等待正在运行的流水线完整退出后再让前端开始下一场。
_SESSION_IDLE = threading.Event()
_SESSION_IDLE.set()
_SESSION_STATE_LOCK = threading.Lock()
_RESET_REQUEST_LOCK = threading.Lock()
_RESET_PENDING = threading.Event()
# 用户点“结束会议”时置位；喂流循环每块检查一次，走正常收尾与回填
_CANCEL = threading.Event()
# 重置或浏览器断开时置位；跳过后续回填并尽快释放会议锁。
_RESET = threading.Event()

# 模型单例：首次请求/预加载时装载，之后复用。
_MODELS: Optional[Dict] = None
_MODELS_LOCK = threading.Lock()

# 测试注入点：替换为假流水线即可在无模型环境跑通 HTTP 契约。
_PIPELINE: Callable[..., Dict] = run


class FfmpegMissing(RuntimeError):
    """需要转码但 PATH 中找不到 ffmpeg。"""


def _get_models() -> Dict:
    """模块级模型单例（双重检查锁）。"""
    global _MODELS
    if _MODELS is None:
        with _MODELS_LOCK:
            if _MODELS is None:
                from web_server import get_models
                _MODELS = get_models(need_stream=True, need_offline=False)
    return _MODELS


def _run_pipeline(wav, session_dir, **kwargs) -> Dict:
    """薄包装：测试 monkeypatch `_PIPELINE` 后仍走同一条调用路径。"""
    return _PIPELINE(wav, session_dir, **kwargs)


def _safe_filename(raw: Optional[str]) -> str:
    """只取 basename；客户端字符串永远不能作为路径使用。"""
    name = Path(str(raw or "").replace("\\", "/")).name.strip()
    name = name.replace("\x00", "")
    if name in ("", ".", ".."):
        return "upload.bin"
    # Windows 非法文件名字符统一替换，避免落盘直接失败
    for ch in '<>:"|?*':
        name = name.replace(ch, "_")
    return name


def _extract_multipart(body: bytes, content_type: str) -> Optional[Tuple[str, bytes]]:
    """取第一个带 filename 的 part。使用 email.parser，不用已废弃的 cgi。"""
    header = (f"Content-Type: {content_type}\r\n"
              "MIME-Version: 1.0\r\n\r\n").encode("utf-8", errors="replace")
    try:
        message = BytesParser().parsebytes(header + body)
    except Exception:
        return None
    if not message.is_multipart():
        return None
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        filename = part.get_filename()
        if not filename:
            continue
        return filename, part.get_payload(decode=True) or b""
    return None


def _needs_conversion(src: Path) -> bool:
    """已经是 16 kHz 单声道（soundfile 能识别）则跳过 ffmpeg。"""
    try:
        import soundfile as sf
        info = sf.info(str(src))
    except Exception:
        return True
    return not (info.samplerate == 16000 and info.channels == 1)


def _convert_to_16k_mono(src: Path, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FfmpegMissing("ffmpeg")
    proc = subprocess.run(
        # 多声道取 ch0，不用 -ac 1 下混：8 通道平均会相位抵消，ch0 实测好 6-11 个点
        [ffmpeg, "-y", "-i", str(src), "-af", "pan=mono|c0=c0", "-ar", "16000",
         "-c:a", "pcm_s16le", str(dst)],
        capture_output=True,
    )
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        detail = proc.stderr.decode("utf-8", errors="replace")[-500:].strip()
        raise RuntimeError(f"ffmpeg 转码失败（exit {proc.returncode}）: {detail}")


def _inject_bridge(html: bytes) -> bytes:
    """在 </body> 前注入桥接脚本标签；已存在标记时不重复注入。

    只在响应内存中改写，磁盘上的 diar_web.html 保持不变。
    """
    if BRIDGE_MARKER in html:
        return html
    index = html.rfind(b"</body>")
    if index < 0:
        return html
    return html[:index] + BRIDGE_TAG + html[index:]


def _count_clusters(mapping: Dict) -> int:
    labels = set()
    for row in mapping.values():
        if not isinstance(row, dict):
            continue
        label = row.get("spk_label", row.get("spk"))
        if label is not None:
            labels.add(label)
    return len(labels)


class StreamRequestHandler(BaseHTTPRequestHandler):
    """单请求处理器：GET 静态页/健康检查，POST 流式会议。"""

    protocol_version = "HTTP/1.1"
    server_version = "MeetingStream/0.1"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        # 必须在 BaseHTTPRequestHandler.__init__（会立刻进入 handle 循环）之前初始化
        self._dead = False
        self._session_active = False
        self._write_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    # ---- 普通响应 ----

    def _send_json(self, status: int, obj: Dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _send_html(self) -> None:
        try:
            # 每次请求都读盘，注入只发生在响应副本中
            data = _inject_bridge(INDEX_FILE.read_bytes())
        except OSError:
            self._send_json(404, {"error": "前端文件不存在"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _send_bridge(self) -> None:
        """返回桥接脚本本身；每次请求实时读取磁盘内容。"""
        try:
            data = BRIDGE_FILE.read_bytes()
        except OSError:
            self._send_json(404, {"error": "桥接脚本不存在"})
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
        except OSError:
            self._dead = True

    # ---- chunked NDJSON ----

    def _start_ndjson(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _send_chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _end_chunks_safely(self) -> None:
        try:
            with self._write_lock:
                if not self._dead:
                    self._end_chunks()
        except OSError:
            self._dead = True

    def _emit_event(self, ev: Dict) -> None:
        """写一行 NDJSON 并 flush。客户端断线后置 _dead 并彻底静默。"""
        if self._dead:
            return
        line = (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            # 润色线程、回填心跳线程与主线程都会回调，必须串行写 chunk
            with self._write_lock:
                if self._dead:
                    return
                self._send_chunk(line)
                self.wfile.flush()
        except (BrokenPipeError, OSError):
            self._dead = True
            if self._session_active:
                # 刷新/关闭页面会断开 NDJSON 响应；把它当作重置，
                # 不要让后台继续处理完整段音频或启动说话人回填。
                _RESET.set()
                _CANCEL.set()

    # ---- 路由 ----

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        path = urlsplit(self.path).path
        if path == "/":
            self._send_html()
        elif path == "/meeting_bridge.js":
            self._send_bridge()
        elif path == "/api/health":
            self._send_json(200, {
                "ok": True,
                "busy": _SESSION_LOCK.locked(),
                "resetting": _RESET_PENDING.is_set(),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        parsed = urlsplit(self.path)
        if parsed.path == "/api/stream_meeting/cancel":
            _CANCEL.set()
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/api/stream_meeting/reset":
            # 阻止新会议抢在旧会议退出前取得锁；reset 请求之间也串行化。
            with _RESET_REQUEST_LOCK:
                with _SESSION_STATE_LOCK:
                    _RESET_PENDING.set()
                    _RESET.set()
                    _CANCEL.set()
                idle = _SESSION_IDLE.wait(timeout=120)
                with _SESSION_STATE_LOCK:
                    _RESET_PENDING.clear()
                self._send_json(200 if idle else 202, {
                    "ok": idle,
                    "busy": not idle,
                    "error": None if idle else "上一场会议仍在停止，请稍后重试重置",
                })
            return
        if parsed.path != "/api/stream_meeting":
            self._send_json(404, {"error": "not found"})
            return

        query = parse_qs(parsed.query)
        name = query.get("name", [None])[0]
        fast = query.get("fast", ["0"])[0].lower() in ("1", "true", "yes")

        self._dead = False
        with _SESSION_STATE_LOCK:
            if _RESET_PENDING.is_set():
                acquired = False
                busy_error = "正在重置上一场会议，请稍后再试"
            else:
                acquired = _SESSION_LOCK.acquire(blocking=False)
                busy_error = "已有会议在运行"
            if acquired:
                _SESSION_IDLE.clear()
                _CANCEL.clear()
                _RESET.clear()
                self._session_active = True
        if not acquired:
            self._send_json(409, {"error": busy_error})
            return

        tmpdir: Optional[Path] = None
        response_started = False
        try:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                self._send_json(411, {"error": "缺少 Content-Length"})
                return
            try:
                length = int(content_length)
            except ValueError:
                self._send_json(400, {"error": "Content-Length 非法"})
                return
            if length < 0:
                self._send_json(400, {"error": "Content-Length 非法"})
                return
            if length > MAX_UPLOAD_BYTES:
                self._send_json(413, {"error": "上传文件超过 500 MB 上限"})
                return

            body = self.rfile.read(length)
            if len(body) != length:
                self._send_json(400, {"error": "请求体不完整"})
                return
            if _RESET.is_set():
                self._send_json(409, {"error": "会议已重置"})
                return

            content_type = self.headers.get("Content-Type", "")
            part_name: Optional[str] = None
            payload = body
            if content_type.lower().startswith("multipart/form-data"):
                found = _extract_multipart(body, content_type)
                if found is None:
                    self._send_json(400, {"error": "multipart 中缺少带文件名的 part"})
                    return
                part_name, payload = found
            if not payload:
                self._send_json(400, {"error": "上传内容为空"})
                return

            # 只把 basename 拼进临时目录，客户端字符串绝不参与路径拼接
            filename = _safe_filename(name or part_name or "upload.bin")
            tmpdir = Path(tempfile.mkdtemp(prefix="meeting_upload_"))
            src = tmpdir / filename
            src.write_bytes(payload)

            wav_path = src
            if _needs_conversion(src):
                wav_path = tmpdir / "converted_16k.wav"
                try:
                    _convert_to_16k_mono(src, wav_path)
                except FfmpegMissing:
                    # 规格要求：此时仍以 NDJSON 告知致命错误，然后关闭响应
                    self._start_ndjson()
                    response_started = True
                    self._emit_event({
                        "t": "error",
                        "msg": "需要 ffmpeg 转码：未找到 ffmpeg 可执行文件，请安装并加入 PATH",
                        "fatal": True,
                    })
                    self._end_chunks_safely()
                    return

            session_dir = new_session_dir(SESSION_ROOT)
            models = _get_models()

            self._start_ndjson()
            response_started = True

            _run_pipeline(
                wav_path, session_dir,
                fast=fast, models=models,
                on_event=self._emit_event,
                on_progress=self._emit_event,
                should_stop=_CANCEL.is_set,
                should_abort=_RESET.is_set,
            )

            if _RESET.is_set():
                return

            # run() 返回时 session.json 已合并完成；diarization.json 可能存在也可能不存在
            store = SessionStore(session_dir)
            session_obj = store.read_json("session.json") or {}
            mapping = store.read_json("diarization.json") or {}
            utterances = session_obj.get("utterances") or []
            total = len(utterances) if utterances else len(mapping)

            self._emit_event({
                "t": "speaker_diarized",
                "n_clusters": _count_clusters(mapping),
                "mapping": mapping,
                "batch": {"done": len(mapping), "total": total},
                "turns": session_obj.get("turns", []),
            })
            self._emit_event({"t": "finish"})
            self._end_chunks_safely()
        except Exception as exc:  # noqa: BLE001 - 单场会议异常不得杀死服务
            if _RESET.is_set():
                return
            msg = f"{type(exc).__name__}: {exc}"
            if response_started:
                self._emit_event({"t": "error", "msg": msg, "fatal": True})
                self._emit_event({"t": "finish"})
                self._end_chunks_safely()
            else:
                self._send_json(500, {"error": msg})
        finally:
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
            self._session_active = False
            with _SESSION_STATE_LOCK:
                _SESSION_LOCK.release()
                _SESSION_IDLE.set()


class _SingleInstanceServer(ThreadingHTTPServer):
    """禁止 SO_REUSEADDR 双绑。

    Windows 上 HTTPServer 默认 allow_reuse_address=1，两个进程可以同时绑同一端口；
    结果旧进程会静默接管请求，跑的是旧代码。2026-10-17 实际踩过：turn 层上线后
    页面仍由旧进程服务，session.json 没有 turns。
    """

    allow_reuse_address = False


def make_server(host: str = SERVICE_HOST, port: int = SERVICE_PORT) -> ThreadingHTTPServer:
    try:
        server = _SingleInstanceServer((host, port), StreamRequestHandler)
    except OSError as exc:
        raise SystemExit(
            f"端口 {host}:{port} 绑定失败（可能已有服务在运行）: {exc}"
        ) from exc
    server.daemon_threads = True
    return server


def serve(host: str = SERVICE_HOST, port: int = SERVICE_PORT, preload: bool = True) -> None:
    server = make_server(host, port)
    print(f"[stream_service] 已启动 http://{host}:{port}  (Ctrl+C 停止)", flush=True)
    if preload:
        started = time.monotonic()
        try:
            _get_models()
            print(f"[stream_service] 模型预加载完成 {time.monotonic() - started:.1f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - 预加载失败仍允许启动，请求时会再报错
            print(f"[stream_service] 模型预加载失败（请求时重试）: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="会议回放通道 HTTP 服务（NDJSON 流式）")
    parser.add_argument("--port", type=int, default=SERVICE_PORT)
    parser.add_argument("--no-preload", action="store_true", help="启动时不加载模型")
    args = parser.parse_args(argv)
    serve(port=args.port, preload=not args.no_preload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
