# -*- coding: utf-8 -*-
"""
说话人分离本地测试台后端（零依赖 stdlib http.server + ffmpeg）。
接口：
  GET  /                -> diar_web.html
  GET  /api/lanes       -> 方法列表
  POST /api/compare     -> 四泳道对比（NDJSON 流）
  POST /api/llm         -> LLM 后校正
启动: python web_server.py   -> http://127.0.0.1:8734/
"""
import json, os, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")
import numpy as np

PORT = 8734
UPLOAD = os.path.join(HERE, "output", "web_uploads")
os.makedirs(UPLOAD, exist_ok=True)

# 本地快照路径（绕开 modelscope 缓存解析——共享缓存被并发修改时 id 解析会失败/触发联网）
SEACO_LOCAL = os.path.join(r"C:\Users\EDY\.cache\modelscope\models",
                           "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                           "snapshots", "master")
VAD_LOCAL = os.path.join(HERE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")

_models = {}
_mlock = threading.Lock()


def parse_compare_params(q):
    """校验 /api/compare 查询参数 → (params, error)。开关矩阵集中在此（可单测）。"""
    denoise = q.get("denoise", ["auto"])[0]
    reassign = q.get("reassign", ["off"])[0]
    snap = q.get("snap", ["v0"])[0]
    if denoise not in ("auto", "off", "on"):
        return None, f"denoise 只支持 auto|off|on（收到 {denoise}）"
    if reassign not in ("on", "off"):
        return None, f"reassign 只支持 on|off（收到 {reassign}）"
    if snap not in ("v0", "v1"):
        return None, f"snap 只支持 v0|v1（收到 {snap}）"
    if snap == "v0" and reassign == "on":
        return None, ("snapshot_v0 无 reassign 形参（快照内也无 reassign_core），"
                      "snap=v0 + reassign=on 不可用")
    return {"denoise": denoise, "reassign": reassign, "snap": snap,
            "name": q.get("name", ["upload.wav"])[0]}, None


def lane_specs(version, snap_ver, reassign_mode, claude_versions):
    """泳道 (key, label, kind)，顺序 = 前端显示顺序。

    claude_versions 由 claude_a2.visible_versions() 提供（用户要求界面只留 v7/v8）；
    DSH 两条（当前 / 最优快照）是**本仓主算法**，始终在界面上，不由 claude 侧决定。
    """
    snap_label = ("DSH v0（历史最优）" if snap_ver == "v0"
                  else "DSH v1 快照（与当前代码逐字节相同）")
    lanes = [(k, claude_versions[k]["label"], "claude") for k in claude_versions]
    lanes += [("cur", f"DSH {version} 当前·重分配{reassign_mode}", "dsh"),
              ("snap", snap_label, "dsh")]
    return lanes


def get_models(progress=None):
    def p(msg):
        if progress:
            progress(msg)
    with _mlock:
        if "vad" not in _models:
            from a2_port import load_models
            p("正在加载 VAD + CAM++ 声纹模型（首次约 30~60 秒）…")
            _models["vad"], _models["spk"] = load_models()
        if "asr" not in _models:
            from funasr import AutoModel
            from funasr_backend import load_funasr, wrap_funasr
            # 转写后端：Fun-ASR-Nano（2026-09-08 AB 选定，2m 远场 found 54→58-60%/
            # 38→43%）；加载失败回退 SEACO 并保持 generate(ndarray) 老接口
            try:
                p("正在加载 Fun-ASR-Nano 转写模型（约 30 秒）…")
                _models["asr"] = wrap_funasr(load_funasr())
            except Exception as e:
                p(f"Fun-ASR-Nano 加载失败（{type(e).__name__}），回退 SEACO …")
                _models["asr"] = AutoModel(model=SEACO_LOCAL, disable_update=True,
                                           disable_pbar=True)
    return _models


def normalize_audio(src, dst):
    # 沙箱下不可用管道捕获子进程输出（EPERM），用 DEVNULL 重定向
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le", dst],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def run_a2ovl(wav, progress=None, reassign=False):
    from a2_ovl import run as run_ovl
    m = get_models(progress)
    if progress:
        progress(f"A2 重叠分离：滑窗 + 密集短窗 + 重叠检测中…"
                 f"（段级重分配 {'开' if reassign else '关'}）")
    return run_ovl(wav, m["vad"], m["spk"], m["asr"], progress=progress,
                   reassign=reassign)


_snap_lock = threading.Lock()
_snap_mods = {}  # snap 名 -> (ovl_mod, port_mod, re_mod)；模块对象必须复用


def run_a2ovl_snapshot(wav, progress=None, snap="v0"):
    """历史版（snapshot_<snap> 冻结副本）：
    importlib 独立加载快照 a2_ovl.py + a2_port.py（+ 若存在则 reassign_core.py），
    运行时临时把 sys.modules 里的同名模块指向冻结副本——快照 run() 内部的
    `from a2_port import ...` / `from reassign_core import ...` 只认这些名字；
    不交换的话，Stage B 一改仓库根的 reassign_core.py，快照泳道会静默跑新核。
    与当前版共用已加载模型，不重复占显存。
    注意：module_from_spec 每次返回新空模块，exec_module 只在首次执行——
    必须缓存模块对象复用（第二次调用直接拿新空模块会 AttributeError: no run）。"""
    import importlib.util
    snap_dir = os.path.join(HERE, "snapshot_" + snap)
    m = get_models(progress)
    if progress:
        progress(f"快照 snapshot_{snap} 推理中…")
    with _snap_lock:  # sys.modules 交换是全局状态，串行化
        if snap not in _snap_mods:
            mods = {}
            for alias, fname in (("a2port_snap", "a2_port.py"),
                                 ("reassign_snap", "reassign_core.py"),
                                 ("a2ovl_snap", "a2_ovl.py")):
                p = os.path.join(snap_dir, fname)
                if not os.path.exists(p):
                    if fname in ("a2_port.py", "a2_ovl.py"):
                        raise RuntimeError(f"snapshot_{snap} 缺 {fname}")
                    continue                      # 快照无 reassign_core 是正常情况
                spec = importlib.util.spec_from_file_location(alias, p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[alias] = mod
                spec.loader.exec_module(mod)
                mods[fname] = mod
            _snap_mods[snap] = mods
        mods = _snap_mods[snap]
        saved = {n: sys.modules.get(n) for n in ("a2_port", "reassign_core")}
        sys.modules["a2_port"] = mods["a2_port.py"]
        if "reassign_core.py" in mods:
            sys.modules["reassign_core"] = mods["reassign_core.py"]
        try:
            return mods["a2_ovl.py"].run(wav, m["vad"], m["spk"], m["asr"],
                                         progress=progress)
        finally:
            for n, v in saved.items():
                if v is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = v



class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_chunk(self, line: str):
        data = (line + "\n").encode("utf-8")
        # 浏览器刷新/关页/取消运行时会直接掐断 SSE 连接（WinError 10053/10054）。
        # 若在此处抛出，异常处理链会继续向同一个死连接写错误块，级联崩溃；
        # 因此这里统一静默降级并标记 _dead，后续 _emit 直接返回。
        try:
            self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()
        except OSError:
            self._dead = True

    def _emit(self, obj):
        if getattr(self, "_dead", False):
            return
        self._send_chunk(json.dumps(obj, ensure_ascii=False))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/lanes":
            # 方法列表动态下发：DSH 两条（当前 / 最优快照）+ claude 的可见泳道。
            # DSH 侧 = 本仓主算法（当前 VERSION 与 snapshot 冻结副本），claude 侧由
            # claude_a2.visible_versions() 自行收敛（用户要求界面只留 v7/v8）。
            from a2_ovl import VERSION
            from claude_a2 import visible_versions
            lanes = [{"key": k, "label": v["label"], "kind": "claude"}
                     for k, v in visible_versions().items()]
            lanes += [{"key": "cur",
                       "label": f"DSH {VERSION} 当前（重分配见左栏开关）",
                       "kind": "dsh"},
                      {"key": "snap", "label": "DSH 快照（v0/v1 见左栏开关）",
                       "kind": "dsh"}]
            self._json({"lanes": lanes}, 200)
            return
        if self.path in ("/", "/index.html"):
            self._serve_file(os.path.join(HERE, "diar_web.html"), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/llm":
            self._handle_llm(parsed)
            return
        if parsed.path == "/api/compare":
            self._handle_compare(parsed)
            return
    def _finalize(self, result, method_label):
        g2 = result["diag"].get("gap2")
        result["method"] = method_label
        result["verdict"] = "unreliable" if (g2 is not None and g2 < 0.06) \
            else "reliable"
        return result

    def _handle_compare(self, parsed):
        """POST /api/compare?name=xx.wav&lanes=cur,snap,c1&denoise=auto&reassign=off&snap=v0
        - lanes: 逗号分隔要跑的方法（缺省=全部；cur/snap=DSH 两版，其余=claude VERSIONS）
        - denoise: auto(默认，嘈杂自动降噪)/off(不降噪)/on(强制降噪)
        - reassign: off(默认，当前发布行为)/on(段级重分配，评审用开关)
        - snap: v0(默认)/v1 —— 快照泳道指向哪个冻结副本
        NDJSON 流事件：progress(msg)（无 lane=全局）/progress(lane,msg)/done(lane,result)/finish。"""
        q = parse_qs(parsed.query)
        params, err = parse_compare_params(q)
        if err:
            self._json({"error": err}, 400)
            return
        name = params["name"]
        denoise_mode, reassign_mode, snap_ver = (params["denoise"],
                                                 params["reassign"],
                                                 params["snap"])
        lanes_param = q.get("lanes", [""])[0]
        lanes_wanted = [l.strip() for l in lanes_param.split(",") if l.strip()] or None
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        if not data:
            self._json({"error": "empty upload"}, 400)
            return
        ts = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        ext = os.path.splitext(name)[1].lower() or ".bin"
        raw = os.path.join(UPLOAD, ts + ext)
        wav = os.path.join(UPLOAD, ts + "_16k.wav")
        with open(raw, "wb") as f:
            f.write(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def progress(lane=None, msg=None):
            if msg is None:
                msg, lane = lane, None
            try:
                self._emit({"t": "progress", "lane": lane, "msg": msg})
            except Exception:
                pass

        try:
            progress(msg="音频转码中（ffmpeg → 16kHz 单声道）…")
            normalize_audio(raw, wav)
            # 自动环境检测（零训练：VAD 噪声底分位数法，-70dB 分界 23/23 全对）
            from env_detect import detect_env
            env_diag = detect_env(
                wav, get_models(lambda msg: progress("cur", msg)).get("vad"))
            progress(msg=f"环境检测：噪声底 {env_diag['noise_db']} dB → "
                         f"判定『{env_diag['verdict']}』…")
            wav_eff = wav
            should_denoise = (denoise_mode == "on") or (
                denoise_mode == "auto" and env_diag["verdict"] == "嘈杂")
            if should_denoise:
                progress(msg="执行 FRCRN 降噪前置…")
                try:
                    from denoise_frcrn import denoise_file
                    wav_eff = os.path.join(UPLOAD, ts + "_frcrn.wav")
                    denoise_file(wav, wav_eff)
                    progress(msg="降噪完成")
                except Exception as e:
                    progress(msg=f"降噪失败，回退原始音频（{type(e).__name__}）")
                    wav_eff = wav
            # 泳道顺序：claude 在前（前端只显示 v7/v8），DSH 两版在后（显式调用才跑）
            from a2_ovl import VERSION
            from claude_a2 import run_claude_lane, visible_versions
            all_lanes = lane_specs(VERSION, snap_ver, reassign_mode,
                                   visible_versions())
            wanted = [l for l in all_lanes
                      if lanes_wanted is None or l[0] in lanes_wanted]
            for lane, label, kind in wanted:
                progress(lane, f"{label}：推理中…")
                t0 = time.time()
                try:
                    if kind == "dsh":
                        if lane == "cur":
                            result = run_a2ovl(
                                wav_eff, progress=lambda msg: progress(lane, msg),
                                reassign=(reassign_mode == "on"))
                        else:
                            result = run_a2ovl_snapshot(
                                wav_eff, progress=lambda msg: progress(lane, msg),
                                snap=snap_ver)
                        self._finalize(result, label)
                    else:
                        result = run_claude_lane(wav_eff, get_models(progress),
                                                 lane,
                                                 progress=lambda msg: progress(lane, msg))
                    result["file"] = name
                    result["elapsed"] = round(time.time() - t0, 1)
                    result["env"] = env_diag
                    self._emit({"t": "done", "lane": lane, "result": result})
                except Exception as e:
                    self._emit({"t": "error", "lane": lane,
                                "msg": f"{type(e).__name__}: {e}"})
            self._emit({"t": "finish"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emit({"t": "error", "msg": f"{type(e).__name__}: {e}"})
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

    def _handle_llm(self, parsed):
        """POST /api/llm：body = 分离结果 JSON；调用 DeepSeek 后校正，NDJSON 流返回。
        Key 从服务进程环境变量 DEEPSEEK_API_KEY 读取（需在带 key 的终端启动服务）。"""
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        try:
            result = json.loads(data.decode("utf-8"))
        except Exception:
            self._json({"error": "bad body"}, 400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def progress(msg):
            try:
                self._emit({"t": "progress", "msg": msg})
            except Exception:
                pass

        try:
            progress("调用 DeepSeek 进行后校正（约 30~60 秒）…")
            from llm_correct import correct
            ns, names, _ = correct(result)
            out = {k: v for k, v in result.items() if k != "sentence_info"}
            out["sentence_info"] = ns
            out["method"] = str(result.get("method", "a2ovl")) + "+llm"
            out["names"] = names
            self._emit({"t": "done", "result": out})
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._emit({"t": "error", "msg": f"LLM: {type(e).__name__}: {e}"})
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

    def _serve_file(self, path, ctype):
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 页面是"边改边看"的测试台：不加 no-store 时浏览器会缓存旧 HTML，
        # 改完刷新仍是旧界面（2026-09-15 前端更新踩过这个坑）。
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class QuietThreadingServer(ThreadingHTTPServer):
    """客户端断开连接属正常行为，屏蔽其栈回显（其余错误照常打印）。"""
    def handle_error(self, request, client_address):
        import sys
        e = sys.exc_info()[1]
        if isinstance(e, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


if __name__ == "__main__":
    print(f"说话人分离测试台: http://127.0.0.1:{PORT}/", flush=True)
    print("模型在首次请求时加载（约 1~2 分钟），进度会实时推送到页面。", flush=True)
    # 防双监听：Windows 上 allow_reuse_address=True 会让第二个进程也绑定同一端口，
    # 请求随机落到新旧两个进程（曾两次踩坑）。禁用后第二个实例直接报错退出。
    QuietThreadingServer.allow_reuse_address = False
    try:
        QuietThreadingServer(("127.0.0.1", PORT), Handler).serve_forever()
    except OSError as e:
        print(f"[已退出] 端口 {PORT} 被占用（可能已有另一个 web_server.py 实例在跑），"
              f"请先关掉旧进程再启动。错误：{e}", flush=True)
        sys.exit(1)
