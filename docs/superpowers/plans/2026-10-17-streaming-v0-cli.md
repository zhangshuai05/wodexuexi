# 流式会议 v0（纯 CLI 端到端）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一条命令把 16kHz WAV 走完「流式草稿 → 句尾固化 → 异步润色 → 单臂说话人回填」，产出 `output/sessions/s_*/` 六文件契约，不启 HTTP。

**Architecture:** `run_offline.py` 是唯一编排者：持有 `StreamingSession`（复用 `.zhang/streaming_asr_engine.py`，只读）、一个单消费者润色线程（新写）、以及在同一进程内复用已加载 `models` 的回填调用（新包装 `.zhang/offline_diarizer.py`）。所有落盘经 `SessionStore` 统一走「临时文件 + `os.replace`」原子写；`build_session_json()` 幂等合并三份产物，任一方完成即可调用。

**Tech Stack:** Python 3.12（conda `funasr`）、stdlib（`queue` / `threading` / `json` / `urllib`）、`soundfile`、已有 funasr 模型。无 Web 框架、无数据库、无新依赖。

## Global Constraints

- 解释器：`C:\Users\31817\miniconda3\envs\funasr\python.exe`（3.12.15）。跑任何脚本前 `$env:PYTHONIOENCODING="utf-8"`。
- **只读依赖**：绝不修改 `D:\Deepseek\diarization_cl-main` 下任何文件。全部新代码在 `D:\Deepseek\Meeting_main`。
- 时间戳只能由采样点计数得出：`samples * 1000 // 16000`。**禁止 `time.time()` 参与 `start_ms` / `end_ms`**。
- `spk` 是 **0-based 整数**；`spk_label` 是 **`speaker_{spk}`（全小写）**。
- 润色只许改同音字与标点；三道守卫：长度增幅 > 50%、缩减 > 30%、数字集合变化 ⇒ 丢弃改写、保留原文。
- v0 **不做**：HTTP、WebSocket、双臂（T1b）、重叠说话人（`overlap_spk` 恒为 `null`）。
- 所有 JSON 写入必须原子（temp + `os.replace`）；append-only 文件用 `.jsonl` / `.ndjson`，每行一个 JSON 对象。

---

### Task 0: 项目骨架 + 上游挂载点

**Files:**
- Create: `D:\Deepseek\Meeting_main\.gitignore`
- Create: `D:\Deepseek\Meeting_main\pytest.ini`
- Create: `D:\Deepseek\Meeting_main\bootstrap.py`
- Test: `D:\Deepseek\Meeting_main\tests\test_bootstrap.py`

**Interfaces:**
- Consumes: 无
- Produces: `bootstrap.install() -> None`（幂等）、`bootstrap.check_paths() -> dict[str, bool]`、`bootstrap.check() -> dict[str, bool | str]`

- [ ] **Step 1: 初始化仓库与忽略规则**

```powershell
cd D:\Deepseek\Meeting_main
git init
```

`.gitignore`：

```gitignore
__pycache__/
.pytest_cache/
*.pyc
output/
```

- [ ] **Step 2: 写 pytest 配置**

`pytest.ini`：

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -m "not slow"
markers =
    slow: 需要真实模型或长音频，默认不跑
```

- [ ] **Step 3: 写失败测试**

`tests/test_bootstrap.py`：

```python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

import bootstrap


def test_upstream_paths_exist():
    paths = bootstrap.check_paths()
    assert paths, "check_paths 不应返回空"
    missing = [name for name, ok in paths.items() if not ok]
    assert not missing, f"缺失的上游路径: {missing}"
    assert (Path(r"D:\Deepseek\diarization_cl-main") / ".zhang").is_dir()


def test_install_is_idempotent():
    import sys
    bootstrap.install()
    before = list(sys.path)
    bootstrap.install()
    assert list(sys.path) == before, "重复 install 不应改变 sys.path"


@pytest.mark.slow
def test_upstream_modules_importable():
    result = bootstrap.check()
    bad = {k: v for k, v in result.items() if v is not True}
    assert not bad, f"导入失败: {bad}"
```

- [ ] **Step 4: 运行测试确认失败**

```powershell
cd D:\Deepseek\Meeting_main
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_bootstrap.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bootstrap'`

- [ ] **Step 5: 实现 bootstrap.py**

```python
# -*- coding: utf-8 -*-
"""Meeting_main 与 diarization_cl-main 的唯一耦合点。

只往 sys.path 里加路径，不复制、不修改上游任何代码。
上游自身的 bootstrap 见 AliMeeting/run_3dspeaker_pipeline.py:34-37（同源）。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\Deepseek\diarization_cl-main")

# 本项目根目录：必须始终排在 sys.path 最前，避免被上游同名模块遮蔽
_LOCAL = Path(__file__).resolve().parent

_SUBPATHS: tuple[Path, ...] = (
    REPO / ".zhang",
    REPO / "AliMeeting",
    REPO / "streaming_asr",
    REPO / "scripts" / "evaluation",
)


def install() -> None:
    """幂等地把上游路径插到 sys.path 最前面。

    上游目录之后，本项目目录始终被强制放在 sys.path 首位，
    保证与上游同名的本地模块（如 run_offline、stream_llm_polish）不被遮蔽。
    """
    for path in _SUBPATHS:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    local = str(_LOCAL)
    if local in sys.path:
        sys.path.remove(local)
    sys.path.insert(0, local)


def check_paths() -> dict[str, bool]:
    """只检查目录存在，不触发任何 import。"""
    install()
    return {path.name: path.is_dir() for path in _SUBPATHS}


def check() -> dict[str, bool | str]:
    """真正尝试 import 上游入口。慢（会加载 funasr），仅 slow 测试用。"""
    install()
    result: dict[str, bool | str] = {}
    for name in ("offline_diarizer", "streaming_asr_engine", "web_server", "claude_a2"):
        try:
            __import__(name)
            result[name] = True
        except Exception as exc:  # noqa: BLE001 - 自检需要看到任何失败原因
            result[name] = f"{type(exc).__name__}: {exc}"
    return result
```

- [ ] **Step 6: 运行测试确认通过**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_bootstrap.py -v
```

Expected: 2 passed, 1 deselected（slow 被 `addopts` 排除）

- [ ] **Step 7: 提交**

```powershell
git add .gitignore pytest.ini bootstrap.py tests/test_bootstrap.py
git commit -m "chore: scaffold Meeting_main with read-only upstream bootstrap"
```

---

### Task 1: SessionStore —— 六文件契约

**Files:**
- Create: `D:\Deepseek\Meeting_main\session_store.py`
- Test: `D:\Deepseek\Meeting_main\tests\test_session_store.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `new_session_dir(root: Path, when: datetime | None = None) -> Path`
  - `SessionStore(session_dir: Path, *, session_id: str | None = None, started_at: str | None = None)`
  - 属性：`.dir`、`.audio_path`
  - `append_utterance(utt: dict) -> None`
  - `append_event(ev: dict) -> None`
  - `write_transcript(utterances: list[dict], duration_ms: int) -> Path`
  - `write_polished(mapping: dict[str, dict]) -> Path`
  - `write_diarization(mapping: dict[str, dict]) -> Path`
  - `read_json(name: str) -> dict | None`
  - `build_session_json() -> dict`

- [ ] **Step 1: 写失败测试**

`tests/test_session_store.py`：

```python
# -*- coding: utf-8 -*-
import json
from datetime import datetime

from session_store import SessionStore, new_session_dir


def _utterance(uid="utt_0001", start=0, end=1000, text="你好"):
    return {"utt_id": uid, "start_ms": start, "end_ms": end, "raw_text": text}


def test_transcript_only_falls_back_to_raw(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)

    obj = store.build_session_json()

    assert obj["status"] == "transcript"
    assert obj["utterances"][0]["polished_text"] == "你好"
    assert obj["utterances"][0]["corrected"] is False
    assert obj["utterances"][0]["bypassed"] is False
    assert obj["utterances"][0]["spk"] is None
    assert obj["utterances"][0]["spk_label"] is None
    assert obj["utterances"][0]["overlap_spk"] is None


def test_polished_is_applied(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.write_polished({
        "utt_0001": {"polished_text": "你好。", "corrected": True,
                     "bypassed": False, "error": None},
    })

    obj = store.build_session_json()

    assert obj["utterances"][0]["polished_text"] == "你好。"
    assert obj["utterances"][0]["corrected"] is True
    assert obj["status"] == "transcript", "润色不改变 status"


def test_diarization_sets_status_and_fields(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.write_diarization({
        "utt_0001": {"spk": 0, "spk_label": "speaker_0"},
    })

    obj = store.build_session_json()

    assert obj["status"] == "diarized"
    assert obj["utterances"][0]["spk"] == 0
    assert obj["utterances"][0]["spk_label"] == "speaker_0"


def test_build_is_idempotent(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)

    first = store.build_session_json()
    second = store.build_session_json()

    assert first == second


def test_append_only_files(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance(_utterance())
    store.append_utterance(_utterance("utt_0002", 1000, 2000, "再见"))
    store.append_event({"t": "asr_partial", "utt_id": "utt_0001", "text": "你"})

    lines = (store.dir / "utterances.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["utt_id"] == "utt_0002"
    assert len((store.dir / "events.ndjson").read_text(encoding="utf-8").splitlines()) == 1


def test_no_tmp_files_left_behind(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([_utterance()], duration_ms=1000)
    store.build_session_json()

    assert list(store.dir.glob("*.tmp")) == []


def test_new_session_dir_collision(tmp_path):
    when = datetime(2026, 10, 17, 14, 32, 0)
    first = new_session_dir(tmp_path, when=when)
    second = new_session_dir(tmp_path, when=when)

    assert first.name == "s_20261017_143200"
    assert second.name == "s_20261017_143200_1"
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_session_store.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'session_store'`

- [ ] **Step 3: 实现 session_store.py**

```python
# -*- coding: utf-8 -*-
"""Session 目录契约：六文件读写 + 原子落盘 + 最终合并。

文件职责严格隔离（用户 2026-10-17 定稿）：
    session_audio.wav   流式边收边写（由 StreamingSession 负责，本模块不管）
    utterances.jsonl    每个 asr_final append 一行 {utt_id,start_ms,end_ms,raw_text}
    transcript.json     T+0 全量文本，status="transcript"
    polished.json       润色结果 {utt_id:{polished_text,corrected,bypassed,error}}
    diarization.json    回填结果 {utt_id:{spk,spk_label}}
    session.json        幂等合并后的最终态
    events.ndjson       所有流式事件（前端回放用）
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

UTTERANCE_KEYS = ("utt_id", "start_ms", "end_ms", "raw_text")

# 进程内追加锁：主线程与润色线程会并发 append，Windows 下需串行化整次写入
_APPEND_LOCK = threading.Lock()


def new_session_dir(root: Path, when: Optional[datetime] = None) -> Path:
    """创建 output/sessions/s_YYYYMMDD_HHMMSS（重名时追加 _1、_2…）。"""
    when = when or datetime.now()
    base = "s_" + when.strftime("%Y%m%d_%H%M%S")
    root = Path(root)
    candidate = root / base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_ndjson(path: Path, obj: Any) -> None:
    # 先序列化再持锁打开写入，避免两线程的写入交错破坏行
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)


class SessionStore:
    """一个 session 目录的读写门面。

    合并锁保证 polished 线程与回填路径并发调用 build_session_json() 时不会互相覆盖。
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        session_id: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or self.dir.name
        self.started_at = started_at or datetime.now().astimezone().isoformat(timespec="seconds")
        self.audio_path = self.dir / "session_audio.wav"
        self._merge_lock = threading.Lock()

    # ---- 写入 ----

    def append_utterance(self, utt: Dict[str, Any]) -> None:
        row = {key: utt.get(key) for key in UTTERANCE_KEYS}
        _append_ndjson(self.dir / "utterances.jsonl", row)

    def append_event(self, ev: Dict[str, Any]) -> None:
        _append_ndjson(self.dir / "events.ndjson", ev)

    def write_transcript(self, utterances: List[Dict[str, Any]], duration_ms: int) -> Path:
        obj = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "status": "transcript",
            "audio": str(self.audio_path),
            "started_at": self.started_at,
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration_ms": int(duration_ms),
            "utterances": [
                {key: utt.get(key) for key in UTTERANCE_KEYS} for utt in utterances
            ],
        }
        path = self.dir / "transcript.json"
        _atomic_write_json(path, obj)
        return path

    def write_polished(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "polished.json"
        _atomic_write_json(path, mapping)
        return path

    def write_diarization(self, mapping: Dict[str, Dict[str, Any]]) -> Path:
        path = self.dir / "diarization.json"
        _atomic_write_json(path, mapping)
        return path

    # ---- 读取 ----

    def read_json(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.dir / name
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- 合并 ----

    def build_session_json(self) -> Dict[str, Any]:
        """读 transcript + 可选的 polished / diarization，幂等合成 session.json。"""
        with self._merge_lock:
            transcript = self.read_json("transcript.json") or {"utterances": []}
            polished = self.read_json("polished.json") or {}
            diarization = self.read_json("diarization.json")

            merged = []
            for utt in transcript.get("utterances", []):
                uid = utt["utt_id"]
                polish = polished.get(uid, {})
                speaker = (diarization or {}).get(uid, {})
                merged.append({
                    **utt,
                    "polished_text": polish.get("polished_text", utt.get("raw_text", "")),
                    "corrected": bool(polish.get("corrected", False)),
                    "bypassed": bool(polish.get("bypassed", False)),
                    "error": polish.get("error"),
                    "spk": speaker.get("spk"),
                    "spk_label": speaker.get("spk_label"),
                    "overlap_spk": None,
                })

            obj = {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "status": "diarized" if diarization else "transcript",
                "audio": str(self.audio_path),
                "started_at": transcript.get("started_at", self.started_at),
                "ended_at": transcript.get("ended_at"),
                "duration_ms": transcript.get("duration_ms"),
                "utterances": merged,
            }
            _atomic_write_json(self.dir / "session.json", obj)
            return obj
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_session_store.py -v
```

Expected: 7 passed

- [ ] **Step 5: 提交**

```powershell
git add session_store.py tests/test_session_store.py
git commit -m "feat: session store with atomic writes and idempotent merge"
```

---

### Task 2: 句尾异步润色器

**Files:**
- Create: `D:\Deepseek\Meeting_main\stream_llm_polish.py`
- Test: `D:\Deepseek\Meeting_main\tests\test_llm_polish.py`

**Interfaces:**
- Consumes: `bootstrap.install()`（需要 `streaming_asr` 在 sys.path 上以复用 `llm_correct._get_key` / `extract_json`）
- Produces:
  - `is_short(raw: str) -> bool`
  - `accept_rewrite(raw: str, polished: str) -> bool`
  - `Polisher(on_result: Callable[[dict], None], *, call=None, key=None, timeout=60)`
  - `Polisher.submit(ev: dict) -> None`（非阻塞）
  - `Polisher.close(timeout: float | None = None) -> dict[str, dict]`
  - `Polisher.results() -> dict[str, dict]`

- [ ] **Step 1: 写失败测试**

`tests/test_llm_polish.py`：

```python
# -*- coding: utf-8 -*-
import json
import threading
import time

import bootstrap

bootstrap.install()

from stream_llm_polish import Polisher, accept_rewrite, is_short  # noqa: E402


def _event(uid="utt_0001", raw="大家早上好今天我们把思露整理一下"):
    return {"t": "asr_final", "utt_id": uid, "start_ms": 0, "end_ms": 2000, "raw_text": raw}


def test_is_short_ignores_punctuation():
    assert is_short("好的") is True
    assert is_short("嗯。") is True
    assert is_short("三点钟") is False  # 3 个有效字，超过 SHORT_CIRCUIT_CHARS=2


def test_accept_rewrite_allows_punctuation_and_homophone():
    raw = "大家早上好今天我们把思露整理一下"
    polished = "大家早上好，今天我们把思路整理一下。"
    assert accept_rewrite(raw, polished) is True


def test_accept_rewrite_rejects_growth():
    assert accept_rewrite("你好", "你好你好你好你好你好") is False


def test_accept_rewrite_rejects_shrink():
    assert accept_rewrite("我们今天把思路整理一下", "今天") is False


def test_accept_rewrite_rejects_digit_change():
    assert accept_rewrite("我们3点开会", "我们4点开会") is False


def test_accept_rewrite_rejects_empty():
    assert accept_rewrite("你好", "   ") is False


def test_short_sentence_is_bypassed_without_calling_llm():
    calls = []
    results = []
    polisher = Polisher(results.append, call=lambda prompt: calls.append(prompt) or '{"polished":"x"}')

    polisher.submit(_event(raw="好的"))
    out = polisher.close(timeout=5)

    assert calls == []
    assert out["utt_0001"]["bypassed"] is True
    assert out["utt_0001"]["polished_text"] == "好的。"
    assert results[0]["t"] == "llm_confirmed"


def test_normal_sentence_calls_llm_and_reports_corrected():
    def fake_call(prompt):
        assert "上文语境" in prompt
        return json.dumps({"polished": "大家早上好，今天我们把思路整理一下。"}, ensure_ascii=False)

    results = []
    polisher = Polisher(results.append, call=fake_call)
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["corrected"] is True
    assert out["utt_0001"]["bypassed"] is False
    assert out["utt_0001"]["error"] is None
    assert results[0]["polished_text"] == "大家早上好，今天我们把思路整理一下。"


def test_llm_exception_fails_soft():
    def boom(prompt):
        raise RuntimeError("network down")

    results = []
    polisher = Polisher(results.append, call=boom)
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["corrected"] is False
    assert "network down" in out["utt_0001"]["error"]


def test_unparseable_output_fails_soft():
    results = []
    polisher = Polisher(results.append, call=lambda prompt: "这不是 JSON")
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["error"] == "LLM 输出无法解析为 JSON"


def test_guard_rejection_keeps_raw():
    results = []
    # 27 字 > raw 16 字 × 1.5 = 24，必须触发增幅守卫
    polisher = Polisher(results.append, call=lambda prompt: '{"polished":"这是一句完全不同的另一句话而且非常非常非常冗长冗长的话"}')
    polisher.submit(_event())
    out = polisher.close(timeout=5)

    assert out["utt_0001"]["polished_text"] == "大家早上好今天我们把思露整理一下"
    assert out["utt_0001"]["corrected"] is False
    assert out["utt_0001"]["error"] == "守卫驳回改写"


def test_submit_is_non_blocking_and_preserves_order():
    gate = threading.Event()

    def slow_call(prompt):
        gate.wait(2)
        return json.dumps({"polished": "改好了。"}, ensure_ascii=False)

    results = []
    polisher = Polisher(results.append, call=slow_call)

    started = time.monotonic()
    for i in range(3):
        polisher.submit(_event(uid=f"utt_{i:04d}"))
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, "submit 必须立即返回"

    gate.set()
    out = polisher.close(timeout=10)
    assert set(out) == {"utt_0000", "utt_0001", "utt_0002"}
    assert [ev["utt_id"] for ev in results] == ["utt_0000", "utt_0001", "utt_0002"]
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_llm_polish.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'stream_llm_polish'`

- [ ] **Step 3: 实现 stream_llm_polish.py**

```python
# -*- coding: utf-8 -*-
"""句尾异步 DeepSeek 润色：单消费者 FIFO，保守纠错，失败降级。

契约见 STREAMING_PLAN.md §6.2：
    允许：改同音/误识别字、补标点
    禁止：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语
    守卫：长度增幅 > 50% / 缩减 > 30% / 数字集合变化 => 丢弃改写，保留 raw_text
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.request
from typing import Callable, Dict, List, Optional

from llm_correct import _get_key, extract_json

MODEL = "deepseek-chat"
API = "https://api.deepseek.com/chat/completions"

SHORT_CIRCUIT_CHARS = 2
MAX_GROWTH = 1.5
MAX_SHRINK = 0.7
CONTEXT_SENTENCES = 3
PUNCTUATION = "，。！？、,.!? 　"

PROMPT_TMPL = """你是 ASR 转写纠错器。只做两件事：改同音/误识别的字，补标点。
严格禁止：改词序、增删内容、改数字、改专有名词、删口语重复、去语气词、改写成书面语。
不确定的地方保持原样——宁可少改，不可改错。

上文语境（仅用于判断同音字，不要续写、不要回应）：
{context}

待纠错句子：
{raw}

输出严格 JSON，不要解释，不要 markdown 代码块：
{{"polished": "纠错后的整句"}}"""


def is_short(raw: str) -> bool:
    """去掉标点与空白后 <= 2 个字符，视为短句。"""
    core = "".join(ch for ch in raw if ch not in PUNCTUATION)
    return len(core) <= SHORT_CIRCUIT_CHARS


def _digits(text: str) -> List[str]:
    return sorted(ch for ch in text if ch.isdigit())


def accept_rewrite(raw: str, polished: str) -> bool:
    """三道守卫。任一道不过就丢弃改写。"""
    candidate = polished.strip()
    if not candidate:
        return False
    if candidate == raw.strip():
        return True
    if len(candidate) > len(raw) * MAX_GROWTH:
        return False
    if len(candidate) < len(raw) * MAX_SHRINK:
        return False
    if _digits(candidate) != _digits(raw):
        return False
    return True


def _default_call(prompt: str, key: Optional[str] = None, timeout: float = 60) -> str:
    resolved = _get_key(key)
    if not resolved:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY（进程环境或 Windows 用户级环境变量任一处配置即可）")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }).encode("utf-8")
    request = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {resolved}",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


class Polisher:
    """单消费者 FIFO。submit 非阻塞；结果按提交顺序回调 on_result。"""

    def __init__(
        self,
        on_result: Callable[[Dict], None],
        *,
        call: Optional[Callable[[str], str]] = None,
        key: Optional[str] = None,
        timeout: float = 60,
    ) -> None:
        self._on_result = on_result
        self._timeout = timeout
        if call is not None:
            self._call: Callable[[str], str] = call
        else:
            def _bound(prompt: str) -> str:
                return _default_call(prompt, key=key, timeout=timeout)
            self._call = _bound

        self._queue: "queue.Queue[Optional[Dict]]" = queue.Queue()
        self._results: Dict[str, Dict] = {}
        self._context: List[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="llm-polish", daemon=True)
        self._thread.start()

    # ---- 对外 ----

    def submit(self, ev: Dict) -> None:
        """非阻塞。ev 至少含 utt_id / raw_text / start_ms / end_ms。"""
        self._queue.put({
            "utt_id": ev["utt_id"],
            "raw_text": ev.get("raw_text", ""),
            "start_ms": ev.get("start_ms"),
            "end_ms": ev.get("end_ms"),
        })

    def results(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._results)

    def close(self, timeout: Optional[float] = None) -> Dict[str, Dict]:
        self._queue.put(None)
        self._thread.join(timeout)
        return self.results()

    # ---- 内部 ----

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                result = self._polish_one(item["raw_text"])
                with self._lock:
                    self._results[item["utt_id"]] = result
                self._context.append(result["polished_text"])
                del self._context[:-CONTEXT_SENTENCES]
                self._on_result({
                    "t": "llm_confirmed",
                    "utt_id": item["utt_id"],
                    "start_ms": item["start_ms"],
                    "end_ms": item["end_ms"],
                    "raw_text": item["raw_text"],
                    "polished_text": result["polished_text"],
                    "corrected": result["corrected"],
                    "bypassed": result["bypassed"],
                    "error": result["error"],
                })
            finally:
                self._queue.task_done()

    def _polish_one(self, raw: str) -> Dict:
        base = {
            "polished_text": raw,
            "corrected": False,
            "bypassed": False,
            "error": None,
        }
        if is_short(raw):
            text = raw.strip()
            if text and text[-1] not in "，。！？、,.!?":
                text += "。"
            return {**base, "polished_text": text, "bypassed": True}

        try:
            context = "\n".join(self._context) if self._context else "（无）"
            reply = self._call(PROMPT_TMPL.format(context=context, raw=raw))
            obj = extract_json(reply)
            if not obj or "polished" not in obj:
                return {**base, "error": "LLM 输出无法解析为 JSON"}
            candidate = str(obj["polished"]).strip()
            if not accept_rewrite(raw, candidate):
                return {**base, "error": "守卫驳回改写"}
            return {
                **base,
                "polished_text": candidate,
                "corrected": candidate != raw.strip(),
            }
        except Exception as exc:  # noqa: BLE001 - 失败必须降级而不是卡死队列
            return {**base, "error": f"{type(exc).__name__}: {exc}"}
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_llm_polish.py -v
```

Expected: 12 passed

- [ ] **Step 5: 提交**

```powershell
git add stream_llm_polish.py tests/test_llm_polish.py
git commit -m "feat: single-consumer async LLM polisher with conservative guards"
```

---

### Task 3: 单臂回填 + 契约转换

**Files:**
- Create: `D:\Deepseek\Meeting_main\diar_backfill.py`
- Test: `D:\Deepseek\Meeting_main\tests\test_diar_backfill.py`
- Modify: `D:\Deepseek\Meeting_main\STREAMING_PLAN.md`（§三 / §五 的 `spk_label` 改为 `speaker_{p}`）

**Interfaces:**
- Consumes: `SessionStore`、`bootstrap.install()`、`offline_diarizer.backfill_speakers(wav_path, utterances, models, mode) -> dict[str, str]`
- Produces:
  - `to_contract(mapping: dict[str, str]) -> dict[str, dict]`
  - `load_utterances(session_dir: Path) -> list[dict]`
  - `run_backfill(session_dir, *, models=None, mode="auto", backfill=None, force=False) -> dict[str, dict]`
  - CLI：`python diar_backfill.py --session <dir> [--mode auto|class|vad] [--force]`

- [ ] **Step 1: 写失败测试**

`tests/test_diar_backfill.py`：

```python
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import bootstrap

bootstrap.install()

import pytest

from diar_backfill import load_utterances, run_backfill, to_contract  # noqa: E402
from session_store import SessionStore  # noqa: E402


def test_to_contract_converts_one_based_label_to_zero_based_spk():
    out = to_contract({"utt_0001": "Speaker_3"})
    assert out == {"utt_0001": {"spk": 2, "spk_label": "speaker_2"}}


def test_to_contract_handles_two_speakers():
    out = to_contract({"a": "Speaker_1", "b": "Speaker_2"})
    assert out["a"] == {"spk": 0, "spk_label": "speaker_0"}
    assert out["b"] == {"spk": 1, "spk_label": "speaker_1"}


def test_load_utterances_reads_jsonl_and_keeps_last(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "旧"})
    store.append_utterance({"utt_id": "u1", "start_ms": 0, "end_ms": 10, "raw_text": "新"})
    store.append_utterance({"utt_id": "u2", "start_ms": 10, "end_ms": 20, "raw_text": "二"})

    utts = load_utterances(store.dir)

    assert [u["utt_id"] for u in utts] == ["u1", "u2"]
    assert utts[0]["raw_text"] == "新"
    assert set(utts[0]) == {"utt_id", "start_ms", "end_ms", "raw_text"}


def test_run_backfill_writes_contract_and_merges(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"}],
        duration_ms=100,
    )
    store.audio_path.write_bytes(b"RIFF")

    called = {}

    def fake_backfill(wav_path, utterances, models, mode):
        called["args"] = (Path(wav_path).name, [u["utt_id"] for u in utterances], models, mode)
        return {"u1": "Speaker_2"}

    out = run_backfill(store.dir, models={"vad": "x", "spk": "y"},
                       backfill=fake_backfill)

    assert called["args"] == ("session_audio.wav", ["u1"], {"vad": "x", "spk": "y"}, "auto")
    assert out == {"u1": {"spk": 1, "spk_label": "speaker_1"}}
    assert json.loads((store.dir / "diarization.json").read_text(encoding="utf-8")) == out
    session = store.build_session_json()
    assert session["status"] == "diarized"
    assert session["utterances"][0]["spk"] == 1


def test_run_backfill_is_idempotent_unless_forced(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript(
        [{"utt_id": "u1", "start_ms": 0, "end_ms": 100, "raw_text": "你好"}],
        duration_ms=100,
    )
    store.audio_path.write_bytes(b"RIFF")
    store.write_diarization({"u1": {"spk": 0, "spk_label": "speaker_0"}})

    def should_not_run(*args, **kwargs):
        raise AssertionError("已有 diarization.json 时不应重跑")

    out = run_backfill(store.dir, backfill=should_not_run)
    assert out == {"u1": {"spk": 0, "spk_label": "speaker_0"}}


def test_run_backfill_requires_audio(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([], duration_ms=0)

    with pytest.raises(SystemExit):
        run_backfill(store.dir, backfill=lambda *a: {})


def test_run_backfill_empty_utterances_writes_empty_mapping(tmp_path):
    store = SessionStore(tmp_path / "s_1")
    store.write_transcript([], duration_ms=0)
    store.audio_path.write_bytes(b"RIFF")

    out = run_backfill(store.dir, backfill=lambda *a: {})
    assert out == {}
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_diar_backfill.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'diar_backfill'`

- [ ] **Step 3: 实现 diar_backfill.py**

```python
# -*- coding: utf-8 -*-
"""单臂回填：调 .zhang/offline_diarizer.backfill_speakers，转成 session 契约。

底层返回 {utt_id: "Speaker_{p+1}"}（p 是 0-based 簇标签）；
契约要   {utt_id: {"spk": p, "spk_label": f"speaker_{p}"}}。
所以 spk = int(label.split("_")[1]) - 1 —— 这个 off-by-one 只在这里换算一次。

独立 CLI（崩溃后重跑）：
    python diar_backfill.py --session output/sessions/s_xxx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bootstrap


def to_contract(mapping: Dict[str, str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for utt_id, label in mapping.items():
        index = int(str(label).split("_")[1]) - 1
        out[utt_id] = {"spk": index, "spk_label": f"speaker_{index}"}
    return out


def load_utterances(session_dir: Path) -> List[Dict]:
    """从 utterances.jsonl 读句表，按 utt_id 去重（后写覆盖先写）。"""
    path = Path(session_dir) / "utterances.jsonl"
    if not path.exists():
        return []
    rows: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["utt_id"]] = {
                "utt_id": row["utt_id"],
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "raw_text": row.get("raw_text", ""),
            }
    return list(rows.values())


def _real_backfill(wav_path, utterances, models, mode):
    from offline_diarizer import backfill_speakers
    return backfill_speakers(wav_path, utterances, models=models, mode=mode)


def run_backfill(
    session_dir,
    *,
    models: Optional[Dict] = None,
    mode: str = "auto",
    backfill: Optional[Callable] = None,
    force: bool = False,
) -> Dict[str, Dict]:
    bootstrap.install()
    from session_store import SessionStore

    store = SessionStore(Path(session_dir))
    existing = store.read_json("diarization.json")
    if existing is not None and not force:
        return existing

    if not store.audio_path.exists():
        raise SystemExit(f"缺少音频文件: {store.audio_path}")

    utterances = load_utterances(store.dir)
    if not utterances:
        store.write_diarization({})
        store.build_session_json()
        return {}

    fn = backfill or _real_backfill
    raw = fn(str(store.audio_path), utterances, models, mode)
    mapping = to_contract(raw)

    store.write_diarization(mapping)
    store.build_session_json()
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="单臂说话人回填（可独立重跑）")
    parser.add_argument("--session", required=True, help="session 目录，如 output/sessions/s_xxx")
    parser.add_argument("--mode", default="auto", choices=("auto", "class", "vad"))
    parser.add_argument("--force", action="store_true", help="忽略已有 diarization.json")
    args = parser.parse_args(argv)

    bootstrap.install()
    from web_server import get_models

    started = time.time()
    models = get_models(need_stream=False, need_offline=False)
    load_seconds = time.time() - started
    print(f"[backfill] 模型就绪 {load_seconds:.1f}s", flush=True)

    started = time.time()
    mapping = run_backfill(Path(args.session), models=models, mode=args.mode, force=args.force)
    elapsed = time.time() - started

    speakers = sorted({v["spk_label"] for v in mapping.values()})
    print(f"[backfill] {len(mapping)} 句 -> {len(speakers)} 人 {speakers} "
          f"({elapsed:.1f}s，不含模型加载)", flush=True)
    print(f"[backfill] 写入 {Path(args.session) / 'diarization.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_diar_backfill.py -v
```

Expected: 7 passed

- [ ] **Step 5: 同步 STREAMING_PLAN.md 的 `spk_label` 口径**

把 `STREAMING_PLAN.md` §三 契约块与 §五 `speaker_diarized` 示例中的
`"Speaker_1"` / `Speaker_{p+1}` 全部改为 **`"speaker_0"` / `speaker_{p}`**，
并在 §三 锚定铁律 3 写明「底层返回 `Speaker_{p+1}`，服务层只换算一次」。

- [ ] **Step 6: 提交**

```powershell
git add diar_backfill.py tests/test_diar_backfill.py STREAMING_PLAN.md
git commit -m "feat: single-arm backfill with CLI and zero-based speaker contract"
```

---

### Task 4: run_offline.py —— 端到端编排

**Files:**
- Create: `D:\Deepseek\Meeting_main\run_offline.py`
- Test: `D:\Deepseek\Meeting_main\tests\test_run_offline.py`

**Interfaces:**
- Consumes: `bootstrap`、`SessionStore` / `new_session_dir`、`Polisher`、`run_backfill`、`streaming_asr_engine.StreamingSession`
- Produces:
  - `iter_audio_chunks(wav_path, chunk_ms, max_seconds=0.0) -> Iterator[tuple[bytes, bool]]`
  - `run(wav, session_dir, *, chunk_ms=600, fast=False, no_llm=False, no_diar=False, max_seconds=0.0, models=None, session_factory=None, polisher_factory=None, backfill_fn=None) -> dict`
  - CLI：`python run_offline.py --wav <path> [--session-dir ...] [--fast] [--no-llm] [--no-diar]`

- [ ] **Step 1: 写失败测试（用假引擎，不加载真实模型）**

`tests/test_run_offline.py`：

```python
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import bootstrap

bootstrap.install()

import run_offline  # noqa: E402


class FakeSession:
    """按脚本吐出事件，模拟 StreamingSession 的接口。"""

    def __init__(self, wav_path):
        self.wav_path = Path(wav_path)
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        self.wav_path.write_bytes(b"RIFF")
        self.completed_utterances = []
        self.processed_ms = 0

    def feed_pcm_chunk(self, pcm_bytes, is_final=False):
        events = [{"t": "asr_partial", "utt_id": "utt_0001", "start_ms": 0,
                   "current_ms": 100, "text": "你"}]
        if is_final:
            utt = {"utt_id": "utt_0001", "start_ms": 0, "end_ms": 200, "raw_text": "你好"}
            self.completed_utterances.append(utt)
            self.processed_ms = 200
            events.append({"t": "asr_final", **utt})
        return events


class FakePolisher:
    def __init__(self, on_result):
        self.on_result = on_result
        self.submitted = []
        self.closed = False

    def submit(self, ev):
        self.submitted.append(ev)
        self.on_result({
            "t": "llm_confirmed", "utt_id": ev["utt_id"],
            "start_ms": ev["start_ms"], "end_ms": ev["end_ms"],
            "raw_text": ev["raw_text"], "polished_text": "你好。",
            "corrected": True, "bypassed": False, "error": None,
        })

    def close(self, timeout=None):
        self.closed = True
        return {"utt_0001": {"polished_text": "你好。", "corrected": True,
                             "bypassed": False, "error": None}}

    def results(self):
        return {}


def _write_wav(path: Path, seconds: float = 0.4):
    samples = int(16000 * seconds)
    sf.write(str(path), np.zeros(samples, dtype=np.int16), 16000, subtype="PCM_16")


def test_run_writes_all_contract_files(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_1"

    obj = run_offline.run(
        wav, session_dir,
        chunk_ms=200, fast=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        backfill_fn=lambda wav_path, utts, models, mode: {"utt_0001": "Speaker_1"},
        models={"vad": None, "spk": None},
    )

    files = {p.name for p in session_dir.iterdir()}
    assert {"utterances.jsonl", "events.ndjson", "transcript.json",
            "polished.json", "diarization.json", "session.json"} <= files

    assert obj["status"] == "diarized"
    utt = obj["utterances"][0]
    assert utt["raw_text"] == "你好"
    assert utt["polished_text"] == "你好。"
    assert utt["spk"] == 0
    assert utt["spk_label"] == "speaker_0"

    transcript = json.loads((session_dir / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["status"] == "transcript"
    assert transcript["duration_ms"] == 200


def test_no_diar_and_no_llm(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_2"

    obj = run_offline.run(
        wav, session_dir,
        chunk_ms=200, fast=True, no_llm=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        models={},
    )

    assert obj["status"] == "transcript"
    assert obj["utterances"][0]["spk"] is None
    assert not (session_dir / "diarization.json").exists()


def test_events_ndjson_contains_all_events(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav)
    session_dir = tmp_path / "s_3"

    run_offline.run(
        wav, session_dir, chunk_ms=200, fast=True, no_diar=True,
        session_factory=lambda p, m: FakeSession(p),
        polisher_factory=FakePolisher,
        models={},
    )

    events = [json.loads(line) for line in
              (session_dir / "events.ndjson").read_text(encoding="utf-8").splitlines()]
    kinds = [e["t"] for e in events]
    assert kinds.count("asr_partial") >= 2
    assert kinds.count("asr_final") == 1
    assert kinds.count("llm_confirmed") == 1


def test_iter_audio_chunks_rejects_non_16k(tmp_path):
    wav = tmp_path / "bad.wav"
    sf.write(str(wav), np.zeros(8000, dtype=np.int16), 8000, subtype="PCM_16")
    with pytest.raises(SystemExit):
        list(run_offline.iter_audio_chunks(wav, 600))


def test_iter_audio_chunks_respects_max_seconds(tmp_path):
    wav = tmp_path / "in.wav"
    _write_wav(wav, seconds=2.0)
    chunks = list(run_offline.iter_audio_chunks(wav, 600, max_seconds=0.6))
    assert sum(len(c) for c, _ in chunks) == int(16000 * 0.6) * 2
    assert chunks[-1][1] is True
```

- [ ] **Step 2: 运行测试确认失败**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_run_offline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'run_offline'`

- [ ] **Step 3: 实现 run_offline.py**

```python
# -*- coding: utf-8 -*-
"""v0 端到端：WAV -> 流式三态 -> 异步润色 -> 单臂回填 -> session 目录。

用法：
    python run_offline.py --wav D:/Deepseek/yinp/smoke_clip_180s.wav --fast --no-llm
    python run_offline.py --wav <16k mono wav>            # 实时节奏
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import bootstrap


SAMPLE_RATE = 16000
MAX_END_SILENCE_MS = 1000  # 2026-10-17 拍板：1000ms（450 会把 ASR 切成碎片）


def iter_audio_chunks(
    wav_path, chunk_ms: int = 600, max_seconds: float = 0.0
) -> Iterator[Tuple[bytes, bool]]:
    """按 chunk_ms 切 16k 单声道 int16；最后一块的 is_last 为 True。"""
    import soundfile as sf

    info = sf.info(str(wav_path))
    if info.samplerate != SAMPLE_RATE:
        raise SystemExit(f"需要 16kHz，实际 {info.samplerate}Hz: {wav_path}")

    frames = int(max_seconds * SAMPLE_RATE) if max_seconds > 0 else -1
    audio, _ = sf.read(str(wav_path), dtype="int16", always_2d=True, frames=frames)
    audio = audio[:, 0]

    step = max(1, int(SAMPLE_RATE * chunk_ms / 1000))
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        yield chunk.tobytes(), (start + step) >= len(audio)


def _default_session(wav_path, models):
    from streaming_asr_engine import StreamingSession
    return StreamingSession(str(wav_path),
                            asr_model=models["stream_asr"],
                            vad_model=models["vad"],
                            max_end_silence_time=MAX_END_SILENCE_MS)


def run(
    wav,
    session_dir,
    *,
    chunk_ms: int = 600,
    fast: bool = False,
    no_llm: bool = False,
    no_diar: bool = False,
    max_seconds: float = 0.0,
    models: Optional[Dict] = None,
    session_factory: Optional[Callable] = None,
    polisher_factory: Optional[Callable] = None,
    backfill_fn: Optional[Callable] = None,
) -> Dict:
    bootstrap.install()
    from session_store import SessionStore
    from stream_llm_polish import Polisher

    store = SessionStore(Path(session_dir))

    if models is None:
        from web_server import get_models
        models = get_models(need_stream=True, need_offline=False)

    session = (session_factory or _default_session)(store.audio_path, models)

    polished_buffer: Dict[str, Dict] = {}

    def on_polish(ev: Dict) -> None:
        store.append_event(ev)
        polished_buffer[ev["utt_id"]] = {
            "polished_text": ev["polished_text"],
            "corrected": ev["corrected"],
            "bypassed": ev["bypassed"],
            "error": ev["error"],
        }
        # 每句完成即落盘：崩了也不丢已润色的部分
        store.write_polished(polished_buffer)
        store.build_session_json()

    polisher = None
    if not no_llm:
        polisher = (polisher_factory or Polisher)(on_polish)

    def handle(ev: Dict) -> None:
        if ev.get("t") == "asr_final":
            store.append_utterance(ev)
            store.append_event(ev)
            if polisher is not None:
                polisher.submit(ev)
        else:
            store.append_event(ev)

    clock = time.monotonic()
    fed_seconds = 0.0
    for chunk, is_last in iter_audio_chunks(wav, chunk_ms, max_seconds):
        for ev in session.feed_pcm_chunk(chunk, is_final=is_last):
            handle(ev)
        if not fast:
            fed_seconds += len(chunk) / 2 / SAMPLE_RATE
            delay = fed_seconds - (time.monotonic() - clock)
            if delay > 0:
                time.sleep(delay)

    store.write_transcript(session.completed_utterances, duration_ms=session.processed_ms)
    store.build_session_json()

    if not no_diar:
        from diar_backfill import run_backfill
        run_backfill(store.dir, models=models, backfill=backfill_fn)

    if polisher is not None:
        results = polisher.close(timeout=None)
        store.write_polished(results)
        store.build_session_json()

    return store.read_json("session.json") or {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v0 端到端 CLI 转写")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--session-root", default="output/sessions")
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--chunk-ms", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--fast", action="store_true", help="全速推流，不按墙钟节流")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-diar", action="store_true")
    args = parser.parse_args(argv)

    bootstrap.install()
    from session_store import new_session_dir

    session_dir = (Path(args.session_dir) if args.session_dir
                   else new_session_dir(Path(args.session_root)))

    started = time.time()
    obj = run(
        Path(args.wav), session_dir,
        chunk_ms=args.chunk_ms, fast=args.fast,
        no_llm=args.no_llm, no_diar=args.no_diar,
        max_seconds=args.max_seconds,
    )
    elapsed = time.time() - started

    utts = obj.get("utterances", [])
    speakers = sorted({u["spk_label"] for u in utts if u.get("spk_label")})
    summary = {
        "session_dir": str(session_dir),
        "status": obj.get("status"),
        "utterances": len(utts),
        "speakers": speakers,
        "duration_ms": obj.get("duration_ms"),
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_run_offline.py -v
```

Expected: 5 passed

- [ ] **Step 5: 跑全量快测试**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest -q
```

Expected: 全部 passed（slow 被排除）

- [ ] **Step 6: 提交**

```powershell
git add run_offline.py tests/test_run_offline.py
git commit -m "feat: offline CLI orchestrating streaming ASR, async polish and backfill"
```

---

### Task 5: 真音频冒烟 + README

**Files:**
- Create: `D:\Deepseek\Meeting_main\README.md`
- Test: `D:\Deepseek\Meeting_main\tests\test_smoke_real.py`（slow）

**Interfaces:**
- Consumes: `run.run(...)`
- Produces: 无（验证任务）

- [ ] **Step 1: 写 slow 冒烟测试**

`tests/test_smoke_real.py`：

```python
# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

import bootstrap

bootstrap.install()

import run_offline  # noqa: E402

SMOKE = Path(r"D:\Deepseek\yinp\smoke_clip_180s.wav")


@pytest.mark.slow
@pytest.mark.skipif(not SMOKE.exists(), reason="缺少 smoke_clip_180s.wav")
def test_real_smoke_transcript_only(tmp_path):
    """只看转写链路；--no-diar 让这个测试在 1 分钟内可完成。"""
    obj = run_offline.run(
        SMOKE, tmp_path / "s_smoke",
        chunk_ms=600, fast=True, no_llm=True, no_diar=True,
        max_seconds=30.0,
    )
    assert obj["status"] == "transcript"
    assert obj["utterances"], "30 秒内应至少锁定一句"
    for utt in obj["utterances"]:
        assert utt["end_ms"] > utt["start_ms"] >= 0
        assert utt["raw_text"].strip()


@pytest.mark.slow
@pytest.mark.skipif(not SMOKE.exists(), reason="缺少 smoke_clip_180s.wav")
def test_real_smoke_full(tmp_path):
    """完整链路含回填。耗时约 1-5 分钟。"""
    obj = run_offline.run(
        SMOKE, tmp_path / "s_full",
        chunk_ms=600, fast=True, no_llm=True,
    )
    assert obj["status"] == "diarized"
    assert all(u["spk"] is not None for u in obj["utterances"])
    assert all(u["spk_label"].startswith("speaker_") for u in obj["utterances"])
```

- [ ] **Step 2: 跑 slow 冒烟（先只跑转写那条）**

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe -m pytest tests/test_smoke_real.py -m slow -k transcript -v -s
```

Expected: PASS。若失败，先看 `session_dir/events.ndjson` 的事件序列，再查 `session_audio.wav` 是否非空。

- [ ] **Step 3: 手动跑完整 180 秒（不带 LLM，避免依赖 API Key）**

```powershell
$env:PYTHONIOENCODING="utf-8"
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav `
    --session-dir output/sessions/s_smoke_180 `
    --fast --no-llm
```

Expected: 打印 `{"status": "diarized", "utterances": N, "speakers": ["speaker_0", ...]}`。
把实际耗时记进 README。

- [ ] **Step 4: 如果配了 DEEPSEEK_API_KEY，再跑一次带润色**

```powershell
C:\Users\31817\miniconda3\envs\funasr\python.exe run_offline.py `
    --wav D:\Deepseek\yinp\smoke_clip_180s.wav `
    --session-dir output/sessions/s_smoke_llm --fast
```

Expected: `session.json` 里至少有一条 `corrected=True`；没有 key 时 `error` 字段会写明缺 key，但 `status` 仍为 `diarized`（fail-soft 生效）。

- [ ] **Step 5: 写 README.md**

内容必须包含：环境变量、两条命令（实时 / `--fast`）、六文件契约表、`diar_backfill.py` 独立重跑命令、以及实测耗时（写实测值，不要照抄本计划里的估计）。

- [ ] **Step 6: 提交**

```powershell
git add README.md tests/test_smoke_real.py
git commit -m "test: real-audio smoke coverage and v0 README"
```

---

## 自审记录

- **Spec 覆盖**：STREAMING_PLAN §三（Utterance 契约）→ Task 1；§四 阶段 2（异步润色）→ Task 2；§四 阶段 3（单臂回填）→ Task 3；§七 双模输入（v0 只做回放）→ Task 4；§4.1 分阶段落盘 → Task 1 + Task 4 的 `on_polish` 增量写。
- **明确不做**：§6.1 T1b 双臂、§5.3 WebSocket、§4.1 的「分批下发」——已在 Global Constraints 里写明 v0 范围。
- **类型一致性**：`run_backfill(session_dir, *, models, mode, backfill, force)` 在 Task 3 定义、Task 4 调用；`Polisher(on_result, *, call, key, timeout)` 在 Task 2 定义、Task 4 以 `polisher_factory` 注入；`SessionStore.build_session_json()` 三处调用。
- **已知偏差**：`polished.json` 在 Task 2 的契约里是「跑完后落盘」，实现改成了「每句完成即原子重写」——这是严格改进（崩溃安全），格式不变。
