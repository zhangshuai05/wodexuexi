# -*- coding: utf-8 -*-
"""`v12.diarize` 必须把 K 上限**显式**传给 `v11._candidate_clusters`。

## 为什么需要这个测试

`v11._candidate_clusters(E, max_speakers=MAX_SPEAKERS)` 的默认值在 **`def` 执行时**
（模块导入时）就绑定成当时的 `v11.MAX_SPEAKERS` = **4**。
`v12._v11_window_params` 改的是**模块变量**，改不动已经绑好的默认值。

后果不是报错，是**假设空间变小**：候选只剩 k=1..4，
而 `_select_candidate(forced_k=5)` 里 `next(..., None)` 找不到就**跳过**、
落到 `eligible` 过滤、返回 `candidates[0]` = k=1 —— **静默回退，表现为成功**。

这就是 0928 挖出的那个活 bug，也是本会话第四次出现"派发错但无人报错"。
本测试从 `diarize` **自己的调用点**进去（不是直接调 `_candidate_clusters`）：
删掉 `max_speakers=MAX_SPEAKERS` 后，`n_windows=60` 却只拿到 k≤4，必红。

用假 `_extract_windows` 和空 wav，不加载任何模型。
"""
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

N_BLOBS = 6          # > 4：旧代码的 K 上限会把它截掉，新代码不会
PER_BLOB = 10


def _blob_vectors():
    """6 个互相正交方向的团，每个 10 窗 —— 每个 k 都能分出 eligible 的簇。"""
    basis = np.eye(64, dtype=np.float32)[:N_BLOBS]
    return np.vstack([np.tile(basis[i], (PER_BLOB, 1)) for i in range(N_BLOBS)])


def _silent_wav(tmp_path):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return str(path)


def _run(tmp_path, monkeypatch):
    """跑 `diarize`，把嵌入与 VAD 换成合成量；返回它的结果字典。"""
    vectors = _blob_vectors()
    monkeypatch.setattr(v11, "_vad_segs", lambda *a, **k: [(0.0, 1.0)])
    monkeypatch.setattr(v11, "_extract_windows",
                        lambda *a, **k: (vectors, np.arange(len(vectors)) * 0.1, 0))
    # 短窗复核路径也要能跑（它是 k==1 分支，本测试不会进，但别让它炸）
    monkeypatch.setattr(v12, "_refine_windows", lambda *a, **k: (vectors, np.arange(len(vectors)) * 0.15))
    models = {"vad": None, "spk": None}
    return v12.diarize(_silent_wav(tmp_path), models)


def test_diarize_reaches_k_above_v11_def_time_cap(tmp_path, monkeypatch):
    """`diarize` 造出的 K 候选必须超过 v11 那个 def 时绑死的 4。"""
    assert v11._candidate_clusters.__defaults__ == (4,), \
        "前提变了：v11 的默认值不再绑 4，本测试的判据需重写"

    result = _run(tmp_path, monkeypatch)
    assert result["n_windows"] == N_BLOBS * PER_BLOB
    ks = [item["k"] for item in result["candidates"]]
    assert max(ks) >= N_BLOBS, (
        f"候选 K 只到 {max(ks)}（{ks}）——"
        f"`diarize` 应传 max_speakers=MAX_SPEAKERS 而没传，被 v11 的默认值 4 截断"
    )


def test_forced_k_above_cap_actually_selects_that_k():
    """`_select_candidate(forced_k)` 找不到候选时会**静默返回 k=1**，必须能断言住。

    这是"注入失败表现为注入成功"的那条：生产路径一旦截断到 4，
    `forced_k=5` 不会报错，只会给你 k=1 的段表，而下游照常出数。
    """
    candidates, _ = v11._candidate_clusters(_blob_vectors(), max_speakers=v12.MAX_SPEAKERS)
    for k in (5, 10):
        assert v11._select_candidate(candidates, forced_k=k)["k"] == k, \
            f"forced_k={k} 没选出 k={k}——候选表被截断或注入静默失败"

    truncated, _ = v11._candidate_clusters(_blob_vectors())      # 复现旧行为
    assert v11._select_candidate(truncated, forced_k=5)["k"] != 5, \
        "截断到 4 时 forced_k=5 竟能选出 5，说明 v11 默认值已不是 4"
