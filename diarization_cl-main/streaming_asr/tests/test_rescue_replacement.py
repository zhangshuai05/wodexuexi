# -*- coding: utf-8 -*-
"""`v12.diarize` 的 **rescue 替换分支**（控制流回归，合成数据）。

## 为什么需要

路径等价检查在 dev 12 场上跑了 26 个对比，但 `L_R003S01C02` 是唯一进入 rescue
的场次，而它救援窗的 k=2..10 **全部不合格**，`_select_candidate` 输出 k=1，
**未替换**（`rescued=False`）。

所以"rescue 分支**执行了但不生效**"已被覆盖，
**"rescue 真的替换了结果"从未被验过** —— 即下面这段的整体替换：

    if rselected is not None and rselected["k"] > 1:
        E, times, labels = rE, rtimes, rselected["labels"].copy()
        candidates, selected = rcandidates, rselected
        embedding_errors += rescue_errors
        rescued = True

替换后 `E/times/labels` 三个量必须**同时**换成救援窗的，否则下游
`_centroids(E, labels)` / `scores = E @ centroids.T` / `_segment_windows(times, ...)`
会拿混搭的数据算出无意义的段表 —— 而这类混搭**不会报错**。

## 本测试只验控制流

合成嵌入 + 假 VAD，**不宣称任何真实音频效果**。
它回答的是"替换这一步接对了没有"，不是"救援在真数据上有用"。

## 判据都锚在真实代码路径上

  * `k=1` 由**真的** `_select_candidate` 从合成候选里选出（不是硬编造返回值）；
  * rescue 分支由 `WIN_S` 的**实际取值**区分（主窗 2.0 / 救援窗 0.6）；
  * `k>1` 由**真的** `_select_candidate` 从救援候选里选出；
  * 段表由**真的** `_segment_windows` 切出。
"""
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

# ⚠️ 主窗与救援窗的窗数**必须不同**。若两者相同，`times` 与 `labels` 混搭后
# 长度仍然一致，段表照常切出、断言全过 —— 混搭就被掩盖了。这是本测试初版的
# 漏洞：主窗 24 / 救援窗 24 时，把"只换 labels 不换 times"注进去仍 2 passed。
MAIN_WINDOWS = 20       # 主窗：单簇
RESCUE_WINDOWS = 24     # 救援窗：两簇各 12，均 >= MIN_CLUSTER_WINDOWS(4)


def _blob(center_index, dim=64):
    """单个方向的归一化向量 —— 同方向即同一簇。"""
    vec = np.zeros(dim, dtype=np.float32)
    vec[center_index] = 1.0
    return vec


def _single_cluster():
    """主窗：全部同方向 → k=2 的两簇里必有一个是空簇 → 不合格 → 选 k=1。"""
    return np.stack([_blob(0)] * MAIN_WINDOWS)


def _two_clusters():
    """救援窗：两个等大且互相正交的簇 → k=2 合格且 silhouette > 0。"""
    half = RESCUE_WINDOWS // 2
    return np.stack([_blob(0)] * half + [_blob(1)] * (RESCUE_WINDOWS - half))


def _silent_wav(tmp_path):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return str(path)


def _install_fakes(monkeypatch, seen):
    """替换模型相关入口；按**实际 WIN_S** 分发主窗 / 救援窗的嵌入。

    用 `WIN_S` 而不是调用计数来区分阶段，是因为这才是生产代码真正的区分方式
    （`_v11_window_params` 换的就是它）。调用计数会掩盖"阶段进错"的 bug。
    """
    monkeypatch.setattr(v11, "_vad_segs", lambda *a, **k: [(0.0, 1.0)])

    def fake_extract(audio, rate, vad_segments, spk_model):
        if abs(v11.WIN_S - v12.WIN_S) < 1e-9:
            seen.append("main")
            emb = _single_cluster()
        elif abs(v11.WIN_S - v12.REFINE_WIN_S) < 1e-9:
            seen.append("rescue")
            emb = _two_clusters()
        else:
            raise AssertionError(f"未预期的 WIN_S={v11.WIN_S}")
        times = np.arange(len(emb), dtype=float) * 0.15
        return emb, times, 0

    monkeypatch.setattr(v11, "_extract_windows", fake_extract)
    # 主窗 k=1 → 不进 refine 分支；救援分支也不调它。返回空以钉住这一点。
    monkeypatch.setattr(v12, "_refine_windows", lambda *a, **k: (np.empty((0, 64)), []))


def test_rescue_branch_replaces_result_when_it_finds_multiple_speakers(tmp_path, monkeypatch):
    """救援窗选出 k>1 时，K / 标签 / 时间轴 / 段表**必须整体替换**。"""
    seen = []
    _install_fakes(monkeypatch, seen)
    result = v12.diarize(_silent_wav(tmp_path), {"vad": None, "spk": None})

    # 判据 1：确实走了 rescue 分支（两个阶段都到过）
    assert seen == ["main", "rescue"], f"阶段序列异常：{seen}"

    # 判据 2：真的替换了（这是此前从未被验证的那一步）
    assert result["rescued"] is True, "救援选出 k>1 却没置 rescued"
    assert result["refined"] is False
    assert result["k"] == 2, f"最终 k 应为 2，实得 {result['k']}"

    # 判据 3：E / times / labels 三者**同时**换成救援窗的，不得混搭
    # 长度必须等于**救援窗**的窗数（主窗只有 MAIN_WINDOWS 个）：
    # 若只换了 labels 没换 times，这里立刻不等 —— 初版漏掉的正是这条。
    assert len(result["labels"]) == RESCUE_WINDOWS, \
        f"labels 未换成救援窗的（{len(result['labels'])} != {RESCUE_WINDOWS}）"
    assert len(result["times"]) == RESCUE_WINDOWS, \
        f"times 未换成救援窗的（{len(result['times'])} != {RESCUE_WINDOWS}）"
    assert len(set(result["labels"].tolist())) == 2, "救援窗标签不是两簇"
    # 两簇各占一半：若 labels 与 times 错位，这个等式会破
    counts = sorted(int((result["labels"] == l).sum()) for l in (0, 1))
    assert counts == [RESCUE_WINDOWS // 2, RESCUE_WINDOWS // 2], f"簇尺寸异常：{counts}"

    # 判据 4：时间轴取自救援窗（0.15 步长），不是主窗（1.0 步长）
    times = np.asarray(result["times"], dtype=float)
    assert abs((times[1] - times[0]) - v12.REFINE_HOP_S) < 1e-9, \
        f"时间轴步长为 {times[1]-times[0]}，应为救援窗的 {v12.REFINE_HOP_S}"

    # 判据 5：段表由真的 `_segment_windows` 切出，且窗口参数已切到 0.6s
    assert result["segments"], "段表为空——替换后未重新切段"
    assert result["window_seconds"] == v12.REFINE_WIN_S
    assert result["hop_seconds"] == v12.REFINE_HOP_S
    for seg in result["segments"]:
        assert seg["end"] > seg["start"], f"非法段：{seg}"
        assert seg["spk"] in (0, 1), f"段标签越界：{seg['spk']}"

    # 判据 6：candidates 也一并换成救援窗的候选表（否则下游读到的 k 与段表不符）
    # 主窗是**单簇**，任何 k>=2 的划分都会产生空簇 ⇒ 没有合格候选；
    # 救援窗两簇均衡 ⇒ k=2 合格。故"存在合格的 k>=2"正是"候选表已被替换"的判据。
    # （原先这里写 `result["candidates"] is result["candidates"]`——同对象取两次恒真，
    #   对任何 payload 都成立，断言不了任何东西。）
    assert any(c["k"] >= 2 and c.get("eligible") for c in result["candidates"]), \
        f"候选表里没有合格的 k>=2 —— 说明 candidates 仍是主窗（单簇）那份，未被替换：" \
        f"{[(c['k'], c.get('eligible')) for c in result['candidates']]}"


def test_single_speaker_stays_unrescued_when_rescue_finds_nothing(tmp_path, monkeypatch):
    """反向对照：救援窗也只分出一簇时，**不得**替换。

    没有这条，上面的测试无法区分"替换逻辑正确"与"无条件替换"。
    """
    seen = []
    _install_fakes(monkeypatch, seen)
    # 把救援窗也改成单簇
    def fake_extract(audio, rate, vad_segments, spk_model):
        stage = "main" if abs(v11.WIN_S - v12.WIN_S) < 1e-9 else "rescue"
        seen.append(stage)
        emb = _single_cluster()
        return emb, np.arange(len(emb), dtype=float) * 0.15, 0

    monkeypatch.setattr(v11, "_extract_windows", fake_extract)
    result = v12.diarize(_silent_wav(tmp_path), {"vad": None, "spk": None})

    assert seen == ["main", "rescue"], f"阶段序列异常：{seen}"
    assert result["rescued"] is False, "救援没分出多人却替换了"
    assert result["k"] == 1


def test_candidates_criterion_distinguishes_main_from_rescue_table():
    """反证判据 6 是**承重**的：主窗（单簇）那份候选表必须不满足它。

    没有这条，判据 6 可以退化成对任何候选表都成立的恒真式 ——
    初版就是 `result["candidates"] is result["candidates"]`（同对象取两次），
    对任何 payload 都为真，断言不了"候选表已被替换"。
    """
    main_cands, _ = v11._candidate_clusters(_single_cluster(),
                                            max_speakers=v12.MAX_SPEAKERS)
    assert not any(c["k"] >= 2 and c.get("eligible") for c in main_cands), \
        "主窗单簇竟存在合格的 k>=2 —— 判据 6 区分不了主窗表与救援窗表"

    rescue_cands, _ = v11._candidate_clusters(_two_clusters(),
                                              max_speakers=v12.MAX_SPEAKERS)
    assert any(c["k"] >= 2 and c.get("eligible") for c in rescue_cands), \
        "救援窗两簇均衡却无合格 k>=2 —— 判据 6 会误伤正确结果"
