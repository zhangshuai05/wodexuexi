# -*- coding: utf-8 -*-
"""Fun-ASR-Nano 转写后端适配器（选定后端，2026-09-08 AB 胜出：安静 2m found 54→58-60%、
嘈杂 38→43%，SenseVoice/SEACO 全文路径均不敌）：
对外保持 SEACO 风格接口 generate(input=ndarray|path) -> [{"text": ...}]，
内部写临时 wav 走 FunASR（其 ndarray 输入在 funasr 1.3.14 下有 NoneType bug），
并剥掉 <|xx|> 特殊 token。三个 Claude 版本共用同一转写模型。
模型路径：只读共享 modelscope 缓存中的 Fun-ASR-Nano 快照。
"""
import os, re, tempfile, threading

HERE = os.path.dirname(os.path.abspath(__file__))
FUNASR_CACHE = os.path.join(
    r"C:\Users\EDY\.cache\modelscope\models",
    "FunAudioLLM--Fun-ASR-Nano-2512", "snapshots", "master")
SPECIAL = re.compile(r"<\|[^|]*\|>")

_counter = 0
_lock = threading.Lock()


def clean(s):
    return re.sub(r"\s+", "", SPECIAL.sub("", s or ""))


def load_funasr():
    """加载 Fun-ASR-Nano（返回 funasr AutoModel 实例，未包装）。"""
    from funasr import AutoModel
    model_dir = FUNASR_CACHE
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Fun-ASR-Nano 快照缺失: {model_dir}（需先下载 FunAudioLLM/Fun-ASR-Nano-2512）")
    return AutoModel(model=model_dir, disable_update=True, disable_pbar=True)


class FunASRAdapter:
    """SEACO 风格接口适配：generate(input=ndarray|path|list) -> [{"text": ...}]。
    每段写临时 wav（FunASR 的 ndarray 输入路径有 bug），调用后即删。"""

    asr_backend = "funasr_seg"

    def __init__(self, model=None, tmpdir=None):
        self._m = model if model is not None else load_funasr()
        self._tmp = tmpdir or os.path.join(tempfile.gettempdir(), "a2ovl_funasr_tmp")
        os.makedirs(self._tmp, exist_ok=True)

    def _one(self, inp):
        global _counter
        if isinstance(inp, str):
            path = inp
            owned = False
        else:
            import soundfile as sf
            import numpy as np
            x = np.asarray(inp)
            if x.ndim > 1:
                x = x[:, 0]
            with _lock:
                _counter += 1
                path = os.path.join(
                    self._tmp, f"seg_{os.getpid()}_{_counter}.wav")
            sf.write(path, x.astype(np.float32), 16000)
            owned = True
        try:
            r = self._m.generate(input=path)
            return clean((r[0].get("text") if r else "") or "")
        finally:
            if owned:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def generate(self, input=None, **kwargs):
        if isinstance(input, (list, tuple)):
            return [{"text": self._one(x)} for x in input]
        return [{"text": self._one(input)}]


def wrap_funasr(model):
    return FunASRAdapter(model=model)
