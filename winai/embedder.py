"""Эмбеддинги текста на Intel NPU через OpenVINO (с откатом на GPU/CPU).

Модель своя и маленькая (multilingual-e5-small), а не из общего движка: индексы Recall и
поиска по файлам должны всегда строиться одной и той же моделью.
NPU требует статические формы, поэтому модель переформатируется в [1, SEQ_LEN],
а вход добивается паддингом.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import config, runtime

MODEL_ID = "intfloat/multilingual-e5-small"  # ru/en, 384-мерный вектор
MODEL_DIR = config.MODELS / "e5-small-ov"
SEQ_LEN = 128
DIM = 384


def ensure_model():
    if (MODEL_DIR / "openvino_model.xml").exists():
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print("Экспорт модели в OpenVINO IR (один раз)...")
    exe = Path(sys.executable).parent / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")
    subprocess.check_call([
        str(exe), "export", "openvino", "--model", MODEL_ID,
        "--task", "feature-extraction", "--weight-format", "fp16", str(MODEL_DIR),
    ])


def chunk(text: str, max_chunks=6, words=70):
    w = text.split()
    chunks = [" ".join(w[i:i + words]) for i in range(0, len(w), words)]
    return chunks[:max_chunks]


class Embedder:
    def __init__(self):
        from transformers import AutoTokenizer
        ensure_model()
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = runtime.core().read_model(MODEL_DIR / "openvino_model.xml")
        model.reshape({inp.get_any_name(): [1, SEQ_LEN] for inp in model.inputs})
        self.compiled, self.device = runtime.compile_model(model, "embed", "embedder")
        self.names = [i.get_any_name() for i in self.compiled.inputs]

    def _encode_one(self, text: str) -> np.ndarray:
        enc = self.tok(text, padding="max_length", truncation=True,
                       max_length=SEQ_LEN, return_tensors="np")
        feed = {n: enc[n].astype(np.int64) for n in self.names if n in enc}
        if "token_type_ids" in self.names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros((1, SEQ_LEN), np.int64)
        out = self.compiled(feed)[self.compiled.outputs[0]]  # [1, L, H]
        mask = enc["attention_mask"][..., None].astype(np.float32)
        vec = (out * mask).sum(1) / max(mask.sum(), 1.0)
        vec = vec[0]
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed_passages(self, texts):
        if not texts:
            return np.zeros((0, DIM), np.float32)
        return np.stack([self._encode_one("passage: " + t) for t in texts])

    def embed_query(self, text):
        return self._encode_one("query: " + text)

    chunk = staticmethod(chunk)


_shared = None


def shared() -> Embedder:
    """Один эмбеддер на процесс."""
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
