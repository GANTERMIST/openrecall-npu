"""Эмбеддинги текста на Intel NPU через OpenVINO (с откатом на GPU/CPU).

NPU требует статические формы, поэтому модель переформатируется в [1, SEQ_LEN],
а вход добивается паддингом.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"  # ru/en, 384-мерный вектор
MODEL_DIR = Path(__file__).parent / "models" / "e5-small-ov"
SEQ_LEN = 128


def ensure_model():
    if (MODEL_DIR / "openvino_model.xml").exists():
        return
    MODEL_DIR.parent.mkdir(exist_ok=True)
    print("Экспорт модели в OpenVINO IR (один раз)...")
    exe = Path(sys.executable).parent / "optimum-cli.exe"
    subprocess.check_call([
        str(exe), "export", "openvino", "--model", MODEL_ID,
        "--task", "feature-extraction", "--weight-format", "fp16", str(MODEL_DIR),
    ])


class Embedder:
    def __init__(self, device_priority=("NPU", "GPU", "CPU")):
        ensure_model()
        core = ov.Core()
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = core.read_model(MODEL_DIR / "openvino_model.xml")
        model.reshape({inp.get_any_name(): [1, SEQ_LEN] for inp in model.inputs})
        self.device = None
        for dev in device_priority:
            if dev not in core.available_devices:
                continue
            try:
                self.compiled = core.compile_model(model, dev)
                self.device = dev
                break
            except Exception as e:  # NPU-драйвер может не поддержать какой-то оператор
                print(f"[embedder] {dev} не удалось: {str(e).splitlines()[0]}")
        if self.device is None:
            raise RuntimeError("Нет подходящего устройства OpenVINO")
        self.names = [i.get_any_name() for i in self.compiled.inputs]
        print(f"[embedder] устройство: {self.device}")

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
        return np.stack([self._encode_one("passage: " + t) for t in texts]) if texts else np.zeros((0, 384), np.float32)

    def embed_query(self, text):
        return self._encode_one("query: " + text)

    def chunk(self, text: str, max_chunks=6, words=70):
        w = text.split()
        chunks = [" ".join(w[i:i + words]) for i in range(0, len(w), words)]
        return chunks[:max_chunks]
