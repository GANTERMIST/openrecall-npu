"""Настройки WinAI.

Значения по умолчанию ниже. Переопределить можно в config.toml в корне проекта
(те же ключи) или переменными окружения WINAI_<КЛЮЧ> (например WINAI_POWER_MODE=eco).
"""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

DEFAULTS = {
    # Общий движок инференса (npu-llm-hub, OpenAI-совместимый API)
    "engine_url": "http://127.0.0.1:8100/v1",
    "engine_model": "default",
    "engine_whisper": "whisper-small",
    # Если движок не запущен: "local" — поднять свою LLM через OpenVINO GenAI, "off" — не надо
    "llm_fallback": "local",
    "llm_model": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
    "whisper_model": "OpenVINO/whisper-small-fp16-ov",
    "whisper_source": "openai/whisper-small",  # если готового IR нет — экспорт через optimum
    "image_model": "OpenVINO/LCM_Dreamshaper_v7-int8-ov",
    "inpaint_url": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "sr_model": "single-image-super-resolution-1032",
    # auto: от батареи — eco, от сети — performance
    "power_mode": "auto",
    "language": "ru",
    "hotkey_clicktodo": "ctrl+alt+q",
    "panel_port": 8765,
    # Через сколько минут простоя выгружать модель из памяти
    "idle_unload_min": 10,
}


def _load():
    cfg = dict(DEFAULTS)
    f = ROOT / "config.toml"
    if f.exists():
        with open(f, "rb") as fh:
            cfg.update(tomllib.load(fh))
    for k, v in DEFAULTS.items():
        env = os.environ.get("WINAI_" + k.upper())
        if env is not None:
            cfg[k] = type(v)(env)
    return cfg


CFG = _load()


def get(key):
    return CFG[key]
