"""Питание и выбор устройства.

На Core Ultra (Lunar Lake) NPU самый экономный: лёгкие постоянные задачи
(эмбеддинги, Whisper, сегментация) идут на него всегда, чтобы iGPU оставался свободным.
Тяжёлые разовые задачи (LLM, генерация картинок) — на GPU от сети, а от батареи
LLM сначала пробует NPU.
"""
import os

from . import config

# Порядок устройств: вид задачи -> профиль -> список
ORDERS = {
    "embed":  {"eco": ["NPU", "CPU", "GPU"], "performance": ["NPU", "GPU", "CPU"]},
    "audio":  {"eco": ["NPU", "CPU", "GPU"], "performance": ["NPU", "GPU", "CPU"]},
    "vision": {"eco": ["NPU", "CPU", "GPU"], "performance": ["GPU", "NPU", "CPU"]},
    "llm":    {"eco": ["NPU", "GPU", "CPU"], "performance": ["GPU", "NPU", "CPU"]},
    "image":  {"eco": ["GPU", "CPU"],        "performance": ["GPU", "CPU"]},
}


def battery():
    """(процент, от_сети) или None, если батареи нет."""
    try:
        import psutil
        b = psutil.sensors_battery()
    except Exception:
        return None
    if b is None:
        return None
    return round(b.percent), bool(b.power_plugged)


def on_battery() -> bool:
    b = battery()
    return b is not None and not b[1]


def profile() -> str:
    mode = config.get("power_mode")
    if mode in ("eco", "performance"):
        return mode
    return "eco" if on_battery() else "performance"


def devices(kind: str) -> list[str]:
    """Порядок устройств для задачи. WINAI_DEVICE_<KIND>=GPU,CPU переопределяет."""
    env = os.environ.get("WINAI_DEVICE_" + kind.upper())
    if env:
        return [d.strip().upper() for d in env.split(",") if d.strip()]
    return list(ORDERS[kind][profile()])


def status() -> dict:
    b = battery()
    return {"profile": profile(), "battery": b[0] if b else None,
            "plugged": b[1] if b else True}
