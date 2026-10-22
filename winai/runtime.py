"""Общие помощники OpenVINO: ядро с кэшем, компиляция с откатом устройств,
скачивание моделей и выгрузка простаивающих моделей (ОЗУ на ноутбуке общая с iGPU)."""
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import config, power

_core = None


def core():
    global _core
    if _core is None:
        import openvino as ov
        _core = ov.Core()
        cache = config.MODELS / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        # кэш скомпилированных блобов: NPU компилирует модель долго только в первый раз
        _core.set_property({"CACHE_DIR": str(cache)})
    return _core


def available() -> list[str]:
    try:
        return core().available_devices
    except Exception:
        return []


def first_working(kind: str, build, label: str = ""):
    """Пробует build(device) по порядку power.devices(kind). Возвращает (результат, устройство)."""
    have = available()
    errors = []
    for dev in power.devices(kind):
        if not any(h.startswith(dev) for h in have):
            continue
        try:
            obj = build(dev)
            print(f"[{label or kind}] устройство: {dev}")
            return obj, dev
        except Exception as e:  # драйвер NPU может не поддержать модель — идём дальше
            errors.append(f"{dev}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            print(f"[{label or kind}] {errors[-1]}")
    raise RuntimeError(f"{label or kind}: ни одно устройство не подошло ({'; '.join(errors) or 'нет устройств'})")


def compile_model(model, kind: str, label: str = ""):
    props = {"PERFORMANCE_HINT": "LATENCY"}
    return first_working(kind, lambda d: core().compile_model(model, d, props), label)


def hf_model(repo: str, source: str | None = None, task: str | None = None, extra=()) -> Path:
    """Папка с моделью OpenVINO. repo вида OpenVINO/... скачивается готовым,
    иначе (или если скачать не вышло) source экспортируется через optimum-cli."""
    dest = config.MODELS / repo.replace("/", "--")
    if dest.exists() and any(dest.glob("*.xml")):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        print(f"Скачиваю {repo} (один раз)...")
        snapshot_download(repo, local_dir=str(dest))
        if any(dest.glob("*.xml")):
            return dest
    except Exception as e:
        print(f"Не удалось скачать {repo}: {e}")
        if not source:
            raise
    print(f"Экспорт {source} в OpenVINO IR (один раз, может занять несколько минут)...")
    exe = Path(sys.executable).parent / ("optimum-cli.exe" if sys.platform == "win32" else "optimum-cli")
    cmd = [str(exe), "export", "openvino", "--model", source, *extra]
    if task:
        cmd += ["--task", task]
    subprocess.check_call(cmd + [str(dest)])
    return dest


def url_file(url: str, name: str | None = None) -> Path:
    dest = config.MODELS / "files" / (name or url.rsplit("/", 1)[-1])
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


class Pool:
    """Лениво загружает модели и выгружает те, что не использовались idle_unload_min минут."""

    def __init__(self):
        self._items = {}  # имя -> [объект, время последнего доступа]
        self._lock = threading.Lock()
        self._reaper = None

    def get(self, name, loader):
        with self._lock:
            if name not in self._items:
                self._items[name] = [loader(), time.time()]
            item = self._items[name]
            item[1] = time.time()
            self._start_reaper()
            return item[0]

    def drop(self, name):
        with self._lock:
            self._items.pop(name, None)

    def names(self):
        return list(self._items)

    def reap(self, now=None):
        limit = config.get("idle_unload_min") * 60
        now = now or time.time()
        with self._lock:
            for name in [n for n, (_, t) in self._items.items() if now - t > limit]:
                print(f"[pool] выгружаю {name} (простаивает)")
                del self._items[name]

    def _start_reaper(self):
        if self._reaper is None:
            def loop():
                while True:
                    time.sleep(60)
                    self.reap()
            self._reaper = threading.Thread(target=loop, daemon=True)
            self._reaper.start()


POOL = Pool()
