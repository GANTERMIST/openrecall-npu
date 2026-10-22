"""WinAI — локальные аналоги ИИ-функций Windows (Copilot+ PC) на Intel NPU/GPU.

  python -m winai                    панель + значок в трее
  python -m winai panel              только панель (http://127.0.0.1:8765)
  python -m winai doctor             что работает на этом компьютере
  python -m winai recall ...         Recall (capture / search / purge)
  python -m winai clicktodo          Click to Do по горячей клавише
  python -m winai captions ...       Live Captions с переводом
  python -m winai studio ...         Studio Effects для камеры
  python -m winai files ...          поиск по файлам по смыслу
  python -m winai photo ...          upscale / erase / create
  python -m winai chat "вопрос"      спросить локальную LLM
  python -m winai download [--all]   скачать модели заранее (--all — и LLM/Whisper, даже если есть движок)
"""
import sys


def doctor():
    import importlib

    from . import config, llm, power, runtime
    print("Питание:", power.status())
    print("Устройства OpenVINO:", runtime.available() or "openvino не установлен")
    print(f"Общий движок {config.get('engine_url')}:", "запущен" if llm.engine_up(force=True) else "не запущен")
    for kind in power.ORDERS:
        print(f"  порядок устройств для {kind}: {' → '.join(power.devices(kind))}")
    mods = {"openvino_genai": "LLM/Whisper/Cocreator", "mss": "скриншоты", "keyboard": "горячая клавиша",
            "soundcard": "звук для субтитров", "mediapipe": "Studio Effects", "pyvirtualcam": "виртуальная камера",
            "cv2": "фото/камера", "pypdf": "PDF в поиске", "docx": "Word в поиске", "pystray": "трей",
            "winrt.windows.media.ocr": "Windows OCR", "rapidocr_openvino": "запасной OCR"}
    for m, what in mods.items():
        try:
            importlib.import_module(m)
            print(f"  [ok] {what}")
        except Exception as e:
            print(f"  [--] {what}: {type(e).__name__}: {e}")


def download(everything=False):
    """Скачать модели заранее, чтобы функции не ждали при первом использовании."""
    from PIL import Image

    from . import config, embedder, llm, photos, runtime
    steps = [("эмбеддинги для поиска", embedder.ensure_model),
             ("картинки (Cocreator)", lambda: runtime.hf_model(config.get("image_model"))),
             ("удаление объектов", lambda: runtime.url_file(config.get("inpaint_url"), "lama.onnx")),
             ("апскейл", lambda: photos.upscale(Image.new("RGB", (8, 8))))]
    if everything or not llm.engine_up(force=True):
        steps += [("LLM", lambda: runtime.hf_model(config.get("llm_model"))),
                  ("Whisper", lambda: runtime.hf_model(config.get("whisper_model"), config.get("whisper_source"),
                                                       extra=("--weight-format", "fp16")))]
    failed = 0
    for name, fn in steps:
        print(f"== {name}")
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"   не вышло: {e}")
    print("Модели готовы." if not failed else f"Не скачалось: {failed}. Эти функции докачают модель при первом запуске.")


def _log_if_windowless():
    # pythonw.exe (запуск без консоли) не даёт stdout — пишем в файл
    if sys.stdout is None or sys.stderr is None:
        from . import config
        logs = config.DATA / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(logs / "winai.log", "a", encoding="utf-8", buffering=1)


def main(argv=None):
    _log_if_windowless()
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "tray"
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
    elif cmd == "tray":
        from . import tray
        tray.main()
    elif cmd == "panel":
        from . import panel
        panel.run(open_browser="--no-browser" not in argv)
    elif cmd == "doctor":
        doctor()
    elif cmd == "download":
        download("--all" in argv)
    elif cmd == "recall":
        from . import recall
        recall.main(argv)
    elif cmd == "clicktodo":
        from . import clicktodo
        clicktodo.main(argv)
    elif cmd == "captions":
        from . import captions
        captions.main(argv)
    elif cmd == "studio":
        from . import studio
        studio.main(argv)
    elif cmd == "files":
        from . import filesearch
        filesearch.main(argv)
    elif cmd == "photo":
        from . import photos
        photos.main(argv)
    elif cmd == "chat":
        from . import llm
        for piece in llm.stream([{"role": "user", "content": " ".join(argv)}]):
            print(piece, end="", flush=True)
        print()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
