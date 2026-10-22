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
