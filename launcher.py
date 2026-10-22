"""WinAI.exe — лаунчер. Только стандартная библиотека, собирается PyInstaller'ом.

При первом запуске (и после обновления requirements.txt) сам готовит окружение:
скачивает uv, через него Python 3.12 и зависимости в папку программы, показывая окно
с ходом установки. Затем запускает `pythonw -m winai` (трей + панель) и закрывается.
Если WinAI уже запущен — просто открывает панель.

  WinAI.exe              подготовить при необходимости и запустить
  WinAI.exe --setup      принудительно переустановить зависимости
  WinAI.exe --models     заодно скачать все модели заранее
"""
import hashlib
import io
import os
import queue
import socket
import subprocess
import sys
import threading
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
PORT = 8765
WIN = sys.platform == "win32"


def app_dir() -> Path:
    return Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent


APP = app_dir()
VENV = APP / ".venv"
TOOLS = APP / "tools"
STAMP = VENV / ".winai-requirements"


def venv_python(windowless=False) -> Path:
    if WIN:
        return VENV / "Scripts" / ("pythonw.exe" if windowless else "python.exe")
    return VENV / "bin" / "python"


def requirements_hash() -> str:
    return hashlib.sha256((APP / "requirements.txt").read_bytes()).hexdigest()


def needs_setup() -> bool:
    if not venv_python().exists() or not STAMP.exists():
        return True
    return STAMP.read_text().strip() != requirements_hash()


def panel_running() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def ensure_uv(log) -> Path:
    uv = TOOLS / ("uv.exe" if WIN else "uv")
    if uv.exists():
        return uv
    log("Скачиваю uv (менеджер Python)...")
    TOOLS.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(UV_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith("uv.exe"))
        uv.write_bytes(z.read(name))
    return uv


def run(cmd, log, env=None):
    log("> " + " ".join(str(c) for c in cmd))
    flags = subprocess.CREATE_NO_WINDOW if WIN else 0
    p = subprocess.Popen([str(c) for c in cmd], cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, creationflags=flags, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        log(line.rstrip())
    if p.wait() != 0:
        raise RuntimeError(f"команда завершилась с кодом {p.returncode}")


def setup(log, models=False):
    env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(TOOLS / "python"), UV_LINK_MODE="copy",
               PYTHONUTF8="1")
    uv = ensure_uv(log)
    if not venv_python().exists():
        log("Ставлю Python 3.12 и создаю окружение...")
        run([uv, "venv", "--python", "3.12", VENV], log, env)
    log("Ставлю зависимости (первый раз 5–15 минут, ~2 ГБ)...")
    run([uv, "pip", "install", "--python", venv_python(), "-r", APP / "requirements.txt"], log, env)
    STAMP.write_text(requirements_hash())
    if models:
        log("Скачиваю модели (несколько ГБ)...")
        run([venv_python(), "-X", "utf8", "-m", "winai", "download"], log, env)
    log("Готово.")


def launch():
    flags = subprocess.DETACHED_PROCESS if WIN else 0
    subprocess.Popen([str(venv_python(windowless=True)), "-m", "winai"], cwd=APP, creationflags=flags,
                     close_fds=True)


def setup_window(models):
    """Окно с логом установки. Возвращает True, если всё поставилось."""
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("WinAI — подготовка")
    root.geometry("720x420")
    ttk.Label(root, text="Готовлю WinAI к первому запуску. Нужен интернет.", padding=10).pack(anchor="w")
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(fill="x", padx=10)
    bar.start(12)
    text = tk.Text(root, wrap="word", font=("Consolas", 9), height=18)
    text.pack(fill="both", expand=True, padx=10, pady=10)
    lines, result = queue.Queue(), {}

    def work():
        try:
            setup(lines.put, models)
            result["ok"] = True
        except Exception as e:
            lines.put(f"ОШИБКА: {e}")
            result["ok"] = False
        lines.put(None)

    def pump():
        try:
            while True:
                line = lines.get_nowait()
                if line is None:
                    bar.stop()
                    if result.get("ok"):
                        root.after(800, root.destroy)
                    else:
                        ttk.Button(root, text="Закрыть", command=root.destroy).pack(pady=(0, 10))
                    return
                text.insert("end", line + "\n")
                text.see("end")
        except queue.Empty:
            pass
        root.after(100, pump)

    threading.Thread(target=work, daemon=True).start()
    pump()
    root.mainloop()
    return result.get("ok", False)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    force, models = "--setup" in argv, "--models" in argv
    if panel_running() and not (force or models):
        webbrowser.open(f"http://127.0.0.1:{PORT}/")
        return 0
    if force or models or needs_setup():
        if force and STAMP.exists():
            STAMP.unlink()
        if not setup_window(models):
            return 1
    if not panel_running():
        launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
