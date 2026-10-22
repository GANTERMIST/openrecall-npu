"""Click to Do: горячая клавиша → выделить область экрана → что сделать с этим.

  python -m winai clicktodo            # ждёт Ctrl+Alt+Q (меняется в config.toml: hotkey_clicktodo)
  python -m winai clicktodo --now      # сразу выделение, без ожидания клавиши

Действия: скопировать текст, объяснить, кратко, перевести, ответить на вопрос с экрана,
найти в интернете, найти в Recall, открыть картинку во вкладке «Фото» (увеличить/убрать объект).
Esc или правая кнопка — отмена выделения.
"""
import queue
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime

from . import config, llm

ACTIONS = {
    "Объяснить": "Объясни простыми словами, что это и что тут важно. Отвечай по-русски, кратко.",
    "Кратко": "Перескажи суть в 3–5 пунктах по-русски.",
    "Ответить": "На экране вопрос, задание или сообщение. Дай ответ или решение по-русски.",
    "Переписать": "Перепиши текст грамотнее и понятнее, сохрани язык оригинала. Выведи только текст.",
}


def dpi_aware():
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()


def grab_screen():
    import mss
    from PIL import Image
    with mss.mss() as sct:
        mon = sct.monitors[0]
        raw = sct.grab(mon)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"), mon


def norm_rect(x0, y0, x1, y1):
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def select_region(root):
    """Полноэкранный стоп-кадр, пользователь тянет прямоугольник. Возвращает PIL-картинку или None."""
    import tkinter as tk

    from PIL import ImageEnhance, ImageTk
    shot, mon = grab_screen()
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.attributes("-topmost", True)
    win.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
    dim = ImageTk.PhotoImage(ImageEnhance.Brightness(shot).enhance(0.55))
    cv = tk.Canvas(win, highlightthickness=0, cursor="crosshair")
    cv.pack(fill="both", expand=True)
    cv.create_image(0, 0, image=dim, anchor="nw")
    cv.create_text(mon["width"] // 2, 40, text="Выделите область · Esc — отмена", fill="white",
                   font=("Segoe UI", 16))
    st = {"rect": None, "x": 0, "y": 0, "res": None, "bright": None}

    def down(e):
        st["x"], st["y"] = e.x, e.y
        st["rect"] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#4cc2ff", width=2)

    def move(e):
        if st["rect"]:
            cv.coords(st["rect"], st["x"], st["y"], e.x, e.y)

    def up(e):
        x0, y0, x1, y1 = norm_rect(st["x"], st["y"], e.x, e.y)
        if x1 - x0 > 5 and y1 - y0 > 5:
            st["res"] = shot.crop((x0, y0, x1, y1))
        win.destroy()

    cv.bind("<ButtonPress-1>", down)
    cv.bind("<B1-Motion>", move)
    cv.bind("<ButtonRelease-1>", up)
    win.bind("<Escape>", lambda e: win.destroy())
    cv.bind("<Button-3>", lambda e: win.destroy())
    win.focus_force()
    win.grab_set()
    root.wait_window(win)
    return st["res"]


def save_clip(img):
    d = config.DATA / "clips"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"clip_{datetime.now():%Y%m%d_%H%M%S}.png"
    img.save(p)
    return p


def result_window(root, img):
    import tkinter as tk
    from tkinter import ttk

    from .ocr import ocr
    text = ocr(img).strip()
    win = tk.Toplevel(root)
    win.title("Click to Do")
    win.attributes("-topmost", True)
    win.geometry("560x460")
    top = ttk.Frame(win, padding=8)
    top.pack(fill="x")
    body = tk.Text(win, wrap="word", font=("Segoe UI", 11), height=10)
    body.pack(fill="both", expand=True, padx=8)
    body.insert("1.0", text or "(текст не найден — можно открыть картинку во вкладке «Фото»)")
    status = ttk.Label(win, text=f"распознано слов: {len(text.split())}", padding=6)
    status.pack(fill="x")
    port = config.get("panel_port")

    def current():
        return body.get("1.0", "end").strip()

    def show(answer_iter, title):
        status.config(text=title + "…")
        out = queue.Queue()

        def run():
            try:
                for piece in answer_iter():
                    out.put(piece)
            except Exception as e:
                out.put(f"\n[ошибка: {e}]")
            out.put(None)

        threading.Thread(target=run, daemon=True).start()
        body.delete("1.0", "end")

        def pump():
            try:
                while True:
                    piece = out.get_nowait()
                    if piece is None:
                        status.config(text=title + " — готово")
                        return
                    body.insert("end", piece)
                    body.see("end")
            except queue.Empty:
                pass
            win.after(50, pump)

        pump()

    def act(name):
        src = text or current()
        if name == "Перевести":
            target = "en" if _mostly_cyrillic(src) else config.get("language")
            show(lambda: iter([llm.translate(src, target)]), "Перевод")
        else:
            show(lambda: llm.stream([{"role": "system", "content": ACTIONS[name]},
                                     {"role": "user", "content": src}]), name)

    def copy():
        win.clipboard_clear()
        win.clipboard_append(current())
        status.config(text="скопировано")

    buttons = [("Копировать", copy)]
    buttons += [(n, lambda n=n: act(n)) for n in ("Объяснить", "Кратко", "Перевести", "Ответить", "Переписать")]
    buttons += [
        ("Найти в сети", lambda: webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote(current()[:200]))),
        ("Найти в Recall", lambda: webbrowser.open(f"http://127.0.0.1:{port}/#recall?q=" + urllib.parse.quote(current()[:100]))),
        ("Фото…", lambda: webbrowser.open(f"http://127.0.0.1:{port}/#photo?file=" + urllib.parse.quote(save_clip(img).name))),
    ]
    for i, (label, fn) in enumerate(buttons):
        ttk.Button(top, text=label, command=fn).grid(row=i // 5, column=i % 5, padx=2, pady=2, sticky="ew")
    win.bind("<Escape>", lambda e: win.destroy())
    win.focus_force()


def _mostly_cyrillic(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum("а" <= c.lower() <= "я" or c in "ёЁ" for c in letters) > len(letters) / 2


def main(argv):
    import argparse
    import tkinter as tk
    ap = argparse.ArgumentParser(prog="winai clicktodo")
    ap.add_argument("--now", action="store_true")
    a = ap.parse_args(argv)
    dpi_aware()
    root = tk.Tk()
    root.withdraw()
    requests = queue.Queue()

    def handle():
        img = select_region(root)
        if img is not None:
            result_window(root, img)

    def poll():
        try:
            while True:
                requests.get_nowait()
                handle()
        except queue.Empty:
            pass
        root.after(100, poll)

    if a.now:
        root.after(10, lambda: (handle(), root.after(300, _quit_when_closed, root)))
    else:
        import keyboard  # глобальная горячая клавиша
        hk = config.get("hotkey_clicktodo")
        keyboard.add_hotkey(hk, lambda: requests.put(1))
        print(f"Click to Do: нажмите {hk}. Ctrl+C в консоли — выход.")
        poll()
    root.mainloop()


def _quit_when_closed(root):
    if not [w for w in root.winfo_children() if w.winfo_exists()]:
        root.destroy()
    else:
        root.after(300, _quit_when_closed, root)
