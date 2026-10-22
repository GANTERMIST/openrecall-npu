"""Значок в трее: открыть панель, включить/выключить службы. Панель работает в фоне."""
import threading
import webbrowser

from . import config, panel, services


def _icon():
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, 60, 60), 14, fill=(0, 103, 192))
    d.text((15, 16), "AI", fill="white", font_size=28)
    return img


def main():
    try:
        import pystray
    except ImportError:
        print("pystray не установлен — запускаю только панель")
        panel.run()
        return
    threading.Thread(target=panel.run, kwargs={"open_browser": False}, daemon=True).start()
    url = f"http://127.0.0.1:{config.get('panel_port')}/"

    def toggle(name):
        def fn(icon, item):
            services.stop(name) if services.running(name) else services.start(name)
        return fn

    def checked(name):
        return lambda item: services.running(name)

    def quit_(icon, item):
        services.stop_all()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Открыть панель", lambda i, it: webbrowser.open(url), default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Recall", toggle("recall"), checked=checked("recall")),
        pystray.MenuItem("Click to Do (Ctrl+Alt+Q)", toggle("clicktodo"), checked=checked("clicktodo")),
        pystray.MenuItem("Live Captions", toggle("captions"), checked=checked("captions")),
        pystray.MenuItem("Studio Effects", toggle("studio"), checked=checked("studio")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Выход", quit_),
    )
    pystray.Icon("winai", _icon(), "WinAI", menu).run()
