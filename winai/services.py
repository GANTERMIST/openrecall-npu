"""Фоновые службы (Recall, Click to Do, субтитры, камера) как отдельные процессы."""
import json
import subprocess
import sys

from . import config

COMMANDS = {
    "recall": ["recall", "capture"],
    "clicktodo": ["clicktodo"],
    "captions": ["captions"],
    "studio": ["studio"],
}
AUTOSTART = config.DATA / "autostart.json"
LOGS = config.DATA / "logs"

_procs = {}  # имя -> (Popen, args)


def build_args(name, opts):
    args = list(COMMANDS[name])
    opts = opts or {}
    if name == "captions":
        if opts.get("source") == "mic":
            args += ["--source", "mic"]
        if opts.get("translate"):
            args += ["--translate", str(opts["translate"])]
    elif name == "studio":
        if opts.get("blur") is False:
            args.append("--no-blur")
        if opts.get("frame") is False:
            args.append("--no-frame")
        if opts.get("preview"):
            args.append("--preview")
    elif name == "recall" and opts.get("interval"):
        args += ["--interval", str(float(opts["interval"]))]
    return args


def start(name, opts=None):
    if running(name):
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{name}.log", "a", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    args = build_args(name, opts)
    p = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "winai", *args], cwd=str(config.ROOT),
                         stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    _procs[name] = (p, opts or {})
    _save()


def stop(name):
    item = _procs.pop(name, None)
    if item:
        item[0].terminate()
    _save()


def running(name) -> bool:
    item = _procs.get(name)
    return bool(item) and item[0].poll() is None


def status():
    return {n: {"running": running(n), "opts": _procs.get(n, (None, {}))[1]} for n in COMMANDS}


def tail(name, lines=30):
    f = LOGS / f"{name}.log"
    if not f.exists():
        return ""
    return "\n".join(f.read_text(encoding="utf-8", errors="ignore").splitlines()[-lines:])


def _save():
    AUTOSTART.parent.mkdir(parents=True, exist_ok=True)
    AUTOSTART.write_text(json.dumps({n: o for n, (p, o) in _procs.items()}), encoding="utf-8")


def restore():
    """Запустить службы, которые были включены в прошлый раз."""
    if AUTOSTART.exists():
        for name, opts in json.loads(AUTOSTART.read_text(encoding="utf-8")).items():
            if name in COMMANDS:
                start(name, opts)


def stop_all():
    for name in list(_procs):
        item = _procs.pop(name)
        item[0].terminate()
