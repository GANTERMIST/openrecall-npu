"""Веб-панель WinAI: http://127.0.0.1:8765 (только этот компьютер)."""
import io
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file

from . import config, filesearch, llm, power, recall, runtime, services

WEB = Path(__file__).parent / "web"
CLIPS = config.DATA / "clips"
app = Flask(__name__)
_index_job = {"running": False, "log": []}
_gpu_lock = threading.Lock()  # тяжёлые задачи по одной: память общая с iGPU


@app.before_request
def same_origin_only():
    # Чужие сайты в браузере не должны дёргать локальную панель
    origin = request.headers.get("Origin")
    if origin and not origin.startswith((f"http://127.0.0.1:{_port()}", f"http://localhost:{_port()}")):
        abort(403)


def _port():
    return config.get("panel_port")


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


def _image(field):
    from PIL import Image
    f = request.files.get(field)
    return Image.open(f.stream) if f else None


@app.get("/")
def index():
    return send_file(WEB / "index.html")


@app.get("/api/status")
def status():
    return jsonify(power=power.status(), devices=runtime.available(), engine=llm.engine_up(),
                   engine_url=config.get("engine_url"), services=services.status(),
                   loaded=runtime.POOL.names(), indexing=_index_job["running"])


@app.post("/api/service/<name>/<action>")
def service(name, action):
    if name not in services.COMMANDS:
        abort(404)
    if action == "start":
        services.start(name, request.get_json(silent=True) or {})
    else:
        services.stop(name)
    time.sleep(0.3)
    return jsonify(services.status())


@app.get("/api/log/<name>")
def log(name):
    if name not in services.COMMANDS:
        abort(404)
    return Response(services.tail(name), mimetype="text/plain")


@app.get("/api/recall")
def recall_search():
    q = request.args.get("q", "").strip()
    res = recall.search(q, 24) if q else []
    for r in res:
        r["when"] = datetime.fromtimestamp(r["ts"]).strftime("%d.%m %H:%M")
        r["text"] = r["text"][:400]
        r.pop("path")
    return jsonify(res)


@app.get("/img/recall/<int:i>")
def recall_img(i):
    p = recall.shot_path(i)
    if not p or not os.path.exists(p):
        abort(404)
    return send_file(p)


@app.get("/api/files")
def files_search():
    q = request.args.get("q", "").strip()
    return jsonify(filesearch.search(q, 20) if q else [])


@app.post("/api/files/index")
def files_index():
    body = request.get_json(force=True)
    folders = [f for f in body.get("folders", []) if f and os.path.isdir(f)]
    if not folders:
        return jsonify(error="папка не найдена"), 400
    if _index_job["running"]:
        return jsonify(error="индексация уже идёт"), 409

    def job():
        _index_job.update(running=True, log=[])
        try:
            filesearch.index(folders, bool(body.get("images")), log=_index_job["log"].append)
        except Exception as e:
            _index_job["log"].append(f"ошибка: {e}")
        _index_job["running"] = False

    threading.Thread(target=job, daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/files/progress")
def files_progress():
    return jsonify(running=_index_job["running"], log=_index_job["log"][-5:])


@app.post("/api/files/open")
def files_open():
    path = request.get_json(force=True).get("path", "")
    # открывать можно только то, что есть в индексе
    if not filesearch.db().execute("SELECT 1 FROM files WHERE path=?", (path,)).fetchone():
        abort(404)
    if sys.platform == "win32":
        if request.args.get("folder"):
            import subprocess
            subprocess.Popen(["explorer", "/select,", path])
        else:
            os.startfile(path)
    return jsonify(ok=True)


@app.get("/clips/<name>")
def clip(name):
    p = CLIPS / Path(name).name
    if not p.exists():
        abort(404)
    return send_file(p)


@app.post("/api/photo/<op>")
def photo(op):
    from . import photos
    img = _image("image")
    try:
        with _gpu_lock:
            if op == "upscale" and img:
                return _png(photos.upscale(img))
            if op == "erase" and img:
                return _png(photos.erase(img, _image("mask")))
            if op == "create":
                return _png(photos.create(request.form.get("prompt", ""), _image("sketch"),
                                          float(request.form.get("strength", 0.75))))
    except Exception as e:
        return jsonify(error=str(e)), 500
    abort(400)


@app.post("/api/chat")
def chat():
    messages = request.get_json(force=True).get("messages", [])

    def gen():
        try:
            yield from llm.stream(messages, max_tokens=1200)
        except Exception as e:
            yield f"\n[ошибка: {e}]"

    return Response(gen(), mimetype="text/plain; charset=utf-8")


def run(open_browser=True):
    services.restore()
    url = f"http://127.0.0.1:{_port()}/"
    print("Панель:", url)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run("127.0.0.1", _port(), threaded=True)
    finally:
        services.stop_all()
