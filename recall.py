"""OpenRecall: локальный аналог Windows Recall. Всё хранится на этом компьютере.

  python recall.py capture [--interval 10]   # фоновая запись экрана
  python recall.py search "что искал"        # поиск из консоли
  python recall.py serve                     # веб-интерфейс http://127.0.0.1:8765
  python recall.py purge --days 30           # удалить старше N дней
"""
import argparse
import ctypes
import sqlite3
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SHOTS = DATA / "shots"
DB = DATA / "recall.db"

# Не записывать окна, в заголовке которых есть эти слова (приватность)
BLOCKLIST = ["password", "пароль", "incognito", "инкогнито", "inprivate", "keepass",
             "bitwarden", "1password", "банк", "bank", "сбербанк", "tinkoff", "т-банк"]


def db():
    DATA.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS shots(id INTEGER PRIMARY KEY, ts REAL, path TEXT, title TEXT, text TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, title, content='shots', content_rowid='id');
    CREATE TABLE IF NOT EXISTS emb(shot_id INTEGER, vec BLOB);
    CREATE INDEX IF NOT EXISTS emb_shot ON emb(shot_id);
    """)
    return con


def active_title() -> str:
    u = ctypes.windll.user32
    h = u.GetForegroundWindow()
    n = u.GetWindowTextLengthW(h)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(h, buf, n + 1)
    return buf.value


def small(img):
    return np.asarray(img.convert("L").resize((64, 36)), dtype=np.int16)


def capture(interval: float, threshold: float):
    import mss
    from PIL import Image

    from embedder import Embedder
    from ocr import ocr

    emb = Embedder()
    con = db()
    last = None
    print(f"Запись каждые {interval}с. Ctrl+C — остановить.")
    with mss.mss() as sct:
        while True:
            try:
                title = active_title()
                if any(b in title.lower() for b in BLOCKLIST):
                    time.sleep(interval)
                    continue
                raw = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                s = small(img)
                if last is not None and np.abs(s - last).mean() < threshold:
                    time.sleep(interval)
                    continue
                last = s
                t0 = time.time()
                text = ocr(img)
                ts = time.time()
                path = SHOTS / (datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S") + ".jpg")
                img.save(path, quality=65)
                cur = con.execute("INSERT INTO shots(ts,path,title,text) VALUES(?,?,?,?)",
                                  (ts, str(path), title, text))
                sid = cur.lastrowid
                con.execute("INSERT INTO fts(rowid,text,title) VALUES(?,?,?)", (sid, text, title))
                chunks = emb.chunk(f"{title}. {text}")
                for v in emb.embed_passages(chunks):
                    con.execute("INSERT INTO emb VALUES(?,?)", (sid, v.astype(np.float32).tobytes()))
                con.commit()
                print(f"[{datetime.now():%H:%M:%S}] {title[:50]!r} слов={len(text.split())} {time.time()-t0:.1f}с")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("ошибка:", repr(e))
            time.sleep(interval)


def search(query: str, k: int = 10, emb=None):
    con = db()
    from embedder import Embedder
    emb = emb or Embedder()
    q = emb.embed_query(query)
    rows = con.execute("SELECT shot_id, vec FROM emb").fetchall()
    scores = {}
    if rows:
        ids = np.array([r[0] for r in rows])
        mat = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
        sims = mat @ q
        for sid, s in zip(ids, sims):
            scores[int(sid)] = max(scores.get(int(sid), -1), float(s))
    # ключевой поиск даёт бонус за точное совпадение слов
    words = [w for w in query.replace('"', " ").split() if len(w) > 2]
    if words:
        try:
            fq = " OR ".join(f'"{w}"*' for w in words)
            for sid, in con.execute("SELECT rowid FROM fts WHERE fts MATCH ? LIMIT 100", (fq,)):
                scores[sid] = scores.get(sid, 0) + 0.15
        except sqlite3.OperationalError:
            pass
    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    out = []
    for sid, sc in top:
        r = con.execute("SELECT ts,path,title,text FROM shots WHERE id=?", (sid,)).fetchone()
        out.append(dict(id=sid, score=sc, ts=r[0], path=r[1], title=r[2], text=r[3]))
    return out


def serve(port: int):
    from flask import Flask, render_template_string, request, send_file

    from embedder import Embedder
    emb = Embedder()
    app = Flask(__name__)
    page = """<!doctype html><meta charset=utf-8><title>OpenRecall</title>
    <style>body{font-family:Segoe UI,sans-serif;background:#111;color:#eee;max-width:1100px;margin:20px auto}
    input{width:75%;padding:10px;font-size:16px}button{padding:10px 18px;font-size:16px}
    .c{display:inline-block;width:340px;margin:8px;vertical-align:top;background:#1c1c1c;padding:8px;border-radius:8px}
    img{width:100%;border-radius:4px}small{color:#999}</style>
    <h2>OpenRecall <small>({{dev}})</small></h2>
    <form><input name=q value="{{q}}" autofocus placeholder="что вы видели на экране?"><button>Найти</button></form>
    {% for r in res %}<div class=c><a href="/img/{{r.id}}" target=_blank><img src="/img/{{r.id}}"></a>
    <div>{{r.title[:60]}}</div><small>{{r.when}} · {{'%.2f'|format(r.score)}}</small></div>{% endfor %}"""

    @app.route("/")
    def index():
        q = request.args.get("q", "")
        res = search(q, 18, emb) if q else []
        for r in res:
            r["when"] = datetime.fromtimestamp(r["ts"]).strftime("%d.%m %H:%M")
        return render_template_string(page, q=q, res=res, dev=emb.device)

    @app.route("/img/<int:i>")
    def img(i):
        p = db().execute("SELECT path FROM shots WHERE id=?", (i,)).fetchone()
        return send_file(p[0])

    app.run("127.0.0.1", port)


def purge(days: int):
    con = db()
    cut = time.time() - days * 86400
    for sid, p in con.execute("SELECT id,path FROM shots WHERE ts<?", (cut,)).fetchall():
        Path(p).unlink(missing_ok=True)
        con.execute("DELETE FROM emb WHERE shot_id=?", (sid,))
        con.execute("DELETE FROM shots WHERE id=?", (sid,))
    con.execute("INSERT INTO fts(fts) VALUES('rebuild')")
    con.commit()
    print("готово")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--interval", type=float, default=10); c.add_argument("--threshold", type=float, default=3.0)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    v = sub.add_parser("serve"); v.add_argument("--port", type=int, default=8765)
    p = sub.add_parser("purge"); p.add_argument("--days", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "capture":
        capture(a.interval, a.threshold)
    elif a.cmd == "search":
        for r in search(a.query, a.k):
            print(f"{r['score']:.2f} {datetime.fromtimestamp(r['ts']):%d.%m %H:%M} {r['title'][:50]} -> {r['path']}")
    elif a.cmd == "serve":
        serve(a.port)
    else:
        purge(a.days)
