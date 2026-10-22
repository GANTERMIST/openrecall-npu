"""Recall: локальный аналог Windows Recall. Всё хранится на этом компьютере.

  python -m winai recall capture [--interval 10]   # фоновая запись экрана
  python -m winai recall search "что искал"        # поиск из консоли
  python -m winai recall purge --days 30           # удалить старше N дней
Поиск с картинками — во вкладке «Recall» панели (python -m winai panel).
"""
import ctypes
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, power, vecsearch

DATA = config.DATA
SHOTS = DATA / "shots"
DB = DATA / "recall.db"

# Не записывать окна, в заголовке которых есть эти слова (приватность)
BLOCKLIST = ["password", "пароль", "incognito", "инкогнито", "inprivate", "keepass",
             "bitwarden", "1password", "банк", "bank", "сбербанк", "tinkoff", "т-банк"]
LOW_BATTERY = 20  # ниже этого процента от батареи запись ставится на паузу


def db():
    SHOTS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS shots(id INTEGER PRIMARY KEY, ts REAL, path TEXT, title TEXT, text TEXT);
    CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, title, content='shots', content_rowid='id');
    CREATE TABLE IF NOT EXISTS emb(shot_id INTEGER, vec BLOB);
    CREATE INDEX IF NOT EXISTS emb_shot ON emb(shot_id);
    """)
    return con


def active_title() -> str:
    if sys.platform != "win32":
        return ""
    u = ctypes.windll.user32
    h = u.GetForegroundWindow()
    n = u.GetWindowTextLengthW(h)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(h, buf, n + 1)
    return buf.value


def blocked(title: str) -> bool:
    t = title.lower()
    return any(b in t for b in BLOCKLIST)


def small(img):
    return np.asarray(img.convert("L").resize((64, 36)), dtype=np.int16)


def pace(interval: float):
    """Интервал записи с учётом питания: от батареи вдвое реже, при низком заряде — пауза (None)."""
    b = power.battery()
    if b is None or b[1]:
        return interval
    if b[0] < LOW_BATTERY:
        return None
    return interval * 2


def capture(interval: float, threshold: float):
    import mss
    from PIL import Image

    from .embedder import shared
    from .ocr import ocr

    emb = shared()
    con = db()
    last = None
    print(f"Запись каждые {interval}с (от батареи — реже). Ctrl+C — остановить.")
    with mss.mss() as sct:
        while True:
            wait = pace(interval)
            try:
                if wait is None:
                    time.sleep(60)
                    continue
                title = active_title()
                if blocked(title):
                    time.sleep(wait)
                    continue
                raw = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                s = small(img)
                if last is not None and np.abs(s - last).mean() < threshold:
                    time.sleep(wait)
                    continue
                last = s
                t0 = time.time()
                text = ocr(img)
                add_shot(con, emb, img, title, text)
                print(f"[{datetime.now():%H:%M:%S}] {title[:50]!r} слов={len(text.split())} {time.time()-t0:.1f}с")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("ошибка:", repr(e))
            time.sleep(wait or 0)


def add_shot(con, emb, img, title, text, ts=None):
    ts = ts or time.time()
    path = SHOTS / (datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
    img.save(path, quality=65)
    sid = con.execute("INSERT INTO shots(ts,path,title,text) VALUES(?,?,?,?)",
                      (ts, str(path), title, text)).lastrowid
    con.execute("INSERT INTO fts(rowid,text,title) VALUES(?,?,?)", (sid, text, title))
    for v in emb.embed_passages(emb.chunk(f"{title}. {text}")):
        con.execute("INSERT INTO emb VALUES(?,?)", (sid, v.astype(np.float32).tobytes()))
    con.commit()
    return sid


def search(query: str, k: int = 10, emb=None):
    from .embedder import shared
    con = db()
    emb = emb or shared()
    top = vecsearch.hybrid(con, emb.embed_query(query), query, "SELECT shot_id, vec FROM emb", "fts", k)
    out = []
    for sid, sc in top:
        r = con.execute("SELECT ts,path,title,text FROM shots WHERE id=?", (sid,)).fetchone()
        if r:
            out.append(dict(id=sid, score=sc, ts=r[0], path=r[1], title=r[2], text=r[3]))
    return out


def shot_path(i: int):
    r = db().execute("SELECT path FROM shots WHERE id=?", (i,)).fetchone()
    return r and r[0]


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


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="winai recall")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--interval", type=float, default=10)
    c.add_argument("--threshold", type=float, default=3.0)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=8)
    p = sub.add_parser("purge")
    p.add_argument("--days", type=int, default=30)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        capture(a.interval, a.threshold)
    elif a.cmd == "search":
        for r in search(a.query, a.k):
            print(f"{r['score']:.2f} {datetime.fromtimestamp(r['ts']):%d.%m %H:%M} {r['title'][:50]} -> {r['path']}")
    else:
        purge(a.days)
