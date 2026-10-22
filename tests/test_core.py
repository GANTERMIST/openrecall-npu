"""Тесты логики, которая не требует Windows, NPU и скачанных моделей."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image

from winai import captions, config, embedder, photos, power, runtime, services, studio, vecsearch


class FakeEmb:
    """Мешок слов вместо нейросети: одинаковые слова — близкие векторы."""
    device = "FAKE"
    chunk = staticmethod(embedder.chunk)

    def _v(self, text):
        v = np.zeros(embedder.DIM, np.float32)
        for w in text.lower().replace(".", " ").split():
            v[hash(w) % embedder.DIM] += 1
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_passages(self, texts):
        return np.stack([self._v(t) for t in texts]) if texts else np.zeros((0, embedder.DIM), np.float32)

    def embed_query(self, text):
        return self._v(text)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    from winai import filesearch, recall
    monkeypatch.setattr(filesearch, "DB", tmp_path / "files.db")
    monkeypatch.setattr(recall, "DB", tmp_path / "recall.db")
    monkeypatch.setattr(recall, "SHOTS", tmp_path / "shots")
    return tmp_path


# ---------- питание ----------

@pytest.mark.parametrize("bat,expected", [(None, "performance"), ((80, True), "performance"), ((80, False), "eco")])
def test_profile_auto(monkeypatch, bat, expected):
    monkeypatch.setattr(power, "battery", lambda: bat)
    monkeypatch.setitem(config.CFG, "power_mode", "auto")
    assert power.profile() == expected


def test_devices_light_tasks_prefer_npu(monkeypatch):
    for bat in [(50, False), (50, True)]:
        monkeypatch.setattr(power, "battery", lambda b=bat: b)
        assert power.devices("embed")[0] == "NPU"
        assert power.devices("audio")[0] == "NPU"
    monkeypatch.setattr(power, "battery", lambda: (50, True))
    assert power.devices("llm")[0] == "GPU"
    monkeypatch.setattr(power, "battery", lambda: (50, False))
    assert power.devices("llm")[0] == "NPU"


def test_devices_env_override(monkeypatch):
    monkeypatch.setenv("WINAI_DEVICE_LLM", "cpu, gpu")
    assert power.devices("llm") == ["CPU", "GPU"]


def test_first_working_falls_back(monkeypatch):
    monkeypatch.setattr(runtime, "available", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(power, "devices", lambda kind: ["NPU", "GPU", "CPU"])
    tried = []

    def build(d):
        tried.append(d)
        if d == "NPU":
            raise RuntimeError("unsupported op")
        return "ok-" + d

    assert runtime.first_working("x", build) == ("ok-GPU", "GPU")
    assert tried == ["NPU", "GPU"]


def test_recall_pace(monkeypatch):
    from winai import recall
    monkeypatch.setattr(power, "battery", lambda: None)
    assert recall.pace(10) == 10
    monkeypatch.setattr(power, "battery", lambda: (60, False))
    assert recall.pace(10) == 20
    monkeypatch.setattr(power, "battery", lambda: (10, False))
    assert recall.pace(10) is None


def test_pool_unloads_idle(monkeypatch):
    monkeypatch.setitem(config.CFG, "idle_unload_min", 1)
    pool = runtime.Pool()
    pool._start_reaper = lambda: None
    calls = []
    pool.get("m", lambda: calls.append(1) or "model")
    pool.get("m", lambda: calls.append(1) or "model")
    assert calls == [1]
    pool.reap(now=__import__("time").time() + 120)
    assert pool.names() == []


# ---------- поиск ----------

def test_recall_add_and_search(tmp_data):
    from winai import recall
    con = recall.db()
    emb = FakeEmb()
    img = Image.new("RGB", (40, 30))
    recall.add_shot(con, emb, img, "Telegram", "встреча с Петром в пятницу кафе")
    recall.add_shot(con, emb, img, "VS Code", "def main(): print hello world python")
    res = recall.search("кафе пятница", emb=emb)
    assert res[0]["title"] == "Telegram"


def test_filesearch_index_update_delete(tmp_data, tmp_path):
    from winai import filesearch
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rent.txt").write_text("договор аренды квартиры на лето", encoding="utf-8")
    (docs / "code.py").write_text("import numpy as np\nprint('matrix')", encoding="utf-8")
    (docs / "cp.txt").write_bytes("отчёт по практике".encode("cp1251"))
    (docs / "skip.exe").write_bytes(b"MZ")
    emb = FakeEmb()
    assert filesearch.index([docs], emb=emb, log=lambda *_: None) == 3
    assert filesearch.search("аренды квартиры", emb=emb)[0]["path"].endswith("rent.txt")
    assert filesearch.search("отчёт практике", emb=emb)[0]["path"].endswith("cp.txt")
    # повторный проход ничего не переиндексирует
    assert filesearch.index([docs], emb=emb, log=lambda *_: None) == 0
    (docs / "rent.txt").unlink()
    filesearch.index([docs], emb=emb, log=lambda *_: None)
    assert all(not r["path"].endswith("rent.txt") for r in filesearch.search("аренды", emb=emb))


def test_filesearch_docx_pdf(tmp_path):
    from winai import filesearch
    import docx
    d = docx.Document()
    d.add_paragraph("Курсовая работа по нейросетям")
    d.save(tmp_path / "a.docx")
    assert "нейросетям" in filesearch.extract(tmp_path / "a.docx")


def test_hybrid_bonus_for_exact_words(tmp_data):
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.executescript("CREATE TABLE e(id INTEGER, vec BLOB); CREATE VIRTUAL TABLE f USING fts5(text);")
    v = np.ones(4, np.float32) / 2
    for i, t in [(1, "alpha beta"), (2, "gamma delta")]:
        con.execute("INSERT INTO e VALUES(?,?)", (i, v.tobytes()))
        con.execute("INSERT INTO f(rowid,text) VALUES(?,?)", (i, t))
    top = vecsearch.hybrid(con, v, "gamma", "SELECT id, vec FROM e", "f")
    assert top[0][0] == 2


# ---------- субтитры ----------

def test_segmenter_splits_on_silence():
    seg = captions.Segmenter(silence_s=0.3, max_s=5, min_s=0.2)
    n = int(captions.RATE * captions.BLOCK)
    rng = np.random.default_rng(0)
    quiet = lambda: rng.normal(0, 0.001, n).astype(np.float32)
    voice = lambda: (0.2 * np.sin(np.arange(n) / 5)).astype(np.float32)
    out = [seg.push(b) for b in [quiet()] * 10 + [voice()] * 8 + [quiet()] * 5]
    phrases = [o for o in out if o is not None]
    assert len(phrases) == 1
    assert len(phrases[0]) >= 8 * n


def test_segmenter_cuts_long_speech():
    seg = captions.Segmenter(silence_s=0.5, max_s=1.0, min_s=0.2)
    n = int(captions.RATE * captions.BLOCK)
    voice = (0.2 * np.sin(np.arange(n) / 5)).astype(np.float32)
    phrases = [p for p in (seg.push(voice) for _ in range(35)) if p is not None]
    assert len(phrases) == 3


def test_segmenter_ignores_clicks():
    seg = captions.Segmenter(silence_s=0.3, min_s=0.5)
    n = int(captions.RATE * captions.BLOCK)
    click = np.full(n, 0.3, np.float32)
    quiet = np.zeros(n, np.float32)
    assert all(seg.push(b) is None for b in [click] + [quiet] * 6)


def test_clean_hallucinations():
    assert captions.clean("Продолжение следует...") == ""
    assert captions.clean("  Привет, как дела?  ") == "Привет, как дела?"


def test_wav_header():
    data = captions._wav(np.zeros(16000, np.float32))
    assert data[:4] == b"RIFF" and len(data) == 44 + 32000


# ---------- движок (поддельный OpenAI-совместимый сервер) ----------

@pytest.fixture
def fake_engine(monkeypatch):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"default"}]}')

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen[self.path] = (self.headers.get("Content-Type"), body)
            self.send_response(200)
            self.end_headers()
            if self.path.endswith("/chat/completions"):
                for piece in ["При", "вет"]:
                    self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": piece}}]})}\n\n'.encode())
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps({"text": " hello "}).encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setitem(config.CFG, "engine_url", f"http://127.0.0.1:{srv.server_port}/v1")
    from winai import llm
    llm._seen[:] = [0.0, False]
    yield seen
    srv.shutdown()
    llm._seen[:] = [0.0, False]


def test_llm_uses_engine(fake_engine):
    from winai import llm
    assert llm.chat([{"role": "user", "content": "hi"}]) == "Привет"
    body = json.loads(fake_engine["/v1/chat/completions"][1])
    assert body["stream"] and body["model"] == "default"


def test_llm_engine_down_and_fallback_off(monkeypatch):
    from winai import llm
    monkeypatch.setitem(config.CFG, "engine_url", "http://127.0.0.1:9/v1")
    monkeypatch.setitem(config.CFG, "llm_fallback", "off")
    llm._seen[:] = [0.0, False]
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_captions_engine_transcribe(fake_engine):
    rec = captions.Recognizer("ru")
    assert rec.use_engine
    assert rec(np.zeros(8000, np.float32)) == "hello"
    ctype, body = fake_engine["/v1/audio/transcriptions"]
    assert ctype.startswith("multipart/form-data") and b'name="language"\r\n\r\nru' in body and b"RIFF" in body


# ---------- фото ----------

def test_tiled_matches_whole_image():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (101, 157, 3), dtype=np.uint8)
    up = lambda t: np.repeat(np.repeat(t, 2, 0), 2, 1)
    out = photos.tiled(arr, 40, 50, 2, up, margin=4)
    assert out.shape == (202, 314, 3)
    assert np.array_equal(out, up(arr))


def test_crop_box_inside_image():
    m = np.zeros((300, 800), bool)
    m[280:295, 780:795] = True
    t, l, b, r = photos.crop_box(m)
    assert 0 <= t < b <= 300 and 0 <= l < r <= 800
    assert m[t:b, l:r].sum() == m.sum()


def test_erase_and_upscale_fallbacks(monkeypatch):
    # моделей нет (сеть закрыта) — должны сработать классические методы
    monkeypatch.setattr(runtime.POOL, "get", lambda *a: (_ for _ in ()).throw(RuntimeError("нет модели")))
    img = Image.new("RGB", (64, 48), (200, 50, 50))
    img.paste((0, 0, 255), (20, 20, 30, 30))
    mask = Image.new("L", (64, 48), 0)
    mask.paste(255, (20, 20, 30, 30))
    out = np.asarray(photos.erase(img, mask))
    assert abs(int(out[25, 25, 0]) - 200) < 40 and out[25, 25, 2] < 100
    assert photos.upscale(img).size == (256, 192)


# ---------- камера ----------

def test_framer_zooms_on_face_and_stays_inside():
    f = studio.Framer(1280, 720, smooth=1.0)
    x, y, w, h = f.update((1100, 50, 120, 120))
    assert w < 1280 and h < 720
    assert 0 <= x and x + w <= 1280 and 0 <= y and y + h <= 720
    assert f.update(None) == (0, 0, 1280, 720)


def test_compose_blurs_background_only():
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[::2] = 255
    mask = np.zeros((40, 40), np.float32)
    mask[:, :20] = 1
    out = studio.compose(frame, mask)
    assert np.array_equal(out[:, :20], frame[:, :20])
    assert out[:, 25:].std() < frame[:, 25:].std()


# ---------- службы ----------

def test_service_args():
    assert services.build_args("captions", {"source": "mic", "translate": "ru"}) == \
        ["captions", "--source", "mic", "--translate", "ru"]
    assert services.build_args("studio", {"blur": False, "frame": True}) == ["studio", "--no-blur"]
    assert services.build_args("recall", {}) == ["recall", "capture"]
