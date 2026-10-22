"""Live Captions с переводом: субтитры поверх всех окон к звуку из колонок или микрофона.

  python -m winai captions                      # звук компьютера, субтитры как есть
  python -m winai captions --translate ru       # с переводом на русский
  python -m winai captions --source mic

Распознавание: Whisper общего движка (/v1/audio/transcriptions), а если он не запущен —
свой Whisper через OpenVINO GenAI на NPU. Перевод: LLM (движок или локальная).
Окно субтитров можно перетаскивать мышью, двойной клик — закрыть.
"""
import io
import json
import queue
import threading
import time
import urllib.request
import uuid
import wave

import numpy as np

from . import config, llm, runtime

RATE = 16000
BLOCK = 0.1  # секунд в одном блоке


class Segmenter:
    """Режет поток на фразы по паузам: фраза заканчивается после silence_s тишины
    или по достижении max_s. Порог тишины подстраивается под фоновый шум."""

    def __init__(self, silence_s=0.5, max_s=6.0, min_s=0.6):
        self.silence_blocks = int(silence_s / BLOCK)
        self.max_blocks = int(max_s / BLOCK)
        self.min_blocks = int(min_s / BLOCK)
        self.noise = 0.003
        self.buf, self.quiet, self.speaking = [], 0, False

    def push(self, block: np.ndarray):
        """Возвращает готовую фразу (np.float32) или None."""
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        loud = rms > max(self.noise * 3, 0.004)
        if not loud:
            self.noise = 0.95 * self.noise + 0.05 * rms
        if loud:
            self.speaking, self.quiet = True, 0
        elif self.speaking:
            self.quiet += 1
        if self.speaking:
            self.buf.append(block)
        done = self.speaking and (self.quiet >= self.silence_blocks or len(self.buf) >= self.max_blocks)
        if not done:
            return None
        seg = np.concatenate(self.buf)
        enough = len(self.buf) - self.quiet >= self.min_blocks
        self.buf, self.quiet, self.speaking = [], 0, False
        return seg if enough else None


def _wav(samples: np.ndarray) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return bio.getvalue()


def _engine_transcribe(samples, language):
    b = uuid.uuid4().hex
    parts = []
    fields = {"model": config.get("engine_whisper"), "response_format": "json"}
    if language:
        fields["language"] = language
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + _wav(samples) + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    headers = llm._headers()
    headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    url = config.get("engine_url").rstrip("/") + "/audio/transcriptions"
    with urllib.request.urlopen(urllib.request.Request(url, b"".join(parts), headers), timeout=60) as r:
        return json.loads(r.read())["text"].strip()


def _local_whisper():
    import openvino_genai as ov_genai
    path = runtime.hf_model(config.get("whisper_model"), config.get("whisper_source"),
                            extra=("--weight-format", "fp16"))
    pipe, _ = runtime.first_working("audio", lambda d: ov_genai.WhisperPipeline(str(path), d), "whisper")
    return pipe


class Recognizer:
    def __init__(self, language=None):
        self.language = language
        self.use_engine = llm.engine_up(force=True)
        print("[captions] распознавание:", "общий движок" if self.use_engine else "локальный Whisper")

    def __call__(self, samples: np.ndarray) -> str:
        if self.use_engine:
            try:
                return _engine_transcribe(samples, self.language)
            except Exception as e:
                print(f"[captions] движок не ответил ({e}), перехожу на локальный Whisper")
                self.use_engine = False
        pipe = runtime.POOL.get("whisper", _local_whisper)
        kw = {"task": "transcribe"}
        if self.language:
            kw["language"] = f"<|{self.language}|>"
        return pipe.generate(samples.tolist(), **kw).texts[0].strip()


def audio_blocks(source: str):
    """Блоки по BLOCK секунд, моно 16 кГц. source: speakers (звук системы) или mic."""
    import soundcard as sc
    if source == "mic":
        dev = sc.default_microphone()
    else:  # WASAPI loopback: «микрофон», который слышит колонки
        dev = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    n = int(RATE * BLOCK)
    with dev.recorder(samplerate=RATE, channels=1, blocksize=n) as rec:
        while True:
            data = rec.record(numframes=n)
            yield data.reshape(-1).astype(np.float32)


# Фразы, которые Whisper «слышит» в тишине и музыке
HALLUCINATIONS = ("продолжение следует", "субтитры", "спасибо за просмотр", "thank you for watching",
                  "редактор субтитров", "подписывайтесь")


def clean(text: str) -> str:
    t = text.strip()
    if not t or any(h in t.lower() for h in HALLUCINATIONS) and len(t) < 60:
        return ""
    return t


def pipeline(source, language, translate, out: queue.Queue):
    recog = Recognizer(language)
    seg = Segmenter()
    work = queue.Queue(maxsize=3)

    def worker():
        while True:
            samples = work.get()
            t0 = time.time()
            try:
                text = clean(recog(samples))
                if not text:
                    continue
                tr = llm.translate(text, translate, max_tokens=200) if translate and translate != language else ""
                out.put((text, tr))
                print(f"[{time.time()-t0:.1f}с] {text}" + (f"  →  {tr}" if tr else ""))
            except Exception as e:
                out.put((f"ошибка: {e}", ""))

    threading.Thread(target=worker, daemon=True).start()
    for block in audio_blocks(source):
        phrase = seg.push(block)
        if phrase is not None:
            if work.full():  # не успеваем — выкидываем самую старую фразу, чтобы не отставать
                try:
                    work.get_nowait()
                except queue.Empty:
                    pass
            work.put(phrase)


def overlay(out: queue.Queue):
    import tkinter as tk
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.85)
    root.configure(bg="#111")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w = int(sw * 0.7)
    root.geometry(f"{w}x110+{(sw - w) // 2}+{sh - 190}")
    orig = tk.Label(root, text="Слушаю…", fg="#9aa", bg="#111", font=("Segoe UI", 12), wraplength=w - 30,
                    justify="left", anchor="w")
    main = tk.Label(root, text="", fg="#fff", bg="#111", font=("Segoe UI", 18), wraplength=w - 30,
                    justify="left", anchor="w")
    orig.pack(fill="x", padx=14, pady=(8, 0))
    main.pack(fill="x", padx=14)
    drag = {}
    root.bind("<Button-1>", lambda e: drag.update(x=e.x, y=e.y))
    root.bind("<B1-Motion>", lambda e: root.geometry(f"+{e.x_root - drag['x']}+{e.y_root - drag['y']}"))
    root.bind("<Double-Button-1>", lambda e: root.destroy())

    def poll():
        try:
            while True:
                text, tr = out.get_nowait()
                if tr:
                    orig.config(text=text)
                    main.config(text=tr)
                else:
                    orig.config(text=main.cget("text"))
                    main.config(text=text)
        except queue.Empty:
            pass
        root.after(100, poll)

    poll()
    root.mainloop()


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="winai captions")
    ap.add_argument("--source", choices=["speakers", "mic"], default="speakers")
    ap.add_argument("--language", default=None, help="язык речи (ru, en...). Пусто — определить самому")
    ap.add_argument("--translate", default=None, help="перевести на язык (ru, en...)")
    ap.add_argument("--console", action="store_true", help="без окна, только в консоль")
    a = ap.parse_args(argv)
    out = queue.Queue()
    t = threading.Thread(target=pipeline, args=(a.source, a.language, a.translate, out), daemon=True)
    t.start()
    if a.console:
        t.join()
    else:
        overlay(out)
