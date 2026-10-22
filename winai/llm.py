"""Локальная LLM (аналог Phi Silica).

Сначала используется общий движок npu-llm-hub (OpenAI-совместимый API, по умолчанию
http://127.0.0.1:8100/v1), чтобы модель не грузилась в память дважды. Если он не запущен
и llm_fallback = "local", своя модель поднимается через OpenVINO GenAI.
"""
import json
import os
import queue
import threading
import time
import urllib.request

from . import config, runtime

_seen = [0.0, False]  # время проверки, доступен ли движок


def _headers():
    h = {"Content-Type": "application/json"}
    key = os.environ.get("NPU_HUB_API_KEY")
    if key:
        h["Authorization"] = "Bearer " + key
    return h


def _req(path, body=None, timeout=300):
    url = config.get("engine_url").rstrip("/") + path
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(url, data, _headers()), timeout=timeout)


def engine_up(force=False) -> bool:
    if not force and time.time() - _seen[0] < 30:
        return _seen[1]
    try:
        with _req("/models", timeout=1.5) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    _seen[:] = [time.time(), ok]
    return ok


def _engine_stream(messages, max_tokens, temperature):
    body = {"model": config.get("engine_model"), "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": temperature}
    with _req("/chat/completions", body) as r:
        for line in r:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except (ValueError, KeyError, IndexError):
                continue
            if delta.get("content"):
                yield delta["content"]


def _local_pipe():
    import openvino_genai as ov_genai
    path = runtime.hf_model(config.get("llm_model"))
    pipe, _ = runtime.first_working("llm", lambda d: ov_genai.LLMPipeline(str(path), d), "llm")
    return pipe


def _local_stream(messages, max_tokens, temperature):
    import openvino_genai as ov_genai
    pipe = runtime.POOL.get("llm", _local_pipe)
    q = queue.Queue()

    def run():
        try:
            cfg = ov_genai.GenerationConfig()
            cfg.max_new_tokens = max_tokens
            if temperature > 0:
                cfg.do_sample, cfg.temperature = True, temperature
            def on_token(piece):
                q.put(piece)
                return False  # не останавливать

            pipe.generate(ov_genai.ChatHistory(messages), cfg, on_token)
        except Exception as e:
            q.put(e)
        q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


def stream(messages, max_tokens=700, temperature=0.3):
    """Генератор кусочков ответа."""
    if engine_up():
        try:
            yield from _engine_stream(messages, max_tokens, temperature)
            return
        except OSError as e:
            _seen[:] = [time.time(), False]
            print(f"[llm] движок недоступен ({e}), пробую локальную модель")
    if config.get("llm_fallback") != "local":
        raise RuntimeError(f"Движок {config.get('engine_url')} не запущен, а llm_fallback = off")
    yield from _local_stream(messages, max_tokens, temperature)


def chat(messages, **kw) -> str:
    return "".join(stream(messages, **kw))


def ask(instruction: str, text: str, **kw) -> str:
    return chat([{"role": "system", "content": instruction}, {"role": "user", "content": text}], **kw)


LANG_NAMES = {"ru": "русский", "en": "английский", "de": "немецкий", "fr": "французский",
              "es": "испанский", "zh": "китайский", "ja": "японский", "uz": "узбекский",
              "tg": "таджикский", "tr": "турецкий", "ar": "арабский"}


def translate(text: str, lang: str = "ru", **kw) -> str:
    name = LANG_NAMES.get(lang, lang)
    return ask(f"Переведи текст пользователя на {name} язык. Если он уже на этом языке, верни как есть. "
               "Выведи только перевод, без пояснений.", text, **kw).strip()
