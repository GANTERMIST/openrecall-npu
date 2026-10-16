# openrecall-npu

Локальный аналог Windows Recall: периодические скриншоты, OCR, семантический поиск.
Эмбеддинги (multilingual-e5-small) считаются на Intel NPU через OpenVINO, с откатом на GPU/CPU.
Все данные хранятся локально в `data/`.

## Установка
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Модель экспортируется в OpenVINO IR автоматически при первом запуске.

## Использование
```
python recall.py capture --interval 10   # запись
python recall.py search "запрос"         # поиск из консоли
python recall.py serve                   # веб-интерфейс http://127.0.0.1:8765
python recall.py purge --days 30         # удалить старые записи
```
OCR: встроенный Windows OCR (русский при наличии языкового пакета), запасной — RapidOCR.
Окна, в заголовке которых есть слова из `BLOCKLIST` (password, incognito, банк...), не записываются.

**Внимание:** скриншоты могут содержать пароли и личные данные — храните `data/` в безопасности.
