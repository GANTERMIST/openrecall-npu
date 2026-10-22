"""OCR: сначала встроенный Windows OCR (понимает русский, если стоит языковой пакет),
запасной вариант — RapidOCR на OpenVINO (CPU)."""
import asyncio
import io

_win = None
_rapid = None


async def _win_ocr(img):
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

    global _win
    if _win is None:
        _win = OcrEngine.try_create_from_user_profile_languages()
    buf = io.BytesIO()
    img.save(buf, "PNG")
    stream = InMemoryRandomAccessStream()
    w = DataWriter(stream)
    w.write_bytes(buf.getvalue())
    await w.store_async()
    stream.seek(0)
    dec = await BitmapDecoder.create_async(stream)
    bmp = await dec.get_software_bitmap_async()
    res = await _win.recognize_async(bmp)
    return res.text


def ocr(img) -> str:
    try:
        return asyncio.run(_win_ocr(img)).strip()
    except Exception as e:
        global _rapid
        if _rapid is None:
            print(f"[ocr] Windows OCR недоступен ({type(e).__name__}), использую RapidOCR")
            from rapidocr_openvino import RapidOCR
            _rapid = RapidOCR()
        import numpy as np
        res, _ = _rapid(np.array(img.convert("RGB")))
        return "\n".join(r[1] for r in (res or []))
