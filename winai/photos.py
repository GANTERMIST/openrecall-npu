"""Фото: увеличение разрешения, удаление объектов, Cocreator (картинка по описанию/наброску).

  python -m winai photo upscale in.jpg out.png
  python -m winai photo erase in.jpg mask.png out.png      # белое на маске — что убрать
  python -m winai photo create "кот в космосе, акварель" out.png [--sketch draft.png]

Если модель не скачалась, апскейл и удаление работают классическими методами OpenCV
(хуже, но без сети), чтобы функция не пропадала совсем.
"""
import numpy as np
from PIL import Image, ImageFilter

from . import config, power, runtime

SR_URLS = [
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{n}/FP16/{n}.{e}",
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/2/{n}/FP16/{n}.{e}",
]
LAMA = 512


# ---------- общее ----------

def tiled(arr, tile_h, tile_w, scale, fn, margin=8):
    """Прогоняет fn по плиткам tile_h x tile_w (с перекрытием margin) и склеивает результат x scale.
    arr: HxWx3 uint8. fn(плитка tile_h x tile_w) -> плитка (tile_h*scale) x (tile_w*scale)."""
    h, w = arr.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), np.uint8)
    step_h, step_w = tile_h - 2 * margin, tile_w - 2 * margin
    padded = np.pad(arr, ((margin, tile_h), (margin, tile_w), (0, 0)), mode="edge")
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            res = fn(padded[y:y + tile_h, x:x + tile_w])
            hh, ww = min(step_h, h - y), min(step_w, w - x)
            m = margin * scale
            out[y * scale:(y + hh) * scale, x * scale:(x + ww) * scale] = res[m:m + hh * scale, m:m + ww * scale]
    return out


# ---------- супер-разрешение ----------

def _sr_model():
    n = config.get("sr_model")
    last = None
    for tpl in SR_URLS:
        try:
            xml = runtime.url_file(tpl.format(n=n, e="xml"), f"{n}.xml")
            runtime.url_file(tpl.format(n=n, e="bin"), f"{n}.bin")
            model = runtime.core().read_model(xml)
            compiled, _ = runtime.compile_model(model, "vision", "upscale")
            return compiled
        except Exception as e:
            last = e
    raise RuntimeError(f"модель апскейла не загрузилась: {last}")


def upscale(img: Image.Image) -> Image.Image:
    """x4 (OMZ single-image-super-resolution-1032) на NPU/GPU; без модели — Lanczos x4 + резкость."""
    img = img.convert("RGB")
    try:
        net = runtime.POOL.get("sr", _sr_model)
    except Exception as e:
        print(f"[upscale] {e}; использую классический апскейл")
        big = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
        return big.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    import cv2
    lr_in, bic_in = net.inputs[0], net.inputs[1]
    _, _, th, tw = lr_in.shape
    _, _, bh, bw = bic_in.shape
    scale = bh // th

    def run(tile_rgb):
        bgr = tile_rgb[:, :, ::-1]
        bic = cv2.resize(bgr, (bw, bh), interpolation=cv2.INTER_CUBIC)
        res = net({lr_in: bgr.transpose(2, 0, 1)[None].astype(np.float32),
                   bic_in: bic.transpose(2, 0, 1)[None].astype(np.float32)})[net.outputs[0]][0]
        if res.max() <= 2.0:  # модель отдаёт 0..1
            res = res * 255
        return np.clip(res, 0, 255).astype(np.uint8).transpose(1, 2, 0)[:, :, ::-1]

    arr = np.asarray(img)
    return Image.fromarray(tiled(arr, int(th), int(tw), int(scale), run))


# ---------- удаление объектов ----------

def _lama():
    path = runtime.url_file(config.get("inpaint_url"), "lama.onnx")
    model = runtime.core().read_model(path)
    model.reshape({model.inputs[0].get_any_name(): [1, 3, LAMA, LAMA],
                   model.inputs[1].get_any_name(): [1, 1, LAMA, LAMA]})
    compiled, _ = runtime.compile_model(model, "vision", "erase")
    return compiled


def crop_box(mask: np.ndarray, pad=1.0, min_side=LAMA):
    """Квадрат вокруг маски с запасом (контекст для заливки), в пределах картинки."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(max(y1 - y0, x1 - x0) * (1 + pad))
    side = min(max(side, min_side), max(h, w))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    top = int(np.clip(cy - side // 2, 0, max(h - side, 0)))
    left = int(np.clip(cx - side // 2, 0, max(w - side, 0)))
    return top, left, min(top + side, h), min(left + side, w)


def erase(img: Image.Image, mask: Image.Image) -> Image.Image:
    import cv2
    img = img.convert("RGB")
    m = np.asarray(mask.convert("L").resize(img.size)) > 127
    if not m.any():
        return img
    m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    arr = np.asarray(img).copy()
    try:
        net = runtime.POOL.get("lama", _lama)
    except Exception as e:
        print(f"[erase] {e}; использую классическую заливку OpenCV")
        return Image.fromarray(cv2.inpaint(arr, m.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA))
    t, l, b, r = crop_box(m)
    crop, cm = arr[t:b, l:r], m[t:b, l:r]
    x = cv2.resize(crop, (LAMA, LAMA), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    xm = (cv2.resize(cm.astype(np.uint8), (LAMA, LAMA), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    out = net({net.inputs[0]: x.transpose(2, 0, 1)[None], net.inputs[1]: xm[None, None]})[net.outputs[0]][0]
    if out.max() <= 2.0:
        out = out * 255
    out = np.clip(out, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    out = cv2.resize(out, (r - l, b - t), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 3)[..., None]
    arr[t:b, l:r] = (out * soft + crop * (1 - soft)).astype(np.uint8)
    return Image.fromarray(arr)


# ---------- Cocreator ----------

def _t2i():
    import openvino_genai as ov_genai
    path = runtime.hf_model(config.get("image_model"))
    pipe, _ = runtime.first_working("image", lambda d: ov_genai.Text2ImagePipeline(str(path), d), "cocreator")
    return pipe


def _i2i():
    import openvino_genai as ov_genai
    path = runtime.hf_model(config.get("image_model"))
    pipe, _ = runtime.first_working("image", lambda d: ov_genai.Image2ImagePipeline(str(path), d), "cocreator")
    return pipe


def create(prompt: str, sketch: Image.Image | None = None, strength=0.75, seed=None) -> Image.Image:
    import openvino as ov
    lcm = "lcm" in config.get("image_model").lower()
    eco = power.profile() == "eco"
    steps = (4 if eco else 6) if lcm else (12 if eco else 25)
    kw = {"num_inference_steps": steps}
    if seed is not None:
        kw["rng_seed"] = int(seed)
    if sketch is not None:
        try:
            pipe = runtime.POOL.get("i2i", _i2i)
            arr = np.asarray(sketch.convert("RGB").resize((512, 512)))[None]
            res = pipe.generate(prompt, ov.Tensor(arr), strength=strength, **kw)
            return Image.fromarray(res.data[0])
        except Exception as e:
            print(f"[cocreator] набросок не поддержан ({e}), рисую только по описанию")
    pipe = runtime.POOL.get("t2i", _t2i)
    res = pipe.generate(prompt, width=512, height=512, **kw)
    return Image.fromarray(res.data[0])


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="winai photo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upscale")
    u.add_argument("src")
    u.add_argument("dst")
    e = sub.add_parser("erase")
    e.add_argument("src")
    e.add_argument("mask")
    e.add_argument("dst")
    c = sub.add_parser("create")
    c.add_argument("prompt")
    c.add_argument("dst")
    c.add_argument("--sketch")
    c.add_argument("--strength", type=float, default=0.75)
    a = ap.parse_args(argv)
    if a.cmd == "upscale":
        upscale(Image.open(a.src)).save(a.dst)
    elif a.cmd == "erase":
        erase(Image.open(a.src), Image.open(a.mask)).save(a.dst)
    else:
        create(a.prompt, Image.open(a.sketch) if a.sketch else None, a.strength).save(a.dst)
    print("сохранено:", a.dst)
