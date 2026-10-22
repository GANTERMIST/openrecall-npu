"""Studio Effects: размытие или замена фона и автокадрирование для веб-камеры.

  python -m winai studio                       # размытие + кадрирование → виртуальная камера
  python -m winai studio --no-frame --bg room.jpg
  python -m winai studio --preview --no-cam    # только окно предпросмотра

Результат отдаётся как виртуальная камера (pyvirtualcam). На Windows для этого нужна
установленная OBS Studio (её драйвер «OBS Virtual Camera»), саму OBS запускать не надо.
В Zoom/Telegram/Discord/Teams выберите камеру «OBS Virtual Camera».
Сегментация человека — MediaPipe (лёгкая, на CPU, ~2 мс на кадр в 256x144).
"""
import time

import numpy as np

from . import power


class Framer:
    """Автокадрирование: плавно ведёт кадр за лицом, чтобы лицо было ~1/3 высоты кадра."""

    def __init__(self, w, h, face_frac=0.33, smooth=0.12, max_zoom=2.0):
        self.w, self.h = w, h
        self.face_frac, self.smooth, self.max_zoom = face_frac, smooth, max_zoom
        self.cx, self.cy, self.zoom = w / 2, h / 2, 1.0

    def update(self, face):
        """face: (x, y, w, h) в пикселях или None."""
        if face is not None:
            fx, fy, fw, fh = face
            tx, ty = fx + fw / 2, fy + fh / 2 + fh * 0.4  # чуть ниже лица — видно плечи
            tz = float(np.clip(self.face_frac * self.h / max(fh, 1), 1.0, self.max_zoom))
        else:
            tx, ty, tz = self.w / 2, self.h / 2, 1.0
        a = self.smooth
        self.cx += (tx - self.cx) * a
        self.cy += (ty - self.cy) * a
        self.zoom += (tz - self.zoom) * a
        return self.box()

    def box(self):
        cw, ch = self.w / self.zoom, self.h / self.zoom
        x0 = float(np.clip(self.cx - cw / 2, 0, self.w - cw))
        y0 = float(np.clip(self.cy - ch / 2, 0, self.h - ch))
        return int(x0), int(y0), int(cw), int(ch)


def compose(frame, mask, background=None, blur=21):
    """mask: 0..1 (1 — человек), размер кадра. background: картинка того же размера или None (размыть)."""
    import cv2
    if background is None:
        small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))
        k = blur | 1
        background = cv2.resize(cv2.GaussianBlur(small, (k, k), 0), (frame.shape[1], frame.shape[0]))
    m = mask[..., None]
    return (frame * m + background * (1 - m)).astype(np.uint8)


class Effects:
    def __init__(self, w, h, blur=True, frame=True, bg_path=None):
        import cv2
        import mediapipe as mp
        self.w, self.h = w, h
        self.blur, self.frame = blur, frame
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.faces = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.framer = Framer(w, h)
        self.bg = cv2.resize(cv2.imread(bg_path), (w, h)) if bg_path else None
        self.mask = None
        self.n = 0
        self.face = None

    def __call__(self, bgr):
        import cv2
        self.n += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = bgr
        if self.blur or self.bg is not None:
            m = self.seg.process(rgb).segmentation_mask
            m = cv2.resize(m, (self.w, self.h))
            # сглаживание во времени убирает мерцание краёв
            self.mask = m if self.mask is None else 0.6 * m + 0.4 * self.mask
            soft = np.clip((self.mask - 0.3) / 0.4, 0, 1)
            out = compose(bgr, soft, self.bg)
        if self.frame:
            if self.n % 5 == 1:  # лицо ищем не в каждом кадре — экономия
                r = self.faces.process(rgb)
                self.face = None
                if r.detections:
                    b = r.detections[0].location_data.relative_bounding_box
                    self.face = (b.xmin * self.w, b.ymin * self.h, b.width * self.w, b.height * self.h)
            x, y, cw, ch = self.framer.update(self.face)
            out = cv2.resize(out[y:y + ch, x:x + cw], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return out


def run(camera=0, width=1280, height=720, blur=True, frame=True, bg=None, cam_out=True, preview=False):
    import cv2
    fps = 15 if power.profile() == "eco" else 30
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("камера не открылась (занята другим приложением?)")
    h, w = frame0.shape[:2]
    fx = Effects(w, h, blur, frame, bg)
    vcam = None
    if cam_out:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Виртуальная камера: {vcam.device} {w}x{h}@{fps}. Ctrl+C — стоп.")
    try:
        while True:
            t0 = time.time()
            ok, f = cap.read()
            if not ok:
                break
            out = fx(f)
            if vcam:
                vcam.send(out)
            if preview:
                cv2.imshow("WinAI Studio Effects (q — выход)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if vcam:
                vcam.sleep_until_next_frame()
            else:
                time.sleep(max(0, 1 / fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if vcam:
            vcam.close()
        if preview:
            cv2.destroyAllWindows()


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="winai studio")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--no-frame", action="store_true")
    ap.add_argument("--bg", help="картинка для замены фона")
    ap.add_argument("--no-cam", action="store_true", help="не создавать виртуальную камеру")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args(argv)
    run(a.camera, blur=not a.no_blur, frame=not a.no_frame, bg=a.bg,
        cam_out=not a.no_cam, preview=a.preview or a.no_cam)
