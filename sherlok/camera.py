"""Frame sources. Every source returns RGB uint8 numpy arrays.

  --source qnx            Pi Camera Module 3 on QNX (needs qnx_camera_helper.py next to main.py)
  --source qnx:2          a different QNX camera unit
  --source webcam         laptop webcam via OpenCV (development)
  --source path.png       a single image file, repeated forever
  --source some/folder    every image in a folder, in name order, looping
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


class Source:
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class QnxSource(Source):
    def __init__(self, unit: int = 3):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from qnx_camera_helper import QnxCamera  # from QNX codelab mediapipe-camera-python
        self.cam = QnxCamera(unit)
        self.cam.open()

    def read(self):
        rgb, _ = self.cam.read(2.0)
        return rgb

    def close(self):
        self.cam.close()


class WebcamSource(Source):
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("webcam not available")

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


class FileSource(Source):
    def __init__(self, path: str, delay: float = 0.2):
        p = Path(path)
        self.files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
        if not self.files:
            raise RuntimeError(f"no images in {path}")
        self.i, self.delay = 0, delay

    def read(self):
        f = self.files[self.i % len(self.files)]
        self.i += 1
        time.sleep(self.delay)
        return load_rgb(f)


def load_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def open_source(spec: str) -> Source:
    if spec.startswith("qnx"):
        unit = int(spec.split(":")[1]) if ":" in spec else 3
        return QnxSource(unit)
    if spec.startswith("webcam"):
        idx = int(spec.split(":")[1]) if ":" in spec else 0
        return WebcamSource(idx)
    return FileSource(spec)
