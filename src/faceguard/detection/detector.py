"""Face detection adapter.

Uses OpenCV's Haar cascade when OpenCV is installed; otherwise falls back to a
centre crop so the rest of the pipeline still runs on a pre-cropped face. Swap
in RetinaFace/MTCNN here without touching the anti-fraud core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y:self.y + self.h, self.x:self.x + self.w]


class FaceDetector:
    def __init__(self):
        self._cascade = None
        try:
            import cv2

            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(path)
            self._cv2 = cv2
        except Exception:
            self._cascade = None

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        img = np.asarray(image)
        if self._cascade is not None and not self._cascade.empty():
            gray = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            faces = self._cascade.detectMultiScale(gray.astype("uint8"), 1.1, 5)
            return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        # Fallback: assume the whole frame is a pre-cropped face.
        h, w = img.shape[:2]
        return [FaceBox(0, 0, w, h)]

    def detect_largest(self, image: np.ndarray) -> FaceBox:
        boxes = self.detect(image)
        return max(boxes, key=lambda b: b.w * b.h)
