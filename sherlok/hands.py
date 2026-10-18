"""Optional: MediaPipe pretrained hand detector -> 'board obstructed' (yellow).

On QNX: sudo apk add mediapipe
Model:  curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""
from __future__ import annotations

import numpy as np


class HandDetector:
    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.det = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=model_path),
                num_hands=2, min_hand_detection_confidence=0.4))

    def hands_present(self, rgb: np.ndarray) -> bool:
        small = rgb[::2, ::2]  # half resolution is plenty
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        return len(self.det.detect(img).hand_landmarks) > 0

    def close(self):
        self.det.close()
