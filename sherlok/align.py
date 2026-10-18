"""Re-align every frame to the empty-board reference using the ArUco markers (DICT_4X4_50) stuck on the
board corners. Fixes the board sliding / being bumped between builds, so calibration + empty photo stay valid.

- detects markers at normal and 2x scale (small markers)
- >=2 markers shared with the reference: similarity transform (rotation + scale + shift)
- 1 shared marker: shift only
- learns markers the reference photo missed from frames that see them together with known ones
"""
from __future__ import annotations

import cv2
import numpy as np

_AD = cv2.aruco


class Aligner:
    def __init__(self, ref_rgb: np.ndarray):
        p = _AD.DetectorParameters()
        p.adaptiveThreshWinSizeMax = 53
        p.minMarkerPerimeterRate = 0.01
        p.cornerRefinementMethod = _AD.CORNER_REFINE_SUBPIX
        self.det = _AD.ArucoDetector(_AD.getPredefinedDictionary(_AD.DICT_4X4_50), p)
        self.ref = self.detect(ref_rgb)
        self.shape = ref_rgb.shape[:2]
        self.last_info = ""
        print(f"[align] reference markers: {sorted(self.ref)}")

    def detect(self, rgb: np.ndarray) -> dict:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = {}
        c, ids, _ = self.det.detectMarkers(g)
        if ids is not None:
            found.update({int(i): cc.reshape(4, 2) for i, cc in zip(ids.ravel(), c)})
        up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        c, ids, _ = self.det.detectMarkers(up)
        if ids is not None:
            for i, cc in zip(ids.ravel(), c):
                found.setdefault(int(i), cc.reshape(4, 2) / 2.0)
        return found

    def align(self, rgb: np.ndarray) -> np.ndarray:
        m = self.detect(rgb)
        common = sorted(set(m) & set(self.ref))
        h, w = self.shape
        if not common:
            self.last_info = "no shared markers - frame NOT aligned"
            return rgb
        src = np.concatenate([m[i] for i in common]).astype(np.float32)
        dst = np.concatenate([self.ref[i] for i in common]).astype(np.float32)
        if len(common) >= 2:
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        else:
            M = None
        if M is None:
            t = (dst - src).mean(0)
            M = np.array([[1, 0, t[0]], [0, 1, t[1]]], np.float32)
        # learn markers the reference didn't see
        if len(common) >= 2:
            for i in set(m) - set(self.ref):
                pts = np.hstack([m[i], np.ones((4, 1), np.float32)]) @ M.T
                self.ref[i] = pts.astype(np.float32)
        self.last_info = f"aligned on markers {common}, shift ({M[0, 2]:.0f},{M[1, 2]:.0f}) px"
        return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
