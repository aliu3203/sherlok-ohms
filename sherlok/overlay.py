"""Draw a debug image: holes, detected parts, ends, expected pins, verdict."""
from __future__ import annotations

import cv2
import numpy as np

from .board import Board
from .check import GREEN, RED, YELLOW, Result
from .vision import Frame

COL = {GREEN: (40, 200, 60), YELLOW: (240, 190, 20), RED: (230, 40, 40)}


def draw(board: Board, f: Frame, res: Result | None, schema: dict | None, state: str) -> np.ndarray:
    img = f.warped.copy()
    for i, h in enumerate(board.holes):
        x, y = int(h.xy[0]), int(h.xy[1])
        c = (255, 140, 0) if f.p_cov[i] > 0.5 else (150, 150, 150)
        cv2.circle(img, (x, y), 2, c, -1)
    if schema:
        for p in schema["parts"]:
            for pin in p["pins"]:
                h = board.by_name.get(pin)
                if h:
                    cv2.circle(img, (int(h.xy[0]), int(h.xy[1])), 10, (60, 120, 255), 2)
    for k, part in enumerate(f.parts):
        ok = res is not None and k in res.matched.values()
        c = (40, 200, 60) if ok else (230, 40, 40) if part.conf >= 0.8 else (240, 190, 20)
        if part.contour is not None:
            cv2.drawContours(img, [part.contour], -1, c, 2)
        a, b = (board.holes[e] for e in part.ends)
        cv2.line(img, tuple(int(v) for v in a.xy), tuple(int(v) for v in b.xy), c, 1)
        for h in (a, b):
            cv2.circle(img, (int(h.xy[0]), int(h.xy[1])), 6, c, -1)
            cv2.putText(img, h.name, (int(h.xy[0]) + 6, int(h.xy[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    bar = np.zeros((40 + 18 * (len(res.issues) if res else 0), img.shape[1], 3), np.uint8)
    bar[:] = COL.get(state, (90, 90, 90))
    cv2.putText(bar, state.upper(), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    if res:
        for j, (sev, msg) in enumerate(res.issues):
            cv2.putText(bar, f"[{sev}] {msg}", (8, 56 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([bar, img])
