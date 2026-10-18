"""Breadboard geometry: calibration clicks -> homography -> every hole's position and electrical node.

Canonical (warped) image: hole pitch = PITCH px, columns along x, rows a..j along y.
Rows a-e are y=0..4; rows f-j start after the center channel (default 3 pitches from e to f).
Rails are defined by clicking their first and last hole; holes are spaced in groups of 5
with a 1-pitch gap between groups (standard boards). Set group_size 0 for uniform spacing.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PITCH = 24          # canonical px per 0.1" hole pitch
MARGIN = 2.0        # pitches of margin around everything in the canonical image
ROWS_TOP = "abcde"
ROWS_BOT = "fghij"


@dataclass
class Hole:
    name: str            # e.g. "a5" or "tp3"
    node: str            # electrical node, e.g. "A5" (a-e of col 5), "B5" (f-j), "R:tp"
    xy: tuple            # center in canonical image (px)


class Board:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.columns = int(cfg["columns"])
        self.gap = float(cfg.get("center_gap_pitches", 3))
        clicks = cfg["clicks"]  # image coordinates
        src = np.float32([clicks["a1"], clicks["a_last"], clicks["j_last"], clicks["j1"]])
        y_j = 4 + self.gap + 4
        dst = np.float32([[0, 0], [self.columns - 1, 0], [self.columns - 1, y_j], [0, y_j]]) * PITCH
        H = cv2.getPerspectiveTransform(src, dst)

        pts = {}  # name -> canonical xy (before shift)
        for ci in range(self.columns):
            col = ci + 1
            for ri, r in enumerate(ROWS_TOP):
                pts[f"{r}{col}"] = (ci * PITCH, ri * PITCH)
            for ri, r in enumerate(ROWS_BOT):
                pts[f"{r}{col}"] = (ci * PITCH, (4 + self.gap + ri) * PITCH)

        rail_pts = {}
        for rname, rc in cfg.get("rails", {}).items():
            a = cv2.perspectiveTransform(np.float32([[rc["first"]]]), H)[0, 0]
            b = cv2.perspectiveTransform(np.float32([[rc["last"]]]), H)[0, 0]
            n, gs = int(rc["count"]), int(rc.get("group_size", 5))
            pos = [(i // gs) * (gs + 1) + i % gs if gs > 0 else i for i in range(n)]
            span = max(pos[-1], 1)
            for i, p in enumerate(pos):
                t = p / span
                rail_pts[f"{rname}{i + 1}"] = tuple(a + (b - a) * t)
        pts.update(rail_pts)

        allxy = np.array(list(pts.values()), np.float32)
        lo = allxy.min(0) - MARGIN * PITCH
        hi = allxy.max(0) + MARGIN * PITCH
        T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)
        self.H = T @ H
        self.size = (int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1])))  # (w, h)

        rails_split = cfg.get("rails_split", {})  # e.g. {"tp": 25} -> tp1..25 and tp26.. are separate nodes
        self.holes: list[Hole] = []
        for name, (x, y) in pts.items():
            xy = (float(x - lo[0]), float(y - lo[1]))
            self.holes.append(Hole(name, self.node_of(name, rails_split), xy))
        self.by_name = {h.name: h for h in self.holes}
        self.xy = np.array([h.xy for h in self.holes], np.float32)

    @staticmethod
    def node_of(name: str, rails_split: dict | None = None) -> str:
        name = name.strip().lower()
        if name[0] in ROWS_TOP and name[1:].isdigit():
            return f"A{int(name[1:])}"
        if name[0] in ROWS_BOT and name[1:].isdigit():
            return f"B{int(name[1:])}"
        rail = name.rstrip(string.digits)
        idx = name[len(rail):]
        if rails_split and rail in rails_split and idx and int(idx) > int(rails_split[rail]):
            return f"R:{rail}:2"
        return f"R:{rail}"

    def warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.H, self.size, flags=cv2.INTER_LINEAR)

    @classmethod
    def load(cls, path: str | Path) -> "Board":
        return cls(json.loads(Path(path).read_text()))
