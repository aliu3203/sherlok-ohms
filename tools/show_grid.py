"""Draw the computed hole grid on the warped image so you can check calibration.

  python3 tools/show_grid.py ref/empty.png [--board board.json] [--out debug/grid.png]
Every dot must sit on a hole. If dots drift in the middle rows, adjust center_gap_pitches in board.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sherlok.board import Board  # noqa: E402
from sherlok.camera import load_rgb, save_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--board", default="board.json")
    ap.add_argument("--out", default="debug/grid.png")
    a = ap.parse_args()
    b = Board.load(a.board)
    w = b.warp(load_rgb(a.image))
    for h in b.holes:
        x, y = int(h.xy[0]), int(h.xy[1])
        cv2.circle(w, (x, y), 3, (255, 0, 0), -1)
        if h.name[1:] in ("1", "5", "10", "15", "20", "25", "30") or h.name.endswith("1"):
            cv2.putText(w, h.name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    save_rgb(a.out, w)
    print(f"wrote {a.out} ({w.shape[1]}x{w.shape[0]}), {len(b.holes)} holes")


if __name__ == "__main__":
    main()
