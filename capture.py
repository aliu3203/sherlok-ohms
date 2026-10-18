"""Save frames from the camera (data collection + reference photos).

  python3 capture.py --out ref/empty.png                       # one frame: the empty-board reference
  python3 capture.py --out captures/led_demo --count 15 --every 1.0   # 15 frames of a build, 1 s apart
Change lighting / bend wires / move hands between frames to get variety.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sherlok.camera import open_source, save_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qnx")
    ap.add_argument("--out", required=True, help="file.png for one frame, or a folder")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=5, help="frames to discard while auto-exposure settles")
    a = ap.parse_args()
    src = open_source(a.source)
    try:
        for _ in range(a.warmup):
            src.read()
        out = Path(a.out)
        for i in range(a.count):
            rgb = src.read()
            path = out if out.suffix else out / f"{int(time.time() * 1000)}.png"
            save_rgb(path, rgb)
            print(f"saved {path} ({rgb.shape[1]}x{rgb.shape[0]})", flush=True)
            if i + 1 < a.count:
                time.sleep(a.every)
    finally:
        if hasattr(src, "close"):
            src.close()


if __name__ == "__main__":
    main()
