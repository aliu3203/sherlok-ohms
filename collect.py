"""Guided data collection on the Pi. One command, follow the prompts.

  python3 collect.py schemas/*.txt

What it does:
  1. If ref/empty.png doesn't exist: asks you to clear the board, saves it + empty frames.
  2. For each schema: shows the parts, waits for you to build it and press Enter,
     then takes frames 1 s apart. Nudge wires / move the lamp a little between frames.
  Frames go to captures/<schema name>/ (the folder name IS the label - don't rename).
Re-running adds more frames; type 's' to skip a schema, 'q' to quit.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sherlok.camera import open_source, save_rgb
from sherlok.check import load_schema


def grab(src, folder: Path, n: int, every: float, tip: str):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rgb = src.read()
        save_rgb(folder / f"{int(time.time() * 1000)}.png", rgb)
        print(f"  [{i + 1}/{n}] saved   {tip if i % 3 == 0 else ''}", flush=True)
        if i + 1 < n:
            time.sleep(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schemas", nargs="*", help="schema files (.txt or .json); none = just capture the empty board")
    ap.add_argument("--source", default="qnx")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--out", default="captures")
    ap.add_argument("--empty", default="ref/empty.png")
    a = ap.parse_args()

    src = open_source(a.source)
    try:
        for _ in range(5):  # let auto-exposure settle
            src.read()
        out = Path(a.out)
        if not Path(a.empty).exists():
            input("\n>> Clear the board completely (no parts, no hands), then press Enter ")
            for _ in range(3):
                src.read()
            save_rgb(a.empty, src.read())
            print(f"  saved {a.empty} (empty-board reference)")
            grab(src, out / "empty", 5, a.every, "")
            print("  Now calibrate: copy ref/empty.png to a laptop and run tools/calibrate.py\n")

        for sp in a.schemas:
            s = load_schema(sp)
            name = Path(sp).stem
            n_have = len(list((out / name).glob("*.png"))) if (out / name).exists() else 0
            print(f"\n=== {name}  ({n_have} frames so far)")
            for p in s["parts"]:
                print(f"    {p['id']:4} {p.get('type', ''):9} {' -> '.join(p['pins'])}")
            ans = input(">> Build EXACTLY this, hands away, then Enter (s = skip, q = quit) ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            grab(src, out / name, a.frames, a.every, "<- nudge a wire / shift the light, keep it plugged in")
        print("\nDone. Copy to the laptop for training:  scp -r qnxuser@<pi-ip>:~/sherlok-ohms/captures .")
    finally:
        if hasattr(src, "close"):
            src.close()


if __name__ == "__main__":
    main()
