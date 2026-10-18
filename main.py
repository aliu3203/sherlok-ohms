"""Live check: camera -> warp -> change detection (+ optional TFLite hole model, MediaPipe hands)
-> compare with schema -> GPIO lights + debug image.

  python3 main.py --source qnx --schema schemas/led_demo.json
  python3 main.py --source captures/test --schema schemas/led_demo.json --once   # offline test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sherlok.board import Board
from sherlok.camera import load_rgb, open_source, save_rgb
from sherlok.check import YELLOW, Checker, Debouncer, Result, load_schema
from sherlok.overlay import draw
from sherlok.signal import Lights
from sherlok.vision import Vision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qnx")
    ap.add_argument("--board", default="board.json")
    ap.add_argument("--empty", default="ref/empty.png", help="photo of the empty board, same mount + lighting")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--model", default=None, help="optional hole classifier .tflite")
    ap.add_argument("--hands", default=None, help="optional MediaPipe hand_landmarker.task")
    ap.add_argument("--debug", default="debug/latest.png")
    ap.add_argument("--debug-every", type=int, default=5)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds an error must persist before RED")
    ap.add_argument("--once", action="store_true", help="process one frame, print, exit")
    a = ap.parse_args()

    board = Board.load(a.board)
    tuning = json.loads(Path("thresholds.json").read_text()) if Path("thresholds.json").exists() else {}
    vision = Vision(board, load_rgb(a.empty), tuning.get("vision"), a.model)
    schema = load_schema(a.schema)
    checker = Checker(board, schema, tuning.get("high_conf", 0.8))
    deb = Debouncer(a.hold)
    lights = Lights(tuning.get("pins"))
    hands = None
    if a.hands:
        from sherlok.hands import HandDetector
        hands = HandDetector(a.hands)
    src = open_source(a.source)
    print(f"checking against '{schema.get('name', a.schema)}' ({len(schema['parts'])} parts)")

    n, last_msg = 0, None
    try:
        while True:
            t0 = time.monotonic()
            rgb = src.read()
            if hands and hands.hands_present(rgb):
                res, f = Result(YELLOW, [("info", "hand detected - can't verify")]), None
            else:
                f = vision.analyze(rgb)
                res = checker.check(f)
            state = deb.update(res.state)
            lights.show(state)
            msg = "; ".join(m for _, m in res.issues) or "all parts match"
            if msg != last_msg or a.once:
                print(f"[{state:6}] frame={res.state:6} {msg}  ({(time.monotonic() - t0) * 1000:.0f} ms)", flush=True)
                last_msg = msg
            if f is not None and (a.once or n % a.debug_every == 0):
                save_rgb(a.debug, draw(board, f, res, schema, state))
            n += 1
            if a.once:
                break
    except KeyboardInterrupt:
        pass
    finally:
        lights.close()
        if hasattr(src, "close"):
            src.close()
        if hands:
            hands.close()


if __name__ == "__main__":
    main()
