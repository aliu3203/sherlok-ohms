"""Click the corner holes on a photo from the Pi camera -> board.json. Run on a laptop (needs a GUI).

  pip install opencv-python numpy
  python3 tools/calibrate.py ref/empty.png --columns 30 --rails tp,tn,bp,bn --rail-count 25

Click the CENTER of each hole asked for (zoom: the window shows the image at full size if it fits).
Keys: u = undo last click, q = abort. Then copy board.json to the Pi.
If the camera or board moves, redo this.
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--columns", type=int, required=True, help="63 for full-size, 30 for half-size boards")
    ap.add_argument("--rails", default="", help="comma list of rail names you use, e.g. tp,tn (top +/-), bp,bn")
    ap.add_argument("--rail-count", type=int, default=25, help="holes per rail (50 full-size, 25 half-size)")
    ap.add_argument("--out", default="board.json")
    ap.add_argument("--scale", type=float, default=0.0, help="display scale (0 = auto fit 1400 px)")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    s = a.scale or min(1.0, 1400 / max(img.shape[:2]))
    disp = cv2.resize(img, None, fx=s, fy=s)

    asks = [("a1", "hole a1"), ("a_last", f"hole a{a.columns}"),
            ("j_last", f"hole j{a.columns}"), ("j1", "hole j1")]
    for r in [x for x in a.rails.split(",") if x]:
        asks += [(f"{r}.first", f"FIRST hole of rail '{r}' (hole {r}1, nearest column 1)"),
                 (f"{r}.last", f"LAST hole of rail '{r}' (hole {r}{a.rail_count})")]
    pts: list = []

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(asks):
            pts.append((x / s, y / s))
            print(f"  {asks[len(pts) - 1][0]} = ({x / s:.1f}, {y / s:.1f})")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while len(pts) < len(asks):
        view = disp.copy()
        for (x, y) in pts:
            cv2.circle(view, (int(x * s), int(y * s)), 5, (0, 0, 255), -1)
        cv2.putText(view, f"click: {asks[len(pts)][1]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        if k == ord("q"):
            sys.exit("aborted")
    cv2.destroyAllWindows()

    clicks = {k: list(p) for (k, _), p in zip(asks, pts)}
    cfg = {"columns": a.columns, "center_gap_pitches": 3,
           "clicks": {k: clicks[k] for k in ("a1", "a_last", "j_last", "j1")}, "rails": {}}
    for r in [x for x in a.rails.split(",") if x]:
        cfg["rails"][r] = {"first": clicks[f"{r}.first"], "last": clicks[f"{r}.last"],
                           "count": a.rail_count, "group_size": 5}
    with open(a.out, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"wrote {a.out}. Check it: python3 tools/show_grid.py {a.image}")


if __name__ == "__main__":
    main()
