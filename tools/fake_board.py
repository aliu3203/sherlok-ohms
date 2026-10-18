"""Generate synthetic camera photos of a breadboard (empty + built from a schema) plus a matching
board.json, to test the pipeline without hardware.

  python3 tools/fake_board.py --schema schemas/led_demo.json --out fake/
  python3 main.py --source fake/built.png --empty fake/empty.png --board fake/board.json \
                  --schema schemas/led_demo.json --once
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sherlok.check import load_schema  # noqa: E402

P = 30  # px per pitch in the synthetic board texture


def hole_xy(name: str, cols: int):
    """Board-texture coordinates (px) of a hole. Rails: tp/tn above row a, bp/bn below row j."""
    ox, oy = 3 * P, 5 * P
    r, rest = name[0], name[1:]
    if r in "abcde" and rest.isdigit():
        return ox + (int(rest) - 1) * P, oy + "abcde".index(r) * P
    if r in "fghij" and rest.isdigit():
        return ox + (int(rest) - 1) * P, oy + (7 + "fghij".index(r)) * P
    rail = name.rstrip("0123456789")
    i = int(name[len(rail):]) - 1
    pos = (i // 5) * 6 + i % 5
    y = {"tp": oy - 3 * P, "tn": oy - 4 * P, "bp": oy + 14 * P, "bn": oy + 15 * P}[rail]
    return ox + P // 2 + pos * P, y


def render(cols: int, rail_count: int, parts: list, rng) -> np.ndarray:
    W, H = (cols + 5) * P, 22 * P
    img = np.full((H, W, 3), (232, 228, 220), np.uint8)
    names = [f"{r}{c}" for r in "abcdefghij" for c in range(1, cols + 1)]
    names += [f"{r}{i}" for r in ("tp", "tn", "bp", "bn") for i in range(1, rail_count + 1)]
    for n in names:
        x, y = hole_xy(n, cols)
        cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), (60, 60, 60), -1)
    cv2.line(img, (0, 5 * P + int(5.5 * P)), (W, 5 * P + int(5.5 * P)), (200, 195, 185), 8)
    colors = [(220, 30, 30), (30, 30, 200), (30, 160, 40), (230, 200, 20), (240, 120, 20)]
    for k, p in enumerate(parts):
        a, b = (np.array(hole_xy(x, cols)) for x in p["pins"])
        t = p.get("type", "wire")
        if t == "resistor":
            cv2.line(img, tuple(a), tuple(b), (170, 170, 170), 3)
            m = (a + b) // 2
            d = (b - a) / max(np.linalg.norm(b - a), 1)
            q1, q2 = (m - d * P * 0.9).astype(int), (m + d * P * 0.9).astype(int)
            cv2.line(img, tuple(q1), tuple(q2), (210, 180, 130), 12)
        elif t == "led":
            cv2.line(img, tuple(a), tuple(b), (170, 170, 170), 3)
            m = (a + b) // 2
            cv2.circle(img, tuple(m), int(P * 0.8), (240, 60, 60), -1)
        else:
            cv2.line(img, tuple(a), tuple(b), colors[k % len(colors)], 9)
        for e in (a, b):
            cv2.circle(img, tuple(e), 6, (150, 150, 150), -1)
    return img


def to_camera(tex: np.ndarray, M: np.ndarray, size, rng, gain: float) -> np.ndarray:
    cam = cv2.warpPerspective(tex, M, size, borderValue=(90, 90, 90)).astype(np.float32)
    cam = cam * gain + rng.normal(0, 3, cam.shape)
    return np.clip(cam, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--wrong", default=None, help="optional second schema rendered as built.png instead (to test RED)")
    ap.add_argument("--columns", type=int, default=30)
    ap.add_argument("--rail-count", type=int, default=25)
    ap.add_argument("--out", default="fake")
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    built_parts = load_schema(a.wrong or a.schema)["parts"]

    empty = render(a.columns, a.rail_count, [], rng)
    built = render(a.columns, a.rail_count, built_parts, rng)
    h, w = empty.shape[:2]
    size = (1600, 900)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[180, 120], [1450, 90], [1500, 820], [130, 790]])  # mild perspective
    M = cv2.getPerspectiveTransform(src, dst)
    cv2.imwrite(str(out / "empty.png"), cv2.cvtColor(to_camera(empty, M, size, rng, 1.0), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "built.png"), cv2.cvtColor(to_camera(built, M, size, rng, 0.92), cv2.COLOR_RGB2BGR))

    def cam_xy(name):
        p = cv2.perspectiveTransform(np.float32([[hole_xy(name, a.columns)]]), M)[0, 0]
        return [float(p[0]), float(p[1])]

    cfg = {"columns": a.columns, "center_gap_pitches": 3,
           "clicks": {"a1": cam_xy("a1"), "a_last": cam_xy(f"a{a.columns}"),
                      "j_last": cam_xy(f"j{a.columns}"), "j1": cam_xy("j1")},
           "rails": {r: {"first": cam_xy(f"{r}1"), "last": cam_xy(f"{r}{a.rail_count}"),
                         "count": a.rail_count, "group_size": 5} for r in ("tp", "tn", "bp", "bn")}}
    (out / "board.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out}/empty.png, {out}/built.png, {out}/board.json")


if __name__ == "__main__":
    main()
