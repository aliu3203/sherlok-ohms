"""Auto-label hole crops from captured builds. No hand labeling.

Folder layout (one folder per schema you built, named after the schema file):
  captures/led_demo/*.png      <- built exactly as schemas/led_demo.json
  captures/empty/*.png         <- empty board (optional, more 'empty' examples)

Labels per hole:  plugged = a schema pin;  covered = changed vs empty board but not a pin;
                  empty = unchanged.  Ambiguous holes are skipped.

  python3 train/make_dataset.py --captures captures --schemas schemas --out train/holes.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sherlok.board import Board  # noqa: E402
from sherlok.camera import IMG_EXT, load_rgb  # noqa: E402
from sherlok.check import find_schema, load_schema  # noqa: E402
from sherlok.vision import CLASSES, Vision, crop_holes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="captures")
    ap.add_argument("--schemas", default="schemas")
    ap.add_argument("--board", default="board.json")
    ap.add_argument("--empty", default="ref/empty.png")
    ap.add_argument("--out", default="train/holes.npz")
    ap.add_argument("--empty-per-frame", type=int, default=40, help="random empty holes kept per frame")
    a = ap.parse_args()

    board = Board.load(a.board)
    vision = Vision(board, load_rgb(a.empty))
    rng = np.random.default_rng(0)
    X, Y, G = [], [], []
    if vision.aligner is not None:  # warm-up: learn all marker positions before aligning for real
        for f in sorted(Path(a.captures).glob("*/*.png")):
            vision.aligner.align(load_rgb(f))
        print(f"[align] markers known: {sorted(vision.aligner.ref)}")
    for d in sorted(p for p in Path(a.captures).iterdir() if p.is_dir()):
        sch_path = find_schema(a.schemas, d.name)
        pins = set()
        if sch_path:
            for p in load_schema(sch_path)["parts"]:
                pins.update(p["pins"])
        elif d.name != "empty":
            print(f"skip {d} (no schemas/{d.name}.txt or .json)")
            continue
        pin_idx = {i for i, h in enumerate(board.holes) if h.name in pins}
        frames = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        for f in frames:
            fr = vision.analyze(load_rgb(f))
            if fr.obstructed:
                continue
            crops = crop_holes(fr.warped, board.xy)
            lab = np.full(len(board.holes), -1)
            for i in range(len(board.holes)):
                if i in pin_idx:
                    # label from the schema = the TRUE hole, whatever it looks like from this camera angle
                    lab[i] = CLASSES.index("plugged")
                elif fr.p_cov[i] > 0.7:
                    lab[i] = CLASSES.index("covered")
                elif fr.p_cov[i] < 0.2:
                    lab[i] = CLASSES.index("empty")
            empty = np.where(lab == 0)[0]
            keep = np.concatenate([np.where(lab > 0)[0],
                                   rng.choice(empty, min(len(empty), a.empty_per_frame), replace=False)])
            X.append(crops[keep]); Y.append(lab[keep]); G += [d.name] * len(keep)
        print(f"{d.name}: {len(frames)} frames")
    X, Y = np.concatenate(X), np.concatenate(Y)
    np.savez_compressed(a.out, x=X, y=Y, group=np.array(G))
    print(f"wrote {a.out}: {len(Y)} crops; per class", {c: int((Y == i).sum()) for i, c in enumerate(CLASSES)})


if __name__ == "__main__":
    main()
