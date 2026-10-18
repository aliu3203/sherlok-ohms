"""Grab one frame on the Pi (QNX) and write it plus the grid overlay.

    python -m tools.pi_capture [--board half] [--out out/]

Needs qnx_camera_helper.py from github.com/qnx/codelabs (mediapipe-camera-python)
in the repo root or on PYTHONPATH. Copy the raw PNG back to the laptop for tests.
"""
import argparse
from pathlib import Path
import sys
import time

import cv2

from grid import get_spec
from tools.overlay import process


def capture(unit, warmup):
    from qnx_camera_helper import QnxCamera
    cam = QnxCamera(unit=unit)
    cam.open()
    try:
        rgb = None
        for _ in range(warmup + 1):  # let exposure settle
            rgb, _ = cam.read()
        if rgb is None:
            raise RuntimeError('Camera returned no frame')
        return rgb
    finally:
        getattr(cam, 'close', lambda: None)()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--board', help='full or half (default: $SHERLOK_BOARD or full)')
    parser.add_argument('--unit', type=int, default=3, help='camera unit (3 = Camera Module 3 on DISP0)')
    parser.add_argument('--warmup', type=int, default=10, help='frames to discard first')
    parser.add_argument('--out', type=Path, default=Path('out'))
    args = parser.parse_args()
    rgb = capture(args.unit, args.warmup)
    stem = args.out / time.strftime('capture_%Y%m%d_%H%M%S')
    stem.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{stem}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f'{stem}.png')
    sys.exit(0 if process(rgb, get_spec(args.board), stem) else 1)


if __name__ == '__main__':
    main()
