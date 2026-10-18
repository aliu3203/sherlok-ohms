"""Printable ArUco stickers (IDs 0-3, DICT_4X4_50) for the board corners.

    python -m tools.make_markers [--board half] [--size-mm 8] [--out out/]

Writes markers.pdf and markers.png at 300 DPI. Print at 100% (no "fit to page"),
check the 50 mm bar, cut along the gray lines, and keep the white border:
it is the quiet zone the detector needs. Marker size does not enter the math;
only each marker's *center* (middle of the black square) relative to hole a1
does, so measure those after sticking and enter them in grid/spec.py.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from grid import get_spec
from grid.spec import ARUCO_DICT, MARKER_IDS

DPI = 300
QUIET_CELLS = 2  # white border, in marker cells; 2 avoids the sticker edge crowding out the marker
PAGES = {'letter': (215.9, 279.4), 'a4': (210.0, 297.0)}


def mm(value):
    return round(value / 25.4 * DPI)


def sticker(marker_id, size_mm):
    cell = size_mm / 6  # 4x4 code + 1-cell black border on each side
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                           marker_id, mm(size_mm))
    q = mm(QUIET_CELLS * cell)
    out = cv2.copyMakeBorder(marker, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(out, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=170)  # cut line


def sheet(size_mm, paper, sets):
    w, h = PAGES[paper]
    page = np.full((mm(h), mm(w)), 255, np.uint8)
    margin, gap, label = mm(15), mm(6), mm(4)
    x, y = margin, margin
    text = f'DICT_4X4_50  marker {size_mm:g} mm  print at 100%  IDs clockwise from top-left: 0 1 2 3'
    cv2.putText(page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    y += mm(6)
    cv2.rectangle(page, (x, y), (x + mm(50), y + mm(2)), 0, -1)
    cv2.putText(page, '50 mm', (x + mm(52), y + mm(2)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    y += mm(10)
    for _ in range(sets):
        x = margin
        for marker_id in MARKER_IDS:
            s = sticker(marker_id, size_mm)
            if y + s.shape[0] + label > page.shape[0] - margin:
                return page
            page[y:y + s.shape[0], x:x + s.shape[1]] = s
            cv2.putText(page, f'ID {marker_id}', (x, y + s.shape[0] + label),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
            x += s.shape[1] + gap
        y += s.shape[0] + label + gap
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--board', help='full or half; sets the default marker size')
    parser.add_argument('--size-mm', type=float, help='black square edge (default: spec.marker_size_mm)')
    parser.add_argument('--paper', choices=sorted(PAGES), default='letter')
    parser.add_argument('--sets', type=int, default=3, help='copies of IDs 0-3 (spares)')
    parser.add_argument('--out', type=Path, default=Path('out'))
    args = parser.parse_args()
    size = args.size_mm or get_spec(args.board).marker_size_mm
    page = sheet(size, args.paper, args.sets)
    args.out.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # laptop-only dependency
    image = Image.fromarray(page)
    for name in ('markers.png', 'markers.pdf'):
        image.save(args.out / name, dpi=(DPI, DPI), resolution=DPI)
        print(args.out / name)


if __name__ == '__main__':
    main()
