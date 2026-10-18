"""Draw the computed hole grid on a photo to check the constants in grid/spec.py.

    python -m tools.overlay photo.png [--board half] [--out out/]

Writes <stem>_overlay.png (camera view) and <stem>_warped.png (top-down).
If dots miss the holes, re-measure marker_centers_mm. Headless, so it also runs on the Pi.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from grid import GridRegistrar, detect_markers, get_spec

TERMINAL = (60, 220, 60)
RAIL = (255, 150, 0)
MARKER = (230, 0, 230)


def _labels(grid):
    n, rail_n = grid.spec.columns, grid.spec.rail_holes_per_row
    rails = [f'{r}_{i}' for r in grid.spec.rail_rows for i in (1, rail_n)]
    return {'a1', f'a{n}', 'j1', f'j{n}', 'e1', 'f1', *rails}


def draw_holes(rgb, grid, xy):
    """RGB copy with a ring on every hole; xy is grid.xy_image or grid.xy_warped."""
    out = rgb.copy()
    pitch = np.median(np.linalg.norm(xy[grid.index['a2']] - xy[grid.index['a1']]))
    radius = max(2, round(pitch * .3))
    font = max(.35, pitch / 40)
    labels = _labels(grid)
    for name, (x, y) in zip(grid.names, xy):
        color = TERMINAL if name[0] in grid.spec.rows and name[1:].isdigit() else RAIL
        cv2.circle(out, (round(x), round(y)), radius, color, 1, cv2.LINE_AA)
        if name in labels:
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            tx = min(round(x + radius + 2), out.shape[1] - tw - 1)  # keep labels inside the image
            ty = max(round(y - radius - 2), th + 1)
            cv2.putText(out, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font, color, 1, cv2.LINE_AA)
    return out


def process(rgb, spec, stem):
    """Register one RGB frame and write the overlays; returns the HoleGrid or None."""
    registrar = GridRegistrar(spec)
    grid = registrar.update(rgb)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if grid is None:
        print(f'Registration failed: {registrar.last_error}', file=sys.stderr)
        try:
            centers = detect_markers(rgb)
        except ValueError:
            centers = {}
        out = rgb.copy()
        for marker_id, (x, y) in centers.items():
            cv2.drawMarker(out, (round(x), round(y)), MARKER, cv2.MARKER_CROSS, 40, 2)
            cv2.putText(out, str(marker_id), (round(x) + 12, round(y) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, MARKER, 2)
        _write(f'{stem}_markers.png', out)
        return None
    _write(f'{stem}_overlay.png', draw_holes(rgb, grid, grid.xy_image))
    _write(f'{stem}_warped.png', draw_holes(grid.warp(rgb), grid, grid.xy_warped))
    print(f'{spec.name} board: {len(grid.names)} holes, '
          f'{grid.px_per_pitch:.1f} px/pitch median, {grid.pitches_px.min():.1f} min '
          f'(target >= 20); warp {grid.warp_size[0]}x{grid.warp_size[1]}')
    return grid


def _write(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f'Cannot write {path}')
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', type=Path)
    parser.add_argument('--board', help='full or half (default: $SHERLOK_BOARD or full)')
    parser.add_argument('--out', type=Path, default=Path('out'))
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        sys.exit(f'Cannot read {args.image}')
    grid = process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), get_spec(args.board), args.out / args.image.stem)
    sys.exit(0 if grid else 1)


if __name__ == '__main__':
    main()
