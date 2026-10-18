# sherlok-ohms

## Hole grid (`grid/`)

Four ArUco markers (DICT_4X4_50, IDs 0-3 clockwise from top-left) at the
breadboard's corners → one homography → the pixel center of every hole.
No clicks, no ML. The markers must sit **at the height of the board's top
surface** (stickers on the board, or spacers), so markers and holes share a plane.

```python
from grid import GridRegistrar
registrar = GridRegistrar()        # board from $SHERLOK_BOARD (full | half)
grid = registrar.update(rgb)       # None until all 4 markers have been seen once
grid.image_xy('a1')                # full-res pixel center
grid.warp(rgb)                     # top-down board; holes at constant grid.xy_warped
grid.nearest(x, y).hole            # 'c12', or None if far from / between holes
grid.stale                         # True: markers hidden this frame, last good grid reused
```

### Setup
1. `python -m tools.make_markers`: print `out/markers.pdf` at 100% and check the 50 mm bar.
2. Place IDs 0-3 clockwise as the camera sees them, starting top-left. Keep the white border.
3. Measure each marker's center (middle of the black square) relative to hole a1's center
   (+x toward the last column, +y toward row j). Measure the rail offsets too, then fill in the
   `# MEASURE` values in `grid/spec.py`. Set `row_a_at_bottom=False` if, with column 1 on
   your left, row a is the *top* row.
4. `python -m tools.overlay photo.png [--board half]`: rings should sit on the holes, and px/pitch should be ≥ 20.

On the Pi (QNX): `apk add` opencv and numpy, copy `qnx_camera_helper.py` from
github.com/qnx/codelabs into the repo root, then run `python -m tools.pi_capture`.

### Laptop dev
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```
