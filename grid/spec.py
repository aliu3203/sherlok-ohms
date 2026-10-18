"""Board and fiducial constants: the single source of truth for the hole grid.

Board frame: origin at the center of hole a1, +x toward the last column,
+y toward row j, units in mm. Every hole and every marker center is
expressed in this frame, so where the board sits on a sheet never matters.

The markers must sit at the same height as the breadboard's top surface
(stickers on the board itself, or spacers). Then markers and holes are
coplanar and one homography maps all of them exactly.

Values marked MEASURE vary by manufacturer or by how the stickers were
placed. Measure them on the real board and check with tools/overlay.py.
"""
from dataclasses import dataclass
import os

import cv2

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)  # clockwise: top-left, top-right, bottom-right, bottom-left

PX_PER_MM = 10.0       # top-down warp scale; 10 px/mm = 25.4 px per hole pitch
WARP_MARGIN_MM = 5.0   # extra border around the outermost holes in the warp


@dataclass(frozen=True)
class BoardSpec:
    name: str
    columns: int
    rail_rows: dict          # rail name -> y offset in pitches from row a (negative = above a)
    rail_first_x_pitches: float  # x of each rail's first hole, in pitches from column 1
    rail_holes_per_row: int
    marker_centers_mm: dict  # marker id -> (x, y) in the board frame
    marker_size_mm: float    # black square edge; only affects detectability, not the math
    rows: str = 'abcdefghij'
    pitch_mm: float = 2.54
    trench_pitches: int = 3  # e -> f center spacing
    rail_group: int = 5      # rail holes come in groups of 5 ...
    rail_gap_pitches: int = 1  # ... separated by one missing hole
    rail_split: bool = False   # True if each rail is cut electrically in the middle

    @property
    def terminal_width_mm(self):
        return (self.columns - 1) * self.pitch_mm

    @property
    def terminal_height_mm(self):
        return (len(self.rows) - 1 + self.trench_pitches - 1) * self.pitch_mm


# 830-point board: 63 columns, 4 rails of 50 holes (10 groups of 5).
FULL_830 = BoardSpec(
    name='full',
    columns=63,
    rail_rows={'top_minus': -3, 'top_plus': -2, 'bottom_plus': 13, 'bottom_minus': 14},  # MEASURE
    rail_first_x_pitches=2,  # MEASURE (rail span 58 pitches, centered on 62)
    rail_holes_per_row=50,
    marker_centers_mm={0: (0.0, -7.0), 1: (157.48, -7.0),  # MEASURE
                       2: (157.48, 34.94), 3: (0.0, 34.94)},
    marker_size_mm=8.0,
)

# 400-point board: 30 columns, 4 rails of 25 holes (5 groups of 5).
HALF_400 = BoardSpec(
    name='half',
    columns=30,
    rail_rows={'top_minus': -3, 'top_plus': -2, 'bottom_plus': 13, 'bottom_minus': 14},  # MEASURE
    rail_first_x_pitches=0.5,  # MEASURE (rail span 28 pitches, centered on 29)
    rail_holes_per_row=25,
    marker_centers_mm={0: (-1.0, -7.0), 1: (74.66, -7.0),  # MEASURE
                       2: (74.66, 34.94), 3: (-1.0, 34.94)},
    marker_size_mm=8.0,
)

BOARDS = {spec.name: spec for spec in (FULL_830, HALF_400)}
ACTIVE_BOARD = os.environ.get('SHERLOK_BOARD', 'full')


def get_spec(name=None):
    name = name or ACTIVE_BOARD
    if name not in BOARDS:
        raise ValueError(f'Unknown board {name!r}; expected one of {sorted(BOARDS)}')
    return BOARDS[name]
