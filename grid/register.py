"""ArUco markers -> homography -> pixel center of every hole.

Images are RGB (QnxCamera's output) or grayscale. Only numpy and cv2 are
used, with no GUI, so this runs unchanged on the Pi under QNX.
"""
from dataclasses import dataclass
import copy
import functools

import cv2
import numpy as np

from .layout import hole_positions_mm
from .spec import ARUCO_DICT, MARKER_IDS, PX_PER_MM, WARP_MARGIN_MM, get_spec


@dataclass(frozen=True)
class HoleMatch:
    hole: str | None      # None when the point is too far from a hole or ambiguous
    nearest_hole: str
    distance_px: float
    pitch_px: float
    reason: str | None


@functools.cache
def _detector(grouped=True):
    params = cv2.aruco.DetectorParameters()
    # Line fit on the marker edges: ~10x tighter centers than SUBPIX under perspective.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    if not grouped:
        params.minMarkerDistanceRate = .01
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)


def _detect(gray, grouped):
    corners, ids, _ = _detector(grouped).detectMarkers(gray)
    found = {}
    for quad, marker_id in zip(corners, [] if ids is None else ids.ravel()):
        if int(marker_id) in MARKER_IDS:
            found.setdefault(int(marker_id), []).append(quad.reshape(4, 2))
    return found


def _quad_center(corners):
    # The projected center of a square is where its image diagonals cross;
    # the mean of the four corners drifts under perspective.
    p0, p1, p2, p3 = np.asarray(corners, dtype=float).reshape(4, 2)
    d1, d2 = p2 - p0, p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return (p0 + p1 + p2 + p3) / 4
    t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
    return p0 + t * d1


def detect_markers(image):
    """{marker_id: (x, y) center} for registration IDs found in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    found = _detect(gray, grouped=True)
    if not set(MARKER_IDS) <= set(found):
        # A sticker's white edge forms a concentric square that OpenCV's grouping
        # can keep instead of the marker. Retry ungrouped; keep the largest quad.
        for marker_id, quads in _detect(gray, grouped=False).items():
            found.setdefault(marker_id, quads)
    centers = {}
    for marker_id, quads in found.items():
        quads.sort(key=lambda q: cv2.contourArea(q), reverse=True)
        center = _quad_center(quads[0])
        side = np.sqrt(cv2.contourArea(quads[0]))
        if any(np.linalg.norm(_quad_center(q) - center) > side for q in quads[1:]):
            raise ValueError(f'Duplicate registration marker ID {marker_id}')
        centers[marker_id] = center
    return centers


def warp_transform(spec):
    """(H_mm_to_warp, (width, height)) for the fixed top-down board image."""
    xy = np.array(list(hole_positions_mm(spec).values()))
    x0, y0 = xy.min(axis=0) - WARP_MARGIN_MM
    x1, y1 = xy.max(axis=0) + WARP_MARGIN_MM
    s = PX_PER_MM
    H = np.array([[s, 0, -s * x0], [0, s, -s * y0], [0, 0, 1]], dtype=float)
    return H, (int(np.ceil((x1 - x0) * s)) + 1, int(np.ceil((y1 - y0) * s)) + 1)


def _project(H, xy):
    return cv2.perspectiveTransform(np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


class HoleGrid:
    """Hole centers for one registration; positions in both the camera image and the warp."""

    def __init__(self, spec, H_mm_to_img, *, stale=False):
        self.spec = spec
        self.H_mm_to_img = H_mm_to_img
        self.stale = stale
        holes = hole_positions_mm(spec)
        self.names = tuple(holes)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.xy_mm = np.array(list(holes.values()), dtype=float)
        self.xy_image = _project(H_mm_to_img, self.xy_mm)
        self.H_mm_to_warp, self.warp_size = warp_transform(spec)
        self.xy_warped = _project(self.H_mm_to_warp, self.xy_mm)
        d = np.linalg.norm(self.xy_image[:, None] - self.xy_image[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.pitches_px = d.min(axis=1)  # local hole spacing in the camera image

    @property
    def px_per_pitch(self):
        return float(np.median(self.pitches_px))

    def as_stale(self):
        grid = copy.copy(self)  # arrays are shared, never mutated
        grid.stale = True
        return grid

    def image_xy(self, hole):
        return tuple(self.xy_image[self.index[hole]])

    def warp(self, image):
        """Top-down board image where hole centers sit at the constant xy_warped."""
        H = self.H_mm_to_warp @ np.linalg.inv(self.H_mm_to_img)
        return cv2.warpPerspective(image, H, self.warp_size, flags=cv2.INTER_LINEAR)

    def nearest(self, x, y, *, max_fraction=.4, margin_fraction=.25):
        """Assign an image point to a hole, refusing when it is far or between two holes."""
        distances = np.linalg.norm(self.xy_image - [x, y], axis=1)
        first, second = np.argsort(distances)[:2]
        pitch = float(self.pitches_px[first])
        reason = ('too_far_from_hole' if distances[first] > max_fraction * pitch else
                  'ambiguous_neighbor' if distances[second] - distances[first] < margin_fraction * pitch
                  else None)
        return HoleMatch(None if reason else self.names[first], self.names[first],
                         float(distances[first]), pitch, reason)


def grid_from_markers(spec, centers):
    """Build a HoleGrid from {marker_id: image center}; raises ValueError on bad geometry."""
    if not set(MARKER_IDS) <= set(centers):
        raise ValueError(f'Missing markers {sorted(set(MARKER_IDS) - set(centers))}')
    src = np.float32([spec.marker_centers_mm[i] for i in MARKER_IDS])
    dst = np.float32([centers[i] for i in MARKER_IDS])
    for quad in (src, dst):
        if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < 1e-3:
            raise ValueError('Markers must form a convex, nondegenerate quadrilateral')
    # Any camera rotation preserves winding order; a flip means IDs are misplaced.
    if np.sign(cv2.contourArea(src, oriented=True)) != np.sign(cv2.contourArea(dst, oriented=True)):
        raise ValueError('Marker order is mirrored; check that IDs 0-3 run clockwise from top-left')
    H = cv2.getPerspectiveTransform(src, dst).astype(float)
    if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('Degenerate homography')
    return HoleGrid(spec, H)


class GridRegistrar:
    """Per-frame registration that falls back to the last good grid when markers are hidden."""

    def __init__(self, spec=None):
        self.spec = spec or get_spec()
        self.last = None
        self.last_error = None

    def update(self, image):
        """HoleGrid for this frame, the last good one with stale=True, or None."""
        try:
            grid = grid_from_markers(self.spec, detect_markers(image))
        except ValueError as exc:
            self.last_error = str(exc)
            return self.last.as_stale() if self.last else None
        self.last, self.last_error = grid, None
        return grid
