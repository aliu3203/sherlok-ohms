"""Render a camera view of a board straight from its spec, with known ground truth."""
import cv2
import numpy as np

from grid.layout import hole_positions_mm
from grid.spec import ARUCO_DICT, MARKER_IDS

IMAGE_SIZE = (2304, 1296)  # Camera Module 3 stream on the Pi


def camera_homography(spec, *, px_per_mm=10., angle_deg=0., perspective=.06, seed=0):
    """Random mm -> image homography: rotate/scale about the image center, then jitter corners."""
    rng = np.random.default_rng(seed)
    quad_mm = np.float32([spec.marker_centers_mm[i] for i in MARKER_IDS])
    center_mm = quad_mm.mean(axis=0)
    a = np.deg2rad(angle_deg)
    rot = px_per_mm * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad_img = (quad_mm - center_mm) @ rot.T + np.array(IMAGE_SIZE) / 2
    span = np.ptp(quad_img, axis=0).max()
    quad_img += rng.uniform(-perspective, perspective, quad_img.shape) * span
    return cv2.getPerspectiveTransform(quad_mm, quad_img.astype(np.float32)).astype(float)


def render(spec, H, *, noise=3., hide_markers=(), seed=0):
    """RGB image of holes and marker stickers seen through mm -> image homography H."""
    w, h = IMAGE_SIZE
    image = np.full((h, w, 3), 175, np.uint8)
    holes = np.array(list(hole_positions_mm(spec).values()))
    pts = cv2.perspectiveTransform(holes.reshape(-1, 1, 2), H).reshape(-1, 2)
    radius = 0.5 * spec.pitch_mm * np.linalg.norm(H[:2, :2] @ [1, 0]) * .4
    for x, y in pts:
        cv2.circle(image, (round(x * 16), round(y * 16)), round(radius * 16), (40, 40, 40), -1,
                   cv2.LINE_AA, shift=4)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    for marker_id in MARKER_IDS:
        if marker_id in hide_markers:
            continue
        # 4x4 code + 1-cell black border = 6 cells; add a 1-cell white quiet zone each side.
        cell = 40
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 6 * cell)
        sticker = cv2.copyMakeBorder(marker, cell, cell, cell, cell, cv2.BORDER_CONSTANT, value=255)
        m = sticker.shape[0]
        size_mm = spec.marker_size_mm * 8 / 6
        cx, cy = spec.marker_centers_mm[marker_id]
        # Sticker pixel u spans [u-0.5, u+0.5]; map its outer edges onto the sticker in mm.
        scale = size_mm / m
        A = np.array([[scale, 0, cx - size_mm / 2 + scale / 2],
                      [0, scale, cy - size_mm / 2 + scale / 2], [0, 0, 1]])
        M = H @ A
        warped = cv2.warpPerspective(sticker, M, IMAGE_SIZE, flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full_like(sticker, 255), M, IMAGE_SIZE, flags=cv2.INTER_AREA)
        alpha = (mask.astype(float) / 255)[..., None]
        image = (image * (1 - alpha) + warped[..., None] * alpha).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    return image


def truth_xy(spec, H):
    holes = np.array(list(hole_positions_mm(spec).values()))
    return cv2.perspectiveTransform(holes.reshape(-1, 1, 2), H).reshape(-1, 2)


def marker_truth_xy(spec, H):
    pts = np.array([spec.marker_centers_mm[i] for i in MARKER_IDS], dtype=float)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
