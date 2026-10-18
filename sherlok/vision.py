"""What changed on the board vs the empty reference, and which holes each part plugs into.

Pipeline on the warped (canonical) board image:
  1. per-pixel color difference vs the empty-board reference (Lab, lighting-normalized)
  2. per-hole "covered" probability (mean difference around the hole center)
  3. connected blobs of changed pixels = parts / wires
  4. for each blob, its two ends = the covered holes farthest apart along the blob's long axis
     (or, if a trained hole classifier is loaded, the holes it calls "plugged")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import PITCH, Board

CLASSES = ("empty", "covered", "plugged")  # hole classifier classes
CROP = 2 * PITCH                            # hole crop size for the classifier (48 px)


@dataclass
class Part:
    holes: list            # indices into board.holes covered by this blob
    ends: tuple            # (hole index, hole index)
    conf: float            # 0..1 confidence in the end assignment
    area: int
    bbox: tuple
    contour: np.ndarray = field(repr=False, default=None)


@dataclass
class Frame:
    warped: np.ndarray
    diff: np.ndarray        # per-pixel difference (float32)
    mask: np.ndarray        # changed-pixel mask (uint8)
    p_cov: np.ndarray       # per-hole covered probability
    p_plug: np.ndarray | None
    parts: list
    obstructed: bool        # huge change (hand / arm) -> can't verify


def _lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


class Vision:
    def __init__(self, board: Board, empty_rgb: np.ndarray, cfg: dict | None = None, model_path: str | None = None):
        self.b = board
        c = {"px_thresh": 18.0, "hole_thresh": 14.0, "l_weight": 0.5, "min_area_pitch2": 0.6,
             "obstructed_frac": 0.25, "hole_radius": 0.25}
        c.update(cfg or {})
        self.c = c
        self.aligner = None
        if c.get("align", True):
            try:
                from .align import Aligner
                self.aligner = Aligner(empty_rgb)
            except Exception as e:  # no cv2.aruco
                print(f"[align] disabled: {e}")
        self.ref_lab = _lab(board.warp(empty_rgb))
        self.ref_gray = cv2.GaussianBlur(cv2.cvtColor(board.warp(empty_rgb), cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
        self.ref_med_L = float(np.median(self.ref_lab[..., 0]))
        h, w = self.ref_lab.shape[:2]
        # one disk mask per hole, as flat pixel index arrays, for fast per-hole means
        r = c["hole_radius"] * PITCH
        yy, xx = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
        disk = (xx ** 2 + yy ** 2) <= r * r
        self.hole_px = []
        for (x, y) in board.xy:
            xs = np.clip(np.round(x).astype(int) + xx[disk], 0, w - 1)
            ys = np.clip(np.round(y).astype(int) + yy[disk], 0, h - 1)
            self.hole_px.append(ys * w + xs)
        self.model = HoleClassifier(model_path) if model_path else None

    def refine(self, warped):
        """Sub-hole alignment of the warped board to the empty reference (ECC, rotation + shift)."""
        if not self.c.get("refine", True):
            return warped
        g = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(self.ref_gray, g, M, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error:
            return warped
        if abs(M[0, 2]) > PITCH * 0.6 or abs(M[1, 2]) > PITCH * 0.6:  # too big = probably locked onto the wrong hole
            return warped
        h, w = warped.shape[:2]
        return cv2.warpAffine(warped, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def diff_map(self, warped_rgb):
        lab = _lab(warped_rgb)
        lab[..., 0] += self.ref_med_L - float(np.median(lab[..., 0]))  # global brightness match
        d = lab - self.ref_lab
        d[..., 0] *= self.c["l_weight"]                                  # shadows mostly change L
        dist = np.sqrt((d ** 2).sum(-1))
        return cv2.GaussianBlur(dist, (5, 5), 0)

    def analyze(self, rgb: np.ndarray) -> Frame:
        if self.aligner is not None:
            rgb = self.aligner.align(rgb)
        warped = self.refine(self.b.warp(rgb))
        dist = self.diff_map(warped)
        flat = dist.ravel()
        score = np.array([flat[p].mean() for p in self.hole_px], np.float32)
        t = self.c["hole_thresh"]
        p_cov = 1.0 / (1.0 + np.exp(-(score - t) / (0.25 * t)))

        mask = (dist > self.c["px_thresh"]).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        obstructed = (mask > 0).mean() > self.c["obstructed_frac"]

        p_plug = self.model.predict(warped, self.b.xy) if self.model else None

        parts = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = self.c["min_area_pitch2"] * PITCH * PITCH
        hx = np.clip(np.round(self.b.xy[:, 0]).astype(int), 0, mask.shape[1] - 1)
        hy = np.clip(np.round(self.b.xy[:, 1]).astype(int), 0, mask.shape[0] - 1)
        # assign each covered hole to the nearest blob (within assign_pitch): thin metal leads often
        # don't show up in the blob mask, but the hole they plug into does change.
        blobs = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
        # with a model, holes it calls plugged count even if they look unchanged (parallax / occlusion)
        covered = np.where((p_cov > 0.5) | ((p_plug > 0.5) if p_plug is not None else False))[0]
        hole_label = np.zeros(len(self.b.holes), int)
        if blobs and len(covered):
            dists = np.stack([cv2.distanceTransform(np.where(labels == k, 0, 255).astype(np.uint8),
                                                    cv2.DIST_L2, 5)[hy[covered], hx[covered]] for k in blobs])
            best = dists.argmin(0)
            ok = dists[best, np.arange(len(covered))] <= self.c.get("assign_pitch", 1.5) * PITCH
            hole_label[covered[ok]] = np.array(blobs)[best[ok]]
        for k in blobs:
            area = int(stats[k, cv2.CC_STAT_AREA])
            idx = np.where(hole_label == k)[0]
            if p_plug is not None:
                plugged = [i for i in idx if p_plug[i] > 0.5]
                if len(plugged) >= 2:
                    idx_for_ends = np.array(plugged)
                else:
                    idx_for_ends = idx
            else:
                idx_for_ends = idx
            if len(idx_for_ends) < 2:
                continue
            ends, conf = self._ends(labels == k, idx_for_ends)
            if p_plug is not None:
                conf = min(1.0, conf * 0.5 + 0.5 * float(min(p_plug[ends[0]], p_plug[ends[1]])))
            cnts, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = (int(v) for v in stats[k, :4])
            parts.append(Part(idx.tolist(), ends, conf, area, (x, y, w, h), cnts[0] if cnts else None))
        return Frame(warped, dist, mask, p_cov, p_plug, parts, obstructed)

    def _ends(self, blob: np.ndarray, idx: np.ndarray):
        ys, xs = np.nonzero(blob)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mean = pts.mean(0)
        cov = np.cov((pts - mean).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, -1]
        elong = float(np.sqrt(max(evals[-1], 1e-6) / max(evals[0], 1e-6)))
        proj = (self.b.xy[idx] - mean) @ axis
        order = np.argsort(proj)
        i0, i1 = int(idx[order[0]]), int(idx[order[-1]])
        # gap between the extreme hole and the next one along the axis: clear ends -> confident
        sp = np.sort(proj)
        span = sp[-1] - sp[0]
        conf = 1.0
        if elong < 2.0:            # compact blob (LED, button): ends are a guess
            conf = 0.5
        if span < 1.5 * PITCH:     # ends in adjacent holes
            conf = min(conf, 0.6)
        return (i0, i1), conf


class HoleClassifier:
    """Optional TFLite model: crop around each hole -> P(empty, covered, plugged)."""

    def __init__(self, path: str):
        try:
            import tflite_runtime.interpreter as tfl
        except ImportError:  # laptop: full TensorFlow
            import tensorflow.lite as tfl  # type: ignore
        self.it = tfl.Interpreter(model_path=path, num_threads=4)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.n = None

    def predict(self, warped: np.ndarray, xy: np.ndarray) -> np.ndarray:
        crops = crop_holes(warped, xy)
        if self.n != len(crops):
            self.it.resize_tensor_input(self.inp["index"], [len(crops), CROP, CROP, 3])
            self.it.allocate_tensors()
            self.n = len(crops)
        self.it.set_tensor(self.inp["index"], crops.astype(self.inp["dtype"]))
        self.it.invoke()
        probs = self.it.get_tensor(self.out["index"])
        return probs[:, CLASSES.index("plugged")]


def crop_holes(warped: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h = CROP // 2
    pad = cv2.copyMakeBorder(warped, h, h, h, h, cv2.BORDER_REFLECT)
    out = np.empty((len(xy), CROP, CROP, 3), np.uint8)
    for i, (x, y) in enumerate(xy):
        cx, cy = int(round(x)) + h, int(round(y)) + h
        out[i] = pad[cy - h:cy + h, cx - h:cx + h]
    return out
