import numpy as np
import pytest

from grid import FULL_830, HALF_400, GridRegistrar, detect_markers, grid_from_markers
from synthetic import camera_homography, marker_truth_xy, render, truth_xy

CASES = [(spec, angle, seed) for spec in (FULL_830, HALF_400)
         for angle, seed in ((0, 1), (7, 2), (-90, 3), (180, 4))]


@pytest.mark.parametrize('spec, angle, seed', CASES)
def test_every_hole_lands_on_truth(spec, angle, seed):
    H = camera_homography(spec, angle_deg=angle, seed=seed)
    grid = GridRegistrar(spec).update(render(spec, H, seed=seed))
    assert grid is not None and not grid.stale
    truth = truth_xy(spec, H)
    error = np.linalg.norm(grid.xy_image - truth, axis=1)
    assert error.max() < .1 * grid.px_per_pitch, error.max()
    # Labels follow the marker IDs, so rotating the camera never relabels holes.
    for name in ('a1', f'j{spec.columns}', 'a_minus_1', 'j_minus_1'):
        assert grid.nearest(*truth[grid.index[name]]).hole == name


def test_nearest_refuses_far_and_ambiguous_points():
    H = camera_homography(FULL_830, seed=5)
    grid = grid_from_markers(FULL_830, dict(enumerate(marker_truth_xy(FULL_830, H))))
    a1, a2 = grid.image_xy('a1'), grid.image_xy('a2')
    assert grid.nearest(*a1).hole == 'a1'
    step = np.subtract(a2, a1)
    assert grid.nearest(*(a1 + .39 * step)).reason == 'ambiguous_neighbor'
    assert grid.nearest(*(a1 + .5 * step)).reason == 'too_far_from_hole'
    far = np.add(a1, 1.3 * np.subtract(a1, grid.image_xy('b1')))  # into the gap beyond row a
    assert grid.nearest(*far).hole is None


def test_hidden_marker_falls_back_to_last_good_grid():
    H = camera_homography(FULL_830, seed=6)
    registrar = GridRegistrar(FULL_830)
    good = registrar.update(render(FULL_830, H))
    occluded = render(FULL_830, H, hide_markers=(2,))
    assert set(detect_markers(occluded)) == {0, 1, 3}
    stale = registrar.update(occluded)
    assert stale.stale and np.array_equal(stale.xy_image, good.xy_image)
    assert 'Missing markers [2]' in registrar.last_error
    assert GridRegistrar(FULL_830).update(occluded) is None


def test_mirrored_marker_order_is_rejected():
    H = camera_homography(FULL_830, seed=7)
    c = dict(enumerate(marker_truth_xy(FULL_830, H)))
    with pytest.raises(ValueError, match='mirrored'):
        grid_from_markers(FULL_830, {0: c[1], 1: c[0], 2: c[3], 3: c[2]})


def test_warp_puts_holes_at_constant_positions():
    H = camera_homography(FULL_830, angle_deg=20, seed=8)
    image = render(FULL_830, H, noise=0)
    grid = GridRegistrar(FULL_830).update(image)
    top = grid.warp(image)
    assert top.shape[1::-1] == grid.warp_size
    half_pitch = grid.xy_warped[grid.index['c11']] - grid.xy_warped[grid.index['c10']]
    for name in ('a1', 'c10', 'e32', 'f32', 'j63'):
        x, y = np.round(grid.xy_warped[grid.index[name]]).astype(int)
        mx, my = np.round(grid.xy_warped[grid.index[name]] + half_pitch / 2).astype(int)
        assert top[y, x].mean() < 90 < top[my, mx].mean(), name


def test_marker_crowded_by_sticker_edge_is_still_found():
    # OpenCV's default candidate grouping drops marker 2 in this view.
    H = camera_homography(HALF_400, px_per_mm=18 * .6, angle_deg=57, seed=21)
    centers = detect_markers(render(HALF_400, H, seed=21))
    assert set(centers) == {0, 1, 2, 3}
    error = np.linalg.norm(np.array([centers[i] for i in range(4)]) - marker_truth_xy(HALF_400, H), axis=1)
    assert error.max() < .1
