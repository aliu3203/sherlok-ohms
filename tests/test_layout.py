import pytest

from grid import FULL_830, HALF_400, hole_positions_mm, node_of


@pytest.mark.parametrize('spec, count', [(FULL_830, 830), (HALF_400, 400)])
def test_hole_count(spec, count):
    assert len(hole_positions_mm(spec)) == count


@pytest.mark.parametrize('spec', [FULL_830, HALF_400])
def test_terminal_geometry(spec):
    holes = hole_positions_mm(spec)
    n = spec.columns
    assert holes['a1'] == (0, 0)
    assert holes[f'a{n}'][0] == pytest.approx((n - 1) * 2.54)
    assert holes['b1'][1] - holes['a1'][1] == pytest.approx(2.54)
    assert holes['f1'][1] - holes['e1'][1] == pytest.approx(7.62)  # trench
    assert holes[f'j{n}'] == pytest.approx((spec.terminal_width_mm, spec.terminal_height_mm))


def test_rail_groups_skip_one_hole():
    holes = hole_positions_mm(FULL_830)
    x = lambda i: holes[f'top_plus_{i}'][0]
    assert x(2) - x(1) == pytest.approx(2.54)
    assert x(6) - x(5) == pytest.approx(2 * 2.54)
    assert x(50) - x(1) == pytest.approx(58 * 2.54)


def test_nodes():
    assert node_of(FULL_830, 'a5') == node_of(FULL_830, 'e5') == 'left:5'
    assert node_of(FULL_830, 'f5') == node_of(FULL_830, 'j5') == 'right:5'
    assert node_of(FULL_830, 'top_plus_1') == node_of(FULL_830, 'top_plus_50') == 'rail:top_plus'
    with pytest.raises(ValueError):
        node_of(HALF_400, 'a31')
    with pytest.raises(ValueError):
        node_of(FULL_830, 'top_plus_51')
