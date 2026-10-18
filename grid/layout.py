"""Pure geometry: every hole's position in the board frame, and its electrical node."""
import re


def hole_positions_mm(spec):
    """Ordered {name: (x_mm, y_mm)}: terminals a1..j{N}, then rails a_minus_1, ..."""
    p = spec.pitch_mm
    half = len(spec.rows) // 2
    holes = {}
    for r, row in enumerate(spec.rows):
        y = (r + (spec.trench_pitches - 1 if r >= half else 0)) * p
        for c in range(spec.columns):
            holes[f'{row}{c + 1}'] = (c * p, y)
    for rail, y_pitches in spec.rail_rows.items():
        for i in range(spec.rail_holes_per_row):
            x = spec.rail_first_x_pitches + i + (i // spec.rail_group) * spec.rail_gap_pitches
            holes[f'{rail}_{i + 1}'] = (x * p, y_pitches * p)
    return holes


def node_of(spec, hole):
    """Electrical node: 'ae:5' (a5-e5), 'fj:5' (f5-j5), or 'rail:a_plus' (rail beside row a; '...:1'/'...:2' if split)."""
    m = re.fullmatch(rf'([{spec.rows}])(\d+)', hole)
    if m and 1 <= int(m[2]) <= spec.columns:
        half = len(spec.rows) // 2
        side = spec.rows[:half] if spec.rows.index(m[1]) < half else spec.rows[half:]
        return f'{side[0]}{side[-1]}:{m[2]}'
    rail, _, index = hole.rpartition('_')
    if rail in spec.rail_rows and index.isdigit() and 1 <= int(index) <= spec.rail_holes_per_row:
        if spec.rail_split:
            return f'rail:{rail}:{1 if int(index) <= spec.rail_holes_per_row // 2 else 2}'
        return f'rail:{rail}'
    raise ValueError(f'Unknown hole {hole!r} on {spec.name} board')
