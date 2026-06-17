from __future__ import print_function

import numpy as np


FRAMEWORK_SPECIES = {1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14}


def framework_mask_from_entries(entries):
    return np.array([int(entry[1]) in FRAMEWORK_SPECIES for entry in entries], dtype=bool)


def cart_to_frac(coords, cell):
    cell = np.asarray(cell, dtype=float)
    coords = np.asarray(coords, dtype=float)
    return np.linalg.solve(cell.T, coords.T).T


def frac_to_cart(frac, cell):
    cell = np.asarray(cell, dtype=float)
    frac = np.asarray(frac, dtype=float)
    return np.dot(frac, cell)


def axis_gap_stats(values):
    vals = np.sort(np.asarray(values, dtype=float) % 1.0)
    if vals.size == 0:
        return {
            "largest_gap": None,
            "largest_gap_start": None,
            "largest_gap_center": None,
            "fractional_span": None,
        }
    if vals.size == 1:
        v = float(vals[0])
        return {
            "largest_gap": 1.0,
            "largest_gap_start": v,
            "largest_gap_center": (v + 0.5) % 1.0,
            "fractional_span": 0.0,
        }
    gaps = []
    for idx in range(vals.size - 1):
        gaps.append(float(vals[idx + 1] - vals[idx]))
    gaps.append(float(vals[0] + 1.0 - vals[-1]))
    max_idx = int(np.argmax(gaps))
    gap_start = float(vals[max_idx])
    gap = float(gaps[max_idx])
    center = (gap_start + 0.5 * gap) % 1.0
    return {
        "largest_gap": gap,
        "largest_gap_start": gap_start,
        "largest_gap_center": center,
        "fractional_span": float(1.0 - gap),
    }


def _is_triclinic(cell):
    cell = np.asarray(cell, dtype=float)
    offdiag = [cell[1, 0], cell[2, 0], cell[2, 1]]
    return any(abs(x) > 1.0e-12 for x in offdiag)


def recenter_framework_largest_gap(entries, cell, enabled=True):
    coords = np.array([[float(entry[3]), float(entry[4]), float(entry[5])] for entry in entries], dtype=float).reshape((-1, 3))
    framework_mask = framework_mask_from_entries(entries)
    framework_count = int(np.sum(framework_mask))
    warnings = []
    if _is_triclinic(cell):
        warnings.append("triclinic_cell_handled")
    frac = cart_to_frac(coords, cell)
    before = []
    for dim in range(3):
        before.append(axis_gap_stats(frac[framework_mask, dim]))

    def audit_without_shift(applied):
        return {
            "applied": bool(applied),
            "method": "largest_gap_to_boundary",
            "framework_atom_count": framework_count,
            "shift_fractional_x": 0.0,
            "shift_fractional_y": 0.0,
            "shift_fractional_z": 0.0,
            "largest_gap_before_x": before[0]["largest_gap"],
            "largest_gap_before_y": before[1]["largest_gap"],
            "largest_gap_before_z": before[2]["largest_gap"],
            "largest_gap_after_x": before[0]["largest_gap"],
            "largest_gap_after_y": before[1]["largest_gap"],
            "largest_gap_after_z": before[2]["largest_gap"],
            "fractional_span_before_x": before[0]["fractional_span"],
            "fractional_span_before_y": before[1]["fractional_span"],
            "fractional_span_before_z": before[2]["fractional_span"],
            "fractional_span_after_x": before[0]["fractional_span"],
            "fractional_span_after_y": before[1]["fractional_span"],
            "fractional_span_after_z": before[2]["fractional_span"],
            "largest_gap_center_before_x": before[0]["largest_gap_center"],
            "largest_gap_center_before_y": before[1]["largest_gap_center"],
            "largest_gap_center_before_z": before[2]["largest_gap_center"],
            "largest_gap_center_after_x": before[0]["largest_gap_center"],
            "largest_gap_center_after_y": before[1]["largest_gap_center"],
            "largest_gap_center_after_z": before[2]["largest_gap_center"],
            "warnings": sorted(set(warnings)),
        }

    if not enabled:
        return entries, audit_without_shift(False)
    if framework_count < 2:
        warnings.append("framework_atom_count_too_small")
        return entries, audit_without_shift(False)

    after = []
    shifts = []
    for dim in range(3):
        shift = -float(before[dim]["largest_gap_center"])
        shifts.append(shift)
        frac[:, dim] = (frac[:, dim] + shift) % 1.0
        after.append(axis_gap_stats(frac[framework_mask, dim]))
    new_coords = frac_to_cart(frac, cell)
    updated = []
    for entry, coord in zip(entries, new_coords):
        item = list(entry)
        item[3] = float(coord[0])
        item[4] = float(coord[1])
        item[5] = float(coord[2])
        updated.append(item)
    for axis, stats in zip(("x", "y", "z"), after):
        span = stats["fractional_span"]
        if span is not None and span < 0.15:
            warnings.append("possible_large_empty_cell_or_box_size_issue_{}".format(axis))
    audit = {
        "applied": True,
        "method": "largest_gap_to_boundary",
        "framework_atom_count": framework_count,
        "shift_fractional_x": float(shifts[0]),
        "shift_fractional_y": float(shifts[1]),
        "shift_fractional_z": float(shifts[2]),
        "largest_gap_before_x": before[0]["largest_gap"],
        "largest_gap_before_y": before[1]["largest_gap"],
        "largest_gap_before_z": before[2]["largest_gap"],
        "largest_gap_after_x": after[0]["largest_gap"],
        "largest_gap_after_y": after[1]["largest_gap"],
        "largest_gap_after_z": after[2]["largest_gap"],
        "fractional_span_before_x": before[0]["fractional_span"],
        "fractional_span_before_y": before[1]["fractional_span"],
        "fractional_span_before_z": before[2]["fractional_span"],
        "fractional_span_after_x": after[0]["fractional_span"],
        "fractional_span_after_y": after[1]["fractional_span"],
        "fractional_span_after_z": after[2]["fractional_span"],
        "largest_gap_center_before_x": before[0]["largest_gap_center"],
        "largest_gap_center_before_y": before[1]["largest_gap_center"],
        "largest_gap_center_before_z": before[2]["largest_gap_center"],
        "largest_gap_center_after_x": after[0]["largest_gap_center"],
        "largest_gap_center_after_y": after[1]["largest_gap_center"],
        "largest_gap_center_after_z": after[2]["largest_gap_center"],
        "warnings": sorted(set(warnings)),
    }
    return updated, audit
