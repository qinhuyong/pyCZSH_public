from __future__ import print_function

import math

import numpy as np


FRAMEWORK_SPECIES = {1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 14}


def _coords(entries):
    return np.array([[float(e[3]), float(e[4]), float(e[5])] for e in entries], dtype=float).reshape((-1, 3))


def _framework_mask(entries):
    return np.array([int(e[1]) in FRAMEWORK_SPECIES for e in entries], dtype=bool)


def cart_to_frac(coords, cell):
    cell = np.asarray(cell, dtype=float)
    coords = np.asarray(coords, dtype=float)
    return np.linalg.solve(cell.T, coords.T).T


def frac_to_cart(frac, cell):
    return np.dot(np.asarray(frac, dtype=float), np.asarray(cell, dtype=float))


def restricted_triclinic_bounds(cell):
    cell = np.asarray(cell, dtype=float)
    lx = float(cell[0, 0])
    ly = float(cell[1, 1])
    lz = float(cell[2, 2])
    xy = float(cell[1, 0])
    xz = float(cell[2, 0])
    yz = float(cell[2, 1])
    return {
        "xlo_bound": min(0.0, xy, xz, xy + xz),
        "xhi_bound": lx + max(0.0, xy, xz, xy + xz),
        "ylo_bound": min(0.0, yz),
        "yhi_bound": ly + max(0.0, yz),
        "zlo_bound": 0.0,
        "zhi_bound": lz,
        "xy": xy,
        "xz": xz,
        "yz": yz,
        "lx": lx,
        "ly": ly,
        "lz": lz,
    }


def cell_lengths(cell):
    cell = np.asarray(cell, dtype=float)
    return [float(np.linalg.norm(cell[i, :])) for i in range(3)]


def _span(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return None, None, None
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    return lo, hi, hi - lo


def compute_framework_occupancy_summary(entries, cell, label="final", recenter_summary=None):
    coords = _coords(entries)
    mask = _framework_mask(entries)
    fcoords = coords[mask]
    warnings = []
    summary = {
        "label": label,
        "framework_atom_count": int(np.sum(mask)),
        "total_atom_count": int(len(entries)),
        "warnings": warnings,
    }
    if fcoords.size == 0:
        warnings.append("no_framework_atoms")
        return summary

    frac = cart_to_frac(fcoords, cell) % 1.0
    lengths = cell_lengths(cell)
    for axis, idx in zip(("x", "y", "z"), range(3)):
        lo, hi, span = _span(fcoords[:, idx])
        flo, fhi, fspan = _span(frac[:, idx])
        summary["{}_min".format(axis)] = lo
        summary["{}_max".format(axis)] = hi
        summary["{}_span".format(axis)] = span
        summary["frac_min_{}".format(axis)] = flo
        summary["frac_max_{}".format(axis)] = fhi
        summary["frac_span_{}".format(axis)] = fspan
        summary["occ_ratio_{}".format(axis)] = None if lengths[idx] == 0 else float(span) / float(lengths[idx])
        if fspan is not None and fspan < 0.5:
            warnings.append("possible_underfilled_supercell_{}".format(axis))
            warnings.append("possible_oversized_cell_{}".format(axis))
            warnings.append("possible_translation_issue_{}".format(axis))
    if recenter_summary:
        summary["recenter_applied"] = bool(recenter_summary.get("applied"))
        for axis in ("x", "y", "z"):
            summary["recenter_largest_gap_after_{}".format(axis)] = recenter_summary.get("largest_gap_after_{}".format(axis))
            summary["recenter_fractional_span_after_{}".format(axis)] = recenter_summary.get("fractional_span_after_{}".format(axis))
    summary["occupancy_warning"] = ";".join(sorted(set(warnings)))
    return summary


def compute_cell_geometry_summary(cell_generation, cell_export=None, entries=None):
    gen = np.asarray(cell_generation, dtype=float)
    exp = np.asarray(cell_export if cell_export is not None else cell_generation, dtype=float)
    warnings = []
    diff = exp - gen
    roundtrip_error = None
    if entries is not None and len(entries):
        coords = _coords(entries)
        frac = cart_to_frac(coords, gen)
        back = frac_to_cart(frac, gen)
        roundtrip_error = float(np.max(np.abs(back - coords)))
        if roundtrip_error > 1.0e-8:
            warnings.append("cart_frac_roundtrip_error")
    if float(np.max(np.abs(diff))) > 1.0e-8:
        warnings.append("generation_export_cell_matrix_mismatch")
    bounds = restricted_triclinic_bounds(exp)
    axis_widths = [
        bounds["xhi_bound"] - bounds["xlo_bound"],
        bounds["yhi_bound"] - bounds["ylo_bound"],
        bounds["zhi_bound"] - bounds["zlo_bound"],
    ]
    lengths = cell_lengths(exp)
    if any(axis_widths[i] > lengths[i] * 1.2 for i in range(3) if lengths[i] > 0):
        warnings.append("restricted_triclinic_bounding_box_larger_than_cell_vectors")
        warnings.append("visualization_tool_may_show_large_outer_brick")
    return {
        "cell_matrix_generation": gen.tolist(),
        "cell_matrix_export": exp.tolist(),
        "cell_matrix_max_abs_delta": float(np.max(np.abs(diff))),
        "is_triclinic": bool(abs(bounds["xy"]) > 1.0e-12 or abs(bounds["xz"]) > 1.0e-12 or abs(bounds["yz"]) > 1.0e-12),
        "xy": bounds["xy"],
        "xz": bounds["xz"],
        "yz": bounds["yz"],
        "cell_vector_lengths": lengths,
        "restricted_triclinic_axis_aligned_widths": axis_widths,
        "cart_frac_roundtrip_max_error": roundtrip_error,
        "export_consistency_passed": not any(w == "generation_export_cell_matrix_mismatch" or w == "cart_frac_roundtrip_error" for w in warnings),
        "warnings": sorted(set(warnings)),
    }


def audit_supercell_population(crystal_rs, size):
    requested_shape = [int(x) for x in size]
    expected = int(requested_shape[0] * requested_shape[1] * requested_shape[2])
    occupied = []
    missing = []
    duplicates = []
    seen = set()
    for i in range(requested_shape[0]):
        for j in range(requested_shape[1]):
            for k in range(requested_shape[2]):
                item = crystal_rs[i, j, k]
                slot = [i, j, k]
                if item is None or item == 0:
                    missing.append(slot)
                    continue
                occupied.append(slot)
                key = tuple(slot)
                if key in seen:
                    duplicates.append(slot)
                seen.add(key)
    warnings = []
    if missing:
        warnings.append("missing_translation_slots")
    if duplicates:
        warnings.append("duplicate_translation_slots")
    if len(occupied) < expected:
        warnings.append("possible_underfilled_supercell")
    return {
        "requested_shape": requested_shape,
        "expected_brick_count": expected,
        "actual_brick_count": int(len(occupied)),
        "unique_translation_vectors": occupied,
        "translation_vector_count": int(len(occupied)),
        "missing_translation_slots": missing,
        "duplicate_translation_slots": duplicates,
        "occupancy_grid_summary": {
            "occupied_slots": int(len(occupied)),
            "missing_slots": int(len(missing)),
            "duplicate_slots": int(len(duplicates)),
        },
        "supercell_population_warning": ";".join(sorted(set(warnings))),
        "warnings": sorted(set(warnings)),
    }


def audit_triclinic_export_consistency(cell_generation, cell_export=None, entries=None):
    return compute_cell_geometry_summary(cell_generation, cell_export=cell_export, entries=entries)


def audit_deduplication(stage_counts):
    counts = dict(stage_counts or {})
    before = counts.get("atom_count_before_dedup")
    after = counts.get("atom_count_after_dedup")
    warnings = []
    removed = None
    frac = None
    present = before is not None or after is not None
    if before is not None and after is not None:
        removed = int(before) - int(after)
        frac = 0.0 if int(before) == 0 else float(removed) / float(before)
        if removed > 0:
            warnings.append("atoms_removed_by_dedup")
        if frac is not None and frac > 0.05:
            warnings.append("suspicious_large_atom_drop")
    else:
        warnings.append("no_dedup_step_detected")
    return {
        "dedup_step_present": bool(present),
        "atom_count_before_dedup": before,
        "atom_count_after_dedup": after,
        "removed_atom_count": removed,
        "removed_atom_fraction": frac,
        "suspicious_large_atom_drop": bool(frac is not None and frac > 0.05),
        "dedup_warning": ";".join(sorted(set(warnings))),
        "warnings": sorted(set(warnings)),
    }


def classify_root_cause(occupancy, cell_geometry, supercell_population, dedup):
    warnings = []
    for obj in (occupancy, cell_geometry, supercell_population, dedup):
        warnings.extend(obj.get("warnings", []))
    if any("generation_export_cell_matrix_mismatch" == w for w in warnings):
        return "export inconsistency"
    if any("cart_frac_roundtrip_error" == w for w in warnings):
        return "export inconsistency"
    if any(w in ("missing_translation_slots", "possible_underfilled_supercell") for w in supercell_population.get("warnings", [])):
        return "incomplete supercell population"
    if dedup.get("suspicious_large_atom_drop"):
        return "erroneous deduplication"
    if any(str(w).startswith("possible_underfilled_supercell_") for w in occupancy.get("warnings", [])):
        return "oversized cell"
    if any(w == "visualization_tool_may_show_large_outer_brick" for w in cell_geometry.get("warnings", [])):
        return "representation only"
    return "representation only"


def compact_orthogonal_visual_entries(entries):
    coords = _coords(entries)
    if len(entries) == 0:
        return [], np.eye(3), {"warnings": ["no_atoms"]}
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    spans = maxs - mins
    pad = np.array([2.0, 2.0, 2.0])
    visual_cell = np.diag(spans + 2.0 * pad)
    shifted = coords - mins + pad
    out = []
    for entry, coord in zip(entries, shifted):
        item = list(entry)
        item[3] = float(coord[0])
        item[4] = float(coord[1])
        item[5] = float(coord[2])
        out.append(item)
    summary = {
        "visualization_only": True,
        "method": "axis_aligned_compact_orthogonal_bbox",
        "source_cartesian_min": [float(x) for x in mins],
        "source_cartesian_max": [float(x) for x in maxs],
        "source_cartesian_span": [float(x) for x in spans],
        "padding_angstrom": [float(x) for x in pad],
        "visual_cell_matrix": visual_cell.tolist(),
        "warnings": [
            "visualization_only_not_for_validation",
            "does_not_preserve_periodic_triclinic_cell",
        ],
    }
    return out, visual_cell, summary
