from __future__ import print_function

import os
import random
import sys
import copy
import csv
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from mod_construct_brick_Y import get_all_bricks, pieces
from mod_construct_supercell_Y import check_move_water_hydrogens, get_angles, get_full_coordinates, resize_crystal
from mod_sample import fill_water, sample_Ca_Si_ratio
from mod_write_Y import get_lammps_input_cementff, write_cementff4_mapping_json, write_cementff4_zinc_input
from mod_zinc import (
    apply_charge_balance,
    apply_zinc_modification,
    apply_zinc_sites,
    build_multi_zinc_summary,
    build_zinc_candidate_site_report,
    finalize_zinc_summary,
    inspect_zinc_candidates,
    remap_zinc_angles,
    select_multi_zinc_sites,
    total_charge,
    validate_no_zinc_bonds,
    write_zinc_summary,
)
from validate_cementff_data import parse_data


UNITCELL = np.array(
    [
        [6.7352, 0.0, 0.0],
        [-4.071295, 6.209521, 0.0],
        [0.7037701, -6.2095578, 13.9936836],
    ]
)


MULTI_ZN_SITE_MODES = {
    "multi_q2b",
    "multi_q1",
    "q1_q2b_single_structure_mixture",
}


def _ratio_or_none(numerator, denominator):
    return None if denominator in (None, 0) else float(numerator) / float(denominator)


def write_json(path, obj):
    with open(path, "w") as f:
        import json
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_single_row_csv(path, row, fields=None):
    fields = fields or sorted(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


COMPOSITION_SUMMARY_FIELDS = [
    "N_Ca",
    "N_Si_parent_before_zn",
    "N_Si_final",
    "N_Zn",
    "target_Ca_Si",
    "actual_Ca_Si_final",
    "target_Zn_Si",
    "actual_Zn_Si_final",
    "target_Zn_count",
    "actual_Zn_count",
    "site_mode",
    "seed",
    "within_Ca_Si_tolerance",
    "within_Zn_Si_tolerance",
    "Ca_Si_error",
    "Zn_Si_error",
]


def compute_composition_summary(
    data_file=None,
    entries=None,
    target_ca_si=1.7,
    target_zn_si=None,
    target_zn_count=None,
    site_mode=None,
    seed=None,
    n_si_parent_before_zn=None,
    ca_si_tol=None,
    zn_si_tol=None,
):
    """Count real Ca, Si, and Zn centers without counting shell pseudoatoms."""
    if data_file:
        data = parse_data(data_file)
        atoms = data["atoms"].values()
        n_ca = sum(1 for atom in atoms if int(atom["type"]) == 1)
        n_si = sum(1 for atom in data["atoms"].values() if int(atom["type"]) == 2)
        n_zn = sum(1 for atom in data["atoms"].values() if int(atom["type"]) == 9)
    elif entries is not None:
        n_ca = sum(1 for entry in entries if int(entry[1]) in (1, 9))
        n_si = sum(1 for entry in entries if int(entry[1]) in (2, 10))
        n_zn = sum(1 for entry in entries if int(entry[1]) == 14)
    else:
        raise ValueError("compute_composition_summary requires data_file or entries")
    actual_ca_si = _ratio_or_none(n_ca, n_si)
    actual_zn_si = _ratio_or_none(n_zn, n_si)
    ca_error = None if actual_ca_si is None or target_ca_si is None else actual_ca_si - float(target_ca_si)
    zn_error = None if actual_zn_si is None or target_zn_si is None else actual_zn_si - float(target_zn_si)
    return {
        "N_Ca": int(n_ca),
        "N_Si_parent_before_zn": None if n_si_parent_before_zn is None else int(n_si_parent_before_zn),
        "N_Si_final": int(n_si),
        "N_Zn": int(n_zn),
        "target_Ca_Si": None if target_ca_si is None else float(target_ca_si),
        "actual_Ca_Si_final": actual_ca_si,
        "target_Zn_Si": None if target_zn_si is None else float(target_zn_si),
        "actual_Zn_Si_final": actual_zn_si,
        "target_Zn_count": None if target_zn_count is None else int(target_zn_count),
        "actual_Zn_count": int(n_zn),
        "site_mode": site_mode,
        "seed": None if seed is None else int(seed),
        "within_Ca_Si_tolerance": None if ca_error is None or ca_si_tol is None else abs(ca_error) <= float(ca_si_tol),
        "within_Zn_Si_tolerance": None if zn_error is None or zn_si_tol is None else abs(zn_error) <= float(zn_si_tol),
        "Ca_Si_error": ca_error,
        "Zn_Si_error": zn_error,
    }


def _target_count_from_ratio(target_zn_si, n_si_parent):
    return max(1, int(round(float(target_zn_si) * int(n_si_parent))))


def _multi_counts_from_total(mode, total):
    total = max(1, int(total))
    if mode == "multi_q2b":
        return 0, total
    if mode == "multi_q1":
        return total, 0
    if mode == "q1_q2b_single_structure_mixture":
        q1_weight, q2b_weight = _q1_q2b_ratio_from_env()
        q1_raw = float(total) * q1_weight / float(q1_weight + q2b_weight)
        q1 = int(math.floor(q1_raw + 0.5))
        q1 = min(total, max(0, q1))
        q2b = total - q1
        if total > 1 and q1 == 0:
            q1, q2b = 1, total - 1
        if total > 1 and q2b == 0:
            q1, q2b = total - 1, 1
        return q1, q2b
    raise ValueError("Unsupported multi-Zn site mode {}".format(mode))


def _q1_q2b_ratio_from_env():
    raw = os.environ.get("PYCSH_ZN_Q1_Q2B_RATIO", "1:1").strip()
    parts = raw.replace(",", ":").split(":")
    if len(parts) != 2:
        raise ValueError("PYCSH_ZN_Q1_Q2B_RATIO must look like '1:1'")
    q1_weight = float(parts[0])
    q2b_weight = float(parts[1])
    if q1_weight < 0 or q2b_weight < 0 or (q1_weight + q2b_weight) <= 0:
        raise ValueError("PYCSH_ZN_Q1_Q2B_RATIO weights must be non-negative and non-zero in total")
    return q1_weight, q2b_weight


def _multi_zn_min_distance_from_env(default=5.0):
    raw = os.environ.get("PYCSH_ZN_MIN_ZN_ZN_DISTANCE")
    if raw is None or str(raw).strip() == "":
        return float(default)
    value = float(raw)
    if value < 0:
        raise ValueError("PYCSH_ZN_MIN_ZN_ZN_DISTANCE must be non-negative")
    return value


def apply_multi_zinc_modification(
    entries,
    crystal_dict,
    bonds,
    angles,
    supercell,
    site_mode,
    seed,
    n_q1,
    n_q2b,
    ca_si_ratio,
    target_zn_si,
    min_zn_zn_distance=5.0,
    max_attempts=100,
):
    candidates = inspect_zinc_candidates(crystal_dict)
    candidate_site_report = build_zinc_candidate_site_report(
        candidates, entries, bonds, angles, supercell, False, True, 1.95
    )
    selected_sites, rejected_candidates, _ = select_multi_zinc_sites(
        candidates,
        candidate_site_report,
        entries,
        bonds,
        supercell,
        site_mode,
        seed,
        n_q1=n_q1,
        n_q2b=n_q2b,
        min_zn_zn_distance=min_zn_zn_distance,
        max_attempts=max_attempts,
    )
    entries_m = copy.deepcopy(entries)
    bonds_m = copy.deepcopy(bonds)
    angles_m = copy.deepcopy(angles)
    crystal_dict_m = copy.deepcopy(crystal_dict)
    entries_m, crystal_dict_m = apply_zinc_sites(entries_m, crystal_dict_m, selected_sites)
    hydroxylation_records = apply_charge_balance(
        entries_m,
        bonds_m,
        angles_m,
        selected_sites,
        supercell,
        "hydroxylate_two_oxygens",
        False,
        True,
        1.95,
    )
    topology = remap_zinc_angles(entries_m, angles_m)
    topology["zinc_bonds"] = validate_no_zinc_bonds(entries_m, bonds_m)
    zinc_summary = build_multi_zinc_summary(
        entries_m,
        selected_sites,
        candidates,
        candidate_site_report,
        rejected_candidates,
        0.0 if target_zn_si is None else target_zn_si,
        ca_si_ratio,
        supercell,
        site_mode,
        seed,
        min_zn_zn_distance,
    )
    zinc_summary["hydroxylation_records"] = hydroxylation_records
    zinc_summary["topology_validation"] = topology
    zinc_summary["total_charge_after_hydroxylation"] = total_charge(entries_m)
    zinc_summary["charge_residual_final"] = total_charge(entries_m)
    zinc_summary["total_charge_residual"] = total_charge(entries_m)
    zinc_summary["N_Os_converted_to_Oh"] = sum(len(record["hydroxylated_oxygens"]) for record in hydroxylation_records)
    zinc_summary["N_H_added_for_Zn_OH"] = zinc_summary["N_Os_converted_to_Oh"]
    return entries_m, crystal_dict_m, bonds_m, angles_m, zinc_summary


def generate_structure(
    output_dir,
    prefix,
    enable_zinc=False,
    zn_ratio=0.03,
    site_type="Q2b_Zn",
    seed=23743,
    target_ca_si=1.7,
    target_w_si=0.2,
    ca_si_width=None,
    composition_widths=None,
    target_zn_si=None,
    target_zn_count=None,
    ca_si_tol=None,
    zn_si_tol=None,
):
    if target_zn_si is not None and target_zn_count is not None:
        raise ValueError("target_zn_si and target_zn_count are mutually exclusive")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    np.random.seed(seed)
    random.seed(seed + 10)
    size = (2, 2, 2)
    supercell = np.zeros((3, 3))
    for i in range(3):
        supercell[i, :] = UNITCELL[i, :] * size[i]
    bricks, sorted_bricks = get_all_bricks(pieces)
    widths = composition_widths if composition_widths is not None else [0.06, 0.08, 0.08]
    widths = list(widths)
    if ca_si_width is not None:
        widths[0] = float(ca_si_width)
    crystal, n_ca, n_si, r_sioh, r_caoh, mcl, n_water, _ = sample_Ca_Si_ratio(
        sorted_bricks, target_ca_si, target_w_si, size[0] * size[1] * size[2], widths
    )
    water_in_crystal = fill_water(crystal, n_water)
    crystal_rs, water_in_crystal_rs = resize_crystal(crystal, water_in_crystal, size)
    entries, bonds, crystal_dict, water_dict = get_full_coordinates(
        crystal_rs, water_in_crystal_rs, size, pieces, False, [0]
    )
    angles = get_angles(crystal_dict, water_dict, size)
    zinc_summary = None
    requested_zn_count = None
    if enable_zinc:
        if target_zn_count is not None:
            requested_zn_count = int(target_zn_count)
        elif target_zn_si is not None:
            requested_zn_count = _target_count_from_ratio(target_zn_si, n_si)
        if site_type in MULTI_ZN_SITE_MODES:
            if requested_zn_count is None:
                requested_zn_count = _target_count_from_ratio(zn_ratio, n_si)
            n_q1, n_q2b = _multi_counts_from_total(site_type, requested_zn_count)
            entries, crystal_dict, bonds, angles, zinc_summary = apply_multi_zinc_modification(
                entries,
                crystal_dict,
                bonds,
                angles,
                supercell,
                site_type,
                seed,
                n_q1,
                n_q2b,
                n_ca / n_si,
                target_zn_si if target_zn_si is not None else zn_ratio,
                min_zn_zn_distance=_multi_zn_min_distance_from_env(),
            )
        else:
            generation_zn_ratio = zn_ratio
            if target_zn_count is not None:
                generation_zn_ratio = float(target_zn_count) / float(n_si)
            elif target_zn_si is not None:
                generation_zn_ratio = 1.0 / float(n_si)
            entries, crystal_dict, zinc_summary = apply_zinc_modification(
                entries, crystal_dict, supercell, generation_zn_ratio, site_type, seed, n_ca / n_si,
                "hydroxylate_two_oxygens", bonds, angles, False, True, 1.95
            )
            zinc_summary = finalize_zinc_summary(
                entries, bonds, angles, supercell, zinc_summary, "hydroxylate_two_oxygens", False
            )
        if zinc_summary is not None:
            zinc_summary["requested_target_Ca_Si"] = float(target_ca_si)
            zinc_summary["requested_target_Zn_Si"] = None if target_zn_si is None else float(target_zn_si)
            zinc_summary["requested_target_Zn_count"] = requested_zn_count
    entries, _, _ = check_move_water_hydrogens(entries)
    data_name = prefix + ("_cementff_zn.data" if enable_zinc else "_cementff.data")
    data_path = os.path.join(output_dir, data_name)
    water_summary = get_lammps_input_cementff(data_path, entries, bonds, angles, supercell, zinc_summary)
    mapping_path = os.path.join(output_dir, "cementff_mapping_summary.json")
    write_cementff4_mapping_json(mapping_path, enable_zinc)
    water_path = os.path.join(output_dir, "water_summary.json")
    with open(water_path, "w") as f:
        import json
        json.dump(water_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    composition_summary = compute_composition_summary(
        data_file=data_path,
        target_ca_si=target_ca_si,
        target_zn_si=target_zn_si if target_zn_si is not None else (zn_ratio if enable_zinc else None),
        target_zn_count=requested_zn_count,
        site_mode=site_type if enable_zinc else "pure_csh",
        seed=seed,
        n_si_parent_before_zn=n_si,
        ca_si_tol=ca_si_tol,
        zn_si_tol=zn_si_tol,
    )
    composition_json = os.path.join(output_dir, "composition_summary.json")
    composition_csv = os.path.join(output_dir, "composition_summary.csv")
    write_json(composition_json, composition_summary)
    write_single_row_csv(composition_csv, composition_summary, COMPOSITION_SUMMARY_FIELDS)
    result = {
        "data_file": data_path,
        "mapping_summary": mapping_path,
        "water_summary": water_path,
        "composition_summary": composition_json,
        "composition_summary_csv": composition_csv,
        "zinc_summary": None,
        "forcefield": None,
    }
    if enable_zinc:
        zinc_path = os.path.join(output_dir, "zinc_summary.json")
        write_zinc_summary(zinc_path, zinc_summary)
        ff_path = os.path.join(output_dir, "in.CementFF4_Zn")
        write_cementff4_zinc_input(ff_path)
        result["zinc_summary"] = zinc_path
        result["forcefield"] = ff_path
    return result
