from __future__ import print_function

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

_IMPORT_CWD = os.getcwd()
try:
    os.chdir(PACKAGE_ROOT)
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
    from validate_cementff_data import validate
    from forcefields.build_cementff4_zn import build as build_forcefield
    from lammps_templates.build_inputs import build as build_lammps_inputs
    from periodic_recenter import recenter_framework_largest_gap
    from cell_audit import (
        audit_deduplication,
        audit_supercell_population,
        audit_triclinic_export_consistency,
        classify_root_cause,
        compact_orthogonal_visual_entries,
        compute_framework_occupancy_summary,
    )
finally:
    os.chdir(_IMPORT_CWD)


UNITCELL = np.array(
    [
        [6.7352, 0.0, 0.0],
        [-4.071295, 6.209521, 0.0],
        [0.7037701, -6.2095578, 13.9936836],
    ]
)

SITE_MODE_TO_INTERNAL = {
    "q2b_only": "Q2b_Zn",
    "q1_only": "Q1_Zn",
    "multi_q2b": "multi_q2b",
    "multi_q1": "multi_q1",
    "q1_q2b_single_structure_mixture": "q1_q2b_single_structure_mixture",
}

VALID_LABELS = {
    "valid_static_candidate",
    "valid_q1_zn_candidate",
    "valid_q2b_zn_candidate",
    "valid_multi_q2b_zn_candidate",
    "valid_multi_q1_zn_candidate",
    "valid_multi_q1_q2b_zn_candidate",
    "needs_static_relaxation",
}

SUMMARY_FIELDS = [
    "model_id",
    "seed",
    "site_mode",
    "target_Ca_Si",
    "actual_Ca_Si_final",
    "Ca_Si_error",
    "target_W_Si",
    "target_Zn_Si",
    "actual_Zn_Si_final",
    "Zn_Si_error",
    "target_Zn_count",
    "actual_Zn_count",
    "target_q1_q2b_ratio",
    "actual_q1_q2b_ratio",
    "target_N_Q1_Zn",
    "target_N_Q2b_Zn",
    "actual_N_Q1_Zn",
    "actual_N_Q2b_Zn",
    "N_Ca",
    "N_Si_parent_before_zn",
    "N_Si_final",
    "N_Zn",
    "validation_label",
    "validation_passed",
    "accepted",
    "coordination_quality",
    "failure_reason",
    "data_file",
    "validation_json",
    "zinc_summary",
    "composition_summary",
    "recenter_applied",
    "recenter_summary_path",
    "framework_occupancy_summary",
    "cell_geometry_summary",
    "brick_placement_summary",
    "dedup_audit_summary",
    "framework_frac_span_x",
    "framework_frac_span_y",
    "framework_frac_span_z",
    "occupancy_warning",
    "expected_brick_count",
    "actual_brick_count",
    "supercell_population_warning",
    "export_consistency_passed",
    "dedup_warning",
    "cell_audit_root_cause_category",
    "lammps_dir",
    "postmin_raw_data_path",
    "postmin_internal_data_path",
    "postmin_validation_path",
    "postmin_validation_label",
    "postmin_validation_passed",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows, fields=SUMMARY_FIELDS):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def ratio_or_none(numerator, denominator):
    return None if denominator in (None, 0) else float(numerator) / float(denominator)


def target_count_from_ratio(target_zn_si, n_si_parent):
    if target_zn_si is None:
        return 0
    count = int(round(float(target_zn_si) * int(n_si_parent)))
    return max(1, count) if float(target_zn_si) > 0.0 else 0


def mixture_counts(total, q1_fraction):
    total = int(total)
    if total < 2:
        total = 2
    fraction = float(q1_fraction)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("--q1-q2b-ratio must be between 0.0 and 1.0")
    q1 = int(math.floor(float(total) * fraction + 0.5))
    q1 = min(total, max(0, q1))
    q2b = total - q1
    if q1 == 0:
        q1, q2b = 1, total - 1
    if q2b == 0:
        q1, q2b = total - 1, 1
    return int(q1), int(q2b)


def mode_target_counts(site_mode, target_zn_count, q1_q2b_ratio):
    count = max(0, int(target_zn_count or 0))
    if site_mode == "q2b_only":
        return {"N_Zn_target": 1 if count else 0, "N_Q1_target": 0, "N_Q2b_target": 1 if count else 0}
    if site_mode == "q1_only":
        return {"N_Zn_target": 1 if count else 0, "N_Q1_target": 1 if count else 0, "N_Q2b_target": 0}
    if site_mode == "multi_q2b":
        return {"N_Zn_target": count, "N_Q1_target": 0, "N_Q2b_target": count}
    if site_mode == "multi_q1":
        return {"N_Zn_target": count, "N_Q1_target": count, "N_Q2b_target": 0}
    if site_mode == "q1_q2b_single_structure_mixture":
        q1, q2b = mixture_counts(count, q1_q2b_ratio)
        return {"N_Zn_target": q1 + q2b, "N_Q1_target": q1, "N_Q2b_target": q2b}
    raise ValueError("Unsupported site_mode {}".format(site_mode))


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
    entries_m = [list(item) for item in entries]
    bonds_m = [list(item) for item in bonds]
    angles_m = [list(item) for item in angles]
    import copy

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


def composition_summary(data_file, target_ca_si, target_w_si, target_zn_si, target_counts, site_mode, seed, n_si_parent):
    from validate_cementff_data import parse_data

    data = parse_data(data_file)
    atoms = data["atoms"].values()
    n_ca = sum(1 for atom in atoms if int(atom["type"]) == 1)
    n_si = sum(1 for atom in data["atoms"].values() if int(atom["type"]) == 2)
    n_zn = sum(1 for atom in data["atoms"].values() if int(atom["type"]) == 9)
    actual_ca_si = ratio_or_none(n_ca, n_si)
    actual_zn_si = ratio_or_none(n_zn, n_si)
    return {
        "N_Ca": int(n_ca),
        "N_Si_parent_before_zn": int(n_si_parent),
        "N_Si_final": int(n_si),
        "N_Zn": int(n_zn),
        "target_Ca_Si": float(target_ca_si),
        "target_W_Si": float(target_w_si),
        "actual_Ca_Si_final": actual_ca_si,
        "Ca_Si_error": None if actual_ca_si is None else actual_ca_si - float(target_ca_si),
        "target_Zn_Si": None if target_zn_si is None else float(target_zn_si),
        "actual_Zn_Si_final": actual_zn_si,
        "Zn_Si_error": None if actual_zn_si is None or target_zn_si is None else actual_zn_si - float(target_zn_si),
        "target_Zn_count": int(target_counts["N_Zn_target"]),
        "actual_Zn_count": int(n_zn),
        "target_N_Q1_Zn": int(target_counts["N_Q1_target"]),
        "target_N_Q2b_Zn": int(target_counts["N_Q2b_target"]),
        "site_mode": site_mode,
        "seed": int(seed),
    }


def generate_structure(
    model_id,
    internal_dir,
    site_mode,
    seed,
    target_ca_si=1.7,
    target_w_si=0.2,
    target_zn_si=0.05,
    target_zn_count=None,
    q1_q2b_ratio=0.5,
    min_zn_zn_distance=5.0,
    recenter=True,
    visual_export_dir=None,
):
    ensure_dir(internal_dir)
    np.random.seed(int(seed))
    random.seed(int(seed) + 10)
    size = (2, 2, 2)
    supercell = np.zeros((3, 3))
    for i in range(3):
        supercell[i, :] = UNITCELL[i, :] * size[i]
    bricks, sorted_bricks = get_all_bricks(pieces)
    crystal, n_ca, n_si, _, _, _, n_water, _ = sample_Ca_Si_ratio(
        sorted_bricks, target_ca_si, target_w_si, size[0] * size[1] * size[2], [0.06, 0.08, 0.08]
    )
    target_count = int(target_zn_count) if target_zn_count is not None else target_count_from_ratio(target_zn_si, n_si)
    target_counts = mode_target_counts(site_mode, target_count, q1_q2b_ratio)
    contains_zinc = bool(target_counts["N_Zn_target"] > 0)
    water_in_crystal = fill_water(crystal, n_water)
    crystal_rs, water_in_crystal_rs = resize_crystal(crystal, water_in_crystal, size)
    entries, bonds, crystal_dict, water_dict = get_full_coordinates(
        crystal_rs, water_in_crystal_rs, size, pieces, False, [0]
    )
    stage_counts = {
        "atom_count_after_full_coordinates": int(len(entries)),
        "bond_count_after_full_coordinates": int(len(bonds)),
    }
    brick_summary = audit_supercell_population(crystal_rs, size)
    angles = get_angles(crystal_dict, water_dict, size)
    zinc_summary = None
    if contains_zinc:
        internal_site = SITE_MODE_TO_INTERNAL[site_mode]
        if site_mode in ("multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture"):
            entries, crystal_dict, bonds, angles, zinc_summary = apply_multi_zinc_modification(
                entries,
                crystal_dict,
                bonds,
                angles,
                supercell,
                internal_site,
                seed,
                target_counts["N_Q1_target"],
                target_counts["N_Q2b_target"],
                n_ca / float(n_si),
                target_zn_si,
                min_zn_zn_distance=min_zn_zn_distance,
            )
        else:
            single_ratio = 1.0 / float(n_si)
            entries, crystal_dict, zinc_summary = apply_zinc_modification(
                entries,
                crystal_dict,
                supercell,
                single_ratio,
                internal_site,
                seed,
                n_ca / float(n_si),
                "hydroxylate_two_oxygens",
                bonds,
                angles,
                False,
                True,
                1.95,
            )
            zinc_summary = finalize_zinc_summary(entries, bonds, angles, supercell, zinc_summary, "hydroxylate_two_oxygens", False)
        zinc_summary["requested_target_Ca_Si"] = float(target_ca_si)
        zinc_summary["requested_target_Zn_Si"] = None if target_zn_si is None else float(target_zn_si)
        zinc_summary["requested_target_Zn_count"] = int(target_counts["N_Zn_target"])
        zinc_summary["target_q1_q2b_ratio"] = None if site_mode != "q1_q2b_single_structure_mixture" else float(q1_q2b_ratio)
        zinc_summary["target_N_Q1_Zn"] = int(target_counts["N_Q1_target"])
        zinc_summary["target_N_Q2b_Zn"] = int(target_counts["N_Q2b_target"])
    stage_counts["atom_count_after_zinc"] = int(len(entries))
    stage_counts["bond_count_after_zinc"] = int(len(bonds))
    entries, _, _ = check_move_water_hydrogens(entries)
    stage_counts["atom_count_after_water_hydrogen_check"] = int(len(entries))
    entries, recenter_summary = recenter_framework_largest_gap(entries, supercell, enabled=bool(recenter))
    stage_counts["atom_count_after_recenter"] = int(len(entries))
    recenter_file = os.path.join(internal_dir, "periodic_recenter_summary.json")
    write_json(recenter_file, recenter_summary)
    occupancy_summary = compute_framework_occupancy_summary(entries, supercell, label="final_recentered", recenter_summary=recenter_summary)
    cell_summary = audit_triclinic_export_consistency(supercell, cell_export=supercell, entries=entries)
    dedup_summary = audit_deduplication(stage_counts)
    root_cause_category = classify_root_cause(occupancy_summary, cell_summary, brick_summary, dedup_summary)
    occupancy_file = os.path.join(internal_dir, "framework_occupancy_summary.json")
    cell_file = os.path.join(internal_dir, "cell_geometry_summary.json")
    brick_file = os.path.join(internal_dir, "brick_placement_summary.json")
    dedup_file = os.path.join(internal_dir, "dedup_audit_summary.json")
    write_json(occupancy_file, occupancy_summary)
    write_json(cell_file, cell_summary)
    write_json(brick_file, brick_summary)
    write_json(dedup_file, dedup_summary)
    data_file = os.path.join(internal_dir, model_id + ".data")
    water_summary = get_lammps_input_cementff(data_file, entries, bonds, angles, supercell, zinc_summary)
    mapping_file = os.path.join(internal_dir, model_id + "_cementff_mapping.json")
    write_cementff4_mapping_json(mapping_file, contains_zinc)
    water_file = os.path.join(internal_dir, model_id + "_water_summary.json")
    write_json(water_file, water_summary)
    zinc_file = None
    if zinc_summary is not None:
        zinc_file = os.path.join(internal_dir, model_id + "_zinc_summary.json")
        write_zinc_summary(zinc_file, zinc_summary)
        write_cementff4_zinc_input(os.path.join(internal_dir, "in.CementFF4_Zn"))
    comp = composition_summary(data_file, target_ca_si, target_w_si, target_zn_si, target_counts, site_mode, seed, n_si)
    comp_file = os.path.join(internal_dir, model_id + "_composition_summary.json")
    write_json(comp_file, comp)
    visual_clean_file = None
    visual_summary_file = None
    if visual_export_dir is not None:
        ensure_dir(visual_export_dir)
        visual_entries, visual_cell, visual_summary = compact_orthogonal_visual_entries(entries)
        visual_clean_file = os.path.join(visual_export_dir, model_id + ".visual_orthogonal.clean.data")
        write_clean_entries(visual_clean_file, visual_entries, bonds, angles, visual_cell)
        visual_summary_file = os.path.join(visual_export_dir, model_id + "_visual_orthogonal_summary.json")
        write_json(visual_summary_file, visual_summary)
    return {
        "data_file": data_file,
        "zinc_summary": zinc_file,
        "composition_summary": comp_file,
        "mapping_summary": mapping_file,
        "water_summary": water_file,
        "target_counts": target_counts,
        "recenter_summary": recenter_file,
        "recenter_applied": bool(recenter_summary.get("applied")),
        "framework_occupancy_summary": occupancy_file,
        "cell_geometry_summary": cell_file,
        "brick_placement_summary": brick_file,
        "dedup_audit_summary": dedup_file,
        "cell_audit_root_cause_category": root_cause_category,
        "visual_orthogonal_clean_data": visual_clean_file,
        "visual_orthogonal_summary": visual_summary_file,
        "audit_metrics": {
            "framework_frac_span_x": occupancy_summary.get("frac_span_x"),
            "framework_frac_span_y": occupancy_summary.get("frac_span_y"),
            "framework_frac_span_z": occupancy_summary.get("frac_span_z"),
            "occupancy_warning": occupancy_summary.get("occupancy_warning"),
            "expected_brick_count": brick_summary.get("expected_brick_count"),
            "actual_brick_count": brick_summary.get("actual_brick_count"),
            "supercell_population_warning": brick_summary.get("supercell_population_warning"),
            "export_consistency_passed": cell_summary.get("export_consistency_passed"),
            "dedup_warning": dedup_summary.get("dedup_warning"),
        },
    }


def coordination_values(validation):
    return [
        int(site.get("coordination_2p5", 0))
        for site in validation.get("zinc", {}).get("zinc_sites", []) or []
    ]


def coordination_quality(values):
    vals = [int(v) for v in values]
    if not vals:
        return "no_zinc"
    if any(v < 4 for v in vals):
        return "undercoordinated_failed"
    if all(v == 4 for v in vals):
        return "ideal_fourfold"
    if any(v > 4 for v in vals):
        return "overcoordinated"
    return "minimum_valid"


def failure_reason(validation, accepted, ideal_only, quality, exc=None):
    if exc is not None:
        return "{}: {}".format(type(exc).__name__, exc)
    label = validation.get("classification")
    if label not in VALID_LABELS:
        reasons = validation.get("reasons") or []
        return "validation failed: {}{}".format(label, "; " + "; ".join(reasons) if reasons else "")
    if ideal_only and quality != "ideal_fourfold":
        return "outside requested coordination quality window"
    if not accepted:
        return "not accepted"
    return ""


def actual_q1_q2b(zinc_summary_path):
    if not zinc_summary_path:
        return 0, 0, None
    zinc = read_json(zinc_summary_path)
    q1 = int(zinc.get("N_Q1_Zn", zinc.get("n_Q1_Zn", 0)) or 0)
    q2b = int(zinc.get("N_Q2b_Zn", zinc.get("n_Q2b_Zn", 0)) or 0)
    ratio = None if (q1 + q2b) == 0 else float(q1) / float(q1 + q2b)
    return q1, q2b, ratio


def export_clean_data(internal_data, clean_data):
    with open(internal_data) as src:
        lines = src.readlines()
    out = []
    skip = False
    for line in lines:
        if line.strip() == "CS-Info":
            skip = True
            continue
        if skip:
            continue
        out.append(line)
    with open(clean_data, "w") as dst:
        dst.writelines(out)


def write_clean_entries(path, entries, bonds, angles, cell):
    get_lammps_input_cementff(path, entries, bonds, angles, cell, zinc_summary=None, sanitize_water=False)
    export_clean_data(path, path)


def read_csinfo(data_file):
    from validate_cementff_data import parse_data

    return parse_data(data_file)["csinfo"]


def append_csinfo(reference_data, raw_data, output_data):
    csinfo = read_csinfo(reference_data)
    with open(raw_data) as f:
        text = f.read().rstrip()
    with open(output_data, "w") as f:
        f.write(text)
        f.write("\n\nCS-Info\n\n")
        for atom_id in sorted(csinfo):
            f.write("{:8d} {:8d}\n".format(atom_id, csinfo[atom_id]))
    return output_data


def run_lammps_input(lammps_command, input_file, run_dir, log_name):
    cmd = [lammps_command, "-in", os.path.basename(input_file), "-log", log_name]
    result = subprocess.call(cmd, cwd=run_dir)
    if result != 0:
        raise RuntimeError("LAMMPS command failed with exit code {}: {}".format(result, " ".join(cmd)))


def run_one_model(args_dict):
    model_index = int(args_dict["model_index"])
    model_id = "model_{:06d}".format(model_index)
    seed = int(args_dict["seed_start"]) + model_index - 1 if args_dict.get("seed") is None else int(args_dict["seed"]) + model_index - 1
    model_dir = os.path.join(args_dict["output_dir"], "structures", model_id)
    internal_dir = os.path.join(model_dir, "internal")
    lammps_dir = os.path.join(model_dir, "lammps")
    postmin_dir = os.path.join(model_dir, "postmin")
    try:
        generation = generate_structure(
            model_id,
            internal_dir,
            args_dict["site_mode"],
            seed,
            target_ca_si=args_dict["target_ca_si"],
            target_w_si=args_dict["target_w_si"],
            target_zn_si=args_dict.get("target_zn_si"),
            target_zn_count=args_dict.get("target_zn_count"),
            q1_q2b_ratio=args_dict["q1_q2b_ratio"],
            min_zn_zn_distance=args_dict["min_zn_zn_distance"],
            recenter=args_dict.get("recenter", True),
            visual_export_dir=lammps_dir if args_dict.get("export_clean_data") else None,
        )
        validation = validate(generation["data_file"], expected_zinc_site_type=None, zinc_summary_path=generation.get("zinc_summary"))
        validation_file = os.path.join(internal_dir, model_id + "_validation.json")
        write_json(validation_file, validation)
        composition = read_json(generation["composition_summary"])
        q1, q2b, actual_ratio = actual_q1_q2b(generation.get("zinc_summary"))
        audit_metrics = generation.get("audit_metrics", {})
        coords = coordination_values(validation)
        quality = coordination_quality(coords)
        validation_passed = validation.get("classification") in VALID_LABELS
        accepted = validation_passed and (quality == "ideal_fourfold" if args_dict["ideal_only"] else True)
        lammps_outputs = {}
        if args_dict["build_lammps_inputs"] or args_dict["run_static_relaxation"] or args_dict["run_quasistatic"]:
            ff_result = build_forcefield(lammps_dir)
            lammps_outputs = build_lammps_inputs(generation["data_file"], ff_result["forcefield"], lammps_dir, model_id)
        if args_dict["export_clean_data"]:
            ensure_dir(lammps_dir)
            clean_data = os.path.join(lammps_dir, model_id + ".clean.data")
            export_clean_data(generation["data_file"], clean_data)
            lammps_outputs["clean_data"] = clean_data
            if generation.get("visual_orthogonal_clean_data"):
                lammps_outputs["visual_orthogonal_clean_data"] = generation.get("visual_orthogonal_clean_data")
                lammps_outputs["visual_orthogonal_summary"] = generation.get("visual_orthogonal_summary")
        if args_dict["run_static_relaxation"]:
            if not lammps_outputs:
                raise ValueError("--run-static-relaxation requires LAMMPS inputs")
            ensure_dir(postmin_dir)
            run_lammps_input(args_dict["lammps_command"], lammps_outputs["minimize_static"], lammps_dir, "log.minimize_static")
            raw_postmin = os.path.join(lammps_dir, model_id + "_minimized_static.raw.data")
            postmin_raw_data_path = None
            postmin_internal_data_path = None
            postmin_validation_path = None
            postmin_validation_label = None
            postmin_validation_passed = False
            if os.path.exists(raw_postmin):
                postmin_raw_data_path = os.path.join(postmin_dir, model_id + "_minimized_static.raw.data")
                shutil.copy2(raw_postmin, postmin_raw_data_path)
                postmin_internal_data_path = os.path.join(postmin_dir, model_id + "_postmin_internal.data")
                append_csinfo(generation["data_file"], postmin_raw_data_path, postmin_internal_data_path)
                postmin_validation = validate(
                    postmin_internal_data_path,
                    expected_zinc_site_type=None,
                    zinc_summary_path=generation.get("zinc_summary"),
                )
                postmin_validation_path = os.path.join(postmin_dir, model_id + "_postmin_validation.json")
                write_json(postmin_validation_path, postmin_validation)
                postmin_validation_label = postmin_validation.get("classification")
                postmin_validation_passed = postmin_validation_label in VALID_LABELS
                lammps_outputs.update(
                    {
                        "postmin_raw_data_path": postmin_raw_data_path,
                        "postmin_internal_data_path": postmin_internal_data_path,
                        "postmin_validation_path": postmin_validation_path,
                        "postmin_validation_label": postmin_validation_label,
                        "postmin_validation_passed": postmin_validation_passed,
                    }
                )
        if args_dict["run_quasistatic"]:
            if not args_dict["run_static_relaxation"]:
                raise ValueError("--run-quasistatic requires --run-static-relaxation")
            run_lammps_input(args_dict["lammps_command"], lammps_outputs["elastic_x_plus"], lammps_dir, "log.elastic_x_plus")
            run_lammps_input(args_dict["lammps_command"], lammps_outputs["elastic_x_minus"], lammps_dir, "log.elastic_x_minus")
        row = dict(composition)
        row.update(
            {
                "model_id": model_id,
                "seed": seed,
                "site_mode": args_dict["site_mode"],
                "target_q1_q2b_ratio": args_dict["q1_q2b_ratio"] if args_dict["site_mode"] == "q1_q2b_single_structure_mixture" else None,
                "actual_q1_q2b_ratio": actual_ratio,
                "actual_N_Q1_Zn": q1,
                "actual_N_Q2b_Zn": q2b,
                "validation_label": validation.get("classification"),
                "validation_passed": bool(validation_passed),
                "accepted": bool(accepted),
                "coordination_quality": quality,
                "failure_reason": failure_reason(validation, accepted, args_dict["ideal_only"], quality),
                "data_file": generation["data_file"],
                "validation_json": validation_file,
                "zinc_summary": generation.get("zinc_summary"),
                "composition_summary": generation["composition_summary"],
                "recenter_applied": generation.get("recenter_applied"),
                "recenter_summary_path": generation.get("recenter_summary"),
                "framework_occupancy_summary": generation.get("framework_occupancy_summary"),
                "cell_geometry_summary": generation.get("cell_geometry_summary"),
                "brick_placement_summary": generation.get("brick_placement_summary"),
                "dedup_audit_summary": generation.get("dedup_audit_summary"),
                "framework_frac_span_x": audit_metrics.get("framework_frac_span_x"),
                "framework_frac_span_y": audit_metrics.get("framework_frac_span_y"),
                "framework_frac_span_z": audit_metrics.get("framework_frac_span_z"),
                "occupancy_warning": audit_metrics.get("occupancy_warning"),
                "expected_brick_count": audit_metrics.get("expected_brick_count"),
                "actual_brick_count": audit_metrics.get("actual_brick_count"),
                "supercell_population_warning": audit_metrics.get("supercell_population_warning"),
                "export_consistency_passed": audit_metrics.get("export_consistency_passed"),
                "dedup_warning": audit_metrics.get("dedup_warning"),
                "cell_audit_root_cause_category": generation.get("cell_audit_root_cause_category"),
                "lammps_dir": lammps_dir if lammps_outputs else None,
                "postmin_raw_data_path": lammps_outputs.get("postmin_raw_data_path"),
                "postmin_internal_data_path": lammps_outputs.get("postmin_internal_data_path"),
                "postmin_validation_path": lammps_outputs.get("postmin_validation_path"),
                "postmin_validation_label": lammps_outputs.get("postmin_validation_label"),
                "postmin_validation_passed": lammps_outputs.get("postmin_validation_passed"),
            }
        )
        write_json(os.path.join(model_dir, "model_manifest.json"), {"row": row, "generation": generation, "lammps_outputs": lammps_outputs})
        return row
    except Exception as exc:
        ensure_dir(internal_dir)
        failure = {
            "model_id": model_id,
            "seed": seed,
            "site_mode": args_dict["site_mode"],
            "target_Ca_Si": args_dict["target_ca_si"],
            "target_W_Si": args_dict["target_w_si"],
            "target_Zn_Si": args_dict.get("target_zn_si"),
            "target_Zn_count": args_dict.get("target_zn_count"),
            "target_q1_q2b_ratio": args_dict["q1_q2b_ratio"],
            "validation_label": "not_generated",
            "validation_passed": False,
            "accepted": False,
            "coordination_quality": "not_generated",
            "failure_reason": failure_reason({}, False, args_dict["ideal_only"], "not_generated", exc=exc),
            "data_file": None,
        }
        write_json(os.path.join(internal_dir, model_id + "_failure.json"), failure)
        return failure


def count_rows(rows, key):
    counts = Counter(str(row.get(key) or "none") for row in rows)
    return [{key: name, "count": counts[name]} for name in sorted(counts)]


def representative_rows(rows):
    accepted = [row for row in rows if str(row.get("accepted")).lower() == "true" or row.get("accepted") is True]
    quality_rank = {"ideal_fourfold": 0, "minimum_valid": 1, "overcoordinated": 2, "no_zinc": 3}
    accepted.sort(key=lambda row: (
        quality_rank.get(row.get("coordination_quality"), 9),
        abs(float(row.get("Ca_Si_error") or 0.0)),
        abs(float(row.get("Zn_Si_error") or 0.0)),
        int(row.get("seed") or 0),
    ))
    return {"best_overall": accepted[0]} if accepted else {}
