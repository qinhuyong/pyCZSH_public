from __future__ import print_function

import copy
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs
from mod_construct_brick_Y import get_all_bricks, pieces
from mod_construct_supercell_Y import check_move_water_hydrogens, get_angles, get_full_coordinates, resize_crystal
from mod_sample import fill_water, sample_Ca_Si_ratio
from mod_write_Y import get_lammps_input_cementff, write_cementff4_mapping_json
from mod_zinc import (
    ZN_CHARGE,
    ZN_SPECIE,
    apply_charge_balance,
    apply_zinc_sites,
    attach_q1_scores_to_candidates,
    build_zinc_candidate_site_report,
    build_zinc_summary,
    finalize_zinc_summary,
    inspect_zinc_candidates,
    q1_nearest_oxygen_records,
    q1_static_rank_tuple,
    validate_no_zinc_bonds,
    vector_pbc,
    write_zinc_summary,
)
from postprocess.analyze_structure import analyze as analyze_structure
from q1_diagnostics import compare_pre_post
from validate_cementff_data import parse_data, validate


UNITCELL = np.array(
    [
        [6.7352, 0.0, 0.0],
        [-4.071295, 6.209521, 0.0],
        [0.7037701, -6.2095578, 13.9936836],
    ]
)

CSV_FIELDS = [
    "candidate_id",
    "variant",
    "site_atom_id",
    "piece",
    "score",
    "initial_classification",
    "postmin_classification",
    "candidate_classification",
    "postmin_valid",
    "intended_four_still_closest",
    "pre_mean_zn_o",
    "pre_max_zn_o",
    "pre_mean_tetrahedral_deviation",
    "pre_max_tetrahedral_deviation",
    "pre_min_o_o",
    "post_mean_zn_o",
    "post_max_zn_o",
    "post_coordination_2p5",
    "clash_score",
    "reposition_enabled",
    "reposition_displacement",
    "candidate_dir",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def find_lammps():
    env = os.environ.get("LAMMPS_EXE")
    candidates = [env] if env else []
    candidates += ["lmp", "lmp_serial", "lmp_mpi", "lammps"]
    for candidate in candidates:
        if not candidate:
            continue
        path = shutil.which(candidate) or (candidate if os.path.exists(candidate) else None)
        if path:
            return path
    return None


def run_lammps(lmp, input_dir, input_name):
    out_path = os.path.join(input_dir, input_name + ".stdout.txt")
    err_path = os.path.join(input_dir, input_name + ".stderr.txt")
    proc = subprocess.run(
        [lmp, "-in", input_name],
        cwd=input_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    with open(out_path, "w") as f:
        f.write(proc.stdout)
    with open(err_path, "w") as f:
        f.write(proc.stderr)
    return {
        "input": os.path.join(input_dir, input_name),
        "returncode": proc.returncode,
        "stdout": out_path,
        "stderr": err_path,
        "ok": proc.returncode == 0,
    }


def append_csinfo(reference_data, raw_data, output_data):
    csinfo = parse_data(reference_data)["csinfo"]
    with open(raw_data) as f:
        text = f.read().rstrip()
    with open(output_data, "w") as f:
        f.write(text)
        f.write("\n\nCS-Info\n\n")
        for atom_id in sorted(csinfo):
            f.write("{:8d} {:8d}\n".format(atom_id, csinfo[atom_id]))
    return output_data


def generate_base(seed):
    np.random.seed(seed)
    random.seed(seed + 10)
    size = (2, 2, 2)
    supercell = np.zeros((3, 3))
    for i in range(3):
        supercell[i, :] = UNITCELL[i, :] * size[i]
    bricks, sorted_bricks = get_all_bricks(pieces)
    crystal, n_ca, n_si, r_sioh, r_caoh, mcl, n_water, _ = sample_Ca_Si_ratio(
        sorted_bricks, 1.7, 0.2, size[0] * size[1] * size[2], [0.06, 0.08, 0.08]
    )
    water_in_crystal = fill_water(crystal, n_water)
    crystal_rs, water_in_crystal_rs = resize_crystal(crystal, water_in_crystal, size)
    entries, bonds, crystal_dict, water_dict = get_full_coordinates(
        crystal_rs, water_in_crystal_rs, size, pieces, False, [0]
    )
    angles = get_angles(crystal_dict, water_dict, size)
    return {
        "entries": entries,
        "bonds": bonds,
        "angles": angles,
        "crystal_dict": crystal_dict,
        "supercell": supercell,
        "ca_si_ratio": n_ca / float(n_si),
        "n_si": n_si,
    }


def find_entry(entries, atom_id):
    for entry in entries:
        if int(entry[0]) == int(atom_id):
            return entry
    return None


def centroid_reposition(entries, site, intended_ids, supercell, fraction=1.0):
    zn_entry = find_entry(entries, site["atom_id"])
    if zn_entry is None or len(intended_ids) < 4:
        return {"enabled": True, "applied": False, "reason": "missing Zn entry or fewer than four intended O atoms"}
    zn_coord = np.array(zn_entry[3:], dtype=float)
    vectors = []
    for oid in intended_ids[:4]:
        o_entry = find_entry(entries, oid)
        if o_entry is not None:
            vectors.append(vector_pbc(zn_coord, o_entry[3:], supercell))
    if len(vectors) < 4:
        return {"enabled": True, "applied": False, "reason": "fewer than four intended O coordinates found"}
    centroid = zn_coord + sum(vectors) / float(len(vectors))
    new_coord = zn_coord + (centroid - zn_coord) * float(fraction)
    displacement = new_coord - zn_coord
    zn_entry[3] = float(new_coord[0])
    zn_entry[4] = float(new_coord[1])
    zn_entry[5] = float(new_coord[2])
    site["coord"] = [float(new_coord[0]), float(new_coord[1]), float(new_coord[2])]
    return {
        "enabled": True,
        "applied": True,
        "method": "centroid_of_four_intended_motif_O_atoms",
        "fraction": float(fraction),
        "old_coord": [float(x) for x in zn_coord],
        "new_coord": [float(x) for x in new_coord],
        "displacement_vector": [float(x) for x in displacement],
        "displacement_magnitude": float(np.linalg.norm(displacement)),
    }


def geometry_summary(report):
    geom = report.get("q1_geometry_diagnostics", {}) or {}
    dist = geom.get("Zn_O_distances_A", {}) or {}
    dev = geom.get("tetrahedral_angle_deviation_deg", {}) or {}
    return {
        "pre_mean_zn_o": dist.get("mean"),
        "pre_max_zn_o": dist.get("max"),
        "pre_mean_tetrahedral_deviation": dev.get("mean"),
        "pre_max_tetrahedral_deviation": dev.get("max"),
        "pre_min_o_o": geom.get("minimum_O_O_separation_A"),
    }


def clash_score(report):
    score = 0.0
    for record in report.get("q1_hydroxylation_records", []):
        for oxy in record.get("hydroxylated_oxygens", []):
            for metric in oxy.get("H_contact_metrics", {}).values():
                distance = metric.get("distance")
                if distance is not None:
                    score += max(0.0, 1.5 - float(distance))
            for metric in oxy.get("converted_O_contact_metrics", {}).values():
                distance = metric.get("distance")
                if distance is not None:
                    score += max(0.0, 1.5 - float(distance))
    return float(score)


def summarize_post_geometry(compare):
    geom = compare.get("post_minimization_motif_geometry", {})
    distances = geom.get("intended_Zn_O_distances_A", [])
    vals = [float(item["distance"]) for item in distances if item.get("distance") is not None]
    return {
        "post_mean_zn_o": None if not vals else float(sum(vals) / len(vals)),
        "post_max_zn_o": None if not vals else float(max(vals)),
    }


def intended_four_still_closest(compare):
    context = compare.get("context") or {}
    intended = set(int(x) for x in context.get("intended_oxygen_ids", []))
    after = set(int(item["atom_id"]) for item in compare.get("nearest_four_after_minimization", []))
    return bool(intended and intended == after)


def post_coordination_2p5(validation):
    sites = validation.get("zinc", {}).get("zinc_sites", [])
    if not sites:
        return None
    return sites[0].get("coordination_2p5")


def classify_candidate(initial_validation, post_validation):
    if post_validation is None:
        return "postmin_not_run"
    if post_validation["classification"] == "valid_q1_zn_candidate":
        return "postmin_valid"
    return "postmin_failed"


def rank_tuple(row):
    return (
        0 if row.get("postmin_valid") else 1,
        0 if row.get("intended_four_still_closest") else 1,
        row.get("post_max_zn_o") if row.get("post_max_zn_o") is not None else 999.0,
        row.get("pre_mean_tetrahedral_deviation") if row.get("pre_mean_tetrahedral_deviation") is not None else 999.0,
        row.get("clash_score") if row.get("clash_score") is not None else 999.0,
    )


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def build_candidate(base, candidates, candidate_report, candidate_record, candidate_id, variant, out_dir, lmp):
    candidate_dir = os.path.join(out_dir, "candidates", "candidate_{:04d}_{}".format(candidate_id, variant))
    ensure_dir(candidate_dir)
    entries = copy.deepcopy(base["entries"])
    bonds = copy.deepcopy(base["bonds"])
    angles = copy.deepcopy(base["angles"])
    crystal_dict = copy.deepcopy(base["crystal_dict"])
    site = copy.deepcopy(candidate_record["site"])
    site["q1_selection_score"] = float(candidate_record["report"].get("selection_score", -1.0e9))
    site["q1_selection_report"] = candidate_record["report"]
    site["q1_passed_preconditions"] = True

    apply_zinc_sites(entries, crystal_dict, [site])
    hydroxylation_records = apply_charge_balance(
        entries, bonds, angles, [site], base["supercell"], "hydroxylate_two_oxygens", False, True, 1.95
    )
    intended_ids = [
        int(item["atom_id"])
        for item in candidate_record["report"].get("pre_minimization_nearest_four_zn_o_atoms", [])
    ]
    reposition_record = {"enabled": False, "applied": False}
    if variant == "repositioned":
        reposition_record = centroid_reposition(entries, site, intended_ids, base["supercell"])
    summary = build_zinc_summary(
        entries,
        [site],
        candidates,
        1.0 / float(base["n_si"]),
        base["ca_si_ratio"],
        base["supercell"],
        "Q1_Zn",
        23743,
    )
    summary["total_charge_before_zinc"] = None
    summary["total_charge_after_zinc_before_hydroxylation"] = None
    summary["total_charge_after_hydroxylation"] = sum(float(entry[2]) for entry in entries)
    summary["total_charge_residual"] = sum(float(entry[2]) for entry in entries)
    summary["charge_residual_final"] = sum(float(entry[2]) for entry in entries)
    summary["N_Os_converted_to_Oh"] = sum(len(record["hydroxylated_oxygens"]) for record in hydroxylation_records)
    summary["N_H_added_for_Zn_OH"] = summary["N_Os_converted_to_Oh"]
    summary["hydroxylation_records"] = hydroxylation_records
    summary["allow_hydroxylate_bridging_oxygen"] = False
    summary["precondition_zinc_geometry"] = True
    summary["target_Zn_O_distance"] = 1.95
    summary["candidate_site_report"] = candidate_report
    summary["q1_screening_candidate_id"] = candidate_id
    summary["q1_screening_variant"] = variant
    summary["q1_reposition_zn"] = reposition_record
    summary["Q1_Zn_motif_assumption"] = (
        "v1.3.2 screening candidate: one topology-valid Q1/terminal silicate Si is replaced by Zn(+2); "
        "two safe terminal/non-bridging O core-shell pairs are converted to Oh-Hoh; no Zn-O bonds are added."
    )
    summary = finalize_zinc_summary(entries, bonds, angles, base["supercell"], summary, "hydroxylate_two_oxygens", False)
    entries, _, _ = check_move_water_hydrogens(entries)

    data_path = os.path.join(candidate_dir, "q1_zn_candidate.data")
    water_summary = get_lammps_input_cementff(data_path, entries, bonds, angles, base["supercell"], summary)
    water_path = os.path.join(candidate_dir, "water_summary.json")
    with open(water_path, "w") as f:
        json.dump(water_summary, f, indent=2, sort_keys=True)
        f.write("\n")
    zinc_path = os.path.join(candidate_dir, "zinc_summary.json")
    write_zinc_summary(zinc_path, summary)
    write_cementff4_mapping_json(os.path.join(candidate_dir, "cementff_mapping_summary.json"), True)
    ff = build_forcefield(candidate_dir)
    inputs = build_inputs(data_path, ff["forcefield"], os.path.join(candidate_dir, "lammps_inputs"), "q1_zn")
    initial_validation = validate(data_path, expected_zinc_site_type="Q1_Zn", zinc_summary_path=zinc_path)
    initial_validation_path = os.path.join(candidate_dir, "validation_initial.json")
    with open(initial_validation_path, "w") as f:
        json.dump(initial_validation, f, indent=2, sort_keys=True)
        f.write("\n")

    post_validation = None
    compare = {}
    steps = {}
    post_data = None
    if lmp:
        for input_name in ("in.read_check", "in.run0", "in.minimize_static"):
            steps[input_name] = run_lammps(lmp, os.path.join(candidate_dir, "lammps_inputs"), input_name)
            if not steps[input_name]["ok"]:
                break
        raw = os.path.join(candidate_dir, "lammps_inputs", "q1_zn_minimized_static.raw.data")
        post_data = os.path.join(candidate_dir, "q1_zn_postmin.data")
        if all(step.get("ok") for step in steps.values()) and os.path.exists(raw):
            append_csinfo(data_path, raw, post_data)
            post_validation = validate(post_data, expected_zinc_site_type="Q1_Zn", zinc_summary_path=zinc_path)
            post_validation_path = os.path.join(candidate_dir, "validation_postmin.json")
            with open(post_validation_path, "w") as f:
                json.dump(post_validation, f, indent=2, sort_keys=True)
                f.write("\n")
            compare_path = os.path.join(candidate_dir, "q1_zn_pre_post_coordination_compare.json")
            compare = compare_pre_post(data_path, post_data, zinc_path, compare_path)
            try:
                analyze_structure(post_data, candidate_dir)
            except Exception as exc:
                compare["postprocess_error"] = "{}: {}".format(type(exc).__name__, exc)

    geo = geometry_summary(candidate_record["report"])
    post_geo = summarize_post_geometry(compare)
    row = {
        "candidate_id": candidate_id,
        "variant": variant,
        "site_atom_id": int(site["atom_id"]),
        "piece": site["piece"],
        "score": float(candidate_record["report"].get("selection_score", -1.0e9)),
        "initial_classification": initial_validation["classification"],
        "postmin_classification": None if post_validation is None else post_validation["classification"],
        "candidate_classification": classify_candidate(initial_validation, post_validation),
        "postmin_valid": bool(post_validation and post_validation["classification"] == "valid_q1_zn_candidate"),
        "intended_four_still_closest": intended_four_still_closest(compare),
        "pre_mean_zn_o": geo["pre_mean_zn_o"],
        "pre_max_zn_o": geo["pre_max_zn_o"],
        "pre_mean_tetrahedral_deviation": geo["pre_mean_tetrahedral_deviation"],
        "pre_max_tetrahedral_deviation": geo["pre_max_tetrahedral_deviation"],
        "pre_min_o_o": geo["pre_min_o_o"],
        "post_mean_zn_o": post_geo["post_mean_zn_o"],
        "post_max_zn_o": post_geo["post_max_zn_o"],
        "post_coordination_2p5": None if post_validation is None else post_coordination_2p5(post_validation),
        "clash_score": clash_score(candidate_record["report"]),
        "reposition_enabled": bool(variant == "repositioned"),
        "reposition_displacement": reposition_record.get("displacement_magnitude"),
        "candidate_dir": candidate_dir,
    }
    record = {
        "row": row,
        "candidate_dir": candidate_dir,
        "data": data_path,
        "zinc_summary": zinc_path,
        "water_summary": water_path,
        "forcefield": ff,
        "inputs": inputs,
        "validation_initial": initial_validation_path,
        "steps": steps,
        "post_data": post_data,
        "pre_post_compare": os.path.join(candidate_dir, "q1_zn_pre_post_coordination_compare.json") if compare else None,
    }
    return record


def main():
    seed = int(os.environ.get("PYCSH_ZN_Q1_SCREEN_SEED", "23743"))
    top_n = int(os.environ.get("PYCSH_ZN_Q1_SCREEN_TOP_N", "6"))
    include_repositioned = os.environ.get("PYCSH_ZN_Q1_REPOSITION_ZN") == "1"
    out_dir = os.path.join("output_Y", "workflow_v1", "q1_motif_screening")
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "candidates"))
    base = generate_base(seed)
    candidates = inspect_zinc_candidates(base["crystal_dict"])
    candidate_report = build_zinc_candidate_site_report(
        candidates, base["entries"], base["bonds"], base["angles"], base["supercell"], False, True, 1.95
    )
    candidates = attach_q1_scores_to_candidates(candidates, candidate_report)
    valid_records = []
    report_by_atom = {int(item["candidate_atom_id"]): item for item in candidate_report.get("Q1_Zn", [])}
    for site in candidates.get("Q1_Zn", []):
        report = report_by_atom.get(int(site["atom_id"]))
        if not report or not report.get("passed_preconditions"):
            continue
        valid_records.append({"site": site, "report": report})
    valid_records.sort(key=lambda item: q1_static_rank_tuple(item["site"], seed))
    selected = valid_records[:top_n]
    lmp = find_lammps()
    results = []
    rows = []
    candidate_counter = 1
    for candidate in selected:
        variants = ["base"]
        if include_repositioned:
            variants.append("repositioned")
        for variant in variants:
            result = build_candidate(base, candidates, candidate_report, candidate, candidate_counter, variant, out_dir, lmp)
            results.append(result)
            rows.append(result["row"])
            candidate_counter += 1
    ranked = sorted(rows, key=rank_tuple)
    summary_csv = os.path.join(out_dir, "q1_motif_screen_summary.csv")
    summary_json = os.path.join(out_dir, "q1_motif_screen_summary.json")
    best_json = os.path.join(out_dir, "best_candidate.json")
    manifest_json = os.path.join(out_dir, "screening_manifest.json")
    write_csv(summary_csv, rows)
    summary = {
        "workflow": "v1.3.2-Q1-motif-screening",
        "scope": "controlled Q1_Zn static motif screening; no force-field or validation-threshold changes",
        "seed": seed,
        "top_n": top_n,
        "reposition_variants_enabled": include_repositioned,
        "lammps_executable": lmp,
        "n_q1_candidates_total": len(candidates.get("Q1_Zn", [])),
        "n_q1_candidates_topology_valid": len(valid_records),
        "n_screened": len(rows),
        "any_postmin_valid": any(row["postmin_valid"] for row in rows),
        "rows": rows,
        "ranked_rows": ranked,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    best = ranked[0] if ranked else None
    with open(best_json, "w") as f:
        json.dump(
            {
                "selection_policy": "postmin_valid first, then retained intended four-O shell, post-min max Zn-O, pre-min tetrahedral deviation, clash score",
                "best_candidate": best,
                "note": None if (best and best.get("postmin_valid")) else "No screened Q1_Zn motif passed post-min validation.",
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    with open(manifest_json, "w") as f:
        json.dump(
            {
                "workflow": "v1.3.2-Q1-motif-screening",
                "finite_temperature_md": "not run",
                "q1_added_to_default_mechanics": False,
                "forcefield_parameters_changed": False,
                "validation_semantics_changed": False,
                "outputs": {
                    "summary_csv": summary_csv,
                    "summary_json": summary_json,
                    "best_candidate": best_json,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(json.dumps({"ok": True, "any_postmin_valid": summary["any_postmin_valid"], "summary": summary_json}, indent=2))


if __name__ == "__main__":
    main()
