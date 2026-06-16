from __future__ import print_function

import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs
from mod_construct_brick_Y import get_all_bricks, pieces
from mod_construct_supercell_Y import check_move_water_hydrogens, get_angles, get_full_coordinates, resize_crystal
from mod_sample import fill_water, sample_Ca_Si_ratio
from mod_write_Y import get_lammps_input_cementff, write_cementff4_mapping_json, write_cementff4_zinc_input
from mod_zinc import (
    apply_charge_balance,
    apply_zinc_sites,
    build_multi_zinc_summary,
    build_zinc_candidate_site_report,
    inspect_zinc_candidates,
    remap_zinc_angles,
    select_multi_zinc_sites,
    total_charge,
    validate_no_zinc_bonds,
    write_zinc_summary,
)
from postprocess.analyze_structure import analyze as analyze_structure
from validate_cementff_data import parse_data, validate


UNITCELL = [
    [6.7352, 0.0, 0.0],
    [-4.071295, 6.209521, 0.0],
    [0.7037701, -6.2095578, 13.9936836],
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
    return {"returncode": proc.returncode, "stdout": out_path, "stderr": err_path, "ok": proc.returncode == 0}


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


def sample_parent(seed):
    np.random.seed(seed)
    random.seed(seed + 10)
    size = (2, 2, 2)
    supercell = np.zeros((3, 3))
    for i in range(3):
        supercell[i, :] = [UNITCELL[i][j] * size[i] for j in range(3)]
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
    return entries, bonds, crystal_dict, angles, supercell, n_ca / n_si, n_si


def select_multi_candidates(entries, bonds, crystal_dict, angles, supercell, mode, seed, n_q1, n_q2b, min_zn_zn_distance, max_attempts):
    candidates = inspect_zinc_candidates(crystal_dict)
    candidate_site_report = build_zinc_candidate_site_report(
        candidates,
        entries,
        bonds,
        angles,
        supercell,
        False,
        True,
        1.95,
    )
    selected, rejected, pools = select_multi_zinc_sites(
        candidates,
        candidate_site_report,
        entries,
        bonds,
        supercell,
        mode,
        seed,
        n_q1=n_q1,
        n_q2b=n_q2b,
        min_zn_zn_distance=min_zn_zn_distance,
        max_attempts=max_attempts,
    )
    return candidates, candidate_site_report, selected, rejected, pools


def apply_multi_motif(entries, crystal_dict, bonds, angles, supercell, selected_sites):
    entries = copy.deepcopy(entries)
    bonds = copy.deepcopy(bonds)
    angles = copy.deepcopy(angles)
    crystal_dict = copy.deepcopy(crystal_dict)
    entries, crystal_dict = apply_zinc_sites(entries, crystal_dict, selected_sites)
    hydroxylation_records = apply_charge_balance(
        entries,
        bonds,
        angles,
        selected_sites,
        supercell,
        "hydroxylate_two_oxygens",
        False,
        True,
        1.95,
    )
    return entries, crystal_dict, bonds, angles, hydroxylation_records


def mode_to_counts(mode, n_q1, n_q2b):
    if mode == "multi_q2b":
        return 0, 2 if n_q2b is None else int(n_q2b)
    if mode == "multi_q1":
        return 2 if n_q1 is None else int(n_q1), 0
    if mode == "q1_q2b_single_structure_mixture":
        return 1 if n_q1 is None else int(n_q1), 1 if n_q2b is None else int(n_q2b)
    raise ValueError("Unsupported mode {}".format(mode))


def zinc_validation_by_id(validation):
    by_id = {}
    for site in validation.get("zinc", {}).get("zinc_sites", []):
        by_id[str(int(site["Zn"]))] = site
    return by_id


def write_pre_post_compare(path, zinc_summary, validation_initial, validation_post):
    initial_by_id = zinc_validation_by_id(validation_initial)
    post_by_id = zinc_validation_by_id(validation_post)
    centers = []
    for center in zinc_summary.get("zn_centers", []):
        zn_id = str(int(center["atom_id"]))
        before = initial_by_id.get(zn_id, {})
        after = post_by_id.get(zn_id, {})
        centers.append({
            "zn_atom_id": int(zn_id),
            "motif_type": center.get("motif_type"),
            "coordination_initial_2p5A": before.get("coordination_2p5"),
            "coordination_postmin_2p5A": after.get("coordination_2p5"),
            "nearest_oxygen_initial": before.get("nearest_oxygen", []),
            "nearest_oxygen_postmin": after.get("nearest_oxygen", []),
            "center_passed_postmin": bool(after.get("coordination_2p5", 0) >= 4),
        })
    out = {
        "all_centers_passed_postmin": all(item["center_passed_postmin"] for item in centers),
        "centers": centers,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return out


def attach_postmin_to_summary(zinc_summary, validation_post):
    post_by_id = zinc_validation_by_id(validation_post)
    for center in zinc_summary.get("zn_centers", []):
        zn_id = str(int(center["atom_id"]))
        post = post_by_id.get(zn_id, {})
        center["postmin_Zn_O_distances"] = post.get("nearest_oxygen", [])
        center["postmin_Zn_O_coordination_2p5A"] = post.get("coordination_2p5")
        center["center_passed_postmin"] = bool(post.get("coordination_2p5", 0) >= 4)
    for site in zinc_summary.get("selected_sites", []):
        zn_id = str(int(site["atom_id"]))
        post = post_by_id.get(zn_id, {})
        site["postmin_Zn_O_distances"] = post.get("nearest_oxygen", [])
        site["postmin_Zn_O_coordination_2p5A"] = post.get("coordination_2p5")
        site["center_passed_postmin"] = bool(post.get("coordination_2p5", 0) >= 4)
    zinc_summary["postmin_diagnostics"] = [
        {
            "zn_atom_id": int(center["atom_id"]),
            "motif_type": center.get("motif_type"),
            "postmin_Zn_O_coordination_2p5A": center.get("postmin_Zn_O_coordination_2p5A"),
            "postmin_Zn_O_distances": center.get("postmin_Zn_O_distances", []),
            "center_passed_postmin": center.get("center_passed_postmin"),
        }
        for center in zinc_summary.get("zn_centers", [])
    ]
    return zinc_summary


def enrich_centers_with_hydroxylation(zinc_summary, hydroxylation_records):
    by_zn = {
        int(record["zn_atom_id"]): record.get("hydroxylated_oxygens", [])
        for record in hydroxylation_records
    }
    for center in zinc_summary.get("zn_centers", []):
        center["hydroxylated_O_pairs"] = by_zn.get(int(center["atom_id"]), [])
    for site in zinc_summary.get("selected_sites", []):
        site["hydroxylated_O_pairs"] = by_zn.get(int(site["atom_id"]), [])
    zinc_summary["N_Os_converted_to_Oh"] = sum(len(x) for x in by_zn.values())
    zinc_summary["N_H_added_for_Zn_OH"] = zinc_summary["N_Os_converted_to_Oh"]
    return zinc_summary


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def compact_selected_sites(selected_sites):
    return [
        {
            "atom_id": int(site["atom_id"]),
            "motif": site.get("motif"),
            "piece": site.get("piece"),
            "cell": site.get("cell"),
            "planned_hydroxylated_oxygen_ids": site.get("planned_hydroxylated_oxygen_ids", []),
        }
        for site in selected_sites
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate a single-structure multi-Zn C-S-H candidate.")
    parser.add_argument("--mode", choices=["multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-q1", type=int, default=None)
    parser.add_argument("--n-q2b", type=int, default=None)
    parser.add_argument("--min-zn-zn-distance", type=float, default=5.0)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--run-static-relaxation", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join("output_Y", "workflow_v1", "multi_zn_structure"))
    args = parser.parse_args()
    ensure_dir(args.out_dir)
    mode_dir = args.out_dir
    entries, bonds, crystal_dict, angles, supercell, ca_si_ratio, n_si = sample_parent(args.seed)
    n_q1, n_q2b = mode_to_counts(args.mode, args.n_q1, args.n_q2b)
    manifest = {
        "workflow": "v1.6-alpha-single-structure-multi-Zn",
        "mode": args.mode,
        "seed": int(args.seed),
        "n_q1": int(n_q1),
        "n_q2b": int(n_q2b),
        "min_zn_zn_distance": float(args.min_zn_zn_distance),
        "max_attempts": int(args.max_attempts),
        "run_static_relaxation": bool(args.run_static_relaxation),
        "finite_temperature_md": "not run",
        "single_structure_mixed_Q1_Q2b": args.mode == "q1_q2b_single_structure_mixture",
        "old_mixed_Q1_Q2b_Zn_site_type": False,
    }
    write_json(os.path.join(mode_dir, "generation_manifest.json"), manifest)
    try:
        candidates, candidate_site_report, selected_sites, rejected_candidates, pools = select_multi_candidates(
            entries,
            bonds,
            crystal_dict,
            angles,
            supercell,
            args.mode,
            args.seed,
            n_q1,
            n_q2b,
            args.min_zn_zn_distance,
            args.max_attempts,
        )
        entries_m, crystal_dict_m, bonds_m, angles_m, hydroxylation_records = apply_multi_motif(entries, crystal_dict, bonds, angles, supercell, selected_sites)
        topology = remap_zinc_angles(entries_m, angles_m)
        topology["zinc_bonds"] = validate_no_zinc_bonds(entries_m, bonds_m)
        zinc_summary = build_multi_zinc_summary(
            entries_m,
            selected_sites,
            candidates,
            candidate_site_report,
            rejected_candidates,
            0.05,
            ca_si_ratio,
            supercell,
            args.mode,
            args.seed,
            args.min_zn_zn_distance,
        )
        zinc_summary["hydroxylation_records"] = hydroxylation_records
        zinc_summary = enrich_centers_with_hydroxylation(zinc_summary, hydroxylation_records)
        zinc_summary["topology_validation"] = topology
        zinc_summary["total_charge_after_hydroxylation"] = total_charge(entries_m)
        zinc_summary["charge_residual_final"] = total_charge(entries_m)
        zinc_summary["total_charge_residual"] = total_charge(entries_m)
        zinc_summary["selection_summary"] = {
            "n_selected": len(selected_sites),
            "rejected_candidates": rejected_candidates,
        }
        zinc_summary["postmin_diagnostics"] = []
        data_path = os.path.join(mode_dir, "multi_zn_cementff_zn.data")
        water_summary = get_lammps_input_cementff(data_path, entries_m, bonds_m, angles_m, supercell, zinc_summary)
        mapping_path = os.path.join(mode_dir, "cementff_mapping_summary.json")
        write_cementff4_mapping_json(mapping_path, True)
        water_path = os.path.join(mode_dir, "water_summary.json")
        write_json(water_path, water_summary)
        zinc_path = os.path.join(mode_dir, "multi_zinc_summary.json")
        write_zinc_summary(zinc_path, zinc_summary)
        ff_path = os.path.join(mode_dir, "in.CementFF4_Zn")
        write_cementff4_zinc_input(ff_path)
        build_forcefield(mode_dir)
        build_inputs(data_path, ff_path, os.path.join(mode_dir, "lammps_inputs"), "multi_zn")
        validation_initial = validate(data_path, expected_zinc_site_type="multi_Zn", zinc_summary_path=zinc_path)
        write_json(os.path.join(mode_dir, "validation_initial.json"), validation_initial)
        report = {
            "ok": validation_initial["classification"].startswith("valid_multi_"),
            "classification": validation_initial["classification"],
            "selected_sites": compact_selected_sites(selected_sites),
            "rejected_candidates": rejected_candidates,
            "zinc_summary": zinc_path,
        }
        if args.run_static_relaxation and report["ok"]:
            lmp = find_lammps()
            if not lmp:
                report["ok"] = False
                report["failure_reason"] = "No LAMMPS executable found"
            else:
                input_dir = os.path.join(mode_dir, "lammps_inputs")
                steps = {}
                for input_name in ("in.read_check", "in.run0", "in.minimize_static"):
                    steps[input_name] = run_lammps(lmp, input_dir, input_name)
                    if not steps[input_name]["ok"]:
                        report["ok"] = False
                        report["failure_reason"] = "{} failed".format(input_name)
                        break
                if report["ok"]:
                    raw = os.path.join(input_dir, "multi_zn_minimized_static.raw.data")
                    final = os.path.join(input_dir, "multi_zn_minimized_static.data")
                    append_csinfo(data_path, raw, final)
                    validation_post = validate(final, expected_zinc_site_type="multi_Zn", zinc_summary_path=zinc_path)
                    zinc_summary = attach_postmin_to_summary(zinc_summary, validation_post)
                    write_zinc_summary(zinc_path, zinc_summary)
                    write_json(os.path.join(mode_dir, "validation_postmin.json"), validation_post)
                    compare = write_pre_post_compare(
                        os.path.join(mode_dir, "multi_zn_pre_post_coordination_compare.json"),
                        zinc_summary,
                        validation_initial,
                        validation_post,
                    )
                    try:
                        analyze_structure(final, os.path.join(mode_dir, "postprocess"))
                    except Exception as exc:
                        report["structure_analysis_warning"] = str(exc)
                    report["coordination_compare"] = compare
                    report["postmin_validation"] = validation_post["classification"]
                    report["ok"] = validation_post["classification"].startswith("valid_multi_")
        write_json(os.path.join(mode_dir, "generation_report.json"), report)
        print(json.dumps(report, indent=2, sort_keys=True))
    except Exception as exc:
        failure = {"ok": False, "error_type": type(exc).__name__, "error": str(exc), "mode": args.mode, "seed": args.seed}
        write_json(os.path.join(mode_dir, "generation_failure.json"), failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise


if __name__ == "__main__":
    main()
