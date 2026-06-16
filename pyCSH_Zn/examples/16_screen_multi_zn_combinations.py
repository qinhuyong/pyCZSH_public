from __future__ import print_function

import argparse
import csv
import importlib.util
import itertools
import json
import os
import shutil
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

GEN15_PATH = os.path.join(SCRIPT_DIR, "15_generate_multi_zn_structure.py")
spec = importlib.util.spec_from_file_location("multi_zn_alpha", GEN15_PATH)
alpha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alpha)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs
from mod_write_Y import get_lammps_input_cementff, write_cementff4_mapping_json, write_cementff4_zinc_input
from mod_zinc import (
    build_multi_candidate_pools,
    build_multi_zinc_summary,
    build_zinc_candidate_site_report,
    inspect_zinc_candidates,
    pairwise_zn_zn_distances,
    site_hydroxylated_oxygen_ids,
    total_charge,
    validate_no_zinc_bonds,
    remap_zinc_angles,
    write_zinc_summary,
)
from postprocess.analyze_structure import analyze as analyze_structure
from validate_cementff_data import validate


CSV_FIELDS = [
    "candidate_id",
    "mode",
    "seed",
    "selected_sites",
    "motif_types",
    "min_Zn_Zn_distance",
    "hydroxylated_O_pairs",
    "has_OH_conflict",
    "initial_validation_label",
    "postmin_validation_label",
    "initial_valid",
    "postmin_valid",
    "initial_ok",
    "postmin_ok",
    "per_center_coordination",
    "failed_center_count",
    "failed_center_ids",
    "overcoordinated_center_count",
    "coordination_quality",
    "failed_center_atom_id",
    "failure_reason",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def mode_counts(mode, n_q1, n_q2b):
    if mode == "multi_q2b":
        return 0, 2 if n_q2b is None else int(n_q2b)
    if mode == "multi_q1":
        return 2 if n_q1 is None else int(n_q1), 0
    if mode == "q1_q2b_single_structure_mixture":
        return 1 if n_q1 is None else int(n_q1), 1 if n_q2b is None else int(n_q2b)
    raise ValueError("Unsupported mode {}".format(mode))


def expected_label(mode):
    if mode == "multi_q2b":
        return "valid_multi_q2b_zn_candidate"
    if mode == "multi_q1":
        return "valid_multi_q1_zn_candidate"
    return "valid_multi_q1_q2b_zn_candidate"


def combination_iter(pools, mode, n_q1, n_q2b, max_combinations):
    q1_count, q2b_count = mode_counts(mode, n_q1, n_q2b)
    q1_pool = pools["Q1_Zn"][: max(max_combinations * 4, q1_count)]
    q2b_pool = pools["Q2b_Zn"][: max(max_combinations * 4, q2b_count)]
    if q1_count and q2b_count:
        iterator = itertools.product(itertools.combinations(q1_pool, q1_count), itertools.combinations(q2b_pool, q2b_count))
        for q1_items, q2b_items in iterator:
            yield [(dict(site), "Q1_Zn") for site in q1_items] + [(dict(site), "Q2b_Zn") for site in q2b_items]
    elif q1_count:
        for items in itertools.combinations(q1_pool, q1_count):
            yield [(dict(site), "Q1_Zn") for site in items]
    else:
        for items in itertools.combinations(q2b_pool, q2b_count):
            yield [(dict(site), "Q2b_Zn") for site in items]


def prepare_selected(raw_combo, entries, bonds, supercell, min_zn_zn_distance):
    selected = []
    used_si = set()
    used_oxy = set()
    rejected = []
    for site, motif in raw_combo:
        atom_id = int(site["atom_id"])
        if atom_id in used_si:
            return None, True, "duplicate substituted Si site"
        prepared = dict(site)
        prepared["motif"] = motif
        hydrox = site_hydroxylated_oxygen_ids(prepared, entries, bonds, supercell, False)
        overlap = sorted(used_oxy.intersection(hydrox))
        if overlap:
            rejected.append({"candidate_atom_id": atom_id, "motif": motif, "rejection_reason": "hydroxylated O conflict {}".format(overlap)})
            return None, True, "hydroxylated O conflict {}".format(overlap)
        selected.append(prepared)
        used_si.add(atom_id)
        used_oxy.update(hydrox)
        prepared["planned_hydroxylated_oxygen_ids"] = sorted(hydrox)
    distances = pairwise_zn_zn_distances(selected, supercell)
    min_distance = min([item["distance"] for item in distances], default=None)
    if min_distance is not None and min_distance < float(min_zn_zn_distance):
        return None, False, "Zn-Zn distance {:.3f} below {:.3f} A".format(min_distance, float(min_zn_zn_distance))
    return selected, False, None


def compact_sites(selected):
    return [
        {
            "atom_id": int(site["atom_id"]),
            "motif": site.get("motif"),
            "piece": site.get("piece"),
            "hydroxylated": site.get("planned_hydroxylated_oxygen_ids", []),
        }
        for site in selected
    ]


def failed_center_summary(validation):
    failed = validation.get("multi_zinc_failed_centers", [])
    if not failed:
        return None, ""
    return failed[0].get("zn_atom_id"), failed[0].get("reason")


def coordination_values_from_validation(validation):
    return [
        int(site.get("coordination_2p5", 0))
        for site in validation.get("zinc", {}).get("zinc_sites", [])
    ]


def coordination_quality(values):
    vals = [int(v) for v in values]
    if not vals:
        return "undercoordinated_failed"
    if any(v < 4 for v in vals):
        return "undercoordinated_failed"
    if all(v == 4 for v in vals):
        return "ideal_fourfold"
    if any(v > 4 for v in vals):
        return "overcoordinated"
    return "minimum_valid"


def failed_center_ids_from_validation(validation):
    return [
        str(item.get("zn_atom_id"))
        for item in validation.get("multi_zinc_failed_centers", [])
        if item.get("zn_atom_id") is not None
    ]


def run_candidate(candidate_dir, candidate_id, mode, seed, selected, parent, candidate_site_report, rejected_candidates, min_zn_zn_distance, run_static):
    ensure_dir(candidate_dir)
    entries, bonds, crystal_dict, angles, supercell, ca_si_ratio, n_si = parent
    entries_m, crystal_dict_m, bonds_m, angles_m, hydroxylation_records = alpha.apply_multi_motif(
        entries, crystal_dict, bonds, angles, supercell, selected
    )
    topology = remap_zinc_angles(entries_m, angles_m)
    topology["zinc_bonds"] = validate_no_zinc_bonds(entries_m, bonds_m)
    zinc_summary = build_multi_zinc_summary(
        entries_m,
        selected,
        {"Q1_Zn": [], "Q2b_Zn": []},
        candidate_site_report,
        rejected_candidates,
        0.05,
        ca_si_ratio,
        supercell,
        mode,
        seed,
        min_zn_zn_distance,
    )
    zinc_summary["hydroxylation_records"] = hydroxylation_records
    zinc_summary = alpha.enrich_centers_with_hydroxylation(zinc_summary, hydroxylation_records)
    zinc_summary["topology_validation"] = topology
    zinc_summary["charge_residual_final"] = total_charge(entries_m)
    zinc_summary["total_charge_residual"] = total_charge(entries_m)
    data_path = os.path.join(candidate_dir, "multi_zn_cementff_zn.data")
    water_summary = get_lammps_input_cementff(data_path, entries_m, bonds_m, angles_m, supercell, zinc_summary)
    write_json(os.path.join(candidate_dir, "water_summary.json"), water_summary)
    write_cementff4_mapping_json(os.path.join(candidate_dir, "cementff_mapping_summary.json"), True)
    zinc_path = os.path.join(candidate_dir, "multi_zinc_summary.json")
    write_zinc_summary(zinc_path, zinc_summary)
    ff_path = os.path.join(candidate_dir, "in.CementFF4_Zn")
    write_cementff4_zinc_input(ff_path)
    build_forcefield(candidate_dir)
    build_inputs(data_path, ff_path, os.path.join(candidate_dir, "lammps_inputs"), "multi_zn")
    validation_initial = validate(data_path, expected_zinc_site_type="multi_Zn", zinc_summary_path=zinc_path)
    write_json(os.path.join(candidate_dir, "validation_initial.json"), validation_initial)
    post_label = ""
    post_ok = False
    post_coordination = []
    failure_reason = ""
    failed_center = None
    compare = None
    if run_static and validation_initial["classification"].startswith("valid_multi_"):
        lmp = alpha.find_lammps()
        if not lmp:
            failure_reason = "No LAMMPS executable found"
        else:
            input_dir = os.path.join(candidate_dir, "lammps_inputs")
            for input_name in ("in.read_check", "in.run0", "in.minimize_static"):
                step = alpha.run_lammps(lmp, input_dir, input_name)
                if not step["ok"]:
                    failure_reason = "{} failed".format(input_name)
                    break
            if not failure_reason:
                raw = os.path.join(input_dir, "multi_zn_minimized_static.raw.data")
                final = os.path.join(input_dir, "multi_zn_minimized_static.data")
                alpha.append_csinfo(data_path, raw, final)
                validation_post = validate(final, expected_zinc_site_type="multi_Zn", zinc_summary_path=zinc_path)
                post_coordination = coordination_values_from_validation(validation_post)
                write_json(os.path.join(candidate_dir, "validation_postmin.json"), validation_post)
                post_label = validation_post["classification"]
                post_ok = post_label == expected_label(mode)
                failed_center, failure_reason = failed_center_summary(validation_post)
                zinc_summary = alpha.attach_postmin_to_summary(zinc_summary, validation_post)
                write_zinc_summary(zinc_path, zinc_summary)
                compare = alpha.write_pre_post_compare(
                    os.path.join(candidate_dir, "multi_zn_pre_post_coordination_compare.json"),
                    zinc_summary,
                    validation_initial,
                    validation_post,
                )
                try:
                    analyze_structure(final, os.path.join(candidate_dir, "postprocess"))
                    src = os.path.join(candidate_dir, "postprocess", "structure_analysis.json")
                    if os.path.exists(src):
                        shutil.copyfile(src, os.path.join(candidate_dir, "structure_analysis.json"))
                except Exception as exc:
                    failure_reason = failure_reason or "postprocess failed: {}".format(exc)
    elif run_static:
        failure_reason = "initial validation failed: {}".format(validation_initial["classification"])
    distances = pairwise_zn_zn_distances(selected, supercell)
    min_distance = min([item["distance"] for item in distances], default=None)
    initial_coordination = coordination_values_from_validation(validation_initial)
    final_coordination = post_coordination or initial_coordination
    failed_ids = failed_center_ids_from_validation(validation_post) if "validation_post" in locals() else []
    row = {
        "candidate_id": candidate_id,
        "mode": mode,
        "seed": seed,
        "selected_sites": ";".join(str(site["atom_id"]) for site in selected),
        "motif_types": ";".join(site["motif"] for site in selected),
        "min_Zn_Zn_distance": min_distance,
        "hydroxylated_O_pairs": ";".join(",".join(str(x) for x in site.get("planned_hydroxylated_oxygen_ids", [])) for site in selected),
        "has_OH_conflict": False,
        "initial_validation_label": validation_initial["classification"],
        "postmin_validation_label": post_label,
        "initial_valid": validation_initial["classification"].startswith("valid_multi_"),
        "postmin_valid": post_ok,
        "initial_ok": validation_initial["classification"].startswith("valid_multi_"),
        "postmin_ok": post_ok,
        "per_center_coordination": ";".join(str(x) for x in final_coordination),
        "failed_center_count": len(failed_ids),
        "failed_center_ids": ";".join(failed_ids),
        "overcoordinated_center_count": sum(1 for value in final_coordination if int(value) > 4),
        "coordination_quality": coordination_quality(final_coordination),
        "failed_center_atom_id": failed_center,
        "failure_reason": failure_reason,
    }
    write_json(os.path.join(candidate_dir, "candidate_result.json"), {"row": row, "coordination_compare": compare})
    return row


def screen_mode(args, mode):
    out_root = args.out_dir
    ensure_dir(out_root)
    parent = alpha.sample_parent(args.seed)
    entries, bonds, crystal_dict, angles, supercell, ca_si_ratio, n_si = parent
    candidates = inspect_zinc_candidates(crystal_dict)
    candidate_site_report = build_zinc_candidate_site_report(candidates, entries, bonds, angles, supercell, False, True, 1.95)
    pools = build_multi_candidate_pools(candidates, candidate_site_report, args.seed, entries, bonds, supercell)
    rows = []
    tried = 0
    skipped = []
    mode_dir = os.path.join(out_root, "candidates")
    ensure_dir(mode_dir)
    for raw_combo in combination_iter(pools, mode, args.n_q1, args.n_q2b, args.max_combinations * 10):
        selected, conflict, reason = prepare_selected(raw_combo, entries, bonds, supercell, args.min_zn_zn_distance)
        if selected is None:
            skipped.append({"mode": mode, "reason": reason, "has_OH_conflict": conflict})
            continue
        tried += 1
        candidate_id = "candidate_{:06d}_{}".format(tried, mode)
        candidate_dir = os.path.join(mode_dir, candidate_id)
        try:
            row = run_candidate(
                candidate_dir,
                candidate_id,
                mode,
                args.seed,
                selected,
                parent,
                candidate_site_report,
                skipped[-20:],
                args.min_zn_zn_distance,
                args.run_static_relaxation,
            )
        except Exception as exc:
            row = {
                "candidate_id": candidate_id,
                "mode": mode,
                "seed": args.seed,
                "initial_ok": False,
                "postmin_ok": False,
                "failure_reason": "{}: {}".format(type(exc).__name__, exc),
            }
            write_json(os.path.join(candidate_dir, "candidate_exception.json"), {"error": str(exc), "traceback": traceback.format_exc()})
        rows.append(row)
        if tried >= args.max_combinations:
            break
    valid = [row for row in rows if str(row.get("postmin_validation_label")) == expected_label(mode) and str(row.get("postmin_valid")).lower() == "true"]
    quality_rank = {"ideal_fourfold": 0, "minimum_valid": 1, "overcoordinated": 2, "undercoordinated_failed": 3}
    valid.sort(key=lambda row: (quality_rank.get(row.get("coordination_quality"), 9), float(row.get("min_Zn_Zn_distance") or 0.0) * -1.0))
    best = valid[0] if valid else None
    best_name = {
        "multi_q2b": "best_multi_q2b_candidate.json",
        "multi_q1": "best_multi_q1_candidate.json",
        "q1_q2b_single_structure_mixture": "best_q1_q2b_mixed_candidate.json",
    }[mode]
    write_json(os.path.join(out_root, best_name), {
        "mode": mode,
        "found_postmin_valid_candidate": bool(best),
        "best_candidate": best,
        "status": "postmin_valid_candidate_found" if best else "no_postmin_valid_candidate",
    })
    return rows, skipped, best


def main():
    parser = argparse.ArgumentParser(description="Screen single-structure multi-Zn site combinations.")
    parser.add_argument("--mode", choices=["multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture", "all"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-q1", type=int, default=None)
    parser.add_argument("--n-q2b", type=int, default=None)
    parser.add_argument("--max-combinations", type=int, default=10)
    parser.add_argument("--min-zn-zn-distance", type=float, default=5.0)
    parser.add_argument("--run-static-relaxation", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join("output_Y", "workflow_v1", "multi_zn_screening"))
    args = parser.parse_args()
    ensure_dir(args.out_dir)
    modes = ["multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture"] if args.mode == "all" else [args.mode]
    all_rows = []
    best_by_mode = {}
    skipped_by_mode = {}
    for mode in modes:
        rows, skipped, best = screen_mode(args, mode)
        all_rows.extend(rows)
        best_by_mode[mode] = best
        skipped_by_mode[mode] = skipped
    write_csv(os.path.join(args.out_dir, "multi_zn_screening_summary.csv"), all_rows)
    summary = {
        "ok": True,
        "mode": args.mode,
        "n_screened": len(all_rows),
        "n_postmin_valid": sum(1 for row in all_rows if row.get("postmin_ok") in (True, "True", "true")),
        "best_by_mode": best_by_mode,
        "skipped_by_mode": skipped_by_mode,
        "status_by_mode": {
            mode: ("postmin_valid_candidate_found" if best_by_mode.get(mode) else "no_postmin_valid_candidate")
            for mode in modes
        },
        "finite_temperature_md": "not run",
    }
    write_json(os.path.join(args.out_dir, "multi_zn_screening_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
