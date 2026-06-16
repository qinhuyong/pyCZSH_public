from __future__ import print_function

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from examples.workflow_common import generate_structure
from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_lammps_inputs
from validate_cementff_data import validate


DEFAULT_CA_SI = "1.2,1.5,1.7"
DEFAULT_ZN_SI = "0,0.03,0.06"
DEFAULT_Q1_Q2B_RATIO = "1:1"
RATIO_SOURCE_NOTE = (
    "Atomic-Level Structure of Zinc-Modified Cementitious Calcium Silicate Hydrate reports "
    "that at (Zn/Si)i=0.15, Q(1,Zn) and Q(2p,Zn) each constitute 10% of all silicate "
    "species. This script maps Q(2p,Zn) to the pyCSH_Zn Q2b_Zn motif selector and uses "
    "Q1_Zn:Q2b_Zn = 1:1 by default."
)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_float_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def ratio_label(value):
    return ("{:.4g}".format(float(value))).replace("-", "m").replace(".", "p")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    fields = [
        "model_id",
        "contains_zn",
        "target_ca_si",
        "target_zn_si",
        "actual_ca_si_final",
        "actual_zn_si_final",
        "n_ca",
        "n_si",
        "n_zn",
        "n_q1_zn",
        "n_q2b_zn",
        "actual_q1_q2b_ratio",
        "requested_q1_q2b_ratio",
        "min_zn_zn_distance",
        "seed",
        "attempt",
        "site_mode",
        "validation_classification",
        "data_file",
        "forcefield",
        "lammps_dir",
        "read_check_input",
        "run0_input",
        "minimize_static_input",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_one(args, target_ca_si, target_zn_si, index):
    last_exc = None
    for attempt in range(int(args.max_attempts)):
        try:
            return build_one_attempt(args, target_ca_si, target_zn_si, index, attempt)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= int(args.max_attempts):
                break
            print(
                "Retrying Ca/Si={} Zn/Si={} with a new seed after {}: {}".format(
                    target_ca_si, target_zn_si, type(exc).__name__, exc
                )
            )
    raise last_exc


def build_one_attempt(args, target_ca_si, target_zn_si, index, attempt):
    contains_zn = float(target_zn_si) > 0.0
    model_id = "CaSi_{}_ZnSi_{}".format(ratio_label(target_ca_si), ratio_label(target_zn_si))
    model_dir = os.path.join(args.output_dir, model_id)
    prefix = "zn_csh" if contains_zn else "pure_csh"
    data_prefix = "{}_{}".format(prefix, model_id)
    seed = int(args.seed_start) + int(index) + int(attempt) * int(args.seed_stride)
    ensure_dir(model_dir)

    previous_ratio = os.environ.get("PYCSH_ZN_Q1_Q2B_RATIO")
    previous_min_distance = os.environ.get("PYCSH_ZN_MIN_ZN_ZN_DISTANCE")
    if contains_zn and args.site_mode == "q1_q2b_single_structure_mixture":
        os.environ["PYCSH_ZN_Q1_Q2B_RATIO"] = args.q1_q2b_ratio
    if contains_zn:
        os.environ["PYCSH_ZN_MIN_ZN_ZN_DISTANCE"] = str(args.min_zn_zn_distance)
    try:
        result = generate_structure(
            model_dir,
            data_prefix,
            enable_zinc=contains_zn,
            site_type=args.site_mode,
            seed=seed,
            target_ca_si=float(target_ca_si),
            target_w_si=float(args.target_w_si),
            ca_si_width=float(args.ca_si_tol),
            target_zn_si=float(target_zn_si) if contains_zn else None,
            ca_si_tol=float(args.ca_si_tol),
            zn_si_tol=float(args.zn_si_tol) if contains_zn else None,
        )
    finally:
        if previous_ratio is None:
            os.environ.pop("PYCSH_ZN_Q1_Q2B_RATIO", None)
        else:
            os.environ["PYCSH_ZN_Q1_Q2B_RATIO"] = previous_ratio
        if previous_min_distance is None:
            os.environ.pop("PYCSH_ZN_MIN_ZN_ZN_DISTANCE", None)
        else:
            os.environ["PYCSH_ZN_MIN_ZN_ZN_DISTANCE"] = previous_min_distance
    validation = validate(
        result["data_file"],
        expected_zinc_site_type=None,
        zinc_summary_path=result.get("zinc_summary"),
    )
    validation_path = os.path.join(model_dir, "validation.json")
    write_json(validation_path, validation)

    ff_result = build_forcefield(model_dir)
    lammps_dir = os.path.join(model_dir, "lammps_inputs")
    lammps_inputs = build_lammps_inputs(
        result["data_file"],
        ff_result["forcefield"],
        lammps_dir,
        data_prefix,
    )
    composition = read_json(result["composition_summary"])
    n_q1_zn = 0
    n_q2b_zn = 0
    if result.get("zinc_summary"):
        zinc_summary = read_json(result["zinc_summary"])
        n_q1_zn = int(zinc_summary.get("N_Q1_Zn", zinc_summary.get("n_Q1_Zn", 0)) or 0)
        n_q2b_zn = int(zinc_summary.get("N_Q2b_Zn", zinc_summary.get("n_Q2b_Zn", 0)) or 0)
    actual_q1_q2b_ratio = None
    if n_q1_zn or n_q2b_zn:
        actual_q1_q2b_ratio = "{}:{}".format(n_q1_zn, n_q2b_zn)
    row = {
        "model_id": model_id,
        "contains_zn": contains_zn,
        "target_ca_si": float(target_ca_si),
        "target_zn_si": float(target_zn_si),
        "actual_ca_si_final": composition.get("actual_Ca_Si_final"),
        "actual_zn_si_final": composition.get("actual_Zn_Si_final") if contains_zn else 0.0,
        "n_ca": composition.get("N_Ca"),
        "n_si": composition.get("N_Si_final"),
        "n_zn": composition.get("N_Zn"),
        "n_q1_zn": n_q1_zn,
        "n_q2b_zn": n_q2b_zn,
        "actual_q1_q2b_ratio": actual_q1_q2b_ratio,
        "requested_q1_q2b_ratio": args.q1_q2b_ratio if contains_zn else None,
        "min_zn_zn_distance": float(args.min_zn_zn_distance) if contains_zn else None,
        "seed": seed,
        "attempt": int(attempt) + 1,
        "site_mode": args.site_mode if contains_zn else "pure_csh",
        "validation_classification": validation.get("classification"),
        "data_file": result["data_file"],
        "forcefield": ff_result["forcefield"],
        "lammps_dir": lammps_dir,
        "read_check_input": lammps_inputs["read_check"],
        "run0_input": lammps_inputs["run0"],
        "minimize_static_input": lammps_inputs["minimize_static"],
    }
    write_json(os.path.join(model_dir, "batch_model_manifest.json"), {
        "generation": result,
        "forcefield": ff_result,
        "lammps_inputs": lammps_inputs,
        "composition": composition,
        "validation": validation_path,
        "summary_row": row,
    })
    return row


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate pure C-S-H and Zn-C-S-H LAMMPS inputs over Ca/Si and Zn/Si target ratios."
    )
    parser.add_argument("--ca-si", default=DEFAULT_CA_SI, help="Comma-separated target Ca/Si ratios.")
    parser.add_argument("--zn-si", default=DEFAULT_ZN_SI, help="Comma-separated target Zn/Si ratios; 0 means no Zn.")
    parser.add_argument("--ca-si-tol", type=float, default=0.15)
    parser.add_argument("--zn-si-tol", type=float, default=0.05)
    parser.add_argument("--target-w-si", type=float, default=0.2)
    parser.add_argument("--site-mode", choices=("multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture"), default="multi_q2b")
    parser.add_argument(
        "--q1-q2b-ratio",
        default=DEFAULT_Q1_Q2B_RATIO,
        help="Requested Q1_Zn:Q2b_Zn weights for q1_q2b_single_structure_mixture.",
    )
    parser.add_argument("--seed-start", type=int, default=26000)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--min-zn-zn-distance", type=float, default=5.0)
    parser.add_argument("--output-dir", default=os.path.join("output_Y", "workflow_v1", "ca_zn_ratio_lammps_matrix"))
    return parser


def main():
    args = build_parser().parse_args()
    ca_si_values = parse_float_list(args.ca_si)
    zn_si_values = parse_float_list(args.zn_si)
    ensure_dir(args.output_dir)
    rows = []
    index = 0
    for target_ca_si in ca_si_values:
        for target_zn_si in zn_si_values:
            index += 1
            rows.append(build_one(args, target_ca_si, target_zn_si, index))
    summary = {
        "ok": True,
        "n_models": len(rows),
        "target_ca_si_values": ca_si_values,
        "target_zn_si_values": zn_si_values,
        "site_mode": args.site_mode,
        "requested_q1_q2b_ratio": args.q1_q2b_ratio,
        "min_zn_zn_distance": float(args.min_zn_zn_distance),
        "q1_q2b_ratio_source_note": RATIO_SOURCE_NOTE,
        "output_dir": args.output_dir,
        "summary_csv": os.path.join(args.output_dir, "ca_zn_ratio_lammps_matrix_summary.csv"),
        "models": rows,
    }
    write_csv(summary["summary_csv"], rows)
    write_json(os.path.join(args.output_dir, "ca_zn_ratio_lammps_matrix_manifest.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
