from __future__ import print_function

import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from validate_cementff_data import validate
from workflow_common import generate_structure


SITE_MODE_TO_GENERATOR = {
    "q2b_only": "Q2b_Zn",
    "q1_only": "Q1_Zn",
    "multi_q2b": "multi_q2b",
    "multi_q1": "multi_q1",
    "q1_q2b_single_structure_mixture": "q1_q2b_single_structure_mixture",
}

VALID_LABELS = {
    "valid_q2b_zn_candidate",
    "valid_q1_zn_candidate",
    "valid_multi_q2b_zn_candidate",
    "valid_multi_q1_zn_candidate",
    "valid_multi_q1_q2b_zn_candidate",
    "needs_static_relaxation",
}

CSV_FIELDS = [
    "model_id",
    "seed",
    "site_mode",
    "target_Ca_Si",
    "actual_Ca_Si_final",
    "Ca_Si_error",
    "within_Ca_Si_tolerance",
    "target_Zn_Si",
    "actual_Zn_Si_final",
    "Zn_Si_error",
    "within_Zn_Si_tolerance",
    "target_Zn_count",
    "actual_Zn_count",
    "N_Ca",
    "N_Si_parent_before_zn",
    "N_Si_final",
    "N_Zn",
    "composition_match",
    "validation_label",
    "validation_passed",
    "accepted_for_composition_target",
    "coordination_quality",
    "per_center_coordination_2p5A",
    "failed_center_count",
    "failure_reason",
    "data_file",
    "model_dir",
    "zinc_summary",
    "validation_json",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path, rows, fields=CSV_FIELDS):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def read_json(path):
    with open(path) as f:
        return json.load(f)


def boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


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


def failure_reason(validation, composition_match, quality, ideal_only, exc=None):
    if exc is not None:
        return "{}: {}".format(type(exc).__name__, exc)
    label = validation.get("classification")
    if label not in VALID_LABELS:
        reasons = validation.get("reasons") or []
        return "validation failed: {}{}".format(label, "; " + "; ".join(reasons) if reasons else "")
    if not composition_match:
        return "outside requested composition window"
    if ideal_only and quality != "ideal_fourfold":
        return "outside requested coordination quality window"
    return ""


def mode_target_count(args, site_mode):
    if args.target_zn_count is not None:
        return int(args.target_zn_count)
    return None


def run_one(args, index, seed):
    model_id = "model_{:06d}".format(index)
    model_dir = os.path.join(args.output_dir, "models", model_id)
    ensure_dir(model_dir)
    generator_site = SITE_MODE_TO_GENERATOR[args.site_mode]
    prefix = args.site_mode
    try:
        result = generate_structure(
            model_dir,
            prefix,
            enable_zinc=True,
            site_type=generator_site,
            seed=seed,
            target_ca_si=args.target_ca_si,
            target_w_si=0.2,
            ca_si_width=args.ca_si_tol,
            target_zn_si=args.target_zn_si,
            target_zn_count=mode_target_count(args, args.site_mode),
            ca_si_tol=args.ca_si_tol,
            zn_si_tol=args.zn_si_tol,
        )
        validation = validate(result["data_file"], expected_zinc_site_type=None, zinc_summary_path=result["zinc_summary"])
        validation_path = os.path.join(model_dir, "validation.json")
        write_json(validation_path, validation)
        composition = read_json(result["composition_summary"])
        coords = coordination_values(validation)
        quality = coordination_quality(coords)
        composition_match = boolish(composition.get("within_Ca_Si_tolerance")) and boolish(composition.get("within_Zn_Si_tolerance"))
        validation_passed = validation.get("classification") in VALID_LABELS
        quality_passed = (quality == "ideal_fourfold") if args.ideal_only else True
        accepted = validation_passed and composition_match and quality_passed
        failed_centers = validation.get("multi_zinc_failed_centers", []) or []
        row = dict(composition)
        row.update({
            "model_id": model_id,
            "seed": int(seed),
            "site_mode": args.site_mode,
            "composition_match": bool(composition_match),
            "validation_label": validation.get("classification"),
            "validation_passed": bool(validation_passed),
            "accepted_for_composition_target": bool(accepted),
            "coordination_quality": quality,
            "per_center_coordination_2p5A": ";".join(str(v) for v in coords),
            "failed_center_count": len(failed_centers),
            "failure_reason": failure_reason(validation, composition_match, quality, args.ideal_only),
            "data_file": result["data_file"],
            "model_dir": model_dir,
            "zinc_summary": result["zinc_summary"],
            "validation_json": validation_path,
        })
        write_json(os.path.join(model_dir, "candidate_result.json"), row)
        return row
    except Exception as exc:
        write_json(os.path.join(model_dir, "candidate_failure.json"), {
            "model_id": model_id,
            "seed": int(seed),
            "site_mode": args.site_mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return {
            "model_id": model_id,
            "seed": int(seed),
            "site_mode": args.site_mode,
            "target_Ca_Si": float(args.target_ca_si),
            "target_Zn_Si": float(args.target_zn_si) if args.target_zn_si is not None else None,
            "target_Zn_count": args.target_zn_count,
            "composition_match": False,
            "validation_label": "not_generated",
            "validation_passed": False,
            "accepted_for_composition_target": False,
            "coordination_quality": "not_generated",
            "failure_reason": failure_reason({}, False, "not_generated", args.ideal_only, exc=exc),
            "model_dir": model_dir,
        }


def count_rows(rows, key):
    counts = Counter(str(row.get(key) or "none") for row in rows)
    return [{key: name, "count": counts[name]} for name in sorted(counts)]


def representative_rows(rows):
    accepted = [row for row in rows if boolish(row.get("accepted_for_composition_target"))]
    quality_rank = {"ideal_fourfold": 0, "minimum_valid": 1, "overcoordinated": 2, "no_zinc": 9, "undercoordinated_failed": 9}
    accepted.sort(key=lambda row: (
        quality_rank.get(row.get("coordination_quality"), 9),
        abs(float(row.get("Ca_Si_error") or 0.0)),
        abs(float(row.get("Zn_Si_error") or 0.0)),
        int(row.get("seed") or 0),
    ))
    reps = {}
    if accepted:
        reps["best_overall"] = accepted[0]
    for quality in ("ideal_fourfold", "minimum_valid", "overcoordinated"):
        pool = [row for row in accepted if row.get("coordination_quality") == quality]
        if pool:
            reps["best_" + quality] = pool[0]
    return reps


def build_parser():
    parser = argparse.ArgumentParser(description="Generate and screen Zn-modified C-S-H models against target composition windows.")
    parser.add_argument("--target-ca-si", type=float, default=1.7)
    parser.add_argument("--ca-si-tol", type=float, default=0.15)
    parser.add_argument("--target-zn-si", type=float, default=None)
    parser.add_argument("--zn-si-tol", type=float, default=0.05)
    parser.add_argument("--target-zn-count", type=int, default=None)
    parser.add_argument("--site-mode", choices=sorted(SITE_MODE_TO_GENERATOR), default="multi_q2b")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--output-dir", default=os.path.join("output_Y", "workflow_v1", "composition_targeted"))
    parser.add_argument(
        "--include-overcoordinated",
        action="store_true",
        help=(
            "Compatibility flag only. Overcoordinated candidates are included by "
            "default as minimum-valid candidates unless --ideal-only is used."
        ),
    )
    parser.add_argument(
        "--ideal-only",
        action="store_true",
        help="Accept only candidates with coordination_quality == ideal_fourfold.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.target_zn_si is not None and args.target_zn_count is not None:
        raise SystemExit("--target-zn-si and --target-zn-count are mutually exclusive")
    if args.target_zn_si is None and args.target_zn_count is None:
        raise SystemExit("Provide --target-zn-si or --target-zn-count")
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "models"))
    seeds = [int(args.seed_start) + i for i in range(int(args.n_models))]
    manifest = {
        "workflow": "v1.12-composition-target-interface",
        "description": "target-window composition screening; exact final Ca/Si and Zn/Si are not guaranteed",
        "target_Ca_Si": float(args.target_ca_si),
        "Ca_Si_tolerance": float(args.ca_si_tol),
        "target_Zn_Si": None if args.target_zn_si is None else float(args.target_zn_si),
        "Zn_Si_tolerance": float(args.zn_si_tol),
        "target_Zn_count": None if args.target_zn_count is None else int(args.target_zn_count),
        "site_mode": args.site_mode,
        "n_models": len(seeds),
        "seeds": seeds,
        "include_overcoordinated": bool(args.include_overcoordinated),
        "ideal_only": bool(args.ideal_only),
        "coordination_acceptance_policy": (
            "overcoordinated candidates are included by default as minimum-valid candidates; "
            "--include-overcoordinated is a compatibility no-op; --ideal-only restricts "
            "accepted candidates to ideal_fourfold"
        ),
        "finite_temperature_md": "not run",
        "mechanics": "not run",
    }
    write_json(os.path.join(args.output_dir, "composition_target_manifest.json"), manifest)
    rows = [run_one(args, index, seed) for index, seed in enumerate(seeds, start=1)]
    matched = [row for row in rows if boolish(row.get("accepted_for_composition_target"))]
    unmatched_valid = [
        row for row in rows
        if boolish(row.get("validation_passed")) and not boolish(row.get("composition_match"))
    ]
    rejected = [row for row in rows if not boolish(row.get("validation_passed"))]
    write_csv(os.path.join(args.output_dir, "composition_target_summary.csv"), rows)
    write_csv(os.path.join(args.output_dir, "composition_matched_models.csv"), matched)
    write_csv(os.path.join(args.output_dir, "composition_unmatched_valid_models.csv"), unmatched_valid)
    write_csv(os.path.join(args.output_dir, "composition_rejected_models.csv"), rejected)
    write_csv(os.path.join(args.output_dir, "composition_failure_reason_summary.csv"), count_rows(rows, "failure_reason"), ["failure_reason", "count"])
    write_csv(os.path.join(args.output_dir, "composition_coordination_quality_summary.csv"), count_rows(rows, "coordination_quality"), ["coordination_quality", "count"])
    representatives = representative_rows(rows)
    write_json(os.path.join(args.output_dir, "representative_composition_matched_models.json"), representatives)
    summary = {
        "ok": True,
        "n_models": len(rows),
        "n_composition_matched": len(matched),
        "n_unmatched_valid": len(unmatched_valid),
        "n_rejected": len(rejected),
        "manifest": os.path.join(args.output_dir, "composition_target_manifest.json"),
        "target_window_screening": True,
        "exact_composition_guaranteed": False,
        "rows": rows,
    }
    write_json(os.path.join(args.output_dir, "composition_target_summary.json"), summary)
    print(json.dumps({
        "ok": True,
        "n_models": len(rows),
        "n_composition_matched": len(matched),
        "n_unmatched_valid": len(unmatched_valid),
        "n_rejected": len(rejected),
        "summary": os.path.join(args.output_dir, "composition_target_summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
