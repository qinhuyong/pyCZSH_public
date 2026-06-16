from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs
from validate_cementff_data import parse_data, validate
from workflow_common import generate_structure

try:
    from postprocess.analyze_structure import analyze as analyze_structure
except Exception:
    analyze_structure = None


MODES = ("q2b_only", "q1_only", "q1_q2b_mixture")
VALID_ZN_LABELS = ("valid_q1_zn_candidate", "valid_q2b_zn_candidate")
CSV_FIELDS = [
    "model_id",
    "seed",
    "requested_mode",
    "target_Zn_Si",
    "actual_Zn_Si",
    "target_Q1_fraction",
    "actual_Q1_count",
    "actual_Q2b_count",
    "n_Zn_total",
    "initial_validation_label",
    "postmin_validation_label",
    "initial_ok",
    "postmin_ok",
    "charge_residual",
    "min_Zn_O_distance",
    "mean_Zn_O_distance",
    "max_Zn_O_distance",
    "Zn_O_coordination_2p5A",
    "usable_for_mechanics",
    "failure_reason",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_seed_list(text):
    if not text:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


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


def choose_site_type(mode, q1_fraction, seed):
    if mode == "q2b_only":
        return "Q2b_Zn"
    if mode == "q1_only":
        return "Q1_Zn"
    rng = random.Random(int(seed))
    return "Q1_Zn" if rng.random() < float(q1_fraction) else "Q2b_Zn"


def expected_label(site_type):
    return "valid_q1_zn_candidate" if site_type == "Q1_Zn" else "valid_q2b_zn_candidate"


def summarize_zinc(validation):
    zinc = validation.get("zinc", {}) or {}
    sites = zinc.get("zinc_sites", []) or []
    distances = []
    coord = []
    for site in sites:
        coord.append(site.get("coordination_2p5"))
        for rec in site.get("nearest_oxygen", []) or []:
            if rec.get("distance") is not None:
                distances.append(float(rec["distance"]))
    return {
        "n_Zn_total": zinc.get("n_zinc", 0),
        "min_Zn_O_distance": min(distances) if distances else None,
        "mean_Zn_O_distance": sum(distances) / len(distances) if distances else None,
        "max_Zn_O_distance": max(distances) if distances else None,
        "Zn_O_coordination_2p5A": ";".join(str(x) for x in coord) if coord else "",
    }


def count_actual_sites(zinc_summary_path):
    if not zinc_summary_path or not os.path.exists(zinc_summary_path):
        return {"actual_Zn_Si": None, "actual_Q1_count": 0, "actual_Q2b_count": 0}
    with open(zinc_summary_path) as f:
        summary = json.load(f)
    return {
        "actual_Zn_Si": summary.get("actual_Zn_Si_original_ratio", summary.get("actual_Zn_Si_ratio")),
        "actual_Q1_count": int(summary.get("N_Q1_Zn", 0) or 0),
        "actual_Q2b_count": int(summary.get("N_Q2b_Zn", 0) or 0),
    }


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


def model_id(index):
    return "model_{:06d}".format(index)


def effective_zn_ratio(target_zn_si):
    # v1.4 intentionally keeps one Zn motif per independently generated model.
    # A tiny positive ratio triggers the existing one-Zn path without enabling
    # unsafe multi-Zn motif coupling inside a single structure.
    return 1.0e-9


def run_static_relaxation(lmp, model_dir, data_file, site_type, zinc_summary_path, prefix):
    input_dir = os.path.join(model_dir, "lammps_inputs")
    steps = {}
    for input_name in ("in.read_check", "in.run0", "in.minimize_static"):
        steps[input_name] = run_lammps(lmp, input_dir, input_name)
        if not steps[input_name]["ok"]:
            return {
                "ok": False,
                "steps": steps,
                "failure_reason": "{} failed with return code {}".format(input_name, steps[input_name]["returncode"]),
            }
    raw = os.path.join(input_dir, "{}_minimized_static.raw.data".format(prefix))
    final = os.path.join(input_dir, "{}_minimized_static.data".format(prefix))
    if not os.path.exists(raw):
        return {"ok": False, "steps": steps, "failure_reason": "LAMMPS minimize did not write raw data"}
    append_csinfo(data_file, raw, final)
    validation = validate(final, expected_zinc_site_type=site_type, zinc_summary_path=zinc_summary_path)
    validation_path = os.path.join(model_dir, "validation_postmin.json")
    write_json(validation_path, validation)
    analysis_path = None
    if analyze_structure is not None:
        try:
            analysis_dir = os.path.join(model_dir, "postprocess")
            analyze_structure(final, analysis_dir)
            analysis_path = os.path.join(analysis_dir, "structure_analysis.json")
        except Exception as exc:
            analysis_path = "postprocess failed: {}".format(exc)
    return {
        "ok": validation["classification"] == expected_label(site_type),
        "steps": steps,
        "postmin_data": final,
        "validation": validation,
        "validation_path": validation_path,
        "structure_analysis": analysis_path,
        "failure_reason": None if validation["classification"] == expected_label(site_type) else "post-min validation classified as {}".format(validation["classification"]),
    }


def run_model(index, seed, args, lmp):
    mid = model_id(index)
    model_dir = os.path.join(args.out_dir, "models", mid)
    if os.path.abspath(model_dir).startswith(os.path.abspath(os.path.join(args.out_dir, "models"))) and os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)
    site_type = choose_site_type(args.mode, args.q1_fraction, seed)
    prefix = "{}_{}".format(mid, site_type.lower())
    config = {
        "model_id": mid,
        "seed": int(seed),
        "requested_mode": args.mode,
        "selected_site_type": site_type,
        "target_Zn_Si": float(args.target_zn_si),
        "effective_Zn_Si_used_for_generation": effective_zn_ratio(args.target_zn_si),
        "target_Q1_fraction": float(args.q1_fraction),
        "one_Zn_motif_per_structure": True,
        "run_static_relaxation": bool(args.run_static_relaxation),
        "keep_postmin_valid_only": bool(args.keep_postmin_valid_only),
    }
    write_json(os.path.join(model_dir, "input_config.json"), config)
    base_row = {
        "model_id": mid,
        "seed": int(seed),
        "requested_mode": args.mode,
        "target_Zn_Si": float(args.target_zn_si),
        "target_Q1_fraction": float(args.q1_fraction),
        "initial_ok": False,
        "postmin_ok": False,
        "usable_for_mechanics": False,
        "failure_reason": "",
    }
    try:
        result = generate_structure(
            model_dir,
            prefix,
            enable_zinc=True,
            zn_ratio=effective_zn_ratio(args.target_zn_si),
            site_type=site_type,
            seed=int(seed),
        )
        ff_result = build_forcefield(model_dir)
        inputs_result = build_inputs(result["data_file"], ff_result["forcefield"], os.path.join(model_dir, "lammps_inputs"), prefix)
        manifest = {
            "input_config": config,
            "generated": result,
            "forcefield": ff_result,
            "lammps_inputs": inputs_result,
        }
        write_json(os.path.join(model_dir, "generation_manifest.json"), manifest)
        initial_validation = validate(result["data_file"], expected_zinc_site_type=site_type, zinc_summary_path=result["zinc_summary"])
        write_json(os.path.join(model_dir, "validation_initial.json"), initial_validation)
        counts = count_actual_sites(result["zinc_summary"])
        zn_stats = summarize_zinc(initial_validation)
        row = dict(base_row)
        row.update(counts)
        row.update(zn_stats)
        row["initial_validation_label"] = initial_validation["classification"]
        row["initial_ok"] = initial_validation["classification"] == expected_label(site_type)
        row["charge_residual"] = initial_validation.get("total_charge")
        if not row["initial_ok"]:
            row["failure_reason"] = "initial validation classified as {}".format(initial_validation["classification"])
            return {"row": row, "accepted": False, "model_dir": model_dir}
        if args.initial_only or not args.run_static_relaxation:
            row["postmin_validation_label"] = ""
            row["postmin_ok"] = False
            row["usable_for_mechanics"] = False
            return {"row": row, "accepted": row["initial_ok"] and not args.keep_postmin_valid_only, "model_dir": model_dir}
        if not lmp:
            row["failure_reason"] = "LAMMPS executable not found"
            return {"row": row, "accepted": False, "model_dir": model_dir}
        relax = run_static_relaxation(lmp, model_dir, result["data_file"], site_type, result["zinc_summary"], prefix)
        write_json(os.path.join(model_dir, "static_relaxation_report.json"), relax)
        post_validation = relax.get("validation")
        if post_validation:
            post_counts = count_actual_sites(result["zinc_summary"])
            post_stats = summarize_zinc(post_validation)
            row.update(post_counts)
            row.update(post_stats)
            row["postmin_validation_label"] = post_validation["classification"]
            row["postmin_ok"] = post_validation["classification"] == expected_label(site_type)
            row["charge_residual"] = post_validation.get("total_charge")
        else:
            row["postmin_validation_label"] = ""
            row["postmin_ok"] = False
        row["usable_for_mechanics"] = bool(row["postmin_ok"])
        row["failure_reason"] = "" if row["postmin_ok"] else (relax.get("failure_reason") or "post-min validation failed")
        return {"row": row, "accepted": bool(row["postmin_ok"]), "model_dir": model_dir}
    except Exception as exc:
        failure = {
            "ok": False,
            "model_id": mid,
            "seed": int(seed),
            "site_type": site_type,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "fallback_used": False,
        }
        write_json(os.path.join(model_dir, "generation_failure.json"), failure)
        row = dict(base_row)
        row["failure_reason"] = "{}: {}".format(type(exc).__name__, exc)
        return {"row": row, "accepted": False, "model_dir": model_dir}


def build_parser():
    parser = argparse.ArgumentParser(description="Generate constrained random Zn-C-S-H static candidate ensembles.")
    parser.add_argument("--n-models", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1400)
    parser.add_argument("--seed-list", default=None, help="Comma-separated seed list. Overrides --n-models/--seed-start.")
    parser.add_argument("--mode", choices=MODES, default="q1_q2b_mixture")
    parser.add_argument("--q1-fraction", type=float, default=0.5)
    parser.add_argument("--target-zn-si", type=float, default=0.05)
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--run-static-relaxation", action="store_true")
    parser.add_argument("--keep-postmin-valid-only", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join("output_Y", "workflow_v1", "zn_csh_ensemble"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.q1_fraction < 0.0 or args.q1_fraction > 1.0:
        raise SystemExit("--q1-fraction must be between 0 and 1")
    if args.n_models <= 0:
        raise SystemExit("--n-models must be positive")
    seeds = parse_seed_list(args.seed_list)
    if seeds is None:
        seeds = [args.seed_start + i for i in range(args.n_models)]
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "models"))
    lmp = find_lammps() if args.run_static_relaxation and not args.initial_only else None
    manifest = {
        "workflow": "v1.4-Zn-C-S-H-ensemble-generator",
        "mode": args.mode,
        "n_models": len(seeds),
        "seeds": seeds,
        "target_Zn_Si": float(args.target_zn_si),
        "q1_fraction": float(args.q1_fraction),
        "one_Zn_motif_per_structure": True,
        "mixture_policy": "q1_q2b_mixture is ensemble-level assignment of Q1_Zn or Q2b_Zn per independent model; it does not enable mixed_Q1_Q2b_Zn site type.",
        "target_Zn_Si_policy": "v1.4 records the requested target_Zn_Si, but generates one Zn motif per structure; actual_Zn_Si is reported per model.",
        "run_static_relaxation": bool(args.run_static_relaxation),
        "initial_only": bool(args.initial_only),
        "lammps_executable": lmp,
        "finite_temperature_md": "not run",
        "scope": "static candidate ensemble generation; mechanics is downstream opt-in for post-min valid structures",
    }
    write_json(os.path.join(args.out_dir, "ensemble_manifest.json"), manifest)
    rows = []
    for i, seed in enumerate(seeds, start=1):
        result = run_model(i, seed, args, lmp)
        rows.append(result["row"])
    accepted = [row for row in rows if row.get("postmin_ok") or (not args.run_static_relaxation and row.get("initial_ok") and not args.keep_postmin_valid_only)]
    rejected = [row for row in rows if row not in accepted]
    summary = {
        "ok": True,
        "mode": args.mode,
        "n_models": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted_model_ids": [row["model_id"] for row in accepted],
        "rejected_model_ids": [row["model_id"] for row in rejected],
        "postmin_valid_labels": sorted(set(row.get("postmin_validation_label") for row in accepted if row.get("postmin_validation_label"))),
        "one_Zn_motif_per_structure": True,
        "single_structure_mixed_Q1_Q2b_supported": False,
        "rows": rows,
    }
    write_csv(os.path.join(args.out_dir, "ensemble_summary.csv"), rows)
    write_csv(os.path.join(args.out_dir, "accepted_models.csv"), accepted)
    write_csv(os.path.join(args.out_dir, "rejected_models.csv"), rejected)
    write_json(os.path.join(args.out_dir, "ensemble_summary.json"), summary)
    print(json.dumps({
        "ok": summary["ok"],
        "mode": args.mode,
        "n_models": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "summary": os.path.join(args.out_dir, "ensemble_summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
