from __future__ import print_function

import argparse
import json
import multiprocessing
import os
import sys
import time


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.join(REPO_ROOT, "pyCSH_Zn")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from workflow import (
    SITE_MODE_TO_INTERNAL,
    count_rows,
    ensure_dir,
    representative_rows,
    run_one_model,
    write_csv,
    write_json,
)


VERSION = "v2.1.1-public-polish"


class PyCZSHArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        if "not allowed with argument" in message and "--target-zn" in message:
            message = message + "; --target-zn-si and --target-zn-count are mutually exclusive"
        super(PyCZSHArgumentParser, self).error(message)


def build_parser():
    parser = PyCZSHArgumentParser(
        description=(
            "pyCZSH {}. Generate and validate pure or Zn-modified C-S-H structures "
            "with a unified pyCZSH workflow."
        ).format(VERSION)
    )
    parser.add_argument("--version", action="version", version="pyCZSH {}".format(VERSION))
    parser.add_argument("--target-ca-si", type=float, default=1.7)
    parser.add_argument("--target-w-si", type=float, default=0.2)
    zn_group = parser.add_mutually_exclusive_group()
    zn_group.add_argument(
        "--target-zn-si",
        type=float,
        default=None,
        help="Target Zn/Si ratio. Defaults to 0.05 when neither Zn option is provided.",
    )
    zn_group.add_argument(
        "--target-zn-count",
        type=int,
        default=None,
        help="Target Zn atom count. Mutually exclusive with --target-zn-si.",
    )
    parser.add_argument(
        "--q1-q2b-ratio",
        type=float,
        default=0.5,
        help="Target fraction N_Q1_Zn / N_Zn_total for q1_q2b_single_structure_mixture.",
    )
    parser.add_argument(
        "--site-mode",
        choices=sorted(SITE_MODE_TO_INTERNAL),
        default="q1_q2b_single_structure_mixture",
        help=(
            "Zn placement mode: q1_only/q2b_only are single-Zn modes; "
            "multi_q1/multi_q2b/q1_q2b_single_structure_mixture place multiple motifs in one structure."
        ),
    )
    parser.add_argument("--n-models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--output-dir", default="output_pyczsh")
    parser.add_argument("--ideal-only", action="store_true")
    parser.add_argument("--build-lammps-inputs", action="store_true")
    parser.add_argument("--no-lammps", action="store_true", help="Compatibility no-op; LAMMPS is off unless explicitly requested.")
    parser.add_argument("--run-static-relaxation", action="store_true")
    parser.add_argument(
        "--run-quasistatic",
        action="store_true",
        help="Run plus/minus small-strain x-direction diagnostic input checks; not final mechanics.",
    )
    parser.add_argument("--export-clean-data", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-zn-zn-distance", type=float, default=5.0)
    parser.add_argument("--lammps-command", default="lammps")
    return parser


def validate_args(args):
    if args.target_zn_si is None and args.target_zn_count is None:
        args.target_zn_si = 0.05
    if args.q1_q2b_ratio < 0.0 or args.q1_q2b_ratio > 1.0:
        raise SystemExit("--q1-q2b-ratio must be between 0.0 and 1.0")
    if args.n_models < 1:
        raise SystemExit("--n-models must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.target_zn_count is not None and args.target_zn_count < 0:
        raise SystemExit("--target-zn-count must be non-negative")
    if args.run_quasistatic and not args.run_static_relaxation:
        raise SystemExit("--run-quasistatic requires --run-static-relaxation")
    if args.run_static_relaxation or args.run_quasistatic:
        args.build_lammps_inputs = True
    if args.site_mode == "q1_q2b_single_structure_mixture" and args.q1_q2b_ratio in (0.0, 1.0):
        print(
            "Warning: q1_q2b_single_structure_mixture needs both Q1_Zn and Q2b_Zn; "
            "the workflow will allocate at least one of each when N_Zn_total >= 2."
        )
    return args


def args_to_dict(args, output_dir):
    target_zn_si = None if args.target_zn_count is not None else args.target_zn_si
    return {
        "output_dir": output_dir,
        "target_ca_si": float(args.target_ca_si),
        "target_w_si": float(args.target_w_si),
        "target_zn_si": target_zn_si,
        "target_zn_count": args.target_zn_count,
        "q1_q2b_ratio": float(args.q1_q2b_ratio),
        "site_mode": args.site_mode,
        "seed": args.seed,
        "seed_start": int(args.seed_start),
        "ideal_only": bool(args.ideal_only),
        "build_lammps_inputs": bool(args.build_lammps_inputs),
        "run_static_relaxation": bool(args.run_static_relaxation),
        "run_quasistatic": bool(args.run_quasistatic),
        "export_clean_data": bool(args.export_clean_data),
        "min_zn_zn_distance": float(args.min_zn_zn_distance),
        "lammps_command": args.lammps_command,
    }


def run_models(base_args, n_models, workers):
    tasks = []
    for idx in range(1, int(n_models) + 1):
        item = dict(base_args)
        item["model_index"] = idx
        tasks.append(item)
    if int(workers) == 1:
        return [run_one_model(task) for task in tasks]
    pool = multiprocessing.Pool(processes=int(workers))
    try:
        return pool.map(run_one_model, tasks)
    finally:
        pool.close()
        pool.join()


def main(argv=None):
    start = time.time()
    args = validate_args(build_parser().parse_args(argv))
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "structures"))
    ensure_dir(os.path.join(output_dir, "logs"))
    manifest = {
        "program": "pyCZSH",
        "version": VERSION,
        "recommended_entry_point": "main_pyczsh.py",
        "output_dir": output_dir,
        "target_Ca_Si": float(args.target_ca_si),
        "target_W_Si": float(args.target_w_si),
        "target_Zn_Si": None if args.target_zn_count is not None else float(args.target_zn_si),
        "target_Zn_count": args.target_zn_count,
        "site_mode": args.site_mode,
        "target_q1_q2b_ratio": float(args.q1_q2b_ratio) if args.site_mode == "q1_q2b_single_structure_mixture" else None,
        "n_models": int(args.n_models),
        "seed": args.seed,
        "seed_start": int(args.seed_start),
        "build_lammps_inputs": bool(args.build_lammps_inputs),
        "run_static_relaxation": bool(args.run_static_relaxation),
        "run_quasistatic": bool(args.run_quasistatic),
        "export_clean_data": bool(args.export_clean_data),
        "workers": int(args.workers),
        "notes": [
            "Target Ca/Si and Zn/Si are requested target-window values, not guaranteed exact final compositions.",
            "Internal data files retain CS-Info for validation and core-shell metadata.",
            "Clean data export is optional and is intended only for external reading or visualization convenience.",
            "LAMMPS static relaxation and small-strain x-direction diagnostic checks are opt-in.",
        ],
    }
    write_json(os.path.join(output_dir, "manifest.json"), manifest)
    rows = run_models(args_to_dict(args, output_dir), args.n_models, args.workers)
    accepted = [row for row in rows if row.get("accepted")]
    rejected = [row for row in rows if not row.get("accepted")]
    write_csv(os.path.join(output_dir, "composition_summary.csv"), rows)
    write_json(os.path.join(output_dir, "composition_summary.json"), {"models": rows})
    write_csv(os.path.join(output_dir, "accepted_models.csv"), accepted)
    write_csv(os.path.join(output_dir, "rejected_models.csv"), rejected)
    write_csv(
        os.path.join(output_dir, "failure_reason_summary.csv"),
        count_rows(rows, "failure_reason"),
        ["failure_reason", "count"],
    )
    write_csv(
        os.path.join(output_dir, "coordination_quality_summary.csv"),
        count_rows(rows, "coordination_quality"),
        ["coordination_quality", "count"],
    )
    write_json(os.path.join(output_dir, "representative_models.json"), representative_rows(rows))
    elapsed = time.time() - start
    summary = {
        "ok": True,
        "version": VERSION,
        "output_dir": output_dir,
        "n_models": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "elapsed_seconds": elapsed,
    }
    write_json(os.path.join(output_dir, "run_summary.json"), summary)
    with open(os.path.join(output_dir, "logs", "run_summary.txt"), "w") as f:
        f.write("pyCZSH {}\n".format(VERSION))
        f.write("output_dir: {}\n".format(output_dir))
        f.write("n_models: {}\n".format(len(rows)))
        f.write("n_accepted: {}\n".format(len(accepted)))
        f.write("n_rejected: {}\n".format(len(rejected)))
        f.write("elapsed_seconds: {:.3f}\n".format(elapsed))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
