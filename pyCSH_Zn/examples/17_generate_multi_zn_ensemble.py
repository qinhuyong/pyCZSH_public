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
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SCREEN16_PATH = os.path.join(SCRIPT_DIR, "16_screen_multi_zn_combinations.py")
spec = importlib.util.spec_from_file_location("multi_zn_screen", SCREEN16_PATH)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

from mod_zinc import build_multi_candidate_pools, build_zinc_candidate_site_report, inspect_zinc_candidates


ENSEMBLE_MODES = (
    "multi_q2b_ensemble",
    "multi_q1_ensemble",
    "q1_q2b_single_structure_mixed_ensemble",
    "mixed_multi_zn_ensemble",
)
INTERNAL_MODES = ("multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture")
VALID_MULTI_LABELS = (
    "valid_multi_q1_zn_candidate",
    "valid_multi_q2b_zn_candidate",
    "valid_multi_q1_q2b_zn_candidate",
)
CSV_FIELDS = [
    "model_id",
    "seed",
    "requested_mode",
    "internal_mode",
    "n_q1_requested",
    "n_q2b_requested",
    "n_q1_actual",
    "n_q2b_actual",
    "n_Zn_total",
    "target_Zn_Si",
    "actual_Zn_Si",
    "selected_candidate_id",
    "selected_Si_sites",
    "motif_types",
    "initial_validation_label",
    "postmin_validation_label",
    "initial_valid",
    "postmin_valid",
    "coordination_quality",
    "per_center_coordination",
    "failed_center_count",
    "failed_center_ids",
    "overcoordinated_center_count",
    "total_excess_coordination",
    "min_Zn_Zn_distance",
    "max_Zn_O_distance",
    "mean_Zn_O_distance",
    "charge_residual",
    "accepted",
    "mechanics_ready",
    "failure_reason",
    "postmin_data_path",
    "model_dir",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows, fields=CSV_FIELDS):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def run_lammps_with_timeout(lmp, input_dir, input_name):
    timeout = int(os.environ.get("PYCSH_MULTI_ZN_LAMMPS_TIMEOUT", "180"))
    out_path = os.path.join(input_dir, input_name + ".stdout.txt")
    err_path = os.path.join(input_dir, input_name + ".stderr.txt")
    try:
        proc = subprocess.run(
            [lmp, "-in", input_name],
            cwd=input_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
        ok = returncode == 0
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nLAMMPS step timed out after {} seconds\n".format(timeout)
        returncode = 124
        ok = False
    with open(out_path, "w") as f:
        f.write(stdout)
    with open(err_path, "w") as f:
        f.write(stderr)
    return {"returncode": returncode, "stdout": out_path, "stderr": err_path, "ok": ok}


def parse_seed_list(text):
    if not text:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def model_id(index):
    return "model_{:06d}".format(index)


def internal_mode_for_model(args, index, seed):
    if args.mode == "multi_q2b_ensemble":
        return "multi_q2b"
    if args.mode == "multi_q1_ensemble":
        return "multi_q1"
    if args.mode == "q1_q2b_single_structure_mixed_ensemble":
        return "q1_q2b_single_structure_mixture"
    choices = ["multi_q2b", "multi_q1", "q1_q2b_single_structure_mixture"]
    return choices[(index - 1) % len(choices)]


def requested_counts(internal_mode, args):
    if internal_mode == "multi_q2b":
        return 0, max(2, int(args.n_q2b or 2))
    if internal_mode == "multi_q1":
        return max(2, int(args.n_q1 or 2)), 0
    return int(args.n_q1 or 1), int(args.n_q2b or 1)


def expected_label(internal_mode):
    return screen.expected_label(internal_mode)


def boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_coord(text):
    vals = []
    for item in str(text or "").replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(int(float(item)))
        except ValueError:
            pass
    return vals


def validation_zno_stats(validation):
    distances = []
    for site in validation.get("zinc", {}).get("zinc_sites", []) or []:
        for rec in site.get("nearest_oxygen", []) or []:
            if rec.get("distance") is not None:
                distances.append(float(rec["distance"]))
    return {
        "max_Zn_O_distance": max(distances) if distances else None,
        "mean_Zn_O_distance": (sum(distances) / len(distances)) if distances else None,
    }


def zinc_summary_counts(path, fallback_mode, n_q1, n_q2b):
    if not path or not os.path.exists(path):
        return {
            "n_q1_actual": n_q1,
            "n_q2b_actual": n_q2b,
            "n_Zn_total": n_q1 + n_q2b,
            "actual_Zn_Si": None,
        }
    summary = read_json(path)
    return {
        "n_q1_actual": int(summary.get("N_Q1_Zn", 0) or sum(1 for c in summary.get("zn_centers", []) if c.get("motif_type") == "Q1_Zn")),
        "n_q2b_actual": int(summary.get("N_Q2b_Zn", 0) or sum(1 for c in summary.get("zn_centers", []) if c.get("motif_type") == "Q2b_Zn")),
        "n_Zn_total": int(summary.get("N_Zn_added", 0) or summary.get("n_Zn_total", 0) or len(summary.get("zn_centers", [])) or n_q1 + n_q2b),
        "actual_Zn_Si": summary.get("actual_Zn_Si_original_ratio", summary.get("actual_Zn_Si_ratio")),
    }


def total_excess_coordination(coord):
    return sum(max(0, int(v) - 4) for v in coord)


def row_rank(row, prefer_ideal):
    coord = parse_coord(row.get("per_center_coordination"))
    quality_rank = {"ideal_fourfold": 0, "minimum_valid": 1, "overcoordinated": 2, "undercoordinated_failed": 9}
    if not prefer_ideal:
        quality_rank["overcoordinated"] = 1
    return (
        0 if boolish(row.get("postmin_valid")) and row.get("postmin_validation_label") == expected_label(row.get("internal_mode")) else 1,
        quality_rank.get(row.get("coordination_quality"), 9),
        int(row.get("overcoordinated_center_count") or 0),
        total_excess_coordination(coord),
        float(row.get("max_Zn_O_distance") or 999.0),
        -float(row.get("min_Zn_Zn_distance") or 0.0),
    )


def select_best(rows, prefer_ideal):
    valid = [
        row for row in rows
        if boolish(row.get("postmin_valid"))
        and row.get("postmin_validation_label") == expected_label(row.get("internal_mode"))
    ]
    pool = valid or rows
    if not pool:
        return None
    return sorted(pool, key=lambda row: row_rank(row, prefer_ideal))[0]


def postmin_data_path(candidate_dir):
    data = os.path.join(candidate_dir, "lammps_inputs", "multi_zn_minimized_static.data")
    return data if os.path.exists(data) else ""


def promote_candidate(candidate_dir, model_dir):
    names = [
        "multi_zn_cementff_zn.data",
        "multi_zinc_summary.json",
        "water_summary.json",
        "cementff_mapping_summary.json",
        "validation_initial.json",
        "validation_postmin.json",
        "multi_zn_pre_post_coordination_compare.json",
        "structure_analysis.json",
        "in.CementFF4_Zn",
    ]
    for name in names:
        src = os.path.join(candidate_dir, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(model_dir, name))
    final = postmin_data_path(candidate_dir)
    if final:
        shutil.copyfile(final, os.path.join(model_dir, "multi_zn_postmin.data"))


def build_model_row(args, index, seed, internal_mode, n_q1, n_q2b, row, model_dir, accepted):
    candidate_id = row.get("candidate_id", "")
    candidate_dir = os.path.join(model_dir, "candidates", candidate_id) if candidate_id else ""
    validation_path = os.path.join(candidate_dir, "validation_postmin.json")
    validation = read_json(validation_path) if os.path.exists(validation_path) else {}
    stats = validation_zno_stats(validation)
    zinc_counts = zinc_summary_counts(os.path.join(candidate_dir, "multi_zinc_summary.json"), internal_mode, n_q1, n_q2b)
    postmin_path = postmin_data_path(candidate_dir)
    coord = parse_coord(row.get("per_center_coordination"))
    mechanics_ready = (
        accepted
        and row.get("postmin_validation_label") in VALID_MULTI_LABELS
        and coord
        and min(coord) >= 4
        and bool(postmin_path)
    )
    out = {
        "model_id": model_id(index),
        "seed": int(seed),
        "requested_mode": args.mode,
        "internal_mode": internal_mode,
        "n_q1_requested": int(n_q1),
        "n_q2b_requested": int(n_q2b),
        "target_Zn_Si": float(args.target_zn_si),
        "selected_candidate_id": candidate_id,
        "selected_Si_sites": row.get("selected_sites", ""),
        "motif_types": row.get("motif_types", ""),
        "initial_validation_label": row.get("initial_validation_label", ""),
        "postmin_validation_label": row.get("postmin_validation_label", ""),
        "initial_valid": boolish(row.get("initial_valid")),
        "postmin_valid": boolish(row.get("postmin_valid")),
        "coordination_quality": row.get("coordination_quality", ""),
        "per_center_coordination": row.get("per_center_coordination", ""),
        "failed_center_count": row.get("failed_center_count", 0),
        "failed_center_ids": row.get("failed_center_ids", ""),
        "overcoordinated_center_count": row.get("overcoordinated_center_count", 0),
        "total_excess_coordination": total_excess_coordination(coord),
        "min_Zn_Zn_distance": row.get("min_Zn_Zn_distance"),
        "max_Zn_O_distance": stats["max_Zn_O_distance"],
        "mean_Zn_O_distance": stats["mean_Zn_O_distance"],
        "charge_residual": validation.get("total_charge"),
        "accepted": bool(accepted),
        "mechanics_ready": bool(mechanics_ready),
        "failure_reason": "" if accepted else (row.get("failure_reason") or "no post-min valid multi-Zn candidate found"),
        "postmin_data_path": postmin_path,
        "model_dir": model_dir,
    }
    out.update(zinc_counts)
    return out


def screen_model(args, index, seed):
    mid = model_id(index)
    model_dir = os.path.join(args.out_dir, "models", mid)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)
    internal_mode = internal_mode_for_model(args, index, seed)
    n_q1, n_q2b = requested_counts(internal_mode, args)
    config = {
        "model_id": mid,
        "seed": int(seed),
        "requested_mode": args.mode,
        "internal_mode": internal_mode,
        "n_q1": int(n_q1),
        "n_q2b": int(n_q2b),
        "target_Zn_Si": float(args.target_zn_si),
        "max_combinations_per_model": int(args.max_combinations_per_model),
        "min_zn_zn_distance": float(args.min_zn_zn_distance),
        "run_static_relaxation": bool(args.run_static_relaxation),
        "prefer_ideal_fourfold": bool(args.prefer_ideal_fourfold),
        "finite_temperature_md": "not run",
    }
    write_json(os.path.join(model_dir, "input_config.json"), config)
    try:
        parent = screen.alpha.sample_parent(seed)
        entries, bonds, crystal_dict, angles, supercell, ca_si_ratio, n_si = parent
        candidates = inspect_zinc_candidates(crystal_dict)
        candidate_site_report = build_zinc_candidate_site_report(candidates, entries, bonds, angles, supercell, False, True, 1.95)
        pools = build_multi_candidate_pools(candidates, candidate_site_report, seed, entries, bonds, supercell)
        rows = []
        skipped = []
        tried = 0
        for raw_combo in screen.combination_iter(pools, internal_mode, n_q1, n_q2b, args.max_combinations_per_model * 10):
            selected, conflict, reason = screen.prepare_selected(raw_combo, entries, bonds, supercell, args.min_zn_zn_distance)
            if selected is None:
                skipped.append({"mode": internal_mode, "reason": reason, "has_OH_conflict": conflict})
                continue
            tried += 1
            candidate_id = "candidate_{:06d}_{}".format(tried, internal_mode)
            candidate_dir = os.path.join(model_dir, "candidates", candidate_id)
            try:
                row = screen.run_candidate(
                    candidate_dir,
                    candidate_id,
                    internal_mode,
                    seed,
                    selected,
                    parent,
                    candidate_site_report,
                    skipped[-20:],
                    args.min_zn_zn_distance,
                    args.run_static_relaxation,
                )
            except Exception as exc:
                ensure_dir(candidate_dir)
                row = {
                    "candidate_id": candidate_id,
                    "mode": internal_mode,
                    "seed": seed,
                    "initial_valid": False,
                    "postmin_valid": False,
                    "failure_reason": "{}: {}".format(type(exc).__name__, exc),
                }
                write_json(os.path.join(candidate_dir, "candidate_exception.json"), {"error": str(exc), "traceback": traceback.format_exc()})
            row["internal_mode"] = internal_mode
            rows.append(row)
            if boolish(row.get("postmin_valid")) and row.get("postmin_validation_label") == expected_label(internal_mode):
                break
            if tried >= args.max_combinations_per_model:
                break
        best = select_best(rows, args.prefer_ideal_fourfold)
        accepted = bool(best and boolish(best.get("postmin_valid")) and best.get("postmin_validation_label") == expected_label(internal_mode))
        if best:
            candidate_dir = os.path.join(model_dir, "candidates", best["candidate_id"])
            promote_candidate(candidate_dir, model_dir)
            write_json(os.path.join(model_dir, "selected_candidate_summary.json"), best)
        else:
            best = {
                "candidate_id": "",
                "internal_mode": internal_mode,
                "initial_valid": False,
                "postmin_valid": False,
                "failure_reason": "no candidate combination could be generated",
            }
        generation_manifest = {
            "workflow": "v1.7-multi-Zn-ensemble-generator",
            "input_config": config,
            "n_combinations_screened": len(rows),
            "n_postmin_valid": sum(1 for row in rows if boolish(row.get("postmin_valid"))),
            "selected_candidate": best,
            "skipped_candidates": skipped,
            "selection_policy": "postmin valid first; matching valid_multi label; ideal_fourfold before overcoordinated; lower excess coordination; lower max Zn-O distance",
        }
        write_json(os.path.join(model_dir, "generation_manifest.json"), generation_manifest)
        model_row = build_model_row(args, index, seed, internal_mode, n_q1, n_q2b, best, model_dir, accepted)
        return model_row
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "fallback_used": False}
        write_json(os.path.join(model_dir, "generation_failure.json"), failure)
        return {
            "model_id": mid,
            "seed": int(seed),
            "requested_mode": args.mode,
            "internal_mode": internal_mode,
            "n_q1_requested": int(n_q1),
            "n_q2b_requested": int(n_q2b),
            "n_q1_actual": 0,
            "n_q2b_actual": 0,
            "n_Zn_total": 0,
            "target_Zn_Si": float(args.target_zn_si),
            "accepted": False,
            "mechanics_ready": False,
            "failure_reason": "{}: {}".format(type(exc).__name__, exc),
            "model_dir": model_dir,
        }


def aggregate_counts(rows, key):
    counts = Counter(row.get(key) or "unknown" for row in rows)
    return [{"name": name, "count": counts[name]} for name in sorted(counts)]


def write_named_count_csv(path, rows, name_field):
    fields = [name_field, "count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name_field: row["name"], "count": row["count"]})


def write_survival_summary(path, rows):
    out = []
    for mode in sorted(set(row.get("internal_mode") for row in rows)):
        subset = [row for row in rows if row.get("internal_mode") == mode]
        accepted = [row for row in subset if boolish(row.get("accepted"))]
        out.append({
            "internal_mode": mode,
            "n_models": len(subset),
            "n_accepted": len(accepted),
            "survival_rate": (len(accepted) / len(subset)) if subset else None,
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["internal_mode", "n_models", "n_accepted", "survival_rate"])
        writer.writeheader()
        for row in out:
            writer.writerow(row)
    return out


def representative_selection(rows):
    accepted = [row for row in rows if boolish(row.get("accepted"))]
    reps = {}
    def best(where):
        pool = [row for row in accepted if where(row)]
        return sorted(pool, key=lambda row: row_rank(row, True))[0] if pool else None
    reps["best_multi_q2b_ideal_fourfold"] = best(lambda row: row.get("internal_mode") == "multi_q2b" and row.get("coordination_quality") == "ideal_fourfold")
    reps["best_multi_q2b"] = best(lambda row: row.get("internal_mode") == "multi_q2b")
    reps["best_multi_q1"] = best(lambda row: row.get("internal_mode") == "multi_q1")
    reps["best_q1_q2b_mixed"] = best(lambda row: row.get("internal_mode") == "q1_q2b_single_structure_mixture")
    reps["best_ideal_fourfold"] = best(lambda row: row.get("coordination_quality") == "ideal_fourfold")
    reps["best_minimum_valid_overcoordinated"] = best(lambda row: row.get("coordination_quality") == "overcoordinated")
    return {key: value for key, value in reps.items() if value is not None}


def svg_bar(path, title, labels, values, ylabel="count"):
    width, height, margin = 760, 420, 58
    plot_w, plot_h = width - 2 * margin, height - 2 * margin
    max_v = max([float(v) for v in values] or [1.0])
    max_v = max(max_v, 1.0)
    gap = plot_w / max(len(values), 1)
    bar_w = gap * 0.62
    colors = ["#4f7cac", "#59a14f", "#e15759", "#f28e2b", "#76b7b2", "#af7aa1"]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{}</text>'.format(width / 2, title),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333"/>'.format(margin, height - margin, width - margin),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333"/>'.format(margin, margin, height - margin),
        '<text x="18" y="{}" transform="rotate(-90 18,{})" text-anchor="middle" font-family="Arial" font-size="12">{}</text>'.format(height / 2, height / 2, ylabel),
    ]
    for i, value in enumerate(values):
        x = margin + i * gap + (gap - bar_w) / 2
        h = plot_h * (float(value) / max_v)
        y = height - margin - h
        lines.append('<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" fill="{}"/>'.format(x, y, bar_w, h, colors[i % len(colors)]))
        lines.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{}</text>'.format(x + bar_w / 2, y - 6, value))
        lines.append('<text x="{:.2f}" y="{}" text-anchor="middle" font-family="Arial" font-size="10">{}</text>'.format(x + bar_w / 2, height - margin + 18, labels[i]))
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def svg_hist(path, title, values, xlabel, bins=12):
    vals = [float(v) for v in values if v not in (None, "")]
    if not vals:
        svg_bar(path, title, ["none"], [0], "count")
        return
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1.0e-12:
        labels, counts = ["{:.3g}".format(vmin)], [len(vals)]
    else:
        step = (vmax - vmin) / bins
        counts = [0] * bins
        for value in vals:
            idx = min(bins - 1, int((value - vmin) / step))
            counts[idx] += 1
        labels = ["{:.2f}".format(vmin + (i + 0.5) * step) for i in range(bins)]
    svg_bar(path, title + " (" + xlabel + ")", labels, counts)


def write_plots(out_dir, rows):
    plots = os.path.join(out_dir, "plots")
    ensure_dir(plots)
    accepted = sum(1 for row in rows if boolish(row.get("accepted")))
    rejected = len(rows) - accepted
    svg_bar(os.path.join(plots, "accepted_rejected_counts.svg"), "Accepted vs rejected", ["accepted", "rejected"], [accepted, rejected])
    mode_counts = Counter(row.get("internal_mode") for row in rows)
    svg_bar(os.path.join(plots, "motif_type_counts.svg"), "Motif mode counts", list(mode_counts), [mode_counts[k] for k in mode_counts])
    survival = []
    for mode in sorted(mode_counts):
        subset = [row for row in rows if row.get("internal_mode") == mode]
        survival.append(100.0 * sum(1 for row in subset if boolish(row.get("accepted"))) / len(subset))
    svg_bar(os.path.join(plots, "postmin_survival_rate_by_mode.svg"), "Post-min survival by mode", sorted(mode_counts), [round(x, 3) for x in survival], "survival (%)")
    quality = Counter(row.get("coordination_quality") or "unknown" for row in rows)
    svg_bar(os.path.join(plots, "coordination_quality_counts.svg"), "Coordination quality", list(quality), [quality[k] for k in quality])
    svg_hist(os.path.join(plots, "Zn_Zn_distance_distribution.svg"), "Zn-Zn distance", [row.get("min_Zn_Zn_distance") for row in rows], "Angstrom")
    coords = []
    for row in rows:
        coords.extend(parse_coord(row.get("per_center_coordination")))
    svg_hist(os.path.join(plots, "Zn_O_coordination_distribution.svg"), "Zn-O coordination", coords, "coordination")
    failures = Counter((row.get("failure_reason") or "none") for row in rows if not boolish(row.get("accepted")))
    svg_bar(os.path.join(plots, "failure_reason_counts.svg"), "Failure reasons", list(failures) or ["none"], [failures[k] for k in failures] or [0])


def build_parser():
    parser = argparse.ArgumentParser(description="Generate and analyze a multi-Zn C-S-H ensemble.")
    parser.add_argument("--mode", choices=ENSEMBLE_MODES, required=True)
    parser.add_argument("--n-models", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=8100)
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--n-q1", type=int, default=None)
    parser.add_argument("--n-q2b", type=int, default=None)
    parser.add_argument("--q1-fraction", type=float, default=0.5)
    parser.add_argument("--target-zn-si", type=float, default=0.05)
    parser.add_argument("--max-combinations-per-model", type=int, default=10)
    parser.add_argument("--min-zn-zn-distance", type=float, default=5.0)
    parser.add_argument("--run-static-relaxation", action="store_true")
    parser.add_argument("--keep-only-postmin-valid", action="store_true")
    parser.add_argument("--prefer-ideal-fourfold", action="store_true", default=True)
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--out-dir", default=os.path.join("output_Y", "workflow_v1", "multi_zn_ensemble"))
    return parser


def main():
    args = build_parser().parse_args()
    screen.alpha.run_lammps = run_lammps_with_timeout
    seeds = parse_seed_list(args.seed_list)
    if seeds is None:
        seeds = [int(args.seed_start) + i for i in range(int(args.n_models))]
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "models"))
    manifest = {
        "workflow": "v1.7-multi-Zn-ensemble-generator",
        "mode": args.mode,
        "n_models": len(seeds),
        "seeds": seeds,
        "target_Zn_Si": float(args.target_zn_si),
        "max_combinations_per_model": int(args.max_combinations_per_model),
        "min_zn_zn_distance": float(args.min_zn_zn_distance),
        "run_static_relaxation": bool(args.run_static_relaxation),
        "keep_only_postmin_valid": bool(args.keep_only_postmin_valid),
        "prefer_ideal_fourfold": bool(args.prefer_ideal_fourfold),
        "finite_temperature_md": "not run",
        "scope": "multi-Zn ensemble generation and analysis only; not batch mechanics",
    }
    write_json(os.path.join(args.out_dir, "multi_zn_ensemble_manifest.json"), manifest)
    rows = []
    for index, seed in enumerate(seeds, start=1):
        row = screen_model(args, index, seed)
        rows.append(row)
    accepted = [row for row in rows if boolish(row.get("accepted"))]
    rejected = [row for row in rows if not boolish(row.get("accepted"))]
    mechanics_ready = [row for row in accepted if boolish(row.get("mechanics_ready"))]
    survival_rows = write_survival_summary(os.path.join(args.out_dir, "motif_survival_summary.csv"), rows)
    quality_rows = aggregate_counts(rows, "coordination_quality")
    failure_rows = aggregate_counts(rejected, "failure_reason")
    write_named_count_csv(os.path.join(args.out_dir, "coordination_quality_summary.csv"), quality_rows, "coordination_quality")
    write_named_count_csv(os.path.join(args.out_dir, "failure_reason_summary.csv"), failure_rows, "failure_reason")
    representatives = representative_selection(rows)
    write_json(os.path.join(args.out_dir, "representative_multi_zn_models.json"), representatives)
    write_csv(os.path.join(args.out_dir, "multi_zn_ensemble_summary.csv"), rows)
    write_csv(os.path.join(args.out_dir, "accepted_multi_zn_models.csv"), accepted)
    write_csv(os.path.join(args.out_dir, "rejected_multi_zn_models.csv"), rejected)
    write_csv(os.path.join(args.out_dir, "mechanics_ready_multi_zn_models.csv"), mechanics_ready)
    summary = {
        "ok": True,
        "mode": args.mode,
        "n_models": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_mechanics_ready": len(mechanics_ready),
        "coordination_quality_counts": {row["name"]: row["count"] for row in quality_rows},
        "motif_survival": survival_rows,
        "representative_model_ids": {key: value.get("model_id") for key, value in representatives.items()},
        "valid_multi_q1_q2b_present": any(row.get("postmin_validation_label") == "valid_multi_q1_q2b_zn_candidate" for row in accepted),
        "rows": rows,
    }
    write_json(os.path.join(args.out_dir, "multi_zn_ensemble_summary.json"), summary)
    if args.write_plots:
        write_plots(args.out_dir, rows)
    print(json.dumps({
        "ok": True,
        "mode": args.mode,
        "n_models": len(rows),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_mechanics_ready": len(mechanics_ready),
        "summary": os.path.join(args.out_dir, "multi_zn_ensemble_summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
