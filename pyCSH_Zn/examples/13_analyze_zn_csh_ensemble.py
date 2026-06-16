from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)


VALID_LABELS = ("valid_q1_zn_candidate", "valid_q2b_zn_candidate")
SUMMARY_FIELDS = [
    "metric",
    "value",
]
MOTIF_FIELDS = [
    "motif",
    "requested_count",
    "accepted_count",
    "rejected_count",
    "postmin_survival_rate",
]
FAILURE_FIELDS = [
    "failure_reason",
    "failure_stage",
    "count",
]
MECHANICS_FIELDS = [
    "rank",
    "model_id",
    "seed",
    "motif_type",
    "model_directory",
    "postmin_data_path",
    "postmin_validation_label",
    "Zn_O_coordination_2p5A",
    "max_Zn_O_distance",
    "mean_Zn_O_distance",
    "charge_residual",
    "selection_score",
    "reason_selected",
]


def resolve_path(path):
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(REPO_ROOT, path)),
        os.path.abspath(os.path.join(ROOT, path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def as_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_coord(value):
    parts = [x for x in str(value or "").replace(",", ";").split(";") if x.strip()]
    vals = []
    for item in parts:
        try:
            vals.append(float(item))
        except ValueError:
            pass
    return vals


def motif_type(row):
    label = row.get("postmin_validation_label") or row.get("initial_validation_label") or ""
    if "q1" in label.lower():
        return "Q1_Zn"
    if "q2b" in label.lower():
        return "Q2b_Zn"
    if int(as_float(row.get("actual_Q1_count"), 0) or 0) > 0:
        return "Q1_Zn"
    if int(as_float(row.get("actual_Q2b_count"), 0) or 0) > 0:
        return "Q2b_Zn"
    return "unknown"


def stats(values):
    vals = [float(x) for x in values if x is not None]
    if not vals:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(vals),
        "min": min(vals),
        "mean": sum(vals) / len(vals),
        "max": max(vals),
    }


def classify_failure(row):
    reason = (row.get("failure_reason") or "").strip()
    if "LAMMPS" in reason or "return code" in reason or "executable" in reason or "in." in reason:
        return "LAMMPS failure"
    if not row.get("initial_validation_label") or "Error:" in reason or "ValueError:" in reason:
        return "generation failure"
    if not as_bool(row.get("initial_ok")):
        return "initial_validation failure"
    if not as_bool(row.get("postmin_ok")):
        return "postmin_validation failure"
    return "other"


def find_postmin_data(ensemble_dir, row):
    model_dir = os.path.join(ensemble_dir, "models", row.get("model_id", ""))
    input_dir = os.path.join(model_dir, "lammps_inputs")
    if not os.path.isdir(input_dir):
        return None
    suffix = "_minimized_static.data"
    candidates = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith(suffix) and not name.endswith(".raw.data")
    ]
    return sorted(candidates)[0] if candidates else None


def normalized_path(path):
    if not path:
        return None
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return path


def row_score(row):
    coord_vals = parse_coord(row.get("Zn_O_coordination_2p5A"))
    coord = min(coord_vals) if coord_vals else 0.0
    max_dist = as_float(row.get("max_Zn_O_distance"), 999.0)
    mean_dist = as_float(row.get("mean_Zn_O_distance"), 999.0)
    charge = abs(as_float(row.get("charge_residual"), 999.0))
    target = 1.95
    return (
        1 if as_bool(row.get("postmin_ok")) else 0,
        coord,
        -max_dist,
        -abs(mean_dist - target),
        -charge,
    )


def selection_score(row):
    tup = row_score(row)
    return 1000.0 * tup[0] + 100.0 * tup[1] + 10.0 * tup[2] + tup[3] + tup[4]


def mechanics_ready(row, min_coordination, max_zn_o):
    if not as_bool(row.get("postmin_ok")):
        return False
    if row.get("postmin_validation_label") not in VALID_LABELS:
        return False
    coord_vals = parse_coord(row.get("Zn_O_coordination_2p5A"))
    if not coord_vals or min(coord_vals) < float(min_coordination):
        return False
    max_dist = as_float(row.get("max_Zn_O_distance"), 999.0)
    return max_dist <= float(max_zn_o)


def select_representatives(rows, ensemble_dir, top_n, min_coordination, max_zn_o, prefer_balanced):
    ready = [row for row in rows if mechanics_ready(row, min_coordination, max_zn_o)]
    ready = sorted(ready, key=row_score, reverse=True)
    selected = []
    if prefer_balanced:
        for motif in ("Q1_Zn", "Q2b_Zn"):
            for row in ready:
                if motif_type(row) == motif and row not in selected:
                    selected.append(row)
                    break
                if len(selected) >= top_n:
                    break
    for row in ready:
        if len(selected) >= top_n:
            break
        if row not in selected:
            selected.append(row)
    records = []
    for rank, row in enumerate(selected[:top_n], start=1):
        postmin_data = find_postmin_data(ensemble_dir, row)
        motif = motif_type(row)
        reason = "mechanics-ready post-min valid {}; ranked by coordination, Zn-O distance, and charge residual".format(motif)
        if prefer_balanced:
            reason += "; balanced Q1/Q2b preference enabled"
        records.append({
            "rank": rank,
            "model_id": row.get("model_id"),
            "seed": row.get("seed"),
            "motif_type": motif,
            "model_directory": normalized_path(os.path.join(ensemble_dir, "models", row.get("model_id", ""))),
            "postmin_data_path": normalized_path(postmin_data),
            "postmin_validation_label": row.get("postmin_validation_label"),
            "Zn_O_coordination_2p5A": row.get("Zn_O_coordination_2p5A"),
            "max_Zn_O_distance": row.get("max_Zn_O_distance"),
            "mean_Zn_O_distance": row.get("mean_Zn_O_distance"),
            "charge_residual": row.get("charge_residual"),
            "selection_score": selection_score(row),
            "reason_selected": reason,
        })
    return ready, records


def svg_bar(path, title, labels, values, ylabel="count"):
    width = 720
    height = 420
    margin = 56
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_v = max(values) if values else 1.0
    max_v = max(max_v, 1.0)
    bar_w = plot_w / max(len(values), 1) * 0.62
    gap = plot_w / max(len(values), 1)
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
        lines.append('<text x="{:.2f}" y="{}" text-anchor="middle" font-family="Arial" font-size="11">{}</text>'.format(x + bar_w / 2, height - margin + 18, labels[i]))
    lines.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def svg_hist(path, title, values, xlabel, bins=12):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        svg_bar(path, title, ["none"], [0], "count")
        return
    vmin = min(vals)
    vmax = max(vals)
    if abs(vmax - vmin) < 1.0e-12:
        labels = ["{:.3g}".format(vmin)]
        counts = [len(vals)]
    else:
        step = (vmax - vmin) / bins
        counts = [0] * bins
        for value in vals:
            idx = min(bins - 1, int((value - vmin) / step))
            counts[idx] += 1
        labels = ["{:.2f}".format(vmin + (i + 0.5) * step) for i in range(bins)]
    svg_bar(path, title + " (" + xlabel + ")", labels, counts, "count")


def analyze(args):
    ensemble_dir = resolve_path(args.ensemble_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else os.path.join(os.path.dirname(ensemble_dir), "zn_csh_ensemble_analysis")
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    if args.write_plots:
        ensure_dir(plots_dir)

    summary_rows = read_csv(os.path.join(ensemble_dir, "ensemble_summary.csv"))
    accepted_rows = read_csv(os.path.join(ensemble_dir, "accepted_models.csv"))
    rejected_rows = read_csv(os.path.join(ensemble_dir, "rejected_models.csv"))
    ensemble_summary_path = os.path.join(ensemble_dir, "ensemble_summary.json")
    ensemble_summary = {}
    if os.path.exists(ensemble_summary_path):
        with open(ensemble_summary_path) as f:
            ensemble_summary = json.load(f)

    total = len(summary_rows)
    accepted = len(accepted_rows)
    rejected = len(rejected_rows)
    motif_counts = {}
    for motif in ("Q1_Zn", "Q2b_Zn", "unknown"):
        requested = [row for row in summary_rows if motif_type(row) == motif]
        acc = [row for row in accepted_rows if motif_type(row) == motif]
        rej = [row for row in rejected_rows if motif_type(row) == motif]
        if motif == "unknown" and not requested:
            continue
        motif_counts[motif] = {
            "motif": motif,
            "requested_count": len(requested),
            "accepted_count": len(acc),
            "rejected_count": len(rej),
            "postmin_survival_rate": (len(acc) / len(requested)) if requested else None,
        }
    motif_rows = list(motif_counts.values())

    max_dist = [as_float(row.get("max_Zn_O_distance")) for row in accepted_rows]
    mean_dist = [as_float(row.get("mean_Zn_O_distance")) for row in accepted_rows]
    min_dist = [as_float(row.get("min_Zn_O_distance")) for row in accepted_rows]
    charge = [as_float(row.get("charge_residual")) for row in accepted_rows]
    actual_ratio = [as_float(row.get("actual_Zn_Si")) for row in accepted_rows]
    coords = []
    for row in accepted_rows:
        coords.extend(parse_coord(row.get("Zn_O_coordination_2p5A")))

    failure_counter = Counter()
    failure_stage_counter = Counter()
    failure_rows = []
    for row in rejected_rows:
        reason = (row.get("failure_reason") or "unspecified").strip() or "unspecified"
        stage = classify_failure(row)
        failure_counter[(reason, stage)] += 1
        failure_stage_counter[stage] += 1
    for (reason, stage), count in sorted(failure_counter.items(), key=lambda item: (-item[1], item[0][1], item[0][0])):
        failure_rows.append({"failure_reason": reason, "failure_stage": stage, "count": count})

    ready_rows, representative = select_representatives(
        accepted_rows,
        ensemble_dir,
        args.top_n,
        args.min_coordination,
        args.max_zn_o,
        args.prefer_balanced_q1_q2b,
    )
    ready_records = []
    for rank, row in enumerate(ready_rows, start=1):
        postmin_data = find_postmin_data(ensemble_dir, row)
        ready_records.append({
            "rank": rank,
            "model_id": row.get("model_id"),
            "seed": row.get("seed"),
            "motif_type": motif_type(row),
            "model_directory": normalized_path(os.path.join(ensemble_dir, "models", row.get("model_id", ""))),
            "postmin_data_path": normalized_path(postmin_data),
            "postmin_validation_label": row.get("postmin_validation_label"),
            "Zn_O_coordination_2p5A": row.get("Zn_O_coordination_2p5A"),
            "max_Zn_O_distance": row.get("max_Zn_O_distance"),
            "mean_Zn_O_distance": row.get("mean_Zn_O_distance"),
            "charge_residual": row.get("charge_residual"),
            "selection_score": selection_score(row),
            "reason_selected": "mechanics-ready candidate; not production mechanics",
        })

    analysis = {
        "ok": True,
        "workflow": "v1.5-ensemble-analysis-and-selection",
        "ensemble_dir": normalized_path(ensemble_dir),
        "source_ensemble_summary": ensemble_summary,
        "n_total": total,
        "n_accepted": accepted,
        "n_rejected": rejected,
        "accepted_fraction": (accepted / total) if total else None,
        "rejected_fraction": (rejected / total) if total else None,
        "motif_survival": motif_counts,
        "accepted_model_statistics": {
            "Zn_O_coordination_2p5A": stats(coords),
            "min_Zn_O_distance": stats(min_dist),
            "mean_Zn_O_distance": stats(mean_dist),
            "max_Zn_O_distance": stats(max_dist),
            "charge_residual": stats(charge),
            "actual_Zn_Si": stats(actual_ratio),
            "empty_failure_reason_count": sum(1 for row in accepted_rows if not (row.get("failure_reason") or "").strip()),
        },
        "failure_stage_counts": dict(failure_stage_counter),
        "mechanics_ready_count": len(ready_records),
        "representative_count": len(representative),
        "representative_model_ids": [row["model_id"] for row in representative],
        "scope": "ensemble statistics and representative selection only; mechanics_ready_models.csv is an input manifest for later opt-in mechanics",
        "non_scope": [
            "finite-temperature MD",
            "final elastic constants",
            "production mechanical properties",
            "single-structure mixed Q1+Q2b",
            "single-structure multi-Zn",
        ],
    }

    summary_table = [
        {"metric": "n_total", "value": total},
        {"metric": "n_accepted", "value": accepted},
        {"metric": "n_rejected", "value": rejected},
        {"metric": "accepted_fraction", "value": analysis["accepted_fraction"]},
        {"metric": "rejected_fraction", "value": analysis["rejected_fraction"]},
        {"metric": "mechanics_ready_count", "value": len(ready_records)},
        {"metric": "representative_count", "value": len(representative)},
    ]
    write_json(os.path.join(out_dir, "ensemble_analysis_summary.json"), analysis)
    write_csv(os.path.join(out_dir, "ensemble_analysis_summary.csv"), summary_table, SUMMARY_FIELDS)
    write_csv(os.path.join(out_dir, "motif_survival_summary.csv"), motif_rows, MOTIF_FIELDS)
    write_csv(os.path.join(out_dir, "failure_reason_summary.csv"), failure_rows, FAILURE_FIELDS)
    write_csv(os.path.join(out_dir, "mechanics_ready_models.csv"), ready_records, MECHANICS_FIELDS)
    write_json(os.path.join(out_dir, "representative_models.json"), {"models": representative})
    manifest = {
        "workflow": "v1.5-ensemble-analysis-and-selection",
        "ensemble_dir": normalized_path(ensemble_dir),
        "out_dir": normalized_path(out_dir),
        "top_n": int(args.top_n),
        "select_for_mechanics": bool(args.select_for_mechanics),
        "min_coordination": float(args.min_coordination),
        "max_zn_o": float(args.max_zn_o),
        "prefer_balanced_q1_q2b": bool(args.prefer_balanced_q1_q2b),
        "write_plots": bool(args.write_plots),
    }
    write_json(os.path.join(out_dir, "analysis_manifest.json"), manifest)

    if args.write_plots:
        svg_bar(
            os.path.join(plots_dir, "accepted_rejected_counts.svg"),
            "Accepted vs rejected models",
            ["accepted", "rejected"],
            [accepted, rejected],
        )
        svg_bar(
            os.path.join(plots_dir, "survival_rate_by_motif.svg"),
            "Post-min survival rate by motif",
            [row["motif"] for row in motif_rows],
            [round((row["postmin_survival_rate"] or 0.0) * 100.0, 3) for row in motif_rows],
            "survival (%)",
        )
        svg_hist(os.path.join(plots_dir, "zn_o_distance_distribution.svg"), "Accepted Zn-O max distance", max_dist, "Angstrom")
        svg_hist(os.path.join(plots_dir, "coordination_distribution.svg"), "Accepted Zn-O coordination", coords, "coordination")
        svg_bar(
            os.path.join(plots_dir, "failure_reason_counts.svg"),
            "Failure reason counts",
            [row["failure_stage"] for row in failure_rows],
            [int(row["count"]) for row in failure_rows],
        )

    return {
        "ok": True,
        "out": os.path.join(out_dir, "ensemble_analysis_summary.json"),
        "n_accepted": accepted,
        "n_rejected": rejected,
        "mechanics_ready_count": len(ready_records),
        "representative_model_ids": [row["model_id"] for row in representative],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze and select representative models from a v1.4 Zn-C-S-H ensemble.")
    parser.add_argument("--ensemble-dir", default=os.path.join(ROOT, "output_Y", "workflow_v1", "zn_csh_ensemble"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--select-for-mechanics", action="store_true")
    parser.add_argument("--min-coordination", type=float, default=4.0)
    parser.add_argument("--max-zn-o", type=float, default=2.5)
    parser.add_argument("--prefer-balanced-q1-q2b", action="store_true")
    parser.add_argument("--write-plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
