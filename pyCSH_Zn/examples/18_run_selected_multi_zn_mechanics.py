from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

MECH07_PATH = os.path.join(SCRIPT_DIR, "07_run_quasistatic_mechanics.py")
spec = importlib.util.spec_from_file_location("mechanics_base", MECH07_PATH)
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)

try:
    from postprocess.analyze_structure import analyze as analyze_structure
except Exception:
    analyze_structure = None


VALID_MULTI_LABELS = (
    "valid_multi_q1_zn_candidate",
    "valid_multi_q2b_zn_candidate",
    "valid_multi_q1_q2b_zn_candidate",
)
DEFAULT_STRAINS = (-0.003, -0.002, -0.001, 0.001, 0.002, 0.003)
CSV_FIELDS = [
    "model_id",
    "seed",
    "requested_mode",
    "motif_class",
    "n_q1_actual",
    "n_q2b_actual",
    "n_Zn_total",
    "coordination_quality_reference",
    "per_center_coordination_reference",
    "strain",
    "actual_strain",
    "initial_lx",
    "final_lx",
    "stress_xx_bar",
    "stress_xx_GPa",
    "pressure_bar",
    "pressure_GPa",
    "energy_initial",
    "energy_final",
    "validation_label_after_strain",
    "per_center_coordination_after_strain",
    "coordination_quality_after_strain",
    "case_ok",
    "failure_reason",
    "model_dir",
    "case_dir",
]
MODEL_FIELDS = [
    "model_id",
    "seed",
    "requested_mode",
    "motif_class",
    "coordination_quality_reference",
    "n_cases",
    "n_ok_cases",
    "model_ok",
    "failure_reason",
    "model_dir",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def resolve_path(path):
    if not path:
        return ""
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


def boolish(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_strains(text):
    if not text:
        return list(DEFAULT_STRAINS)
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def case_name(strain):
    prefix = "x_plus" if strain > 0 else "x_minus"
    return "{}_{:.3f}".format(prefix, abs(float(strain)))


def coordination_values(validation):
    return [
        int(site.get("coordination_2p5", 0))
        for site in validation.get("zinc", {}).get("zinc_sites", []) or []
    ]


def coordination_quality(values):
    vals = [int(v) for v in values]
    if not vals or any(v < 4 for v in vals):
        return "undercoordinated_failed"
    if all(v == 4 for v in vals):
        return "ideal_fourfold"
    if any(v > 4 for v in vals):
        return "overcoordinated"
    return "minimum_valid"


def motif_class(row):
    label = row.get("postmin_validation_label", "")
    if label == "valid_multi_q1_zn_candidate":
        return "multi_q1"
    if label == "valid_multi_q2b_zn_candidate":
        return "multi_q2b"
    if label == "valid_multi_q1_q2b_zn_candidate":
        return "q1_q2b_single_structure_mixture"
    return row.get("internal_mode") or row.get("requested_mode") or "unknown"


def preflight_model(row, require_ideal, include_overcoordinated):
    label = row.get("postmin_validation_label", "")
    quality = row.get("coordination_quality", "")
    if not boolish(row.get("postmin_valid")):
        return False, "postmin_valid is false"
    if label not in VALID_MULTI_LABELS:
        return False, "post-min validation label is {}".format(label)
    if quality == "undercoordinated_failed":
        return False, "reference is undercoordinated_failed"
    if require_ideal and quality != "ideal_fourfold":
        return False, "require_ideal_fourfold excludes {}".format(quality)
    if quality == "overcoordinated" and not include_overcoordinated and not require_ideal:
        return False, "overcoordinated model requires --include-overcoordinated"
    model_dir = resolve_path(row.get("model_dir"))
    postmin = resolve_path(row.get("postmin_data_path"))
    if not os.path.isdir(model_dir):
        return False, "model_dir does not exist"
    if not os.path.exists(postmin):
        return False, "post-min data path does not exist"
    validation_path = os.path.join(model_dir, "validation_postmin.json")
    zinc_summary = os.path.join(model_dir, "multi_zinc_summary.json")
    if not os.path.exists(validation_path):
        return False, "validation_postmin.json not found"
    validation = json.load(open(validation_path))
    coords = coordination_values(validation)
    if validation.get("classification") not in VALID_MULTI_LABELS:
        return False, "reference validation JSON label is {}".format(validation.get("classification"))
    if not coords or min(coords) < 4:
        return False, "reference has Zn coordination below 4"
    if not os.path.exists(zinc_summary):
        return False, "multi_zinc_summary.json not found"
    return True, ""


def run_case(lmp, model, strain, out_model_dir):
    cname = case_name(strain)
    case_dir = os.path.join(out_model_dir, "strain_cases", cname)
    ensure_dir(case_dir)
    base_data = model["postmin_data"]
    stripped = os.path.join(case_dir, os.path.basename(base_data).replace(".data", "_no_csinfo.data"))
    mechanics.strip_csinfo(base_data, stripped)
    input_name = "in.{}".format(cname)
    input_path = os.path.join(case_dir, input_name)
    raw_name = mechanics.write_lammps_input(input_path, stripped, model["forcefield"], cname, strain)
    run = mechanics.run_lammps(lmp, case_dir, input_name)
    raw_path = os.path.join(case_dir, raw_name)
    final_data = os.path.join(case_dir, cname + "_deformed_minimized.data")
    failure = ""
    validation = None
    analysis_path = None
    if run["ok"] and os.path.exists(raw_path):
        mechanics.append_csinfo(base_data, raw_path, final_data)
        validation = mechanics.validate(final_data, expected_zinc_site_type="multi_Zn", zinc_summary_path=model["zinc_summary"])
        write_json(os.path.join(case_dir, "validation_postmin_strained.json"), validation)
        if analyze_structure is not None:
            try:
                analysis_dir = os.path.join(case_dir, "postprocess")
                analyze_structure(final_data, analysis_dir)
                src = os.path.join(analysis_dir, "structure_analysis.json")
                if os.path.exists(src):
                    analysis_path = os.path.join(case_dir, "structure_analysis.json")
                    shutil.copyfile(src, analysis_path)
            except Exception as exc:
                analysis_path = "postprocess failed: {}".format(exc)
    else:
        failure = "LAMMPS failed or raw data was not written"
    thermo = mechanics.parse_thermo(run["stdout"])
    base_box = mechanics.parse_data(base_data)["box"]
    final_box = mechanics.parse_data(final_data)["box"] if os.path.exists(final_data) else {}
    initial_lx = base_box.get("lx")
    final_lx = final_box.get("lx")
    actual_strain = final_lx / initial_lx - 1.0 if initial_lx and final_lx else None
    coords = coordination_values(validation) if validation else []
    quality = coordination_quality(coords)
    label = validation.get("classification") if validation else ""
    if validation and label not in VALID_MULTI_LABELS:
        failure = failure or "strained validation label {}".format(label)
    if coords and min(coords) < 4:
        failure = failure or "strained Zn center below 4 coordination"
    case_ok = bool(validation and not failure and label in VALID_MULTI_LABELS and coords and min(coords) >= 4)
    stress_bar = thermo["final"].get("Pxx")
    pressure_bar = thermo["final"].get("Press")
    row = {
        "model_id": model["model_id"],
        "seed": model.get("seed"),
        "requested_mode": model.get("requested_mode"),
        "motif_class": model.get("motif_class"),
        "n_q1_actual": model.get("n_q1_actual"),
        "n_q2b_actual": model.get("n_q2b_actual"),
        "n_Zn_total": model.get("n_Zn_total"),
        "coordination_quality_reference": model.get("coordination_quality"),
        "per_center_coordination_reference": model.get("per_center_coordination"),
        "strain": strain,
        "actual_strain": actual_strain,
        "initial_lx": initial_lx,
        "final_lx": final_lx,
        "stress_xx_bar": stress_bar,
        "stress_xx_GPa": mechanics.divide_or_none(stress_bar, 10000.0),
        "pressure_bar": pressure_bar,
        "pressure_GPa": mechanics.divide_or_none(pressure_bar, 10000.0),
        "energy_initial": thermo["initial"].get("PotEng"),
        "energy_final": thermo["final"].get("PotEng"),
        "validation_label_after_strain": label,
        "per_center_coordination_after_strain": ";".join(str(x) for x in coords),
        "coordination_quality_after_strain": quality,
        "case_ok": case_ok,
        "failure_reason": failure,
        "model_dir": model["source_model_dir"],
        "case_dir": case_dir,
    }
    write_json(os.path.join(case_dir, "case_summary.json"), {
        "row": row,
        "run": run,
        "strained_input": stripped,
        "minimized_data": final_data if os.path.exists(final_data) else None,
        "validation": os.path.join(case_dir, "validation_postmin_strained.json") if validation else None,
        "structure_analysis": analysis_path,
        "same_reference_for_all_strains": base_data,
    })
    return row


def svg_bar(path, title, labels, values):
    mechanics.plot_svg(path, title, [("count", [{"actual_strain": i, "value": v} for i, v in enumerate(values)])], "actual_strain", "value", "count")


def write_plots(out_dir, rows, model_rows):
    plots = os.path.join(out_dir, "plots")
    ensure_dir(plots)
    series = []
    by_model = defaultdict(list)
    by_mode = defaultdict(list)
    by_quality = defaultdict(list)
    for row in rows:
        by_model[row["model_id"]].append(row)
        by_mode[row["motif_class"]].append(row)
        by_quality[row["coordination_quality_reference"]].append(row)
    for model_id, vals in sorted(by_model.items()):
        series.append((model_id, vals))
    mechanics.plot_svg(os.path.join(plots, "stress_strain_all_models.svg"), "Selected multi-Zn stress_xx vs actual strain", series, "actual_strain", "stress_xx_GPa", "Pxx (GPa)")
    mechanics.plot_svg(os.path.join(plots, "stress_strain_by_mode.svg"), "Multi-Zn stress_xx by mode", sorted(by_mode.items()), "actual_strain", "stress_xx_GPa", "Pxx (GPa)")
    mechanics.plot_svg(os.path.join(plots, "stress_strain_by_coordination_quality.svg"), "Multi-Zn stress_xx by coordination quality", sorted(by_quality.items()), "actual_strain", "stress_xx_GPa", "Pxx (GPa)")
    mechanics.plot_svg(os.path.join(plots, "stress_strain_representative_models.svg"), "Representative multi-Zn stress_xx", series[: min(4, len(series))], "actual_strain", "stress_xx_GPa", "Pxx (GPa)")
    failures = Counter(row.get("failure_reason") or "ok" for row in rows if not boolish(row.get("case_ok")))
    labels = list(failures) or ["none"]
    vals = [failures[x] for x in labels] or [0]
    svg_bar(os.path.join(plots, "failed_cases_by_reason.svg"), "Failed cases by reason", labels, vals)


def build_parser():
    parser = argparse.ArgumentParser(description="Run selected multi-Zn quasi-static mechanics.")
    parser.add_argument("--models-csv", default=os.path.join(ROOT, "output_Y", "workflow_v1", "multi_zn_ensemble", "mechanics_ready_multi_zn_models.csv"))
    parser.add_argument("--output-dir", default=os.path.join("output_Y", "workflow_v1", "selected_multi_zn_mechanics"))
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--mode-filter", default=None)
    parser.add_argument("--require-ideal-fourfold", action="store_true")
    parser.add_argument("--include-overcoordinated", action="store_true")
    parser.add_argument("--strain-values", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-plots", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    models_csv = resolve_path(args.models_csv)
    out_dir = args.output_dir
    ensure_dir(out_dir)
    lmp = mechanics.find_lammps()
    strains = parse_strains(args.strain_values)
    manifest = {
        "workflow": "v1.8-selected-multi-Zn-batch-mechanics",
        "models_csv": models_csv,
        "output_dir": out_dir,
        "strain_values": strains,
        "require_ideal_fourfold": bool(args.require_ideal_fourfold),
        "include_overcoordinated": bool(args.include_overcoordinated),
        "finite_temperature_md": "not run",
        "scope": "selected multi-Zn quasi-static mechanics pipeline only; not final elastic constants or production mechanical properties",
        "lammps_executable": lmp,
    }
    write_json(os.path.join(out_dir, "batch_multi_zn_mechanics_manifest.json"), manifest)
    source_rows = read_csv(models_csv)
    selected = []
    skipped = []
    for row in source_rows:
        if args.mode_filter and row.get("internal_mode") != args.mode_filter and row.get("requested_mode") != args.mode_filter:
            continue
        ok, reason = preflight_model(row, args.require_ideal_fourfold, args.include_overcoordinated)
        if not ok:
            skipped.append({"model_id": row.get("model_id"), "reason": reason})
            continue
        selected.append(row)
        if args.max_models and len(selected) >= args.max_models:
            break
    all_rows = []
    model_summaries = []
    if not lmp:
        skipped.append({"model_id": "all", "reason": "LAMMPS executable not found"})
    for row in selected if lmp else []:
        mid = row["model_id"]
        out_model_dir = os.path.join(out_dir, "models", mid)
        if os.path.isdir(out_model_dir) and not args.skip_existing:
            shutil.rmtree(out_model_dir)
        ensure_dir(out_model_dir)
        model = dict(row)
        model["source_model_dir"] = resolve_path(row.get("model_dir"))
        model["postmin_data"] = resolve_path(row.get("postmin_data_path"))
        model["forcefield"] = os.path.join(model["source_model_dir"], "in.CementFF4_Zn")
        model["zinc_summary"] = os.path.join(model["source_model_dir"], "multi_zinc_summary.json")
        model["motif_class"] = motif_class(row)
        case_rows = []
        for strain in strains:
            try:
                case_rows.append(run_case(lmp, model, strain, out_model_dir))
            except Exception as exc:
                case_rows.append({
                    "model_id": mid,
                    "seed": row.get("seed"),
                    "requested_mode": row.get("requested_mode"),
                    "motif_class": model["motif_class"],
                    "n_q1_actual": row.get("n_q1_actual"),
                    "n_q2b_actual": row.get("n_q2b_actual"),
                    "n_Zn_total": row.get("n_Zn_total"),
                    "coordination_quality_reference": row.get("coordination_quality"),
                    "per_center_coordination_reference": row.get("per_center_coordination"),
                    "strain": strain,
                    "case_ok": False,
                    "failure_reason": "{}: {}".format(type(exc).__name__, exc),
                    "model_dir": model["source_model_dir"],
                    "case_dir": os.path.join(out_model_dir, "strain_cases", case_name(strain)),
                })
        all_rows.extend(case_rows)
        ok_cases = [case for case in case_rows if boolish(case.get("case_ok"))]
        model_ok = len(ok_cases) == len(strains)
        model_summary = {
            "model_id": mid,
            "seed": row.get("seed"),
            "requested_mode": row.get("requested_mode"),
            "motif_class": model["motif_class"],
            "coordination_quality_reference": row.get("coordination_quality"),
            "n_cases": len(case_rows),
            "n_ok_cases": len(ok_cases),
            "model_ok": model_ok,
            "failure_reason": "" if model_ok else "one or more strain cases failed",
            "model_dir": model["source_model_dir"],
        }
        model_summaries.append(model_summary)
        write_csv(os.path.join(out_model_dir, "model_mechanics_summary.csv"), case_rows, CSV_FIELDS)
        write_json(os.path.join(out_model_dir, "model_mechanics_summary.json"), {"model": model_summary, "cases": case_rows})
        mechanics.plot_svg(os.path.join(out_model_dir, "stress_strain.svg"), "{} stress_xx".format(mid), [(mid, case_rows)], "actual_strain", "stress_xx_GPa", "Pxx (GPa)")
    accepted_models = [row for row in model_summaries if boolish(row.get("model_ok"))]
    failed_models = [row for row in model_summaries if not boolish(row.get("model_ok"))]
    write_csv(os.path.join(out_dir, "batch_multi_zn_mechanics_summary.csv"), all_rows, CSV_FIELDS)
    write_csv(os.path.join(out_dir, "accepted_multi_zn_mechanics_models.csv"), accepted_models, MODEL_FIELDS)
    write_csv(os.path.join(out_dir, "failed_multi_zn_mechanics_models.csv"), failed_models, MODEL_FIELDS)
    write_csv(os.path.join(out_dir, "stress_strain_summary_by_model.csv"), all_rows, CSV_FIELDS)
    write_csv(os.path.join(out_dir, "stress_strain_summary_by_mode.csv"), all_rows, CSV_FIELDS)
    failure_counts = [{"failure_reason": k, "count": v} for k, v in Counter(row.get("failure_reason") or "ok" for row in all_rows if not boolish(row.get("case_ok"))).items()]
    write_csv(os.path.join(out_dir, "mechanics_failure_reason_summary.csv"), failure_counts, ["failure_reason", "count"])
    summary = {
        "ok": True,
        "n_source_models": len(source_rows),
        "n_selected_models": len(selected),
        "n_accepted_mechanics_models": len(accepted_models),
        "n_failed_mechanics_models": len(failed_models),
        "n_cases": len(all_rows),
        "n_ok_cases": sum(1 for row in all_rows if boolish(row.get("case_ok"))),
        "skipped_models": skipped,
        "accepted_model_ids": [row["model_id"] for row in accepted_models],
        "failed_model_ids": [row["model_id"] for row in failed_models],
        "scope": manifest["scope"],
    }
    write_json(os.path.join(out_dir, "batch_multi_zn_mechanics_summary.json"), summary)
    if args.write_plots:
        write_plots(out_dir, all_rows, model_summaries)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
