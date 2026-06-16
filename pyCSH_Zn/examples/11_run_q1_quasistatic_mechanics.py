from __future__ import print_function

import csv
import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

MECHANICS_PATH = os.path.join(SCRIPT_DIR, "07_run_quasistatic_mechanics.py")
spec = importlib.util.spec_from_file_location("q1_mechanics_base", MECHANICS_PATH)
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)


CSV_FIELDS = [
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
    "validation_label",
]

Q1_TARGET = {
    "name": "q1_zn",
    "label": "Q1_Zn",
    "base_dir": os.path.join("output_Y", "workflow_v1", "q1_zn"),
    "base_data": os.path.join("output_Y", "workflow_v1", "q1_zn", "lammps_inputs", "q1_zn_minimized_static.data"),
    "ff": os.path.join("output_Y", "workflow_v1", "q1_zn", "in.CementFF4_Zn"),
    "expected_classes": ("valid_q1_zn_candidate",),
    "expected_zinc_site_type": "Q1_Zn",
    "zinc_summary": os.path.join("output_Y", "workflow_v1", "q1_zn", "zinc_summary.json"),
}


def require_valid_q1_reference():
    if not os.path.exists(Q1_TARGET["base_data"]):
        return {
            "ok": False,
            "reason": "Q1 post-min reference data not found; run examples/09_run_q1_static_relaxation.py first.",
        }
    validation = mechanics.validate(
        Q1_TARGET["base_data"],
        expected_zinc_site_type="Q1_Zn",
        zinc_summary_path=Q1_TARGET["zinc_summary"],
    )
    return {
        "ok": validation["classification"] == "valid_q1_zn_candidate",
        "classification": validation["classification"],
        "validation": validation,
        "reason": None if validation["classification"] == "valid_q1_zn_candidate" else "Q1 post-min reference is not valid_q1_zn_candidate.",
    }


def compact_row(row):
    return {
        "strain": row.get("strain"),
        "actual_strain": row.get("actual_strain"),
        "initial_lx": row.get("initial_lx"),
        "final_lx": row.get("final_lx"),
        "stress_xx_bar": row.get("stress_xx_bar"),
        "stress_xx_GPa": row.get("stress_xx_GPa"),
        "pressure_bar": row.get("pressure_bar"),
        "pressure_GPa": row.get("pressure_GPa"),
        "energy_initial": row.get("energy_initial"),
        "energy_final": row.get("energy_final"),
        "validation_label": row.get("classification"),
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def plot_stress_strain(path, rows):
    mechanics.plot_svg(
        path,
        "Q1_Zn stress_xx vs actual strain",
        [("Q1_Zn", rows)],
        "actual_strain",
        "stress_xx_GPa",
        "Pxx (GPa)",
    )


def main():
    out_dir = os.path.join("output_Y", "workflow_v1", "mechanics_q1_zn")
    mechanics.ensure_dir(out_dir)
    report = {
        "workflow": "opt-in Q1_Zn quasi-static mechanics smoke test",
        "scope": "Q1_Zn quasi-static input validation only; not final elastic constants or production mechanical properties",
        "finite_temperature_md": "not run",
        "strain_cases": [{"case": name, "strain": strain} for name, strain in mechanics.STRAIN_CASES],
        "q1_added_to_default_mechanics": False,
        "reference": Q1_TARGET["base_data"],
    }
    reference = require_valid_q1_reference()
    report["reference_validation"] = reference
    lmp = mechanics.find_lammps()
    report["lammps_executable"] = lmp
    if not reference["ok"]:
        report["ok"] = False
        report["reason"] = reference["reason"]
    elif not lmp:
        report["ok"] = False
        report["reason"] = "No LAMMPS executable found. Set LAMMPS_EXE or add lmp to PATH."
    else:
        rows = []
        cases = []
        for case_name, strain in mechanics.STRAIN_CASES:
            result = mechanics.run_case(lmp, Q1_TARGET, case_name, strain, out_dir)
            cases.append(mechanics.case_report(result))
            rows.append(result["row"])
        compact_rows = [compact_row(row) for row in rows]
        csv_path = os.path.join(out_dir, "mechanics_summary_q1_zn.csv")
        json_path = os.path.join(out_dir, "mechanics_summary_q1_zn.json")
        plot_path = os.path.join(out_dir, "q1_zn_stress_strain.svg")
        write_csv(csv_path, compact_rows)
        plot_stress_strain(plot_path, rows)
        report["cases"] = cases
        report["summary_csv"] = csv_path
        report["plot"] = plot_path
        report["rows"] = compact_rows
        report["ok"] = all(case["ok"] for case in cases)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps({"ok": report["ok"], "out": json_path}, indent=2))
        return
    json_path = os.path.join(out_dir, "mechanics_summary_q1_zn.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"ok": report["ok"], "out": json_path, "reason": report.get("reason")}, indent=2))


if __name__ == "__main__":
    main()
