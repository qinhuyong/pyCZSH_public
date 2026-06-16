from __future__ import print_function

import csv
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from validate_cementff_data import audit_csinfo, audit_water, audit_zinc, parse_data, validate


STRAIN_CASES = [
    ("x_minus_0.003", -0.003),
    ("x_minus_0.002", -0.002),
    ("x_minus_0.001", -0.001),
    ("x_plus_0.001", 0.001),
    ("x_plus_0.002", 0.002),
    ("x_plus_0.003", 0.003),
]

TARGETS = [
    {
        "name": "pure_csh",
        "label": "Pure C-S-H",
        "base_dir": os.path.join("output_Y", "workflow_v1", "pure_csh"),
        "base_data": os.path.join("output_Y", "workflow_v1", "pure_csh", "lammps_inputs", "pure_csh_minimized_static.data"),
        "ff": os.path.join("output_Y", "workflow_v1", "pure_csh", "in.CementFF4_Zn"),
        "expected_classes": ("valid_static_candidate",),
    },
    {
        "name": "q2b_zn",
        "label": "Q2b_Zn",
        "base_dir": os.path.join("output_Y", "workflow_v1", "q2b_zn"),
        "base_data": os.path.join("output_Y", "workflow_v1", "q2b_zn", "lammps_inputs", "q2b_zn_minimized_static.data"),
        "ff": os.path.join("output_Y", "workflow_v1", "q2b_zn", "in.CementFF4_Zn"),
        "expected_classes": ("valid_q2b_zn_candidate", "needs_static_relaxation"),
    },
    {
        "name": "q1_zn",
        "label": "Q1_Zn",
        "base_dir": os.path.join("output_Y", "workflow_v1", "q1_zn"),
        "base_data": os.path.join("output_Y", "workflow_v1", "q1_zn", "lammps_inputs", "q1_zn_minimized_static.data"),
        "ff": os.path.join("output_Y", "workflow_v1", "q1_zn", "in.CementFF4_Zn"),
        "expected_classes": ("valid_q1_zn_candidate",),
    },
]

CSV_FIELDS = [
    "target",
    "case",
    "strain",
    "actual_strain",
    "initial_lx",
    "final_lx",
    "energy_initial",
    "energy_final",
    "stress_xx_final",
    "stress_xx_bar",
    "stress_xx_GPa",
    "pressure_final",
    "pressure_bar",
    "pressure_GPa",
    "classification",
    "case_pass",
    "charge_assignment_bad_atoms",
    "csinfo_bad_pairs",
    "core_shell_distance_min",
    "core_shell_distance_mean",
    "core_shell_distance_max",
    "water_bad_count",
    "zn_coordination_2p3",
    "zn_coordination_2p5",
]


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


def relpath(path, start):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start))


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_csinfo(path):
    return parse_data(path)["csinfo"]


def append_csinfo(reference_data, raw_data, output_data):
    csinfo = read_csinfo(reference_data)
    with open(raw_data) as f:
        text = f.read().rstrip()
    with open(output_data, "w") as f:
        f.write(text)
        f.write("\n\nCS-Info\n\n")
        for atom_id in sorted(csinfo):
            f.write("{:8d} {:8d}\n".format(atom_id, csinfo[atom_id]))


def strip_csinfo(source_path, output_path):
    with open(source_path) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        header = line.split("#")[0].strip().lower()
        if header in ("cs-info", "csinfo"):
            break
        out.append(line)
    with open(output_path, "w") as f:
        f.writelines(out)


def write_lammps_input(path, base_data, ff_file, case_name, strain):
    run_dir = os.path.dirname(path)
    raw_name = case_name + "_deformed_minimized.raw.data"
    sign = "+" if strain > 0 else "-"
    mag = abs(strain)
    lines = [
        "clear",
        "units metal",
        "dimension 3",
        "atom_style full",
        "boundary p p p",
        "box tilt large",
        "read_data {}".format(relpath(base_data, run_dir)),
        "include {}".format(relpath(ff_file, run_dir)),
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "comm_modify vel yes cutoff 14.0",
        "compute p all pressure NULL virial",
        "thermo 1",
        "thermo_style custom step pe pxx pyy pzz pxy pxz pyz press fnorm fmax lx ly lz",
        "run 0",
        "variable strain equal {:.6f}".format(mag),
        "variable sx equal 1.0{}${{strain}}".format(sign),
        "change_box all x scale ${sx} remap",
        "run 0",
        "min_style cg",
        "min_modify dmax 0.002 line quadratic",
        "minimize 1e-6 1e-8 1000 10000",
        "run 0",
        "write_data {} nocoeff".format(raw_name),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return raw_name


def run_lammps(lmp, run_dir, input_name):
    stdout_path = os.path.join(run_dir, input_name + ".stdout.txt")
    stderr_path = os.path.join(run_dir, input_name + ".stderr.txt")
    proc = subprocess.run(
        [lmp, "-in", input_name],
        cwd=run_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    with open(stdout_path, "w") as f:
        f.write(proc.stdout)
    with open(stderr_path, "w") as f:
        f.write(proc.stderr)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout_path,
        "stderr": stderr_path,
    }


def parse_thermo(stdout_path):
    headers = None
    rows = []
    number = re.compile(r"^[\s+-]?\d")
    with open(stdout_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            if parts[0] == "Step":
                headers = parts
                continue
            if headers and len(parts) == len(headers) and number.match(parts[0]):
                try:
                    rows.append({headers[i]: float(parts[i]) for i in range(len(headers))})
                except ValueError:
                    pass
    if not rows:
        return {"rows": [], "initial": {}, "final": {}}
    return {"rows": rows, "initial": rows[0], "final": rows[-1]}


def summarize_values(values):
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def divide_or_none(value, divisor):
    if value is None:
        return None
    return value / divisor


def structure_metrics(data):
    cs = audit_csinfo(data)
    water = audit_water(data)
    zinc = audit_zinc(data)
    core_shell_distances = [rec["distance"] for rec in cs["pairs"]]
    cs_stats = summarize_values(core_shell_distances)
    zn23 = None
    zn25 = None
    if zinc["zinc_sites"]:
        zn23 = zinc["zinc_sites"][0]["coordination_2p3"]
        zn25 = zinc["zinc_sites"][0]["coordination_2p5"]
    return {
        "core_shell_distance_min": cs_stats["min"],
        "core_shell_distance_mean": cs_stats["mean"],
        "core_shell_distance_max": cs_stats["max"],
        "water_bad_count": water["n_bad_water"],
        "zn_coordination_2p3": zn23,
        "zn_coordination_2p5": zn25,
    }


def q1_mechanics_enabled():
    return os.environ.get("PYCSH_ZN_INCLUDE_Q1_MECHANICS") == "1"


def q1_reference_is_valid(path):
    if not os.path.exists(path):
        return False
    return validate(
        path,
        expected_zinc_site_type="Q1_Zn",
        zinc_summary_path=os.path.join("output_Y", "workflow_v1", "q1_zn", "zinc_summary.json"),
    )["classification"] == "valid_q1_zn_candidate"


def severe_failures(validation, metrics, target_name):
    if validation["charge_assignment"]["n_bad"]:
        return "charge assignment failure"
    if validation["csinfo"]["n_bad_pairs"]:
        return "bad core-shell pair"
    if validation["water"]["n_bad_water"]:
        return "water topology failure"
    if target_name == "q2b_zn":
        if metrics["zn_coordination_2p5"] is None or metrics["zn_coordination_2p5"] < 4:
            return "severe Zn coordination collapse"
    if target_name == "q1_zn":
        if metrics["zn_coordination_2p5"] is None or metrics["zn_coordination_2p5"] < 4:
            return "severe Zn coordination collapse"
    return None


def run_case(lmp, target, case_name, strain, mechanics_dir):
    target_dir = os.path.join(mechanics_dir, target["name"])
    run_dir = os.path.join(target_dir, case_name)
    ensure_dir(run_dir)
    stripped_base = os.path.join(run_dir, os.path.basename(target["base_data"]).replace(".data", "_no_csinfo.data"))
    strip_csinfo(target["base_data"], stripped_base)
    input_name = "in.{}".format(case_name)
    input_path = os.path.join(run_dir, input_name)
    raw_name = write_lammps_input(input_path, stripped_base, target["ff"], case_name, strain)
    run = run_lammps(lmp, run_dir, input_name)
    raw_path = os.path.join(run_dir, raw_name)
    final_data = os.path.join(run_dir, case_name + "_deformed_minimized.data")
    validation = None
    metrics = {}
    fail_reason = None
    if run["ok"] and os.path.exists(raw_path):
        append_csinfo(target["base_data"], raw_path, final_data)
        validation = validate(
            final_data,
            expected_zinc_site_type=target["expected_zinc_site_type"] if "expected_zinc_site_type" in target else None,
            zinc_summary_path=target.get("zinc_summary"),
        )
        validation_path = os.path.splitext(final_data)[0] + "_validation.json"
        with open(validation_path, "w") as f:
            json.dump(validation, f, indent=2, sort_keys=True)
            f.write("\n")
        metrics = structure_metrics(parse_data(final_data))
        fail_reason = severe_failures(validation, metrics, target["name"])
        classification_ok = validation["classification"] in target["expected_classes"]
        case_ok = classification_ok and fail_reason is None
    else:
        validation_path = None
        classification_ok = False
        case_ok = False
        fail_reason = "LAMMPS failed or raw data was not written"
    thermo = parse_thermo(run["stdout"])
    base_box = parse_data(target["base_data"])["box"]
    final_box = parse_data(final_data)["box"] if os.path.exists(final_data) else {}
    initial_lx = base_box.get("lx")
    final_lx = final_box.get("lx")
    actual_strain = final_lx / initial_lx - 1.0 if initial_lx and final_lx else None
    stress_xx_bar = thermo["final"].get("Pxx")
    pressure_bar = thermo["final"].get("Press")
    row = {
        "target": target["name"],
        "case": case_name,
        "strain": strain,
        "actual_strain": actual_strain,
        "initial_lx": initial_lx,
        "final_lx": final_lx,
        "energy_initial": thermo["initial"].get("PotEng"),
        "energy_final": thermo["final"].get("PotEng"),
        "stress_xx_final": stress_xx_bar,
        "stress_xx_bar": stress_xx_bar,
        "stress_xx_GPa": divide_or_none(stress_xx_bar, 10000.0),
        "pressure_final": pressure_bar,
        "pressure_bar": pressure_bar,
        "pressure_GPa": divide_or_none(pressure_bar, 10000.0),
        "classification": validation["classification"] if validation else None,
        "case_pass": case_ok,
        "charge_assignment_bad_atoms": validation["charge_assignment"]["n_bad"] if validation else None,
        "csinfo_bad_pairs": validation["csinfo"]["n_bad_pairs"] if validation else None,
        "core_shell_distance_min": metrics.get("core_shell_distance_min"),
        "core_shell_distance_mean": metrics.get("core_shell_distance_mean"),
        "core_shell_distance_max": metrics.get("core_shell_distance_max"),
        "water_bad_count": metrics.get("water_bad_count"),
        "zn_coordination_2p3": metrics.get("zn_coordination_2p3"),
        "zn_coordination_2p5": metrics.get("zn_coordination_2p5"),
    }
    return {
        "ok": case_ok,
        "fail_reason": fail_reason,
        "case": case_name,
        "strain": strain,
        "input": input_path,
        "run": run,
        "raw_data": raw_path,
        "data": final_data if os.path.exists(final_data) else None,
        "validation": validation_path,
        "thermo": thermo,
        "metrics": metrics,
        "row": row,
    }


def case_report(case_result):
    thermo = case_result["thermo"]
    row = case_result["row"]
    return {
        "case": case_result["case"],
        "strain": case_result["strain"],
        "ok": case_result["ok"],
        "fail_reason": case_result["fail_reason"],
        "classification": row["classification"],
        "input": case_result["input"],
        "raw_data": case_result["raw_data"],
        "data": case_result["data"],
        "validation": case_result["validation"],
        "run": case_result["run"],
        "thermo": {
            "initial": thermo["initial"],
            "final": thermo["final"],
            "n_rows": len(thermo["rows"]),
        },
        "mechanics": {
            "prescribed_strain": row["strain"],
            "actual_strain": row["actual_strain"],
            "initial_lx": row["initial_lx"],
            "final_lx": row["final_lx"],
            "energy_initial": row["energy_initial"],
            "energy_final": row["energy_final"],
            "stress_xx_final": row["stress_xx_final"],
            "stress_xx_bar": row["stress_xx_bar"],
            "stress_xx_GPa": row["stress_xx_GPa"],
            "pressure_final": row["pressure_final"],
            "pressure_bar": row["pressure_bar"],
            "pressure_GPa": row["pressure_GPa"],
        },
        "validation_metrics": {
            "charge_assignment_bad_atoms": row["charge_assignment_bad_atoms"],
            "csinfo_bad_pairs": row["csinfo_bad_pairs"],
            "water_bad_count": row["water_bad_count"],
            "core_shell_distance_min": row["core_shell_distance_min"],
            "core_shell_distance_mean": row["core_shell_distance_mean"],
            "core_shell_distance_max": row["core_shell_distance_max"],
            "zn_coordination_2p3": row["zn_coordination_2p3"],
            "zn_coordination_2p5": row["zn_coordination_2p5"],
        },
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def plot_svg(path, title, series, x_key, y_key, y_label):
    width = 760
    height = 460
    left = 80
    right = 30
    top = 45
    bottom = 70
    points = []
    for label, rows in series:
        vals = [(float(r[x_key]), r[y_key]) for r in rows if r.get(y_key) not in (None, "")]
        vals = [(x, float(y)) for x, y in vals if y is not None]
        points.extend(vals)
    if not points:
        with open(path, "w") as f:
            f.write("<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}'><text x='20' y='40'>No data for {}</text></svg>\n".format(width, height, title))
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    xpad = (xmax - xmin) * 0.08
    ypad = (ymax - ymin) * 0.08
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y):
        return height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}' viewBox='0 0 {} {}'>".format(width, height, width, height),
        "<rect width='100%' height='100%' fill='white'/>",
        "<text x='{}' y='28' font-family='Arial' font-size='18' font-weight='bold'>{}</text>".format(left, title),
        "<line x1='{0}' y1='{1}' x2='{2}' y2='{1}' stroke='#111'/>".format(left, height - bottom, width - right),
        "<line x1='{0}' y1='{1}' x2='{0}' y2='{2}' stroke='#111'/>".format(left, top, height - bottom),
        "<text x='{}' y='{}' font-family='Arial' font-size='12'>{}</text>".format(width / 2 - 45, height - 25, x_key),
        "<text x='18' y='{}' font-family='Arial' font-size='12' transform='rotate(-90 18,{})'>{}</text>".format(height / 2 + 50, height / 2 + 50, y_label),
    ]
    for idx, (label, rows) in enumerate(series):
        vals = sorted([(float(r[x_key]), r[y_key]) for r in rows if r.get(y_key) not in (None, "")], key=lambda x: x[0])
        vals = [(x, float(y)) for x, y in vals if y is not None]
        if not vals:
            continue
        color = colors[idx % len(colors)]
        poly = " ".join("{:.2f},{:.2f}".format(sx(x), sy(y)) for x, y in vals)
        parts.append("<polyline fill='none' stroke='{}' stroke-width='2' points='{}'/>".format(color, poly))
        for x, y in vals:
            parts.append("<circle cx='{:.2f}' cy='{:.2f}' r='4' fill='{}'/>".format(sx(x), sy(y), color))
        parts.append("<rect x='{}' y='{}' width='12' height='12' fill='{}'/>".format(width - 185, top + idx * 22, color))
        parts.append("<text x='{}' y='{}' font-family='Arial' font-size='12'>{}</text>".format(width - 168, top + 10 + idx * 22, label))
    parts.append("<text x='{}' y='{}' font-family='Arial' font-size='11'>x: {:.4g} to {:.4g}; y: {:.4g} to {:.4g}</text>".format(left, height - 45, xmin, xmax, ymin, ymax))
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
        f.write("\n")


def generate_plots(out_dir, pure_rows, q2b_rows, combined_rows):
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)
    plots = {}
    x_key = "actual_strain"
    q1_rows = [row for row in combined_rows if row["target"] == "q1_zn"]
    plots["energy_pure_csh"] = os.path.join(plots_dir, "energy_vs_strain_pure_csh.svg")
    plot_svg(plots["energy_pure_csh"], "Pure C-S-H energy vs actual strain", [("pure C-S-H", pure_rows)], x_key, "energy_final", "Potential energy")
    plots["energy_q2b_zn"] = os.path.join(plots_dir, "energy_vs_strain_q2b_zn.svg")
    plot_svg(plots["energy_q2b_zn"], "Q2b_Zn energy vs actual strain", [("Q2b_Zn", q2b_rows)], x_key, "energy_final", "Potential energy")
    plots["stress_xx_combined"] = os.path.join(plots_dir, "stress_xx_vs_strain.svg")
    plot_svg(plots["stress_xx_combined"], "stress_xx vs actual strain", [("pure C-S-H", pure_rows), ("Q2b_Zn", q2b_rows), ("Q1_Zn", q1_rows)], x_key, "stress_xx_GPa", "Pxx (GPa)")
    plots["zn_coordination_q2b_zn"] = os.path.join(plots_dir, "zn_o_coordination_vs_strain_q2b_zn.svg")
    plot_svg(plots["zn_coordination_q2b_zn"], "Q2b_Zn Zn-O coordination vs actual strain", [("coordination 2.5 A", q2b_rows)], x_key, "zn_coordination_2p5", "Zn-O coordination")
    if q1_rows:
        plots["energy_q1_zn"] = os.path.join(plots_dir, "energy_vs_strain_q1_zn.svg")
        plot_svg(plots["energy_q1_zn"], "Q1_Zn energy vs actual strain", [("Q1_Zn", q1_rows)], x_key, "energy_final", "Potential energy")
        plots["stress_xx_q1_zn"] = os.path.join(plots_dir, "stress_xx_vs_strain_q1_zn.svg")
        plot_svg(plots["stress_xx_q1_zn"], "Q1_Zn stress_xx vs actual strain", [("Q1_Zn", q1_rows)], x_key, "stress_xx_GPa", "Pxx (GPa)")
        plots["zn_coordination_q1_zn"] = os.path.join(plots_dir, "zn_coordination_vs_strain_q1_zn.svg")
        plot_svg(plots["zn_coordination_q1_zn"], "Q1_Zn Zn-O coordination vs actual strain", [("coordination 2.5 A", q1_rows)], x_key, "zn_coordination_2p5", "Zn-O coordination")
    return plots


def main():
    lmp = find_lammps()
    mechanics_dir = os.path.join("output_Y", "workflow_v1", "quasistatic_mechanics")
    ensure_dir(mechanics_dir)
    report = {
        "workflow": "v1.2-quasistatic-mechanics",
        "finite_temperature_md": "not run",
        "scope": "controlled quasi-static mechanics pipeline validation, not final elastic constants",
        "lammps_executable": lmp,
        "strain_cases": [{"case": name, "strain": strain} for name, strain in STRAIN_CASES],
        "targets": [],
    }
    if not lmp:
        report["ok"] = False
        report["reason"] = "No LAMMPS executable found. Set LAMMPS_EXE or add lmp to PATH."
    else:
        combined_rows = []
        rows_by_target = {}
        active_targets = []
        for target in TARGETS:
            if target["name"] == "q1_zn":
                if not q1_mechanics_enabled() or not q1_reference_is_valid(target["base_data"]):
                    continue
                target = dict(target)
                target["expected_zinc_site_type"] = "Q1_Zn"
                target["zinc_summary"] = os.path.join("output_Y", "workflow_v1", "q1_zn", "zinc_summary.json")
            active_targets.append(target)
        for target in active_targets:
            target_record = {"name": target["name"], "cases": []}
            rows_by_target[target["name"]] = []
            for case_name, strain in STRAIN_CASES:
                case_result = run_case(lmp, target, case_name, strain, mechanics_dir)
                target_record["cases"].append(case_report(case_result))
                rows_by_target[target["name"]].append(case_result["row"])
                combined_rows.append(case_result["row"])
            target_record["ok"] = all(case["ok"] for case in target_record["cases"])
            report["targets"].append(target_record)
        pure_csv = os.path.join(mechanics_dir, "mechanics_summary_pure_csh.csv")
        q2b_csv = os.path.join(mechanics_dir, "mechanics_summary_q2b_zn.csv")
        q1_csv = os.path.join(mechanics_dir, "mechanics_summary_q1_zn.csv")
        combined_csv = os.path.join(mechanics_dir, "mechanics_summary_combined.csv")
        write_csv(pure_csv, rows_by_target["pure_csh"])
        write_csv(q2b_csv, rows_by_target["q2b_zn"])
        if "q1_zn" in rows_by_target:
            write_csv(q1_csv, rows_by_target["q1_zn"])
        write_csv(combined_csv, combined_rows)
        report["summary_files"] = {
            "pure_csh": pure_csv,
            "q2b_zn": q2b_csv,
            "combined": combined_csv,
        }
        if "q1_zn" in rows_by_target:
            report["summary_files"]["q1_zn"] = q1_csv
        report["plots"] = generate_plots(mechanics_dir, rows_by_target["pure_csh"], rows_by_target["q2b_zn"], combined_rows)
        report["ok"] = all(target["ok"] for target in report["targets"])
    out = os.path.join(mechanics_dir, "mechanics_summary.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"ok": report["ok"], "out": out}, indent=2))


if __name__ == "__main__":
    main()
