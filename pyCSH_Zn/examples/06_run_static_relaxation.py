from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from validate_cementff_data import parse_data, validate


TARGETS = [
    {
        "name": "pure_csh",
        "data": os.path.join("output_Y", "workflow_v1", "pure_csh", "pure_csh_cementff.data"),
        "input_dir": os.path.join("output_Y", "workflow_v1", "pure_csh", "lammps_inputs"),
        "min_raw": "pure_csh_minimized_static.raw.data",
        "min_with_cs": "pure_csh_minimized_static.data",
        "elastic_plus_raw": "pure_csh_elastic_x_plus.raw.data",
        "elastic_plus_with_cs": "pure_csh_elastic_x_plus.data",
        "elastic_minus_raw": "pure_csh_elastic_x_minus.raw.data",
        "elastic_minus_with_cs": "pure_csh_elastic_x_minus.data",
    },
    {
        "name": "q2b_zn",
        "data": os.path.join("output_Y", "workflow_v1", "q2b_zn", "q2b_zn_cementff_zn.data"),
        "input_dir": os.path.join("output_Y", "workflow_v1", "q2b_zn", "lammps_inputs"),
        "min_raw": "q2b_zn_minimized_static.raw.data",
        "min_with_cs": "q2b_zn_minimized_static.data",
        "elastic_plus_raw": "q2b_zn_elastic_x_plus.raw.data",
        "elastic_plus_with_cs": "q2b_zn_elastic_x_plus.data",
        "elastic_minus_raw": "q2b_zn_elastic_x_minus.raw.data",
        "elastic_minus_with_cs": "q2b_zn_elastic_x_minus.data",
    },
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


def read_csinfo(path):
    data = parse_data(path)
    return data["csinfo"]


def append_csinfo(reference_data, raw_data, output_data):
    csinfo = read_csinfo(reference_data)
    with open(raw_data) as f:
        text = f.read().rstrip()
    with open(output_data, "w") as f:
        f.write(text)
        f.write("\n\nCS-Info\n\n")
        for atom_id in sorted(csinfo):
            f.write("{:8d} {:8d}\n".format(atom_id, csinfo[atom_id]))
    return output_data


def validate_with_cs(reference_data, input_dir, raw_name, final_name):
    raw = os.path.join(input_dir, raw_name)
    final = os.path.join(input_dir, final_name)
    if not os.path.exists(raw):
        return {"ok": False, "reason": "raw LAMMPS data file not found", "raw": raw}
    append_csinfo(reference_data, raw, final)
    result = validate(final)
    out = os.path.splitext(final)[0] + "_validation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return {
        "ok": result["classification"] in ("valid_static_candidate", "valid_q2b_zn_candidate"),
        "data": final,
        "validation": out,
        "classification": result["classification"],
        "charge_assignment_bad_atoms": result["charge_assignment"]["n_bad"],
        "csinfo_bad_pairs": result["csinfo"]["n_bad_pairs"],
    }


def run_target(lmp, target):
    input_dir = target["input_dir"]
    record = {"name": target["name"], "input_dir": input_dir, "steps": {}}
    for input_name in ("in.read_check", "in.run0", "in.minimize_static", "in.elastic_x_plus", "in.elastic_x_minus"):
        record["steps"][input_name] = run_lammps(lmp, input_dir, input_name)
        if not record["steps"][input_name]["ok"]:
            record["ok"] = False
            return record
    record["post_minimization_validation"] = validate_with_cs(
        target["data"], input_dir, target["min_raw"], target["min_with_cs"]
    )
    record["elastic_x_plus_validation"] = validate_with_cs(
        target["data"], input_dir, target["elastic_plus_raw"], target["elastic_plus_with_cs"]
    )
    record["elastic_x_minus_validation"] = validate_with_cs(
        target["data"], input_dir, target["elastic_minus_raw"], target["elastic_minus_with_cs"]
    )
    elastic_ok = record["elastic_x_plus_validation"]["ok"] and record["elastic_x_minus_validation"]["ok"]
    record["ok"] = (
        all(step["ok"] for step in record["steps"].values())
        and record["post_minimization_validation"]["ok"]
        and elastic_ok
    )
    return record


def main():
    lmp = find_lammps()
    report = {
        "workflow": "v1.1-static-relaxation",
        "finite_temperature_md": "not run",
        "lammps_executable": lmp,
        "targets_order": [x["name"] for x in TARGETS],
        "targets": [],
    }
    if not lmp:
        report["ok"] = False
        report["reason"] = "No LAMMPS executable found. Set LAMMPS_EXE or add lmp to PATH."
    else:
        active_targets = []
        for target in TARGETS:
            if target["name"] == "q1_zn" and not os.path.exists(target["data"]):
                continue
            active_targets.append(target)
        report["targets_order"] = [x["name"] for x in active_targets]
        for target in active_targets:
            report["targets"].append(run_target(lmp, target))
        report["ok"] = all(target.get("ok") for target in report["targets"])
    out = os.path.join("output_Y", "workflow_v1", "static_relaxation_report.json")
    out_dir = os.path.dirname(out)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"ok": report["ok"], "out": out}, indent=2))


if __name__ == "__main__":
    main()
