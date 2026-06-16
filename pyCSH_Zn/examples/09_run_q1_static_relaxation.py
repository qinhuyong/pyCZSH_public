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
from q1_diagnostics import compare_pre_post


TARGET = {
    "name": "q1_zn",
    "data": os.path.join("output_Y", "workflow_v1", "q1_zn", "q1_zn_cementff_zn.data"),
    "input_dir": os.path.join("output_Y", "workflow_v1", "q1_zn", "lammps_inputs"),
    "min_raw": "q1_zn_minimized_static.raw.data",
    "min_with_cs": "q1_zn_minimized_static.data",
}


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


def main():
    lmp = find_lammps()
    input_dir = TARGET["input_dir"]
    q1_dir = os.path.join("output_Y", "workflow_v1", "q1_zn")
    zinc_summary = os.path.join(q1_dir, "zinc_summary.json")
    report = {
        "workflow": "v1.3.1-Q1-postmin-diagnostics",
        "lammps_executable": lmp,
        "target": TARGET["name"],
        "finite_temperature_md": "not run",
        "scope": "Q1_Zn post-minimization coordination diagnostics; validation semantics unchanged",
    }
    if not lmp:
        report["ok"] = False
        report["reason"] = "No LAMMPS executable found."
    else:
        steps = {}
        for input_name in ("in.read_check", "in.run0", "in.minimize_static"):
            steps[input_name] = run_lammps(lmp, input_dir, input_name)
            if not steps[input_name]["ok"]:
                report["ok"] = False
                report["steps"] = steps
                break
        else:
            raw = os.path.join(input_dir, TARGET["min_raw"])
            final = os.path.join(input_dir, TARGET["min_with_cs"])
            if os.path.exists(raw):
                append_csinfo(TARGET["data"], raw, final)
                validation = validate(
                    final,
                    expected_zinc_site_type="Q1_Zn",
                    zinc_summary_path=zinc_summary,
                )
                validation_path = os.path.splitext(final)[0] + "_validation.json"
                with open(validation_path, "w") as f:
                    json.dump(validation, f, indent=2, sort_keys=True)
                    f.write("\n")
                compare_path = os.path.join(q1_dir, "q1_zn_pre_post_coordination_compare.json")
                comparison = compare_pre_post(TARGET["data"], final, zinc_summary, compare_path)
                post_min_ok = validation["classification"] in ("valid_q1_zn_candidate",)
                workflow_ok = validation["classification"] in ("valid_q1_zn_candidate", "needs_static_relaxation")
                report["ok"] = workflow_ok
                report["post_min_validation_ok"] = post_min_ok
                report["classification"] = validation["classification"]
                report["validation_reasons"] = validation["reasons"]
                report["validation"] = validation_path
                report["coordination_compare"] = compare_path
                report["nearest_four_before_minimization"] = comparison["nearest_four_before_minimization"]
                report["nearest_four_after_minimization"] = comparison["nearest_four_after_minimization"]
                report["intended_motif_O_moved_outside_2p5A"] = comparison["intended_motif_O_moved_outside_2p5A"]
                if not post_min_ok:
                    report["diagnostic_conclusion"] = (
                        "Q1_Zn static relaxation completed, but post-min validation remains "
                        "{}. See q1_zn_pre_post_coordination_compare.json for the exact Zn-O shell change."
                    ).format(validation["classification"])
            else:
                report["ok"] = False
                report["reason"] = "raw data was not written"
            report["steps"] = steps
    out = os.path.join("output_Y", "workflow_v1", "q1_zn_static_relaxation_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"ok": report["ok"], "out": out}, indent=2))


if __name__ == "__main__":
    main()
