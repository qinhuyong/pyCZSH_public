from __future__ import print_function

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs
from validate_cementff_data import validate
from workflow_common import generate_structure


if __name__ == "__main__":
    out_dir = os.path.join("output_Y", "workflow_v1", "q1_zn")
    try:
        result = generate_structure(out_dir, "q1_zn", enable_zinc=True, zn_ratio=0.03, site_type="Q1_Zn", seed=23743)
    except Exception as exc:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        manifest_path = os.path.join(out_dir, "generation_manifest.json")
        failure = {
            "ok": False,
            "site_type": "Q1_Zn",
            "seed": 23743,
            "zn_ratio": 0.03,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fallback_used": False,
        }
        with open(manifest_path, "w") as f:
            json.dump(failure, f, indent=2, sort_keys=True)
            f.write("\n")
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise SystemExit(1)
    ff_result = build_forcefield(out_dir)
    inputs_result = build_inputs(result["data_file"], ff_result["forcefield"], os.path.join(out_dir, "lammps_inputs"), "q1_zn")
    manifest = {
        "site_type": "Q1_Zn",
        "seed": 23743,
        "zn_ratio": 0.03,
        "data_file": result["data_file"],
        "forcefield": ff_result,
        "inputs": inputs_result,
    }
    manifest_path = os.path.join(out_dir, "generation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    validation = validate(result["data_file"], expected_zinc_site_type="Q1_Zn", zinc_summary_path=result["zinc_summary"])
    validation_path = os.path.join(out_dir, "q1_zn_validation.json")
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2, sort_keys=True)
        f.write("\n")
    validation_q1_path = os.path.join(out_dir, "validation_q1_zn.json")
    with open(validation_q1_path, "w") as f:
        json.dump(validation, f, indent=2, sort_keys=True)
        f.write("\n")
    result["validation"] = validation_path
    result["validation_q1_zn"] = validation_q1_path
    result["generation_manifest"] = manifest_path
    result["forcefield"] = ff_result
    result["inputs"] = inputs_result
    result["classification"] = validation["classification"]
    result["total_charge"] = validation["total_charge"]
    result["zinc"] = validation["zinc"]
    print(json.dumps(result, indent=2, sort_keys=True))
