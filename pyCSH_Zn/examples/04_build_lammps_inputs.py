from __future__ import print_function

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from forcefields.build_cementff4_zn import build as build_forcefield
from lammps_templates.build_inputs import build as build_inputs


if __name__ == "__main__":
    pure_dir = os.path.join("output_Y", "workflow_v1", "pure_csh")
    q2b_dir = os.path.join("output_Y", "workflow_v1", "q2b_zn")
    q1_dir = os.path.join("output_Y", "workflow_v1", "q1_zn")
    pure_ff_result = build_forcefield(pure_dir)
    ff_result = build_forcefield(q2b_dir)
    pure_inputs = build_inputs(
        os.path.join(pure_dir, "pure_csh_cementff.data"),
        pure_ff_result["forcefield"],
        os.path.join(pure_dir, "lammps_inputs"),
        "pure_csh",
    )
    q2b_inputs = build_inputs(
        os.path.join(q2b_dir, "q2b_zn_cementff_zn.data"),
        ff_result["forcefield"],
        os.path.join(q2b_dir, "lammps_inputs"),
        "q2b_zn",
    )
    result = {
        "pure_csh": {"forcefield": pure_ff_result, "inputs": pure_inputs},
        "q2b_zn": {"forcefield": ff_result, "inputs": q2b_inputs},
    }
    q1_data = os.path.join(q1_dir, "q1_zn_cementff_zn.data")
    if os.path.exists(q1_data):
        q1_ff_result = build_forcefield(q1_dir)
        q1_inputs = build_inputs(
            q1_data,
            q1_ff_result["forcefield"],
            os.path.join(q1_dir, "lammps_inputs"),
            "q1_zn",
        )
        result["q1_zn"] = {"forcefield": q1_ff_result, "inputs": q1_inputs}
    print(json.dumps(result, indent=2, sort_keys=True))
