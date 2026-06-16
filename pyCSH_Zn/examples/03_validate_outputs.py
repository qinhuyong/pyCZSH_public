from __future__ import print_function

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from validate_cementff_data import validate


TARGETS = [
    os.path.join("output_Y", "workflow_v1", "pure_csh", "pure_csh_cementff.data"),
    os.path.join("output_Y", "workflow_v1", "q2b_zn", "q2b_zn_cementff_zn.data"),
    os.path.join("output_Y", "workflow_v1", "q1_zn", "q1_zn_cementff_zn.data"),
]


if __name__ == "__main__":
    results = {}
    for target in TARGETS:
        if not os.path.exists(target):
            continue
        result = validate(target)
        out = os.path.splitext(target)[0] + "_validation.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        results[target] = result["classification"]
    print(json.dumps(results, indent=2, sort_keys=True))
