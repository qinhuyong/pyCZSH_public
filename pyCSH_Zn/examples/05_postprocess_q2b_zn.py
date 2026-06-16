from __future__ import print_function

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from postprocess.analyze_structure import analyze


if __name__ == "__main__":
    q2b_dir = os.path.join("output_Y", "workflow_v1", "q2b_zn")
    result = analyze(
        os.path.join(q2b_dir, "q2b_zn_cementff_zn.data"),
        os.path.join(q2b_dir, "postprocess"),
    )
    print(json.dumps({"analysis": os.path.join(q2b_dir, "postprocess", "structure_analysis.json"), "n_atoms": result["n_atoms"]}, indent=2))
