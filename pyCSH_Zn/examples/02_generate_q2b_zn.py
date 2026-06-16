from __future__ import print_function

import json
import os

from workflow_common import generate_structure


OUT = os.path.join("output_Y", "workflow_v1", "q2b_zn")


if __name__ == "__main__":
    result = generate_structure(OUT, "q2b_zn", enable_zinc=True, zn_ratio=0.03, site_type="Q2b_Zn", seed=23743)
    with open(os.path.join(OUT, "generation_manifest.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2))
