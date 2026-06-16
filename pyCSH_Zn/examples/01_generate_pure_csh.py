from __future__ import print_function

import json
import os

from workflow_common import generate_structure


OUT = os.path.join("output_Y", "workflow_v1", "pure_csh")


if __name__ == "__main__":
    result = generate_structure(OUT, "pure_csh", enable_zinc=False, seed=23743)
    with open(os.path.join(OUT, "generation_manifest.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2))
