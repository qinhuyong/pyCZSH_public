from __future__ import print_function

import json
import math
import os

from pbc_utils import minimum_image_vector
from validate_cementff_data import LABELS, parse_data


OXYGEN_TYPES = {3, 5, 6, 11}
TETRAHEDRAL_ANGLE = 109.47


def oxygen_role(atom_type):
    if int(atom_type) == 3:
        return "O(S)"
    if int(atom_type) == 4:
        return "O_shell"
    if int(atom_type) == 5:
        return "Ow"
    if int(atom_type) == 6:
        return "Oh"
    if int(atom_type) == 11:
        return "O_core"
    return "other O"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_zinc_summary(data_file):
    here = os.path.dirname(os.path.abspath(data_file))
    for _ in range(5):
        candidate = os.path.join(here, "zinc_summary.json")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def q1_context_from_summary(summary):
    if not summary or summary.get("Zn_site_type") != "Q1_Zn":
        return None
    selected = summary.get("selected_sites", [])
    if not selected:
        return None
    zn_id = int(selected[0]["atom_id"])
    diagnostics = summary.get("pre_minimization_geometry", {}).get("q1_motif_diagnostics", [])
    diagnostic = diagnostics[0] if diagnostics else {}
    intended = [int(x) for x in diagnostic.get("intended_oxygen_ids", [])]
    if not intended:
        report = selected[0].get("q1_selection_report", {})
        intended = [int(item["atom_id"]) for item in report.get("pre_minimization_nearest_four_zn_o_atoms", [])]
    hydroxylated = []
    for record in summary.get("hydroxylation_records", []):
        if int(record.get("zn_atom_id", -1)) != zn_id:
            continue
        for oxy in record.get("hydroxylated_oxygens", []):
            hydroxylated.append(int(oxy["oxygen_atom_id"]))
    return {
        "zn_atom_id": zn_id,
        "intended_oxygen_ids": intended,
        "hydroxylated_oxygen_ids": sorted(set(hydroxylated)),
    }


def q1_context_from_path(zinc_summary_path):
    if not zinc_summary_path or not os.path.exists(zinc_summary_path):
        return None
    return q1_context_from_summary(load_json(zinc_summary_path))


def atom_coord(atom):
    return atom["x"], atom["y"], atom["z"]


def distance(data, atom_i, atom_j):
    vec = minimum_image_vector(atom_coord(atom_i), atom_coord(atom_j), data["box"])
    return math.sqrt(sum(x * x for x in vec))


def vector_from_center(data, center, atom):
    return minimum_image_vector(atom_coord(center), atom_coord(atom), data["box"])


def angle_deg(v1, v2):
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    cosv = max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def nearest_zn_o_records(data, context, limit=4):
    if context is None:
        return []
    zn_id = int(context["zn_atom_id"])
    if zn_id not in data["atoms"]:
        return []
    zn = data["atoms"][zn_id]
    intended = {int(x) for x in context.get("intended_oxygen_ids", [])}
    hydroxylated = {int(x) for x in context.get("hydroxylated_oxygen_ids", [])}
    records = []
    for atom_id, atom in data["atoms"].items():
        if atom["type"] not in OXYGEN_TYPES:
            continue
        records.append(
            {
                "atom_id": int(atom_id),
                "atom_type": int(atom["type"]),
                "element_type_label": LABELS.get(atom["type"], "type{}".format(atom["type"])),
                "oxygen_role": oxygen_role(atom["type"]),
                "distance": float(distance(data, zn, atom)),
                "belongs_to_intended_motif": bool(int(atom_id) in intended),
                "is_hydroxylated_oxygen": bool(int(atom_id) in hydroxylated),
            }
        )
    records.sort(key=lambda item: (item["distance"], item["atom_id"]))
    return records[:limit]


def motif_geometry(data, context):
    if context is None:
        return {}
    zn_id = int(context["zn_atom_id"])
    intended = [int(x) for x in context.get("intended_oxygen_ids", []) if int(x) in data["atoms"]]
    if zn_id not in data["atoms"]:
        return {}
    zn = data["atoms"][zn_id]
    distances = []
    for oid in intended:
        atom = data["atoms"][oid]
        distances.append(
            {
                "atom_id": oid,
                "atom_type": int(atom["type"]),
                "element_type_label": LABELS.get(atom["type"], "type{}".format(atom["type"])),
                "oxygen_role": oxygen_role(atom["type"]),
                "distance": float(distance(data, zn, atom)),
                "is_hydroxylated_oxygen": bool(oid in set(context.get("hydroxylated_oxygen_ids", []))),
            }
        )
    angles = []
    for i, oid_i in enumerate(intended):
        for oid_j in intended[i + 1:]:
            v1 = vector_from_center(data, zn, data["atoms"][oid_i])
            v2 = vector_from_center(data, zn, data["atoms"][oid_j])
            value = angle_deg(v1, v2)
            angles.append(
                {
                    "atom_id_1": oid_i,
                    "atom_id_2": oid_j,
                    "angle_deg": value,
                    "tetrahedral_deviation_deg": None if value is None else abs(value - TETRAHEDRAL_ANGLE),
                }
            )
    oo = []
    for i, oid_i in enumerate(intended):
        for oid_j in intended[i + 1:]:
            oo.append(
                {
                    "atom_id_1": oid_i,
                    "atom_id_2": oid_j,
                    "distance": float(distance(data, data["atoms"][oid_i], data["atoms"][oid_j])),
                }
            )
    vals = [item["distance"] for item in distances]
    devs = [item["tetrahedral_deviation_deg"] for item in angles if item["tetrahedral_deviation_deg"] is not None]
    return {
        "zn_atom_id": zn_id,
        "intended_oxygen_ids": intended,
        "hydroxylated_oxygen_ids": [int(x) for x in context.get("hydroxylated_oxygen_ids", [])],
        "intended_Zn_O_distances_A": distances,
        "O_Zn_O_angles_deg": angles,
        "tetrahedral_angle_deviation_from_109p47_deg": {
            "count": len(devs),
            "mean": None if not devs else float(sum(devs) / len(devs)),
            "max": None if not devs else float(max(devs)),
        },
        "mean_Zn_O_distance_A": None if not vals else float(sum(vals) / len(vals)),
        "max_Zn_O_distance_A": None if not vals else float(max(vals)),
        "minimum_O_O_separation_A": None if not oo else float(min(item["distance"] for item in oo)),
        "reasonable_zn_o2oh2_like_geometry": bool(
            len(vals) >= 4
            and max(vals) <= 2.8
            and len(devs) >= 6
            and sum(devs) / len(devs) <= 45.0
            and oo
            and min(item["distance"] for item in oo) >= 1.4
        ),
    }


def compare_pre_post(pre_data_file, post_data_file, zinc_summary_path, out_path=None):
    context = q1_context_from_path(zinc_summary_path)
    pre = parse_data(pre_data_file)
    post = parse_data(post_data_file)
    pre_four = nearest_zn_o_records(pre, context, 4)
    post_four = nearest_zn_o_records(post, context, 4)
    pre_ids = [item["atom_id"] for item in pre_four]
    post_ids = [item["atom_id"] for item in post_four]
    intended = set(context.get("intended_oxygen_ids", [])) if context else set()
    post_by_id = {item["atom_id"]: item for item in nearest_zn_o_records(post, context, 9999)}
    moved_out = [
        {
            "atom_id": int(oid),
            "post_distance": post_by_id.get(int(oid), {}).get("distance"),
            "post_rank_in_oxygen_shell": None,
        }
        for oid in intended
        if post_by_id.get(int(oid), {}).get("distance") is not None and post_by_id[int(oid)]["distance"] > 2.5
    ]
    all_post = nearest_zn_o_records(post, context, 9999)
    rank = {item["atom_id"]: idx + 1 for idx, item in enumerate(all_post)}
    for item in moved_out:
        item["post_rank_in_oxygen_shell"] = rank.get(item["atom_id"])
    comparison = {
        "pre_data_file": pre_data_file,
        "post_data_file": post_data_file,
        "zinc_summary": zinc_summary_path,
        "context": context,
        "nearest_four_before_minimization": pre_four,
        "nearest_four_after_minimization": post_four,
        "same_O_atoms_remain_in_first_coordination_shell": set(pre_ids) == set(post_ids),
        "retained_first_shell_atom_ids": sorted(set(pre_ids).intersection(post_ids)),
        "lost_from_first_shell_atom_ids": sorted(set(pre_ids).difference(post_ids)),
        "gained_in_first_shell_atom_ids": sorted(set(post_ids).difference(pre_ids)),
        "fourth_neighbor_after_minimization": post_four[3] if len(post_four) >= 4 else None,
        "intended_motif_O_moved_outside_2p5A": moved_out,
        "pre_minimization_motif_geometry": motif_geometry(pre, context),
        "post_minimization_motif_geometry": motif_geometry(post, context),
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2, sort_keys=True)
            f.write("\n")
    return comparison
