from __future__ import print_function

import argparse
import json
import math
import os
import re

from pbc_utils import finalize_box, minimum_image_distance, parse_box_line


LABELS = {
    1: "Ca",
    2: "Si",
    3: "O_core",
    4: "O_shell",
    5: "Ow",
    6: "Oh",
    7: "Hw",
    8: "Hoh",
    9: "Zn",
    10: "Al",
    11: "Cl",
}
VALID_ATOM_TYPES = set(LABELS)
VALID_BOND_TYPES = {1, 2, 3}
VALID_ANGLE_TYPES = {1, 2, 3, 4, 5}
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(ROOT, "forcefields", "CementFF4_Zn_parameters.json")
CHARGE_TOLERANCE = 1.0e-5


def load_forcefield_charges(path=DEFAULT_DB):
    with open(path) as f:
        db = json.load(f)
    return {int(k): float(v["charge"]) for k, v in db["atom_types"].items()}


def parse_data(path):
    data = {"path": path, "counts": {}, "box": {}, "masses": {}, "atoms": {}, "bonds": [], "angles": [], "csinfo": {}}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^(\d+)\s+(atoms|bonds|angles|atom types|bond types|angle types)$", line)
            if m:
                data["counts"][m.group(2)] = int(m.group(1))
                continue
            parts = line.split()
            if parse_box_line(data["box"], parts):
                continue
            header = line.split("#")[0].strip()
            if header in ("Masses", "Bonds", "Angles", "CS-Info") or header.startswith("Atoms"):
                section = "Atoms" if header.startswith("Atoms") else header
                continue
            p = line.split("#")[0].split()
            if section == "Masses" and len(p) >= 2:
                data["masses"][int(p[0])] = float(p[1])
            elif section == "Atoms" and len(p) >= 7:
                aid = int(p[0])
                typ = int(p[2])
                data["atoms"][aid] = {
                    "id": aid,
                    "mol": int(p[1]),
                    "type": typ,
                    "label": LABELS.get(typ, "type{}".format(typ)),
                    "q": float(p[3]),
                    "x": float(p[4]),
                    "y": float(p[5]),
                    "z": float(p[6]),
                }
            elif section == "Bonds" and len(p) >= 4:
                data["bonds"].append({"id": int(p[0]), "type": int(p[1]), "a1": int(p[2]), "a2": int(p[3])})
            elif section == "Angles" and len(p) >= 5:
                data["angles"].append({"id": int(p[0]), "type": int(p[1]), "a1": int(p[2]), "a2": int(p[3]), "a3": int(p[4])})
            elif section == "CS-Info" and len(p) >= 2:
                data["csinfo"][int(p[0])] = int(p[1])
    data["box"] = finalize_box(data["box"])
    return data


def distance(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def pbc_distance(data, a, b):
    return minimum_image_distance(a, b, data["box"])


def nearest(data, atom_id, types=None, exclude=None):
    exclude = set(exclude or [])
    atom = data["atoms"][atom_id]
    out = []
    for other_id, other in data["atoms"].items():
        if other_id == atom_id or other_id in exclude:
            continue
        if types is not None and other["type"] not in types:
            continue
        out.append({"atom_id": other_id, "type": other["type"], "label": other["label"], "distance": pbc_distance(data, atom, other)})
    return sorted(out, key=lambda x: x["distance"])


def audit_charge_assignment(data, expected_charges, tolerance=CHARGE_TOLERANCE):
    bad = []
    for atom in sorted(data["atoms"].values(), key=lambda x: x["id"]):
        expected = expected_charges.get(atom["type"])
        if expected is None:
            continue
        delta = atom["q"] - expected
        if abs(delta) > tolerance:
            bad.append({
                "atom_id": atom["id"],
                "type": atom["type"],
                "label": atom["label"],
                "actual_charge": atom["q"],
                "expected_charge": expected,
                "delta": delta,
            })
    return {
        "tolerance": tolerance,
        "n_bad": len(bad),
        "bad_atoms": bad,
    }


def audit_csinfo(data):
    bad = []
    pairs = []
    for b in data["bonds"]:
        if b["a1"] in data["atoms"] and b["a2"] in data["atoms"]:
            t = {data["atoms"][b["a1"]]["type"], data["atoms"][b["a2"]]["type"]}
            if t == {3, 4}:
                same = data["csinfo"].get(b["a1"]) == data["csinfo"].get(b["a2"])
                rec = {"bond_id": b["id"], "a1": b["a1"], "a2": b["a2"], "same_csid": same, "distance": pbc_distance(data, data["atoms"][b["a1"]], data["atoms"][b["a2"]])}
                pairs.append(rec)
                if not same:
                    bad.append(rec)
    return {"n_pairs": len(pairs), "n_csinfo": len(data["csinfo"]), "bad_pairs": bad, "pairs": pairs}


def audit_water(data):
    waters = []
    bad = []
    for aid, atom in data["atoms"].items():
        if atom["type"] != 5:
            continue
        bonded_h = []
        for b in data["bonds"]:
            if b["type"] == 2 and aid in (b["a1"], b["a2"]):
                other = b["a2"] if b["a1"] == aid else b["a1"]
                if other in data["atoms"] and data["atoms"][other]["type"] == 7:
                    bonded_h.append(other)
        rec = {"Ow": aid, "Hw": sorted(bonded_h), "molecule_id": atom["mol"], "tip4p_compatible": len(bonded_h) == 2}
        waters.append(rec)
        if len(bonded_h) != 2:
            bad.append(rec)
    return {"n_water": len(waters), "n_bad_water": len(bad), "waters": waters, "bad_waters": bad}


def audit_zinc(data):
    zns = [aid for aid, a in data["atoms"].items() if a["type"] == 9]
    records = []
    for zid in zns:
        zn_o = nearest(data, zid, types={3, 5, 6})[:8]
        records.append({
            "Zn": zid,
            "coordination_2p3": sum(1 for x in zn_o if x["distance"] <= 2.3),
            "coordination_2p5": sum(1 for x in zn_o if x["distance"] <= 2.5),
            "nearest_oxygen": zn_o[:4],
        })
    return {"n_zinc": len(zns), "zinc_sites": records}


def load_zinc_summary(path):
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, ValueError):
        return None


def multi_zinc_site_counts(summary):
    counts = {"Q1_Zn": 0, "Q2b_Zn": 0}
    if not summary:
        return counts
    for site in summary.get("selected_sites", []):
        motif = site.get("motif") or site.get("motif_type")
        if motif in counts:
            counts[motif] += 1
    centers = summary.get("zn_centers", [])
    if centers and not any(counts.values()):
        for center in centers:
            motif = center.get("motif_type")
            if motif in counts:
                counts[motif] += 1
    return counts


def multi_zinc_failed_centers(zinc):
    failed = []
    for site in zinc.get("zinc_sites", []):
        if site.get("coordination_2p5", 0) < 4:
            failed.append({
                "zn_atom_id": site.get("Zn"),
                "coordination_2p5": site.get("coordination_2p5"),
                "nearest_oxygen": site.get("nearest_oxygen", []),
                "reason": "fewer than four O neighbors within 2.5 A",
            })
    return failed


def infer_zinc_site_type(data_file):
    here = os.path.dirname(os.path.abspath(data_file))
    for _ in range(4):
        summary_path = os.path.join(here, "zinc_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                return summary.get("Zn_site_type")
            except (IOError, ValueError):
                return None
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    normalized = data_file.replace("\\", "/").lower()
    if "/q1_zn/" in normalized or normalized.endswith("/q1_zn_cementff_zn.data"):
        return "Q1_Zn"
    return None


def validate(path, db_path=DEFAULT_DB, expected_zinc_site_type=None, zinc_summary_path=None):
    data = parse_data(path)
    expected_charges = load_forcefield_charges(db_path)
    total_charge = sum(a["q"] for a in data["atoms"].values())
    atom_type_bad = [a for a in data["atoms"].values() if a["type"] not in VALID_ATOM_TYPES]
    bond_type_bad = [b for b in data["bonds"] if b["type"] not in VALID_BOND_TYPES]
    angle_type_bad = [a for a in data["angles"] if a["type"] not in VALID_ANGLE_TYPES]
    charge_assignment = audit_charge_assignment(data, expected_charges)
    cs = audit_csinfo(data)
    water = audit_water(data)
    zinc = audit_zinc(data)
    zinc_summary = load_zinc_summary(zinc_summary_path)
    zinc_site_type = expected_zinc_site_type
    if zinc_site_type is None and zinc_summary:
        zinc_site_type = zinc_summary.get("Zn_site_type") or zinc_summary.get("multi_Zn_mode")
    if zinc_site_type is None:
        zinc_site_type = infer_zinc_site_type(path)
    reasons = []
    classification = "valid_static_candidate"
    if charge_assignment["n_bad"]:
        classification = "failed_charge_assignment"
        reasons.append("{} atoms have charges inconsistent with CementFF4_Zn_parameters.json".format(charge_assignment["n_bad"]))
    elif abs(total_charge) > CHARGE_TOLERANCE:
        classification = "failed_charge"
        reasons.append("charge residual {:.6g}".format(total_charge))
    elif atom_type_bad or bond_type_bad or angle_type_bad:
        classification = "failed_topology"
        reasons.append("invalid atom/bond/angle type")
    elif cs["bad_pairs"] or cs["n_csinfo"] != len(data["atoms"]):
        classification = "failed_csinfo"
        reasons.append("CS-Info does not match core-shell pairs")
    elif water["n_bad_water"]:
        classification = "failed_water_contacts"
        reasons.append("invalid TIP4P water topology")
    elif zinc["n_zinc"]:
        if any(site["coordination_2p5"] < 4 for site in zinc["zinc_sites"]):
            if (
                zinc_site_type == "Q1_Zn"
                and all(site["coordination_2p5"] >= 3 for site in zinc["zinc_sites"])
                and all(len(site["nearest_oxygen"]) >= 4 and site["nearest_oxygen"][3]["distance"] <= 3.2 for site in zinc["zinc_sites"])
            ):
                classification = "needs_static_relaxation"
                reasons.append("Q1_Zn has a fourth O neighbor beyond 2.5 A but within 3.2 A")
            else:
                if zinc["n_zinc"] > 1 or str(zinc_site_type or "").startswith("multi_"):
                    classification = "failed_multi_zn_candidate"
                else:
                    classification = "failed_zinc_geometry"
                failed = multi_zinc_failed_centers(zinc)
                reasons.append("Zn has fewer than 4 O neighbors within 2.5 A: {}".format(failed))
        elif zinc["n_zinc"] > 1 or str(zinc_site_type or "").startswith("multi_"):
            motif_counts = multi_zinc_site_counts(zinc_summary)
            if motif_counts["Q1_Zn"] and motif_counts["Q2b_Zn"]:
                classification = "valid_multi_q1_q2b_zn_candidate"
            elif motif_counts["Q1_Zn"]:
                classification = "valid_multi_q1_zn_candidate"
            elif motif_counts["Q2b_Zn"]:
                classification = "valid_multi_q2b_zn_candidate"
            else:
                classification = "failed_multi_zn_candidate"
                reasons.append("multi-Zn motif composition could not be inferred from zinc summary")
        elif zinc_site_type == "Q1_Zn":
            classification = "valid_q1_zn_candidate"
        else:
            classification = "valid_q2b_zn_candidate"
    return {
        "data_file": path,
        "classification": classification,
        "reasons": reasons,
        "counts": {
            "atoms": len(data["atoms"]),
            "bonds": len(data["bonds"]),
            "angles": len(data["angles"]),
            "atom_types": sorted(set(a["type"] for a in data["atoms"].values())),
        },
        "total_charge": total_charge,
        "charge_assignment": charge_assignment,
        "csinfo": {"n_pairs": cs["n_pairs"], "n_entries": cs["n_csinfo"], "n_bad_pairs": len(cs["bad_pairs"])},
        "water": {"n_water": water["n_water"], "n_bad_water": water["n_bad_water"]},
        "zinc": zinc,
        "zinc_site_type": zinc_site_type,
        "multi_zinc_failed_centers": multi_zinc_failed_centers(zinc) if str(zinc_site_type or "").startswith("multi_") or zinc["n_zinc"] > 1 else [],
        "multi_zinc_site_counts": multi_zinc_site_counts(zinc_summary),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--expected-zinc-site-type", default=None)
    parser.add_argument("--zinc-summary", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = validate(args.data_file, args.db, args.expected_zinc_site_type, args.zinc_summary)
    out = args.out or os.path.splitext(args.data_file)[0] + "_validation.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"classification": result["classification"], "out": out}, indent=2))


if __name__ == "__main__":
    main()
