from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pbc_utils import box_volume, minimum_image_vector
from q1_diagnostics import find_zinc_summary, motif_geometry as q1_motif_geometry, nearest_zn_o_records, q1_context_from_path
from validate_cementff_data import parse_data, nearest


def rdf(data, type_a, type_b, rmax=8.0, dr=0.1):
    bins = int(rmax / dr)
    hist = [0] * bins
    atoms_a = [a for a in data["atoms"].values() if a["type"] in type_a]
    atoms_b = [a for a in data["atoms"].values() if a["type"] in type_b]
    volume = box_volume(data["box"])
    number_density_b = len(atoms_b) / volume if volume > 0.0 else 0.0
    for a in atoms_a:
        for b in atoms_b:
            if a["id"] == b["id"]:
                continue
            vec = minimum_image_vector((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]), data["box"])
            r = math.sqrt(sum(x * x for x in vec))
            idx = int(r / dr)
            if 0 <= idx < bins:
                hist[idx] += 1
    rows = []
    for i in range(bins):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        r_mid = (i + 0.5) * dr
        shell_volume = (4.0 / 3.0) * math.pi * (r_outer ** 3 - r_inner ** 3)
        ideal = len(atoms_a) * number_density_b * shell_volume
        g_r = hist[i] / ideal if ideal > 0.0 else 0.0
        rows.append({
            "r": r_mid,
            "g_r": g_r,
            "count": hist[i],
            "shell_volume": shell_volume,
            "number_density_b": number_density_b,
        })
    return rows


def angle(a, b, c):
    v1 = (a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return None
    cosv = max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def pbc_angle(data, a, b, c):
    v1 = minimum_image_vector((b["x"], b["y"], b["z"]), (a["x"], a["y"], a["z"]), data["box"])
    v2 = minimum_image_vector((b["x"], b["y"], b["z"]), (c["x"], c["y"], c["z"]), data["box"])
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 == 0 or n2 == 0:
        return None
    cosv = max(-1.0, min(1.0, sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def zinc_angles(data):
    zn_centered = []
    zn_oh_h = []
    for ang in data["angles"]:
        ids = [ang["a1"], ang["a2"], ang["a3"]]
        if any(i not in data["atoms"] for i in ids):
            continue
        atoms = [data["atoms"][i] for i in ids]
        val = pbc_angle(data, atoms[0], atoms[1], atoms[2])
        if val is None:
            continue
        if atoms[1]["type"] == 9:
            zn_centered.append({"angle_id": ang["id"], "type": ang["type"], "angle": val, "atoms": ids})
        if ang["type"] == 5:
            zn_oh_h.append({"angle_id": ang["id"], "type": ang["type"], "angle": val, "atoms": ids})
    return {"O_Zn_O": zn_centered, "Zn_Oh_H": zn_oh_h}


def zinc_angle_distributions(data):
    records = zinc_angles(data)
    return {
        "O_Zn_O": [x["angle"] for x in records["O_Zn_O"]],
        "Oh_Zn_O": [x["angle"] for x in records["O_Zn_O"]],
        "Oh_Zn_Oh": [x["angle"] for x in records["O_Zn_O"]],
        "Zn_Oh_H": [x["angle"] for x in records["Zn_Oh_H"]],
    }


def summarize(values):
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    vals = sorted(values)
    return {"count": len(vals), "min": vals[0], "mean": sum(vals) / len(vals), "max": vals[-1]}


def water_contacts(data):
    rows = []
    for atom in data["atoms"].values():
        if atom["type"] not in (5, 7):
            continue
        contacts = nearest(data, atom["id"], types={1, 2, 3, 4, 6, 8, 9})[:3]
        rows.append({"atom_id": atom["id"], "label": atom["label"], "nearest": contacts})
    return rows


def analyze(data_file, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data = parse_data(data_file)
    zn_sites = [a["id"] for a in data["atoms"].values() if a["type"] == 9]
    zn_nn = []
    for zid in zn_sites:
        zn_nn.append({"Zn": zid, "nearest_oxygen": nearest(data, zid, types={3, 5, 6})[:8]})
    angle_records = zinc_angles(data)
    angle_distributions = zinc_angle_distributions(data)
    rdf_specs = {
        "Zn_O": ({9}, {3, 5, 6}),
        "Zn_Si": ({9}, {2}),
        "Zn_Ca": ({9}, {1}),
        "Si_O": ({2}, {3, 6}),
        "Ca_O": ({1}, {3, 5, 6}),
    }
    rdf_out = {name: rdf(data, *spec) for name, spec in rdf_specs.items()}
    summary = {
        "data_file": data_file,
        "n_atoms": len(data["atoms"]),
        "zinc_nearest_neighbors": zn_nn,
        "zinc_coordination_2p3": [sum(1 for x in rec["nearest_oxygen"] if x["distance"] <= 2.3) for rec in zn_nn],
        "zinc_coordination_2p5": [sum(1 for x in rec["nearest_oxygen"] if x["distance"] <= 2.5) for rec in zn_nn],
        "angle_summary": {
            "O_Zn_O": summarize(angle_distributions["O_Zn_O"]),
            "Oh_Zn_O": summarize(angle_distributions["Oh_Zn_O"]),
            "Oh_Zn_Oh": summarize(angle_distributions["Oh_Zn_Oh"]),
            "Zn_Oh_H": summarize(angle_distributions["Zn_Oh_H"]),
        },
        "water_contacts": water_contacts(data),
        "rdf_files": {},
        "rdf_definition": {
            "normalized": True,
            "pbc": "minimum-image distance with orthogonal/triclinic boxes",
            "normalization": "count / (N_a * number_density_b * spherical_shell_volume)",
            "box_volume": box_volume(data["box"]),
        },
    }
    zinc_summary_path = find_zinc_summary(data_file)
    q1_context = q1_context_from_path(zinc_summary_path) if zinc_summary_path else None
    if q1_context is not None:
        pre_nearest = []
        pre_geometry = {}
        zinc_summary = None
        if zinc_summary_path and os.path.exists(zinc_summary_path):
            with open(zinc_summary_path) as f:
                zinc_summary = json.load(f)
            selected = zinc_summary.get("selected_sites", [])
            if selected:
                report = selected[0].get("q1_selection_report", {})
                pre_nearest = report.get("pre_minimization_nearest_four_zn_o_atoms", [])
                pre_geometry = report.get("q1_geometry_diagnostics", {}) or {}
        summary["q1_zn_diagnostics"] = {
            "zinc_summary": zinc_summary_path,
            "pre_minimization_nearest_four_zn_o_atoms": pre_nearest,
            "nearest_four_zn_o_atoms": nearest_zn_o_records(data, q1_context, 4),
            "pre_minimization_motif_geometry": pre_geometry,
            "motif_geometry": q1_motif_geometry(data, q1_context),
            "water_contact_diagnostics": water_contacts(data),
        }
    for name, rows in rdf_out.items():
        path = os.path.join(out_dir, "rdf_{}.csv".format(name))
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["r", "g_r", "count", "shell_volume", "number_density_b"])
            writer.writeheader()
            writer.writerows(rows)
        summary["rdf_files"][name] = path
    out_json = os.path.join(out_dir, "structure_analysis.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = analyze(args.data_file, args.out)
    print(json.dumps({"out": os.path.join(args.out, "structure_analysis.json"), "n_atoms": result["n_atoms"]}, indent=2))


if __name__ == "__main__":
    main()
