# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 09:27:48 2025

@author: YERAI
"""

import numpy as np
from mod_pores_Y import *
import os
import sys
import copy
import random
import json
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
from mod_zinc import CEMENTFF4_ANGLE_MAP, CEMENTFF4_TYPE_MAP, write_zinc_summary
from forcefields.build_cementff4_zn import load_database as load_cementff4_db
from forcefields.build_cementff4_zn import write_forcefield as write_cementff4_forcefield_from_db


CEMENTFF4_LAMMPS_TYPES = {
    1: ("Ca", 40.08),
    2: ("Si", 28.10),
    3: ("O", 15.999),
    4: ("O(S)", 0.40),
    5: ("Ow", 15.999),
    6: ("Oh", 15.999),
    7: ("Hw", 1.008),
    8: ("Hoh", 1.008),
    9: ("Zn", 65.38),
    10: ("Al", 26.9815),
    11: ("Cl", 35.45),
}

CEMENTFF4_BOND_MAP = {
    1: "O_core-O_shell",
    2: "Ow-Hw",
    3: "Oh-Hoh",
}

DEFAULT_WATER_CONTACT_CUTOFFS = {
    "water_min_H_Ca": 1.6,
    "water_min_H_Si": 1.6,
    "water_min_H_Zn": 1.4,
    "water_min_H_O_nonbonded": 1.2,
    "water_min_H_H_nonbonded": 1.2,
    "water_min_Ow_Ca": 2.2,
    "water_min_Ow_O": 2.2,
}


def lammps_restricted_triclinic_bounds(supercell):
    xlo = 0.0
    ylo = 0.0
    zlo = 0.0
    lx = float(supercell[0, 0])
    ly = float(supercell[1, 1])
    lz = float(supercell[2, 2])
    xy = float(supercell[1, 0])
    xz = float(supercell[2, 0])
    yz = float(supercell[2, 1])
    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xlo + lx + max(0.0, xy, xz, xy + xz)
    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = ylo + ly + max(0.0, yz)
    return {
        "xlo_bound": xlo_bound,
        "xhi_bound": xhi_bound,
        "ylo_bound": ylo_bound,
        "yhi_bound": yhi_bound,
        "zlo_bound": zlo,
        "zhi_bound": zlo + lz,
        "xy": xy,
        "xz": xz,
        "yz": yz,
    }


def cementff4_bond_type(entry, entries_crystal=None):
    internal_bond_type = int(entry[1])
    if entries_crystal is not None:
        atom_types = {int(atom[0]): int(atom[1]) for atom in entries_crystal}
        atom_i = CEMENTFF4_TYPE_MAP.get(atom_types.get(int(entry[2])), {}).get("lammps_type")
        atom_j = CEMENTFF4_TYPE_MAP.get(atom_types.get(int(entry[3])), {}).get("lammps_type")
        pair = {atom_i, atom_j}
        if pair == {3, 4}:
            return 1
        if pair == {5, 7}:
            return 2
        if pair == {6, 8}:
            return 3
    return internal_bond_type


def cementff4_atom_type(entry):
    internal_type = int(entry[1])
    if internal_type not in CEMENTFF4_TYPE_MAP:
        raise ValueError("No CementFF4 type mapping for internal specie {}".format(internal_type))
    return int(CEMENTFF4_TYPE_MAP[internal_type]["lammps_type"])


def cementff4_atom_label(entry):
    return CEMENTFF4_TYPE_MAP[int(entry[1])]["label"]


def cementff4_csinfo(entries_crystal, entries_bonds):
    by_id = _entry_by_atom_id(entries_crystal)
    csinfo = {}
    pairs = []
    next_id = 1
    for bond in entries_bonds:
        a1 = int(bond[2])
        a2 = int(bond[3])
        if a1 not in by_id or a2 not in by_id:
            continue
        t1 = cementff4_atom_type(by_id[a1])
        t2 = cementff4_atom_type(by_id[a2])
        if {t1, t2} == {3, 4}:
            csinfo[a1] = next_id
            csinfo[a2] = next_id
            pairs.append({"csid": next_id, "core": a1 if t1 == 3 else a2, "shell": a2 if t1 == 3 else a1})
            next_id += 1
    for entry in sorted(entries_crystal, key=lambda x: int(x[0])):
        atom_id = int(entry[0])
        if atom_id not in csinfo:
            csinfo[atom_id] = next_id
            next_id += 1
    return csinfo, pairs


def _entry_by_atom_id(entries_crystal):
    return {int(entry[0]): entry for entry in entries_crystal}


def _periodic_vector(coord_i, coord_j, supercell):
    inv_supercell = np.linalg.inv(supercell)
    delta = np.array(coord_j, dtype=float) - np.array(coord_i, dtype=float)
    frac = np.dot(delta, inv_supercell)
    frac -= np.rint(frac)
    return np.dot(frac, supercell)


def _periodic_distance(coord_i, coord_j, supercell):
    return float(np.linalg.norm(_periodic_vector(coord_i, coord_j, supercell)))


def _water_molecules_from_bonds(entries_crystal, entries_bonds):
    by_id = _entry_by_atom_id(entries_crystal)
    waters = []
    for entry in sorted(entries_crystal, key=lambda x: int(x[0])):
        if cementff4_atom_type(entry) != 5:
            continue
        ow_id = int(entry[0])
        h_bonds = []
        for bond in entries_bonds:
            if cementff4_bond_type(bond, entries_crystal) != 2:
                continue
            a1 = int(bond[2])
            a2 = int(bond[3])
            if ow_id not in (a1, a2):
                continue
            other_id = a2 if a1 == ow_id else a1
            if other_id in by_id and cementff4_atom_type(by_id[other_id]) == 7:
                h_bonds.append((other_id, int(bond[0])))
        waters.append(
            {
                "Ow": ow_id,
                "Hw": sorted([x[0] for x in h_bonds]),
                "bond_ids": [x[1] for x in h_bonds],
            }
        )
    return waters


def _water_angle_ids(entries_angle, waters):
    by_ow = {water["Ow"]: water for water in waters}
    for water in waters:
        water["angle_ids"] = []
    for angle in entries_angle:
        if int(angle[1]) != 1:
            continue
        center = int(angle[3])
        if center not in by_ow:
            continue
        hset = set(by_ow[center]["Hw"])
        if set((int(angle[2]), int(angle[4]))) == hset:
            by_ow[center]["angle_ids"].append(int(angle[0]))
    return waters


def cementff4_molecule_id_map(entries_crystal, entries_bonds, entries_angle, zinc_summary=None):
    """Assign deterministic molecule IDs for CementFF4 LAMMPS output."""
    waters = _water_angle_ids(entries_angle, _water_molecules_from_bonds(entries_crystal, entries_bonds))
    mol_map = {int(entry[0]): 0 for entry in entries_crystal}
    water_records = []
    next_mol = 1
    for water in waters:
        atoms = [water["Ow"]] + water["Hw"]
        if len(water["Hw"]) != 2:
            continue
        for atom_id in atoms:
            mol_map[atom_id] = next_mol
        water_records.append(
            {
                "molecule_id": next_mol,
                "Ow": water["Ow"],
                "Hw": list(water["Hw"]),
                "bond_ids": list(water.get("bond_ids", [])),
                "angle_ids": list(water.get("angle_ids", [])),
            }
        )
        next_mol += 1

    hydroxyl_records = []
    used = set()
    by_id = _entry_by_atom_id(entries_crystal)
    for bond in entries_bonds:
        if cementff4_bond_type(bond, entries_crystal) != 3:
            continue
        a1 = int(bond[2])
        a2 = int(bond[3])
        if a1 not in by_id or a2 not in by_id:
            continue
        types = {cementff4_atom_type(by_id[a1]), cementff4_atom_type(by_id[a2])}
        if types != {6, 8}:
            continue
        if a1 in used or a2 in used:
            continue
        mol_map[a1] = next_mol
        mol_map[a2] = next_mol
        hydroxyl_records.append({"molecule_id": next_mol, "atoms": [a1, a2], "bond_id": int(bond[0])})
        used.update([a1, a2])
        next_mol += 1

    if zinc_summary is not None:
        zinc_summary["cementff4_molecule_id_policy"] = {
            "framework_molecule_id": 0,
            "water_molecules": water_records,
            "hydroxyl_pairs": hydroxyl_records,
        }
    return mol_map, water_records, hydroxyl_records


def update_zinc_classification_for_water(zinc_summary):
    if zinc_summary is None:
        return
    sanitizer = zinc_summary.get("water_sanitizer")
    if not sanitizer:
        return
    if sanitizer.get("rejected"):
        zinc_summary["output_classification"] = "failed_water_sanitization"
        zinc_summary.setdefault("classification_reasons", []).append(
            "one or more water molecules still violate CementFF4 water contact cutoffs after sanitizer"
        )
    elif sanitizer.get("bad_before"):
        if zinc_summary.get("output_classification") == "valid_q2b_zn_candidate":
            zinc_summary["output_classification"] = "needs_static_relaxation"
        zinc_summary.setdefault("classification_reasons", []).append(
            "water contacts were sanitized; static relaxation is required before property calculations"
        )


def _water_contact_metrics(entries_crystal, entries_bonds, supercell, water):
    by_id = _entry_by_atom_id(entries_crystal)
    water_ids = set([water["Ow"]] + water["Hw"])
    metrics = {}

    def nearest(atom_id, target_lammps_types, exclude_ids):
        best = None
        atom = by_id[atom_id]
        for other in entries_crystal:
            other_id = int(other[0])
            if other_id == atom_id or other_id in exclude_ids:
                continue
            if cementff4_atom_type(other) not in target_lammps_types:
                continue
            d = _periodic_distance(atom[3:], other[3:], supercell)
            if best is None or d < best["distance"]:
                best = {
                    "distance": d,
                    "atom_id": other_id,
                    "type": cementff4_atom_type(other),
                    "label": cementff4_atom_label(other),
                }
        return best

    metrics["Ow_Ca"] = nearest(water["Ow"], {1}, water_ids)
    metrics["Ow_Si"] = nearest(water["Ow"], {2}, water_ids)
    metrics["Ow_O"] = nearest(water["Ow"], {3, 4, 5, 6}, water_ids)
    metrics["Ow_Zn"] = nearest(water["Ow"], {9}, water_ids)
    metrics["Hw"] = []
    for h_id in water["Hw"]:
        metrics["Hw"].append(
            {
                "H": h_id,
                "H_O_nonbonded": nearest(h_id, {3, 4, 5, 6}, water_ids),
                "H_H_nonbonded": nearest(h_id, {7, 8}, water_ids),
                "H_Ca": nearest(h_id, {1}, water_ids),
                "H_Si": nearest(h_id, {2}, water_ids),
                "H_Zn": nearest(h_id, {9}, water_ids),
            }
        )
    return metrics


def _water_contact_violations(metrics, cutoffs):
    checks = [
        ("Ow_Ca", "water_min_Ow_Ca"),
        ("Ow_O", "water_min_Ow_O"),
    ]
    violations = []
    for metric_name, cutoff_name in checks:
        value = metrics.get(metric_name)
        if value is not None and value["distance"] < cutoffs[cutoff_name]:
            violations.append({"contact": metric_name, "cutoff": cutoffs[cutoff_name], **value})
    for h_metrics in metrics["Hw"]:
        for metric_name, cutoff_name in [
            ("H_O_nonbonded", "water_min_H_O_nonbonded"),
            ("H_H_nonbonded", "water_min_H_H_nonbonded"),
            ("H_Ca", "water_min_H_Ca"),
            ("H_Si", "water_min_H_Si"),
            ("H_Zn", "water_min_H_Zn"),
        ]:
            value = h_metrics.get(metric_name)
            if value is not None and value["distance"] < cutoffs[cutoff_name]:
                item = {"contact": metric_name, "cutoff": cutoffs[cutoff_name], "H": h_metrics["H"]}
                item.update(value)
                violations.append(item)
    return violations


def _set_water_geometry(entries_crystal, water, h1_vec, h2_vec):
    by_id = _entry_by_atom_id(entries_crystal)
    ow_coord = np.array(by_id[water["Ow"]][3:], dtype=float)
    by_id[water["Hw"][0]][3:] = list(ow_coord + h1_vec)
    by_id[water["Hw"][1]][3:] = list(ow_coord + h2_vec)


def _water_orientation_candidates(entries_crystal, water, supercell):
    by_id = _entry_by_atom_id(entries_crystal)
    ow = np.array(by_id[water["Ow"]][3:], dtype=float)
    current_h1 = _periodic_vector(ow, by_id[water["Hw"][0]][3:], supercell)
    norm = np.linalg.norm(current_h1)
    if norm < 1.0e-8:
        current_h1 = np.array([1.0, 0.0, 0.0])
    else:
        current_h1 = current_h1 / norm
    axes = [
        current_h1,
        -current_h1,
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    oh = 0.9572
    theta = np.radians(104.52)
    candidates = []
    for axis in axes:
        axis = axis / np.linalg.norm(axis)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(axis, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(axis, ref)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        for phi in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            ring = np.cos(phi) * perp1 + np.sin(phi) * perp2
            h1 = oh * axis
            h2 = oh * (np.cos(theta) * axis + np.sin(theta) * ring)
            candidates.append((h1, h2))
    return candidates


def _translate_water(entries_crystal, water, vector):
    by_id = _entry_by_atom_id(entries_crystal)
    for atom_id in [water["Ow"]] + water["Hw"]:
        by_id[atom_id][3:] = list(np.array(by_id[atom_id][3:], dtype=float) + vector)


def sanitize_cementff4_water(entries_crystal, entries_bonds, entries_angle, supercell, zinc_summary=None, cutoffs=None):
    """Deterministically rotate/translate water molecules away from bad contacts."""
    cutoffs = dict(DEFAULT_WATER_CONTACT_CUTOFFS if cutoffs is None else cutoffs)
    waters = _water_angle_ids(entries_angle, _water_molecules_from_bonds(entries_crystal, entries_bonds))
    report = {
        "enabled": True,
        "cutoffs": cutoffs,
        "n_water": len(waters),
        "bad_before": [],
        "repaired_by_rotation": [],
        "repaired_by_translation": [],
        "rejected": [],
        "final_min_contacts": {},
    }
    by_id = _entry_by_atom_id(entries_crystal)
    original_coords = {int(entry[0]): list(entry[3:]) for entry in entries_crystal}

    for water in waters:
        if len(water["Hw"]) != 2:
            report["rejected"].append({"Ow": water["Ow"], "reason": "water does not have exactly two H atoms"})
            continue
        metrics = _water_contact_metrics(entries_crystal, entries_bonds, supercell, water)
        violations = _water_contact_violations(metrics, cutoffs)
        if not violations:
            continue
        report["bad_before"].append({"Ow": water["Ow"], "Hw": list(water["Hw"]), "violations": violations})
        saved = {atom_id: list(by_id[atom_id][3:]) for atom_id in [water["Ow"]] + water["Hw"]}

        repaired = False
        best_trial = None
        best_score = -1.0
        for h1, h2 in _water_orientation_candidates(entries_crystal, water, supercell):
            _set_water_geometry(entries_crystal, water, h1, h2)
            trial_metrics = _water_contact_metrics(entries_crystal, entries_bonds, supercell, water)
            trial_violations = _water_contact_violations(trial_metrics, cutoffs)
            distances = []
            for value in trial_metrics.values():
                if isinstance(value, dict) and "distance" in value:
                    distances.append(value["distance"])
            for h_metrics in trial_metrics["Hw"]:
                for value in h_metrics.values():
                    if isinstance(value, dict) and "distance" in value:
                        distances.append(value["distance"])
            score = min(distances) if distances else 0.0
            if score > best_score:
                best_score = score
                best_trial = (h1, h2, trial_violations)
            if not trial_violations:
                report["repaired_by_rotation"].append({"Ow": water["Ow"], "Hw": list(water["Hw"])})
                repaired = True
                break

        if repaired:
            continue
        for atom_id, coord in saved.items():
            by_id[atom_id][3:] = coord
        if best_trial is not None:
            _set_water_geometry(entries_crystal, water, best_trial[0], best_trial[1])

        ow_coord = np.array(by_id[water["Ow"]][3:], dtype=float)
        directions = []
        for other in entries_crystal:
            other_id = int(other[0])
            if other_id in [water["Ow"]] + water["Hw"]:
                continue
            d = _periodic_distance(ow_coord, other[3:], supercell)
            if d < 3.0:
                v = -_periodic_vector(ow_coord, other[3:], supercell)
                if np.linalg.norm(v) > 1.0e-8:
                    directions.append(v / np.linalg.norm(v))
        if not directions:
            directions = [np.array([1.0, 0.0, 0.0])]
        direction = np.sum(directions, axis=0)
        if np.linalg.norm(direction) < 1.0e-8:
            direction = directions[0]
        direction = direction / np.linalg.norm(direction)

        translated = False
        for step in [0.1, 0.2, 0.3, 0.4, 0.5]:
            test_saved = {atom_id: list(by_id[atom_id][3:]) for atom_id in [water["Ow"]] + water["Hw"]}
            _translate_water(entries_crystal, water, step * direction)
            trial_metrics = _water_contact_metrics(entries_crystal, entries_bonds, supercell, water)
            trial_violations = _water_contact_violations(trial_metrics, cutoffs)
            if not trial_violations:
                report["repaired_by_translation"].append(
                    {"Ow": water["Ow"], "Hw": list(water["Hw"]), "translation": list(step * direction)}
                )
                translated = True
                break
            for atom_id, coord in test_saved.items():
                by_id[atom_id][3:] = coord
        if not translated:
            final_metrics = _water_contact_metrics(entries_crystal, entries_bonds, supercell, water)
            report["rejected"].append(
                {
                    "Ow": water["Ow"],
                    "Hw": list(water["Hw"]),
                    "violations": _water_contact_violations(final_metrics, cutoffs),
                }
            )

    final_min = {}
    for water in waters:
        metrics = _water_contact_metrics(entries_crystal, entries_bonds, supercell, water)
        for key, value in metrics.items():
            if key == "Hw":
                for h_metrics in value:
                    for h_key, h_value in h_metrics.items():
                        if isinstance(h_value, dict) and "distance" in h_value:
                            old = final_min.get(h_key)
                            if old is None or h_value["distance"] < old["distance"]:
                                final_min[h_key] = h_value
                continue
            if isinstance(value, dict) and "distance" in value:
                old = final_min.get(key)
                if old is None or value["distance"] < old["distance"]:
                    final_min[key] = value
    report["final_min_contacts"] = final_min
    if zinc_summary is not None:
        zinc_summary["water_sanitizer"] = report
    return entries_crystal, report


def get_lammps_input_cementff(name, entries_crystal, entries_bonds, entries_angle, supercell, zinc_summary=None, sanitize_water=True):
    """Write a CementFF4-oriented LAMMPS data file with fixed atom type IDs."""
    if sanitize_water:
        entries_crystal, water_sanitizer_report = sanitize_cementff4_water(
            entries_crystal, entries_bonds, entries_angle, supercell, zinc_summary
        )
    mol_map, water_molecule_records, hydroxyl_records = cementff4_molecule_id_map(
        entries_crystal, entries_bonds, entries_angle, zinc_summary
    )
    csinfo, cs_pairs = cementff4_csinfo(entries_crystal, entries_bonds)
    has_zinc = any(int(entry[1]) == 14 for entry in entries_crystal)
    max_atom_type = 9
    max_angle_type = 5

    with open(name, "w") as f:
        f.write("Generated with pyCSH CementFF4 output\n")
        f.write("# Fixed CementFF4 type map; force-field coefficients are in in.CementFF4 or in.CementFF4_Zn.\n\n")
        if zinc_summary is not None:
            f.write("# Zn_site_type: {}\n".format(zinc_summary.get("Zn_site_type")))
            f.write("# target_Zn_Si_ratio: {}\n".format(zinc_summary.get("target_Zn_Si_ratio")))
            f.write("# actual_Zn_Si_ratio: {}\n\n".format(zinc_summary.get("actual_Zn_Si_ratio")))
        f.write("{: 8d} atoms\n".format(len(entries_crystal)))
        f.write("{: 8d} bonds\n".format(len(entries_bonds)))
        f.write("{: 8d} angles\n".format(len(entries_angle)))
        f.write("{: 8d} atom types\n".format(max_atom_type))
        f.write("{: 8d} bond types\n".format(3 if entries_bonds else 0))
        f.write("{: 8d} angle types\n".format(max_angle_type if entries_angle else 0))
        f.write("\n")
        bounds = lammps_restricted_triclinic_bounds(supercell)
        f.write("{: 12.6f} {: 12.6f} xlo xhi\n".format(bounds["xlo_bound"], bounds["xhi_bound"]))
        f.write("{: 12.6f} {: 12.6f} ylo yhi\n".format(bounds["ylo_bound"], bounds["yhi_bound"]))
        f.write("{: 12.6f} {: 12.6f} zlo zhi\n".format(bounds["zlo_bound"], bounds["zhi_bound"]))
        f.write(
            "{: 12.6f} {: 12.6f} {: 12.6f} xy xz yz\n".format(
                bounds["xy"], bounds["xz"], bounds["yz"]
            )
        )
        f.write("\nMasses\n\n")
        for lammps_type in range(1, max_atom_type + 1):
            label, mass = CEMENTFF4_LAMMPS_TYPES[lammps_type]
            f.write("{: 4d} {: 12.6f} # {}\n".format(lammps_type, mass, label))

        f.write("\nAtoms # full\n\n")
        fmt = "{: 8d} {: 8d} {: 8d} {: 10.6f} {: 12.6f} {: 12.6f} {: 12.6f} # {}\n"
        for entry in entries_crystal:
            lammps_type = cementff4_atom_type(entry)
            label = cementff4_atom_label(entry)
            f.write(fmt.format(int(entry[0]), mol_map.get(int(entry[0]), 0), lammps_type, float(entry[2]), *entry[3:], label))

        if entries_bonds:
            f.write("\nBonds\n\n")
            for entry in entries_bonds:
                f.write(
                    "{: 8d} {: 8d} {: 8d} {: 8d}\n".format(
                        int(entry[0]), cementff4_bond_type(entry, entries_crystal), int(entry[2]), int(entry[3])
                    )
                )

        if entries_angle:
            f.write("\nAngles\n\n")
            for entry in entries_angle:
                f.write("{: 8d} {: 8d} {: 8d} {: 8d} {: 8d}\n".format(*entry))
        f.write("\nCS-Info\n\n")
        for atom_id in sorted(csinfo):
            f.write("{: 8d} {: 8d}\n".format(atom_id, csinfo[atom_id]))
    return {
        "molecule_id_policy": {
            "framework_molecule_id": 0,
            "water_molecules": water_molecule_records,
            "hydroxyl_pairs": hydroxyl_records,
        },
        "csinfo": {
            "n_entries": len(csinfo),
            "n_core_shell_pairs": len(cs_pairs),
            "pairs": cs_pairs,
        },
        "water_sanitizer": zinc_summary.get("water_sanitizer") if zinc_summary is not None else water_sanitizer_report,
    }


def write_cementff4_mapping_json(name, zinc_enabled):
    mapping = {
        "atom_type_map": {
            str(k): {"label": v[0], "mass": v[1]}
            for k, v in CEMENTFF4_LAMMPS_TYPES.items()
            if zinc_enabled or k <= 8
        },
        "internal_to_lammps_type_map": CEMENTFF4_TYPE_MAP,
        "angle_type_map": {
            str(k): v for k, v in CEMENTFF4_ANGLE_MAP.items() if zinc_enabled or k <= 3
        },
        "bond_type_map": {str(k): v for k, v in CEMENTFF4_BOND_MAP.items()},
    }
    with open(name, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
        f.write("\n")


def write_cementff4_zinc_input(name, data_file=None):
    write_cementff4_forcefield_from_db(load_cementff4_db(), name)


def write_cementff4_smoke_input(name, data_file, ff_file):
    with open(name, "w") as f:
        f.write("# Minimal pyCSH CementFF4-Zn smoke test\n")
        f.write("clear\n")
        f.write("units metal\n")
        f.write("dimension 3\n")
        f.write("atom_style full\n")
        f.write("boundary p p p\n")
        f.write("box tilt large\n\n")
        f.write("read_data {:}\n".format(os.path.basename(data_file)))
        f.write("include {:}\n".format(os.path.basename(ff_file)))
        f.write("run 0\n")


def write_cementff4_minimize_input(name, data_file, ff_file, minimized_data_file, dump_file):
    with open(name, "w") as f:
        f.write("# Staged pyCSH CementFF4-Zn minimization; no MD is run here.\n")
        f.write("clear\n")
        f.write("units metal\n")
        f.write("dimension 3\n")
        f.write("atom_style full\n")
        f.write("boundary p p p\n")
        f.write("box tilt large\n\n")
        f.write("read_data {:}\n".format(os.path.basename(data_file)))
        f.write("include {:}\n".format(os.path.basename(ff_file)))
        f.write("dump min_dump all custom 100 {:} id type q x y z\n".format(os.path.basename(dump_file)))
        f.write("thermo 1\n")
        f.write("thermo_style custom step pe ebond eangle evdwl ecoul elong fnorm fmax press\n")
        f.write("neighbor 2.0 bin\n")
        f.write("neigh_modify every 1 delay 0 check yes\n")
        f.write("# Stage A: conservative line-search minimization with a small displacement cap.\n")
        f.write("min_style cg\n")
        f.write("min_modify dmax 0.02 line quadratic\n")
        f.write("minimize 1.0e-6 1.0e-8 200 2000\n")
        f.write("# Stage B: slightly larger displacement cap after the worst contacts are reduced.\n")
        f.write("min_modify dmax 0.10 line quadratic\n")
        f.write("minimize 1.0e-6 1.0e-8 1000 10000\n")
        f.write("write_data {:} nocoeff\n".format(os.path.basename(minimized_data_file)))


def get_lammps_input(input_file, entries_crystal, entries_bonds, entries_angle, supercell, unitcell, write_lammps_erica, orthogonal = None, shift = None, diferentiate = None, dpore = None, saturation =None, grid = None):

    N_atom = len(entries_crystal)
    N_bond = len(entries_bonds)
    N_angle = len(entries_angle)
    y=np.zeros(len(entries_angle))
    supercell_orthogonal=np.zeros((3,3))
    shift_done = False
    salt=unitcell[2,2]/2

    coords_Ot = []
    coords_Si = []
    coords_OSi = []
    coords_O = []    
    coords_Ow = []
    coords_Hw = []
    coords_Ca = []
    coords_Ca1 = []
    coords_Ca2 = []
    coords_Si = []
    coords_Si1 = []
    coords_Si2 = []
    coords_Of = []
    coords_OCa = []
    coords_Oh = []
    coords_H = []
    coords_Zn = []
    coords_Mn = []
    
    if orthogonal == True:
        entries_crystal, supercell, unitcell = orthogonal_cell(entries_crystal, supercell, unitcell)
        
    for i in entries_crystal: 
            r = np.array( i[3:] )
            specie = i[1]
            if specie == 5:
                coords_Ow.append(r)
            elif specie == 7:
                coords_Hw.append(r)
            elif specie == 1:
                coords_Ca1.append(r)
                coords_Ca.append(r)                

            elif specie == 9:
                coords_Ca2.append(r)
                coords_Ca.append(r)
            
            elif specie == 2:
                coords_Si.append(r)
                coords_Si1.append(r)

            elif specie == 10:
                coords_Si.append(r)            
                coords_Si2.append(r)

            elif specie == 3:
                coords_Ot.append(r)
                coords_Of.append(r)
            elif specie == 11:
                coords_OCa.append(r)
                coords_Of.append(r)
              
            elif specie == 6:
                coords_Oh.append(r)
                coords_Of.append(r)
         
            elif specie == 8:
                coords_H.append(r)

            elif specie == 12:
                coords_Zn.append(r)
            elif specie == 13:
                coords_Mn.append(r)
    
    if diferentiate==True:    
        for i in entries_crystal:
            r = np.array( i[3:] )
            if i[1] == 3:
                coords_Ot.append(r)    
            if i[1] == 2 or i[1] == 10:
                 coords_Si.append(r)           
        r1=[0,0,0]
        r2=[0,0,0]
        r3=[0,0,0]

        for i in range(0,len(coords_Ot),1):
            r1[0]=coords_Ot[i][0]
            r1[1]=coords_Ot[i][1]
            r1[2]=coords_Ot[i][2]
            Not_Ob = True
            repetido = False
            r=np.array( r1[0:] )
            for j in range(0,len(coords_Si),1):
                r2[0]=coords_Si[j][0]
                r2[1]=coords_Si[j][1]                      
                r2[2]=coords_Si[j][2]     
                dis=((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**(1/2)
                for k in range(0,len(coords_Si),1):
                    r3[0]=coords_Si[k][0]
                    r3[1]=coords_Si[k][1]
                    r3[2]=coords_Si[k][2]                
                    dis1=((r1[0]-r3[0])**2+(r1[1]-r3[1])**2+(r1[2]-r3[2])**2)**(1/2)
                    if dis <= 1.99002 and dis1 <= 1.99002 and j!=k:
                        Not_Ob = False
                        if len(coords_OSi)==0:
                            coords_OSi.append(r)
                            repetido = True
                        else: 
                            for n in range(0,len(coords_OSi),1):
                                if r1[0] == coords_OSi[n][0] and r1[1] == coords_OSi[n][1] and r1[2] == coords_OSi[n][2]:
                                    repetido = True
                            if repetido == False:
                                coords_OSi.append(r)                                                    
                            
            if Not_Ob == True:
                if len(coords_O)==0:
                    coords_O.append(r)
                    repetido = True
                else:
                    for m in range(0,len(coords_O),1):                    
                        if r1[0] == coords_O[m][0] and r1[1] == coords_O[m][1] and r1[2] == coords_O[m][2]:
                            repetido = True
                    if repetido == False:
                        coords_O.append(r)          
        
        for i in entries_crystal:
            if i[1]==3:
                r = np.array( i[3:] )
                for m in range(0,len(coords_OSi),1):
                    if r[0] == coords_OSi[m][0] and r[1] == coords_OSi[m][1] and r[2] == coords_OSi[m][2]:
                        i[1]=12
                        break
                        
    if shift==True :
            if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)              
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            shift_done==True 
            
            
    if dpore != None and dpore != 0:
        n=round(supercell[2,2]/unitcell[2,2])
 
        rel=float(dpore/supercell[2,2])
        if ((n/2) == round(n/2) and shift_done == True) or ((n/2) != round(n/2) and shift_done == False):
            if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)              
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
        if diferentiate == True:
            
            coords_Ca1 = pores_corrected(supercell, salt, n, rel, coords_Ca1)
            coords_Ca2 = pores_corrected(supercell, salt, n, rel, coords_Ca2)            
            coords_Si1 = pores_corrected(supercell, salt, n, rel, coords_Si1)
            coords_Si2 = pores_corrected(supercell, salt, n, rel, coords_Si2)
            coords_OSi = pores_corrected(supercell, salt, n, rel, coords_OSi)
            coords_O = pores_corrected(supercell, salt, n, rel, coords_O)            
            coords_OCa = pores_corrected(supercell, salt, n, rel, coords_OCa)
            coords_Ow,coords_Hw=pores_corrected(supercell, salt, n, rel, coords_Ow, coords_Hw)
            coords_Oh = pores_corrected(supercell, salt, n, rel, coords_Oh)
            coords_H = pores_corrected(supercell, salt, n, rel, coords_H)
            coords_Zn = pores_corrected(supercell, salt, n, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, salt, n, rel, coords_Mn)   
        else:
            coords_Ca = pores_corrected(supercell, salt, n, rel, coords_Ca)
            coords_Si = pores_corrected(supercell, salt, n, rel, coords_Si)
            coords_Of = pores_corrected(supercell, salt, n, rel, coords_Of)            
            coords_Ow,coords_Hw = pores_corrected(supercell, salt, n, rel, coords_Ow, coords_Hw)            
            coords_H = pores_corrected(supercell, salt, n, rel, coords_H)
            coords_Zn = pores_corrected(supercell, salt, n, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, salt, n, rel, coords_Mn)               
        supercell[2,:] += rel*(supercell[2,:])
    

    for entry in entries_crystal:
        if diferentiate == True:
        
            if any(np.array_equal(entry[3:], c) for c in coords_Ca1):
                entry[1] = 1 
        
            if any(np.array_equal(entry[3:], c) for c in coords_Si1):
                entry[1] = 2 
        
            if any(np.array_equal(entry[3:], c) for c in coords_O):
                entry[1] = 3

            if any(np.array_equal(entry[3:], c) for c in coords_Ow):
                entry[1] = 5

            if any(np.array_equal(entry[3:], c) for c in coords_Oh):
                entry[1] = 6

            if any(np.array_equal(entry[3:], c) for c in coords_Hw):
                entry[1] = 7          

            if any(np.array_equal(entry[3:], c) for c in coords_H):
                entry[1] = 8                   

            if any(np.array_equal(entry[3:], c) for c in coords_Ca2):
                entry[1] = 9

            if any(np.array_equal(entry[3:], c) for c in coords_Si2):
                entry[1] = 10
                
            if any(np.array_equal(entry[3:], c) for c in coords_OCa):
                entry[1] = 11          

            if any(np.array_equal(entry[3:], c) for c in coords_OSi):
                entry[1] = 12                   

            if any(np.array_equal(entry[3:], c) for c in coords_Mn):
                entry[1] = 13

            if any(np.array_equal(entry[3:], c) for c in coords_Zn):
                entry[1] = 14
        else:
# Ca
            if any(np.array_equal(entry[3:], c) for c in coords_Ca1) or any(np.array_equal(entry[3:], c) for c in coords_Ca2):
                entry[1] = 1

# Si
            if any(np.array_equal(entry[3:], c) for c in coords_Si1) or any(np.array_equal(entry[3:], c) for c in coords_Si2):
                entry[1] = 2 

# Oxígenos
            if any(np.array_equal(entry[3:], c) for c in coords_O) or any(np.array_equal(entry[3:], c) for c in coords_OSi) or any(np.array_equal(entry[3:], c) for c in coords_OCa):
                entry[1] = 3

# Otros tipos de oxígeno
            if any(np.array_equal(entry[3:], c) for c in coords_Ow):
                entry[1] = 5

            if any(np.array_equal(entry[3:], c) for c in coords_Oh):
                entry[1] = 6

# Hidrógenos
            if any(np.array_equal(entry[3:], c) for c in coords_Hw):
                entry[1] = 7    

            if any(np.array_equal(entry[3:], c) for c in coords_H):
                entry[1] = 8                            

# Manganeso y Zinc
            if any(np.array_equal(entry[3:], c) for c in coords_Mn):
                entry[1] = 13

            if any(np.array_equal(entry[3:], c) for c in coords_Zn):
                entry[1] = 14
    with open(input_file, "w") as f:
        f.write( "Generated with Brickcode \n\n" )
        f.write( "{: 8d} atoms \n".format(N_atom) )
        f.write( "{: 8d} bonds \n".format(N_bond) )
        f.write( "{: 8d} angles \n".format(N_angle) )
        f.write( "{: 8d} atom types \n".format(8) )
        f.write( "{: 8d} bond types \n".format(3) )
        f.write( "{: 8d} angle types \n".format(3) )
        f.write( " \n" )
        f.write( "{: 12.6f} {: 12.6f} xlo xhi \n".format(0.0, supercell[0,0]) )
        f.write( "{: 12.6f} {: 12.6f} ylo yhi \n".format(0.0, supercell[1,1]) )
        f.write( "{: 12.6f} {: 12.6f} zlo zhi \n".format(0.0, supercell[2,2]) )
        f.write( "{: 12.6f} {: 12.6f} {: 12.6f} xy xz yz \n".format( supercell[1,0], supercell[2,0], supercell[2,1] ) )
        f.write( " \n" )
        f.write( "Masses \n" )
        f.write( " \n" )
        f.write( "1 40.08  #Ca_1  \n" )
        f.write( "2 28.10  #Si_1 \n" )        
        f.write( "3 15.59  #O \n" )
        f.write( "4 0.40   #O(S) \n" )
        f.write( "5 16.00  #Ow \n" )
        f.write( "6 16.00  #Oh \n" )
        f.write( "7 1.00   #Hw \n" )
        f.write( "8 1.00   #H \n" )
        f.write( "9 40.08  #Ca_2  \n" )
        f.write( "10 28.10  #Si_2 \n" )
        f.write( "11 15.59  #O-Ca \n" )
        f.write( "12 15.59  #O-Si \n" )
        f.write( "13 54.94  #Mn \n" )
        f.write( "14 65.38  #Zn \n" )


        f.write( " \n" )
        f.write( "Atoms \n" )
        f.write( " \n" )
        fmt = "{: 8d} {: 8d} {: 8d} {: 8.3f} {: 12.6f} {: 12.6f} {: 12.6f}\n"
        molID = 2
        CS_info = []
        for i in entries_crystal:                  
            if i[1] == 3 or i[1] == 12:
                f.write( fmt.format(i[0], molID, *i[1:]) )
                CS_info.append( [i[0], molID] )
            elif i[1] == 4:
                f.write( fmt.format(i[0], molID, *i[1:]) )
                CS_info.append( [i[0], molID] )
                molID += 1    
            else:
                f.write( fmt.format(i[0], 1, *i[1:]) )
                CS_info.append( [i[0], 1] )
        f.write( " \n" )

        f.write( "Bonds \n" )
        f.write( " \n" )
        fmt = "{: 8d} {: 8d} {: 8d} {: 8d} \n"
        for i in entries_bonds:
            f.write( fmt.format(*i) )
        f.write( " \n" )

        f.write( "Angles \n" )
        f.write( " \n" )
        fmt = "{: 8d} {: 8d} {: 8d} {: 8d} {: 8d} \n"
        for i in entries_angle:
            f.write( fmt.format(*i) )
        f.write( " \n" )

        f.write( " \n" )
        fmt = "{: 8d} {: 8d} \n"
        if write_lammps_erica:
            f.write( "CS-Info \n" )
            f.write( "\n" )
            for i in CS_info:
                f.write( fmt.format(*i))


def get_lammps_input_reaxfff(name, entries_crystal, supercell, unitcell, orthogonal = None, shift = None, diferentiate = None, dpore = None, saturation =None , grid = None):

    N_atoms_specie = np.zeros(15,dtype=int)
    N_atoms_specie_1 = np.zeros(15,dtype=int)
    Atom_types = np.zeros(15,dtype=int)
    Atom_types_1 = np.zeros(15,dtype=int)
    coords = [ [] for i in range(16) ]
    entries_crystal_lammps = copy.deepcopy(entries_crystal)
    supercell_lammps = copy.deepcopy(supercell)
    unitcell_lammps = copy.deepcopy(unitcell)

    
    coords_Ca1 = []
    coords_Ca2 = []
    coords_Ca = []    
    coords_Si = []
    coords_Si1 = []
    coords_Si2 = []
    coords_Ot = []
    coords_Ow = []
    coords_Oh = []
    coords_Hw = []
    coords_H = []
    coords_OCa = []
    coords_OSi = []
    coords_O = []    
    coords_Of = []
    coords_Mn = []
    coords_Zn = []
    coords_Na = []
    coords_Cl = []
    entries_crystal_Ow=[]
    entries_crystal_Hw=[]

    #sys.exit()
    if orthogonal == True:
        for entry in entries_crystal_lammps:
            specie = entry[1]        
        
            if specie == 5 :
                entries_crystal_Ow.append(entry)
            elif specie == 7:
                entries_crystal_Hw.append(entry)

        entries_crystal_Ow, entries_crystal_Hw = moleculas_agua(entries_crystal_Ow, entries_crystal_Hw)
 
        entries_crystal_lammps, supercell_lammps, unitcell_lammps = orthogonal_cell(entries_crystal_lammps, supercell_lammps, unitcell_lammps, entries_crystal_Ow, entries_crystal_Hw)
        

    for entry in entries_crystal_lammps:
        r = np.array( entry[3:] )
        specie = entry[1]        

        if specie == 1:
                coords_Ca1.append(r)
                coords_Ca.append(r)                
                N_atoms_specie[0] += 1
                N_atoms_specie_1[0] += 1
                Atom_types[0] = 1
                Atom_types_1[0] = 1

        elif specie == 9:
                coords_Ca2.append(r)
                coords_Ca.append(r)
                N_atoms_specie[1] += 1
                N_atoms_specie_1[0] += 1

                Atom_types[1] = 1
                Atom_types_1[0] = 1

        elif specie == 2:
                coords_Si.append(r)
                coords_Si1.append(r)
                N_atoms_specie[2] += 1
                N_atoms_specie_1[1] += 1
                Atom_types[2] = 1
                Atom_types_1[1] = 1

        elif specie == 10:
                coords_Si.append(r)            
                coords_Si2.append(r)
                N_atoms_specie[3] += 1
                N_atoms_specie_1[1] += 1
                Atom_types[3] = 1
                Atom_types_1[1] = 1

        elif specie == 3:
                coords_Ot.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
                Atom_types_1[2] = 1
                
        elif specie == 11:
                coords_OCa.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
                N_atoms_specie[6] += 1   
                Atom_types[6] = 1
                Atom_types_1[2] = 1

        elif specie == 5:
                coords_Ow.append(r)
                N_atoms_specie[7] += 1
                N_atoms_specie_1[3] += 1
                Atom_types[7] = 1
                Atom_types_1[3] = 1
                
        elif specie == 6:
                coords_Oh.append(r)
                coords_Of.append(r)
                N_atoms_specie[8] += 1
                N_atoms_specie_1[2] += 1
                Atom_types[8] = 1
                Atom_types_1[2] = 1


        elif specie == 7:
                coords_Hw.append(r)
                N_atoms_specie[9] += 1
                N_atoms_specie_1[4] += 1

                Atom_types[9] = 1
                Atom_types_1[4] = 1


        elif specie == 8:
                coords_H.append(r)
                N_atoms_specie[10] += 1
                N_atoms_specie_1[5] += 1

                Atom_types[10] = 1
                Atom_types_1[5] = 1


        elif specie == 12:
                coords_Zn.append(r)
                N_atoms_specie[11] += 1 
                N_atoms_specie_1[6] += 1

                Atom_types[11] = 1
                Atom_types_1[6] = 1


        elif specie == 13:
                coords_Mn.append(r)
                N_atoms_specie[12] += 1 
                N_atoms_specie_1[7] += 1

                Atom_types[12] = 1
                Atom_types_1[7] = 1


        coords[  entry[1]-1 ].append( r )


    coords_Ow, coords_Hw = moleculas_agua(coords_Ow, coords_Hw, True)
   

    if diferentiate==True:
        coords_OSi, coords_O, N_atoms_specie, Atom_types = subtypes(supercell_lammps, coords_Ot, coords_Si, N_atoms_specie, Atom_types)

    if shift==True :
        if diferentiate == True:
            coords_Ca1 = shifting(supercell_lammps, unitcell_lammps, coords_Ca1)
            coords_Ca2 = shifting(supercell_lammps, unitcell_lammps, coords_Ca2)          
            coords_Si1 = shifting(supercell_lammps, unitcell_lammps, coords_Si1)
            coords_Si2 = shifting(supercell_lammps, unitcell_lammps, coords_Si2)
            coords_OSi = shifting(supercell_lammps, unitcell_lammps, coords_OSi)
            coords_O = shifting(supercell_lammps, unitcell_lammps, coords_O)
            coords_OCa = shifting(supercell_lammps, unitcell_lammps, coords_OCa)
            coords_Ow,coords_Hw = shifting(supercell_lammps, unitcell_lammps, coords_Ow,  True, coords_Hw)              
            coords_Oh = shifting(supercell_lammps, unitcell_lammps, coords_Oh)
            coords_H = shifting(supercell_lammps, unitcell_lammps, coords_H)
            coords_Zn = shifting(supercell_lammps, unitcell_lammps, coords_Zn)
            coords_Mn = shifting(supercell_lammps, unitcell_lammps, coords_Mn)  
        else:
            coords_Ca = shifting(supercell_lammps, unitcell_lammps, coords_Ca)
            coords_Si = shifting(supercell_lammps, unitcell_lammps, coords_Si)
            coords_OSi = shifting(supercell_lammps, unitcell_lammps, coords_OSi)
            coords_Of = shifting(supercell_lammps, unitcell_lammps, coords_Of)
            coords_Ow,coords_Hw = shifting(supercell_lammps, unitcell_lammps, coords_Ow,  True, coords_Hw)              
            coords_Oh = shifting(supercell_lammps, unitcell_lammps, coords_Oh)  
            coords_H = shifting(supercell_lammps, unitcell_lammps, coords_H)    
            coords_Zn = shifting(supercell_lammps, unitcell_lammps, coords_Zn)
            coords_Mn = shifting(supercell_lammps, unitcell_lammps, coords_Mn)  
            
            
    if dpore != None and dpore != 0:
        n=round(supercell_lammps[2,2]/unitcell_lammps[2,2])
        rel=float(dpore/supercell_lammps[2,2])

        if diferentiate == True:
            
            coords_Ca1 = pores_corrected(supercell_lammps, n, shift, rel, coords_Ca1)
            coords_Ca2 = pores_corrected(supercell_lammps, n, shift, rel, coords_Ca2)            
            coords_Si1 = pores_corrected(supercell_lammps, n, shift, rel, coords_Si1)
            coords_Si2 = pores_corrected(supercell_lammps, n, shift, rel, coords_Si2)
            coords_OSi = pores_corrected(supercell_lammps, n, shift, rel, coords_OSi)
            coords_O = pores_corrected(supercell_lammps, n, shift, rel, coords_O)            
            coords_OCa = pores_corrected(supercell_lammps, n, shift, rel, coords_OCa)
            coords_Ow,coords_Hw=pores_corrected(supercell_lammps, n, shift, rel, coords_Ow, True, coords_Hw)
            coords_Oh = pores_corrected(supercell_lammps, n, shift, rel, coords_Oh)
            coords_H = pores_corrected(supercell_lammps, n, shift, rel, coords_H)
            coords_Zn = pores_corrected(supercell_lammps, n, shift, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell_lammps, n, shift, rel, coords_Mn)   
            coords_Ca=[]
            for i in range(len(coords_Ca1)):
                coords_Ca.append(coords_Ca1[i])
            for j in range(len(coords_Ca2)):
                coords_Ca.append(coords_Ca2[j])
        else:
            coords_Ca = pores_corrected(supercell_lammps, n, shift, rel, coords_Ca)
            coords_Si = pores_corrected(supercell_lammps, n, shift, rel, coords_Si)
            coords_Of = pores_corrected(supercell_lammps, n, shift, rel, coords_Of)            
            coords_Ow,coords_Hw = pores_corrected(supercell_lammps, n, shift, rel, coords_Ow, True, coords_Hw)  
            coords_H = pores_corrected(supercell_lammps, n, shift, rel, coords_H)
            coords_Zn = pores_corrected(supercell_lammps, n, shift, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell_lammps, n, shift, rel, coords_Mn)               

        if saturation is not None and saturation is not False:
            filled_atoms= mallado(supercell_lammps, rel, grid, n, shift, entries_crystal_lammps, coords_Ca)
            if grid[3] != 0 or grid[3] != None or grid[3] != False:
                filled_atoms = substitute_water(filled_atoms, grid)

            for entry in filled_atoms:
                specie = entry[1]
                r=entry[3:]
                if specie == 7:
                    coords_Hw.append(r)
                    N_atoms_specie[9] += 1
                    N_atoms_specie_1[4] += 1
                elif specie == 5:
                    coords_Ow.append(r)
                    N_atoms_specie[7] += 1
                    N_atoms_specie_1[3] += 1
                elif specie == 14:
                    coords_Na.append(r)
                    N_atoms_specie[13] += 1 
                    N_atoms_specie_1[8] += 1

                    Atom_types[13] = 1
                    Atom_types_1[8] = 1
                elif specie == 15:
                    coords_Cl.append(r)
                    N_atoms_specie[14] += 1 
                    N_atoms_specie_1[9] += 1

                    Atom_types[14] = 1
                    Atom_types_1[9] = 1                
    #print(len(coords_Na))
    #print(len(coords_Cl))
    #sys.exit()
    
    if dpore == None or dpore == 0:
        rel=0           
    supercell_lammps[2,:] += rel*(supercell_lammps[2,:])
    
    with open( name, "w" ) as f:
        f.write( "Generated with Brickcode \n\n" )
        if diferentiate == True:
            f.write( "{: 8d} atoms \n".format(np.sum(N_atoms_specie)) ) 
            f.write( "{: 8d} atom types \n".format(np.sum(Atom_types))) 
        else:
            f.write( "{: 8d} atoms \n".format(np.sum(N_atoms_specie_1)) ) 
            f.write( "{: 8d} atom types \n".format(np.sum(Atom_types_1)))
        f.write( " \n" )
        f.write( "{: 12.6f} {: 12.6f} xlo xhi \n".format(0.0, supercell_lammps[0,0]) )
        f.write( "{: 12.6f} {: 12.6f} ylo yhi \n".format(0.0, supercell_lammps[1,1]) )
        f.write( "{: 12.6f} {: 12.6f} zlo zhi \n".format(0.0, supercell_lammps[2,2]) )
        f.write( "{: 12.6f} {: 12.6f} {: 12.6f} xy xz yz \n".format( supercell_lammps[1,0], supercell_lammps[2,0], supercell_lammps[2,1] ) )
        f.write( " \n" )
        f.write( "Masses \n" )
        f.write( " \n" )
        if diferentiate == True:
            f.write( "1 40.08  #Ca_1  \n" )
            f.write( "2 40.08  #Ca_2  \n" )
            f.write( "3 28.10  #Si_1 \n" )
            f.write( "4 28.10  #Si_2 \n" )   
            f.write( "5 15.79  #O_Ca \n" )
            f.write( "6 15.79  #O_Si \n" )
            f.write( "7 15.79  #O \n" )        
            f.write( "8 15.79  #Ow \n" )
            f.write( "9 15.79  #Oh \n" )
            f.write( "10 1.00  #Hw \n" )
            f.write( "11 1.00  #H \n" )
            f.write( "12 65.38  #Zn \n" )
            f.write( "13 54.94  #Mn \n" )
            f.write( "14 22.99  #Na \n" )
            f.write( "15 35.45  #Cl \n" )
        else:
            f.write( "1 40.08  #Ca  \n" )
            f.write( "2 28.10  #Si \n" )
            f.write( "3 15.79  #O \n" )
            f.write( "4 15.79  #Ow \n" )            
            f.write( "5 1.00   #H \n" )
            f.write( "6 1.00   #Hw \n" )     
            f.write( "12 65.38  #Zn \n" )
            f.write( "13 54.94  #Mn \n" )
            f.write( "14 22.99  #Na \n" )
            f.write( "15 35.45  #Cl \n" )
        f.write( " \n" )
        f.write( "Atoms \n" )
        f.write( " \n" )
        fmt = "{: 8d} {: 8d} {: 8d} {: 12.6f} {: 12.6f} {: 12.6f}\n"

        cont = 1
        types = 0
        if diferentiate == True:
            for i in coords_Ca1:
                types=np.sum(Atom_types[0:1])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Ca2:
                types=np.sum(Atom_types[0:2])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1            
            for i in coords_Si1:
                types=np.sum(Atom_types[0:3])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Si2:
                types=np.sum(Atom_types[0:4])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1   
            for i in coords_OCa:
                types=np.sum(Atom_types[0:5])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_OSi:
                types=np.sum(Atom_types[0:6])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_O:
                types=np.sum(Atom_types[0:7])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Ow:
                types=np.sum(Atom_types[0:8])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Oh:
                types=np.sum(Atom_types[0:9])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1  
            for i in coords_Hw:
                types=np.sum(Atom_types[0:10])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_H:
                types=np.sum(Atom_types[0:11])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Zn:
                types=np.sum(Atom_types[0:12])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Mn:
                types=np.sum(Atom_types[0:13])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1            
            for i in coords_Na:
                types=np.sum(Atom_types[0:14])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Cl:
                types=np.sum(Atom_types[0:15])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1                            
                
        else: 
            for i in coords_Ca:
                types=np.sum(Atom_types_1[0:1])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Si:
                types=np.sum(Atom_types_1[0:2])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Of:
                types=np.sum(Atom_types_1[0:3])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Ow:
                types=np.sum(Atom_types_1[0:4])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1                
            for i in coords_H:
                types=np.sum(Atom_types_1[0:5])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Hw:
                types=np.sum(Atom_types_1[0:6])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Zn:
                types=np.sum(Atom_types_1[0:7])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Mn:
                types=np.sum(Atom_types_1[0:8])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1            
            for i in coords_Na:
                types=np.sum(Atom_types_1[0:9])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1
            for i in coords_Cl:
                types=np.sum(Atom_types_1[0:10])
                f.write( fmt.format(cont, types, 0, *i) )
                cont += 1                                
  
def get_vasp_input(name, entries_crystal, supercell, unitcell, orthogonal = None, shift = None, diferentiate = None, dpore = None , saturation =None, grid = None ):
    N_atoms_specie = np.zeros(13,dtype=int)
    N_atoms_specie_1 = np.zeros(11,dtype=int)
    Atom_types = np.zeros(13,dtype=int)

    shift_done = False
    salt=unitcell[2,2]/2

    #print(supercell[2,:])
    coords = [ [] for i in range(14) ]

    coords_Ca1 = []
    coords_Ca2 = []
    coords_Ca = []    
    coords_Si = []
    coords_Si1 = []
    coords_Si2 = []
    coords_Ot = []
    coords_Ow = []
    coords_Oh = []
    coords_Hw = []
    coords_H = []
    coords_OCa = []
    coords_OSi = []
    coords_O = []    
    coords_Of = []
    coords_Zn = []
    coords_Mn = []
    filled_atoms = []
    
    if orthogonal == True:
        entries_crystal, supercell, unitcell = orthogonal_cell(entries_crystal, supercell, unitcell)

    for entry in entries_crystal:
        specie = entry[1]
        r = np.array( entry[3:] )            
        if specie == 1:
                coords_Ca1.append(r)
                coords_Ca.append(r)                
                N_atoms_specie[0] += 1
                N_atoms_specie_1[0] += 1
        elif specie == 9:
                coords_Ca2.append(r)
                coords_Ca.append(r)
                N_atoms_specie[1] += 1
                N_atoms_specie_1[0] += 1               
        elif specie == 2:
                coords_Si.append(r)
                coords_Si1.append(r)
                N_atoms_specie[2] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 10:
                coords_Si.append(r)            
                coords_Si2.append(r)
                N_atoms_specie[3] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 3:
                coords_Ot.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
        elif specie == 11:
                coords_OCa.append(r)
                coords_Of.append(r)
                N_atoms_specie[6] += 1   
                N_atoms_specie_1[2] += 1                 
        elif specie == 5:
                coords_Ow.append(r)
                N_atoms_specie[7] += 1
                N_atoms_specie_1[3] += 1
        elif specie == 6:
                coords_Oh.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
                N_atoms_specie[8] += 1
        elif specie == 7:
                coords_Hw.append(r)
                N_atoms_specie[9] += 1
                N_atoms_specie_1[4] += 1                
        elif specie == 8:
                coords_H.append(r)
                N_atoms_specie[10] += 1  
                N_atoms_specie_1[5] += 1
        elif specie == 12:
                coords_Zn.append(r)
                N_atoms_specie[11] += 1   
        elif specie == 13:
                coords_Mn.append(r)
                N_atoms_specie[12] += 1           

        coords[  entry[1]-1 ].append( r )
        
    #sys.exit()
 

    if diferentiate==True:
        coords_OSi, coords_O, N_atoms_specie, Atom_types = subtypes(supercell, coords_Ot, coords_Si, N_atoms_specie, Atom_types)
                        
    if shift==True :
        if diferentiate == True:
            coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
            coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
            coords_Si1 = shifting(supercell, unitcell, coords_Si1)
            coords_Si2 = shifting(supercell, unitcell, coords_Si2)
            coords_OSi = shifting(supercell, unitcell, coords_OSi)
            coords_O = shifting(supercell, unitcell, coords_O)
            coords_OCa = shifting(supercell, unitcell, coords_OCa)
            coords_Ow, coords_Hw = shifting(supercell, unitcell, coords_Ow, coords_Hw)

            coords_Oh = shifting(supercell, unitcell, coords_Oh)
            coords_H = shifting(supercell, unitcell, coords_H)
            coords_Zn = shifting(supercell, unitcell, coords_Zn)
            coords_Mn = shifting(supercell, unitcell, coords_Mn)  
        else:
            coords_Ca = shifting(supercell, unitcell, coords_Ca)
            coords_Si = shifting(supercell, unitcell, coords_Si)
            coords_OSi = shifting(supercell, unitcell, coords_OSi)
            coords_Of = shifting(supercell, unitcell, coords_Of)
            coords_Ow, coords_Hw = shifting(supercell, unitcell, coords_Ow, coords_Hw)
            coords_Oh = shifting(supercell, unitcell, coords_Oh)  
            coords_H = shifting(supercell, unitcell, coords_H)    
            coords_Zn = shifting(supercell, unitcell, coords_Zn)
            coords_Mn = shifting(supercell, unitcell, coords_Mn)  
        shift_done=True 
            
            
    if dpore != None and dpore != 0:
        n=round(supercell[2,2]/unitcell[2,2])
        rel=float(dpore/supercell[2,2])
        if ((n/2) == round(n/2) and shift_done == False) or ((n/2) != round(n/2) and shift_done == True):
            print("Automatic shift to center pore")
            if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow, coords_Hw = shifting(supercell, unitcell, coords_Ow, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow, coords_Hw = shifting(supercell, unitcell, coords_Ow, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
        
        if diferentiate == True:
            
            coords_Ca1 = pores_corrected(supercell, n, shift, rel, coords_Ca1)
            coords_Ca2 = pores_corrected(supercell, n, shift, rel, coords_Ca2)            
            coords_Si1 = pores_corrected(supercell, n, shift, rel, coords_Si1)
            coords_Si2 = pores_corrected(supercell, n, shift, rel, coords_Si2)
            coords_OSi = pores_corrected(supercell, n, shift, rel, coords_OSi)
            coords_O = pores_corrected(supercell, n, shift, rel, coords_O)            
            coords_OCa = pores_corrected(supercell, n, shift, rel, coords_OCa)
            coords_Ow,coords_Hw = pores_corrected(supercell, n, shift, rel, coords_Ow, coords_Hw)
            coords_Oh = pores_corrected(supercell, n, shift, rel, coords_Oh)
            coords_H = pores_corrected(supercell, n, shift, rel, coords_H)
            coords_Zn = pores_corrected(supercell, n, shift, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, n, shift, rel, coords_Mn)   
        else:
            coords_Ca = pores_corrected(supercell, n, shift, rel, coords_Ca)
            coords_Si = pores_corrected(supercell, n, shift, rel, coords_Si)
            coords_Of = pores_corrected(supercell, n, shift, rel, coords_Of)            
            coords_Ow,coords_Hw = pores_corrected(supercell, n, shift, rel, coords_Ow, coords_Hw)  
            coords_H = pores_corrected(supercell, n, shift, rel, coords_H)
            coords_Zn = pores_corrected(supercell, n, shift, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, n, shift, rel, coords_Mn)               
            
        if saturation is not None and saturation is not False:
            filled_atoms= mallado(supercell, rel, grid, dpore, entries_crystal)
            #print(len(filled_H))
            for entry in filled_atoms:
                specie = entry[1]
                r=entry[3:]
                if specie == 7:
                    coords_Hw.append(r)
                    N_atoms_specie[9] += 1
                    N_atoms_specie_1[4] += 1
                elif specie == 5:
                        coords_Ow.append(r)
                        N_atoms_specie[7] += 1
                        N_atoms_specie_1[3] += 1
    if dpore == None or dpore == 0:
        rel=0           
    #print(supercell[2,:])
    supercell[2,:] += rel*(supercell[2,:])
    #print(supercell[2,:])
    with open( name, "w" ) as f:
        f.write( "kk \n" )
        f.write( "1.0 \n" )
        if diferentiate == True:
            for i in supercell:
                f.write( "{: 12.6f} {: 12.6f} {: 12.6f} \n".format(*i) )
            f.write( "Ca_1  Cb_2  Si_1  Si_2  OS  O  OC  Ow  Oh  Hw  H Zn Mn \n" )
            f.write( "{: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} \n".format(*N_atoms_specie) )           
        else:          
            for i in supercell:
                f.write( "{: 12.6f} {: 12.6f} {: 12.6f} \n".format(*i) )
            f.write( "Ca  Si  O  Ow   Hw  H  Zn  Mn \n" )
            f.write( "{: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} {: 5d} \n".format(*N_atoms_specie_1) )
        f.write("Cartesian\n")
        fmt = "{: 12.6f} {: 12.6f} {: 12.6f} \n"
        
        #print(len(coords_Ca1),len(coords_Ca2),len(coords_Si1),len(coords_Si2),len(coords_OSi),len(coords_O),len(coords_OCa),len(coords_Ow),len(coords_Oh),len(coords_Hw),len(coords_H))

        if diferentiate == True:
            for i in coords_Ca1:
                f.write( fmt.format(*i) )
            for i in coords_Ca2:
                f.write( fmt.format(*i) )        
            for i in coords_Si1:
                f.write( fmt.format(*i) )
            for i in coords_Si2:
                f.write( fmt.format(*i) )     
            for i in coords_OSi:
                f.write( fmt.format(*i) )                  
            for i in coords_O:
                f.write( fmt.format(*i) )   
            for i in coords_OCa:
                f.write( fmt.format(*i) )      
            for i in coords_Ow:
                f.write( fmt.format(*i) )
            for i in coords_Oh:
                f.write( fmt.format(*i) )
            for i in coords_Hw:
                f.write( fmt.format(*i) )
            for i in coords_H:
                f.write( fmt.format(*i) )
            for i in coords_Zn:
                f.write( fmt.format(*i) )
            for i in coords_Mn:
                f.write( fmt.format(*i) )      
                
        else:
            for i in coords_Ca:
                f.write( fmt.format(*i) )
            for i in coords_Si:
                f.write( fmt.format(*i) )           
            for i in coords_Of:
                f.write( fmt.format(*i) )                    
            for i in coords_Ow:
                f.write( fmt.format(*i) )
            for i in coords_Hw:
                f.write( fmt.format(*i) )
            for i in coords_H:
                f.write( fmt.format(*i) )
            for i in coords_Zn:
                f.write( fmt.format(*i) )
            for i in coords_Mn:
                f.write( fmt.format(*i) )      


def localizar_atomo(coord, n_a):
    for i in range(len(coord)):
        if i==(n_a-1): #número de átomo perdido
            print(coord[i])


def distancia(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2)


def subtypes(supercell, coords_Ot, coords_Si, N_atoms_specie, Atom_types):

    coords_OSi = []
    coords_O = []
    coords_Si_twins = []
    rO = []
    rSi = []
    rt = []
    n=2
    z=0

    for i1 in range(-1,n,1):
        x = i1
        for j1 in range(-1,n,1):
            y = j1
            if (x==1 and y==0) or (x==-1 and y==0) or (x==0 and y==1) or (x==0 and y==-1) or (x==0 and y == 0):
                for p in range(len(coords_Si)):
                    rt = copy.deepcopy(coords_Si[p])
                    rt[:] += x*copy.deepcopy(supercell[0,:])
                    rt[:] += y*copy.deepcopy(supercell[1,:])
                    rt[:] += z*copy.deepcopy(supercell[2,:])
                    coords_Si_twins.append(rt.copy())
    #sys.exit()
    for i in range(len(coords_Ot)):
        rO = copy.deepcopy(coords_Ot[i][:])
        distancias = []

        for j in range(len(coords_Si_twins)):
            rSi = copy.deepcopy(coords_Si_twins[j][:])
            dist = distancia(rO,rSi)
            distancias.append(dist)
        smallest=sorted(distancias)
        if smallest[0] <= 1.99002 and smallest[1] <= 1.99002:
            coords_OSi.append(copy.deepcopy(rO))
            N_atoms_specie[4] += 1
            Atom_types[4] = 1
        else:
            coords_O.append(copy.deepcopy(rO))
            N_atoms_specie[5] += 1
            Atom_types[5] = 1
                #sys.exit()
    return coords_OSi, coords_O, N_atoms_specie, Atom_types


def intento_subtypes(supercell, coords_Ot, coords_Si, N_atoms_specie, Atom_types):                
        r1=[0,0,0]
        r2=[0,0,0]
        r3=[0,0,0]
        for i in range(0,len(coords_Ot),1):
            r1[0]=coords_Ot[i][0]
            r1[1]=coords_Ot[i][1]
            r1[2]=coords_Ot[i][2]
            Not_Ob = True
            repetido = False
            r=np.array( r1[0:] )
            for j in range(0,len(coords_Si_twins),1):
                r2[0]=coords_Si_twins[j][0]
                r2[1]=coords_Si_twins[j][1]                      
                r2[2]=coords_Si_twins[j][2]     
                dis=((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**(1/2)
                for k in range(0,len(coords_Si_twins),1):
                    r3[0]=coords_Si_twins[k][0]
                    r3[1]=coords_Si_twins[k][1]
                    r3[2]=coords_Si_twins[k][2]                
                    dis1=((r1[0]-r3[0])**2+(r1[1]-r3[1])**2+(r1[2]-r3[2])**2)**(1/2)
                    #print(i,j,k)
                    if dis <= 1.99002 and dis1 <= 1.99002 and j!=k:
                        Not_Ob = False
                        if len(coords_OSi)==0:
                            coords_OSi.append(r)
                            repetido = True
                            N_atoms_specie[4] += 1
                            Atom_types[4] = 1

                        else: 
                            for n in range(0,len(coords_OSi),1):
                                if r1[0] == coords_OSi[n][0] and r1[1] == coords_OSi[n][1] and r1[2] == coords_OSi[n][2]:
                                    repetido = True
                            if repetido == False:
                                coords_OSi.append(r)                            
                                N_atoms_specie[4] += 1
                                Atom_types[4] = 1

            #print(len(coords_OSi))
            if Not_Ob == True:
                if len(coords_O)==0:
                    coords_O.append(r)
                    N_atoms_specie[5] += 1
                    Atom_types[5] = 1

                    repetido = True
                else:
                    for m in range(0,len(coords_O),1):                    
                        if r1[0] == coords_O[m][0] and r1[1] == coords_O[m][1] and r1[2] == coords_O[m][2]:
                            repetido = True
                    if repetido == False:
                        N_atoms_specie[5] += 1      
                        Atom_types[5] = 1

                        coords_O.append(r) 
        

def twin_cells(entries_crystal, supercell, x, y, z):
    coord=[]

    entries_crystal_def = copy.deepcopy(entries_crystal)
    for entry in entries_crystal_def:
        r = copy.deepcopy(entry[3:])
        r[:] += x*copy.deepcopy(supercell[0,:])
        r[:] += y*copy.deepcopy(supercell[1,:])
        r[:] += z*copy.deepcopy(supercell[2,:])
        entry[3:] = copy.deepcopy(r[:])
        coord.append(entry)
    return coord    


def orthogonal_cell(entries_crystal, supercell, unitcell, entries_crystal_Ow, entries_crystal_Hw):

    
    #Intento de clipping
    twins = []

    limx = supercell[0,0]
    limy = supercell[1,1]
    limz = supercell[2,2]
    supercell_orto = np.zeros((3,3))
    unitcell_orto = np.zeros((3,3))
    supercell_orto[0,0] = copy.deepcopy(supercell[0,0])
    supercell_orto[1,1] = copy.deepcopy(supercell[1,1])
    supercell_orto[2,2] = copy.deepcopy(supercell[2,2])
    unitcell_orto[0,0] = copy.deepcopy(unitcell[0,0])
    unitcell_orto[1,1] = copy.deepcopy(unitcell[1,1])
    unitcell_orto[2,2] = copy.deepcopy(unitcell[2,2])
    entries_crystal_d = []
    n=1
    for entry in entries_crystal:
        if entry[1] != 5 and entry[1] != 7:
            entries_crystal_d.append(entry.copy())
    corrected_list = []
    while len(corrected_list) <len(entries_crystal):
        n += 1
        orthogonal_entries = []
        orthogonal_entries_Ow = []
        orthogonal_entries_Hw = []
        corrected_list = []

        for i in range(-n,n,1):
            x = i
            for j in range(-n,n,1):
                y = j
                for k in range(-n,n,1):
                    z = k
                    twins = twin_cells(entries_crystal_d, supercell, x, y, z)
                    twins_O = twin_cells(entries_crystal_Ow, supercell, x, y, z)
                    twins_H = twin_cells(entries_crystal_Hw, supercell, x, y, z)
                    for l in range(len(twins)):
                        if twins[l][3]>0 and twins[l][3]<limx:
                            if twins[l][4]>0 and twins[l][4]<limy:
                                if twins[l][5]>0 and twins[l][5]<limz:
                                    orthogonal_entries.append(twins[l].copy())
                    for t in range(len(twins_O)):
                        if twins_O[t][3]>0 and twins_O[t][3]<limx:
                            if twins_O[t][4]>0 and twins_O[t][4]<limy:
                                if twins_O[t][5]>0 and twins_O[t][5]<limz:
                                    orthogonal_entries_Ow.append(twins_O[t].copy())
                    for m in range(len(twins_H)):
                        orthogonal_entries_Hw.append(twins_H[m].copy())
                                    
        orthogonal_entries_Ow, orthogonal_entries_Hw = moleculas_agua(orthogonal_entries_Ow, orthogonal_entries_Hw, True)  

        ordering = 1
        for entry in orthogonal_entries_Ow:
            orthogonal_entries.append(entry)
        for entry in orthogonal_entries_Hw:
            orthogonal_entries.append(entry)        
        for entry in orthogonal_entries:
            if entry not in corrected_list:
                entry[0]=ordering
                ordering += 1
                corrected_list.append(entry.copy())
    return corrected_list, supercell_orto, unitcell_orto


def buscar_indice(lista, array):
    for i, elem in enumerate(lista):
        if np.array_equal(elem, array):
            return i
    return -1


def moleculas_agua(coord_O, coord_H, rep= None ):
    rO=[]
    rH=[]
    coord_H_out=[]

    for i in range(len(coord_O)):
        distancias=[]
        if len(coord_O)>3:
            rO[:]=copy.deepcopy(coord_O[i][3:])

        else:
            rO[:]=copy.deepcopy(coord_O[i])            
        for j in range(len(coord_H)):
            if len(coord_H)>3:
                rH[:]=copy.deepcopy(coord_H[j][3:])
            else:
                rH[:]=copy.deepcopy(coord_H[j])                     
            dist=distancia(rO,rH)
            distancias.append(dist)
        smallest=sorted(distancias)[:2]
 
        indices = [i for i, x in enumerate(distancias) if x == smallest[0]]
        for k in range(len(indices)):
            coord_H_out.append(coord_H[indices[k]].copy())
        if len(indices) == 1:
            index_2= distancias.index(smallest[1]) 
            coord_H_out.append(coord_H[index_2].copy())
    

    return coord_O, coord_H_out
                

def shifting(supercell, unitcell, coord, H = None, coord1 = None):
    r = np.zeros((3))
    coordp = []
    coordp1 = []
    salts = unitcell[2,2]/2
    if H != None:
        for a in range(len(coord)):
            #print(a)
            r=coord[a][:].copy()
            rH_1=coord1[2*a][:].copy()
            rH_2=coord1[2*a+1][:].copy()
            if r[2]<=supercell[2,2]-salts:
                r[:] += unitcell[2,:]/2
                rH_1[:] += unitcell[2,:]/2
                rH_2[:] += unitcell[2,:]/2

            else:
                r[:]=r[:]+unitcell[2,:]/2-supercell[2,:]
                rH_1[:]=rH_1[:]+unitcell[2,:]/2-supercell[2,:]
                rH_2[:]=rH_2[:]+unitcell[2,:]/2-supercell[2,:]

            coordp.append(r.copy())
            coordp1.append(rH_1.copy())
            coordp1.append(rH_2.copy())       
        return coordp, coordp1

    else:
        for i in range(len(coord)):
            r=coord[i][:].copy()
            if r[2]<=supercell[2,2]-salts:
                r[:] += unitcell[2,:]/2
            else:
                r[:]=r[:]+unitcell[2,:]/2-supercell[2,:]
            coordp.append(r)
        return coordp
        
    
def get_xyz_input(name, entries_crystal, supercell, unitcell, orthogonal = None, shift = None, diferentiate = None, dpore = None):

    N_atoms_specie = np.zeros(13,dtype=int)
    N_atoms_specie_1 = np.zeros(11,dtype=int)
    coords = [ [] for i in range(14) ]

    coords_Ca1 = []
    coords_Ca2 = []
    coords_Ca = []    
    coords_Si = []
    coords_Si1 = []
    coords_Si2 = []
    coords_Ot = []
    coords_Ow = []
    coords_Oh = []
    coords_Hw = []
    coords_H = []
    coords_OCa = []
    coords_OSi = []
    coords_O = []    
    coords_Of = []
    coords_Zn = []
    coords_Mn = []
    filled_H = []
    
    if orthogonal == True:
        entries_crystal, supercell, unitcell = orthogonal_cell(entries_crystal, supercell, unitcell)
    for entry in entries_crystal:
        specie = entry[1]
        r = np.array( entry[3:] )            
        if specie == 1:
                coords_Ca1.append(r)
                coords_Ca.append(r)                
                N_atoms_specie[0] += 1
                N_atoms_specie_1[0] += 1
        elif specie == 9:
                coords_Ca2.append(r)
                coords_Ca.append(r)
                N_atoms_specie[1] += 1
                N_atoms_specie_1[0] += 1               
        elif specie == 2:
                coords_Si.append(r)
                coords_Si1.append(r)
                N_atoms_specie[2] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 10:
                coords_Si.append(r)            
                coords_Si2.append(r)
                N_atoms_specie[3] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 3:
                coords_Ot.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
        elif specie == 11:
                coords_OCa.append(r)
                coords_Of.append(r)
                N_atoms_specie[6] += 1   
                N_atoms_specie_1[2] += 1                 
        elif specie == 5:
                coords_Ow.append(r)
                N_atoms_specie[7] += 1
                N_atoms_specie_1[3] += 1
        elif specie == 6:
                coords_Oh.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
                N_atoms_specie[8] += 1
        elif specie == 7:
                coords_Hw.append(r)
                N_atoms_specie[9] += 1
                N_atoms_specie_1[4] += 1                
        elif specie == 8:
                coords_H.append(r)
                N_atoms_specie[10] += 1  
                N_atoms_specie_1[5] += 1
        elif specie == 12:
                coords_Zn.append(r)
                N_atoms_specie[11] += 1   
        elif specie == 13:
                coords_Mn.append(r)
                N_atoms_specie[12] += 1           

        coords[  entry[1]-1 ].append( r )


    together = False
    #sys.exit()
    if together:
        coords_Of = coords_Of + coords_Ow
        N_atoms_specie_1[2] += N_atoms_specie_1[3]
        N_atoms_specie_1[3] = 0
        coords_Ow = []

        coords_H = coords_H + coords_Hw
        N_atoms_specie_1[5] += N_atoms_specie_1[4]
        N_atoms_specie_1[4] = 0
        coords_Hw = []


    if diferentiate == True:
        r1=[0,0,0]
        r2=[0,0,0]
        r3=[0,0,0]
        for i in range(0,len(coords_Ot),1):
            r1[0]=coords_Ot[i][0]
            r1[1]=coords_Ot[i][1]
            r1[2]=coords_Ot[i][2]
            Not_Ob = True
            repetido = False
            r=np.array( r1[0:] )
            for j in range(0,len(coords_Si),1):
                r2[0]=coords_Si[j][0]
                r2[1]=coords_Si[j][1]                      
                r2[2]=coords_Si[j][2]     
                dis=((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**(1/2)
                for k in range(0,len(coords_Si),1):
                    r3[0]=coords_Si[k][0]
                    r3[1]=coords_Si[k][1]
                    r3[2]=coords_Si[k][2]                
                    dis1=((r1[0]-r3[0])**2+(r1[1]-r3[1])**2+(r1[2]-r3[2])**2)**(1/2)
                    if dis <= 1.99002 and dis1 <= 1.99002 and j!=k:
                        Not_Ob = False
                        if len(coords_OSi)==0:
                            coords_OSi.append(r)
                            repetido = True
                            N_atoms_specie[4] += 1
                        else: 
                            for n in range(0,len(coords_OSi),1):
                                if r1[0] == coords_OSi[n][0] and r1[1] == coords_OSi[n][1] and r1[2] == coords_OSi[n][2]:
                                    repetido = True
                            if repetido == False:
                                coords_OSi.append(r)                            
                                N_atoms_specie[4] += 1
            if Not_Ob == True:
                if len(coords_O)==0:
                    coords_O.append(r)
                    N_atoms_specie[5] += 1                
                    repetido = True
                else:
                    for m in range(0,len(coords_O),1):                    
                        if r1[0] == coords_O[m][0] and r1[1] == coords_O[m][1] and r1[2] == coords_O[m][2]:
                            repetido = True
                    if repetido == False:
                        N_atoms_specie[5] += 1                          
                        coords_O.append(r)
            
            
    if dpore != None or dpore != 0:
        salt=unitcell[2,2]/2
        n=round(supercell[2,2]/unitcell[2,2])
        rel=float(dpore/supercell[2,2])
        shift1=(unitcell[2,2]+dpore)/2
        if diferentiate == True:
            coords_Ca1=pores(supercell, salt, n, rel, coords_Ca1)
            coords_Ca2=pores(supercell, salt, n, rel, coords_Ca2)            
            coords_Si1=pores(supercell, salt, n, rel, coords_Si1)
            coords_Si2=pores(supercell, salt, n, rel, coords_Si2)
            coords_OSi=pores(supercell, salt, n, rel, coords_OSi)
            coords_O=pores(supercell, salt, n, rel, coords_O)            
            coords_OCa=pores(supercell, salt, n, rel, coords_OCa)
            coords_Ow,coords_Hw=pores(supercell, salt, n, rel, coords_Ow, coords_Hw)
            coords_Oh=pores(supercell, salt, n, rel, coords_Oh)
            coords_H=pores(supercell, salt, n, rel, coords_H)
            coords_Zn=pores(supercell, salt, n, rel, coords_Zn)
            coords_Mn=pores(supercell, salt, n, rel, coords_Mn)   
        else:
            coords_Ca=pores(supercell, salt, n, rel, coords_Ca)
            coords_Si=pores(supercell, salt, n, rel, coords_Si)
            coords_OSi=pores(supercell, salt, n, rel, coords_OSi)
            coords_Of=pores(supercell, salt, n, rel, coords_Of)            
            coords_Ow,coords_Hw=pores(supercell, salt, n, rel, coords_Ow, coords_Hw)            
            coords_Oh=pores(supercell, salt, n, rel, coords_Oh)
            coords_H=pores(supercell, salt, n, rel, coords_H)
            coords_Zn=pores(supercell, salt, n, rel, coords_Zn)
            coords_Mn=pores(supercell, salt, n, rel, coords_Mn)   
        supercell[2,:]=supercell[2,:]+n*rel*(supercell[2,:])
        

        
    if shift==True:
        if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
        else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)              
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  

    #f.write( " \n" )
    with open( name, "w" ) as f:
        f.write( "{: 12d} \n".format(np.sum(N_atoms_specie)) )
        f.write( " \n" )
        fmt = "{:} {: 12.6f} {: 12.6f} {: 12.6f} \n"
        
        # for i in range(8):
        # #for i in [4, 6]:
        #     for j in coords[i]:
        #         f.write( fmt.format(*j) )

        if diferentiate == True:
            for i in coords_Ca1:
                f.write( fmt.format(*i) )
            for i in coords_Ca2:
                f.write( fmt.format(*i) )        
            for i in coords_Si1:
                f.write( fmt.format(*i) )
            for i in coords_Si2:
                f.write( fmt.format(*i) )     
            for i in coords_OSi:
                f.write( fmt.format(*i) )                  
            for i in coords_O:
                f.write( fmt.format(*i) )   
            for i in coords_OCa:
                f.write( fmt.format(*i) )      
            for i in coords_Ow:
                f.write( fmt.format(*i) )
            for i in coords_Oh:
                f.write( fmt.format(*i) )
            for i in coords_Hw:
                f.write( fmt.format(*i) )
            for i in coords_H:
                f.write( fmt.format(*i) )
            for i in coords_Zn:
                f.write( fmt.format(*i) )
            for i in coords_Mn:
                f.write( fmt.format(*i) )
                
        else:
            for i in coords_Ca:
                f.write( fmt.format(*i) )
            for i in coords_Si:
                f.write( fmt.format(*i) )           
            for i in coords_Of:
                f.write( fmt.format(*i) )                    
            for i in coords_Ow:
                f.write( fmt.format(*i) )
            for i in coords_Hw:
                f.write( fmt.format(*i) )
            for i in coords_H:
                f.write( fmt.format(*i) )
            for i in coords_Zn:
                f.write( fmt.format(*i) )
            for i in coords_Mn:
                f.write( fmt.format(*i) )


def get_log(log_file, size, crystal_rs, water_in_crystal_rs, N_Ca, N_Si, r_SiOH, r_CaOH, MCL, zinc_summary=None ):

    N_Oh = 0
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                N_Oh+= crystal_rs[i,j,k].N_Oh
                
    with open(log_file, "w") as f:
        f.write( "Ca/Si ratio:   {: 8.6f} \n".format(N_Ca/N_Si) )
        f.write( "SiOH/Si ratio: {: 8.6f} \n".format(r_SiOH) )
        f.write( "CaOH/Ca ratio: {: 8.6f} \n".format(r_CaOH) )
        f.write( "MCL:           {: 8.6f} \n".format(MCL) )
        f.write( " \n" )
        if zinc_summary is not None:
            f.write( "Zinc generation: enabled \n" )
            f.write( "Output classification: {:} \n".format(zinc_summary.get("output_classification", "unclassified")) )
            if zinc_summary.get("classification_reasons"):
                f.write( "Classification reasons: {:} \n".format("; ".join(zinc_summary["classification_reasons"])) )
            f.write( "Zn site type:    {:} \n".format(zinc_summary["Zn_site_type"]) )
            f.write( "Zn seed:         {:d} \n".format(zinc_summary["Zn_seed"]) )
            f.write( "Target Zn/Si:    {: 8.6f} \n".format(zinc_summary["target_Zn_Si_ratio"]) )
            f.write( "Actual Zn/Si:    {: 8.6f} \n".format(zinc_summary["actual_Zn_Si_ratio"]) )
            f.write( "Ca/(Si+Zn):      {: 8.6f} \n".format(zinc_summary["Ca_over_Si_plus_Zn_ratio"]) )
            f.write( "N_Si:            {:d} \n".format(zinc_summary["N_Si"]) )
            f.write( "N_Zn:            {:d} \n".format(zinc_summary["N_Zn"]) )
            f.write( "N Os->Oh:        {:d} \n".format(zinc_summary.get("N_Os_converted_to_Oh", 0)) )
            f.write( "N H added ZnOH:  {:d} \n".format(zinc_summary.get("N_H_added_for_Zn_OH", 0)) )
            f.write( "N_Q1_Zn:         {:d} \n".format(zinc_summary["N_Q1_Zn"]) )
            f.write( "N_Q2b_Zn:        {:d} \n".format(zinc_summary["N_Q2b_Zn"]) )
            f.write( "Min Zn-Zn dist:  {:} \n".format(zinc_summary["minimum_Zn_Zn_distance"]) )
            f.write( "Min Zn-O dist:   {:} \n".format(zinc_summary["minimum_Zn_O_distance"]) )
            f.write( "Charge before Zn: {: 8.6f} \n".format(zinc_summary.get("total_charge_before_zinc", 0.0)) )
            f.write( "Charge after Zn before hydroxylation: {: 8.6f} \n".format(
                zinc_summary.get("total_charge_after_zinc_before_hydroxylation", 0.0)
            ) )
            f.write( "Charge after hydroxylation: {: 8.6f} \n".format(
                zinc_summary.get("total_charge_after_hydroxylation", zinc_summary.get("total_charge_residual", 0.0))
            ) )
            f.write( "Charge residual final: {: 8.6f} \n".format(zinc_summary.get("charge_residual_final", 0.0)) )
            if "topology_validation" in zinc_summary:
                f.write( "Remapped O-Zn-O angles: {:d} \n".format(
                    zinc_summary["topology_validation"]["remapped_O_Zn_O_angles"]
                ) )
                f.write( "Remapped Zn-Oh-H angles: {:d} \n".format(
                    zinc_summary["topology_validation"]["remapped_Zn_Oh_H_angles"]
                ) )
            f.write( "Selected Zn sites: \n" )
            for site in zinc_summary["selected_sites"]:
                f.write( "  atom_id={:d} motif={:} cell={:} piece={:} \n".format(
                    site["atom_id"], site["motif"], site["cell"], site["piece"]
                ) )
            f.write( " \n" )
        else:
            f.write( "Zinc generation: disabled \n" )
            f.write( " \n" )
        f.write( "size: {: 3d} {: 3d} {: 3d} \n".format(*size) )
        f.write( " \n" )
        f.write( "Supecell Brick Code: \n" )
        f.write( " Na  Nb  Nc  :   Brick Code \n\n" )
        f.write( "brick_code = { \n" )
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    f.write( "({: 3d}, {: 3d}, {: 3d})  :   {:}, \n".format(i, j, k, crystal_rs[i,j, k].comb) )
        f.write("}\n\n")
        f.write( "water_code = { \n" )
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    f.write( "({: 3d}, {: 3d},{: 3d})  :   {:}, \n".format(i, j, k, water_in_crystal_rs[i,j, k]) )
        f.write("}\n")

        f.write( " \n" )
        f.write( "Charge Distribution: \n" )
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):

                    f.write( "{: 3d} {: 3d} {: 3d}  :   {:} \n".format(i, j, k, crystal_rs[i,j, k].charge) )


def get_siesta_input(name, entries_crystal, supercell, unitcell, orthogonal = None, shift = None, diferentiate = None, dpore = None , saturation =None, grid = None):


    shift_done = False
    salt=unitcell[2,2]/2
    N_atoms_specie = np.zeros(13,dtype=int)
    N_atoms_specie_1 = np.zeros(11,dtype=int)
    coords = [ [] for i in range(14) ]
    coords_Ca1 = []
    coords_Ca2 = []
    coords_Ca = []    
    coords_Si = []
    coords_Si1 = []
    coords_Si2 = []
    coords_Ot = []
    coords_Ow = []
    coords_Oh = []
    coords_Hw = []
    coords_H = []
    coords_OCa = []
    coords_OSi = []
    coords_O = []    
    coords_Of = []
    coords_Zn = []
    coords_Mn = []
    filled_H = []
    
    if orthogonal == True:
        entries_crystal, supercell, unitcell = orthogonal_cell(entries_crystal, supercell, unitcell)

    for entry in entries_crystal:
        specie = entry[1]
        r = np.array( entry[3:] )            
        if specie == 1:
                coords_Ca1.append(r)
                coords_Ca.append(r)                
                N_atoms_specie[0] += 1
                N_atoms_specie_1[0] += 1
        elif specie == 9:
                coords_Ca2.append(r)
                coords_Ca.append(r)
                N_atoms_specie[1] += 1
                N_atoms_specie_1[0] += 1               
        elif specie == 2:
                coords_Si.append(r)
                coords_Si1.append(r)
                N_atoms_specie[2] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 10:
                coords_Si.append(r)            
                coords_Si2.append(r)
                N_atoms_specie[3] += 1
                N_atoms_specie_1[1] += 1
        elif specie == 3:
                coords_Ot.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
        elif specie == 11:
                coords_OCa.append(r)
                coords_Of.append(r)
                N_atoms_specie[6] += 1   
                N_atoms_specie_1[2] += 1                 
        elif specie == 5:
                coords_Ow.append(r)
                N_atoms_specie[7] += 1
                N_atoms_specie_1[3] += 1
        elif specie == 6:
                coords_Oh.append(r)
                coords_Of.append(r)
                N_atoms_specie_1[2] += 1
                N_atoms_specie[8] += 1
        elif specie == 7:
                coords_Hw.append(r)
                N_atoms_specie[9] += 1
                N_atoms_specie_1[4] += 1                
        elif specie == 8:
                coords_H.append(r)
                N_atoms_specie[10] += 1  
                N_atoms_specie_1[5] += 1
        elif specie == 12:
                coords_Zn.append(r)
                N_atoms_specie[11] += 1   
        elif specie == 13:
                coords_Mn.append(r)
                N_atoms_specie[12] += 1           

        coords[  entry[1]-1 ].append( r )



    r1=[0,0,0]
    r2=[0,0,0]
    r3=[0,0,0]

    for i in range(0,len(coords_Ot),1):
        r1[0]=coords_Ot[i][0]
        r1[1]=coords_Ot[i][1]
        r1[2]=coords_Ot[i][2]
        Not_Ob = True
        repetido = False
        r=np.array( r1[0:] )
        for j in range(0,len(coords_Si),1):
            r2[0]=coords_Si[j][0]
            r2[1]=coords_Si[j][1]                      
            r2[2]=coords_Si[j][2]     
            dis=((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**(1/2)
            for k in range(0,len(coords_Si),1):
                r3[0]=coords_Si[k][0]
                r3[1]=coords_Si[k][1]
                r3[2]=coords_Si[k][2]                
                dis1=((r1[0]-r3[0])**2+(r1[1]-r3[1])**2+(r1[2]-r3[2])**2)**(1/2)
                if dis <= 1.99002 and dis1 <= 1.99002 and j!=k:
                    Not_Ob = False
                    if len(coords_OSi)==0:
                        coords_OSi.append(r)
                        repetido = True
                        N_atoms_specie[4] += 1
                    else: 
                        for n in range(0,len(coords_OSi),1):
                            if r1[0] == coords_OSi[n][0] and r1[1] == coords_OSi[n][1] and r1[2] == coords_OSi[n][2]:
                                repetido = True
                        if repetido == False:
                            coords_OSi.append(r)                            
                            N_atoms_specie[4] += 1
                            
        if Not_Ob == True:
            if len(coords_O)==0:
                coords_O.append(r)
                N_atoms_specie[5] += 1                
                repetido = True
            else:
                for m in range(0,len(coords_O),1):                    
                    if r1[0] == coords_O[m][0] and r1[1] == coords_O[m][1] and r1[2] == coords_O[m][2]:
                        repetido = True
                if repetido == False:
                    N_atoms_specie[5] += 1                          
                    coords_O.append(r)              
            
    if shift==True :
            if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)              
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            shift_done==True 
            
            
    if dpore != None and dpore != 0:
        n=(supercell[2,2]/unitcell[2,2])
        rel=float(dpore/supercell[2,2])
        if ((n/2) == round(n/2) and shift_done == True) or ((n/2) != round(n/2) and shift_done == False):
            print('Si',n/2,shift_done)
            if diferentiate == True:
                coords_Ca1 = shifting(supercell, unitcell, coords_Ca1)
                coords_Ca2 = shifting(supercell, unitcell, coords_Ca2)          
                coords_Si1 = shifting(supercell, unitcell, coords_Si1)
                coords_Si2 = shifting(supercell, unitcell, coords_Si2)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_O = shifting(supercell, unitcell, coords_O)
                coords_OCa = shifting(supercell, unitcell, coords_OCa)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Oh = shifting(supercell, unitcell, coords_Oh)
                coords_H = shifting(supercell, unitcell, coords_H)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)  
            else:
                coords_Ca = shifting(supercell, unitcell, coords_Ca)
                coords_Si = shifting(supercell, unitcell, coords_Si)
                coords_OSi = shifting(supercell, unitcell, coords_OSi)
                coords_Of = shifting(supercell, unitcell, coords_Of)
                coords_Ow = shifting(supercell, unitcell, coords_Ow)              
                coords_Oh = shifting(supercell, unitcell, coords_Oh)  
                coords_H = shifting(supercell, unitcell, coords_H)    
                coords_Hw = shifting(supercell, unitcell, coords_Hw)
                coords_Zn = shifting(supercell, unitcell, coords_Zn)
                coords_Mn = shifting(supercell, unitcell, coords_Mn)   

        if diferentiate == True:
            
            coords_Ca1 = pores_corrected(supercell, salt, n, rel, coords_Ca1)
            coords_Ca2 = pores_corrected(supercell, salt, n, rel, coords_Ca2)            
            coords_Si1 = pores_corrected(supercell, salt, n, rel, coords_Si1)
            coords_Si2 = pores_corrected(supercell, salt, n, rel, coords_Si2)
            coords_OSi = pores_corrected(supercell, salt, n, rel, coords_OSi)
            coords_O = pores_corrected(supercell, salt, n, rel, coords_O)            
            coords_OCa = pores_corrected(supercell, salt, n, rel, coords_OCa)
            coords_Ow,coords_Hw=pores_corrected(supercell, salt, n, rel, coords_Ow, coords_Hw)
            coords_Oh = pores_corrected(supercell, salt, n, rel, coords_Oh)
            coords_H = pores_corrected(supercell, salt, n, rel, coords_H)
            coords_Zn = pores_corrected(supercell, salt, n, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, salt, n, rel, coords_Mn)   
        else:
            coords_Ca = pores_corrected(supercell, salt, n, rel, coords_Ca)
            coords_Si = pores_corrected(supercell, salt, n, rel, coords_Si)
            coords_Of = pores_corrected(supercell, salt, n, rel, coords_Of)            
            coords_Ow,coords_Hw = pores_corrected(supercell, salt, n, rel, coords_Ow, coords_Hw)            
            coords_H = pores_corrected(supercell, salt, n, rel, coords_H)
            coords_Zn = pores_corrected(supercell, salt, n, rel, coords_Zn)
            coords_Mn = pores_corrected(supercell, salt, n, rel, coords_Mn)               
        supercell[2,:] += rel*(supercell[2,:])
    #f.write( " \n" )
    with open( name, "w" ) as f:
        f.write( "SystemName        {:} \n".format(name[:-4]) )
        f.write( "SystemLabel       {:} \n".format(name[:-4]) )
        f.write( "NumberOfAtoms     {: 5d} \n".format(np.sum(N_atoms_specie)) )
        f.write( "NumberOfSpecies   {: 5d} \n".format(4) )
        f.write( "NetCharge         {: 5d} \n".format(0) )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "%block ChemicalSpeciesLabel \n" )
        f.write( "1   20   Ca_1 \n" )
        f.write( "2   20   Ca_2 \n" )
        f.write( "3   14   Si_1 \n" )
        f.write( "4   14   Si_2 \n" )        
        f.write( "5   8    O_Ca \n" )
        f.write( "6   8    O_S1 \n" )
        f.write( "7   8    O \n" )        
        f.write( "8   8    Ow \n" )
        f.write( "9   8    Oh \n" )
        f.write( "10   1    Hw \n" )
        f.write( "11   1    H \n" )    
        f.write( "12   65    Zn \n" )
        f.write( "13   55   Mn \n" )    
        f.write( "%endblock ChemicalSpeciesLabel \n" )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "%block PAO.BasisSizes \n" )
        f.write( "Ca_1    DZP \n" )
        f.write( "Ca_2    DZP \n" )
        f.write( "Si_1    DZP \n" )
        f.write( "Si_2   DZP \n" )
        f.write( "O     DZP \n" )
        f.write( "O1     DZP \n" )
        f.write( "Ow     DZP \n" )
        f.write( "Oh     DZP \n" )
        f.write( "Hw     DZP \n" )
        f.write( "H     DZP \n" )
        f.write( "%endblock PAO.BasisSizes \n" )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "%block PAO.Basis \n" )
        f.write( "Ca   5      1.90213 \n" )
        f.write( "n=3   0   1   E    61.56667     4.61281 \n" )
        f.write( "    5.29940 \n" )
        f.write( "          1.00000 \n" )
        f.write( "n=4   0   2   E   164.86383     5.38785 \n" )
        f.write( "      6.76569     4.96452 \n" )
        f.write( "           1.00000     1.00000 \n" )
        f.write( "n=3   1   1   E    86.94959     3.48034 \n" )
        f.write( "        6.32716 \n" )
        f.write( "     1.00000 \n" )
        f.write( "n=4   1   1   E   112.03339     4.98424 \n" )
        f.write( "        7.49434 \n" )
        f.write( "        1.00000 \n" )
        f.write( "n=3   2   1   E    87.65847    5.83989 \n" )
        f.write( "        6.49046 \n" )
        f.write( "         1.00000 \n" )
        f.write( "%endblock PAO.Basis \n" )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "xc.functional          GGA \n" )
        f.write( "xc.authors             PBE \n" )
        f.write( " \n" )
        f.write( "SolutionMethod         diagon \n" )
        f.write( "DM.Tolerance           1.E-4 \n" )
        f.write( "MaxSCFIterations       500 \n" )
        f.write( " \n" )
        f.write( "DM.NumberPulay         5 \n" )
        f.write( "DM.MixingWeight        0.05 \n" )
        f.write( "DM.NumberKick          20 \n" )
        f.write( "DM.KickMixingWeight    0.1 \n" )
        f.write( "DM.MixSCF1            .false. \n" )
        f.write( " \n" )
        f.write( "MeshCutoff             400  Ry \n" )
        f.write( "kgrid_cutoff           25  Bohr \n" )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "%block LatticeVectors \n" )
        for i in range(3):
            f.write( "{: 12.6f} {: 12.6f} {: 12.6f} \n".format( *supercell[i,:] ) )
        f.write( "%endblock LatticeVectors \n" )
        f.write( " \n" )
        f.write( " \n" )
        f.write( "AtomicCoordinatesFormat Ang \n" )
        f.write( "%block AtomicCoordinatesAndAtomicSpecies \n" )
        fmt = "{: 12.6f} {: 12.6f} {: 12.6f} {: 5d} {:} \n"
        if diferentiate == True:
            for i in coords_Ca1:
                f.write( fmt.format(*i, 1, "Ca_1") )
            for i in coords_Ca2:
                f.write( fmt.format(*i, 2, "Ca_2") )        
            for i in coords_Si1:
                f.write( fmt.format(*i, 3, "Si_1") )
            for i in coords_Si2:
                f.write( fmt.format(*i, 4, "Si_2") )           
            for i in coords_OCa:
                f.write( fmt.format(*i, 5, "O_Ca") )
            for i in coords_OSi:
                f.write( fmt.format(*i, 6, "O_Si") )   
            for i in coords_O:
                f.write( fmt.format(*i, 7, "O") )               
            for i in coords_Ow:
                f.write( fmt.format(*i, 8, "Ow") )
            for i in coords_Oh:
                f.write( fmt.format(*i, 9, "Oh") )
            for i in coords_Hw:
                f.write( fmt.format(*i, 10, "Hw") )
            for i in coords_H:
                f.write( fmt.format(*i, 11, "H") )
            for i in coords_Zn:
                f.write( fmt.format(*i, 12, "Zn") )
            for i in coords_Mn:
                f.write( fmt.format(*i, 13, "Mn") )            
        else:
            for i in coords_Ca:
                f.write( fmt.format(*i, 1, "Ca") )
            for i in coords_Si:
                f.write( fmt.format(*i, 2, "Si") )           
            for i in coords_Of:
                f.write( fmt.format(*i, 3, "O") )               
            for i in coords_Ow:
                f.write( fmt.format(*i, 4, "Ow") )
            for i in coords_Hw:
                f.write( fmt.format(*i, 5, "Hw") )
            for i in coords_H:
                f.write( fmt.format(*i, 6, "H") )
            for i in coords_Zn:
                f.write( fmt.format(*i, 12, "Zn") )
            for i in coords_Mn:
                f.write( fmt.format(*i, 13, "Mn") )       
        f.write( "%endblock AtomicCoordinatesAndAtomicSpecies \n" )
        

def get_sorted_log(list_properties):

    sorted_properties = {}
    for i in list_properties:
        Ca_Si = round(i[0], 4)
        SiOH  = round(i[1], 4)
        CaOH  = round(i[2], 4)
        MCL   = round(i[3], 4)

        if Ca_Si in sorted_properties:
            if SiOH in sorted_properties[Ca_Si]:
                if CaOH in sorted_properties[Ca_Si][SiOH]:
                    if MCL in sorted_properties[Ca_Si][SiOH][CaOH]:
                        sorted_properties[Ca_Si][SiOH][CaOH][MCL].append(i[4])
                    else:
                        sorted_properties[Ca_Si][SiOH][CaOH][MCL] = [i[4]]
                else:
                    sorted_properties[Ca_Si][SiOH][CaOH] = {MCL: [i[4]]}
            else:
                sorted_properties[Ca_Si][SiOH] = {CaOH: {MCL: [i[4]]}}
        else:
            sorted_properties[Ca_Si] = {SiOH: {CaOH: {MCL: [i[4]]} } }


    with open("created_samples.log", "w") as f:
        fmt = "Sample: {: 5d}     Ca/Si: {: 8.6f}     SiOH/Si: {: 8.6f}    CaOH/Ca: {: 8.6f}    MCL: {: 8.6f} \n"

        sorted_Ca_Si = sorted(sorted_properties.keys())
        for Ca_Si in sorted_Ca_Si:
            sorted_SiOH = sorted(sorted_properties[Ca_Si].keys())
            for SiOH in sorted_SiOH:
                sorted_CaOH = sorted(sorted_properties[Ca_Si][SiOH].keys())
                for CaOH in sorted_CaOH:
                    sorted_MCL = sorted(sorted_properties[Ca_Si][SiOH][CaOH].keys())
                    for MCL in sorted_MCL:
                        for i in sorted(sorted_properties[Ca_Si][SiOH][CaOH][MCL]):
                            f.write( fmt.format(int(i), Ca_Si, SiOH, CaOH, MCL) )


def write_output( isample, entries_crystal, entries_bonds, entries_angle, size, crystal_rs, water_in_crystal_rs,
                  supercell, N_Ca, N_Si, r_SiOH, r_CaOH, MCL, write_lammps, write_lammps_erica, write_vasp, write_siesta,
                  prefix, unitcell, orthogonal, shift, diferentiate = None, dpore = None, saturation = None , grid = None, guest_ions = None,
                  write_lammps_cementff = False, zinc_summary = None, write_zinc_summary_file = True):

    mypath = os.path.abspath(".")
    path = os.path.join(mypath, "output_Y/")

    if write_lammps_erica:
        name=prefix+"_"+str(isample+1)
        if shift == True:
            name = name +"_shift"
        if orthogonal == True:
            name = name +"_orthogonal"
        if diferentiate == True:
            name = name + "_diferentiate"          
        if dpore != 0 or dpore != None:
            name = name +"_pore="+ str(dpore)   
        name = name +"_erica.data"
        name = os.path.join(path, name)
        get_lammps_input(name, entries_crystal, entries_bonds, entries_angle, supercell, unitcell, write_lammps_erica, orthogonal, shift, diferentiate, dpore, saturation, grid) 
    
    if write_lammps and zinc_summary is None:
        name=prefix+"_reax"+str(isample+1)
        if shift == True:
            name = name +"_shift"
        if orthogonal == True:
            name = name +"_orthogonal"
        if diferentiate == True:
            name = name + "_diferentiate"          
        if dpore != 0 and dpore != None:
            name = name +"_pore="+ str(dpore)
        if saturation != None and saturation != False:
            name = name +"_saturation"
        if guest_ions != None or guest_ions !=False:
            name = name +"_guest_ions"
        name = name +".data"
        name = os.path.join(path, name)
        get_lammps_input_reaxfff(name, entries_crystal, supercell, unitcell, orthogonal, shift, diferentiate, dpore, saturation, grid)
    elif write_lammps and zinc_summary is not None:
        print("Skipping legacy ReaxFF-style LAMMPS writer for Zn-enabled CementFF output.")

    if write_lammps_cementff:
        name=prefix+"_cementff"+str(isample+1)
        if zinc_summary is not None:
            name = name +"_zn"
        name = name +".data"
        name = os.path.join(path, name)
        cementff_metadata = get_lammps_input_cementff(name, entries_crystal, entries_bonds, entries_angle, supercell, zinc_summary)
        update_zinc_classification_for_water(zinc_summary)
        mapping_name = os.path.join(path, prefix+"_cementff_type_mapping_"+str(isample+1)+".json")
        write_cementff4_mapping_json(mapping_name, zinc_summary is not None)
        water_summary_name = os.path.join(path, prefix+"_cementff_water_summary_"+str(isample+1)+".json")
        with open(water_summary_name, "w") as f:
            json.dump(cementff_metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        if zinc_summary is not None:
            ff_name = os.path.join(path, prefix+"_in.CementFF4_Zn_"+str(isample+1))
            write_cementff4_zinc_input(ff_name, name)
            smoke_name = os.path.join(path, prefix+"_smoke_CementFF4_Zn_"+str(isample+1)+".in")
            write_cementff4_smoke_input(smoke_name, name, ff_name)
            min_name = os.path.join(path, prefix+"_minimize_CementFF4_Zn_"+str(isample+1)+".in")
            minimized_data = os.path.join(path, prefix+"_cementff"+str(isample+1)+"_zn_minimized.data")
            minimized_dump = os.path.join(path, prefix+"_cementff"+str(isample+1)+"_zn_minimized.lammpstrj")
            write_cementff4_minimize_input(min_name, name, ff_name, minimized_data, minimized_dump)
            zinc_summary["cementff4_data_file"] = os.path.basename(name)
            zinc_summary["cementff4_forcefield_file"] = os.path.basename(ff_name)
            zinc_summary["cementff4_smoke_input"] = os.path.basename(smoke_name)
            zinc_summary["cementff4_minimize_input"] = os.path.basename(min_name)
            zinc_summary["cementff4_minimized_data_file"] = os.path.basename(minimized_data)
            zinc_summary["cementff4_minimized_dump_file"] = os.path.basename(minimized_dump)

    name = prefix+"_"+str(isample+1)+".log"
    name = os.path.join(path, name)
    get_log(name, size, crystal_rs, water_in_crystal_rs, N_Ca, N_Si, r_SiOH, r_CaOH, MCL, zinc_summary )

    if zinc_summary is not None and write_zinc_summary_file:
        name = prefix+"_zinc_summary_"+str(isample+1)+".json"
        name = os.path.join(path, name)
        write_zinc_summary(name, zinc_summary)
    
    
    if write_vasp:
        name=prefix+"_"+str(isample+1)+"_Y"
        if shift == True:
            name = name +"_shift"
        if orthogonal == True:
            name = name +"_orthogonal"
        if diferentiate == True:
            name = name + "_diferentiate"          
        if dpore != 0 and dpore != None:
            name = name +"_pore = "+ str(dpore)
        if saturation:
            name = name + "_grid"
        name = name + ".vasp"
        name = os.path.join(path, name)
        get_vasp_input(name, entries_crystal, supercell, unitcell, orthogonal, shift, diferentiate, dpore, saturation, grid)
            
        
    if write_siesta:
        name=prefix+"_reax"
        if shift == True:
            name = name +"_shift"
        if orthogonal == True:
            name = name +"_orthogonal"
        if diferentiate == True:
            name = name + "_diferentiate"          
        if dpore != 0 or dpore != None:
            name = name +"_pore="+ str(dpore)   
        name = name +".fdf"
        name = os.path.join(path, name)
        get_siesta_input(name, entries_crystal, supercell, unitcell, orthogonal, shift, diferentiate, dpore, saturation, grid)
