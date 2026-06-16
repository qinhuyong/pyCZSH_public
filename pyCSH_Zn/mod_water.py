import math
import numpy as np


WATER_HARD_CUTOFFS = {
    "water_hard_H_Ca": 1.3,
    "water_hard_H_Si": 1.4,
    "water_hard_H_Zn": 1.3,
    "water_hard_H_O_nonbonded": 1.1,
    "water_hard_H_H_nonbonded": 1.1,
    "water_hard_Ow_O": 1.9,
    "water_hard_Ow_Ca": 1.9,
}

WATER_WARN_CUTOFFS = {
    "water_warn_H_Ca": 1.6,
    "water_warn_H_Si": 1.6,
    "water_warn_H_Zn": 1.4,
    "water_warn_H_O_nonbonded": 1.2,
    "water_warn_H_H_nonbonded": 1.2,
    "water_warn_Ow_O": 2.2,
    "water_warn_Ow_Ca": 2.2,
}

LABELS = {
    1: "Ca",
    9: "Ca",
    2: "Si",
    10: "Si",
    3: "O",
    11: "O",
    4: "O(S)",
    5: "Ow",
    6: "Oh",
    7: "Hw",
    8: "Hoh",
    14: "Zn",
}


def pbc_vector(origin, other, supercell):
    inv_supercell = np.linalg.inv(supercell)
    delta = np.array(other, dtype=float) - np.array(origin, dtype=float)
    frac = np.dot(delta, inv_supercell)
    frac -= np.rint(frac)
    return np.dot(frac, supercell)


def pbc_distance(origin, other, supercell):
    return float(np.linalg.norm(pbc_vector(origin, other, supercell)))


def entry_by_id(entries):
    return {int(entry[0]): entry for entry in entries}


def water_molecules(entries, bonds):
    by_id = entry_by_id(entries)
    waters = []
    for entry in sorted(entries, key=lambda x: int(x[0])):
        if int(entry[1]) != 5:
            continue
        ow_id = int(entry[0])
        hw = []
        bond_ids = []
        for bond in bonds:
            if int(bond[1]) != 2:
                continue
            a1 = int(bond[2])
            a2 = int(bond[3])
            if ow_id not in (a1, a2):
                continue
            other = a2 if a1 == ow_id else a1
            if other in by_id and int(by_id[other][1]) == 7:
                hw.append(other)
                bond_ids.append(int(bond[0]))
        waters.append({"Ow": ow_id, "Hw": sorted(hw), "bond_ids": bond_ids})
    return waters


def nearest_contact(entries, atom_id, target_species, exclude_ids, supercell):
    by_id = entry_by_id(entries)
    atom = by_id[atom_id]
    best = None
    for other in entries:
        other_id = int(other[0])
        if other_id == atom_id or other_id in exclude_ids:
            continue
        if int(other[1]) not in target_species:
            continue
        d = pbc_distance(atom[3:], other[3:], supercell)
        if best is None or d < best["distance"]:
            best = {
                "distance": d,
                "atom_id": other_id,
                "specie": int(other[1]),
                "label": LABELS.get(int(other[1]), str(int(other[1]))),
            }
    return best


def water_contact_metrics(entries, bonds, supercell, water):
    ids = set([water["Ow"]] + water["Hw"])
    metrics = {
        "Ow": water["Ow"],
        "Hw": list(water["Hw"]),
        "Ow_Ca": nearest_contact(entries, water["Ow"], {1, 9}, ids, supercell),
        "Ow_Si": nearest_contact(entries, water["Ow"], {2, 10}, ids, supercell),
        "Ow_O": nearest_contact(entries, water["Ow"], {3, 4, 5, 6, 11}, ids, supercell),
        "Ow_Zn": nearest_contact(entries, water["Ow"], {14}, ids, supercell),
        "H_contacts": [],
    }
    for h_id in water["Hw"]:
        metrics["H_contacts"].append(
            {
                "H": h_id,
                "H_O_nonbonded": nearest_contact(entries, h_id, {3, 4, 5, 6, 11}, ids, supercell),
                "H_H_nonbonded": nearest_contact(entries, h_id, {7, 8}, ids, supercell),
                "H_Ca": nearest_contact(entries, h_id, {1, 9}, ids, supercell),
                "H_Si": nearest_contact(entries, h_id, {2, 10}, ids, supercell),
                "H_Zn": nearest_contact(entries, h_id, {14}, ids, supercell),
            }
        )
    return metrics


def contact_violations(metrics, cutoffs, level):
    prefix = "water_{}_".format(level)
    pairs = [
        ("Ow_O", prefix + "Ow_O"),
        ("Ow_Ca", prefix + "Ow_Ca"),
    ]
    violations = []
    for name, cutoff_name in pairs:
        contact = metrics.get(name)
        if contact is not None and contact["distance"] < cutoffs[cutoff_name]:
            item = {"contact": name, "cutoff": cutoffs[cutoff_name]}
            item.update(contact)
            violations.append(item)
    for h_metrics in metrics["H_contacts"]:
        for name, cutoff_name in [
            ("H_Ca", prefix + "H_Ca"),
            ("H_Si", prefix + "H_Si"),
            ("H_Zn", prefix + "H_Zn"),
            ("H_O_nonbonded", prefix + "H_O_nonbonded"),
            ("H_H_nonbonded", prefix + "H_H_nonbonded"),
        ]:
            contact = h_metrics.get(name)
            if contact is not None and contact["distance"] < cutoffs[cutoff_name]:
                item = {"contact": name, "H": h_metrics["H"], "cutoff": cutoffs[cutoff_name]}
                item.update(contact)
                violations.append(item)
    return violations


def water_geometry_candidates(entries, water, supercell):
    by_id = entry_by_id(entries)
    ow = np.array(by_id[water["Ow"]][3:], dtype=float)
    h1_vec = pbc_vector(ow, by_id[water["Hw"][0]][3:], supercell)
    if np.linalg.norm(h1_vec) < 1.0e-8:
        h1_vec = np.array([1.0, 0.0, 0.0])
    h1_vec = h1_vec / np.linalg.norm(h1_vec)
    axes = [
        h1_vec,
        -h1_vec,
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
    oh = 0.9572
    theta = math.radians(104.52)
    candidates = []
    for axis in axes:
        axis = axis / np.linalg.norm(axis)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(axis, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(axis, ref)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        for phi in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
            ring = math.cos(phi) * perp1 + math.sin(phi) * perp2
            h1 = oh * axis
            h2 = oh * (math.cos(theta) * axis + math.sin(theta) * ring)
            candidates.append((h1, h2))
    return candidates


def set_water_hydrogens(entries, water, h1_vec, h2_vec):
    by_id = entry_by_id(entries)
    ow = np.array(by_id[water["Ow"]][3:], dtype=float)
    by_id[water["Hw"][0]][3:] = list(ow + h1_vec)
    by_id[water["Hw"][1]][3:] = list(ow + h2_vec)


def translate_water(entries, water, vector):
    by_id = entry_by_id(entries)
    for atom_id in [water["Ow"]] + water["Hw"]:
        by_id[atom_id][3:] = list(np.array(by_id[atom_id][3:], dtype=float) + vector)


def repair_water_hard_contacts(entries, bonds, supercell, water, hard_cutoffs=None):
    hard_cutoffs = dict(WATER_HARD_CUTOFFS if hard_cutoffs is None else hard_cutoffs)
    by_id = entry_by_id(entries)
    saved = {atom_id: list(by_id[atom_id][3:]) for atom_id in [water["Ow"]] + water["Hw"]}
    best = None
    best_count = 999
    best_min = -1.0

    for h1, h2 in water_geometry_candidates(entries, water, supercell):
        set_water_hydrogens(entries, water, h1, h2)
        metrics = water_contact_metrics(entries, bonds, supercell, water)
        violations = contact_violations(metrics, hard_cutoffs, "hard")
        distances = []
        for value in metrics.values():
            if isinstance(value, dict) and "distance" in value:
                distances.append(value["distance"])
        for h_metrics in metrics["H_contacts"]:
            for value in h_metrics.values():
                if isinstance(value, dict) and "distance" in value:
                    distances.append(value["distance"])
        min_dist = min(distances) if distances else 0.0
        if len(violations) < best_count or (len(violations) == best_count and min_dist > best_min):
            best = (h1, h2, violations)
            best_count = len(violations)
            best_min = min_dist
        if not violations:
            return "rotation", []

    for atom_id, coord in saved.items():
        by_id[atom_id][3:] = coord
    if best is not None:
        set_water_hydrogens(entries, water, best[0], best[1])

    ow = np.array(by_id[water["Ow"]][3:], dtype=float)
    repel = []
    for other in entries:
        other_id = int(other[0])
        if other_id in [water["Ow"]] + water["Hw"]:
            continue
        d = pbc_distance(ow, other[3:], supercell)
        if d < 3.0:
            v = -pbc_vector(ow, other[3:], supercell)
            if np.linalg.norm(v) > 1.0e-8:
                repel.append(v / np.linalg.norm(v))
    if not repel:
        repel = [np.array([1.0, 0.0, 0.0])]
    direction = np.sum(repel, axis=0)
    if np.linalg.norm(direction) < 1.0e-8:
        direction = repel[0]
    direction = direction / np.linalg.norm(direction)

    for step in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        trial_saved = {atom_id: list(by_id[atom_id][3:]) for atom_id in [water["Ow"]] + water["Hw"]}
        translate_water(entries, water, step * direction)
        metrics = water_contact_metrics(entries, bonds, supercell, water)
        violations = contact_violations(metrics, hard_cutoffs, "hard")
        if not violations:
            return "translation", []
        for atom_id, coord in trial_saved.items():
            by_id[atom_id][3:] = coord
    metrics = water_contact_metrics(entries, bonds, supercell, water)
    return "failed", contact_violations(metrics, hard_cutoffs, "hard")


def screen_and_repair_waters(entries, bonds, supercell, hard_cutoffs=None, warn_cutoffs=None):
    hard_cutoffs = dict(WATER_HARD_CUTOFFS if hard_cutoffs is None else hard_cutoffs)
    warn_cutoffs = dict(WATER_WARN_CUTOFFS if warn_cutoffs is None else warn_cutoffs)
    waters = water_molecules(entries, bonds)
    report = {
        "n_water": len(waters),
        "hard_fail_before": [],
        "repaired_by_rotation": [],
        "repaired_by_translation": [],
        "rejected": [],
        "warning_contacts": [],
        "final_min_contacts": {},
    }
    for water in waters:
        if len(water["Hw"]) != 2:
            report["rejected"].append({"Ow": water["Ow"], "reason": "not exactly two H"})
            continue
        metrics = water_contact_metrics(entries, bonds, supercell, water)
        hard = contact_violations(metrics, hard_cutoffs, "hard")
        if hard:
            report["hard_fail_before"].append({"Ow": water["Ow"], "Hw": list(water["Hw"]), "violations": hard})
            method, remaining = repair_water_hard_contacts(entries, bonds, supercell, water, hard_cutoffs)
            if method == "rotation":
                report["repaired_by_rotation"].append({"Ow": water["Ow"], "Hw": list(water["Hw"])})
            elif method == "translation":
                report["repaired_by_translation"].append({"Ow": water["Ow"], "Hw": list(water["Hw"])})
            else:
                report["rejected"].append({"Ow": water["Ow"], "Hw": list(water["Hw"]), "violations": remaining})

    final_min = {}
    for water in waters:
        metrics = water_contact_metrics(entries, bonds, supercell, water)
        warn = contact_violations(metrics, warn_cutoffs, "warn")
        if warn:
            report["warning_contacts"].append({"Ow": water["Ow"], "Hw": list(water["Hw"]), "violations": warn})
        for key, value in metrics.items():
            if key == "H_contacts":
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
    return entries, report

