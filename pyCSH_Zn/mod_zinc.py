import json
import math
import os
import copy
import hashlib

import numpy as np


ZN_SPECIE = 14
ZN_CHARGE = 2.0
CHARGE_TOLERANCE = 1.0e-6
O_CORE_CHARGE = 0.84819
O_SHELL_CHARGE = -2.84819
OH_CHARGE = -1.4
HOH_CHARGE = 0.4
OH_BOND_DISTANCE = 1.0
ZN_O_CUTOFF = 2.3
H_OVERLAP_CUTOFF = 0.75
H_MIN_DISTANCES = {
    "H_H": 1.2,
    "H_O_nonbonded": 1.2,
    "H_Si": 1.6,
    "H_Ca": 1.6,
    "H_Zn": 1.4,
}
CONVERTED_O_MIN_DISTANCES = {
    "O_Ca": 1.6,
    "O_Si": 1.45,
    "O_O_nonbonded": 1.2,
    "O_Zn": 1.7,
}

SUPPORTED_CHARGE_BALANCE_MODES = {
    "hydroxylate_two_oxygens",
    "fail_if_not_neutral",
    "allow_unbalanced_for_debug",
    "add_interlayer_Ca",
}

CEMENTFF4_TYPE_MAP = {
    1: {"lammps_type": 1, "label": "Ca", "source": "Ca1"},
    9: {"lammps_type": 1, "label": "Ca", "source": "Ca2"},
    2: {"lammps_type": 2, "label": "Si", "source": "Si1"},
    10: {"lammps_type": 2, "label": "Si", "source": "Si2"},
    3: {"lammps_type": 3, "label": "O", "source": "O core"},
    11: {"lammps_type": 3, "label": "O", "source": "OCa core"},
    4: {"lammps_type": 4, "label": "O(S)", "source": "O shell"},
    5: {"lammps_type": 5, "label": "Ow", "source": "water oxygen"},
    6: {"lammps_type": 6, "label": "Oh", "source": "hydroxide oxygen"},
    7: {"lammps_type": 7, "label": "Hw", "source": "water hydrogen"},
    8: {"lammps_type": 8, "label": "Hoh", "source": "hydroxide hydrogen"},
    14: {"lammps_type": 9, "label": "Zn", "source": "Zn substituted Si site"},
}

CEMENTFF4_ANGLE_MAP = {
    1: "Hw-Ow-Hw",
    2: "O-Si-O / Oh-Si-O / Oh-Si-Oh",
    3: "Si-Oh-H",
    4: "O-Zn-O / Oh-Zn-O / Oh-Zn-Oh",
    5: "Zn-Oh-H",
}

Q1_PIECES = {"<L", "<R", ">L", ">R", "<Lo", "<Ro", ">Lo", ">Ro"}
Q2B_PIECES = {"SU", "SD", "SUo", "SDo"}
SUPPORTED_SITE_TYPES = {"Q1_Zn", "Q2b_Zn"}
UNSUPPORTED_SITE_TYPES = {"mixed_Q1_Q2b_Zn", "interlayer_Zn", "Ca_substitution_control"}
OXYGEN_LIKE_TYPES = {3, 4, 5, 6, 11, 12}
ZN_COORDINATION_O_TYPES = {3, 5, 6, 11}
Q1_ZN_O_TARGET = 1.95
Q1_ZN_O_CUTOFF = 3.2
SUPPORTED_Q1_SELECTION_MODES = {"ranked_static", "first_valid", "screening_debug"}


def validate_zinc_site_type(site_type):
    if site_type in UNSUPPORTED_SITE_TYPES:
        raise NotImplementedError(site_type + " is not implemented in v0")
    if site_type not in SUPPORTED_SITE_TYPES:
        raise ValueError(
            "Unknown Zn_site_type {!r}. Expected one of {}".format(
                site_type, sorted(SUPPORTED_SITE_TYPES | UNSUPPORTED_SITE_TYPES)
            )
        )


def validate_charge_balance_mode(mode):
    if mode not in SUPPORTED_CHARGE_BALANCE_MODES:
        raise ValueError(
            "Unknown Zn_charge_balance_mode {!r}. Expected one of {}".format(
                mode, sorted(SUPPORTED_CHARGE_BALANCE_MODES)
            )
        )
    if mode == "add_interlayer_Ca":
        raise NotImplementedError("add_interlayer_Ca is not safely implemented in v2")


def inspect_zinc_candidates(crystal_dict):
    """Return Q1-like and Q2b-like Si centers from the expanded crystal."""
    q1_sites = []
    q2b_sites = []

    for cell, brick_dict in crystal_dict.items():
        for piece_name, piece_entries in brick_dict.items():
            if piece_name not in Q1_PIECES and piece_name not in Q2B_PIECES:
                continue

            si_entries = [entry for entry in piece_entries if entry[1] in (2, 10)]
            if not si_entries:
                raise ValueError(
                    "Cannot identify a Si center for Zn candidate piece "
                    "{!r} at cell {}".format(piece_name, cell)
                )

            for si_entry in si_entries:
                site = {
                    "atom_id": int(si_entry[0]),
                    "cell": [int(cell[0]), int(cell[1]), int(cell[2])],
                    "piece": piece_name,
                    "motif": "Q1_Zn" if piece_name in Q1_PIECES else "Q2b_Zn",
                    "coord": [float(si_entry[3]), float(si_entry[4]), float(si_entry[5])],
                    "original_specie": int(si_entry[1]),
                    "original_charge": float(si_entry[2]),
                }
                if piece_name in Q1_PIECES:
                    q1_sites.append(site)
                else:
                    q2b_sites.append(site)

    if not q1_sites and not q2b_sites:
        raise ValueError("No Q1-like or Q2b-like silicate sites were identified for Zn placement")

    return {"Q1_Zn": q1_sites, "Q2b_Zn": q2b_sites}


def inspect_q1_zinc_candidates(crystal_dict):
    return inspect_zinc_candidates(crystal_dict)["Q1_Zn"]


def oxygen_role_label(specie):
    if int(specie) in (3, 11):
        return "O_core"
    if int(specie) in (4, 12):
        return "O_shell"
    if int(specie) == 5:
        return "Ow"
    if int(specie) == 6:
        return "Oh"
    if int(specie) == 7:
        return "Hw"
    if int(specie) == 8:
        return "Hoh"
    return "other O"


def is_oxygen_like(specie):
    return int(specie) in OXYGEN_LIKE_TYPES


def q1_precondition_score(site_report):
    score = 0.0
    n_safe = int(site_report.get("n_safe_terminal_oxygen", 0))
    score += 30.0 * n_safe
    distances = [
        float(item["distance"])
        for item in site_report.get("neighboring_O_atoms", [])
        if item.get("safe_for_default_hydroxylation")
    ]
    if distances:
        score -= 8.0 * abs(min(distances) - Q1_ZN_O_TARGET)
        score -= 2.0 * abs(sum(distances) / float(len(distances)) - Q1_ZN_O_TARGET)
    if not site_report.get("passed_preconditions", False):
        score -= 100.0
    return float(score)


def q1_nearest_oxygen_records(entries_crystal, supercell, zn_atom_id, intended_oxygen_ids=None, hydroxylated_oxygen_ids=None, limit=4):
    coords = coords_by_atom_id(entries_crystal)
    atom_types = type_by_atom_id(entries_crystal)
    zn_coord = coords[int(zn_atom_id)]
    intended_oxygen_ids = {int(x) for x in (intended_oxygen_ids or [])}
    hydroxylated_oxygen_ids = {int(x) for x in (hydroxylated_oxygen_ids or [])}
    records = []
    for atom_id, specie in atom_types.items():
        if int(specie) not in ZN_COORDINATION_O_TYPES:
            continue
        distance = periodic_distance(zn_coord, coords[atom_id], supercell)
        records.append(
            {
                "atom_id": int(atom_id),
                "atom_type": int(specie),
                "atom_label": oxygen_role_label(specie),
                "distance": float(distance),
                "belongs_to_intended_motif": bool(int(atom_id) in intended_oxygen_ids),
                "oxygen_role": oxygen_role_label(specie),
                "is_hydroxylated_oxygen": bool(int(atom_id) in hydroxylated_oxygen_ids),
            }
        )
    records.sort(key=lambda item: (item["distance"], item["atom_id"]))
    return records[:limit], records


def q1_motif_geometry(entries_crystal, entries_angle, supercell, zn_atom_id, intended_oxygen_ids, hydroxylated_oxygen_ids):
    coords = coords_by_atom_id(entries_crystal)
    intended_ids = [int(x) for x in intended_oxygen_ids]
    zn_coord = coords[int(zn_atom_id)]
    oxygen_coords = []
    all_pairs = []
    if len(intended_ids) >= 2:
        for oid in intended_ids:
            oxygen_coords.append((oid, coords[oid]))
    for i, oid_i in enumerate(intended_ids):
        for oid_j in intended_ids[i + 1:]:
            v1 = vector_pbc(zn_coord, coords[oid_i], supercell)
            v2 = vector_pbc(zn_coord, coords[oid_j], supercell)
            all_pairs.append(
                {
                    "atom_id_1": int(oid_i),
                    "atom_id_2": int(oid_j),
                    "angle_deg": angle_degrees(v1, v2),
                    "deviation_from_tetrahedral_deg": None,
                }
            )
            if all_pairs[-1]["angle_deg"] is not None:
                all_pairs[-1]["deviation_from_tetrahedral_deg"] = abs(all_pairs[-1]["angle_deg"] - 109.47)
    zn_o_distances = [
        {
            "atom_id": int(oid),
            "distance": float(periodic_distance(zn_coord, coords[oid], supercell)),
            "oxygen_role": oxygen_role_label(type_by_atom_id(entries_crystal)[oid]),
            "is_hydroxylated_oxygen": bool(int(oid) in set(int(x) for x in hydroxylated_oxygen_ids or [])),
        }
        for oid in intended_ids
        if oid in coords
    ]
    oo_separations = []
    for i, oid_i in enumerate(intended_ids):
        for oid_j in intended_ids[i + 1:]:
            oo_separations.append(
                {
                    "atom_id_1": int(oid_i),
                    "atom_id_2": int(oid_j),
                    "distance": float(periodic_distance(coords[oid_i], coords[oid_j], supercell)),
                }
            )
    distances = [item["distance"] for item in zn_o_distances]
    angle_deviations = [item["deviation_from_tetrahedral_deg"] for item in all_pairs if item["deviation_from_tetrahedral_deg"] is not None]
    min_oo = min([item["distance"] for item in oo_separations]) if oo_separations else None
    max_dist = max(distances) if distances else None
    mean_dist = sum(distances) / len(distances) if distances else None
    mean_angle_dev = sum(angle_deviations) / len(angle_deviations) if angle_deviations else None
    max_angle_dev = max(angle_deviations) if angle_deviations else None
    hydroxylated_ids = {int(x) for x in (hydroxylated_oxygen_ids or [])}
    hydroxylated_in_intended = [oid for oid in intended_ids if oid in hydroxylated_ids]
    reasonable = bool(
        len(intended_ids) >= 4
        and len(hydroxylated_in_intended) >= 2
        and max_dist is not None
        and max_dist <= 2.8
        and mean_angle_dev is not None
        and mean_angle_dev <= 45.0
        and min_oo is not None
        and min_oo >= 1.4
    )
    return {
        "zn_atom_id": int(zn_atom_id),
        "intended_oxygen_ids": [int(x) for x in intended_ids],
        "hydroxylated_oxygen_ids": sorted(hydroxylated_ids),
        "nearest_four_zn_o_atoms": zn_o_distances,
        "all_nearest_zn_o_atoms": [
            {
                "atom_id": int(item["atom_id"]),
                "atom_type": int(item["atom_type"]),
                "atom_label": item["atom_label"],
                "distance": float(item["distance"]),
                "belongs_to_intended_motif": bool(item["belongs_to_intended_motif"]),
                "oxygen_role": item["oxygen_role"],
                "is_hydroxylated_oxygen": bool(item["is_hydroxylated_oxygen"]),
            }
            for item in q1_nearest_oxygen_records(entries_crystal, supercell, zn_atom_id, intended_oxygen_ids, hydroxylated_oxygen_ids, limit=9999)[1]
        ],
        "Zn_O_distances_A": {
            "count": len(distances),
            "mean": None if mean_dist is None else float(mean_dist),
            "max": None if max_dist is None else float(max_dist),
        },
        "O_Zn_O_angles_deg": all_pairs,
        "tetrahedral_angle_deviation_deg": {
            "count": len(angle_deviations),
            "mean": None if mean_angle_dev is None else float(mean_angle_dev),
            "max": None if max_angle_dev is None else float(max_angle_dev),
        },
        "minimum_O_O_separation_A": None if min_oo is None else float(min_oo),
        "hydroxylated_in_intended_motif": [int(x) for x in hydroxylated_in_intended],
        "reasonable_zn_o2oh2_like_geometry": reasonable,
    }


def q1_selection_score(geometry, precondition_score=0.0):
    if not geometry:
        return float(precondition_score - 1000.0)
    score = float(precondition_score)
    dist_stats = geometry.get("Zn_O_distances_A", {})
    angle_stats = geometry.get("tetrahedral_angle_deviation_deg", {})
    min_oo = geometry.get("minimum_O_O_separation_A")
    if dist_stats.get("max") is not None:
        score -= 25.0 * max(0.0, float(dist_stats["max"]) - Q1_ZN_O_TARGET)
    if dist_stats.get("mean") is not None:
        score -= 5.0 * abs(float(dist_stats["mean"]) - Q1_ZN_O_TARGET)
    if angle_stats.get("mean") is not None:
        score -= 1.5 * float(angle_stats["mean"])
    if angle_stats.get("max") is not None:
        score -= 0.5 * float(angle_stats["max"])
    if min_oo is not None:
        score += 4.0 * float(min_oo)
    if geometry.get("reasonable_zn_o2oh2_like_geometry"):
        score += 50.0
    if len(geometry.get("hydroxylated_in_intended_motif", [])) >= 2:
        score += 10.0
    return float(score)


def q1_static_tie_breaker(site, seed):
    key = "{}:{}:{}:{}".format(
        int(seed),
        int(site["atom_id"]),
        site.get("piece", ""),
        ",".join(str(x) for x in site.get("cell", [])),
    )
    digest = hashlib.sha256(key.encode("ascii")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def q1_static_rank_tuple(site, seed):
    report = site.get("q1_selection_report", {}) or {}
    geom = report.get("q1_geometry_diagnostics", {}) or {}
    dist = geom.get("Zn_O_distances_A", {}) or {}
    dev = geom.get("tetrahedral_angle_deviation_deg", {}) or {}
    hydroxylated = report.get("selected_hydroxylated_oxygen_records", [])
    hydroxylated_safe = all(
        item.get("oxygen_class") == "terminal/non-bridging O"
        for item in hydroxylated
    ) and len(hydroxylated) >= 2
    intended_count = len(geom.get("intended_oxygen_ids", []))
    reasonable = bool(geom.get("reasonable_zn_o2oh2_like_geometry"))
    all_neighbors = report.get("all_nearest_zn_o_atoms", [])
    non_motif_distances = [
        float(item["distance"])
        for item in all_neighbors
        if not item.get("belongs_to_intended_motif", False)
    ]
    shell_isolation = min(non_motif_distances) if non_motif_distances else 999.0
    secondary_shell_reactive_count = sum(
        1
        for item in all_neighbors
        if (
            not item.get("belongs_to_intended_motif", False)
            and item.get("oxygen_role") in ("Oh", "Ow")
            and float(item.get("distance", 999.0)) <= 3.7
        )
    )
    def q(value, digits=6):
        return round(float(value), digits)
    return (
        not bool(site.get("q1_passed_preconditions", False)),
        not bool(hydroxylated_safe),
        not reasonable,
        -q(report.get("selection_score", -1.0e9)),
        q(dist.get("max") if dist.get("max") is not None else 999.0),
        q(abs(float(dist.get("mean") if dist.get("mean") is not None else 999.0) - Q1_ZN_O_TARGET)),
        q(dev.get("mean") if dev.get("mean") is not None else 999.0),
        q(dev.get("max") if dev.get("max") is not None else 999.0),
        -int(intended_count >= 4),
        q(shell_isolation),
        int(secondary_shell_reactive_count),
        -q(geom.get("minimum_O_O_separation_A") if geom.get("minimum_O_O_separation_A") is not None else -999.0),
        q1_static_tie_breaker(site, seed),
    )


def q1_selection_record(site, rank, mode, seed, screened_postmin_valid_atom_ids=None):
    report = site.get("q1_selection_report", {}) or {}
    geom = report.get("q1_geometry_diagnostics", {}) or {}
    dist = geom.get("Zn_O_distances_A", {}) or {}
    dev = geom.get("tetrahedral_angle_deviation_deg", {}) or {}
    all_neighbors = report.get("all_nearest_zn_o_atoms", [])
    non_motif_distances = [
        float(item["distance"])
        for item in all_neighbors
        if not item.get("belongs_to_intended_motif", False)
    ]
    shell_isolation = min(non_motif_distances) if non_motif_distances else None
    secondary_shell_reactive_count = sum(
        1
        for item in all_neighbors
        if (
            not item.get("belongs_to_intended_motif", False)
            and item.get("oxygen_role") in ("Oh", "Ow")
            and float(item.get("distance", 999.0)) <= 3.7
        )
    )
    screened_ids = {int(x) for x in (screened_postmin_valid_atom_ids or [])}
    return {
        "selection_mode": mode,
        "rank": int(rank),
        "selected_site_atom_id": int(site["atom_id"]),
        "selected_piece_name": site.get("piece"),
        "selected_cell": site.get("cell"),
        "deterministic_tie_breaker": q1_static_tie_breaker(site, seed),
        "score": float(report.get("selection_score", -1.0e9)),
        "score_components": report.get("selection_score_components", {}),
        "geometry_score_inputs": {
            "max_intended_Zn_O_distance": dist.get("max"),
            "mean_intended_Zn_O_distance": dist.get("mean"),
            "mean_tetrahedral_angle_deviation_from_109p47": dev.get("mean"),
            "max_tetrahedral_angle_deviation_from_109p47": dev.get("max"),
            "minimum_O_O_distance_among_motif_O": geom.get("minimum_O_O_separation_A"),
            "nearest_non_motif_O_distance": shell_isolation,
            "secondary_shell_reactive_O_count_within_3p7A": int(secondary_shell_reactive_count),
            "secondary_shell_policy": "prefer a compact but non-overlapping static oxygen environment; no post-min result is used",
            "reasonable_zn_o2oh2_like_geometry": geom.get("reasonable_zn_o2oh2_like_geometry"),
        },
        "hydroxylated_oxygen_ids": report.get("selected_hydroxylated_oxygen_ids", []),
        "hydroxylated_oxygen_records": report.get("selected_hydroxylated_oxygen_records", []),
        "intended_four_oxygen_atoms": report.get("pre_minimization_nearest_four_zn_o_atoms", []),
        "matches_previously_screened_postmin_valid_class": bool(int(site["atom_id"]) in screened_ids),
        "note": (
            "Normal Q1 generation uses static pre-min ranking only; post-min screening results are not required "
            "unless examples/10_screen_q1_motifs.py is run explicitly."
        ),
    }


def build_q1_selection_policy(candidates, selected_sites, seed, mode):
    q1_sites = list(candidates.get("Q1_Zn", []))
    ranked_sites = sorted(q1_sites, key=lambda site: q1_static_rank_tuple(site, seed))
    selected_ids = {int(site["atom_id"]) for site in selected_sites}
    records = []
    selected_rank = None
    for idx, site in enumerate(ranked_sites, start=1):
        report = site.get("q1_selection_report", {}) or {}
        rejection = report.get("rejection_reason")
        if int(site["atom_id"]) in selected_ids:
            selected_rank = idx
            rejection = None
        elif not site.get("q1_passed_preconditions", False):
            rejection = rejection or "failed Q1_Zn preconditions"
        else:
            rejection = "lower ranked static Q1 candidate"
        records.append(
            {
                "rank": idx,
                "candidate_atom_id": int(site["atom_id"]),
                "piece": site.get("piece"),
                "passed_preconditions": bool(site.get("q1_passed_preconditions", False)),
                "selection_score": site.get("q1_selection_score"),
                "rejection_reason": rejection,
            }
        )
    selected = selected_sites[0] if selected_sites else None
    return {
        "mode": mode,
        "candidate_pool_size": len(q1_sites),
        "topology_valid_candidate_count": sum(1 for site in q1_sites if site.get("q1_passed_preconditions", False)),
        "selected_candidate_rank": selected_rank,
        "selected_site_atom_id": None if selected is None else int(selected["atom_id"]),
        "selected_piece_name": None if selected is None else selected.get("piece"),
        "policy": [
            "Q1 piece labels are only the first filter.",
            "Candidates must pass terminal/non-bridging O hydroxylation preconditions.",
            "ranked_static mode orders candidates by static ZnO2(OH)2 geometry score, Zn-O distances, tetrahedral angle deviation, O-O separation, and deterministic seed-based tie breaker.",
            "For otherwise tied motifs, ranked_static prefers a compact but non-overlapping secondary oxygen shell using the nearest non-motif O distance.",
            "No post-minimization result is used by normal generation.",
        ],
        "selected_record": None if selected is None else q1_selection_record(selected, selected_rank or 0, mode, seed),
        "candidate_records": records,
    }


def q1_candidate_trial_report(site, entries_crystal, entries_bonds, entries_angle, supercell, allow_hydroxylate_bridging_oxygen=False, precondition_zinc_geometry=True, target_Zn_O_distance=1.95):
    pre_nearest_four, pre_all = q1_nearest_oxygen_records(
        entries_crystal,
        supercell,
        site["atom_id"],
        limit=4,
    )
    base_report = assess_zinc_site_preconditions(
        site,
        entries_crystal,
        entries_bonds,
        entries_angle,
        supercell,
        allow_hydroxylate_bridging_oxygen,
        precondition_zinc_geometry,
        target_Zn_O_distance,
    )
    base_report["selection_score"] = float(q1_precondition_score(base_report))
    base_report["selection_score_components"] = {
        "precondition_score": float(base_report["selection_score"]),
    }
    if not base_report.get("passed_preconditions", False):
        base_report["q1_geometry_diagnostics"] = None
        base_report["selected_hydroxylated_oxygen_ids"] = []
        base_report["selected_hydroxylated_oxygen_records"] = []
        base_report["pre_minimization_nearest_four_zn_o_atoms"] = pre_nearest_four
        base_report["pre_minimization_all_zn_o_atoms"] = pre_all
        base_report["nearest_four_zn_o_atoms"] = []
        return base_report

    trial_entries = copy.deepcopy(entries_crystal)
    trial_bonds = copy.deepcopy(entries_bonds)
    trial_angles = copy.deepcopy(entries_angle)
    for entry in trial_entries:
        if int(entry[0]) == int(site["atom_id"]):
            entry[1] = ZN_SPECIE
            entry[2] = ZN_CHARGE
            break
    hydroxylation_records = hydroxylate_two_oxygens(
        trial_entries,
        trial_bonds,
        trial_angles,
        [site],
        supercell,
        allow_hydroxylate_bridging_oxygen,
        precondition_zinc_geometry,
        target_Zn_O_distance,
    )
    hydroxylated_ids = []
    for record in hydroxylation_records:
        for oxy in record.get("hydroxylated_oxygens", []):
            hydroxylated_ids.append(int(oxy["oxygen_atom_id"]))
    post_nearest_four, post_all = q1_nearest_oxygen_records(
        trial_entries,
        supercell,
        site["atom_id"],
        intended_oxygen_ids=[item["atom_id"] for item in pre_nearest_four],
        hydroxylated_oxygen_ids=hydroxylated_ids,
        limit=4,
    )
    geometry = q1_motif_geometry(
        trial_entries,
        trial_angles,
        supercell,
        site["atom_id"],
        [item["atom_id"] for item in pre_nearest_four],
        hydroxylated_ids,
    )
    selection_score = q1_selection_score(geometry, base_report["selection_score"])
    base_report["selection_score"] = float(selection_score)
    base_report["selection_score_components"] = {
        "precondition_score": float(base_report["selection_score_components"]["precondition_score"]),
        "geometry_score": float(selection_score - base_report["selection_score_components"]["precondition_score"]),
    }
    base_report["q1_geometry_diagnostics"] = geometry
    base_report["pre_minimization_nearest_four_zn_o_atoms"] = pre_nearest_four
    base_report["pre_minimization_all_zn_o_atoms"] = pre_all
    base_report["post_trial_nearest_four_zn_o_atoms"] = post_nearest_four
    base_report["post_trial_all_zn_o_atoms"] = post_all
    base_report["selected_hydroxylated_oxygen_ids"] = sorted(set(hydroxylated_ids))
    base_report["selected_hydroxylated_oxygen_records"] = [
        {
            "atom_id": int(item["oxygen_atom_id"]),
            "original_oxygen_specie": int(item["original_oxygen_specie"]),
            "oxygen_class": item["oxygen_class"],
            "Zn_O_distance_before_hydroxylation": float(item["Zn_O_distance_before_hydroxylation"]),
        }
        for record in hydroxylation_records
        for item in record.get("hydroxylated_oxygens", [])
    ]
    base_report["nearest_four_zn_o_atoms"] = post_nearest_four
    base_report["all_nearest_zn_o_atoms"] = [
        {
            "atom_id": int(item["atom_id"]),
            "atom_type": int(item["atom_type"]),
            "atom_label": item["atom_label"],
            "distance": float(item["distance"]),
            "belongs_to_intended_motif": bool(item["belongs_to_intended_motif"]),
            "oxygen_role": item["oxygen_role"],
            "is_hydroxylated_oxygen": bool(item["is_hydroxylated_oxygen"]),
        }
        for item in post_all
    ]
    base_report["q1_hydroxylation_records"] = hydroxylation_records
    base_report["q1_assumption"] = (
        "Conservative static candidate: replace one Q1/terminal silicate Si center with Zn(+2), "
        "convert two safe terminal/non-bridging O(S) core-shell pairs to Oh-Hoh, and score the resulting "
        "ZnO2(OH)2-like local geometry conservatively."
    )
    return base_report


def count_species(entries):
    counts = {}
    for entry in entries:
        counts[int(entry[1])] = counts.get(int(entry[1]), 0) + 1
    return counts


def coords_by_atom_id(entries):
    return {int(entry[0]): np.array(entry[3:], dtype=float) for entry in entries}


def type_by_atom_id(entries):
    return {int(entry[0]): int(entry[1]) for entry in entries}


def bonded_atom_ids(entries_bonds, atom_id, bond_types=None):
    bonded = []
    for bond in entries_bonds:
        if bond_types is not None and int(bond[1]) not in bond_types:
            continue
        if int(bond[2]) == int(atom_id):
            bonded.append(int(bond[3]))
        elif int(bond[3]) == int(atom_id):
            bonded.append(int(bond[2]))
    return bonded


def minimum_periodic_distance(coord, selected_coords, supercell):
    if not selected_coords:
        return math.inf
    inv_supercell = np.linalg.inv(supercell)
    coord = np.array(coord, dtype=float)
    min_distance = math.inf
    for other in selected_coords:
        delta = coord - np.array(other, dtype=float)
        frac = np.dot(delta, inv_supercell)
        frac -= np.rint(frac)
        delta_pbc = np.dot(frac, supercell)
        min_distance = min(min_distance, float(np.linalg.norm(delta_pbc)))
    return min_distance


def minimum_distance_to_species(coord, entries, species, supercell):
    coords = [entry[3:] for entry in entries if int(entry[1]) in species]
    return minimum_periodic_distance(coord, coords, supercell)


def periodic_distance(coord1, coord2, supercell):
    return minimum_periodic_distance(coord1, [coord2], supercell)


def angle_degrees(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    cosang = float(np.dot(v1, v2) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def vector_pbc(origin, other, supercell):
    inv_supercell = np.linalg.inv(supercell)
    delta = np.array(other, dtype=float) - np.array(origin, dtype=float)
    frac = np.dot(delta, inv_supercell)
    frac -= np.rint(frac)
    return np.dot(frac, supercell)


def total_charge(entries_crystal):
    return float(sum(float(entry[2]) for entry in entries_crystal))


def next_atom_id(entries_crystal):
    return max(int(entry[0]) for entry in entries_crystal) + 1


def find_entry(entries, atom_id):
    for entry in entries:
        if int(entry[0]) == int(atom_id):
            return entry
    return None


def find_shell_for_core(entries_crystal, entries_bonds, core_id):
    for bond in entries_bonds:
        if int(bond[1]) == 3 and int(bond[2]) == int(core_id):
            shell = find_entry(entries_crystal, int(bond[3]))
            if shell is not None and int(shell[1]) == 4:
                return shell, bond
    return None, None


def count_nearby_si_neighbors(atom_id, entries_crystal, supercell, cutoff=2.2, exclude_si_ids=None):
    coords = coords_by_atom_id(entries_crystal)
    atom_types = type_by_atom_id(entries_crystal)
    o_coord = coords[int(atom_id)]
    count = 0
    neighbors = []
    exclude_si_ids = set(exclude_si_ids or [])
    for other_id, specie in atom_types.items():
        if other_id in exclude_si_ids:
            continue
        if specie not in (2, 10):
            continue
        distance = periodic_distance(o_coord, coords[other_id], supercell)
        if distance <= cutoff:
            count += 1
            neighbors.append({"atom_id": int(other_id), "distance": float(distance)})
    return count, neighbors


def classify_oxygen_for_hydroxylation(atom_id, entries_crystal, entries_bonds, supercell, exclude_si_ids=None):
    entry = find_entry(entries_crystal, atom_id)
    if entry is None:
        return {"oxygen_class": "missing", "safe_for_default_hydroxylation": False}

    specie = int(entry[1])
    if specie == 5:
        return {"oxygen_class": "water oxygen", "safe_for_default_hydroxylation": False}
    if specie == 6:
        return {"oxygen_class": "hydroxyl oxygen already", "safe_for_default_hydroxylation": False}
    if specie not in (3, 11):
        return {"oxygen_class": "not core/shell oxygen", "safe_for_default_hydroxylation": False}

    shell, shell_bond = find_shell_for_core(entries_crystal, entries_bonds, atom_id)
    if shell is None or shell_bond is None:
        return {"oxygen_class": "core oxygen without O(S) shell", "safe_for_default_hydroxylation": False}

    si_count, si_neighbors = count_nearby_si_neighbors(atom_id, entries_crystal, supercell, exclude_si_ids=exclude_si_ids)
    if si_count == 0:
        oxygen_class = "terminal/non-bridging O"
        safe = True
    else:
        oxygen_class = "bridging Zn/Si-O-Si oxygen"
        safe = False

    return {
        "oxygen_class": oxygen_class,
        "safe_for_default_hydroxylation": safe,
        "nearby_si_count": int(si_count),
        "nearby_si_neighbors": si_neighbors,
    }


def oxygen_candidates_for_site(site, entries_crystal, entries_bonds, supercell, allow_hydroxylate_bridging_oxygen=False):
    coords = coords_by_atom_id(entries_crystal)
    atom_types = type_by_atom_id(entries_crystal)
    zn_coord = np.array(site["coord"], dtype=float)
    candidates = []
    for atom_id, specie in atom_types.items():
        if specie not in (3, 11):
            continue
        shell, shell_bond = find_shell_for_core(entries_crystal, entries_bonds, atom_id)
        if shell is None:
            continue
        distance = periodic_distance(zn_coord, coords[atom_id], supercell)
        if distance <= ZN_O_CUTOFF:
            classification = classify_oxygen_for_hydroxylation(
                atom_id,
                entries_crystal,
                entries_bonds,
                supercell,
                exclude_si_ids={int(site["atom_id"])},
            )
            safe = bool(classification["safe_for_default_hydroxylation"])
            if classification["oxygen_class"] == "bridging Si-O-Si oxygen" and allow_hydroxylate_bridging_oxygen:
                safe = True
            candidates.append(
                {
                    "atom_id": atom_id,
                    "shell_id": int(shell[0]),
                    "shell_bond_id": int(shell_bond[0]),
                    "distance": distance,
                    "specie": int(specie),
                    "oxygen_class": classification["oxygen_class"],
                    "safe_for_default_hydroxylation": safe,
                    "nearby_si_count": classification.get("nearby_si_count"),
                    "nearby_si_neighbors": classification.get("nearby_si_neighbors", []),
                }
            )
    candidates.sort(key=lambda item: (not item["safe_for_default_hydroxylation"], item["distance"], item["atom_id"]))
    return candidates


def assess_zinc_site_preconditions(
    site,
    entries_crystal,
    entries_bonds,
    entries_angle,
    supercell,
    allow_hydroxylate_bridging_oxygen=False,
    precondition_zinc_geometry=True,
    target_Zn_O_distance=1.95,
):
    neighboring_oxygen = oxygen_candidates_for_site(
        site,
        entries_crystal,
        entries_bonds,
        supercell,
        allow_hydroxylate_bridging_oxygen,
    )
    safe_oxygen = [item for item in neighboring_oxygen if item["safe_for_default_hydroxylation"]]
    reasons = []
    passed = True
    if len(safe_oxygen) < 2:
        passed = False
        reasons.append("fewer than two safe terminal/non-bridging O candidates")

    if passed:
        try:
            trial_entries = copy.deepcopy(entries_crystal)
            trial_bonds = copy.deepcopy(entries_bonds)
            trial_angles = copy.deepcopy(entries_angle)
            for entry in trial_entries:
                if int(entry[0]) == int(site["atom_id"]):
                    entry[1] = ZN_SPECIE
                    entry[2] = ZN_CHARGE
                    break
            hydroxylate_two_oxygens(
                trial_entries,
                trial_bonds,
                trial_angles,
                [site],
                supercell,
                allow_hydroxylate_bridging_oxygen,
                precondition_zinc_geometry,
                target_Zn_O_distance,
            )
        except ValueError as exc:
            passed = False
            reasons.append(str(exc))

    return {
        "candidate_atom_id": int(site["atom_id"]),
        "original_atom_type": int(site["original_specie"]),
        "original_species": int(site["original_specie"]),
        "local_silicate_label": site["piece"],
        "motif": site["motif"],
        "cell": site["cell"],
        "coord": site["coord"],
        "neighboring_O_atoms": [
            {
                "atom_id": int(item["atom_id"]),
                "shell_id": int(item["shell_id"]),
                "distance": float(item["distance"]),
                "oxygen_class": item["oxygen_class"],
                "safe_for_default_hydroxylation": bool(item["safe_for_default_hydroxylation"]),
                "nearby_si_count": item.get("nearby_si_count"),
                "nearby_si_neighbors": item.get("nearby_si_neighbors", []),
            }
            for item in neighboring_oxygen
        ],
        "n_safe_terminal_oxygen": int(len(safe_oxygen)),
        "passed_Q1_Zn_preconditions": bool(passed) if site["motif"] == "Q1_Zn" else None,
        "passed_preconditions": bool(passed),
        "rejection_reason": None if passed else "; ".join(reasons),
    }


def build_zinc_candidate_site_report(
    candidates,
    entries_crystal,
    entries_bonds,
    entries_angle,
    supercell,
    allow_hydroxylate_bridging_oxygen=False,
    precondition_zinc_geometry=True,
    target_Zn_O_distance=1.95,
):
    report = {}
    for motif, sites in candidates.items():
        if motif == "Q1_Zn":
            report[motif] = [
                q1_candidate_trial_report(
                    site,
                    entries_crystal,
                    entries_bonds,
                    entries_angle,
                    supercell,
                    allow_hydroxylate_bridging_oxygen,
                    precondition_zinc_geometry,
                    target_Zn_O_distance,
                )
                for site in sites
            ]
        else:
            report[motif] = [
                assess_zinc_site_preconditions(
                    site,
                    entries_crystal,
                    entries_bonds,
                    entries_angle,
                    supercell,
                    allow_hydroxylate_bridging_oxygen,
                    precondition_zinc_geometry,
                    target_Zn_O_distance,
                )
                for site in sites
            ]
    return report


def attach_q1_scores_to_candidates(candidates, candidate_site_report):
    if not candidate_site_report or "Q1_Zn" not in candidate_site_report:
        return candidates
    by_atom = {
        int(item["candidate_atom_id"]): item
        for item in candidate_site_report.get("Q1_Zn", [])
    }
    for site in candidates.get("Q1_Zn", []):
        record = by_atom.get(int(site["atom_id"]))
        if record is None:
            continue
        site["q1_selection_score"] = float(record.get("selection_score", -1.0e9))
        site["q1_selection_report"] = record
        site["q1_passed_preconditions"] = bool(record.get("passed_preconditions", False))
    return candidates


def select_oxygens_for_hydroxylation(
    site,
    entries_crystal,
    entries_bonds,
    supercell,
    n_oxygen=2,
    allow_hydroxylate_bridging_oxygen=False,
):
    candidates = oxygen_candidates_for_site(
        site, entries_crystal, entries_bonds, supercell, allow_hydroxylate_bridging_oxygen
    )
    safe_candidates = [candidate for candidate in candidates if candidate["safe_for_default_hydroxylation"]]
    if len(safe_candidates) < n_oxygen:
        raise ValueError(
            "Selected Zn site atom_id={} has only {} safe terminal/non-bridging O core/shell "
            "candidates within {:.2f} A; need {}. Candidate classes: {}".format(
                site["atom_id"],
                len(safe_candidates),
                ZN_O_CUTOFF,
                n_oxygen,
                [
                    {
                        "atom_id": item["atom_id"],
                        "class": item["oxygen_class"],
                        "distance": item["distance"],
                    }
                    for item in candidates
                ],
            )
        )
    return safe_candidates[:n_oxygen]


def h_direction_away_from_zn(o_coord, zn_coord):
    direction = np.array(o_coord, dtype=float) - np.array(zn_coord, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 1.0])
    return direction / norm


def min_h_overlap(h_coord, entries_crystal, ignore_ids, supercell):
    distances = []
    for entry in entries_crystal:
        atom_id = int(entry[0])
        if atom_id in ignore_ids:
            continue
        distances.append(periodic_distance(h_coord, entry[3:], supercell))
    return None if not distances else float(min(distances))


def distance_to_type_set(h_coord, entries_crystal, type_set, supercell, ignore_ids=None):
    ignore_ids = set(ignore_ids or [])
    best = {"distance": None, "atom_id": None, "specie": None}
    for entry in entries_crystal:
        atom_id = int(entry[0])
        if atom_id in ignore_ids or int(entry[1]) not in type_set:
            continue
        distance = periodic_distance(h_coord, entry[3:], supercell)
        if best["distance"] is None or distance < best["distance"]:
            best = {"distance": float(distance), "atom_id": atom_id, "specie": int(entry[1])}
    return best


def h_contact_metrics(h_coord, entries_crystal, supercell, bonded_o_id, h_id, zn_id):
    ignore_o = {int(bonded_o_id), int(h_id)}
    return {
        "H_H": distance_to_type_set(h_coord, entries_crystal, {7, 8}, supercell, {int(h_id)}),
        "H_O_nonbonded": distance_to_type_set(h_coord, entries_crystal, {3, 4, 5, 6, 11, 12}, supercell, ignore_o),
        "H_Si": distance_to_type_set(h_coord, entries_crystal, {2, 10}, supercell, {int(h_id)}),
        "H_Ca": distance_to_type_set(h_coord, entries_crystal, {1, 9}, supercell, {int(h_id)}),
        "H_Zn": distance_to_type_set(h_coord, entries_crystal, {ZN_SPECIE}, supercell, {int(h_id)}),
    }


def h_contacts_are_safe(metrics):
    for key, cutoff in H_MIN_DISTANCES.items():
        distance = metrics.get(key, {}).get("distance")
        if distance is not None and distance < cutoff:
            return False
    return True


def oxygen_contact_metrics(o_coord, entries_crystal, supercell, o_id, h_id, zn_id):
    return {
        "O_Ca": distance_to_type_set(o_coord, entries_crystal, {1, 9}, supercell, {int(o_id), int(h_id)}),
        "O_Si": distance_to_type_set(o_coord, entries_crystal, {2, 10}, supercell, {int(o_id), int(h_id), int(zn_id)}),
        "O_O_nonbonded": distance_to_type_set(
            o_coord,
            entries_crystal,
            {3, 4, 5, 6, 11, 12},
            supercell,
            {int(o_id), int(h_id)},
        ),
        "O_Zn": distance_to_type_set(o_coord, entries_crystal, {ZN_SPECIE}, supercell, {int(o_id), int(h_id)}),
    }


def oxygen_contacts_are_safe(metrics):
    for key, cutoff in CONVERTED_O_MIN_DISTANCES.items():
        distance = metrics.get(key, {}).get("distance")
        if distance is not None and distance < cutoff:
            return False
    return True


def orthonormal_basis(axis):
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(axis, trial))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    u = trial - np.dot(trial, axis) * axis
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def trial_h_directions(base_direction):
    base = np.array(base_direction, dtype=float)
    base = base / np.linalg.norm(base)
    directions = [base]
    u, v = orthonormal_basis(base)
    for polar_deg in (30.0, 50.0, 70.0, 90.0, 110.0):
        polar = math.radians(polar_deg)
        for azimuth_deg in range(0, 360, 30):
            azimuth = math.radians(azimuth_deg)
            direction = (
                math.cos(polar) * base
                + math.sin(polar) * (math.cos(azimuth) * u + math.sin(azimuth) * v)
            )
            directions.append(direction / np.linalg.norm(direction))
    return directions


def choose_h_position(o_coord, zn_coord, entries_crystal, supercell, bonded_o_id, h_id, zn_id):
    base = h_direction_away_from_zn(o_coord, zn_coord)
    best = None
    for direction in trial_h_directions(base):
        h_coord = np.array(o_coord, dtype=float) + OH_BOND_DISTANCE * direction
        metrics = h_contact_metrics(h_coord, entries_crystal, supercell, bonded_o_id, h_id, zn_id)
        score = min(
            metrics[key]["distance"] if metrics[key]["distance"] is not None else 99.0
            for key in H_MIN_DISTANCES
        )
        record = {
            "coord": h_coord,
            "direction": direction,
            "metrics": metrics,
            "score": float(score),
            "safe": h_contacts_are_safe(metrics),
        }
        if best is None or record["score"] > best["score"]:
            best = record
        if record["safe"]:
            return record
    return best


def set_entry_coord(entry, coord):
    entry[3] = float(coord[0])
    entry[4] = float(coord[1])
    entry[5] = float(coord[2])


def precondition_converted_oxygen(o_entry, site, supercell, target_Zn_O_distance):
    zn_coord = np.array(site["coord"], dtype=float)
    o_coord = np.array(o_entry[3:], dtype=float)
    vector = vector_pbc(zn_coord, o_coord, supercell)
    distance = float(np.linalg.norm(vector))
    if distance < 1.0e-8:
        return {"moved": False, "old_distance": distance, "new_distance": distance}
    if distance >= target_Zn_O_distance:
        return {"moved": False, "old_distance": distance, "new_distance": distance}
    new_coord = zn_coord + vector / distance * float(target_Zn_O_distance)
    set_entry_coord(o_entry, new_coord)
    return {
        "moved": True,
        "old_distance": distance,
        "new_distance": float(target_Zn_O_distance),
        "old_coord": [float(x) for x in o_coord],
        "new_coord": [float(x) for x in new_coord],
    }


def hydroxylate_two_oxygens(
    entries_crystal,
    entries_bonds,
    entries_angle,
    selected_sites,
    supercell,
    allow_hydroxylate_bridging_oxygen=False,
    precondition_zinc_geometry=True,
    target_Zn_O_distance=1.95,
):
    records = []
    for site in selected_sites:
        selected_oxygens = select_oxygens_for_hydroxylation(
            site, entries_crystal, entries_bonds, supercell, 2, allow_hydroxylate_bridging_oxygen
        )
        site_records = []
        for oxygen in selected_oxygens:
            o_entry = find_entry(entries_crystal, oxygen["atom_id"])
            h_entry = find_entry(entries_crystal, oxygen["shell_id"])
            shell_bond = None
            for bond in entries_bonds:
                if int(bond[0]) == oxygen["shell_bond_id"]:
                    shell_bond = bond
                    break
            if o_entry is None or h_entry is None or shell_bond is None:
                raise ValueError("Internal error while hydroxylating O atom {}".format(oxygen["atom_id"]))

            precondition_record = {"moved": False}
            if precondition_zinc_geometry:
                precondition_record = precondition_converted_oxygen(
                    o_entry, site, supercell, target_Zn_O_distance
                )
            o_coord = np.array(o_entry[3:], dtype=float)
            o_contact_metrics = oxygen_contact_metrics(
                o_coord,
                entries_crystal,
                supercell,
                int(o_entry[0]),
                int(h_entry[0]),
                int(site["atom_id"]),
            )
            if not oxygen_contacts_are_safe(o_contact_metrics):
                if precondition_record.get("moved") and "old_coord" in precondition_record:
                    set_entry_coord(o_entry, precondition_record["old_coord"])
                raise ValueError(
                    "Preconditioning converted O atom {} for Zn atom {} creates unsafe contacts: {}".format(
                        o_entry[0], site["atom_id"], o_contact_metrics
                    )
                )
            placement = choose_h_position(
                o_coord,
                site["coord"],
                entries_crystal,
                supercell,
                int(o_entry[0]),
                int(h_entry[0]),
                int(site["atom_id"]),
            )
            if placement is None or not placement["safe"]:
                raise ValueError(
                    "Could not place Hoh safely for Zn atom {} and O atom {}. Best contact metrics: {}".format(
                        site["atom_id"], o_entry[0], None if placement is None else placement["metrics"]
                    )
                )
            h_coord = placement["coord"]
            direction = placement["direction"]

            o_entry[1] = 6
            o_entry[2] = OH_CHARGE
            h_entry[1] = 8
            h_entry[2] = HOH_CHARGE
            h_entry[3] = float(h_coord[0])
            h_entry[4] = float(h_coord[1])
            h_entry[5] = float(h_coord[2])

            shell_bond[1] = 1
            shell_bond[2] = int(o_entry[0])
            shell_bond[3] = int(h_entry[0])

            angle_id = max([int(angle[0]) for angle in entries_angle] or [0]) + 1
            entries_angle.append([angle_id, 5, int(site["atom_id"]), int(o_entry[0]), int(h_entry[0])])

            overlap = min_h_overlap(h_coord, entries_crystal, {int(o_entry[0]), int(h_entry[0]), int(site["atom_id"])}, supercell)
            site_records.append(
                {
                    "oxygen_atom_id": int(o_entry[0]),
                    "reused_shell_as_H_atom_id": int(h_entry[0]),
                    "modified_bond_id": int(shell_bond[0]),
                    "added_angle_id": int(angle_id),
                    "original_oxygen_specie": int(oxygen["specie"]),
                    "original_shell_atom_id": int(oxygen["shell_id"]),
                    "oxygen_class": oxygen["oxygen_class"],
                    "nearby_si_count_before_hydroxylation": oxygen.get("nearby_si_count"),
                    "nearby_si_neighbors_before_hydroxylation": oxygen.get("nearby_si_neighbors", []),
                    "Zn_O_distance_before_hydroxylation": float(oxygen["distance"]),
                    "Zn_O_preconditioning": precondition_record,
                    "converted_O_contact_metrics": o_contact_metrics,
                    "H_coord": [float(h_coord[0]), float(h_coord[1]), float(h_coord[2])],
                    "H_placement_vector": [float(direction[0]), float(direction[1]), float(direction[2])],
                    "O_H_distance": float(periodic_distance(o_coord, h_coord, supercell)),
                    "H_contact_metrics": placement["metrics"],
                    "min_H_overlap_distance": overlap,
                    "H_placement": "deterministic_overlap_checked_search",
                }
            )
        records.append({"zn_atom_id": int(site["atom_id"]), "hydroxylated_oxygens": site_records})
    return records


def apply_charge_balance(
    entries_crystal,
    entries_bonds,
    entries_angle,
    selected_sites,
    supercell,
    mode,
    allow_hydroxylate_bridging_oxygen=False,
    precondition_zinc_geometry=True,
    target_Zn_O_distance=1.95,
):
    validate_charge_balance_mode(mode)
    if mode == "hydroxylate_two_oxygens":
        return hydroxylate_two_oxygens(
            entries_crystal,
            entries_bonds,
            entries_angle,
            selected_sites,
            supercell,
            allow_hydroxylate_bridging_oxygen,
            precondition_zinc_geometry,
            target_Zn_O_distance,
        )
    if mode in ("fail_if_not_neutral", "allow_unbalanced_for_debug"):
        return []
    raise NotImplementedError(mode + " is not implemented in v2")


def select_zinc_sites(candidates, n_zinc, site_type, seed, supercell, min_zn_zn_distance=3.0, site_filter=None, q1_selection_mode="ranked_static"):
    validate_zinc_site_type(site_type)
    if n_zinc <= 0:
        return []

    if site_type == "Q1_Zn":
        pool = list(candidates["Q1_Zn"])
    elif site_type == "Q2b_Zn":
        pool = list(candidates["Q2b_Zn"])
    else:
        pool = list(candidates["Q1_Zn"]) + list(candidates["Q2b_Zn"])

    if not pool:
        raise ValueError("No candidate sites are available for " + site_type)
    if n_zinc > len(pool):
        raise ValueError(
            "Requested {} Zn sites, but only {} eligible {} sites were found".format(
                n_zinc, len(pool), site_type
            )
        )

    rng = np.random.default_rng(seed)
    if site_type == "Q1_Zn" and q1_selection_mode not in SUPPORTED_Q1_SELECTION_MODES:
        raise ValueError(
            "Unknown Q1 selection mode {!r}. Expected one of {}".format(
                q1_selection_mode, sorted(SUPPORTED_Q1_SELECTION_MODES)
            )
        )
    if site_type == "Q1_Zn" and q1_selection_mode == "ranked_static" and any("q1_selection_score" in site for site in pool):
        pool = sorted(pool, key=lambda item: q1_static_rank_tuple(item, seed))
        order = range(len(pool))
    elif site_type == "Q1_Zn" and q1_selection_mode in ("first_valid", "screening_debug"):
        tie_breakers = {int(site["atom_id"]): float(rng.random()) for site in pool}
        pool = sorted(
            pool,
            key=lambda item: (
                not bool(item.get("q1_passed_preconditions", False)),
                int(item["atom_id"]),
                tie_breakers[int(item["atom_id"])],
            ),
        )
        order = range(len(pool))
    else:
        order = rng.permutation(len(pool))
    selected = []
    selected_coords = []
    skipped_for_distance = 0

    for idx in order:
        site = dict(pool[int(idx)])
        if site_filter is not None and not site_filter(site):
            continue
        distance = minimum_periodic_distance(site["coord"], selected_coords, supercell)
        if distance < min_zn_zn_distance:
            skipped_for_distance += 1
            continue
        selected.append(site)
        selected_coords.append(site["coord"])
        if len(selected) == n_zinc:
            break

    if len(selected) != n_zinc:
        raise ValueError(
            "Could only select {} of {} Zn sites without Zn-Zn distances below {:.2f} A "
            "({} candidates skipped). Lower Zn_Si_ratio or increase the supercell.".format(
                len(selected), n_zinc, min_zn_zn_distance, skipped_for_distance
            )
        )

    return selected


def site_hydroxylated_oxygen_ids(site, entries_crystal, entries_bonds, supercell, allow_hydroxylate_bridging_oxygen=False):
    return {
        int(item["atom_id"])
        for item in select_oxygens_for_hydroxylation(
            site,
            entries_crystal,
            entries_bonds,
            supercell,
            2,
            allow_hydroxylate_bridging_oxygen,
        )
    }


def rank_q2b_site(site, entries_crystal, entries_bonds, supercell):
    oxy = oxygen_candidates_for_site(site, entries_crystal, entries_bonds, supercell, False)
    safe = [item for item in oxy if item["safe_for_default_hydroxylation"]]
    min_dist = min([float(item["distance"]) for item in safe], default=999.0)
    return (len(safe) < 2, min_dist, int(site["atom_id"]))


def multi_mode_counts(mode, n_q1=None, n_q2b=None):
    if mode == "multi_q2b":
        return 0 if n_q1 is None else int(n_q1), 2 if n_q2b is None else int(n_q2b)
    if mode == "multi_q1":
        return 2 if n_q1 is None else int(n_q1), 0 if n_q2b is None else int(n_q2b)
    if mode == "q1_q2b_single_structure_mixture":
        return 1 if n_q1 is None else int(n_q1), 1 if n_q2b is None else int(n_q2b)
    raise ValueError("Unsupported v1.6-alpha multi-Zn mode: {}".format(mode))


def pairwise_zn_zn_distances(selected_sites, supercell):
    out = []
    for i, site_i in enumerate(selected_sites):
        for site_j in selected_sites[i + 1:]:
            out.append({
                "zn_atom_id_1": int(site_i["atom_id"]),
                "zn_atom_id_2": int(site_j["atom_id"]),
                "motif_1": site_i.get("motif"),
                "motif_2": site_j.get("motif"),
                "distance": float(periodic_distance(site_i["coord"], site_j["coord"], supercell)),
            })
    return out


def build_multi_candidate_pools(candidates, candidate_site_report, seed, entries_crystal, entries_bonds, supercell):
    q1_candidates = attach_q1_scores_to_candidates(copy.deepcopy(candidates), candidate_site_report).get("Q1_Zn", [])
    q1_pool = [
        dict(site)
        for site in q1_candidates
        if site.get("q1_passed_preconditions", False)
    ]
    q1_pool.sort(key=lambda site: q1_static_rank_tuple(site, seed))
    q2b_reports = {
        int(item["candidate_atom_id"]): item
        for item in candidate_site_report.get("Q2b_Zn", [])
    }
    q2b_pool = []
    for site in candidates.get("Q2b_Zn", []):
        report = q2b_reports.get(int(site["atom_id"]), {})
        if not report.get("passed_preconditions", False):
            continue
        item = dict(site)
        item["q2b_selection_report"] = report
        q2b_pool.append(item)
    q2b_pool.sort(key=lambda site: rank_q2b_site(site, entries_crystal, entries_bonds, supercell))
    return {"Q1_Zn": q1_pool, "Q2b_Zn": q2b_pool}


def select_multi_zinc_sites(
    candidates,
    candidate_site_report,
    entries_crystal,
    entries_bonds,
    supercell,
    mode,
    seed,
    n_q1=None,
    n_q2b=None,
    min_zn_zn_distance=5.0,
    max_attempts=100,
):
    target_q1, target_q2b = multi_mode_counts(mode, n_q1, n_q2b)
    pools = build_multi_candidate_pools(candidates, candidate_site_report, seed, entries_crystal, entries_bonds, supercell)
    selected = []
    used_si = set()
    used_hydroxylated_o = set()
    rejected = []

    def try_take(site, motif):
        atom_id = int(site["atom_id"])
        if atom_id in used_si:
            return False, "duplicate substituted Si site"
        if minimum_periodic_distance(site["coord"], [item["coord"] for item in selected], supercell) < float(min_zn_zn_distance):
            return False, "Zn-Zn distance below {:.2f} A".format(float(min_zn_zn_distance))
        try:
            hydroxylated = site_hydroxylated_oxygen_ids(site, entries_crystal, entries_bonds, supercell, False)
        except ValueError as exc:
            return False, str(exc)
        overlap = sorted(used_hydroxylated_o.intersection(hydroxylated))
        if overlap:
            return False, "would reuse hydroxylated O core-shell pair(s): {}".format(overlap)
        accepted = dict(site)
        accepted["motif"] = motif
        accepted["planned_hydroxylated_oxygen_ids"] = sorted(hydroxylated)
        selected.append(accepted)
        used_si.add(atom_id)
        used_hydroxylated_o.update(hydroxylated)
        return True, None

    for motif, target in (("Q1_Zn", target_q1), ("Q2b_Zn", target_q2b)):
        attempts = 0
        for site in pools[motif]:
            if sum(1 for item in selected if item["motif"] == motif) >= target:
                break
            attempts += 1
            if attempts > int(max_attempts):
                break
            ok, reason = try_take(site, motif)
            if not ok:
                rejected.append({
                    "candidate_atom_id": int(site["atom_id"]),
                    "motif": motif,
                    "piece": site.get("piece"),
                    "rejection_reason": reason,
                })
        have = sum(1 for item in selected if item["motif"] == motif)
        if have < target:
            raise ValueError(
                "Could only select {} of {} requested {} sites. Rejections: {}".format(
                    have, target, motif, rejected[-10:]
                )
            )
    return selected, rejected, pools


def apply_zinc_sites(entries_crystal, crystal_dict, selected_sites):
    selected_ids = {site["atom_id"] for site in selected_sites}
    touched = set()

    for entry in entries_crystal:
        if int(entry[0]) in selected_ids:
            if int(entry[1]) in (1, 9):
                raise ValueError("Refusing to place Zn on Ca atom id {}".format(entry[0]))
            entry[1] = ZN_SPECIE
            entry[2] = ZN_CHARGE
            touched.add(int(entry[0]))

    for brick_dict in crystal_dict.values():
        for piece_entries in brick_dict.values():
            for entry in piece_entries:
                if int(entry[0]) in selected_ids:
                    entry[1] = ZN_SPECIE
                    entry[2] = ZN_CHARGE

    missing = selected_ids.difference(touched)
    if missing:
        raise ValueError("Selected Zn atom ids were not found in entries_crystal: {}".format(sorted(missing)))

    return entries_crystal, crystal_dict


def remap_zinc_angles(entries_crystal, entries_angle):
    """Convert Zn-centered old Si angle types to CementFF4 Zn angle types."""
    atom_types = type_by_atom_id(entries_crystal)
    selected_zn = {atom_id for atom_id, specie in atom_types.items() if specie == ZN_SPECIE}
    remapped_ozno = 0
    remapped_znohh = 0
    stale_zn_angles = []

    for angle in entries_angle:
        atom1, atom2, atom3 = int(angle[2]), int(angle[3]), int(angle[4])
        if atom2 in selected_zn and int(angle[1]) == 2:
            angle[1] = 4
            remapped_ozno += 1
        elif atom1 in selected_zn and int(angle[1]) == 3:
            angle[1] = 5
            remapped_znohh += 1
        elif atom2 in selected_zn and int(angle[1]) == 4:
            continue
        elif atom1 in selected_zn and int(angle[1]) == 5:
            continue
        elif atom1 in selected_zn or atom2 in selected_zn or atom3 in selected_zn:
            stale_zn_angles.append(list(angle))

    return {
        "remapped_O_Zn_O_angles": remapped_ozno,
        "remapped_Zn_Oh_H_angles": remapped_znohh,
        "stale_zn_angles": stale_zn_angles,
    }


def validate_no_zinc_bonds(entries_crystal, entries_bonds):
    atom_types = type_by_atom_id(entries_crystal)
    zn_ids = {atom_id for atom_id, specie in atom_types.items() if specie == ZN_SPECIE}
    zinc_bonds = []
    for bond in entries_bonds:
        if int(bond[2]) in zn_ids or int(bond[3]) in zn_ids:
            zinc_bonds.append(list(bond))
    if zinc_bonds:
        raise ValueError(
            "CementFF4 v1 does not define Zn-O bonds; refusing Zn-bonded topology: {}".format(zinc_bonds)
        )
    return zinc_bonds


def geometry_metrics(entries_crystal, entries_angle, supercell):
    coords = coords_by_atom_id(entries_crystal)
    atom_types = type_by_atom_id(entries_crystal)
    zn_ids = [atom_id for atom_id, specie in atom_types.items() if specie == ZN_SPECIE]
    o_species = {3, 4, 5, 6, 11, 12}
    silicate_o_species = {3, 6, 11, 12}
    ca_species = {1, 9}
    si_species = {2, 10}

    zn_o_distances = []
    zn_coordination = []
    for zn_id in zn_ids:
        distances = [
            periodic_distance(coords[zn_id], coord, supercell)
            for atom_id, coord in coords.items()
            if atom_types[atom_id] in silicate_o_species
        ]
        neighbors = [d for d in distances if d <= 2.3]
        zn_coordination.append(len(neighbors))
        zn_o_distances.extend(neighbors)

    zn_zn_distances = []
    for i, zn_i in enumerate(zn_ids):
        for zn_j in zn_ids[i + 1:]:
            zn_zn_distances.append(periodic_distance(coords[zn_i], coords[zn_j], supercell))

    si_o_distances = []
    for si_id, si_coord in coords.items():
        if atom_types[si_id] not in si_species:
            continue
        for o_id, o_coord in coords.items():
            if atom_types[o_id] in silicate_o_species:
                distance = periodic_distance(si_coord, o_coord, supercell)
                if distance <= 2.2:
                    si_o_distances.append(distance)

    ca_o_distances = []
    for ca_id, ca_coord in coords.items():
        if atom_types[ca_id] not in ca_species:
            continue
        for o_id, o_coord in coords.items():
            if atom_types[o_id] in o_species:
                distance = periodic_distance(ca_coord, o_coord, supercell)
                if distance <= 3.2:
                    ca_o_distances.append(distance)

    o_zn_o_angles = []
    for angle in entries_angle:
        if int(angle[1]) != 4:
            continue
        a1, center, a3 = int(angle[2]), int(angle[3]), int(angle[4])
        if center not in zn_ids:
            continue
        v1 = vector_pbc(coords[center], coords[a1], supercell)
        v2 = vector_pbc(coords[center], coords[a3], supercell)
        angle_value = angle_degrees(v1, v2)
        if angle_value is not None:
            o_zn_o_angles.append(angle_value)

    def stats(values):
        if not values:
            return {"min": None, "mean": None, "max": None}
        return {
            "min": float(min(values)),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
        }

    return {
        "Zn_O_cutoff_A": ZN_O_CUTOFF,
        "Zn_O_coordination_numbers_cutoff_2p3A": [int(x) for x in zn_coordination],
        "Zn_O_distance_A": stats(zn_o_distances),
        "O_Zn_O_angle_deg": stats(o_zn_o_angles),
        "minimum_Zn_Zn_distance": None if not zn_zn_distances else float(min(zn_zn_distances)),
        "minimum_Si_O_distance": None if not si_o_distances else float(min(si_o_distances)),
        "minimum_Ca_O_distance": None if not ca_o_distances else float(min(ca_o_distances)),
    }


def h_overlap_metrics(hydroxylation_records):
    overlaps = []
    for record in hydroxylation_records or []:
        for oxy in record["hydroxylated_oxygens"]:
            if oxy["min_H_overlap_distance"] is not None:
                overlaps.append(float(oxy["min_H_overlap_distance"]))
    if not overlaps:
        return {"minimum_added_H_overlap_distance": None, "has_severe_H_overlap": False}
    return {
        "minimum_added_H_overlap_distance": float(min(overlaps)),
        "has_severe_H_overlap": bool(min(overlaps) < H_OVERLAP_CUTOFF),
    }


def hydroxylation_topology_audit(entries_crystal, entries_bonds, entries_angle, hydroxylation_records):
    atom_types = type_by_atom_id(entries_crystal)
    audit = []
    bad_records = []
    for record in hydroxylation_records or []:
        for oxy in record["hydroxylated_oxygens"]:
            o_id = int(oxy["oxygen_atom_id"])
            h_id = int(oxy["reused_shell_as_H_atom_id"])
            bonds = [list(bond) for bond in entries_bonds if int(bond[2]) in (o_id, h_id) or int(bond[3]) in (o_id, h_id)]
            shell_bonds = [
                bond for bond in bonds
                if int(bond[1]) == 3
                and (
                    {atom_types.get(int(bond[2])), atom_types.get(int(bond[3]))} == {3, 4}
                    or {atom_types.get(int(bond[2])), atom_types.get(int(bond[3]))} == {11, 4}
                )
            ]
            oh_bonds = [
                bond for bond in bonds
                if {int(bond[2]), int(bond[3])} == {o_id, h_id}
                and {atom_types.get(int(bond[2])), atom_types.get(int(bond[3]))} == {6, 8}
            ]
            angles = [list(angle) for angle in entries_angle if o_id in [int(angle[2]), int(angle[3]), int(angle[4])] or h_id in [int(angle[2]), int(angle[3]), int(angle[4])]]
            item = {
                "oxygen_atom_id": o_id,
                "hoh_atom_id": h_id,
                "oxygen_type": atom_types.get(o_id),
                "hoh_type": atom_types.get(h_id),
                "has_correct_oh_hoh_bond_internal": bool(oh_bonds),
                "expected_cementff_oh_hoh_bond_type": 3,
                "remaining_core_shell_bonds": shell_bonds,
                "bonds_involving_pair": bonds,
                "angles_involving_pair": angles,
            }
            if item["oxygen_type"] != 6 or item["hoh_type"] != 8 or not item["has_correct_oh_hoh_bond_internal"] or shell_bonds:
                bad_records.append(item)
            audit.append(item)
    return {"records": audit, "bad_records": bad_records}


def classify_zinc_output(summary, allow_unbalanced_for_debug=False):
    site_type = summary.get("Zn_site_type")
    valid_label = "valid_q1_zn_candidate" if site_type == "Q1_Zn" else "valid_q2b_zn_candidate"
    classification = valid_label
    reasons = []

    charge_residual = float(summary.get("charge_residual_final", summary.get("total_charge_residual", 0.0)))
    if abs(charge_residual) > CHARGE_TOLERANCE:
        classification = "failed_charge"
        reasons.append("non-neutral charge residual")

    geometry = summary.get("pre_minimization_geometry", summary.get("geometry_validation", {}))
    zn_o = geometry.get("Zn_O_distance_A", {})
    if zn_o.get("min") is None:
        classification = "failed_zinc_geometry"
        reasons.append("Zn has no O neighbors within 2.3 A")
    else:
        coordination = geometry.get("Zn_O_coordination_numbers_cutoff_2p3A", [])
        if any(int(count) < 4 for count in coordination):
            classification = "failed_zinc_geometry"
            reasons.append("Zn has fewer than four O neighbors within 2.3 A")

    h_overlap = geometry.get("added_H_overlap", {})
    if h_overlap.get("has_severe_H_overlap"):
        classification = "failed_zinc_geometry"
        reasons.append("newly added H has severe overlap")
    if h_overlap.get("has_v21_H_contact_violation"):
        classification = "failed_zinc_geometry"
        reasons.append("newly added H violates v2.1 minimum contact cutoffs")

    topology = summary.get("topology_validation", {})
    if topology.get("stale_zn_angles"):
        classification = "failed_topology"
        reasons.append("unmapped Zn angle entries remain")
    if topology.get("zinc_bonds"):
        classification = "failed_topology"
        reasons.append("Zn bonded topology has no defined CementFF4 bond type")
    if topology.get("hydroxylation_topology_audit", {}).get("bad_records"):
        classification = "failed_topology"
        reasons.append("converted O(S)->Oh-Hoh pair still has shell/topology remnants")

    if classification == valid_label:
        reasons.append("charge/topology mapped for static CementFF4-Zn candidate; relaxation still required before property calculations")

    summary["output_classification"] = classification
    summary["classification_reasons"] = reasons
    return summary


def finalize_zinc_summary(
    entries_crystal,
    entries_bonds,
    entries_angle,
    supercell,
    summary,
    charge_balance_mode="fail_if_not_neutral",
    allow_unbalanced_for_debug=False,
):
    if summary is None:
        return None

    summary["total_charge_after_hydroxylation"] = total_charge(entries_crystal)
    summary["charge_residual_final"] = total_charge(entries_crystal)
    topology = remap_zinc_angles(entries_crystal, entries_angle)
    topology["zinc_bonds"] = validate_no_zinc_bonds(entries_crystal, entries_bonds)
    topology["hydroxylation_topology_audit"] = hydroxylation_topology_audit(
        entries_crystal, entries_bonds, entries_angle, summary.get("hydroxylation_records", [])
    )
    summary["topology_validation"] = topology
    summary["pre_minimization_geometry"] = geometry_metrics(entries_crystal, entries_angle, supercell)
    if summary.get("Zn_site_type") == "Q1_Zn":
        q1_diagnostics = []
        for site in summary.get("selected_sites", []):
            zn_id = int(site["atom_id"])
            q1_report = site.get("q1_selection_report", {})
            intended_ids = [
                int(item["atom_id"])
                for item in q1_report.get("pre_minimization_nearest_four_zn_o_atoms", [])
            ]
            if not intended_ids:
                intended_ids = [
                    int(item["atom_id"])
                    for item in q1_report.get("nearest_four_zn_o_atoms", [])
                ]
            hydroxylated_ids = [
                int(oxy["oxygen_atom_id"])
                for record in summary.get("hydroxylation_records", [])
                if int(record.get("zn_atom_id", -1)) == zn_id
                for oxy in record.get("hydroxylated_oxygens", [])
            ]
            nearest_four, all_neighbors = q1_nearest_oxygen_records(
                entries_crystal,
                supercell,
                zn_id,
                intended_oxygen_ids=intended_ids,
                hydroxylated_oxygen_ids=hydroxylated_ids,
                limit=4,
            )
            diagnostic = q1_motif_geometry(
                entries_crystal,
                entries_angle,
                supercell,
                zn_id,
                intended_ids,
                hydroxylated_ids,
            )
            diagnostic["nearest_four_zn_o_atoms"] = nearest_four
            diagnostic["all_nearest_zn_o_atoms"] = all_neighbors
            diagnostic["selection_score"] = site.get("q1_selection_score")
            diagnostic["selection_source"] = "topology-valid Q1 candidate scored before Zn placement"
            q1_diagnostics.append(diagnostic)
        summary["pre_minimization_geometry"]["q1_motif_diagnostics"] = q1_diagnostics
    summary["pre_minimization_geometry"]["added_H_overlap"] = h_overlap_metrics(
        summary.get("hydroxylation_records", [])
    )
    h_violations = []
    for record in summary.get("hydroxylation_records", []):
        for oxy in record["hydroxylated_oxygens"]:
            for key, cutoff in H_MIN_DISTANCES.items():
                distance = oxy.get("H_contact_metrics", {}).get(key, {}).get("distance")
                if distance is not None and distance < cutoff:
                    h_violations.append(
                        {
                            "oxygen_atom_id": oxy["oxygen_atom_id"],
                            "hoh_atom_id": oxy["reused_shell_as_H_atom_id"],
                            "metric": key,
                            "distance": float(distance),
                            "cutoff": float(cutoff),
                        }
                    )
    summary["pre_minimization_geometry"]["added_H_overlap"]["v21_H_contact_violations"] = h_violations
    summary["pre_minimization_geometry"]["added_H_overlap"]["has_v21_H_contact_violation"] = bool(h_violations)
    summary["Zn_charge_balance_mode"] = charge_balance_mode
    summary["allow_unbalanced_for_debug"] = bool(allow_unbalanced_for_debug)
    summary["cementff4_type_mapping"] = CEMENTFF4_TYPE_MAP
    summary["cementff4_angle_mapping"] = CEMENTFF4_ANGLE_MAP
    summary["status_note"] = (
        "v2 uses a charge-balanced ZnO2(OH)2-style substitutional static candidate motif; minimization is required before property calculations."
    )
    summary = classify_zinc_output(summary, allow_unbalanced_for_debug)
    return summary


def build_zinc_summary(
    entries_crystal,
    selected_sites,
    candidates,
    target_zinc_si_ratio,
    ca_si_ratio,
    supercell,
    site_type,
    seed,
):
    counts = count_species(entries_crystal)
    n_si = counts.get(2, 0) + counts.get(10, 0)
    n_zn = counts.get(ZN_SPECIE, 0)
    n_ca = counts.get(1, 0) + counts.get(9, 0)
    n_si_original = n_si + n_zn
    min_zn_zn = minimum_periodic_distance(
        selected_sites[0]["coord"] if selected_sites else [0.0, 0.0, 0.0],
        [site["coord"] for site in selected_sites[1:]] if len(selected_sites) > 1 else [],
        supercell,
    )
    min_zn_o = math.inf
    for site in selected_sites:
        min_zn_o = min(
            min_zn_o,
            minimum_distance_to_species(site["coord"], entries_crystal, {3, 4, 5, 6, 11, 12}, supercell),
        )

    total_charge = float(sum(float(entry[2]) for entry in entries_crystal))
    denominator = n_si + n_zn

    return {
        "enable_zinc": True,
        "Zn_site_type": site_type,
        "Zn_seed": int(seed),
        "target_Zn_Si_ratio": float(target_zinc_si_ratio),
        "actual_Zn_Si_ratio": float(n_zn / n_si) if n_si else None,
        "actual_Zn_Si_original_ratio": float(n_zn / n_si_original) if n_si_original else None,
        "Ca_Si_ratio": float(ca_si_ratio),
        "Ca_over_Si_plus_Zn_ratio": float(n_ca / denominator) if denominator else None,
        "N_Si_original": int(n_si_original),
        "N_Si_final": int(n_si),
        "N_Si": int(n_si),
        "N_Zn": int(n_zn),
        "N_Ca": int(n_ca),
        "N_Q1_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q1_Zn")),
        "N_Q2b_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q2b_Zn")),
        "N_Q1_candidates": int(len(candidates["Q1_Zn"])),
        "N_Q2b_candidates": int(len(candidates["Q2b_Zn"])),
        "selected_sites": [
            {
                "atom_id": site["atom_id"],
                "motif": site["motif"],
                "cell": site["cell"],
                "piece": site["piece"],
                "coord": site["coord"],
                "original_specie": site["original_specie"],
                "q1_selection_score": site.get("q1_selection_score"),
                "q1_selection_report": site.get("q1_selection_report"),
            }
            for site in selected_sites
        ],
        "minimum_Zn_Zn_distance": None if math.isinf(min_zn_zn) else float(min_zn_zn),
        "minimum_Zn_O_distance": None if math.isinf(min_zn_o) else float(min_zn_o),
        "total_charge_before_zinc": None,
        "total_charge_after_zinc_before_hydroxylation": total_charge,
        "total_charge_after_hydroxylation": total_charge,
        "total_charge_residual": total_charge,
        "charge_residual_final": total_charge,
        "N_Os_converted_to_Oh": 0,
        "N_H_added_for_Zn_OH": 0,
        "hydroxylation_records": [],
        "Ca_Si_original": float(n_ca / n_si_original) if n_si_original else None,
        "Ca_Si_final": float(n_ca / n_si) if n_si else None,
        "Zn_Si_original": float(n_zn / n_si_original) if n_si_original else None,
        "Zn_Si_final": float(n_zn / n_si) if n_si else None,
        "notes": [
            "v2 creates a charge-balanced ZnO2(OH)2 substitutional candidate.",
            "v2 does not use guest_ions/substitute and does not randomly replace Ca-layer atoms.",
            "Zn parameters are taken from CementFF4 supplementary information.",
        ],
    }


def nearest_oxygen_summary_for_site(entries_crystal, supercell, zn_id, hydroxylated_ids=None):
    coords = coords_by_atom_id(entries_crystal)
    atom_types = type_by_atom_id(entries_crystal)
    zn_coord = coords[int(zn_id)]
    hydroxylated = {int(x) for x in (hydroxylated_ids or [])}
    records = []
    for atom_id, specie in atom_types.items():
        if specie not in (3, 5, 6):
            continue
        records.append({
            "atom_id": int(atom_id),
            "atom_type": int(specie),
            "oxygen_role": oxygen_role_label(specie),
            "distance": float(periodic_distance(zn_coord, coords[atom_id], supercell)),
            "is_hydroxylated_oxygen": bool(int(atom_id) in hydroxylated),
        })
    records.sort(key=lambda item: item["distance"])
    return records[:8]


def build_multi_zinc_summary(
    entries_crystal,
    selected_sites,
    candidates,
    candidate_site_report,
    rejected_candidates,
    target_zinc_si_ratio,
    ca_si_ratio,
    supercell,
    mode,
    seed,
    min_zn_zn_distance,
):
    counts = count_species(entries_crystal)
    n_si = counts.get(2, 0) + counts.get(10, 0)
    n_zn = counts.get(ZN_SPECIE, 0)
    n_ca = counts.get(1, 0) + counts.get(9, 0)
    n_si_original = n_si + n_zn
    zn_zn = pairwise_zn_zn_distances(selected_sites, supercell)
    selected = []
    for site in selected_sites:
        hydroxylated = site.get("planned_hydroxylated_oxygen_ids", [])
        nearest = nearest_oxygen_summary_for_site(entries_crystal, supercell, site["atom_id"], hydroxylated)
        selected.append({
            "atom_id": int(site["atom_id"]),
            "motif": site["motif"],
            "motif_type": site["motif"],
            "substituted_si_atom_id": int(site["atom_id"]),
            "cell": site.get("cell"),
            "piece": site.get("piece"),
            "coord": site.get("coord"),
            "original_specie": site.get("original_specie"),
            "selected_O_atoms": nearest[:4],
            "planned_hydroxylated_oxygen_ids": hydroxylated,
            "initial_Zn_O_distances": nearest,
            "Zn_O_coordination_2p5A": sum(1 for item in nearest if item["distance"] <= 2.5),
            "center_passed_initial": sum(1 for item in nearest if item["distance"] <= 2.5) >= 4,
            "q1_selection_score": site.get("q1_selection_score"),
            "q1_selection_report": site.get("q1_selection_report"),
            "q2b_selection_report": site.get("q2b_selection_report"),
        })
    return {
        "enable_zinc": True,
        "Zn_site_type": "multi_Zn",
        "multi_Zn_mode": mode,
        "requested_mode": mode,
        "Zn_seed": int(seed),
        "target_Zn_Si_ratio": float(target_zinc_si_ratio),
        "actual_Zn_Si_ratio": float(n_zn / n_si) if n_si else None,
        "actual_Zn_Si_original_ratio": float(n_zn / n_si_original) if n_si_original else None,
        "Ca_Si_ratio": float(ca_si_ratio),
        "N_Si_original": int(n_si_original),
        "N_Si_final": int(n_si),
        "N_Zn": int(n_zn),
        "n_Zn_total": int(n_zn),
        "N_Ca": int(n_ca),
        "N_Q1_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q1_Zn")),
        "N_Q2b_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q2b_Zn")),
        "n_Q1_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q1_Zn")),
        "n_Q2b_Zn": int(sum(1 for site in selected_sites if site["motif"] == "Q2b_Zn")),
        "N_Q1_candidates": int(len(candidates["Q1_Zn"])),
        "N_Q2b_candidates": int(len(candidates["Q2b_Zn"])),
        "selected_sites": selected,
        "zn_centers": selected,
        "Zn_Zn_distances": zn_zn,
        "minimum_Zn_Zn_distance": None if not zn_zn else float(min(item["distance"] for item in zn_zn)),
        "min_zn_zn_distance_required": float(min_zn_zn_distance),
        "rejected_candidates": rejected_candidates,
        "candidate_site_report": candidate_site_report,
        "total_charge_before_zinc": None,
        "total_charge_after_zinc_before_hydroxylation": total_charge(entries_crystal),
        "total_charge_after_hydroxylation": total_charge(entries_crystal),
        "total_charge_residual": total_charge(entries_crystal),
        "charge_residual_final": total_charge(entries_crystal),
        "N_Os_converted_to_Oh": 0,
        "N_H_added_for_Zn_OH": 0,
        "hydroxylation_records": [],
        "notes": [
            "v1.6-alpha creates multiple independent Q1_Zn/Q2b_Zn motifs in one static parent structure.",
            "This is not the unsupported mixed_Q1_Q2b_Zn site type.",
            "Each motif remains independently selected, recorded, and validated.",
        ],
    }


def apply_zinc_modification(
    entries_crystal,
    crystal_dict,
    supercell,
    Zn_Si_ratio,
    Zn_site_type,
    Zn_seed,
    ca_si_ratio,
    charge_balance_mode="hydroxylate_two_oxygens",
    entries_bonds=None,
    entries_angle=None,
    allow_hydroxylate_bridging_oxygen=False,
    precondition_zinc_geometry=True,
    target_Zn_O_distance=1.95,
    q1_selection_mode=None,
):
    validate_zinc_site_type(Zn_site_type)
    validate_charge_balance_mode(charge_balance_mode)
    if q1_selection_mode is None:
        q1_selection_mode = os.environ.get("PYCSH_ZN_Q1_SELECTION_MODE", "ranked_static")
    if Zn_site_type == "Q1_Zn" and q1_selection_mode not in SUPPORTED_Q1_SELECTION_MODES:
        raise ValueError(
            "Unknown PYCSH_ZN_Q1_SELECTION_MODE={!r}. Expected one of {}".format(
                q1_selection_mode, sorted(SUPPORTED_Q1_SELECTION_MODES)
            )
        )
    if charge_balance_mode == "hydroxylate_two_oxygens" and (entries_bonds is None or entries_angle is None):
        raise ValueError("hydroxylate_two_oxygens requires entries_bonds and entries_angle")
    candidates = inspect_zinc_candidates(crystal_dict)
    candidate_site_report = None
    if entries_bonds is not None and entries_angle is not None:
        candidate_site_report = build_zinc_candidate_site_report(
            candidates,
            entries_crystal,
            entries_bonds,
            entries_angle,
            supercell,
            allow_hydroxylate_bridging_oxygen,
            precondition_zinc_geometry,
            target_Zn_O_distance,
        )
        if Zn_site_type == "Q1_Zn":
            candidates = attach_q1_scores_to_candidates(candidates, candidate_site_report)
    counts = count_species(entries_crystal)
    n_si_initial = counts.get(2, 0) + counts.get(10, 0)
    if n_si_initial <= 0:
        raise ValueError("Cannot place Zn because the generated structure contains no Si atoms")

    n_zinc = int(round(float(Zn_Si_ratio) * n_si_initial))
    if float(Zn_Si_ratio) > 0.0 and n_zinc == 0:
        n_zinc = 1

    site_filter = None
    if charge_balance_mode == "hydroxylate_two_oxygens":
        def site_filter(site):
            try:
                trial_entries = copy.deepcopy(entries_crystal)
                trial_bonds = copy.deepcopy(entries_bonds)
                trial_angles = copy.deepcopy(entries_angle)
                for entry in trial_entries:
                    if int(entry[0]) == int(site["atom_id"]):
                        entry[1] = ZN_SPECIE
                        entry[2] = ZN_CHARGE
                        break
                hydroxylate_two_oxygens(
                    trial_entries,
                    trial_bonds,
                    trial_angles,
                    [site],
                    supercell,
                    allow_hydroxylate_bridging_oxygen,
                    precondition_zinc_geometry,
                    target_Zn_O_distance,
                )
                return True
            except ValueError:
                return False

    selected_sites = select_zinc_sites(
        candidates,
        n_zinc,
        Zn_site_type,
        Zn_seed,
        supercell,
        site_filter=site_filter,
        q1_selection_mode=q1_selection_mode,
    )
    charge_before_zinc = total_charge(entries_crystal)
    entries_crystal, crystal_dict = apply_zinc_sites(entries_crystal, crystal_dict, selected_sites)
    charge_after_zinc = total_charge(entries_crystal)
    hydroxylation_records = apply_charge_balance(
        entries_crystal,
        entries_bonds,
        entries_angle,
        selected_sites,
        supercell,
        charge_balance_mode,
        allow_hydroxylate_bridging_oxygen,
        precondition_zinc_geometry,
        target_Zn_O_distance,
    )
    summary = build_zinc_summary(
        entries_crystal,
        selected_sites,
        candidates,
        Zn_Si_ratio,
        ca_si_ratio,
        supercell,
        Zn_site_type,
        Zn_seed,
    )
    summary["total_charge_before_zinc"] = charge_before_zinc
    summary["total_charge_after_zinc_before_hydroxylation"] = charge_after_zinc
    summary["total_charge_after_hydroxylation"] = total_charge(entries_crystal)
    summary["total_charge_residual"] = total_charge(entries_crystal)
    summary["charge_residual_final"] = total_charge(entries_crystal)
    summary["N_Os_converted_to_Oh"] = sum(len(record["hydroxylated_oxygens"]) for record in hydroxylation_records)
    summary["N_H_added_for_Zn_OH"] = summary["N_Os_converted_to_Oh"]
    summary["hydroxylation_records"] = hydroxylation_records
    summary["allow_hydroxylate_bridging_oxygen"] = bool(allow_hydroxylate_bridging_oxygen)
    summary["precondition_zinc_geometry"] = bool(precondition_zinc_geometry)
    summary["target_Zn_O_distance"] = float(target_Zn_O_distance)
    summary["candidate_site_report"] = candidate_site_report
    if Zn_site_type == "Q1_Zn":
        summary["Q1_Zn_selection_policy"] = build_q1_selection_policy(
            candidates,
            selected_sites,
            Zn_seed,
            q1_selection_mode,
        )
    summary["Q1_Zn_motif_assumption"] = (
        "Conservative static candidate: replace one Q1/terminal silicate Si center with Zn(+2), "
        "retain nearby framework O coordination, and convert two safe terminal/non-bridging O core-shell pairs "
        "to Oh-Hoh for explicit charge balance. This is not claimed to be the unique experimental Q(1,Zn) structure."
        if Zn_site_type == "Q1_Zn"
        else None
    )
    return entries_crystal, crystal_dict, summary


def write_zinc_summary(path, summary):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
