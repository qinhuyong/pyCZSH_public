from __future__ import print_function

import argparse
import json
import os

try:
    from validate_forcefield import validate_database, write_report
except ImportError:
    from forcefields.validate_forcefield import validate_database, write_report


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CementFF4_Zn_parameters.json")


def load_database(path=DEFAULT_DB):
    with open(path) as f:
        return json.load(f)


def write_type_map(db, output):
    mapping = {
        "units": db["units"],
        "atom_type_map": db["atom_types"],
        "bond_type_map": db["bond_types"],
        "angle_type_map": db["angle_types"],
        "source": db.get("source"),
    }
    with open(output, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
        f.write("\n")


def write_forcefield(db, output):
    atoms = db["atom_types"]
    bonds = db["bond_types"]
    angles = db["angle_types"]
    with open(output, "w") as f:
        f.write("# Generated from forcefields/CementFF4_Zn_parameters.json\n")
        f.write("# Units: {}\n".format(db["units"]))
        f.write("# Source: {}\n\n".format(db.get("source", "")))
        group_names = {
            "1": "ca",
            "2": "Si",
            "3": "Osicore",
            "4": "Osishell",
            "5": "Ow",
            "6": "Ooh",
            "7": "Hw",
            "8": "Hoh",
            "9": "Zn",
            "10": "Al",
            "11": "Cl",
        }
        for type_id in sorted(atoms, key=lambda x: int(x)):
            atom = atoms[type_id]
            if atom.get("optional") and type_id not in ("10", "11"):
                continue
            f.write("group {} type {}\n".format(group_names.get(type_id, "type" + type_id), type_id))
        f.write("\n")
        f.write("group cores type 3\n")
        f.write("group shells type 4\n")
        f.write("group noWater type 1 2 3 4 6 8 9 10 11\n")
        f.write("group water type 5 7\n\n")
        f.write("pair_style  {}\n\n".format(db["pair_style"]))
        for line in db["pair_coeff_lines"]:
            f.write(line + "\n")
        f.write("\n")
        f.write("bond_style hybrid  harmonic morse\n")
        for type_id in sorted(bonds, key=lambda x: int(x)):
            bond = bonds[type_id]
            if bond["style"] == "harmonic":
                f.write("bond_coeff {:>3s} harmonic {:g} {:g} # {}\n".format(type_id, bond["K"], bond["r0"], bond["label"]))
            elif bond["style"] == "morse":
                f.write("bond_coeff {:>3s} morse {:g} {:g} {:g} # {}\n".format(type_id, bond["D0"], bond["alpha"], bond["r0"], bond["label"]))
        f.write("\nangle_style harmonic\n")
        for type_id in sorted(angles, key=lambda x: int(x)):
            angle = angles[type_id]
            f.write("angle_coeff {:>3s} {:g} {:g} # {}\n".format(type_id, angle["K"], angle["theta"], angle["label"]))
        f.write("\n")
        f.write("kspace_style {}\n".format(db["kspace_style"]))
        f.write("fix tip4p_shake water shake 1e-4 150 0 b 2 a 1\n")


def build(output_dir, db_path=DEFAULT_DB):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    db = load_database(db_path)
    ff_path = os.path.join(output_dir, "in.CementFF4_Zn")
    map_path = os.path.join(output_dir, "cementff4_type_map.json")
    report_path = os.path.join(output_dir, "forcefield_validation_report.json")
    write_forcefield(db, ff_path)
    write_type_map(db, map_path)
    write_report(validate_database(db), report_path)
    return {"forcefield": ff_path, "type_map": map_path, "validation_report": report_path}


def main():
    parser = argparse.ArgumentParser(description="Build CementFF4-Zn include files from the JSON database.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=os.path.join(ROOT, "output_Y", "forcefields"))
    args = parser.parse_args()
    result = build(args.out, args.db)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
