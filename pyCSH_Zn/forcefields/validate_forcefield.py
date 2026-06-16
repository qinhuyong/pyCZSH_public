from __future__ import print_function

import argparse
import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CementFF4_Zn_parameters.json")


PAIR_PARAM_RULES = {
    "buck/coul/long": (3, 4),
    "buck/coul/long/cs": (3, 4),
    "nm/cut/coul/long": (4, 5),
    "lj/cut/tip4p/long": (2, 3),
}


def load_database(path=DEFAULT_DB):
    with open(path) as f:
        return json.load(f)


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def audit_pair_coeff_line(line, atom_type_ids):
    parts = line.split("#", 1)[0].split()
    record = {"line": line, "ok": True, "errors": [], "warnings": []}
    if len(parts) < 5 or parts[0] != "pair_coeff":
        record["ok"] = False
        record["errors"].append("line must start with pair_coeff and include i j style")
        return record
    i, j, style = parts[1], parts[2], parts[3]
    params = parts[4:]
    record.update({"i": i, "j": j, "style": style, "params": params})
    if i not in atom_type_ids or j not in atom_type_ids:
        record["ok"] = False
        record["errors"].append("unknown atom type id")
    if style not in PAIR_PARAM_RULES:
        record["ok"] = False
        record["errors"].append("unknown pair style for CementFF4-Zn audit")
        return record
    min_count, max_count = PAIR_PARAM_RULES[style]
    if not (min_count <= len(params) <= max_count):
        record["ok"] = False
        record["errors"].append(
            "{} expects {}-{} numeric parameters, found {}".format(style, min_count, max_count, len(params))
        )
    if any(not _is_number(x) for x in params):
        record["ok"] = False
        record["errors"].append("all pair_coeff parameters must be numeric")
    if style == "lj/cut/tip4p/long" and len(params) == 3:
        record["warnings"].append("third lj/cut/tip4p/long parameter is interpreted as an optional pair cutoff")
    return record


def validate_database(db):
    atom_types = db.get("atom_types", {})
    atom_type_ids = set(atom_types.keys())
    required_atom_type_ids = set(k for k, v in atom_types.items() if not v.get("optional"))
    records = [audit_pair_coeff_line(line, atom_type_ids) for line in db.get("pair_coeff_lines", [])]
    pair_keys = set()
    duplicates = []
    for rec in records:
        if not rec.get("i") or not rec.get("j"):
            continue
        key = tuple(sorted((int(rec["i"]), int(rec["j"]))))
        if key in pair_keys:
            duplicates.append({"i": rec["i"], "j": rec["j"], "line": rec["line"]})
        pair_keys.add(key)
    expected = int(len(required_atom_type_ids) * (len(required_atom_type_ids) + 1) / 2)
    missing = []
    optional_missing = []
    for i in sorted(int(x) for x in required_atom_type_ids):
        for j in sorted(int(x) for x in required_atom_type_ids):
            if j < i:
                continue
            if (i, j) not in pair_keys:
                missing.append({"i": i, "j": j})
    for i in sorted(int(x) for x in atom_type_ids):
        for j in sorted(int(x) for x in atom_type_ids):
            if j < i or (str(i) in required_atom_type_ids and str(j) in required_atom_type_ids):
                continue
            if (i, j) not in pair_keys:
                optional_missing.append({"i": i, "j": j})
    ok = all(rec["ok"] for rec in records) and not duplicates and not missing
    return {
        "ok": ok,
        "n_pair_coeff_lines": len(records),
        "expected_pair_coeff_lines": expected,
        "pair_coeff_records": records,
        "duplicates": duplicates,
        "missing_pairs": missing,
        "optional_missing_pairs": optional_missing,
        "notes": [
            "This audit validates static LAMMPS pair_coeff syntax and pair coverage.",
            "A generated LAMMPS in.read_check should still be run with the target LAMMPS executable before production use.",
        ],
    }


def write_report(report, output):
    with open(output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Audit CementFF4-Zn pair_coeff syntax.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=os.path.join(ROOT, "output_Y", "forcefields", "forcefield_validation_report.json"))
    args = parser.parse_args()
    db = load_database(args.db)
    report = validate_database(db)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    write_report(report, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
