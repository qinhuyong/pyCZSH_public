from __future__ import print_function

import argparse
import json
import os


def write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def header(data_file, ff_file, run_dir):
    data_ref = os.path.relpath(os.path.abspath(data_file), os.path.abspath(run_dir))
    ff_ref = os.path.relpath(os.path.abspath(ff_file), os.path.abspath(run_dir))
    return [
        "clear",
        "units metal",
        "dimension 3",
        "atom_style full",
        "boundary p p p",
        "box tilt large",
        "fix csinfo all property/atom i_CSID",
        "read_data {}".format(data_ref) + " fix csinfo NULL CS-Info",
        "include {}".format(ff_ref),
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
        "comm_modify vel yes cutoff 14.0",
        "thermo 100",
        "thermo_style custom step pe ebond eangle evdwl ecoul elong fnorm fmax press",
    ]


def build(data_file, ff_file, out_dir, prefix="static"):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    outputs = {}
    outputs["read_check"] = os.path.join(out_dir, "in.read_check")
    write(outputs["read_check"], header(data_file, ff_file, out_dir) + [
        "print \"LAMMPS read_check completed for {}\"".format(prefix),
    ])

    outputs["run0"] = os.path.join(out_dir, "in.run0")
    write(outputs["run0"], header(data_file, ff_file, out_dir) + [
        "run 0",
    ])

    outputs["minimize_static"] = os.path.join(out_dir, "in.minimize_static")
    write(outputs["minimize_static"], header(data_file, ff_file, out_dir) + [
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-6 1e-8 1000 10000",
        "write_data {}_minimized_static.raw.data nocoeff".format(prefix),
    ])

    outputs["static_relax_shell"] = os.path.join(out_dir, "in.static_relax_shell")
    write(outputs["static_relax_shell"], header(data_file, ff_file, out_dir) + [
        "group mobile type 4",
        "group fixed subtract all mobile",
        "fix freeze fixed setforce 0.0 0.0 0.0",
        "min_style fire",
        "min_modify dmax 0.001",
        "minimize 1e-8 1e-10 500 5000",
        "unfix freeze",
        "write_data {}_shell_relaxed_static.raw.data nocoeff".format(prefix),
    ])

    outputs["elastic_x_plus"] = os.path.join(out_dir, "in.elastic_x_plus")
    write(outputs["elastic_x_plus"], header(data_file, ff_file, out_dir) + [
        "# Quasi-static x-direction positive small-strain smoke test.",
        "# This validates the input path only; it is not a full elastic constants calculation.",
        "variable strain equal 0.001",
        "variable sx equal 1.0+${strain}",
        "change_box all x scale ${sx} remap",
        "min_style cg",
        "min_modify dmax 0.002",
        "minimize 1e-6 1e-8 1000 10000",
        "write_data {}_elastic_x_plus.raw.data nocoeff".format(prefix),
    ])

    outputs["elastic_x_minus"] = os.path.join(out_dir, "in.elastic_x_minus")
    write(outputs["elastic_x_minus"], header(data_file, ff_file, out_dir) + [
        "# Quasi-static x-direction negative small-strain smoke test.",
        "# This validates the input path only; it is not a full elastic constants calculation.",
        "variable strain equal 0.001",
        "variable sx equal 1.0-${strain}",
        "change_box all x scale ${sx} remap",
        "min_style cg",
        "min_modify dmax 0.002",
        "minimize 1e-6 1e-8 1000 10000",
        "write_data {}_elastic_x_minus.raw.data nocoeff".format(prefix),
    ])

    manifest = os.path.join(out_dir, "lammps_input_manifest.json")
    with open(manifest, "w") as f:
        json.dump(outputs, f, indent=2, sort_keys=True)
        f.write("\n")
    outputs["manifest"] = manifest
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--ff", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default="static")
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.ff, args.out, args.prefix), indent=2))


if __name__ == "__main__":
    main()
