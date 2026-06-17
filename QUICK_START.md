# Quick Start

Recommended public version: `v2.1.3-cell-geometry-and-supercell-audit`.

Run from the repository root.

Linux:

```bash
python3 main_pyczsh.py
```

Windows:

```powershell
python main_pyczsh.py
```

The default output directory is `output_pyczsh/` relative to the current
working directory.

Periodic framework recentering is enabled by default. It moves the largest
framework gap to the cell boundary for a compact visualization-friendly periodic
representation. Use `--no-recenter` only for debugging or comparison with the
original coordinates.

## Common Commands

Generate one mixed Q1_Zn/Q2b_Zn C-Z-S-H model:

```bash
python main_pyczsh.py
```

Generate ten models:

```bash
python main_pyczsh.py --n-models 10 --seed-start 12000
```

Set target composition:

```bash
python main_pyczsh.py --target-ca-si 1.5 --target-w-si 0.2 --target-zn-si 0.05
```

Use a fixed target Zn count:

```bash
python main_pyczsh.py --target-ca-si 1.7 --target-zn-count 4 --q1-q2b-ratio 0.5
```

Build LAMMPS input files:

```bash
python main_pyczsh.py --build-lammps-inputs
```

Static relaxation is opt-in:

```bash
python main_pyczsh.py --build-lammps-inputs --run-static-relaxation
```

Quasi-static diagnostics are also opt-in:

```bash
python main_pyczsh.py --build-lammps-inputs --run-static-relaxation --run-quasistatic
```

`--run-quasistatic` runs plus/minus small-strain x-direction diagnostic input
checks only. It does not report final elastic constants or production
mechanical properties.

## Visualization Note

OVITO `Wrap at periodic boundaries` wraps atoms independently and may split a
connected C-S-H framework across the displayed box. The default pyCZSH
recentered internal data files are recommended for visualization and replicated
cell inspection. Recentering does not change cell parameters, force-field
parameters, Zn motifs, atom IDs, atom types, bonds, angles, `CS-Info`, or
validation semantics. If large voids remain, inspect box size, brick
translation, and triclinic cell export.

## Cell Audit Outputs

If a recentered model still appears to occupy only a small part of the displayed
box, inspect `framework_occupancy_summary.json` and
`cell_geometry_summary.json`. v2.1.3 also writes
`brick_placement_summary.json` and `dedup_audit_summary.json` for each model.
With `--export-clean-data`, pyCZSH writes
`model_XXXX.visual_orthogonal.clean.data` as an OVITO-only diagnostic export.
Do not use that visual-only file for validation or LAMMPS relaxation.

## Site Mode Reminder

- `q2b_only` generates a single-Zn Q2b_Zn candidate.
- `q1_only` generates a single-Zn Q1_Zn candidate.
- `multi_q2b` generates a single C-S-H structure containing multiple Q2b_Zn motifs.
- `multi_q1` generates a single C-S-H structure containing multiple Q1_Zn motifs.
- `q1_q2b_single_structure_mixture` generates a single C-S-H structure containing both Q1_Zn and Q2b_Zn motifs.

If multiple Zn motifs are required in the same structure, use `multi_q1`,
`multi_q2b`, or `q1_q2b_single_structure_mixture`, not `q1_only` or
`q2b_only`.

Internal pyCZSH `.data` files may contain `CS-Info` for validator core-shell
metadata. External tools may not understand this section; use `.clean.data`
exports only for visualization or external reading convenience.

Legacy workflow scripts are retained in `pyCSH_Zn/examples/` for compatibility,
but the recommended entry point is `main_pyczsh.py`.
