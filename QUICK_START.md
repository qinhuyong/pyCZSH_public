# Quick Start

Recommended public version: `v2.1.1-public-polish`.

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
