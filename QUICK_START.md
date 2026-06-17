# Quick Start

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

Legacy workflow scripts are retained in `pyCSH_Zn/examples/` for compatibility,
but the recommended entry point is `main_pyczsh.py`.
