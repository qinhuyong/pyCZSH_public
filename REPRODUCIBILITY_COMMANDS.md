# Reproducibility Commands

Run from the repository root.

## Syntax Checks

```bash
python -m py_compile main_pyczsh.py pyCSH_Zn/workflow.py
python -m py_compile pyCSH_Zn/examples/01_generate_pure_csh.py
python -m py_compile pyCSH_Zn/examples/02_generate_q2b_zn.py
python -m py_compile pyCSH_Zn/examples/08_generate_q1_zn.py
python -m py_compile pyCSH_Zn/examples/19_generate_composition_targeted_zn_csh.py
```

## Main Smoke Tests

```bash
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.05 --site-mode q1_q2b_single_structure_mixture --q1-q2b-ratio 0.5 --seed-start 20000 --output-dir output_pyczsh_smoke
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.03 --site-mode q2b_only --seed-start 21000 --output-dir output_pyczsh_smoke_q2b
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.06 --site-mode multi_q2b --seed-start 22000 --output-dir output_pyczsh_smoke_multi_q2b
```

If LAMMPS is available:

```bash
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.03 --site-mode q2b_only --seed-start 23000 --output-dir output_pyczsh_smoke_lammps --build-lammps-inputs --run-static-relaxation
```

Do not run `--run-quasistatic` by default; it is an explicit diagnostic path.

## Expected Output

Smoke outputs should be written to the requested `output_pyczsh_*` directories
under the current working directory, not to `pyCSH_Zn/output_Y/`.

Each smoke output should include:

- `manifest.json`
- `composition_summary.csv`
- `accepted_models.csv` or `rejected_models.csv`
- `logs/run_summary.txt`

The summary files report target and actual Ca/Si, Zn/Si, and Q1/Q2b allocation.
