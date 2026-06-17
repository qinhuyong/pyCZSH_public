# Reproducibility Commands

Recommended public version: `v2.1.1-public-polish`.

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
When enabled, it performs plus/minus small-strain x-direction diagnostic input
checks only, not a full mechanics workflow.

## Site Mode Definitions

- `q2b_only` generates a single-Zn Q2b_Zn candidate.
- `q1_only` generates a single-Zn Q1_Zn candidate.
- `multi_q2b` generates a single C-S-H structure containing multiple Q2b_Zn motifs.
- `multi_q1` generates a single C-S-H structure containing multiple Q1_Zn motifs.
- `q1_q2b_single_structure_mixture` generates a single C-S-H structure containing both Q1_Zn and Q2b_Zn motifs.

If multiple Zn motifs are required in the same structure, use `multi_q1`,
`multi_q2b`, or `q1_q2b_single_structure_mixture`, not `q1_only` or
`q2b_only`.

Internal pyCZSH `.data` files may contain `CS-Info`, which is read by generated
LAMMPS inputs via `fix csinfo all property/atom i_CSID` and
`read_data DATAFILE fix csinfo NULL CS-Info`. Use `.clean.data` only for
external reading or visualization convenience.

## Expected Output

Smoke outputs should be written to the requested `output_pyczsh_*` directories
under the current working directory, not to `pyCSH_Zn/output_Y/`.

Each smoke output should include:

- `manifest.json`
- `composition_summary.csv`
- `accepted_models.csv` or `rejected_models.csv`
- `logs/run_summary.txt`

The summary files report target and actual Ca/Si, Zn/Si, and Q1/Q2b allocation.
