# Reproducibility Commands

Recommended public version: `v2.1.3-cell-geometry-and-supercell-audit`.

Run from the repository root.

## Syntax Checks

```bash
python -m py_compile main_pyczsh.py pyCSH_Zn/workflow.py pyCSH_Zn/periodic_recenter.py pyCSH_Zn/cell_audit.py
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
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.03 --site-mode q2b_only --seed-start 24000 --output-dir output_pyczsh_no_recenter --no-recenter
```

If LAMMPS is available:

```bash
python main_pyczsh.py --n-models 1 --target-ca-si 1.7 --target-w-si 0.2 --target-zn-si 0.03 --site-mode q2b_only --seed-start 23000 --output-dir output_pyczsh_smoke_lammps --build-lammps-inputs --run-static-relaxation
```

Do not run `--run-quasistatic` by default; it is an explicit diagnostic path.
When enabled, it performs plus/minus small-strain x-direction diagnostic input
checks only, not a full mechanics workflow.

## Periodic Recentering Audit

Default runs apply largest-gap-to-boundary periodic framework recentering before
validation and final internal data export. The audit file records the applied
fractional shift, largest gaps before and after, framework atom count,
fractional spans, and warnings. This is a coordinate representation change
only; it does not change chemistry, density, topology, `CS-Info`, force-field
parameters, or validation semantics.

OVITO atom-wise wrapping can still split connected frameworks if applied
blindly, so use pyCZSH recentered data files for visualization and replicated
cell inspection. The `--no-recenter` option is intended for debugging or
comparison with the original coordinate representation.

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
- `structures/model_000001/internal/periodic_recenter_summary.json`
- `logs/run_summary.txt`

The summary files report target and actual Ca/Si, Zn/Si, and Q1/Q2b allocation.
They also record whether periodic framework recentering was applied.

v2.1.3 outputs should also include:

- `structures/model_000001/internal/framework_occupancy_summary.json`
- `structures/model_000001/internal/cell_geometry_summary.json`
- `structures/model_000001/internal/brick_placement_summary.json`
- `structures/model_000001/internal/dedup_audit_summary.json`

If `--export-clean-data` is enabled, the LAMMPS folder may also include
`model_000001.visual_orthogonal.clean.data`, which is visualization-only.
