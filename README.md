# pyCZSH public workflow package

pyCZSH constructs and validates pure and Zn-modified calcium silicate hydrate
(C-S-H / C-Z-S-H) atomistic structures for CementFF4-Zn style LAMMPS workflows.

The recommended executable file is:

```text
main_pyczsh.py
```

Legacy workflow scripts are retained under `pyCSH_Zn/examples/` for
compatibility. New runs should use `main_pyczsh.py`.

## Quick Use

Linux:

```bash
python3 main_pyczsh.py
```

Windows:

```powershell
python main_pyczsh.py
```

By default, output is written to `output_pyczsh/` relative to the current
working directory.

## Main Options

- `--seed`
- `--seed-start`
- `--target-ca-si`
- `--target-w-si`
- `--target-zn-si`
- `--target-zn-count`
- `--site-mode`
- `--q1-q2b-ratio`
- `--n-models`
- `--output-dir`
- `--ideal-only`
- `--build-lammps-inputs`
- `--run-static-relaxation`
- `--run-quasistatic`
- `--export-clean-data`
- `--workers`

Default behavior:

```text
site_mode = q1_q2b_single_structure_mixture
target_ca_si = 1.7
target_w_si = 0.2
target_zn_si = 0.05
q1_q2b_ratio = 0.5
n_models = 1
seed_start = 12000
output_dir = output_pyczsh
run_static_relaxation = false
run_quasistatic = false
export_clean_data = false
workers = 1
```

`--q1-q2b-ratio` is the target fraction `N_Q1_Zn / N_Zn_total` for
`q1_q2b_single_structure_mixture`. For example, if the target Zn count is 6 and
`--q1-q2b-ratio 0.33`, the target allocation is approximately 2 Q1_Zn and 4
Q2b_Zn centers. The workflow reports both target and actual Q1/Q2b allocation.

## Example Commands

Generate one default C-Z-S-H model:

```bash
python3 main_pyczsh.py
```

Generate a batch:

```bash
python3 main_pyczsh.py \
  --target-ca-si 1.5 \
  --target-w-si 0.2 \
  --target-zn-si 0.05 \
  --q1-q2b-ratio 0.5 \
  --n-models 10 \
  --output-dir output_pyczsh_ca15_zn005
```

Specify a Zn count:

```bash
python3 main_pyczsh.py \
  --target-ca-si 1.7 \
  --target-w-si 0.2 \
  --target-zn-count 4 \
  --q1-q2b-ratio 0.5 \
  --site-mode q1_q2b_single_structure_mixture
```

Build LAMMPS input files:

```bash
python3 main_pyczsh.py --build-lammps-inputs
```

Run static relaxation only when explicitly requested:

```bash
python3 main_pyczsh.py --build-lammps-inputs --run-static-relaxation
```

Quasi-static diagnostics are opt-in:

```bash
python3 main_pyczsh.py \
  --build-lammps-inputs \
  --run-static-relaxation \
  --run-quasistatic
```

## Site Modes

- `q2b_only`: one Q2b_Zn candidate.
- `q1_only`: one Q1_Zn candidate.
- `multi_q2b`: multiple Q2b_Zn motifs in one C-S-H structure.
- `multi_q1`: multiple Q1_Zn motifs in one C-S-H structure.
- `q1_q2b_single_structure_mixture`: Q1_Zn and Q2b_Zn motifs in the same structure.

## Output Layout

The default output directory is `output_pyczsh/`:

```text
output_pyczsh/
  manifest.json
  composition_summary.csv
  composition_summary.json
  accepted_models.csv
  rejected_models.csv
  failure_reason_summary.csv
  coordination_quality_summary.csv
  representative_models.json
  structures/model_000001/internal/
  structures/model_000001/lammps/
  structures/model_000001/postmin/
  logs/run_summary.txt
```

Internal `.data` files retain `CS-Info` for validation and core-shell metadata.
Optional `.clean.data` files are written only with `--export-clean-data` and are
for external reading or visualization convenience.

## Scope And Limits

This package does not modify CementFF4-Zn force-field parameters. It preserves
the existing validation semantics, including the 2.5 A Zn-O coordination gate.

The workflow is for construction, validation, screening, and opt-in
quasi-static diagnostic infrastructure. It does not provide finite-temperature
MD, `md_ready_candidate` labels, final elastic constants, or production
mechanical properties.

Target Ca/Si and Zn/Si are target-window values, not guaranteed exact final
compositions. Actual Ca/Si, Zn/Si, and Q1/Q2b allocation are reported in the
summary files.
