# Public Package Manifest

This public package contains the runtime code and supporting data needed to run
the pyCZSH workflow.

Recommended public version: `v2.1.2-periodic-framework-recenter`.

Included:

- `main_pyczsh.py`: recommended executable entry point.
- `pyCSH_Zn/workflow.py`: unified workflow functions.
- `pyCSH_Zn/periodic_recenter.py`: largest-gap-to-boundary periodic framework
  recentering for compact visualization-friendly internal data files.
- Core pyCSH_Zn construction, Zn placement, validation, and writing modules.
- `pyCSH_Zn/Blocks_Renamed_Y/`: brick resources required for construction.
- `pyCSH_Zn/forcefields/CementFF4_Zn_parameters.json`: force-field database used by the writer.
- `pyCSH_Zn/lammps_templates/`: LAMMPS input templates.
- `pyCSH_Zn/examples/`: legacy compatibility scripts.
- `README.md`, `QUICK_START.md`, `REPRODUCIBILITY_COMMANDS.md`, `LICENSE`, and citation notes.

Excluded:

- Generated structures and simulation outputs.
- `output_pyczsh/`, `output_Y/`, caches, logs, dumps, and trajectory files.
- Development notes and manuscript drafting files not required to run the public workflow.

Recommended entry point:

```bash
python main_pyczsh.py
```

Default model generation writes `internal/periodic_recenter_summary.json`.
Recentering is a periodic coordinate representation change only and preserves
cell parameters, topology, atom IDs, atom types, charges, bonds, angles,
`CS-Info`, force-field parameters, and validation semantics.
