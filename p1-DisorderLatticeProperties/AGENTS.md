# p1 Disorder Lattice Properties Agent Context

This paper folder is the finite-element and data-generation layer of the repository. It studies quasi-disordered 2D lattice mechanical response under uniaxial/ductile and compact-tension/fracture loading, then prepares the simulation outputs that p2 uses for machine-learning surrogates.

Read `PROJECT_STATUS.md` only for planning, continuation, or handoff work; it records changing priorities without duplicating these durable rules.

## Folder Map

- `SIMscripts/` contains the active Abaqus model-generation, post-processing, Slurm, local-run, and transfer scripts. Read `SIMscripts/AGENTS.md` before editing or running anything there.
- `code/` contains notebooks for data processing, mechanics inspection, validation/convergence plots, stiffness/effective-property calculations, continuum-style displacement plots, and exploratory analysis. Read `code/AGENTS.md` before changing notebooks.
- `resources/` at repo root owns the shared mechanics and data contracts used by this folder, especially `lattices.py`, `abaqus.py`, `calculations.py`, `data_processing.py`, and `MLdata.py`.

## Workflow Overview

The active p1 workflow is:

```text
Abaqus model generation -> .inp/.odb files -> transfer CSV/NPZ files -> processed p1 CSV/NPZ files -> MLdata products for p2
```

In practical terms:

- `A1_FractureToughness-Ductility.py` builds ductile and fracture Abaqus jobs.
- `A2_INpostProcess.py`, `A2_OUTpostProcess.py`, and `A2_FieldOUTpostProcess.py` export input, curve-output, and displacement-field transfer files.
- `DataProcessing.ipynb` and `InputsOutputs.ipynb` convert transfer files into aligned ML-ready inputs, outputs, properties, and field arrays.
- p2 consumes those ML-ready products through `resources.MLdata.DATA`.

## How To Work Here

- Preserve file naming, sample-id alignment, and processed CSV/NPZ schemas unless the task explicitly asks for a migration.
- For any producer/consumer change, use `review-p1-p2-data-contract`; its reference owns the detailed names, shapes, metadata, and consumer map.
- Keep active p1 scripts and notebooks lean. Remove abandoned approaches, stale cells, obsolete helper paths, and superseded experiments as work progresses.
- Treat changes to lattice geometry, node counts, material laws, boundary conditions, smoothing, fracture-index logic, stiffness fits, and outlier rules as scientific changes, not simple refactors.
- Do not run Abaqus, Slurm, archive-transfer, ODB-upgrade, or recursive rename/delete scripts casually. Confirm the target paths and whether the script modifies research artifacts.
- Keep p1-specific workflow code in p1. Move behavior into `resources/` only when it is genuinely shared or needed by p2.
- `SIMscripts/OldScriptVersions/` is the place for historical script preservation. Current p1 scripts and notebooks should not accumulate old experiments or "just in case" fallbacks.
- Do not add p3 assumptions or conventions here; p3 is a separate and less settled paper folder.
