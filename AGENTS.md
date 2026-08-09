# Repository Agent Context

This repository is organized by PhD paper folder with one shared Python package. Start with this file, then read the nearest nested `AGENTS.md` before editing inside a paper folder.

## Repository Map

- `p1-DisorderLatticeProperties/` is the Abaqus FEA, post-processing, and data-generation workflow for quasi-disordered 2D lattices under uniaxial/ductile and compact-tension/fracture loading.
- `p2-DisorderML/` is the machine-learning and surrogate-modeling workflow built on p1 outputs. It trains and diagnoses curve and field surrogates and contains cluster training/HPO scripts.
- `p3-DisorderIcingMitigation/` is intentionally not covered by an AGENTS file yet. Treat it as uncertain and do not add guidance or borrow conventions from it unless the user explicitly asks.
- `resources/` is the shared package used by p1 and p2. Changes there can affect Abaqus scripts, notebooks, ML training, post-processing, and saved artifact loading.

## Environment And Imports

- `pyproject.toml` exposes `resources` as the editable package `phd-shared-resources`.
- `setup.ps1` installs the shared package for standard Python and Abaqus Python; `remove-setup.ps1` reverses that setup.
- Use imports like `from resources.MLdata import DATA` or `from resources.lattices import Geometry`; avoid local `sys.path` hacks unless an Abaqus/HPC launch path truly requires them.
- Standard Python and Abaqus Python are separate environments. Code that touches Abaqus model/session/ODB APIs must be validated in the Abaqus interpreter or by syntax review only when Abaqus is unavailable.

## Data And Path Conventions

- Local p1 data is commonly rooted at `Z:/p1/data/Ti/...`.
- Local p2 saved runs are commonly rooted at `Z:/p2`.
- QMUL HPC archive paths commonly use `/data/SEMS-TaoLab/Niccolo-Forte/...`; scratch paths may use `/gpfs/scratch/...`.
- Generated Abaqus files, transfer CSVs/NPZs, MLdata products, model checkpoints, HPO databases, Slurm logs, and notebook output bulk are research artifacts. Do not delete, rename, or commit them unless the task explicitly asks for that.
- The repository currently ignores `*AGENTS.md`, so these guidance files are local unless the ignore rule is changed or they are force-added.

## Cross-Folder Workflow

- p1 `SIMscripts/` creates Abaqus `.inp` and `.odb` files and exports `transfer/` CSV/NPZ files.
- p1 `code/` notebooks turn `transfer/` files into ML-ready CSV/NPZ data and inspect mechanics/validation results.
- `resources/data_processing.py`, `resources/MLdata.py`, and `resources/MLmetrics.py` form the main bridge from p1 outputs to p2 training and diagnostics.
- p2 notebooks and HPC scripts should treat p1 data as upstream unless the user explicitly asks to change data generation.

## Editing Rules

- Read the script/notebook plus the shared helper it calls before changing behavior.
- Preserve file naming and index conventions between p1 and p2. In particular, `Ductile`, `Fracture`, `UT`, `FT`, `MULTI`, `per`, `disNodes`, `disStruts`, and simulation id `0` for the periodic reference have downstream meaning.
- Keep path overrides explicit in notebooks and scripts. Do not hide new global path constants inside shared modules.
- Ask for clarification before changing a scientific assumption, simulation setup, lattice definition, material law, loss definition, or data-filtering rule that cannot be verified from local context.
- For notebooks, prefer targeted cell/helper edits. Running full notebooks may require large data, Abaqus artifacts, or HPC resources.

## Cleanup Constraint

Treat the user's cleanup preference as a high-priority engineering constraint, not a cosmetic preference.

- Active scripts, notebooks, and helper modules should stay clean, compact, current, and intentional.
- Do not leave behind dated, stale, legacy, failed, partial, duplicated, or superseded code in active work areas.
- If an attempted approach is abandoned, remove it. If a helper exists only because of an old workaround and is no longer needed, remove it.
- If notebook cells are replaced by helper functions, remove the stale code paths and clear stale outputs instead of preserving them "just in case".
- Avoid verbose safety-net layers, fallback branches, legacy compatibility scaffolding, and overly defensive helper functions unless there is a concrete current need.
- Prefer direct, readable code and compact helpers. If a helper is used once and does not clarify the main flow, consider inlining it. If logic is reused or makes notebooks cleaner, move it into the appropriate `resources/` module.
- Important exception: archived or explicitly historical folders, such as `p1-DisorderLatticeProperties/SIMscripts/OldScriptVersions/`, can preserve old versions. Active/current scripts and notebooks should not accumulate old experiments.
- In short: clean up as you go.

## Validation

- For standard Python changes, use targeted import checks, `python -m py_compile`, or focused tests when data is available.
- For shell scripts, use `bash -n` where possible.
- For Abaqus scripts, distinguish syntax checks from real Abaqus execution. Real behavior often depends on Abaqus CAE, license access, and local/HPC file layout.
