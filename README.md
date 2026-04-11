# Niccolo Forte's PhD Repository

This repository is organized by PhD paper, with a shared top-level Python package:

- `p1-DisorderLatticeProperties/`
- `p2-DisorderML/`
- `p3-DisorderIcingMitigation/`
- `resources/` (shared package used across all papers)

## Repository Structure

### `p1-DisorderLatticeProperties/`
- `code/`: notebooks and scripts for lattice properties, data processing, and ML workflows.
- `SIMscripts/`: ABAQUS local/HPC simulation and post-processing scripts.

### `p2-DisorderML/`
- `code/`: paper-specific ML notebooks and scripts.

### `p3-DisorderIcingMitigation/`
- `SIMscripts/`: paper-specific ABAQUS simulation and post-processing scripts.

### `resources/`
Shared reusable Python modules for all papers, for example:
- `calculations.py`
- `lattices.py`
- `MLdata.py`
- `MLfunc.py`
- `MLmodels.py`
- `tokenization.py`
- `utilities.py`

Use imports like:

```python
from resources.module_name import function_name
```

## Python Setup

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` includes `-e .`, so this single command installs:
- third-party dependencies, and
- the local `resources` package in editable mode.

Editable mode means changes to files inside `resources/` are picked up without reinstalling.

Important:
- Run install commands from the repository root.
- Do not use `pip install -e resources` (the project metadata is in root `pyproject.toml`, not inside `resources/`).

## ABAQUS Python Setup

If ABAQUS scripts need to import `resources`, install with the ABAQUS interpreter:

```powershell
abaqus python -m pip install -r requirements.txt
```

If `pip` is not available in your ABAQUS Python, use a `PYTHONPATH` approach or add a small `sys.path` bootstrap in the ABAQUS script.

## Notes

- `phd_shared_resources.egg-info/` is created by editable installs and is expected.
- It is local install metadata and should not be committed.
