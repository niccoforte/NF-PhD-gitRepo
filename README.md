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
- third-party dependencies
- the local `resources` package in editable mode

Editable mode means changes inside `resources/` are picked up without reinstalling.

Important:
- Run install commands from the repository root.
- Do not use `pip install -e resources` (project metadata is in root `pyproject.toml`).

## ABAQUS Python Setup

Recommended one-command setup from repo root:

```powershell
.\setup-abaqus.ps1
```

Manual steps:

```powershell
abaqus python -m pip install -r requirements-abaqus.txt
abaqus python -m pip install --no-build-isolation -e .
```

The setup script:
- checks if ABAQUS Python has `pip`
- tries `ensurepip` if `pip` is missing
- installs third-party deps from `requirements-abaqus.txt`
- tries to install the local repo package (`resources`) into ABAQUS Python in editable mode
- if editable install fails, tries to write a persistent `.pth` entry in ABAQUS `site-packages`
- if ABAQUS `site-packages` is not writable, falls back to local `.abaqus-pydeps/` plus `PYTHONPATH`
- attempts to persist `PYTHONPATH` via `setx` when possible
- verifies imports with `resources.lattices`

## Notes

- `phd_shared_resources.egg-info/` is created by editable installs and is expected.
- This metadata should not be committed.
