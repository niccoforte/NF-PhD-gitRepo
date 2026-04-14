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

## Setup (Python + ABAQUS)

Recommended one-command setup from repo root:

```powershell
.\setup.ps1
```

This installs for both interpreters:
- all dependencies from `requirements.txt` into standard Python
- all dependencies from `requirements-abaqus.txt` into ABAQUS Python
- the local repo package (`resources`) for standard Python
- the local repo package (`resources`) for ABAQUS Python

Important implementation details:
- the script does **not** create `.pydeps/` or `.abaqus-pydeps/`
- if pip install of the local package fails in ABAQUS, setup writes a `.pth` hook (`phd_shared_resources_repo.pth`) in ABAQUS user site-packages as fallback
- setup verifies imports from a temp directory (not repo root), so import checks are real
- if `PIP_NO_INDEX` is set in the shell, setup temporarily unsets it during install and restores it afterwards

## Remove Setup

To uninstall everything installed by setup (both interpreters), run:

```powershell
.\remove-setup.ps1
```

Default behavior:
- standard Python: uninstall local `resources` package and uninstall all packages listed in `requirements.txt`
- ABAQUS Python: uninstall local `resources` package only
- remove fallback `.pth` hooks (`phd_shared_resources_repo.pth`) for both Python and ABAQUS (if present)

Important:
- ABAQUS built-ins like `numpy`/`matplotlib` are intentionally not removed by this script
- verification at the end checks whether `resources` can still be discovered outside repo-root path injection

Useful options:
- `.\remove-setup.ps1 -SkipPythonRequirementsUninstall` (remove only local package setup, keep standard Python requirements installed)

## Notes

- `phd_shared_resources.egg-info/` can be created by editable installs and is expected.
- pip warnings such as `Ignoring invalid distribution -ygments/-ympy` come from existing broken metadata in the Python environment, not from this repository.
