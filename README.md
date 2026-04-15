# Niccolo Forte's PhD Repository

This repository contains all of the scripts used throughout my PhD and is organized by paper, with a shared top-level Python package:

- `p1-DisorderLatticeProperties/`
- `p2-DisorderML/`
- `p3-DisorderIcingMitigation/`
- `resources/` (shared package used across all papers)

## Repository Structure

### `p1-DisorderLatticeProperties/`
Abaqus simulation scripts and data processing code for the FEA and post-process analysis of the mechanical performance of nodal quasi-disordered 2D lattices under uniaxial and compact tension.
- `code/`: notebooks and scripts for lattice properties, data processing, and ML workflows.
- `SIMscripts/`: ABAQUS local/HPC simulation and post-processing scripts.

### `p2-DisorderML/`
The python scripts implemented for the optimisation of mechanical performance through tuning of nodal disorder with Machine Learning.
- `code/`: paper-specific ML notebooks and scripts.

### `p3-DisorderIcingMitigation/`
The simulation and experimental planning scripts towards achieving a mechanism for decreasing ice adhesions to surfaces by leveraging disordered lattices.
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
- setup writes a `.pth` hook (`phd_shared_resources_repo.pth`) in ABAQUS user site-packages for reliability (and to cover local-package pip failures)
- setup verifies imports from a temp directory (not repo root), so import checks are real
- if `PIP_NO_INDEX` is set in the shell, setup temporarily unsets it during install and restores it afterwards

Optional Flags:
- `.\setup.ps1 -OnlyPython` (run only standard Python setup)
- `.\setup.ps1 -OnlyAbaqus` (run only ABAQUS Python setup)

## Remove Setup

To uninstall everything installed by setup (both interpreters), run:

```powershell
.\remove-setup.ps1
```

Default behavior:
- standard Python: uninstall local `resources` package and uninstall all packages listed in `requirements.txt`
- ABAQUS Python: uninstall local `resources` package and uninstall all packages listed in `requirements-abaqus.txt`
- remove fallback `.pth` hooks (`phd_shared_resources_repo.pth`) for both Python and ABAQUS (if present)

Important:
- keep only packages you want removable in `requirements-abaqus.txt` (for example `pandas`)
- verification at the end checks whether `resources` can still be discovered outside repo-root path injection

Optional Flags:
- `.\remove-setup.ps1 -OnlyPython` (run only standard Python removal)
- `.\remove-setup.ps1 -OnlyAbaqus` (run only ABAQUS removal)
- `.\remove-setup.ps1 -SkipPythonRequirementsUninstall` (remove the local `resources` package/hook but keep packages from `requirements.txt` and `requirements-abaqus.txt` installed)

## Notes

- `phd_shared_resources.egg-info/` can be created by editable installs and is expected.
- pip warnings such as `Ignoring invalid distribution -ygments/-ympy` come from existing broken metadata in the Python environment, not from this repository.
