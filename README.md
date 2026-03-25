# Niccolo Forte's PhD Repository

#### Optional but recommended:
After cloning, create a local virtual environment and install Python dependencies from `requirements.txt`.

Access repo directory: `cd path\to\repo\p1git-Lattices`  
Initiate venv: `python -m venv venv`  
Activate venv (Windows): `venv\Scripts\activate`  
Install required packages: `pip install -r requirements.txt`  
Deactivate venv: `deactivate`  

Using a virtual environment keeps this project's package versions isolated from your base Python installation.

## ABAQUS Simulation Files
ABAQUS scripts for local and HPC simulation execution are in `SIMscripts`.

- `A1_FractureToughness-Ductility.py` / `A-HPC-1_FractureToughness-Ductility.py`:
  Main simulation drivers for fracture toughness and ductility runs.
- `A2_INpostProcess.py`, `A2_OUTpostProcess.py` and HPC variants:
  Post-process `.inp`/`.odb` outputs and assemble run data.
- `A3_ContinuumPP.py`:
  Alternative post-processing path for continuum-plot-oriented outputs.
- `B*.sh` scripts:
  HPC submission, scratch-space workflow, transfer, zipping/unzipping, and clean-up utilities.
- `run-local.ps1`:
  Example local execution wrapper calling A1 then A2 post-processing.

The A/HPC scripts are designed to be driven by command-line arguments (`sys.argv`) passed from job scripts, so most run configuration can be controlled from the shell scripts.

Important practical note:
- Update job parameters and paths before running (for example `LAT`, `nnx`, `DIS`, `pDir`, and cluster directory roots).
- `OldScriptVersions/` keeps historical script snapshots for traceability.

## Data Processing
Data processing and visualization notebooks are in `code/`. Shared reusable Python utilities are in `code/resources/`.

Typical data-processing notebooks include:
- `DataProcessing.ipynb`, `AK-DataProcessing.ipynb`
- `InputsOutputs.ipynb`, `AK-InputsOutputs.ipynb`
- `SIMresults.ipynb`, `StiffnessMatrix.ipynb`, `ContinuumPlots.ipynb`
- `Sampling.ipynb`, `FunctionApproximation.ipynb`
- `Tokenization.ipynb`

Core utility modules in `code/resources/`:
- `calculations.py`: reading/cleaning simulation outputs and UT/FT property calculations.
- `lattices.py`: lattice geometry, connectivity, stiffness/compliance, isotropy metrics.
- `MLdata.py`: dataset assembly, filtering/outlier handling, split/save/load helpers.
- `utilities.py`: filename/convention utilities and file-processing helpers.
- `imports.py`: shared import bundle used by multiple notebooks.

## Machine Learning
Machine-learning workflows are primarily notebook-driven in `code/`, with model/training logic centralized in `code/resources/`.

Main ML notebooks include:
- `ML-StressStrain.ipynb`, `AK-ML-StressStrain.ipynb`
- `ML-DisorderDistribution.ipynb`
- `DimensionalityReduction.ipynb`, `Optimization.ipynb`, `ValConvPlots.ipynb`

Core ML code modules:
- `MLmodels.py`: model definitions (MLP, GNN, transformer/sequence blocks, autoencoder, GPR extensions, BoTorch optimizer wrapper).
- `MLfunc.py`: training loops, prediction helpers, losses, metrics, plotting, and hyperparameter optimization utilities.
- `MLdata.py`: data interface (`DATA` class), preprocessing and split workflows for ML pipelines.
- `tokenization.py`: output-informed embedding/tokenization utilities for graph/data representations.

The Python requirements include both classical ML and deep learning stacks (for example `scikit-learn`, `torch`, `torch-geometric`, `tensorflow`, `gpytorch`, `botorch`, and `optuna`). GPU acceleration is optional but recommended for heavier training workflows.
