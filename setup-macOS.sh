#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
CONDA_ROOT=${CONDA_ROOT:-"$HOME/miniforge3"}
CONDA_ENV=${CONDA_ENV:-nf-phd}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

usage() {
    echo "Usage: $0 [-remove]"
}

REMOVE=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        -remove) REMOVE=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; usage; exit 2 ;;
    esac
    shift
done

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "ERROR: Conda was not found at $CONDA_ROOT."
    echo "Install Miniforge there or set CONDA_ROOT to the correct location."
    exit 2
fi

# shellcheck disable=SC1091
. "$CONDA_ROOT/etc/profile.d/conda.sh"

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"
}

if [ "$REMOVE" = true ]; then
    if [ -z "$CONDA_ENV" ] || [ "$CONDA_ENV" = base ]; then
        echo "ERROR: Refusing to remove an empty or base Conda environment."
        exit 2
    fi
    if env_exists; then
        if [ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ]; then
            conda deactivate
        fi
        echo "Removing Conda environment: $CONDA_ENV"
        conda env remove -y -n "$CONDA_ENV"
        echo "Environment removed: $CONDA_ENV"
    else
        echo "Conda environment does not exist: $CONDA_ENV"
    fi
    exit 0
fi

if env_exists; then
    echo "Refreshing existing Conda environment: $CONDA_ENV"
else
    echo "Creating Conda environment: $CONDA_ENV (Python $PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
fi

conda activate "$CONDA_ENV"
unset PIP_NO_INDEX

python -m pip install --upgrade pip setuptools wheel

# Keep NumPy 1.x because the repository verifies np.trapz and several of its
# scientific/ML dependencies still expect the NumPy 1.x ABI.
python -m pip install --upgrade \
    "numpy>=1.26,<2" scipy matplotlib pandas numexpr bottleneck sympy openpyxl \
    ipywidgets jupyter scikit-learn networkx optuna torchbnn torchinfo

# macOS PyTorch wheels are CPU builds on PyPI and do not use the Windows/Linux
# "+cpu" suffix or the download.pytorch.org CPU index.
if [ "$(uname -m)" = "x86_64" ]; then
    python -m pip install --upgrade \
        "torch==2.2.2" "torchvision==0.17.2" "torchaudio==2.2.2"
else
    python -m pip install --upgrade \
        "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
fi

python -m pip install --upgrade \
    torch-geometric gpytorch "botorch>=0.10.0" deepxde

if [ "$(uname -m)" = "x86_64" ]; then
    python -m pip install --upgrade "tensorflow==2.16.2"
else
    python -m pip install --upgrade tensorflow
fi

python -m pip install --upgrade -e "$REPO_ROOT"
python -m pip check

python - <<'PY'
import os
import tempfile
from importlib.metadata import version

os.chdir(tempfile.gettempdir())

import numpy as np
import torch
from resources.lattices import Geometry
from resources.MLdata import DATA
from resources.MLmodels import MODEL, MLP

print("Environment check")
print("  numpy:", np.__version__, "trapz:", hasattr(np, "trapz"))
print("  torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("  tensorflow:", version("tensorflow"))
print("  repository imports: OK")
PY

echo "Environment setup complete: $CONDA_ENV"
echo "Activate it with: conda activate $CONDA_ENV"
