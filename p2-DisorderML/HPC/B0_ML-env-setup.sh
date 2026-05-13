#!/bin/bash

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-nf-ml-gpu}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}
ML_ENV_EXTRA_PIP=${ML_ENV_EXTRA_PIP:-}
UPDATE_EXISTING=${UPDATE_EXISTING:-false}

module load miniforge

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

PKG_MANAGER=mamba
if ! command -v "$PKG_MANAGER" >/dev/null 2>&1; then
    PKG_MANAGER=conda
fi

if ! command -v "$PKG_MANAGER" >/dev/null 2>&1; then
    echo "ERROR: neither mamba nor conda is available after module load."
    exit 3
fi

env_exists=false
if "$PKG_MANAGER" env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    env_exists=true
fi

if [ "$env_exists" = true ] && [ "$UPDATE_EXISTING" != true ]; then
    echo "Conda environment already exists: $CONDA_ENV"
    echo "Use UPDATE_EXISTING=true to reinstall/refresh packages."
    exit 0
fi

if [ "$env_exists" = false ]; then
    echo "Creating conda environment: $CONDA_ENV"
    "$PKG_MANAGER" create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip
else
    echo "Updating existing conda environment: $CONDA_ENV"
fi

conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    "numpy>=1.26,<2" scipy matplotlib pandas numexpr bottleneck sympy openpyxl ipywidgets \
    scikit-learn networkx optuna torchbnn
python -m pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX_URL"
python -m pip install torch-geometric torchinfo gpytorch "botorch>=0.10.0"

if [ -n "$ML_ENV_EXTRA_PIP" ]; then
    python -m pip install $ML_ENV_EXTRA_PIP
fi

python - <<'PY'
import numpy as np
import torch

print("Environment check")
print("  numpy:", np.__version__, "trapz:", hasattr(np, "trapz"))
print("  torch:", torch.__version__)
print("  torch cuda:", torch.version.cuda)
print("  cuda available:", torch.cuda.is_available())
PY

echo "Environment setup complete: $CONDA_ENV"
