#!/bin/bash
#SBATCH -J ML_GPU_NEW
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# For gpu partition runs
##SBATCH -p gpu
##SBATCH -t 240:0:0
# For andrena partitioin runs
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -t 240:0:0
##SBATCH --exclusive   # for 4 GPUs, full A100 node.

# Optional GPU type constraints:
##SBATCH --constraint=ampere  # A100
##SBATCH --constraint=hopper  # H100/H200


set -euo pipefail

HPC_USER=${HPC_USER:-${USER:-exy053}}

# ^^^ RENAME / EDIT FOR EACH ML RUN ^^^
#
# Intended submit workflow:
#   cd /data/home/exy053/p2/UT/MLP
#   sbatch B1_ML-new.sh
#
# Keep a copy or symlink of this B1 script in the submit directory above.
# The Python run script and resources folder are still copied from REPO_ROOT,
# not from the submit directory.
#
# Other submit examples:
#   sbatch B1_ML-new.sh A-HPC-1_run.py --epochs 3 --nsims 64
#   ML_SCRIPT=A-HPC-1_run.py RUN_LABEL=trial-ut-mlp sbatch B1_ML-new.sh -- --epochs 3
#
# ML_SCRIPT may be either:
#   - a filename inside p2-DisorderML/HPC, e.g. A-HPC-1_run.py
#   - a repo-relative path, e.g. p2-DisorderML/HPC/A-HPC-1_run.py
#   - an absolute path on the cluster.
REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
ML_CODE_DIR=${ML_CODE_DIR:-$REPO_ROOT/p2-DisorderML/HPC}
ML_SCRIPT=${ML_SCRIPT:-A-HPC-1_run.py}
RUN_LABEL_PROVIDED=false
if [ -n "${RUN_LABEL:-}" ]; then
    RUN_LABEL_PROVIDED=true
fi
RUN_LABEL=${RUN_LABEL:-}

# Conda/Mamba environment created beforehand on the cluster.
# Leave empty to use the base Miniforge environment after module load.
CONDA_ENV=${CONDA_ENV:-nf-ml-gpu}
CREATE_CONDA_ENV=${CREATE_CONDA_ENV:-false}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}
ML_ENV_EXTRA_PIP=${ML_ENV_EXTRA_PIP:-}

# Data and archive locations. MLdata.py appends "MLdata/..." to DATA(path=...),
# so DATA_ROOT must be the parent directory containing MLdata, not MLdata itself.
# With this default, Python should use DATA(path=os.environ["ML_DATA_ROOT"], ...).
# Optional automation: if Python writes JSON to os.environ["ML_RUN_METADATA"],
# this script uses task/model_type/run_label to choose the final archive path.
DATA_ROOT=${DATA_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
PATH_EXTRA_PROVIDED=false
if [ -n "${PATH_EXTRA:-}" ]; then
    PATH_EXTRA_PROVIDED=true
fi
PATH_EXTRA=${PATH_EXTRA:-}
SUBMIT_P2_ROOT=${SUBMIT_P2_ROOT:-/data/home/$HPC_USER/p2}

zip=false
delete_scratch=true
copy_data_to_scratch=false

SCRATCH_DIR=/gpfs/scratch/$HPC_USER/$SLURM_JOB_ID
ZIP_DIR=$SCRATCH_DIR/zip
RESULT_DIR=$SCRATCH_DIR/results
RUN_METADATA_FILE=$SCRATCH_DIR/run_metadata.json

if [ "$#" -gt 0 ]; then
    if [ "$1" = "--" ]; then
        shift
    else
        ML_SCRIPT=$1
        shift
    fi
fi
ML_ARGS=("$@")

if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL=$(basename "$ML_SCRIPT")
    RUN_LABEL=${RUN_LABEL%.*}
fi

if [ "$PATH_EXTRA_PROVIDED" = false ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    case "$SLURM_SUBMIT_DIR" in
        "$SUBMIT_P2_ROOT"/*)
            PATH_EXTRA=${SLURM_SUBMIT_DIR#"$SUBMIT_P2_ROOT"/}
            ;;
    esac
fi

ARCHIVE_DIR=$ARCHIVE_ROOT/${PATH_EXTRA:+$PATH_EXTRA/}$RUN_LABEL/$SLURM_JOB_ID

if [[ "$ML_SCRIPT" = /* ]]; then
    SCRIPT_SRC=$ML_SCRIPT
elif [[ "$ML_SCRIPT" == */* ]]; then
    SCRIPT_SRC=$REPO_ROOT/$ML_SCRIPT
else
    SCRIPT_SRC=$ML_CODE_DIR/$ML_SCRIPT
fi
SCRIPT_LOCAL=$SCRATCH_DIR/$(basename "$ML_SCRIPT")

sync_if_exists() {
    local src=$1
    local dst=$2
    if [ -e "$src" ]; then
        mkdir -p "$dst"
        rsync -av "$src" "$dst/"
    fi
}

init_conda_shell() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base=$(conda info --base)
        if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            . "$conda_base/etc/profile.d/conda.sh"
        fi
    fi
}

conda_env_exists() {
    local pkg_manager=$1
    "$pkg_manager" env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"
}

create_conda_env_if_requested() {
    if [ -z "$CONDA_ENV" ]; then
        return
    fi

    local pkg_manager=mamba
    if ! command -v "$pkg_manager" >/dev/null 2>&1; then
        pkg_manager=conda
    fi

    if ! command -v "$pkg_manager" >/dev/null 2>&1; then
        /bin/echo "ERROR: neither mamba nor conda is available after module load."
        exit 3
    fi

    if conda_env_exists "$pkg_manager"; then
        /bin/echo "Using existing conda environment: $CONDA_ENV"
        return
    fi

    if [ "$CREATE_CONDA_ENV" != true ]; then
        /bin/echo "ERROR: Conda environment '$CONDA_ENV' does not exist."
        /bin/echo "Available environments:"
        "$pkg_manager" env list
        /bin/echo ""
        /bin/echo "Create it once with:"
        /bin/echo "  cd $SLURM_SUBMIT_DIR"
        /bin/echo "  CREATE_CONDA_ENV=true sbatch B1_ML-new.sh"
        /bin/echo ""
        /bin/echo "Or submit using an existing environment:"
        /bin/echo "  CONDA_ENV=<env_name> sbatch B1_ML-new.sh"
        exit 3
    fi

    /bin/echo "Creating conda environment: $CONDA_ENV"
    "$pkg_manager" create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip

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
}

apply_run_metadata() {
    if [ ! -f "$RUN_METADATA_FILE" ] || ! command -v python >/dev/null 2>&1; then
        return
    fi

    local meta_assignments
    meta_assignments=$(python - "$RUN_METADATA_FILE" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    meta = json.load(f)

def get(*keys):
    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""

task = get("task", "target", "mechMode", "mech_mode").upper()
model_type = get("model_type", "model", "typ").upper()
path_extra = get("path_extra", "archive_subdir")
if not path_extra and task and model_type:
    path_extra = f"{task}/{model_type}"

values = {
    "META_PATH_EXTRA": path_extra,
    "META_RUN_LABEL": get("run_label", "experiment", "name"),
    "META_ARCHIVE_ROOT": get("archive_root"),
}

for key, value in values.items():
    if value:
        print(f"{key}={shlex.quote(value)}")
PY
)

    if [ -n "$meta_assignments" ]; then
        eval "$meta_assignments"
    fi

    if [ "$PATH_EXTRA_PROVIDED" = false ] && [ -n "${META_PATH_EXTRA:-}" ]; then
        PATH_EXTRA=$META_PATH_EXTRA
    fi
    if [ "$RUN_LABEL_PROVIDED" = false ] && [ -n "${META_RUN_LABEL:-}" ]; then
        RUN_LABEL=$META_RUN_LABEL
    fi
    if [ -n "${META_ARCHIVE_ROOT:-}" ]; then
        ARCHIVE_ROOT=$META_ARCHIVE_ROOT
    fi

    ARCHIVE_DIR=$ARCHIVE_ROOT/${PATH_EXTRA:+$PATH_EXTRA/}$RUN_LABEL/$SLURM_JOB_ID
    /bin/echo "Metadata archive path: $ARCHIVE_DIR"
}

sync_run_outputs() {
    mkdir -p "$ARCHIVE_DIR/zip"

    sync_if_exists "$SCRATCH_DIR/models" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/HPO" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/results" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/outputs" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/figures" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/plots" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/wandb" "$ARCHIVE_DIR/zip"
    sync_if_exists "$SCRATCH_DIR/lightning_logs" "$ARCHIVE_DIR/zip"

    find "$SCRATCH_DIR" -maxdepth 1 -type f \
        \( -name "*.csv" -o -name "*.json" -o -name "*.log" -o -name "*.mdl" \
        -o -name "*.npy" -o -name "*.npz" -o -name "*.pkl" -o -name "*.pt" \
        -o -name "*.pth" -o -name "*.sqlite" -o -name "*.db" -o -name "*.txt" \) \
        -exec rsync -av {} "$ARCHIVE_DIR/zip/" \;

    if [ -f "$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID" ]; then
        rsync -av "$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID" "$ARCHIVE_DIR/zip/"
    fi
}

finish() {
    local status=$?
    set +e

    apply_run_metadata
    /bin/echo "Archiving outputs at: $(date)"
    sync_run_outputs

    if [ "$zip" = true ]; then
        mkdir -p "$ZIP_DIR"
        tar -czf "$SCRATCH_DIR/C2_zip-$RUN_LABEL-$SLURM_JOB_ID.tgz" -C "$SCRATCH_DIR" zip
        rsync -av "$SCRATCH_DIR/C2_zip-$RUN_LABEL-$SLURM_JOB_ID.tgz" "$ARCHIVE_DIR/"
    fi

    if [ "$status" -eq 0 ] && [ "$delete_scratch" = true ]; then
        if [[ "$SCRATCH_DIR" == /gpfs/scratch/"$HPC_USER"/"$SLURM_JOB_ID" ]]; then
            rm -rf "$SCRATCH_DIR"
        else
            /bin/echo "Scratch path safety check failed, not deleting: $SCRATCH_DIR"
        fi
    else
        /bin/echo "Scratch kept for debugging: $SCRATCH_DIR"
    fi

    /bin/echo "Job finished with status $status at: $(date)"
    /bin/echo "Data saved in: $ARCHIVE_DIR"
    exit "$status"
}
trap finish EXIT

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting on: $(date), in $(pwd)"
/bin/echo "Job ID: $SLURM_JOB_ID"
/bin/echo "Submit directory: $SLURM_SUBMIT_DIR"
/bin/echo "Script: $ML_SCRIPT"
/bin/echo "Run label: $RUN_LABEL"
/bin/echo "Archive path extra: ${PATH_EXTRA:-<none>}"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$ZIP_DIR"
mkdir -p "$RESULT_DIR"

# Load required modules.
module load miniforge
init_conda_shell
create_conda_env_if_requested
if [ -n "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH=$SCRATCH_DIR:${PYTHONPATH:-}
export ML_DATA_ROOT=$DATA_ROOT
export ML_RESULTS_DIR=$RESULT_DIR
export ML_RUN_METADATA=$RUN_METADATA_FILE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-$SLURM_NTASKS}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

/bin/echo "Python: $(which python)"
python -V
/bin/echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-set by Slurm inside GPU jobs}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    /bin/echo "nvidia-smi is not available in this environment."
fi

if [ ! -e "$SCRIPT_SRC" ]; then
    /bin/echo "ERROR: ML script not found: $SCRIPT_SRC"
    /bin/echo "Create it in p2-DisorderML/HPC or submit with: sbatch B1_ML-new.sh A-HPC-1_run.py"
    exit 2
fi

# Copy only the shared framework and the one run-specific script to scratch.
rsync -av "$REPO_ROOT/resources/" "$SCRATCH_DIR/resources/"
rsync -av "$SCRIPT_SRC" "$SCRIPT_LOCAL"

if [ "$copy_data_to_scratch" = true ]; then
    mkdir -p "$SCRATCH_DIR/data"
    rsync -av "$DATA_ROOT/" "$SCRATCH_DIR/data/"
    export ML_DATA_ROOT=$SCRATCH_DIR/data
fi

cd "$SCRATCH_DIR"
/bin/echo "Working in scratch directory: $(pwd)"
/bin/echo "Running Python ML script at: $(date)"
python -u "$SCRIPT_LOCAL" "${ML_ARGS[@]}"

/bin/echo "ML run completed at: $(date)"
/bin/echo "Processing and archiving outputs..."
