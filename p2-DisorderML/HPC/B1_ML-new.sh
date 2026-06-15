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
##SBATCH -p sae
##SBATCH -A pilot_sae_gpu
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
#   cd /data/home/exy053/p2/UT/Curve/MLP
#   sbatch -J my-run-name B1_ML-new.sh
#
# Keep a copy or symlink of this B1 script in the submit directory above.
# The Python run script and resources folder are still copied from REPO_ROOT,
# not from the submit directory.
#
# Other submit examples. For ordinary model runs, -J becomes the default run descriptor.
# For HPO runs, -J also becomes the default HPO study/folder name.
#   sbatch -J curve-UT-GAT-full B1_ML-new.sh CurveOutputs/A0-HPC_Curve-test.py --task UT --model-type GAT
#   sbatch -J HPC-CurvePCAUT_fullHPO B1_ML-new.sh CurveOutputs/A0-HPC_Curve-CrossModelHPO.py --task UT --output-reduction pca --pca-components 16
#   sbatch -J HPC-CurvePCAFT_fullHPO B1_ML-new.sh CurveOutputs/A0-HPC_Curve-CrossModelHPO.py --task FT --output-reduction pca --pca-components 16
#   ML_SCRIPT=FieldOutputs/A0-HPC_Field-test.py sbatch -J field-UT-TR-full B1_ML-new.sh -- --task UT --model-type TR
#   ML_SCRIPT=FieldToCurve/A0-HPC_FieldToCurve-test.py sbatch -J field-to-curve-FT-full B1_ML-new.sh -- --task FT --output-reduction none
#   ML_SCRIPT=FieldToCurve/A0-HPC_FieldToCurve-CrossModelHPO.py sbatch -J Field2Curve-FT-HPO B1_ML-new.sh -- --task FT
#
# ML_SCRIPT may be either:
#   - a filename inside p2-DisorderML/HPC, e.g. B0-example.py
#   - a path inside p2-DisorderML/HPC, e.g. CurveOutputs/A0-HPC_Curve-test.py
#   - a repo-relative path, e.g. p2-DisorderML/HPC/CurveOutputs/A0-HPC_Curve-test.py
#   - an absolute path on the cluster.
REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
ML_CODE_DIR=${ML_CODE_DIR:-$REPO_ROOT/p2-DisorderML/HPC}
ML_SCRIPT=${ML_SCRIPT:-CurveOutputs/A0-HPC_Curve-test.py}
RUN_LABEL_PROVIDED=false
if [ -n "${RUN_LABEL:-}" ]; then
    RUN_LABEL_PROVIDED=true
fi
RUN_LABEL=${RUN_LABEL:-}

# Conda/Mamba environment created beforehand on the cluster.
# Create/update it with B0_ML-env-setup.sh when needed.
# Leave empty to use the base Miniforge environment after module load.
CONDA_ENV=${CONDA_ENV:-nf-ml-gpu}

# Data and archive locations. MLdata.py appends "MLdata/..." to DATA(path=...),
# so DATA_ROOT must be the parent directory containing MLdata, not MLdata itself.
# ARCHIVE_ROOT receives the framework run layout directly:
#   {UT|FT|MULTI}/{Curve|Field|FieldToCurve}/{Model}/{Run}
# With these defaults, Python should use DATA(path=os.environ["ML_DATA_ROOT"], ...).
DATA_ROOT=${DATA_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-${ARCHIVE_PARENT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}}

zip=false
delete_scratch=true

SCRATCH_DIR=/gpfs/scratch/$HPC_USER/$SLURM_JOB_ID
SCRATCH_RUN_ROOT=$SCRATCH_DIR/mlruns

if [ "$#" -gt 0 ]; then
    if [ "$1" = "--" ]; then
        shift
    else
        ML_SCRIPT=$1
        shift
    fi
fi
ML_ARGS=("$@")

if [ "$RUN_LABEL_PROVIDED" = false ]; then
    for ((i = 0; i < ${#ML_ARGS[@]}; i++)); do
        case "${ML_ARGS[$i]}" in
            --run-label=*)
                RUN_LABEL=${ML_ARGS[$i]#--run-label=}
                RUN_LABEL_PROVIDED=true
                ;;
            --run-label)
                if [ $((i + 1)) -lt ${#ML_ARGS[@]} ]; then
                    RUN_LABEL=${ML_ARGS[$((i + 1))]}
                    RUN_LABEL_PROVIDED=true
                fi
                ;;
        esac
    done
fi

if [ -z "$RUN_LABEL" ]; then
    RUN_LABEL=${SLURM_JOB_NAME:-$(basename "$ML_SCRIPT")}
    RUN_LABEL=${RUN_LABEL%.*}
fi

ARCHIVE_JOB_NAME=${SLURM_JOB_NAME:-$RUN_LABEL}
ARCHIVE_ROOT=${ARCHIVE_ROOT%/}
ML_JOB_NAME=${ML_JOB_NAME:-$ARCHIVE_JOB_NAME}

if [[ "$ML_SCRIPT" = /* ]]; then
    SCRIPT_SRC=$ML_SCRIPT
elif [[ "$ML_SCRIPT" == */* ]]; then
    if [ -e "$ML_CODE_DIR/$ML_SCRIPT" ]; then
        SCRIPT_SRC=$ML_CODE_DIR/$ML_SCRIPT
    else
        SCRIPT_SRC=$REPO_ROOT/$ML_SCRIPT
    fi
else
    SCRIPT_SRC=$ML_CODE_DIR/$ML_SCRIPT
fi
SCRIPT_LOCAL=$SCRATCH_DIR/$(basename "$ML_SCRIPT")

if [ "$RUN_LABEL_PROVIDED" = false ] && [ -e "$SCRIPT_SRC" ] && grep -q -- "--run-label" "$SCRIPT_SRC"; then
    ML_ARGS=(--run-label "$RUN_LABEL" "${ML_ARGS[@]}")
fi

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
    if [ -z "$CONDA_ENV" ]; then
        return 0
    fi

    conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"
}

sync_run_outputs() {
    mkdir -p "$ARCHIVE_ROOT"

    if [ -f "$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID" ]; then
        while IFS= read -r model_file; do
            rsync -av "$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID" "$(dirname "$model_file")/"
        done < <(find "$SCRATCH_RUN_ROOT" -type f -name "*.mdl" 2>/dev/null)
    fi

    if [ -d "$SCRATCH_RUN_ROOT" ]; then
        rsync -av "$SCRATCH_RUN_ROOT"/ "$ARCHIVE_ROOT"/
    fi
}

finish() {
    local status=$?
    set +e

    /bin/echo "Archiving outputs at: $(date)"
    sync_run_outputs

    if [ "$zip" = true ]; then
        tar -czf "$SCRATCH_DIR/C2_mlruns-$RUN_LABEL-$SLURM_JOB_ID.tgz" -C "$SCRATCH_DIR" mlruns
        rsync -av "$SCRATCH_DIR/C2_mlruns-$RUN_LABEL-$SLURM_JOB_ID.tgz" "$ARCHIVE_ROOT/"
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
    /bin/echo "Data saved under: $ARCHIVE_ROOT"
    exit "$status"
}
trap finish EXIT

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting on: $(date), in $(pwd)"
/bin/echo "Job ID: $SLURM_JOB_ID"
/bin/echo "Submit directory: $SLURM_SUBMIT_DIR"
/bin/echo "Script: $ML_SCRIPT"
/bin/echo "Run label: $RUN_LABEL"
/bin/echo "Archive root: $ARCHIVE_ROOT"
/bin/echo "ML job/study name: $ML_JOB_NAME"

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SCRATCH_RUN_ROOT"

# Load required modules.
module load miniforge
init_conda_shell
if [ -n "$CONDA_ENV" ]; then
    if ! conda_env_exists; then
        /bin/echo "ERROR: Conda environment '$CONDA_ENV' does not exist."
        /bin/echo "Create/update it with: bash $ML_CODE_DIR/B0_ML-env-setup.sh"
        exit 3
    fi
    conda activate "$CONDA_ENV"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH=$SCRATCH_DIR:${PYTHONPATH:-}
export ML_DATA_ROOT=$DATA_ROOT
export ML_RUN_ROOT=$SCRATCH_RUN_ROOT
export ML_ARCHIVE_ROOT=$ARCHIVE_ROOT
export ML_JOB_NAME=$ML_JOB_NAME
export ML_RUN_CONTEXT=HPC
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
    /bin/echo "Create it in p2-DisorderML/HPC or submit with: sbatch B1_ML-new.sh CurveOutputs/A0-HPC_Curve-test.py"
    exit 2
fi

# Copy only the shared framework and the one run-specific script to scratch.
rsync -av "$REPO_ROOT/resources/" "$SCRATCH_DIR/resources/"
rsync -av "$SCRIPT_SRC" "$SCRIPT_LOCAL"

cd "$SCRATCH_DIR"
/bin/echo "Working in scratch directory: $(pwd)"
/bin/echo "Running Python ML script at: $(date)"
python -u "$SCRIPT_LOCAL" "${ML_ARGS[@]}"

/bin/echo "ML run completed at: $(date)"
/bin/echo "Processing and archiving outputs..."
