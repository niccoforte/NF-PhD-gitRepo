#!/bin/bash

# SLURM post-processing pass for archived Abaqus runs.
# Submit this from a lattice archive directory or from the parent disorder level, e.g.
#   cd /data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data/disNodes/0.2
#   sbatch /data/home/$USER/00-PhD-gitRepo/p1-DisorderLatticeProperties/SIMscripts/B2_ABAQUS-PPscratch-new.sh

#SBATCH -n 1
#SBATCH -p compute
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=B2_PPscratch
#SBATCH -o %x.o%j
#SBATCH -L abaqus:5

set -euo pipefail

HPC_USER=${HPC_USER:-${USER:-exy053}}

# ^^^ EDIT / OVERRIDE FOR EACH POST-PROCESSING PASS ^^^
# These are the only Abaqus arguments that affect the recursive A2 post-processing pass.
unitCellSize=${unitCellSize:-10}
mode=${mode:-both}
distribution=${distribution:-lhs_uniform}
Hout=${Hout:-200}

# Run from the current directory by default. Override ROOT_DIR if needed.
ROOT_DIR=${ROOT_DIR:-$(pwd)}
ROOT_DIR=${ROOT_DIR%/}
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
SIM_CODE_DIR=${SIM_CODE_DIR:-$REPO_ROOT/p1-DisorderLatticeProperties/SIMscripts}
FIELD_SCRIPT=${FIELD_SCRIPT:-$SIM_CODE_DIR/A2_FieldOUTpostProcess.py}
OUT_SCRIPT=${OUT_SCRIPT:-$SIM_CODE_DIR/A2_OUTpostProcess.py}
IN_SCRIPT=${IN_SCRIPT:-$SIM_CODE_DIR/A2_INpostProcess.py}

export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

has_odb_files() {
    local directory=$1
    compgen -G "$directory/*.odb" >/dev/null
}

run_postprocess() {
    local pDir=$1
    local job_name
    job_name=$(basename "$(dirname "$pDir")")

    local abaqus_args=(
        # LAT and nnx placeholders. A2 scripts infer both from each file name.
        "post"
        "0"
        "$unitCellSize"
        "$mode"
        "post"
        "0"
        "per"
        "0"
        "$distribution"
        "all"
        "1"
        "1"
        "${SLURM_NTASKS:-1}"
        "20"
        "$Hout"
        "$pDir"
    )

    /bin/echo
    /bin/echo "[$job_name] Post-processing: $pDir"
    mkdir -p "$pDir/transfer"

    if [ "$DRY_RUN" = true ]; then
        /bin/echo "DRY_RUN: abaqus cae noGUI=\"$OUT_SCRIPT\" -- ${abaqus_args[*]}"
        /bin/echo "DRY_RUN: abaqus cae noGUI=\"$IN_SCRIPT\" -- ${abaqus_args[*]}"
        /bin/echo "DRY_RUN: abaqus cae noGUI=\"$FIELD_SCRIPT\" -- ${abaqus_args[*]}"
        return 0
    fi

    abaqus cae noGUI="$OUT_SCRIPT" -- "${abaqus_args[@]}"
    abaqus cae noGUI="$IN_SCRIPT" -- "${abaqus_args[@]}"
    abaqus cae noGUI="$FIELD_SCRIPT" -- "${abaqus_args[@]}"
}

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting on: $(date), in $(pwd)"
/bin/echo "Job ID: ${SLURM_JOB_ID:-manual}"
/bin/echo "Root directory: $ROOT_DIR"
/bin/echo "Repo root: $REPO_ROOT"
/bin/echo "Output script: $OUT_SCRIPT"
/bin/echo "Input script: $IN_SCRIPT"
/bin/echo "Field script: $FIELD_SCRIPT"

module load abaqus/2024
module load intel

if [ ! -d "$ROOT_DIR" ]; then
    /bin/echo "ERROR: ROOT_DIR does not exist: $ROOT_DIR"
    exit 2
fi

if [ ! -f "$OUT_SCRIPT" ]; then
    /bin/echo "ERROR: OUT_SCRIPT does not exist: $OUT_SCRIPT"
    exit 3
fi

if [ ! -f "$IN_SCRIPT" ]; then
    /bin/echo "ERROR: IN_SCRIPT does not exist: $IN_SCRIPT"
    exit 3
fi

if [ ! -f "$FIELD_SCRIPT" ]; then
    /bin/echo "ERROR: FIELD_SCRIPT does not exist: $FIELD_SCRIPT"
    exit 3
fi

processed=0
skipped=0
found=0

while IFS= read -r -d '' zip_dir; do
    found=$((found + 1))

    if ! has_odb_files "$zip_dir"; then
        /bin/echo "Skipping $zip_dir: no .odb files."
        skipped=$((skipped + 1))
        continue
    fi

    run_postprocess "$zip_dir"
    processed=$((processed + 1))
done < <(find "$ROOT_DIR" -type d \( -name transfer -o -name __pycache__ \) -prune -o -type d -name zip -print0 | sort -z)

/bin/echo
/bin/echo "Completed at: $(date)"
/bin/echo "Discovered zip directories: $found"
/bin/echo "Processed zip directories: $processed"
/bin/echo "Skipped directories: $skipped"
