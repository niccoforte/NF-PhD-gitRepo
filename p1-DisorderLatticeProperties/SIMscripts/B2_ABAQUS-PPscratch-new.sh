#!/bin/bash

# SLURM post-processing pass for archived Abaqus runs.
# Submit this from the lattice archive directory, e.g.
#   cd /data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data/disNodes/0.2/FCC
#   sbatch /data/home/$USER/00-PhD-gitRepo/p1-DisorderLatticeProperties/SIMscripts/B2_ABAQUS-PPscratch.sh

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
LAT=${LAT:-lat}
nnx=${nnx:-10}
unitCellSize=${unitCellSize:-10}
mode=${mode:-both}
material=${material:-ti}
rD=${rD:-0.2}
DIS=${DIS:-per}
fac=${fac:-0.0}
distribution=${distribution:-lhs_uniform}
target=${target:-all}
initial=${initial:-1}
nJobs=${nJobs:-1}
CPUs=${CPUs:-${SLURM_NTASKS:-8}}
Fout=${Fout:-20}
Hout=${Hout:-200}

# Run from the FCC directory by default. Override ROOT_DIR if needed.
ROOT_DIR=${ROOT_DIR:-$(pwd)}
SKIP_DIRS=${SKIP_DIRS:-16-ps}
DRY_RUN=${DRY_RUN:-false}

REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
SIM_CODE_DIR=${SIM_CODE_DIR:-$REPO_ROOT/p1-DisorderLatticeProperties/SIMscripts}
FIELD_SCRIPT=${FIELD_SCRIPT:-$SIM_CODE_DIR/A2_FieldOUTpostProcess.py}
OUT_SCRIPT=${OUT_SCRIPT:-$SIM_CODE_DIR/A2_OUTpostProcess.py}
IN_SCRIPT=${IN_SCRIPT:-$SIM_CODE_DIR/A2_INpostProcess.py}

export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

should_skip_dir() {
    local dirname=$1
    for skip in $SKIP_DIRS; do
        if [ "$dirname" = "$skip" ]; then
            return 0
        fi
    done
    return 1
}

has_odb_files() {
    local directory=$1
    compgen -G "$directory/*.odb" >/dev/null
}

run_postprocess() {
    local pDir=$1
    local job_name
    job_name=$(basename "$(dirname "$pDir")")

    local abaqus_args=(
        "$LAT"
        "$nnx"
        "$unitCellSize"
        "$mode"
        "$material"
        "$rD"
        "$DIS"
        "$fac"
        "$distribution"
        "$target"
        "$initial"
        "$nJobs"
        "$CPUs"
        "$Fout"
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

for job_dir in "$ROOT_DIR"/*; do
    [ -d "$job_dir" ] || continue
    job_name=$(basename "$job_dir")

    if should_skip_dir "$job_name"; then
        /bin/echo "Skipping requested directory: $job_name"
        skipped=$((skipped + 1))
        continue
    fi

    zip_dir="$job_dir/zip"
    if [ ! -d "$zip_dir" ]; then
        /bin/echo "Skipping $job_name: no zip directory."
        skipped=$((skipped + 1))
        continue
    fi

    if ! has_odb_files "$zip_dir"; then
        /bin/echo "Skipping $job_name: no .odb files in zip directory."
        skipped=$((skipped + 1))
        continue
    fi

    run_postprocess "$zip_dir"
    processed=$((processed + 1))
done

/bin/echo
/bin/echo "Completed at: $(date)"
/bin/echo "Processed zip directories: $processed"
/bin/echo "Skipped directories: $skipped"
