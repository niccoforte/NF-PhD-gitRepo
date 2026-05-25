#!/bin/bash

# Default compute-partition Abaqus run.
#SBATCH -n 8
#SBATCH -p compute
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=JobNameOG
#SBATCH -o %x.o%j
#SBATCH -L abaqus:12

# For parallel CPU partition runs.
##SBATCH -N 4
##SBATCH -n 192
##SBATCH -p parallel
##SBATCH -t 24:0:0
##SBATCH --exclusive
##SBATCH --mem=0

# Optional resource constraints from older runs.
##SBATCH --gres=gpu:2
##SBATCH -p andrena
##SBATCH -A pilot_andrena
##SBATCH -p highmem
##SBATCH --constraint=avx512

set -euo pipefail

HPC_USER=${HPC_USER:-${USER:-exy053}}

# ^^^ RENAME / EDIT FOR EACH ABAQUS RUN ^^^
#
# These defaults preserve the original B1_ABAQUS-new.sh behavior.
# You can still edit them here, or override them at submission time, e.g.
#   LAT=FCC DIS=disNodes PATH_EXTRA=Frequency fac=0.2 sbatch B1_ABAQUS-new_v2.sh
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
pDir=${pDir:-None}

PATH_EXTRA=${PATH_EXTRA:-}

zip=${zip:-false}
delete_scratch=${delete_scratch:-true}

# Location defaults.
REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
SIM_CODE_DIR=${SIM_CODE_DIR:-$REPO_ROOT/p1-DisorderLatticeProperties/SIMscripts}
RESOURCES_SRC=${RESOURCES_SRC:-$REPO_ROOT/resources}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data}

# Allow basic syntax checks outside Slurm without crashing on unset variables.
SLURM_JOB_ID=${SLURM_JOB_ID:-manual-$(date +%y%m%d-%H%M%S)}
SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
SLURM_JOB_NAME=${SLURM_JOB_NAME:-ABAQUS_MANUAL}

SCRATCH_DIR=${SCRATCH_DIR:-/gpfs/scratch/$HPC_USER/$SLURM_JOB_ID}
TRANSFER_DIR=$SCRATCH_DIR/transfer
RESOURCES_DIR=$SCRATCH_DIR/resources
ZIP_DIR=$SCRATCH_DIR/zip
ZIP_TRANSFER_DIR=$ZIP_DIR/transfer
ZIP_RESOURCES_DIR=$ZIP_DIR/resources
SLURM_LOG=$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID

if [ -n "$PATH_EXTRA" ]; then
    ARCHIVE_PARENT=$ARCHIVE_ROOT/$DIS/$PATH_EXTRA/$fac/$LAT
else
    ARCHIVE_PARENT=$ARCHIVE_ROOT/$DIS/$fac/$LAT
fi
ARCHIVE_RUN_DIR=$ARCHIVE_PARENT/$SLURM_JOB_ID
ARCHIVE_ZIP_DIR=$ARCHIVE_RUN_DIR/zip

ABAQUS_ARGS=(
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

copy_glob_to_dir() {
    local pattern=$1
    local dest=$2
    local matches=()

    mkdir -p "$dest"
    shopt -s nullglob
    matches=( $pattern )
    shopt -u nullglob

    if [ "${#matches[@]}" -gt 0 ]; then
        rsync -av "${matches[@]}" "$dest/"
    else
        /bin/echo "No files found for pattern: $pattern"
    fi
}

sync_dir_contents_if_exists() {
    local src=$1
    local dest=$2

    mkdir -p "$dest"
    if [ -d "$src" ]; then
        rsync -av "$src"/ "$dest"/
    else
        /bin/echo "Directory not found, skipping: $src"
    fi
}

write_run_config() {
    cat > "$SCRATCH_DIR/run_config.txt" <<EOF
Job ID: $SLURM_JOB_ID
Job name: $SLURM_JOB_NAME
Submit directory: $SLURM_SUBMIT_DIR
Scratch directory: $SCRATCH_DIR
Archive directory: $ARCHIVE_RUN_DIR

LAT=$LAT
nnx=$nnx
unitCellSize=$unitCellSize
mode=$mode
material=$material
rD=$rD
DIS=$DIS
fac=$fac
distribution=$distribution
target=$target
initial=$initial
nJobs=$nJobs
CPUs=$CPUs
Fout=$Fout
Hout=$Hout
pDir=$pDir
PATH_EXTRA=$PATH_EXTRA
zip=$zip
delete_scratch=$delete_scratch
EOF
}

prepare_scratch() {
    mkdir -p "$TRANSFER_DIR" "$RESOURCES_DIR" "$ZIP_TRANSFER_DIR" "$ZIP_RESOURCES_DIR"
}

copy_inputs() {
    copy_glob_to_dir "$SIM_CODE_DIR/A-HPC-*" "$SCRATCH_DIR"
    copy_glob_to_dir "$SLURM_SUBMIT_DIR/B*" "$SCRATCH_DIR"
    sync_dir_contents_if_exists "$RESOURCES_SRC" "$RESOURCES_DIR"
}

stage_zip() {
    if [ ! -d "$SCRATCH_DIR" ]; then
        /bin/echo "Scratch directory not found, cannot stage zip: $SCRATCH_DIR"
        return 0
    fi

    mkdir -p "$ZIP_DIR" "$ZIP_TRANSFER_DIR" "$ZIP_RESOURCES_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/A*" "$ZIP_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/B*" "$ZIP_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/abaqus*" "$ZIP_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/*.odb" "$ZIP_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/*.inp" "$ZIP_DIR"
    copy_glob_to_dir "$SCRATCH_DIR/run_config.txt" "$ZIP_DIR"
    sync_dir_contents_if_exists "$RESOURCES_DIR" "$ZIP_RESOURCES_DIR"
    sync_dir_contents_if_exists "$TRANSFER_DIR" "$ZIP_TRANSFER_DIR"

    if [ -f "$SLURM_LOG" ]; then
        rsync -av "$SLURM_LOG" "$SCRATCH_DIR/"
        rsync -av "$SLURM_LOG" "$ZIP_DIR/"
    else
        /bin/echo "Slurm log not found yet, skipping: $SLURM_LOG"
    fi
}

sync_archive() {
    mkdir -p "$ARCHIVE_ZIP_DIR"
    if [ -d "$ZIP_DIR" ]; then
        rsync -av "$ZIP_DIR"/ "$ARCHIVE_ZIP_DIR"/
    else
        /bin/echo "Zip staging directory not found, skipping archive sync: $ZIP_DIR"
    fi
}

compress_if_requested() {
    if [ "$zip" != true ]; then
        return 0
    fi

    if [ -d "$TRANSFER_DIR" ]; then
        tar -czf "$SCRATCH_DIR/C1_transfer-$LAT-$DIS-$SLURM_JOB_ID.tgz" -C "$SCRATCH_DIR" transfer
        rsync -av "$SCRATCH_DIR/C1_transfer-$LAT-$DIS-$SLURM_JOB_ID.tgz" "$SLURM_SUBMIT_DIR/"
    fi

    if [ -d "$ZIP_DIR" ]; then
        tar -czf "$SCRATCH_DIR/C2_zip-$LAT-$DIS-$SLURM_JOB_ID.tgz" -C "$SCRATCH_DIR" zip
        mkdir -p "$ARCHIVE_PARENT"
        rsync -av "$SCRATCH_DIR/C2_zip-$LAT-$DIS-$SLURM_JOB_ID.tgz" "$ARCHIVE_PARENT/"
    fi
}

finish() {
    local status=$?
    set +e

    /bin/echo "Archiving outputs at: $(date)"
    stage_zip
    sync_archive
    compress_if_requested

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
    /bin/echo "Data saved in: $ARCHIVE_RUN_DIR"
    exit "$status"
}
trap finish EXIT

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting on: $(date), in $(pwd)"
/bin/echo "Job ID: $SLURM_JOB_ID"
/bin/echo "Submit directory: $SLURM_SUBMIT_DIR"
/bin/echo "Archive directory: $ARCHIVE_RUN_DIR"

# Load required modules.
module load abaqus/2024
module load intel

prepare_scratch
copy_inputs
write_run_config

cd "$SCRATCH_DIR"
/bin/echo "Working in directory: $(pwd)"

abaqus cae noGUI=A-HPC-1_FractureToughness-Ductility.py -- "${ABAQUS_ARGS[@]}"

/bin/echo "Simulation completed at: $(date)"
/bin/echo "Processing outputs..."

abaqus cae noGUI=A-HPC-2_OUTpostProcess.py -- "${ABAQUS_ARGS[@]}"
abaqus cae noGUI=A-HPC-2_INpostProcess.py -- "${ABAQUS_ARGS[@]}"

/bin/echo "Inputs and outputs collected."
/bin/echo "Simulation files staged in: $ZIP_DIR"
/bin/echo "Job completed at: $(date)"
