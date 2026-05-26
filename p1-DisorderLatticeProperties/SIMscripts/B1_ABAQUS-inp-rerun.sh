#!/bin/bash

# Run existing Abaqus .inp files in the submit/current directory, then
# post-process the resulting .odb files in place.
# Copy this script, the A2_*postProcess.py scripts, and resources/ into the
# scratch directory containing the .inp files, then submit from that directory.

#SBATCH -n 8
#SBATCH -p compute
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=ABAQUS_INP
#SBATCH -o %x.o%j
#SBATCH -L abaqus:12

set -euo pipefail

LAT=${LAT:-FCC}
nnx=${nnx:-20}
unitCellSize=${unitCellSize:-10}
mode=${mode:-both}
material=${material:-ti}
rD=${rD:-0.2}
DIS=${DIS:-disNodes}
fac=${fac:-0.2}
distribution=${distribution:-lhs_uniform}
target=${target:-all}
initial=${initial:-1}
nJobs=${nJobs:-1}
CPUs=${CPUs:-${SLURM_NTASKS:-8}}
Fout=${Fout:-20}
Hout=${Hout:-200}
pDir=${pDir:-None}

# Set this when the directory contains unrelated input files.
# Example: INP_GLOB='*Fracture*kagome*20disNodes*20[1-6].inp'
INP_GLOB=${INP_GLOB:-*.inp}

POSTPROCESS=${POSTPROCESS:-true}
RUN_IN=${RUN_IN:-true}
RUN_OUT=${RUN_OUT:-true}
RUN_FIELD=${RUN_FIELD:-true}

SCRIPT_DIR=${SCRIPT_DIR:-$(pwd)}
WORK_DIR=${WORK_DIR:-$(pwd)}

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

require_file() {
    local path=$1
    if [ ! -f "$path" ]; then
        /bin/echo "ERROR: Required file not found: $path"
        exit 2
    fi
}

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting on: $(date), in $(pwd)"
/bin/echo "Job ID: ${SLURM_JOB_ID:-manual}"
/bin/echo "Work directory: $WORK_DIR"
/bin/echo "Input glob: $INP_GLOB"
/bin/echo "CPUs: $CPUs"

module load abaqus/2024
module load intel

cd "$WORK_DIR"

if [ ! -d "resources" ]; then
    /bin/echo "ERROR: resources/ directory not found in $WORK_DIR"
    /bin/echo "Copy resources/ here, or run from a directory where Abaqus Python can import resources."
    exit 2
fi

if [ "$POSTPROCESS" = true ]; then
    [ "$RUN_OUT" = true ] && require_file "$SCRIPT_DIR/A2_OUTpostProcess.py"
    [ "$RUN_IN" = true ] && require_file "$SCRIPT_DIR/A2_INpostProcess.py"
    [ "$RUN_FIELD" = true ] && require_file "$SCRIPT_DIR/A2_FieldOUTpostProcess.py"
fi

shopt -s nullglob
inp_files=( $INP_GLOB )
shopt -u nullglob

if [ "${#inp_files[@]}" -eq 0 ]; then
    /bin/echo "ERROR: No .inp files matched: $INP_GLOB"
    exit 1
fi

/bin/echo "Found ${#inp_files[@]} input files."

for inp_file in "${inp_files[@]}"; do
    run_job_name=${inp_file%.inp}

    /bin/echo "----------------------------------------------------"
    /bin/echo "Running Abaqus job: $run_job_name"
    /bin/echo "Input file: $inp_file"
    /bin/echo "Time: $(date)"
    /bin/echo "----------------------------------------------------"

    abaqus job="$run_job_name" input="$inp_file" cpus="$CPUs" mp_mode=THREADS interactive

    /bin/echo "Finished job: $run_job_name at $(date)"
done

if [ "$POSTPROCESS" = true ]; then
    /bin/echo "Processing outputs at: $(date)"
    mkdir -p transfer

    if [ "$RUN_OUT" = true ]; then
        abaqus cae noGUI="$SCRIPT_DIR/A2_OUTpostProcess.py" -- "${ABAQUS_ARGS[@]}"
    fi

    if [ "$RUN_IN" = true ]; then
        abaqus cae noGUI="$SCRIPT_DIR/A2_INpostProcess.py" -- "${ABAQUS_ARGS[@]}"
    fi

    if [ "$RUN_FIELD" = true ]; then
        abaqus cae noGUI="$SCRIPT_DIR/A2_FieldOUTpostProcess.py" -- "${ABAQUS_ARGS[@]}"
    fi
fi

/bin/echo "Finished at: $(date)"
/bin/echo "Outputs are in: $WORK_DIR"
/bin/echo "Transfer files are in: $WORK_DIR/transfer"
