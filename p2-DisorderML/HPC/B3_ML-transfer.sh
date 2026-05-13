#!/bin/bash

set -euo pipefail

REMOTE=${REMOTE:-exy053@login.hpc.qmul.ac.uk}
REMOTE_ROOT=${REMOTE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}

if [ -d "Z:/" ]; then
    LOCAL_ROOT=${LOCAL_ROOT:-Z:/p2}
else
    LOCAL_ROOT=${LOCAL_ROOT:-$(pwd)/p2-MLruns}
fi

prompt_default() {
    local var_name=$1
    local prompt=$2
    local default=${3:-}
    local value=${!var_name:-}

    if [ -n "$value" ]; then
        return
    fi

    if [ -n "$default" ]; then
        read -r -p "$prompt [$default]: " value
        value=${value:-$default}
    else
        read -r -p "$prompt: " value
    fi

    printf -v "$var_name" "%s" "$value"
}

PATH_EXTRA=${PATH_EXTRA:-${1:-}}
RUN_LABEL=${RUN_LABEL:-${2:-}}
JOB_ID=${JOB_ID:-${3:-}}

prompt_default PATH_EXTRA "Archive path extra, e.g. UT/GAT"
prompt_default RUN_LABEL "Run label, e.g. ut-gat-1h"
prompt_default JOB_ID "Slurm job ID, or all"
prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"

REMOTE_BASE=$REMOTE_ROOT/${PATH_EXTRA#/}/$RUN_LABEL
LOCAL_BASE=$LOCAL_ROOT/${PATH_EXTRA#/}/$RUN_LABEL

mkdir -p "$LOCAL_BASE"

transfer_job() {
    local job_id=$1
    local remote_zip=$REMOTE_BASE/$job_id/zip
    local local_job=$LOCAL_BASE/$job_id

    mkdir -p "$local_job"
    echo "Remote: $REMOTE:$remote_zip"
    echo "Local:  $local_job"
    (
        cd "$local_job"
        scp -r "$REMOTE:$remote_zip" .
    )
}

if [ "$JOB_ID" = "all" ]; then
    echo "Finding jobs under: $REMOTE:$REMOTE_BASE"
    JOB_IDS=$(ssh "$REMOTE" "for d in '$REMOTE_BASE'/*; do [ -d \"\$d\" ] && basename \"\$d\"; done")
    if [ -z "$JOB_IDS" ]; then
        echo "No jobs found under: $REMOTE_BASE"
        exit 1
    fi
    for job_id in $JOB_IDS; do
        transfer_job "$job_id"
    done
else
    transfer_job "$JOB_ID"
fi

echo "Transfer complete."
echo "Saved under: $LOCAL_BASE"
