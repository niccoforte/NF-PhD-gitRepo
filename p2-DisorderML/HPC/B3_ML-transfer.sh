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

RUN_PATH=${RUN_PATH:-}

if [ -z "$RUN_PATH" ] && [ "$#" -ge 4 ]; then
    TASK=$1
    DATA_DESCRIPTOR=$2
    MODEL_NAME=$3
    RUN_DESCRIPTOR=$4
    RUN_PATH=$TASK/$DATA_DESCRIPTOR/$MODEL_NAME/$RUN_DESCRIPTOR
elif [ -z "$RUN_PATH" ] && [ "$#" -ge 1 ]; then
    RUN_PATH=$1
fi

if [ -z "$RUN_PATH" ]; then
    TASK=${TASK:-}
    DATA_DESCRIPTOR=${DATA_DESCRIPTOR:-}
    MODEL_NAME=${MODEL_NAME:-}
    RUN_DESCRIPTOR=${RUN_DESCRIPTOR:-}

    prompt_default TASK "Task, e.g. UT, FT, MULTI"
    prompt_default DATA_DESCRIPTOR "Data descriptor"
    prompt_default MODEL_NAME "Model, e.g. MLP, GAT, GCN, TR"
    prompt_default RUN_DESCRIPTOR "Run descriptor, e.g. HPC-ut-gat-260513-143012, HPO/my-study, or all"

    if [ "$RUN_DESCRIPTOR" = "all" ]; then
        RUN_PATH=$TASK/$DATA_DESCRIPTOR/$MODEL_NAME
    else
        RUN_PATH=$TASK/$DATA_DESCRIPTOR/$MODEL_NAME/$RUN_DESCRIPTOR
    fi
fi

prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"

RUN_PATH=${RUN_PATH#/}
REMOTE_PATH=$REMOTE_ROOT/$RUN_PATH
LOCAL_PATH=$LOCAL_ROOT/$RUN_PATH
LOCAL_PARENT=$(dirname "$LOCAL_PATH")

mkdir -p "$LOCAL_PARENT"

echo "Remote: $REMOTE:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
scp -r "$REMOTE:$REMOTE_PATH" "$LOCAL_PARENT/"

echo "Transfer complete."
echo "Saved under: $LOCAL_PATH"
