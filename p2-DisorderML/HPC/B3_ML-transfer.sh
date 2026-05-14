#!/bin/bash

set -euo pipefail

REMOTE=${REMOTE:-exy053@login.hpc.qmul.ac.uk}
REMOTE_ROOT=${REMOTE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}

# Usage examples:
#   bash B3_ML-transfer.sh "UT/MLP/HPC-ut-mlp-260514-142233"
#   bash B3_ML-transfer.sh UT MLP HPC-ut-mlp-260514-142233
#   bash B3_ML-transfer.sh UT MLP HPO HPC-MLP_full_hOpt
#   bash B3_ML-transfer.sh UT HPO HPC-GNN_full_hOpt GAT

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

prompt_required() {
    local var_name=$1
    local prompt=$2
    local value=${!var_name:-}

    while [ -z "$value" ]; do
        read -r -p "$prompt: " value
    done

    printf -v "$var_name" "%s" "$value"
}

RUN_PATH=${RUN_PATH:-}

if [ -z "$RUN_PATH" ] && [ "$#" -eq 4 ]; then
    TASK=$1
    MODEL_NAME=$2
    HPO_MARKER=$3
    RUN_DESCRIPTOR=$4
    if [ "${MODEL_NAME^^}" = "HPO" ]; then
        RUN_PATH=$TASK/HPO/$HPO_MARKER/$RUN_DESCRIPTOR
    elif [ "${HPO_MARKER^^}" = "HPO" ]; then
        RUN_PATH=$TASK/$MODEL_NAME/HPO/$RUN_DESCRIPTOR
    else
        /bin/echo "ERROR: Four-argument form must be either:"
        /bin/echo "  TASK MODEL HPO RUN_DESCRIPTOR"
        /bin/echo "  TASK HPO RUN_DESCRIPTOR MODEL"
        exit 2
    fi
elif [ -z "$RUN_PATH" ] && [ "$#" -eq 3 ]; then
    TASK=$1
    MODEL_NAME=$2
    RUN_DESCRIPTOR=$3
    RUN_PATH=$TASK/$MODEL_NAME/$RUN_DESCRIPTOR
elif [ -z "$RUN_PATH" ] && [ "$#" -eq 1 ]; then
    RUN_PATH=$1
elif [ -z "$RUN_PATH" ] && [ "$#" -gt 0 ]; then
    /bin/echo "ERROR: Use one raw RUN_PATH, regular form TASK MODEL RUN_DESCRIPTOR, or HPO forms:"
    /bin/echo "  TASK MODEL HPO RUN_DESCRIPTOR"
    /bin/echo "  TASK HPO RUN_DESCRIPTOR MODEL"
    exit 2
fi

if [ -z "$RUN_PATH" ]; then
    TASK=${TASK:-}
    MODEL_NAME=${MODEL_NAME:-}
    RUN_DESCRIPTOR=${RUN_DESCRIPTOR:-}
    RUN_KIND=${RUN_KIND:-}

    prompt_required TASK "Task, e.g. UT, FT, MULTI"
    prompt_default RUN_KIND "Run kind: regular, model-hpo, compare-hpo" "regular"
    prompt_required MODEL_NAME "Model, e.g. MLP, GAT, GCN, Transformer"
    prompt_required RUN_DESCRIPTOR "Run descriptor, e.g. HPC-ut-gat-260513-143012, HPC-MLP_full_hOpt, or all"

    case "${RUN_KIND,,}" in
        regular)
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/$MODEL_NAME
            else
                RUN_PATH=$TASK/$MODEL_NAME/$RUN_DESCRIPTOR
            fi
            ;;
        model-hpo)
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/$MODEL_NAME/HPO
            else
                RUN_PATH=$TASK/$MODEL_NAME/HPO/$RUN_DESCRIPTOR
            fi
            ;;
        compare-hpo)
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/HPO
            else
                RUN_PATH=$TASK/HPO/$RUN_DESCRIPTOR/$MODEL_NAME
            fi
            ;;
        *)
            /bin/echo "ERROR: RUN_KIND must be regular, model-hpo, or compare-hpo."
            exit 2
            ;;
    esac
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
