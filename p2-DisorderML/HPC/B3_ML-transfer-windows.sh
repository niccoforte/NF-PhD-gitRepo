#!/bin/bash

# Windows/Git Bash transfer entry point. On macOS use B3_ML-transfer-mac.sh.

set -euo pipefail

REMOTE=${REMOTE:-exy053@login.hpc.qmul.ac.uk}
REMOTE_ROOT=${REMOTE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}

# Usage examples:
#   bash B3_ML-transfer-windows.sh "UT/Curve/MLP/ut-mlp-260514-142233"
#   bash B3_ML-transfer-windows.sh UT Curve MLP ut-mlp-260514-142233
#   bash B3_ML-transfer-windows.sh UT FieldToCurve Transformer field-to-curve-UT-full
#   bash B3_ML-transfer-windows.sh MULTI Dual Transformer dual-MULTI-TR-full
#   bash B3_ML-transfer-windows.sh UT Curve MLP HPO MLP_full_hOpt
#   bash B3_ML-transfer-windows.sh UT Field HPO cross_model_hOpt
#   bash B3_ML-transfer-windows.sh UT Field HPO cross_model_hOpt GAT  # optional single model subfolder

if [ ! -d "Z:/" ] && [ -z "${LOCAL_ROOT:-}" ]; then
    echo "Z: is unavailable; set LOCAL_ROOT to an external data directory." >&2
    exit 2
fi
LOCAL_ROOT=${LOCAL_ROOT:-Z:/p2}

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

if [ -z "$RUN_PATH" ] && [ "$#" -eq 5 ]; then
    TASK=$1
    OUTPUT_KIND=$2
    MODEL_NAME=$3
    HPO_MARKER=$4
    RUN_DESCRIPTOR=$5
    if [ "${MODEL_NAME^^}" = "HPO" ]; then
        RUN_PATH=$TASK/$OUTPUT_KIND/HPO/$HPO_MARKER/$RUN_DESCRIPTOR
    elif [ "${HPO_MARKER^^}" = "HPO" ]; then
        RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME/HPO/$RUN_DESCRIPTOR
    else
        /bin/echo "ERROR: Five-argument HPO form must be either:"
        /bin/echo "  TASK OUTPUT_KIND MODEL HPO RUN_DESCRIPTOR"
        /bin/echo "  TASK OUTPUT_KIND HPO RUN_DESCRIPTOR MODEL"
        exit 2
    fi
elif [ -z "$RUN_PATH" ] && [ "$#" -eq 4 ]; then
    TASK=$1
    OUTPUT_KIND=$2
    MODEL_NAME=$3
    RUN_DESCRIPTOR=$4
    if [ "${MODEL_NAME^^}" = "HPO" ]; then
        RUN_PATH=$TASK/$OUTPUT_KIND/HPO/$RUN_DESCRIPTOR
    else
        RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME/$RUN_DESCRIPTOR
    fi
elif [ -z "$RUN_PATH" ] && [ "$#" -eq 1 ]; then
    RUN_PATH=$1
elif [ -z "$RUN_PATH" ] && [ "$#" -gt 0 ]; then
    /bin/echo "ERROR: Use one raw RUN_PATH, regular form TASK OUTPUT_KIND MODEL RUN_DESCRIPTOR, or HPO forms:"
    /bin/echo "  TASK OUTPUT_KIND MODEL HPO RUN_DESCRIPTOR"
    /bin/echo "  TASK OUTPUT_KIND HPO RUN_DESCRIPTOR"
    /bin/echo "  TASK OUTPUT_KIND HPO RUN_DESCRIPTOR MODEL"
    exit 2
fi

if [ -z "$RUN_PATH" ]; then
    TASK=${TASK:-}
    OUTPUT_KIND=${OUTPUT_KIND:-}
    MODEL_NAME=${MODEL_NAME:-}
    RUN_DESCRIPTOR=${RUN_DESCRIPTOR:-}
    RUN_KIND=${RUN_KIND:-}

    prompt_required TASK "Task, e.g. UT, FT, MULTI"
    prompt_required OUTPUT_KIND "Output kind, e.g. Curve, Field, FieldToCurve, or Dual"
    prompt_default RUN_KIND "Run kind: regular, model-hpo, compare-hpo" "regular"

    case "${RUN_KIND,,}" in
        regular)
            prompt_required MODEL_NAME "Model, e.g. MLP, GAT, GCN, Transformer"
            prompt_required RUN_DESCRIPTOR "Run descriptor, e.g. ut-gat-260513-143012 or all"
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME
            else
                RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME/$RUN_DESCRIPTOR
            fi
            ;;
        model-hpo)
            prompt_required MODEL_NAME "Model, e.g. MLP, GAT, GCN, Transformer"
            prompt_required RUN_DESCRIPTOR "Model-specific HPO descriptor, e.g. MLP_full_hOpt or all"
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME/HPO
            else
                RUN_PATH=$TASK/$OUTPUT_KIND/$MODEL_NAME/HPO/$RUN_DESCRIPTOR
            fi
            ;;
        compare-hpo)
            prompt_required RUN_DESCRIPTOR "Cross-model HPO descriptor, e.g. cross_model_hOpt or all"
            if [ "$RUN_DESCRIPTOR" = "all" ]; then
                RUN_PATH=$TASK/$OUTPUT_KIND/HPO
            else
                RUN_PATH=$TASK/$OUTPUT_KIND/HPO/$RUN_DESCRIPTOR
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
case "$RUN_PATH" in
    ''|*/|*//*|*[!a-zA-Z0-9_./-]*) echo "Invalid archive-relative run path." >&2; exit 2 ;;
esac
case "/$RUN_PATH/" in
    */../*|*/./*) echo "Path traversal is not allowed." >&2; exit 2 ;;
esac
case "$LOCAL_ROOT" in
    /*|[a-zA-Z]:/*) ;;
    *) echo "LOCAL_ROOT must be an absolute external data directory." >&2; exit 2 ;;
esac
case "/${LOCAL_ROOT%/}/" in
    */../*|*/./*) echo "LOCAL_ROOT must not contain traversal components." >&2; exit 2 ;;
esac
command -v git >/dev/null || { echo "Git is required to verify the destination." >&2; exit 2; }
EXISTING_PARENT=${LOCAL_ROOT%/}/$RUN_PATH
while [ ! -d "$EXISTING_PARENT" ]; do EXISTING_PARENT=$(dirname "$EXISTING_PARENT"); done
if [ "$(git -C "$EXISTING_PARENT" rev-parse --is-inside-work-tree 2>/dev/null || true)" = true ]; then
    echo "LOCAL_ROOT must be outside all Git checkouts." >&2; exit 2
fi

REMOTE_PATH=$REMOTE_ROOT/$RUN_PATH
LOCAL_PATH=$LOCAL_ROOT/$RUN_PATH
LOCAL_PARENT=$(dirname "$LOCAL_PATH")

mkdir -p "$LOCAL_PARENT"

echo "Remote: $REMOTE:$REMOTE_PATH"
echo "Local:  $LOCAL_PATH"
scp -r "$REMOTE:$REMOTE_PATH" "$LOCAL_PARENT/"

echo "Transfer complete."
echo "Saved under: $LOCAL_PATH"
