#!/bin/bash

set -euo pipefail

REMOTE=${REMOTE:-exy053@login.hpc.qmul.ac.uk}
REMOTE_ROOT=${REMOTE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data}

if [ -d "Z:/" ]; then
    LOCAL_ROOT=${LOCAL_ROOT:-Z:/p1/data/Ti}
else
    LOCAL_ROOT=${LOCAL_ROOT:-$(pwd)/p1-data-Ti}
fi

# Usage examples:
#   bash B3_ABAQUS-transfer.sh
#   bash B3_ABAQUS-transfer.sh both FCC Frequency
#   bash B3_ABAQUS-transfer.sh disNodes FCC Frequency 9919196
#   bash B3_ABAQUS-transfer.sh per FCC all
#   bash B3_ABAQUS-transfer.sh download-zip disNodes FCC Frequency 9919196
#   bash B3_ABAQUS-transfer.sh download-zip per FCC 9919196
#   bash B3_ABAQUS-transfer.sh upload-transfer FCC Frequency
#
# Interactive/default mode downloads per/0.0 and disNodes/<EXTRA>/0.2
# transfer contents into their separate local transfer directories.

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

local_transfer_dir() {
    local dis=$1
    local extra=$2
    local fac=$3
    local lat=$4

    if [ "$dis" = "per" ]; then
        printf "%s/per/%s/transfer" "$LOCAL_ROOT" "$lat"
    elif [ -n "$extra" ]; then
        printf "%s/%s/%s/%s/%s/transfer" "$LOCAL_ROOT" "$dis" "$extra" "$fac" "$lat"
    else
        printf "%s/%s/%s/%s/transfer" "$LOCAL_ROOT" "$dis" "$fac" "$lat"
    fi
}

local_zip_parent_dir() {
    local dis=$1
    local extra=$2
    local fac=$3
    local lat=$4
    local job_id=$5

    if [ "$dis" = "per" ]; then
        printf "%s/per/%s/%s" "$LOCAL_ROOT" "$lat" "$job_id"
    elif [ -n "$extra" ]; then
        printf "%s/%s/%s/%s/%s/%s" "$LOCAL_ROOT" "$dis" "$extra" "$fac" "$lat" "$job_id"
    else
        printf "%s/%s/%s/%s/%s" "$LOCAL_ROOT" "$dis" "$fac" "$lat" "$job_id"
    fi
}

remote_job_selector() {
    local job_id=$1

    if [ "$job_id" = "all" ]; then
        printf "[0-9]*"
    else
        printf "%s" "$job_id"
    fi
}

remote_case_path() {
    local dis=$1
    local extra=$2
    local fac=$3
    local lat=$4
    local job_id=$5
    local selector

    selector=$(remote_job_selector "$job_id")
    if [ "$dis" = "per" ]; then
        printf "%s/per/%s/%s/%s" "$REMOTE_ROOT" "$fac" "$lat" "$selector"
    elif [ -n "$extra" ]; then
        printf "%s/%s/%s/%s/%s/%s" "$REMOTE_ROOT" "$dis" "$extra" "$fac" "$lat" "$selector"
    else
        printf "%s/%s/%s/%s/%s" "$REMOTE_ROOT" "$dis" "$fac" "$lat" "$selector"
    fi
}

rsync_remote_glob() {
    local remote_glob=$1
    local dest=$2

    mkdir -p "$dest"
    /bin/echo "Remote: $REMOTE:$remote_glob"
    /bin/echo "Local:  $dest"

    set +e
    if command -v rsync >/dev/null 2>&1; then
        rsync -av "$REMOTE:$remote_glob" "$dest/"
    else
        /bin/echo "rsync not found locally; falling back to scp."
        scp "$REMOTE:$remote_glob" "$dest/"
    fi
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        /bin/echo "WARNING: transfer command returned status $rc for: $remote_glob"
        /bin/echo "This usually means the remote path did not match any files, or the transfer was interrupted."
    fi

    return 0
}

download_transfer_case() {
    local dis=$1
    local extra=$2
    local fac=$3
    local lat=$4
    local job_id=$5
    local dest=$6
    local base

    base=$(remote_case_path "$dis" "$extra" "$fac" "$lat" "$job_id")
    rsync_remote_glob "$base/zip/transfer/*" "$dest"
}

download_zip_case() {
    local dis=$1
    local extra=$2
    local fac=$3
    local lat=$4
    local job_id=$5
    local parent
    local base

    if [ "$job_id" = "all" ]; then
        /bin/echo "ERROR: download-zip requires a specific JOB_ID, not all."
        exit 2
    fi

    parent=$(local_zip_parent_dir "$dis" "$extra" "$fac" "$lat" "$job_id")
    base=$(remote_case_path "$dis" "$extra" "$fac" "$lat" "$job_id")
    mkdir -p "$parent"

    /bin/echo "Remote: $REMOTE:$base/zip"
    /bin/echo "Local:  $parent/zip"
    if command -v rsync >/dev/null 2>&1; then
        rsync -av "$REMOTE:$base/zip" "$parent/"
    else
        /bin/echo "rsync not found locally; falling back to scp."
        scp -r "$REMOTE:$base/zip" "$parent/"
    fi
}

download_both_default() {
    local lat=$1
    local extra=$2
    local job_id=$3
    local dis_dest
    local per_dest

    dis_dest=$(local_transfer_dir "disNodes" "$extra" "0.2" "$lat")
    per_dest=$(local_transfer_dir "per" "" "0.0" "$lat")

    mkdir -p "$dis_dest" "$per_dest"
    download_transfer_case "per" "" "0.0" "$lat" "$job_id" "$per_dest"
    download_transfer_case "disNodes" "$extra" "0.2" "$lat" "$job_id" "$dis_dest"

    /bin/echo "Default transfer complete."
    /bin/echo "disNodes files saved under: $dis_dest"
    /bin/echo "per files saved under:      $per_dest"
}

upload_transfer() {
    local lat=$1
    local extra=$2
    local local_dir=${3:-Z:/p1/sims/Ti/FrequencyDisorder/$lat/transfer}
    local remote_dir=${4:-$REMOTE_ROOT/disNodes/$extra/0.2/$lat/local/zip/transfer}

    if [ ! -d "$local_dir" ]; then
        /bin/echo "ERROR: Local transfer directory not found: $local_dir"
        exit 2
    fi

    ssh "$REMOTE" "mkdir -p '$remote_dir'"
    if command -v rsync >/dev/null 2>&1; then
        rsync -av "$local_dir"/ "$REMOTE:$remote_dir"/
    else
        /bin/echo "rsync not found locally; falling back to scp."
        scp -r "$local_dir"/* "$REMOTE:$remote_dir"/
    fi
    /bin/echo "Upload complete."
    /bin/echo "Remote: $REMOTE:$remote_dir"
}

extract_tgz() {
    local tgz=$1
    local dest=${2:-$(pwd)}

    if [ ! -f "$tgz" ]; then
        /bin/echo "ERROR: tgz file not found: $tgz"
        exit 2
    fi

    mkdir -p "$dest"
    tar -xvzf "$tgz" -C "$dest"
    /bin/echo "Extracted $tgz into: $dest"
}

MODE=${MODE:-}
LAT=${LAT:-}
EXTRA=${EXTRA:-}
DIS=${DIS:-}
FAC=${FAC:-}
JOB_ID=${JOB_ID:-all}

if [ "$#" -gt 0 ]; then
    case "$1" in
        both|per|disNodes|download-zip|upload-transfer|extract-tgz)
            MODE=$1
            shift
            ;;
        *)
            MODE=${MODE:-both}
            ;;
    esac
fi

case "${MODE:-}" in
    "")
        prompt_default MODE "Mode: both, per, disNodes, download-zip, upload-transfer, extract-tgz" "both"
        ;;
esac

case "$MODE" in
    both)
        LAT=${1:-$LAT}
        EXTRA=${2:-$EXTRA}
        JOB_ID=${3:-$JOB_ID}
        prompt_required LAT "LAT"
        prompt_default EXTRA "Extra Path Spec, e.g. Frequency, Target-xs, validation; leave empty for none" ""
        prompt_default JOB_ID "Job ID, or all" "$JOB_ID"
        prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"
        download_both_default "$LAT" "$EXTRA" "$JOB_ID"
        ;;
    per)
        LAT=${1:-$LAT}
        JOB_ID=${2:-$JOB_ID}
        prompt_required LAT "LAT"
        prompt_default JOB_ID "Job ID, or all" "$JOB_ID"
        prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"
        DEST=$(local_transfer_dir "per" "" "0.0" "$LAT")
        download_transfer_case "per" "" "0.0" "$LAT" "$JOB_ID" "$DEST"
        ;;
    disNodes)
        LAT=${1:-$LAT}
        EXTRA=${2:-$EXTRA}
        JOB_ID=${3:-$JOB_ID}
        FAC=${FAC:-0.2}
        prompt_required LAT "LAT"
        prompt_default EXTRA "Extra Path Spec, e.g. Frequency, Target-xs, validation; leave empty for none" ""
        prompt_default FAC "fac" "$FAC"
        prompt_default JOB_ID "Job ID, or all" "$JOB_ID"
        prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"
        DEST=$(local_transfer_dir "disNodes" "$EXTRA" "$FAC" "$LAT")
        download_transfer_case "disNodes" "$EXTRA" "$FAC" "$LAT" "$JOB_ID" "$DEST"
        ;;
    download-zip)
        DIS=${1:-$DIS}
        LAT=${2:-$LAT}
        prompt_default DIS "DIS, e.g. per or disNodes" "disNodes"
        prompt_required LAT "LAT"
        if [ "$DIS" = "per" ]; then
            EXTRA=""
            FAC=${FAC:-0.0}
            JOB_ID=${3:-$JOB_ID}
        else
            EXTRA=${3:-$EXTRA}
            JOB_ID=${4:-$JOB_ID}
            prompt_default EXTRA "Extra Path Spec, e.g. Frequency, Target-xs, validation; leave empty for none" ""
            FAC=${FAC:-0.2}
        fi
        prompt_default FAC "fac" "$FAC"
        prompt_required JOB_ID "Job ID"
        prompt_default LOCAL_ROOT "Local root" "$LOCAL_ROOT"
        download_zip_case "$DIS" "$EXTRA" "$FAC" "$LAT" "$JOB_ID"
        ;;
    upload-transfer)
        LAT=${1:-$LAT}
        EXTRA=${2:-$EXTRA}
        LOCAL_DIR=${3:-${LOCAL_DIR:-}}
        REMOTE_DIR=${4:-${REMOTE_DIR:-}}
        prompt_required LAT "LAT"
        prompt_required EXTRA "Extra Path Spec, e.g. Frequency"
        prompt_default LOCAL_DIR "Local transfer directory" "Z:/p1/sims/Ti/FrequencyDisorder/$LAT/transfer"
        prompt_default REMOTE_DIR "Remote transfer directory" "$REMOTE_ROOT/disNodes/$EXTRA/0.2/$LAT/local/zip/transfer"
        upload_transfer "$LAT" "$EXTRA" "$LOCAL_DIR" "$REMOTE_DIR"
        ;;
    extract-tgz)
        TGZ=${1:-${TGZ:-}}
        DEST=${2:-${DEST:-$(pwd)}}
        prompt_required TGZ "tgz file"
        prompt_default DEST "Destination directory" "$DEST"
        extract_tgz "$TGZ" "$DEST"
        ;;
    *)
        /bin/echo "ERROR: Unknown mode: $MODE"
        /bin/echo "Use: both, per, disNodes, download-zip, upload-transfer, extract-tgz"
        exit 2
        ;;
esac
