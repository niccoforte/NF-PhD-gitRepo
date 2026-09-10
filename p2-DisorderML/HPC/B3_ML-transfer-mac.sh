#!/bin/bash
# macOS Bash 3.2-compatible saved-run download. No remote writes or deletions.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOCAL_ROOT=${LOCAL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)/data}
REMOTE=${REMOTE:-apocrita}
REMOTE_ROOT=${REMOTE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$HOME/.ssh/codex-apocrita}
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi
case "$#" in
    1) RUN_PATH=$1 ;;
    4) RUN_PATH=$1/$2/$3/$4 ;;
    5)
        case "$3/$4" in
            HPO/*|hpo/*|*/HPO|*/hpo) RUN_PATH=$1/$2/$3/$4/$5 ;;
            *) echo "Five arguments require an HPO marker." >&2; exit 2 ;;
        esac ;;
    *) echo "Usage: bash $0 [--dry-run] RUN_PATH" >&2
       echo "   or: bash $0 [--dry-run] TASK OUTPUT_KIND MODEL RUN_NAME" >&2
       echo "Example: bash $0 MULTI Dual Transformer dual-MULTI-test-260907" >&2
       exit 2 ;;
esac
# Restrict paths to archive-relative, shell-safe components; never accept traversal.
case "$RUN_PATH" in
    ''|/*|*/|*//*|*[!a-zA-Z0-9_./-]*) echo "Invalid archive-relative run path." >&2; exit 2 ;;
esac
case "/$RUN_PATH/" in
    */../*|*/./*) echo "Path traversal is not allowed." >&2; exit 2 ;;
esac
case "$REMOTE_ROOT" in
    /*) ;;
    *) echo "REMOTE_ROOT must be absolute." >&2; exit 2 ;;
esac
SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=15"
if [ -S "$SSH_CONTROL_PATH" ]; then
    SSH_COMMAND="$SSH_COMMAND -S \"$SSH_CONTROL_PATH\""
fi
LOCAL_PATH=${LOCAL_ROOT%/}/$RUN_PATH
REMOTE_PATH=${REMOTE_ROOT%/}/$RUN_PATH
echo "Remote: $REMOTE:$REMOTE_PATH/"
echo "Local:  $LOCAL_PATH/"
OPTIONS=(-a --checksum --partial --timeout=60 --itemize-changes)
# A dry run may create empty destination directories, but never copies files.
mkdir -p "$LOCAL_PATH"
if [ "$DRY_RUN" = true ]; then
    OPTIONS+=(--dry-run)
fi
# Trailing slashes avoid nested duplicate run folders on repeated downloads.
rsync "${OPTIONS[@]}" -e "$SSH_COMMAND" "$REMOTE:$REMOTE_PATH/" "$LOCAL_PATH/"
echo "Transfer check complete (dry-run=$DRY_RUN)."
