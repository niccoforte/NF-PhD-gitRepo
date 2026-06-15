#!/bin/bash
#SBATCH -J ML_HPO_RESUME
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=7500M
#SBATCH --gres=gpu:1

# For gpu partition runs
##SBATCH -p sae
##SBATCH -A pilot_sae_gpu
##SBATCH -t 240:0:0

# For andrena partition runs
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -t 240:0:0
##SBATCH --exclusive   # for 4 GPUs, full A100 node.

# Optional GPU type constraints:
##SBATCH --constraint=ampere  # A100
##SBATCH --constraint=hopper  # H100/H200

set -euo pipefail

usage() {
    cat <<'EOF'
Resume an archived cross-model Optuna HPO run.

Usage:
  sbatch -J resume-name B2_ML-resumeHPO.sh [options] TASK/OUTPUT/HPO/STUDY [-- EXTRA_SCRIPT_ARGS...]

Examples:
  sbatch -J fUT-fHPO-resume B2_ML-resumeHPO.sh UT/Field/HPO/fUT-fHPO

  sbatch -J fUT-fHPO-resume B2_ML-resumeHPO.sh \
      --target-trials 80 \
      UT/Field/HPO/fUT-fHPO \
      -- --components U1,U2

Options:
  --archive-root PATH     Archive root. Default: /data/SEMS-TaoLab/Niccolo-Forte/p2
  --data-root PATH        DATA root passed through ML_DATA_ROOT. Default: archive root
  --repo-root PATH        Git repository root. Default: /data/home/$USER/00-PhD-gitRepo
  --script PATH           Override the HPO Python script if it cannot be inferred.
  --study-name NAME       Override the study folder/name. Default: basename of STUDY.
  --target-trials N       Target finished trials per model. Default: infer from log or sibling studies.
  --models LIST           Restrict models to resume, e.g. GCN,GAT,TR.
  --no-progress           Pass --no-progress to the HPO entry point.
  --dry-run               Print the detected resume plan without running Python training.
  -h, --help              Show this help.

Notes:
  The path may be absolute or relative to --archive-root. The script expects the
  active cross-model HPO layout:
      ARCHIVE_ROOT/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/HPO/{Study}/{Model}/full_study.db

  Path-only resume works for default HPO runs. If the original job used extra
  flags such as PCA settings, custom components, or reduced nsims, pass the same
  script flags after "--".
EOF
}

die() {
    /bin/echo "ERROR: $*" >&2
    exit 2
}

init_conda_shell() {
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base=$(conda info --base)
        if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            . "$conda_base/etc/profile.d/conda.sh"
        fi
    fi
}

conda_env_exists() {
    if [ -z "$CONDA_ENV" ]; then
        return 0
    fi

    conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"
}

find_source_log() {
    local search_dir=$1
    local log
    local logs=()

    while IFS= read -r log; do
        logs+=("$log")
    done < <(find "$search_dir" -maxdepth 1 -type f -name "*.o[0-9]*" 2>/dev/null | sort)

    if [ "${#logs[@]}" -eq 0 ]; then
        while IFS= read -r log; do
            logs+=("$log")
        done < <(find "$search_dir" -mindepth 2 -maxdepth 2 -type f -name "*.o[0-9]*" 2>/dev/null | sort)
    fi

    for log in "${logs[@]}"; do
        if grep -q "^Script:" "$log" && grep -q "^Archive root:" "$log"; then
            printf '%s\n' "$log"
            return 0
        fi
    done

    return 1
}

infer_script_from_output_kind() {
    local output_kind
    output_kind=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$output_kind" in
        curve)
            printf '%s\n' "CurveOutputs/A0-HPC_Curve-CrossModelHPO.py"
            ;;
        field)
            printf '%s\n' "FieldOutputs/A0-HPC_Field-CrossModelHPO.py"
            ;;
        fieldtocurve)
            printf '%s\n' "FieldToCurve/A0-HPC_FieldToCurve-CrossModelHPO.py"
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_script_source() {
    local script=$1
    if [[ "$script" = /* ]]; then
        printf '%s\n' "$script"
    elif [[ "$script" == */* ]]; then
        if [ -e "$ML_CODE_DIR/$script" ]; then
            printf '%s\n' "$ML_CODE_DIR/$script"
        else
            printf '%s\n' "$REPO_ROOT/$script"
        fi
    else
        printf '%s\n' "$ML_CODE_DIR/$script"
    fi
}

sync_resume_outputs() {
    if [ -d "${SCRATCH_RUN_ROOT:-}" ]; then
        mkdir -p "$ARCHIVE_ROOT"
        rsync -av "$SCRATCH_RUN_ROOT"/ "$ARCHIVE_ROOT"/
    fi
}

sync_resume_log() {
    local log_file
    if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_SUBMIT_DIR:-}" ] || [ -z "${SLURM_JOB_NAME:-}" ]; then
        return 0
    fi

    log_file="$SLURM_SUBMIT_DIR/$SLURM_JOB_NAME.o$SLURM_JOB_ID"
    if [ -f "$log_file" ] && [ -d "${HPO_ARCHIVE_DIR:-}" ]; then
        rsync -av "$log_file" "$HPO_ARCHIVE_DIR/"
    fi
}

finish() {
    local status=$?
    set +e

    /bin/echo "Archiving resumed HPO outputs at: $(date)"
    sync_resume_outputs
    /bin/echo "Resume job finished with status $status at: $(date)"
    /bin/echo "Data saved under: ${ARCHIVE_ROOT:-unknown}"
    sync_resume_log

    if [ "${delete_scratch:-true}" = true ] && [ "$status" -eq 0 ] && [ -n "${SCRATCH_DIR:-}" ]; then
        if [[ "$SCRATCH_DIR" == /gpfs/scratch/"$HPC_USER"/"$JOB_ID" ]]; then
            rm -rf "$SCRATCH_DIR"
        else
            /bin/echo "Scratch path safety check failed, not deleting: $SCRATCH_DIR"
        fi
    elif [ -n "${SCRATCH_DIR:-}" ]; then
        /bin/echo "Scratch kept for debugging: $SCRATCH_DIR"
    fi

    exit "$status"
}

HPC_USER=${HPC_USER:-${USER:-exy053}}
JOB_ID=${SLURM_JOB_ID:-manual-$$}

REPO_ROOT=${REPO_ROOT:-/data/home/$HPC_USER/00-PhD-gitRepo}
ML_CODE_DIR=${ML_CODE_DIR:-$REPO_ROOT/p2-DisorderML/HPC}
CONDA_ENV=${CONDA_ENV:-nf-ml-gpu}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
DATA_ROOT=${DATA_ROOT:-}

ARCHIVE_ARG=
ML_SCRIPT=
STUDY_NAME=
TARGET_TRIALS=
MODEL_FILTER=
dry_run=false
delete_scratch=true
no_progress=false
SCRIPT_EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --)
            shift
            SCRIPT_EXTRA_ARGS=("$@")
            break
            ;;
        --archive-root)
            [ "$#" -ge 2 ] || die "--archive-root requires a value."
            ARCHIVE_ROOT=$2
            shift 2
            ;;
        --archive-root=*)
            ARCHIVE_ROOT=${1#--archive-root=}
            shift
            ;;
        --data-root)
            [ "$#" -ge 2 ] || die "--data-root requires a value."
            DATA_ROOT=$2
            shift 2
            ;;
        --data-root=*)
            DATA_ROOT=${1#--data-root=}
            shift
            ;;
        --repo-root)
            [ "$#" -ge 2 ] || die "--repo-root requires a value."
            REPO_ROOT=$2
            ML_CODE_DIR=$REPO_ROOT/p2-DisorderML/HPC
            shift 2
            ;;
        --repo-root=*)
            REPO_ROOT=${1#--repo-root=}
            ML_CODE_DIR=$REPO_ROOT/p2-DisorderML/HPC
            shift
            ;;
        --script)
            [ "$#" -ge 2 ] || die "--script requires a value."
            ML_SCRIPT=$2
            shift 2
            ;;
        --script=*)
            ML_SCRIPT=${1#--script=}
            shift
            ;;
        --study-name)
            [ "$#" -ge 2 ] || die "--study-name requires a value."
            STUDY_NAME=$2
            shift 2
            ;;
        --study-name=*)
            STUDY_NAME=${1#--study-name=}
            shift
            ;;
        --target-trials)
            [ "$#" -ge 2 ] || die "--target-trials requires a value."
            TARGET_TRIALS=$2
            shift 2
            ;;
        --target-trials=*)
            TARGET_TRIALS=${1#--target-trials=}
            shift
            ;;
        --models)
            [ "$#" -ge 2 ] || die "--models requires a value."
            MODEL_FILTER=$2
            shift 2
            ;;
        --models=*)
            MODEL_FILTER=${1#--models=}
            shift
            ;;
        --no-progress)
            no_progress=true
            shift
            ;;
        --dry-run)
            dry_run=true
            delete_scratch=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "Unknown B2 option: $1. Put script-specific options after '--'."
            ;;
        *)
            if [ -z "$ARCHIVE_ARG" ]; then
                ARCHIVE_ARG=$1
            else
                die "Unexpected positional argument: $1"
            fi
            shift
            ;;
    esac
done

[ -n "$ARCHIVE_ARG" ] || { usage >&2; die "Provide a cross-model HPO archive path."; }

ARCHIVE_ROOT=${ARCHIVE_ROOT%/}
if [ -z "$DATA_ROOT" ]; then
    DATA_ROOT=$ARCHIVE_ROOT
fi

if [[ "$ARCHIVE_ARG" = /* ]]; then
    HPO_ARCHIVE_DIR=${ARCHIVE_ARG%/}
else
    HPO_ARCHIVE_DIR=$ARCHIVE_ROOT/${ARCHIVE_ARG#/}
fi

[ -d "$HPO_ARCHIVE_DIR" ] || die "Archive HPO path not found: $HPO_ARCHIVE_DIR"

case "$HPO_ARCHIVE_DIR" in
    "$ARCHIVE_ROOT"/*)
        HPO_REL=${HPO_ARCHIVE_DIR#"$ARCHIVE_ROOT"/}
        ;;
    *)
        die "Archive path must be inside ARCHIVE_ROOT. Got '$HPO_ARCHIVE_DIR' outside '$ARCHIVE_ROOT'."
        ;;
esac

IFS='/' read -r TASK OUTPUT_KIND HPO_TOKEN STUDY_FOLDER MAYBE_MODEL EXTRA_PART _ <<< "$HPO_REL"
if [ "$HPO_TOKEN" != "HPO" ] || [ -z "${STUDY_FOLDER:-}" ]; then
    die "Expected archive path like TASK/OUTPUT/HPO/STUDY, got: $HPO_REL"
fi

if [ -n "${MAYBE_MODEL:-}" ]; then
    if [ -f "$HPO_ARCHIVE_DIR/full_study.db" ] && [ -z "$MODEL_FILTER" ]; then
        MODEL_FILTER=$MAYBE_MODEL
        HPO_ARCHIVE_DIR=$(dirname "$HPO_ARCHIVE_DIR")
        HPO_REL=$(dirname "$HPO_REL")
    else
        die "Provide the HPO study root, not a nested path: $HPO_REL"
    fi
fi

if [ -n "${EXTRA_PART:-}" ]; then
    die "Archive path is too deep for a cross-model HPO root: $HPO_REL"
fi

if [ -z "$STUDY_NAME" ]; then
    STUDY_NAME=$STUDY_FOLDER
fi

SOURCE_LOG=$(find_source_log "$HPO_ARCHIVE_DIR" || true)
if [ -n "$SOURCE_LOG" ]; then
    LOG_SCRIPT=$(sed -n 's/^Script:[[:space:]]*//p' "$SOURCE_LOG" | head -n 1)
    LOG_ARGS=$(sed -n 's/^Python args:[[:space:]]*//p' "$SOURCE_LOG" | tail -n 1)
    LOG_ARCHIVE_ROOT=$(sed -n 's/^Archive root:[[:space:]]*//p' "$SOURCE_LOG" | head -n 1)
    LOG_BASE=$(basename "$SOURCE_LOG")
    ORIGINAL_JOB_ID=${LOG_BASE##*.o}
    ORIGINAL_JOB_NAME=${LOG_BASE%".o$ORIGINAL_JOB_ID"}
else
    LOG_SCRIPT=
    LOG_ARGS=
    LOG_ARCHIVE_ROOT=
    ORIGINAL_JOB_ID=
    ORIGINAL_JOB_NAME=
fi

if [ -z "$ML_SCRIPT" ]; then
    if [ -n "$LOG_SCRIPT" ]; then
        ML_SCRIPT=$LOG_SCRIPT
    else
        ML_SCRIPT=$(infer_script_from_output_kind "$OUTPUT_KIND") || die "Could not infer HPO script for output kind '$OUTPUT_KIND'. Use --script."
    fi
fi

SCRIPT_SRC=$(resolve_script_source "$ML_SCRIPT")
[ -e "$SCRIPT_SRC" ] || die "ML script not found: $SCRIPT_SRC"

SCRATCH_DIR=/gpfs/scratch/$HPC_USER/$JOB_ID
SCRATCH_RUN_ROOT=$SCRATCH_DIR/mlruns
SCRATCH_HPO_DIR=$SCRATCH_RUN_ROOT/$HPO_REL
PLAN_FILE=$SCRATCH_DIR/resume_plan.tsv
SCRIPT_LOCAL=$SCRATCH_DIR/$(basename "$ML_SCRIPT")

trap finish EXIT

/bin/echo "Running on host: $(hostname)"
/bin/echo "Starting HPO resume on: $(date), in $(pwd)"
/bin/echo "Resume job ID: ${SLURM_JOB_ID:-manual}"
/bin/echo "Source archive path: $HPO_ARCHIVE_DIR"
/bin/echo "Resume archive root: $ARCHIVE_ROOT"
/bin/echo "Relative HPO path: $HPO_REL"
/bin/echo "Task/output: $TASK / $OUTPUT_KIND"
/bin/echo "Study folder/name: $STUDY_NAME"
/bin/echo "Resume script: $ML_SCRIPT"
if [ -n "$SOURCE_LOG" ]; then
    /bin/echo "Detected source log: $SOURCE_LOG"
    /bin/echo "Original job: ${ORIGINAL_JOB_NAME:-unknown} ${ORIGINAL_JOB_ID:-unknown}"
    if [ -n "$LOG_ARGS" ]; then
        /bin/echo "Original Python args line: $LOG_ARGS"
    fi
    if [ -n "$LOG_ARCHIVE_ROOT" ] && [ "$LOG_ARCHIVE_ROOT" != "$ARCHIVE_ROOT" ]; then
        /bin/echo "Warning: source log archive root differs: $LOG_ARCHIVE_ROOT"
    fi
else
    /bin/echo "No source Slurm log found under the HPO archive path."
fi

mkdir -p "$SCRATCH_DIR"
mkdir -p "$SCRATCH_RUN_ROOT"

module load miniforge
init_conda_shell
if [ -n "$CONDA_ENV" ]; then
    if ! conda_env_exists; then
        die "Conda environment '$CONDA_ENV' does not exist. Create/update it with: bash $ML_CODE_DIR/B0_ML-env-setup.sh"
    fi
    conda activate "$CONDA_ENV"
fi

export TARGET_TRIALS
export MODEL_FILTER
export HPO_ARCHIVE_DIR
export STUDY_NAME
export SOURCE_LOG
export PLAN_FILE

python - <<'PY'
import os
import re
from pathlib import Path

import optuna
from optuna.trial import TrialState


def normalize_model(value):
    key = str(value).strip().lower()
    aliases = {
        "transformer": "transformer",
        "tr": "transformer",
        "gcn": "gcn",
        "gat": "gat",
        "gnn": "gcn",
        "mlp": "mlp",
    }
    return aliases.get(key, key)


def model_arg(value):
    key = normalize_model(value)
    labels = {
        "transformer": "TR",
        "gcn": "GCN",
        "gat": "GAT",
        "mlp": "MLP",
    }
    return labels.get(key, value)


def infer_target_from_log(log_file):
    if not log_file:
        return None
    path = Path(log_file)
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    totals = [int(match) for match in re.findall(r"(?<!\d)\d+/(\d+)\s*\[", text)]
    return max(totals) if totals else None


archive_dir = Path(os.environ["HPO_ARCHIVE_DIR"])
plan_file = Path(os.environ["PLAN_FILE"])
study_base = os.environ["STUDY_NAME"]
target_text = os.environ.get("TARGET_TRIALS", "").strip()
target = int(target_text) if target_text else infer_target_from_log(os.environ.get("SOURCE_LOG", ""))

filters = {
    normalize_model(part)
    for part in re.split(r"[\s,]+", os.environ.get("MODEL_FILTER", "").strip())
    if part
}

study_dirs = sorted(path for path in archive_dir.iterdir() if path.is_dir() and (path / "full_study.db").exists())
if not study_dirs:
    raise SystemExit(f"No model full_study.db files found below {archive_dir}.")

rows = []
for study_dir in study_dirs:
    normalized = normalize_model(study_dir.name)
    if filters and normalized not in filters:
        continue

    db_path = study_dir / "full_study.db"
    storage = f"sqlite:///{db_path.as_posix()}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not summaries:
        raise SystemExit(f"No Optuna studies found in {db_path}.")

    expected_suffix = f"_{study_dir.name}"
    summary = None
    for candidate in summaries:
        if candidate.study_name == f"{study_base}{expected_suffix}":
            summary = candidate
            break
    if summary is None and len(summaries) == 1:
        summary = summaries[0]
    if summary is None:
        names = ", ".join(candidate.study_name for candidate in summaries)
        raise SystemExit(f"Could not choose one study in {db_path}. Found: {names}")

    expected_name = f"{study_base}{expected_suffix}"
    if summary.study_name != expected_name:
        raise SystemExit(
            f"Study name '{summary.study_name}' does not match folder-derived name "
            f"'{expected_name}'. Use --study-name or check the archive path."
        )

    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    trials = study.get_trials(deepcopy=False)
    finished = sum(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in trials)
    running = sum(t.state == TrialState.RUNNING for t in trials)
    failed = sum(t.state == TrialState.FAIL for t in trials)
    rows.append({
        "model_dir": study_dir.name,
        "model_arg": model_arg(study_dir.name),
        "study_name": summary.study_name,
        "finished": finished,
        "running": running,
        "failed": failed,
        "total": len(trials),
    })

if not rows:
    raise SystemExit("No HPO model studies matched the requested model filter.")

if target is None:
    finished_counts = [row["finished"] for row in rows if row["finished"] > 0]
    if not finished_counts:
        raise SystemExit("Could not infer --target-trials from the log or sibling studies.")
    target = max(finished_counts)

with plan_file.open("w", encoding="utf-8") as handle:
    handle.write("model_dir\tmodel_arg\tstudy_name\tfinished\trunning\tfailed\ttotal\ttarget\tremaining\n")
    for row in rows:
        remaining = max(0, target - row["finished"])
        handle.write(
            f"{row['model_dir']}\t{row['model_arg']}\t{row['study_name']}\t"
            f"{row['finished']}\t{row['running']}\t{row['failed']}\t{row['total']}\t"
            f"{target}\t{remaining}\n"
        )
PY

/bin/echo "Detected HPO resume plan:"
column -t -s $'\t' "$PLAN_FILE" || cat "$PLAN_FILE"

if ! awk -F '\t' 'NR > 1 && $9 > 0 { found = 1 } END { exit found ? 0 : 1 }' "$PLAN_FILE"; then
    /bin/echo "All selected HPO studies have reached the target trial count. Nothing to resume."
    exit 0
fi

if [ "$dry_run" = true ]; then
    /bin/echo "Dry run requested; not launching resumed HPO trials."
    exit 0
fi

/bin/echo "Python: $(which python)"
python -V
/bin/echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-set by Slurm inside GPU jobs}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    /bin/echo "nvidia-smi is not available in this environment."
fi

rsync -av "$REPO_ROOT/resources/" "$SCRATCH_DIR/resources/"
rsync -av "$SCRIPT_SRC" "$SCRIPT_LOCAL"
mkdir -p "$SCRATCH_HPO_DIR"
rsync -av "$HPO_ARCHIVE_DIR"/ "$SCRATCH_HPO_DIR"/

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTHONPATH=$SCRATCH_DIR:${PYTHONPATH:-}
export ML_DATA_ROOT=$DATA_ROOT
export ML_RUN_ROOT=$SCRATCH_RUN_ROOT
export ML_ARCHIVE_ROOT=$ARCHIVE_ROOT
export ML_JOB_NAME=$STUDY_NAME
export ML_RUN_CONTEXT=
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-${SLURM_NTASKS:-12}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

cd "$SCRATCH_DIR"
while IFS=$'\t' read -r model_dir model_arg study_name finished running failed total target remaining; do
    if [ "$model_dir" = "model_dir" ] || [ "$remaining" -le 0 ]; then
        continue
    fi

    /bin/echo "Resuming $model_dir: finished=$finished running=$running failed=$failed total=$total target=$target remaining=$remaining"
    run_args=(
        --task "$TASK"
        --models "$model_arg"
        --study-name "$STUDY_NAME"
        --n-trials-per-typ "$remaining"
    )
    if [ "$no_progress" = true ]; then
        run_args+=(--no-progress)
    fi
    run_args+=("${SCRIPT_EXTRA_ARGS[@]}")

    /bin/echo "Running: python -u $SCRIPT_LOCAL ${run_args[*]}"
    python -u "$SCRIPT_LOCAL" "${run_args[@]}"
done < "$PLAN_FILE"

/bin/echo "HPO resume completed at: $(date)"
