#!/bin/bash
# Isolated final-generator follow-up; preview unless --submit.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: $0 SUITE [--submit]"; exit 2; }
suite=$1; shift
[[ "$suite" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
submit=false
if [[ $# = 1 && "$1" = --submit ]]; then submit=true; shift; fi
[[ $# = 0 ]] || exit 2
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
reference=$archive/MULTI/Dual/Transformer/independent-inputs-261015-mixed-s42/model.json
layout=MULTI/Dual/Transformer
entry=studies/Loss/A0-SpatialBalance.py
submit_dir=/data/home/exy053/p2/$layout/$suite
plans=(
    "spatial|FT|42|0"
    "spatial|FT|42|0.001"
    "spatial|FT|42|0.003"
    "spatial|FT|42|0.01"
    "spatial|FT|42|0.1"
)
for plan in "${plans[@]}"; do echo "PLAN $plan (450 epochs;240h/1GPU/12CPU/90000MB)"; done
[[ "$submit" = true ]] || exit 0
test -z "$(git -C "$repo" status --porcelain)"
test -f "$reference"; test ! -e "$submit_dir"
test -z "$(find "$archive/$layout" -maxdepth 1 -name "$suite-*" -print -quit)"
mkdir -p "$submit_dir/source"
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
export REPO_ROOT=$submit_dir/source ML_CODE_DIR=$submit_dir/source/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD) ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tseed\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
gate=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 8:0:0 B1_ML-new.sh \
    "$entry" --reference-model-json "$reference"  --preflight  --run-label "$suite-preflight")
gate=${gate%%;*}
printf 'preflight\t42\t%s\t-\t%s\n' "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
for plan in "${plans[@]}"; do
    IFS='|' read -r variant mode seed coefficient <<< "$plan"
    extra=(--spatial-coefficient "$coefficient")
    role=$variant
    if [[ "$coefficient" != - ]]; then role=$variant-${coefficient//./p}; fi
    label=$suite-$role-$mode-s$seed
    job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
        --dependency="afterok:$gate" --kill-on-invalid-dep=yes B1_ML-new.sh \
        "$entry" --reference-model-json "$reference"  "${extra[@]}" \
        --seed "$seed" --epochs 450 --run-label "$label")
    job=${job%%;*}
    printf '%s-%s\t%s\t%s\t%s\t%s\n' "$role" "$mode" "$seed" "$job" "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
done
cat jobs.tsv
