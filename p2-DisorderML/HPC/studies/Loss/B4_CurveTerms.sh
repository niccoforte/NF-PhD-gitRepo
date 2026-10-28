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
layout=MULTI/FieldToCurve/Transformer
entry=studies/Loss/A0-CurveTerms.py
submit_dir=/data/home/exy053/p2/$layout/$suite
plans=(
    "mse|UT|42|-"
    "mse|FT|42|-"
    "range_mse|UT|42|-"
    "range_mse|FT|42|-"
    "zone|UT|42|-"
    "zone|FT|42|-"
    "derivative|UT|42|-"
    "derivative|FT|42|-"
    "peak_height|UT|42|-"
    "peak_height|FT|42|-"
    "work|UT|42|-"
    "work|FT|42|-"
    "peak_location|UT|42|-"
    "peak_location|FT|42|-"
    "combined|UT|42|-"
    "combined|FT|42|-"
    "mse|UT|43|-"
    "mse|FT|43|-"
    "range_mse|UT|43|-"
    "range_mse|FT|43|-"
    "zone|UT|43|-"
    "zone|FT|43|-"
    "derivative|UT|43|-"
    "derivative|FT|43|-"
    "peak_height|UT|43|-"
    "peak_height|FT|43|-"
    "work|UT|43|-"
    "work|FT|43|-"
    "peak_location|UT|43|-"
    "peak_location|FT|43|-"
    "combined|UT|43|-"
    "combined|FT|43|-"
    "mse|UT|44|-"
    "mse|FT|44|-"
    "range_mse|UT|44|-"
    "range_mse|FT|44|-"
    "zone|UT|44|-"
    "zone|FT|44|-"
    "derivative|UT|44|-"
    "derivative|FT|44|-"
    "peak_height|UT|44|-"
    "peak_height|FT|44|-"
    "work|UT|44|-"
    "work|FT|44|-"
    "peak_location|UT|44|-"
    "peak_location|FT|44|-"
    "combined|UT|44|-"
    "combined|FT|44|-"
)
for plan in "${plans[@]}"; do echo "PLAN $plan (450 epochs;240h/1GPU/12CPU/90000MB)"; done
[[ "$submit" = true ]] || exit 0
test -z "$(git -C "$repo" status --porcelain)"
test -f "$reference"; test ! -e "$submit_dir"
for task in UT FT; do
    test -z "$(find "$archive/$task/FieldToCurve/Transformer" -maxdepth 1 -name "$suite-*" -print -quit)"
done
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
    extra=(--variant "$variant" --mode "$mode")
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
