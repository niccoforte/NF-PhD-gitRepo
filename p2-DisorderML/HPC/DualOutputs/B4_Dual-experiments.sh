#!/bin/bash
# Matched experiments only: B1 owns resources, staging, execution and archiving.
# Run on HPC. No submission without --submit. Existing suite directories refused.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash B4_Dual-experiments.sh SUITE_NAME [--submit]"
    exit 2
fi
suite=$1
if [[ ! "$suite" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Use a simple unique suite name (letters, numbers, underscore, hyphen)."
    exit 2
fi
if [ "$#" -eq 2 ] && [ "$2" != "--submit" ]; then exit 2; fi
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
anchor=$archive/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best/model.json
submit_dir=/data/home/exy053/p2/MULTI/Dual/Transformer/$suite
variants=(baseline partial private crack_face local_graph residual late_frame ft_region true_field detach winner_probe curve_predicted curve_true)
echo "Submit directory: $submit_dir"
echo "Anchor: $anchor"
echo "One 4-hour preflight; thirteen dependent 240-hour, one-GPU B1 jobs."
printf 'Variant: %s\n' "${variants[@]}"
if [ "${2:-}" != "--submit" ]; then exit 0; fi
test -f "$anchor"
test -f "${anchor%.json}.mdl"
test ! -e "$submit_dir"
if [ -n "$(git -C "$repo" status --porcelain)" ]; then
    echo "Refusing to stage from a dirty HPC checkout."
    exit 2
fi
for variant in "${variants[@]}"; do
    test ! -e "$archive/MULTI/Dual/Transformer/$suite-$variant"
done
mkdir -p "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
# Immutable checkout snapshot: queued jobs cannot accidentally pick up later edits.
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
export REPO_ROOT=$submit_dir/source
export ML_CODE_DIR=$REPO_ROOT/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD)
export ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tvariant\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
preflight=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 4:0:0 B1_ML-new.sh \
    DualOutputs/A0-HPC-Dual-preflight.py --run-label "$suite-preflight" --base-model-json "$anchor")
preflight=${preflight%%;*}
printf 'preflight\tall\t%s\t-\t%s\n' "$preflight" "$ML_SOURCE_REVISION" >> jobs.tsv
for variant in "${variants[@]}"; do
    args=(--experiment "$variant" --base-model-json "$anchor" --seed 42 --split-seed 42 --epochs 450)
    dependency=$preflight
    if [[ "$variant" = winner_probe || "$variant" = curve_true || "$variant" = curve_predicted ]]; then
        args+=(--source-model-json "$anchor")
    fi
    if [[ "$variant" = curve_true || "$variant" = curve_predicted ]]; then
        dependency=$preflight:$winner_probe
    fi
    job=$(/opt/slurm/bin/sbatch --parsable -J "$suite-$variant" -t 240:0:0 \
        --dependency="afterok:$dependency" --kill-on-invalid-dep=yes B1_ML-new.sh \
        DualOutputs/A0-HPC-Dual-test.py "${args[@]}")
    job=${job%%;*}
    if [ "$variant" = winner_probe ]; then winner_probe=$job; fi
    printf 'full\t%s\t%s\t%s\t%s\n' "$variant" "$job" "$dependency" "$ML_SOURCE_REVISION" >> jobs.tsv
done
cat jobs.tsv
