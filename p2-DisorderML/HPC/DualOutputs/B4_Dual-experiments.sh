#!/bin/bash
# Matched experiments only: B1 owns resources, staging, execution and archiving.
# Run on HPC. No submission without --submit. Existing suite directories refused.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: bash B4_Dual-experiments.sh SUITE_NAME [--variants CSV] [--preflight-node NODE] [--submit]"
    exit 2
fi
suite=$1
if [[ ! "$suite" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Use a simple unique suite name (letters, numbers, underscore, hyphen)."
    exit 2
fi
shift
submit=false
selected=
preflight_node=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit) submit=true; shift ;;
        --variants|--preflight-node)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then echo "Missing value for $1"; exit 2; fi
            if [ "$1" = --variants ]; then selected=$2; else preflight_node=$2; fi
            shift 2 ;;
        *) echo "Unknown argument: $1"; exit 2 ;;
    esac
done
if [[ -n "$preflight_node" && ! "$preflight_node" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Use one explicit preflight node name."
    exit 2
fi
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
anchor=$archive/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best/model.json
submit_dir=/data/home/exy053/p2/MULTI/Dual/Transformer/$suite
all_variants=(baseline partial private crack_face local_graph residual late_frame ft_region true_field detach winner_probe curve_predicted curve_true)
variants=("${all_variants[@]}")
if [ -n "$selected" ]; then
    if [[ "$selected" = ,* || "$selected" = *, || "$selected" = *,,* ]]; then
        echo "Variant list contains an empty entry."; exit 2
    fi
    IFS=',' read -r -a requested <<< "$selected"
    seen=' '
    for variant in "${requested[@]}"; do
        if [[ ! "$variant" =~ ^[a-z_]+$ || " ${all_variants[*]} sudden " != *" $variant "* || "$seen" = *" $variant "* ]]; then
            echo "Unknown or repeated variant: $variant"; exit 2
        fi
        seen+="$variant "
    done
    if [[ "$seen" = *' curve_predicted '* || "$seen" = *' curve_true '* ]]; then
        if [[ "$seen" != *' winner_probe '* ]]; then
            echo "Curve fits require winner_probe in the same selection."; exit 2
        fi
    fi
    variants=()
    # Preserve dependency order even when the user's CSV order differs.
    for variant in "${all_variants[@]}" sudden; do
        if [[ "$seen" = *" $variant "* ]]; then variants+=("$variant"); fi
    done
fi
gate_args=()
if [[ " ${variants[*]} " = *' sudden '* ]]; then
    if [[ "${variants[*]}" != 'baseline sudden' ]]; then
        echo "Sudden weighting requires the isolated --variants baseline,sudden pair."; exit 2
    fi
    gate_args=(--sudden-suite)
fi
echo "Submit directory: $submit_dir"
echo "Anchor: $anchor"
echo "One 4-hour suite preflight; ${#variants[@]} dependent 240-hour, one-GPU B1 jobs."
if [ -n "$preflight_node" ]; then echo "Preflight node: $preflight_node"; fi
printf 'Variant: %s\n' "${variants[@]}"
if [ "$submit" != true ]; then exit 0; fi
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
preflight_options=()
if [ -n "$preflight_node" ]; then preflight_options+=(--nodelist="$preflight_node"); fi
preflight=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 4:0:0 "${preflight_options[@]}" B1_ML-new.sh \
    DualOutputs/A0-HPC-Dual-preflight.py --run-label "$suite-preflight" --base-model-json "$anchor" "${gate_args[@]}")
preflight=${preflight%%;*}
printf 'preflight\tall\t%s\t-\t%s\n' "$preflight" "$ML_SOURCE_REVISION" >> jobs.tsv
for variant in "${variants[@]}"; do
    args=(--experiment "$variant" --base-model-json "$anchor" --seed 42 --split-seed 42 --epochs 450)
    if [[ "$variant" = sudden ]]; then args+=(--localization-gain 1); fi
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
