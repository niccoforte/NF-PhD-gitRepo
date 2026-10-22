#!/bin/bash
# Preview by default. Immutable source snapshot; B1 owns scratch and archiving.
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: bash B4_Independence.sh SUITE [--submit]"; exit 2; fi
suite=$1; shift
if [[ ! "$suite" =~ ^[A-Za-z0-9_-]+$ ]]; then echo "Invalid suite label"; exit 2; fi
submit=false
if [ "$#" = 1 ] && [ "$1" = --submit ]; then submit=true; shift; fi
if [ "$#" != 0 ]; then echo "Unknown arguments"; exit 2; fi
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
anchor=$archive/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best/model.json
submit_dir=/data/home/exy053/p2/MULTI/Dual/Transformer/$suite
cases=(1i 1ii 2ai 2aii 2bi 2bii 3i 3ii)
seeds=(42 43 44)
echo "Preview: one 8h GPU preflight; six reusable field banks; 24 case fits."
echo "Every full job: 240h, 1 GPU, 12 CPUs, 90GB through B1."
echo "Architecture/training anchor: $anchor; losses reset to equal normalized MSE."
for seed in "${seeds[@]}"; do
    for case in "${cases[@]}"; do echo "Case ${case}p seed $seed"; done
done
if [ "$submit" != true ]; then exit 0; fi
test -f "$anchor"
test ! -e "$submit_dir"
if [ -n "$(git -C "$repo" status --porcelain)" ]; then echo "Refusing dirty HPC source"; exit 2; fi
for seed in "${seeds[@]}"; do
    for role in bank-1i bank-3i 1ip 1iip 2aip 2aiip 2bip 2biip 3ip 3iip; do
        test ! -e "$archive/MULTI/Dual/Transformer/$suite-$role-s$seed"
    done
done
mkdir -p "$submit_dir/source"
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
export REPO_ROOT=$submit_dir/source ML_CODE_DIR=$submit_dir/source/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD)
export ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tseed\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
gate=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 8:0:0 \
    B1_ML-new.sh DualOutputs/A0-HPC-Dual-preflight.py --independence-suite \
    --run-label "$suite-preflight" --base-model-json "$anchor")
gate=${gate%%;*}
printf 'preflight\t42\t%s\t-\t%s\n' "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
for seed in "${seeds[@]}"; do
    common=(--experiment baseline --base-model-json "$anchor" --seed "$seed" --split-seed 42 --epochs 450 --oof-folds 3)
    for bank in 1i 3i; do
        label=$suite-bank-$bank-s$seed
        job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
            --dependency="afterok:$gate" --kill-on-invalid-dep=yes B1_ML-new.sh \
            studies/Independence/A0-Independence.py "${common[@]}" --independence-case "$bank" --prepare-field-bank --run-label "$label")
        job=${job%%;*}
        if [ "$bank" = 1i ]; then independent_job=$job; else shared_job=$job; fi
        printf 'bank-%s\t%s\t%s\t%s\t%s\n' "$bank" "$seed" "$job" "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
    done
    for case in "${cases[@]}"; do
        options=(); dependency=$gate
        case "$case" in
            1i|2ai) bank=1i; dependency=$independent_job ;;
            2bi|3i) bank=3i; dependency=$shared_job ;;
        esac
        if [ "$dependency" != "$gate" ]; then
            options=(--field-bank "$archive/MULTI/Dual/Transformer/$suite-bank-$bank-s$seed")
        fi
        label=$suite-${case}p-s$seed
        job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
            --dependency="afterok:$dependency" --kill-on-invalid-dep=yes B1_ML-new.sh \
            studies/Independence/A0-Independence.py "${common[@]}" --independence-case "$case" --run-label "$label" "${options[@]}")
        job=${job%%;*}
        printf '%sp\t%s\t%s\t%s\t%s\n' "$case" "$seed" "$job" "$dependency" "$ML_SOURCE_REVISION" >> jobs.tsv
    done
done
cat jobs.tsv
