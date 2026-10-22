#!/bin/bash
# Preview first. Separate field-only jobs; B1 owns staging/archive/cleanup.
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "Usage: $0 SUITE [--submit]"; exit 2; fi
suite=$1; shift
[[ "$suite" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
submit=false
if [ "$#" = 1 ] && [ "$1" = --submit ]; then submit=true; shift; fi
[ "$#" = 0 ] || exit 2
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
reference=$archive/MULTI/Dual/Transformer/independent-inputs-261015-mixed-s42/model.json
submit_dir=/data/home/exy053/p2/MULTI/Dual/Transformer/$suite
variants=(control degree graph gated_graph adjacency geometry cnn1 cnn2)
echo "Locality: one disposable8h GPU gate;16 field-only jobs, seed42,450 epochs."
echo "Each full job240h/1GPU/12CPU/90000MB; no OOF banks or curve fits."
for variant in "${variants[@]}"; do
    for mode in UT FT; do echo "PLAN $suite-$variant-$mode-s42"; done
done
if [ "$submit" != true ]; then exit 0; fi
test -f "$reference"; test ! -e "$submit_dir"
test -z "$(git -C "$repo" status --porcelain)"
test -z "$(find "$archive/MULTI/Dual/Transformer" -maxdepth 1 -name "$suite-*" -print -quit)"
mkdir -p "$submit_dir/source"
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
export REPO_ROOT=$submit_dir/source ML_CODE_DIR=$submit_dir/source/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD) ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tseed\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
gate=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 8:0:0 B1_ML-new.sh \
    studies/Locality/A0-Locality.py --reference-model-json "$reference" --preflight --run-label "$suite-preflight")
gate=${gate%%;*}
printf 'preflight\t42\t%s\t-\t%s\n' "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
for variant in "${variants[@]}"; do
    for mode in UT FT; do
        label=$suite-$variant-$mode-s42
        job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
            --dependency="afterok:$gate" --kill-on-invalid-dep=yes B1_ML-new.sh \
            studies/Locality/A0-Locality.py --reference-model-json "$reference" \
            --variant "$variant" --mode "$mode" --seed 42 --epochs 450 --run-label "$label")
        job=${job%%;*}
        printf '%s-%s\t42\t%s\t%s\t%s\n' "$variant" "$mode" "$job" "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
    done
done
cat jobs.tsv
