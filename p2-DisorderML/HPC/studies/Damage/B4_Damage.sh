#!/bin/bash
# Eight independent STATUS-screen fits; preview unless explicitly submitted.
set -euo pipefail
[[ $# -ge 1 ]] || { echo 'Usage: B4_Damage.sh SUITE [--submit]'; exit 2; }
suite=$1;shift
[[ "$suite" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
submit=false
if [[ $# = 1 && "$1" = --submit ]]; then submit=true;shift;fi
[[ $# = 0 ]] || exit 2
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
reference=$archive/MULTI/Dual/Transformer/independent-inputs-261015-mixed-s42/model.json
submit_dir=/data/home/exy053/p2/MULTI/Damage/Transformer/$suite
for mode in UT FT;do
    for variant in control damage weak strong;do echo "PLAN $suite-$variant-$mode-s42 (450 epochs; 240h/1GPU/12CPU/90000MB)";done
done
[[ "$submit" = true ]] || exit 0
test -z "$(git -C "$repo" status --porcelain)"
test -f "$reference";test ! -e "$submit_dir"
for mode in UT FT;do test -f "$archive/MLdata/MULTI-$mode-disNodes-allD.npz";done
test -z "$(find "$archive/MULTI/Damage/Transformer" -maxdepth 1 -name "$suite-*" -print -quit 2>/dev/null || true)"
mkdir -p "$submit_dir/source"
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
export REPO_ROOT=$submit_dir/source ML_CODE_DIR=$submit_dir/source/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD) ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tseed\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
gate=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 8:0:0 B1_ML-new.sh \
    studies/Damage/A0-Damage.py --reference-model-json "$reference" --damage-root "$archive/MLdata" \
    --preflight --run-label "$suite-preflight")
gate=${gate%%;*}
printf 'preflight\t42\t%s\t-\t%s\n' "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
for mode in UT FT;do
    for variant in control damage weak strong;do
        label=$suite-$variant-$mode-s42
        job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
            --dependency="afterok:$gate" --kill-on-invalid-dep=yes B1_ML-new.sh \
            studies/Damage/A0-Damage.py --reference-model-json "$reference" --damage-root "$archive/MLdata" \
            --mode "$mode" --variant "$variant" --seed 42 --epochs 450 --run-label "$label")
        job=${job%%;*}
        printf '%s-%s\t42\t%s\t%s\t%s\n' "$variant" "$mode" "$job" "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
    done
done
cat jobs.tsv
