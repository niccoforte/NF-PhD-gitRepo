#!/bin/bash
# Case1(i) follow-ups. Preview by default; explicit --submit uses B1 unchanged.
set -euo pipefail
if [ "$#" -lt 2 ]; then echo "Usage: $0 SUITE inputs|losses [--submit]"; exit 2; fi
suite=$1; study=$2; shift 2
[[ "$suite" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$study" = inputs || "$study" = losses ]] || exit 2
submit=false
if [ "$#" = 1 ] && [ "$1" = --submit ]; then submit=true; shift; fi
[ "$#" = 0 ] || exit 2
repo=${REPO_ROOT:-/data/home/exy053/00-PhD-gitRepo}
archive=${ARCHIVE_ROOT:-/data/SEMS-TaoLab/Niccolo-Forte/p2}
base=$archive/MULTI/Dual/Transformer
anchor=$base/HPO/dual-joint-hpo1/best/model.json
control=${CONTROL_SUITE:-independence-261013-r1}
submit_dir=/data/home/exy053/p2/MULTI/Dual/Transformer/$suite
echo "Case1(i), $study, seeds42/43/44; reuse $control baseline banks/results."
echo "One8h disposable GPU gate; every full job240h/1GPU/12CPU/90000MB through B1."
if [ "$study" = inputs ]; then
    echo "15 full jobs: three nested-correction banks and12 curve fits."
    echo "Sources: true, mixed, corrected, corrected_mixed; raw predicted control reused."
else
    echo "30 full jobs: twelve field banks, twelve downstream fits, six curve-loss fits."
    echo "Field losses: spatial, temporal, both, sudden. Curve losses: combined_no_location, combined_soft."
fi
if [ "$submit" != true ]; then exit 0; fi
test -f "$anchor"; test ! -e "$submit_dir"
test -z "$(git -C "$repo" status --porcelain)"
for seed in 42 43 44; do
    test -f "$base/$control-bank-1i-s$seed/field_bank.json"
    test -f "$base/$control-1ip-s$seed/model.json"
done
test -z "$(find "$base" -maxdepth 1 -name "$suite-*" -print -quit)"
mkdir -p "$submit_dir/source"
git -C "$repo" archive HEAD | tar -x -C "$submit_dir/source"
cp "$repo/p2-DisorderML/HPC/B1_ML-new.sh" "$submit_dir/"
export REPO_ROOT=$submit_dir/source ML_CODE_DIR=$submit_dir/source/p2-DisorderML/HPC
export ML_SOURCE_REVISION=$(git -C "$repo" rev-parse HEAD) ARCHIVE_ROOT=$archive
cd "$submit_dir"
printf 'role\tseed\tjob_id\tdependency\tsource_revision\n' > jobs.tsv
gate=$(/opt/slurm/bin/sbatch --parsable -J "$suite-preflight" -t 8:0:0 B1_ML-new.sh \
    DualOutputs/A0-HPC-Dual-preflight.py --followup-suite "$study" --base-model-json "$anchor" --run-label "$suite-preflight")
gate=${gate%%;*}
printf 'preflight\t42\t%s\t-\t%s\n' "$gate" "$ML_SOURCE_REVISION" >> jobs.tsv
for seed in 42 43 44; do
    common=(--experiment baseline --base-model-json "$anchor" --independence-case 1i \
            --followup-study "$study" --seed "$seed" --split-seed 42 --epochs 450 --oof-folds 3)
    raw_bank=$base/$control-bank-1i-s$seed
    reference=$base/$control-1ip-s$seed/model.json
    launch() {
        local role=$1 dependency=$2; shift 2
        local label=$suite-$role-s$seed job
        job=$(/opt/slurm/bin/sbatch --parsable -J "$label" -t 240:0:0 \
            --dependency="afterok:$dependency" --kill-on-invalid-dep=yes B1_ML-new.sh \
            DualOutputs/A0-HPC-Dual-test.py "${common[@]}" --run-label "$label" "$@")
        job=${job%%;*}
        printf '%s\t%s\t%s\t%s\t%s\n' "$role" "$seed" "$job" "$dependency" "$ML_SOURCE_REVISION" >> jobs.tsv
        last_job=$job
    }
    if [ "$study" = inputs ]; then
        launch correction "$gate" --field-bank "$raw_bank" --prepare-correction-bank --reference-model-json "$reference"
        correction_job=$last_job; correction=$base/$suite-correction-s$seed
        for source in true mixed corrected corrected_mixed; do
            launch "$source" "$correction_job" --field-bank "$raw_bank" --curve-training-source "$source" --correction-bank "$correction"
        done
    else
        for variant in spatial temporal both sudden; do
            launch bank-$variant "$gate" --prepare-field-bank --field-loss-variant "$variant"
            launch "$variant" "$last_job" --field-bank "$base/$suite-bank-$variant-s$seed" --field-loss-variant "$variant" --reference-model-json "$reference"
        done
        for loss in combined_no_location combined_soft; do
            launch "$loss" "$gate" --field-bank "$raw_bank" --curve-loss-ablation "$loss"
        done
    fi
done
cat jobs.tsv
