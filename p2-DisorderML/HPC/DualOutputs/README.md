# Joint UT/FT runs

`A0-HPC-Dual-test.py` is the real-data runner. `A0-HPC-Dual-trial1.py` is a small, explicit parameter preset calling that same runner: loading, training, checkpointing and diagnostics are not duplicated. `B1_ML-new.sh` stages the companion runner when launching trial 1.

## Validated smoke run

On 7 September 2026, Apocrita job **25868425** completed with exit code 0 through the home → scratch → archive workflow. It used 64 selected pairs, three epochs, batch size 2 and validation diagnostics. Scratch was cleaned after successful archive collection. See [the human-readable report](../../samples/hpc-test-report.md).

```bash
# Submit from the home-side launch directory, with REPO_ROOT pointing to the code.
sbatch --time=00:30:00 -J dual-MULTI-smoke B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py --nsims 64 --epochs 3 --batch 2 --no-range-split
sbatch -J dual-MULTI-trial1 B1_ML-new.sh DualOutputs/A0-HPC-Dual-trial1.py
```

Use a unique job/run label. Both runners collect the same result tree under `MULTI/Dual/Transformer/<run-label>/`: best model, architecture/data metadata, source hashes, per-epoch losses, physical predictions, validity masks, four sets of diagnostics and `results/input_audit/`. The launcher archives this tree and its Slurm log, including available partial artifacts after a failure. The default evaluation split is validation, not the locked test set.

## Trial 1: HPO-informed, not dual-HPO optimised

The opt-in `A0-HPC-Dual-lossTrial.py` reuses this preset with `--field-loss-variant baseline|spatial|temporal|both` and fixed balanced validation selection. Spatial/temporal coefficients default to 0.1 and localisation gain to zero. After comparing those variants, `weighted` requires a positive explicit gain. Do not change the live HPO study or resume it against changed fingerprinted modules. See the root README's controlled displacement-loss section for the independent companion, artifacts and workflow. The new wrapper also saves a same-checkpoint true-field/normal-predicted-field curve comparison. This is a substitution diagnostic, not a separately pretrained curve oracle.

The following independent-run `best_params.json` files were recovered from `/data/SEMS-TaoLab/Niccolo-Forte/p2/` on 7 September 2026. These are source records, not newly tuned dual results.

| Setting | UT field | FT field | UT field → full curve | FT field → full curve | Dual trial 1 |
| --- | ---: | ---: | ---: | ---: | --- |
| Token width | 256 | 256 | 256 | 96 | field 256; curve 256 |
| Attention heads | 4 | 4 | 2 | 4 | 4 in both stages |
| Encoder layers | 4 | 5 | 2 | 2 | field 4; curve 2 |
| Feed-forward multiplier | 4 | 2 | 4 | 6 | 4 in both stages |
| Encoder dropout | 0.185 | 0.264 | 0.269 | 0.031 | field 0.20; curve 0.15 |
| Learning rate | 3.32e-4 | 7.44e-5 | 2.92e-5 | 1.43e-4 | one optimiser: 1e-4 |
| AdamW weight decay | 7.21e-9 | 2.47e-9 | 3.80e-8 | 1.96e-8 | 1e-8 |
| Batch size | 2 | 1 | 8 | 8 | 2 paired specimens |

Source files, relative to that archive root:

- `UT/Field/HPO/fUT-fHPO/Transformer/best_params.json`
- `FT/Field/HPO/fFT-fHPO/Transformer/best_params.json`
- `UT/FieldToCurve/HPO/f2cUTfull-fHPO/Transformer/best_params.json`
- `FT/FieldToCurve/HPO/f2cFTfull-fHPO/Transformer/best_params.json`

Both curve HPO runs used full 201-point targets, mean pooling and a CLS token. Trial 1 retains these choices, with masked mean pooling over body tokens; the CLS token can still influence attention. ReLU encoder activation and learned positional embeddings match all four sources. The canonical order is therefore part of the checkpoint/data contract.

Field capacity matches the four-layer UT reference and is close to the five-layer FT reference. The shared curve stage uses the larger UT width, a common four-head setting and the two-layer depth found in both curve studies. One learning rate must serve the joint network; 1e-4 is an explicit compromise. These settings are a starting point, not a claim that transferring independent HPO optima is optimal.

Trial 1 uses train-standardised MSE for each of the four targets with weights 1, at most 450 epochs, plateau factor 0.7/patience 12 and early-stop patience 75. This is closer to the independent MSE studies than the smoke runner's default physical-curve CombinedCurveLoss. Pooling, head depth, separate attention/head dropouts and normalisation options are not all identical to the legacy architectures: the dual baseline intentionally has simple LayerNorm–Dropout–Linear output heads. Record that distinction in comparisons. CLI arguments override the preset; the effective configuration and preset hash are saved.

## Why keep the synthetic test file?

`test_dual_contract.py` checks active architecture and data invariants without HPC data: exactly two encoders, one call per stage, absent-node masking, joint gradients, normalisation/checkpoint round trips, pin-selection calculations, the trial preset, and runner artifact collection. It is not a legacy compatibility shim or a second training workflow. It catches regressions that a successful three-epoch training run alone cannot identify.

```bash
python -m unittest discover -s p2-DisorderML/HPC/DualOutputs -p 'test_dual_contract.py' -v
```

## Trial 1 evidence and interpretation

Job 26267130 completed in 3 h 37 min with 7,137 training, 793 validation and 881 test specimens. The saved checkpoint is epoch 74; early stopping ended training at 149. The downloaded local folder was renamed `dual-MULTI-trial1`; the remote archive identity remains `dual-MULTI-trial1-260910`.

| Validation output | RMSE | RMSE improvement vs training mean | Diversity ratio |
| --- | ---: | ---: | ---: |
| UT field | 1.0167 | +5.47% | 0.658 |
| FT field | 0.1823 | -23.58% | 0.851 |
| UT curve | 4.4130 | +3.42% | 0.422 |
| FT curve | 6702.4230 | +14.53% | 0.574 |

RMSE units differ across outputs. Field and curve diversity summaries also use different aggregation conventions; inspect per-component plots. UT's global curve R² of 0.964 does not imply strong specimen discrimination: its peak correlation is only 0.417. At epoch 74, normalized validation contributions are UT field 0.1278, FT field 0.00404, UT curve 0.03607, FT curve 0.11366. The low FT field contribution despite negative baseline skill demonstrates why tuning weights and ranking trials are separate problems. These are validation observations, not locked-test results or proof of the cause of underperformance.

## Full joint HPO: preparation and scope

`A0-HPC-Dual-HPO.py` loads the paired data once and calls `resources/MLdualHPO.py`. Exactly two shared-task Transformer stages and one optimizer remain. Defaults: 200 completed/pruned evaluations, 450 epochs maximum, full paired data, the Trial 1 split/seed, 230 hours per job. The unchanged Slurm policy allows ten days. Trial 1 settings are evaluated first under the new ranking; prior numbers are not injected as a fictitious completed Optuna trial.

| Search group | Range/options |
| --- | --- |
| Each stage's width / heads | 96, 128, 256, 384 / 2, 4, 8 (all divisible) |
| Field / curve depth | 2–6 / 1–4 |
| Feed-forward multiplier | 2, 4, 6 per stage |
| Encoder / output-head dropout | 0–0.35 / 0–0.30 per stage |
| Attention dropout / head hidden layer | 0–0.30 / absent or width 1×/2× stage width |
| Activation / positions / pre-norm | ReLU or GELU / learned, none, sinusoidal / false or true per stage |
| Curve pooling | CLS or masked mean; optional CLS token for mean pooling |
| Batch / optimizer | 1, 2, 4, 8 paired specimens / AdamW or Adam |
| Field/base learning rate | log-uniform 1e-5–8e-4 |
| Curve-stage LR multiplier | log-uniform 0.25–4; still one optimizer |
| Weight decay / gradient clipping | log-uniform 1e-9–1e-3 / off, 0.5, 1, 5 |
| Plateau factor / patience | 0.3–0.8 / 8–25 epochs |
| Early-stop patience | 50–100 epochs |
| Four task loss weights | Three independent ratios, each 1/8–8 relative to UT field; normalized to sum 4 |
| Curve loss | Standardized MSE or physical CombinedCurveLoss |
| Combined loss terms | MSE anchor 1; zone-MSE .01–1, derivative .001–.3, peak .01–1, energy .005–.5, peak-location .001–.1 (log ranges) |
| Derivative / soft peak | order 1 or 2 / beta 5–40 |

Three relative weights cover the four-task balance without a redundant overall loss multiplier. Keeping MSE as an anchor avoids unconstrained deletion of pointwise supervision. Curve-specific term ratios are shared across UT/FT, with distinct task weights and existing task-specific zones; independently tuning every term twice would expand an already broad conditional search substantially.

Do not tune data splits, masks, node mappings, mechanics/zone definitions, feature provenance, normalization or seed to obtain a better score. Full curves remain fixed; PCA, new graph connectivity, probabilistic heads and new physical constraints are different experiments. Existing encoder initialization is retained for comparability; independent layer initialization is a separately testable improvement, not silently mixed into Trial 1.

### Ranking and pruning

Ranking uses the mean over four tasks of physical specimen-MSE divided by the validation specimen-MSE of that task's **training-fitted mean**. With fully valid fields this is the physical global MSE ratio. A score of 1 equals the baseline on average; lower is better. Each task ratio is saved. Normalizers and means are fitted on training only; the fixed validation denominator is selection information, not a fitted data transform. None of the tuned loss weights enters this score.

The same fixed score selects checkpoints, drives scheduling/early stopping and is reported to Optuna. This prevents a trial winning merely by reducing difficult-task weights. Raw and weighted training objectives remain logged separately. Equal-task averaging still permits trade-offs: inspect the worst branch as well as the joint winner.

The search uses [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) with 25 startup trials and median pruning after 10 completed startup trials, with a 75-epoch warmup and 10-epoch intervals. The long warmup is intentional because Trial 1's best epoch was 74. The broad search is not exhaustive and cannot promise the best possible model. At Trial 1's cost, ten days fit roughly 64 full runs before overhead; pruning and smaller models can increase evaluations. Resume across jobs if 200 evaluations are not reached. Confirm a small shortlist across independent training seeds before a single locked-test assessment.

### Submit and resume through the existing job workflow

Submit from a home-side study launch directory, with `REPO_ROOT` set to the synchronized repository. No new shell wrapper or policy is introduced:

```bash
sbatch -J dual-joint-hpo1 B1_ML-new.sh DualOutputs/A0-HPC-Dual-HPO.py --study-name dual-joint-hpo1
# Only after the previous job has stopped:
sbatch -J dual-joint-hpo1-resume B1_ML-new.sh DualOutputs/A0-HPC-Dual-HPO.py --study-name dual-joint-hpo1 --resume --target-trials 200
```

The study lives at `MULTI/Dual/Transformer/HPO/dual-joint-hpo1/`. B1 stages code to scratch as usual. The Python runner restores the archived study on explicit resume, trains from scratch for each new trial, and periodically snapshots SQLite and checkpoint/history files to the archive. B1 performs its final archive/log copy on exit.

- `full_study.db`, `trials.csv`: trial states, parameters, intermediate scores and per-task best scores.
- `study_contract.json`: code/data hashes, split IDs, normalizers, fixed baseline denominators and search scoring contract; mismatches reject resume.
- `trials/trial-NNNN/`: checkpoint and loss history, including available pruned/failed-trial artifacts.
- `best/`: current completed winner's `model.mdl`, metadata, history, physical validation predictions and all four diagnostic sets.
- `best_params.json`, `best_trial_user_attrs.json`: selected parameters and recorded outcomes.

Set the dual post-processing notebook's `RUN_DIR` to the study's `best/` directory. The general legacy HPO notebook/B2 loader is not a dual checkpoint consumer. Transfer with `B3_ML-transfer-mac.sh MULTI Dual Transformer HPO dual-joint-hpo1` (or the Windows counterpart).

Resume is **study-level**, not exact mid-epoch/optimizer continuation. A budget-interrupted configuration is queued for retraining from epoch one; completed trials remain. Median-pruned trials are not automatically rerun. A CUDA OOM records a failed trial and continues without secretly reducing batch size. Other unexpected errors stop the run so defects are not disguised as poor hyperparameters.

Only one worker may own the study. An archive-side `study.lock` records job ID/PID. If a hard kill leaves the lock, first verify that job is no longer active before removing that one lock. Do not run two jobs against the same archive. Live SQLite is backed up through the SQLite API, not copied mid-transaction. The 230-hour budget is checked at epoch boundaries and leaves a ten-hour reserve; it is not permission to exceed Slurm's limit.
