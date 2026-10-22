# Joint UT/FT runs

## Bounded damage contract replay

`A0-HPC-Damage-contract.py` is an execution/consumer-layout check for the six exact
fixtures listed in `handoffs/HANDOFF-DAMAGE.md`, not a trainer. It reuses
`resources.damage`; it does not open an ODB, change experiments or join predictions.
Job31976809 already completed the six-fixture CUDA/reload replay on16 October;
saved confirmation is in samples/hpc-cleanup-261017/job-evidence.json. Do not
repeat it as a handoff prerequisite. The full FCC banks are delivered; the
remaining training-consumer/model plan belongs to Improve ML Accuracy.
A new training implementation still needs its own authorised execution gate.

For a separately justified reproduction, package the six NPZ/original-INP pairs
and an existing Case1(i) DATA descriptor:
`--prepare --samples-root PATH --inp-root PATH --consumer-data MODEL_DATA_JSON
--output-dir NEW_DIRECTORY`. This verifies original INP fingerprints, copies only
the explicit files and writes a manifest with thirteen file hashes. No ODB is
copied. Use the canonical Z: audit samples and original INPs, create the bundle
in a temporary directory and verify its hashes before execution.

Preview the arguments and submit through B1 with an immutable source snapshot and
a unique `damage-contract-preflight-*` label: `--manifest FILE
--manifest-sha256 HASH --require-cuda`. One hour is ample for this bounded replay;
retain existing GPU/CPU/memory/account directives. No production damage job is
implied. B1 deletes successful scratch and keeps the home log, not an archive copy.
Remove temporary home fixture/source copies only after terminal success with no
remaining consumer. Local fixture bundles and replay exports are temporary too;
they are not retained after a completed check.
The human report belongs in ignored `samples/damage-hpc-contract/`.

Checks cover STATUS parent/known masks, original INP geometry, FT absence,
reference-connectivity remapping, frame0-only removal and NumPy/Torch/device reload.
`DAMAGE_CONTRACT_REPORT=` in the home log is the machine-readable result. Actual
times remain unchanged; differences from saved shared ML times are reported, not
interpolated or accepted as exact alignment. Legacy scalar-severity NaNs are not
unknown binary labels. Six files are five unique task responses from three
designs, not a training corpus. A later specimen join still needs exact initial
geometry/source and permitted-split proof; locked-test evaluation remains excluded.

## Case 1(i): input and custom-loss studies

The next studies retain fully independent UT/FT models and separately trained
field/curve stages. They reuse the completed `independence-261013-r1` banks and
Case1(i) controls, the same 7,137/793/881 specimen split, seeds42/43/44, HPO-derived
per-stage capacity and optimizer settings, max450 epochs and stopping patience52.
Locked test is untouched. There is no architecture HPO or change to legacy defaults.

```bash
# Preview only. Run inputs first; explicit --submit queues through B1.
bash p2-DisorderML/HPC/DualOutputs/B4_Independent-followups.sh independent-inputs-NEW inputs
LOSS_CURVE_SOURCE=mixed INPUT_SUITE=independent-inputs-261015 \
  FIELD_BANK_SUITE=independent-losses-261018 \
  bash p2-DisorderML/HPC/DualOutputs/B4_Independent-followups.sh independent-losses-NEW losses
```

Every full job requests240h/1GPU/12CPU/90000MB. Each study has one disposable8h
GPU/data/save/reload preflight. Home submission/source/manifest and archive paths
retain the existing `MULTI/Dual/Transformer/<label>` contract, although the models
are independent. Preflights keep only successful home logs; they produce no archive
dependency. No jobs are submitted by a preview or by this documentation.

### Input study (15 full jobs plus gate)

P = out-of-fold predicted training fields, final-generator validation fields.
T = true fields; C = independently corrected predicted fields. Five curve fits
per seed are represented; the existing P control is reused, four are new:

| Curve training | Main deployment input | Additional saved evaluations |
| --- | --- | --- |
| P (existing control) | P | T already saved; C from correction-bank fixed readout |
| T | P | T and C |
| 50/50 P or T | P | T and C |
| C | C | T and P |
| 50/50 C or T | C | T and P |

Mixtures sample one complete source per specimen/update with a private seeded RNG.
They do not interpolate physical histories or double updates. Curve checkpoints
select using deployment-source validation RMSE. True-field evaluation of that
checkpoint is diagnostic, not an independently true-validation-selected oracle.
All actual deployment paths start from disorder and predicted fields.
The true-trained curve fit is also evaluated on held-out TRUE fields, saved in
`results/true_field_curves.npz` and per-task sample metric CSVs. This same-checkpoint
diagnostic measures source mismatch; selection remains predicted-field validation,
so it is not a separately true-validation-selected oracle. The preflight verifies
that this export exists. The input study completed successfully and its saved
validation results were reviewed before preparing the selected-policy loss study.

Each seed first fits a correction bank. The two independent residual Transformers
have width96/4heads/2layers, FF4, GELU, sinusoidal positions and dropout.05; output
residuals start at zero. They consume fields plus existing physical context,
retain FT masking, supervise corrected fields directly and freeze the generators.
They do not receive true fields at deployment or curve gradients during fitting.

Corrected training fields need **nested** cross-fitting. For each original outer
TRAIN fold, the original fit/selection IDs exclude its holdout. Inner3-fold field
banks generate OOF corrector-training inputs within those fit IDs; a corrector
selects using the separate inner-selection specimens, then corrects the untouched
outer holdout. Convert through physical units to the outer training normalization.
A final corrector fits the full original OOF training bank and supplies validation
predictions. This avoids indirect target leakage through first-stage generators.
It adds twelve inner/final field fits per seed and independent corrector fits;
each field-fit set contains separate UT and FT fits. This is substantially more
expensive than source sampling alone. The10-day request
is a ceiling, not an ETA or exact mid-fit-resume guarantee. Nested provenance and
checkpoints are retained as production evidence.

Correction-bank standard results use the original frozen Case1(i) curve readout,
isolating correction alone. New C-source fits then measure curve adaptation.
Extra probes save `corrected_source_curves*`, `uncorrected_source_curves*` and/or
`fixed_readout_curves*` beside the run root, with IDs, physical curves, sample
metrics and train-mean baseline summaries. Standard results always use the declared
deployment source. `correction_bank.json/npz` is a versioned, fingerprinted cache;
the checkpoint includes base+correction field stages and loads via the existing
dual review loader. All specimen, scale, mask and bank provenance checks apply.

### Selected-policy custom-loss study

The18 October review of793 validation specimens/three seeds selects
`corrected_mixed`: mean fixed four-output physical-MSE/baseline-MSE score0.743791,
versus0.756978 for predicted-only. Paired mean RMSE changes are UTfield−0.57%,
FTfield−1.95%, UTcurve−0.63%, FTcurve−0.68%. These are modest validation gains,
not proof of statistical superiority or resolved fracture paths. Mixed raw/true
training is cheaper and slightly better for the UT curve alone. True-trained
curves on true validation fields remain a separate diagnostic, never deployment.
See canonical `samples/field-input-study/README.md` and its `selection.json`.

The cost-aware review and subsequent researcher decision supersede that numerical
selection for continued training: use **Predicted+True (`mixed`)**, without a
corrector. Correction gave only small/mixed accuracy gains at much greater cost;
the original input comparison and its selection file remain measured evidence.

Set `LOSS_CURVE_SOURCE=mixed`, `INPUT_SUITE=independent-inputs-261015` and
`FIELD_BANK_SUITE=independent-losses-261018` to reuse the12 compatible custom-loss
banks. Their field training never used a corrector or curve supervision. The new
suite `independent-losses-261022-mixed` has18 full jobs plus one disposable8h gate:

| Production work | Jobs | Dependency/source |
| --- | ---: | --- |
| Existing custom-loss OOF field banks |0 new| Retain12 banks from18 October |
| Adapted MSE curve fits |12| Matching raw bank;50/50 predicted/true training |
| Combined curve loss, location off/on |6| Original raw field banks;50/50 predicted/true training |

The launcher validates all12 reused jobs before submission: completed0:0 requires
bank arrays/checkpoint/DATA sidecar; running/pending banks become afterok
dependencies in addition to the new gate. `reused-banks.tsv` records that mapping.
The runner still verifies IDs, scales, architecture, training and loss provenance.
Do not pass preflight artifacts as banks. Without `FIELD_BANK_SUITE`, the raw-source
route creates30 new jobs; this is unnecessary for the present continuation.

Frozen reference curve readouts come from the same-seed `mixed` input model. A bank-hash
difference is allowed only for explicitly changed-generator field-loss comparisons;
seed/data/scaler/model/policy checks remain strict and run before costly fitting.
For curve-loss comparisons, original generator state equality is checked.
The64-sample/one-epoch GPU gate exercises all selected source/loss paths,
diagnostics and reload, not full-population split validation or accuracy.

Reuse normalized masked-MSE field and MSE-curve controls under that policy. Test field changes
separately, with independently retrained OOF banks and final generators:

- Spatial signed relative-displacement supervision, coefficient0.1.
- Temporal recorded-frame increment supervision, coefficient0.1.
- Both terms, each0.1.
- Sudden-motion specimen/node/frame weighting, gain1, no derivative penalties.
  This is the existing capped temporal-change/non-affine rule, not displacement
  magnitude weighting. It uses training targets only and needs no inference weights.

Each variant then fits independent MSE curve models using the selected source
policy and its OOF predicted/true fields. Also evaluate the winning
baseline's frozen curve readout on the new fields; this separates
field improvement from downstream adaptation. Fit structured-loss jump scales and
normalizers inside each fold. Selection remains the unchanged physical RMSE ratio,
restricted to the active independent task. Field and curve custom losses are not
combined in these initial comparisons.

Separately, freeze the selected baseline field pipeline and compare `CombinedCurveLoss` with
peak-location weight0 versus corrected soft/soft weight0.02 (beta20); both retain
other existing curve terms/weights. The matched source-policy MSE run is the third
control. "Unchanged baseline fields" means identical upstream model weights and
raw predicted field arrays within this curve-loss comparison. All deployment
evaluations use predicted fields; true-field substitutions remain diagnostics.

The current runner's explicit curve-test preset is MSE1, zone-MSE0.5,
derivative0.05, peak-height0.25, work0.10 and peak-location0/0.02; soft beta20 and
first differences. Zone weights remain1/5/0.2 over UT slices0:65/65:130/130:end
and FT0:85/85:160/160:end. These override the class constructor defaults. Field
coefficients0.1 preserve the earlier controlled-loss setting; sudden gain1 gives
raw weights1 to below2, then specimen-valid mean-one normalization. These are
fixed starting presets, NOT HPO-selected optima or equal gradient contributions.
The inspected `f2cUTfull-fHPO`/`f2cFTfull-fHPO` best_params select MSE, not a custom
coefficient set. Log raw/weighted components and judge against the fixed metrics;
poor results at one setting do not rule out a loss family.
Report hard peak height/location, work, field motion/local errors, mean-relative
RMSE skill, seed variation and log-scale histories. No improvement is assumed.
Gaussian/frame, fixed FT-region, residual and graph changes remain later isolated
tests. Carry forward the winning input policy, then vary losses alone so their
effects remain distinguishable. Submission is explicitly authorised by the
researcher after the completed input review; submission evidence lives in PROJECT_STATUS.

## Preflight retention

Preflights retain only successful logs in the home submission directory. B1
recognizes `preflight` in script/job/run labels, skips archive copies, and removes
scratch on success. They still test training, saving and reload within scratch.
Failure preserves scratch temporarily for diagnosis; cleanup of failed logs,
scratch and useless archives is high priority once corrected. Production banks,
trained checkpoints and active jobs must not be mistaken for disposable tests.
On14 October56 preflight archive directories from nine completed jobs were removed;
their successful home logs remain. Existing immutable submissions are not edited.

## Eight-case independence comparison (12 October)

Recovery: the original gate31417279 completed, but18 full jobs failed before
training because the split guard compared string IDs with integer IDs. The
remaining12 staged jobs never started. The guard now normalizes both sides while
preserving exact membership and order. A regression test covers representation,
ordering and membership; the GPU gate now also runs one full-data epoch before
its subset cases, exercising the formerly skipped production check. User approved
cleanup of the failed scratch copies, metadata-only archives and home logs;
successful preflight artifacts and the original source/manifest remain evidence.
Replacement `independence-261013-r1` submitted from `6b0db82`: gate31448215,
six banks and24 cases31448216–31448245, all full jobs240h and dependency-gated.
Manifest: `/data/home/exy053/p2/MULTI/Dual/Transformer/independence-261013-r1/jobs.tsv`.
Gate31448215 and all30 production jobs31448216–31448245 are verified
COMPLETED0:0. The24 case fits and six reusable field banks are downloaded under
`Z:/p2/MULTI/Dual/Transformer/`, retaining the archive labels after user relocation.
The human review is `../../samples/independence-review/README.md` relative to this
directory. Generated data and samples remain ignored; source notebooks stay in code.

The three-seed validation review recommends Case1(i) as the practical starting
point for further accuracy work. Mean RMSE reductions versus the training-mean
prediction are UTfield6.11%, FTfield24.20%, UTcurve4.55%, FTcurve18.73%.
For current joint Case3(ii), they are5.09%,13.30%,3.43%,−0.17%, respectively.
All end-to-end variants have nearly constant FT curves across specimens, including
Case1(ii), which has no shared UT/FT parameters. This does not identify gradient
conflict as the cause. The fixed configuration, OOF cost and per-task rather than
total-capacity matching limit generalisation of the verdict. Locked test remains
unevaluated; numerical local-motion/curve acceptance margins remain unagreed.

Submitted on13 October as `independence-261013`, immutable source `ec9189d`.
Original GPU gate **31417279** completed on `sbg16`; the full jobs failed as
described above. Use the replacement manifest, not these historical IDs.
Manifest: `/data/home/exy053/p2/MULTI/Dual/Transformer/independence-261013/jobs.tsv`.
Successful replacement archives are under
`/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/`
with labels `independence-261013-r1-<case>p-s<seed>` and matching `bank-1i|3i` labels.
These names supersede the older case numbering in historical notes.
`F` means geometry→field; `C` means field→curve; `(i)` freezes the completed field
fit before curve training; `(ii)` retains end-to-end field+curve supervision.

| Case | Field networks | Curve networks | Independent training groups |
| --- | --- | --- | --- |
| 1(i) | UT, FT separate | UT, FT separate | 4 |
| 1(ii) | UT, FT separate | UT, FT separate | 2: one end-to-end stream per mode |
| 2a(i) | UT, FT separate | shared dual | 3 |
| 2a(ii) | UT, FT separate | shared dual | 1 |
| 2b(i) | shared dual | UT, FT separate | 3 |
| 2b(ii) | shared dual | UT, FT separate | 1 |
| 3(i) | shared dual | shared dual | 2: F then C |
| 3(ii) | shared dual | shared dual | 1: current joint architecture |

Independent stages share **no** tokenizer, context projection, position/task
embedding, encoder, normalization parameters or output head. Every independent
training group gets its own optimizer, scheduler, selection score, stopping and
checkpoint. Group losses contain only that group's targets. A common saved-run
container does not mean shared training. Case 1(ii) has no UT↔FT gradient path.

Main cases have `p` suffix: deployable inference always uses predicted fields.
For `(i)`, training curve inputs are three-fold out-of-fold (OOF) predictions from
outer TRAIN specimens only. Each fold fits scalers and fields on its own fit IDs,
selects on separate inner-validation IDs, and predicts untouched heldout TRAIN IDs.
Predictions are converted through physical units into the outer train-fitted
field normalization. Final generators train on the complete outer training set
and supply outer-validation inputs. Cache provenance includes all fold IDs,
data/configuration fingerprints and seed checks. Cases 1(i)/2a(i) reuse the exact
same independent field bank; 2b(i)/3(i) reuse the same shared field bank.

OOF is more than withholding the final validation/test set: a downstream learner
must not train on unrealistically accurate in-sample predictions from its upstream
learner. Outer test targets are never evaluated. Fold-internal `test` datasets are
explicitly labelled heldout OUTER TRAIN IDs, not the project's locked test set.

Do not double the suite into misleading `t` variants: true-only curve inputs cut
the curve→field gradient, eliminating the defining property of `(ii)`. Every fit
saves true-field **substitution** diagnostics; these are not trained true-field
oracles. Existing oracle studies remain separate source evidence. Auxiliary
true-field supervision and a residual field corrector are deferred until after
the architecture decision, as are all feature/loss ablations.

High-priority follow-ups after the sharing decision: compare custom field/curve
losses against MSE, and curve training on predicted-only, true-only and both field
sources. Evaluate ALL source variants on predicted fields; additional true-field
diagnostics are not deployment results. True-only training does not propagate
curve gradients into the field generator. MSE here is a common control, not a
conclusion that custom losses are inferior.

### Fixed comparison configuration

Use the archived `HPO/dual-joint-hpo1/best/model.json` architecture/training anchor,
not its weights or unequal four-task HPO loss weights. Full runs verify the
anchor's exact paired split IDs. All eight cases use identical normalized masked
field MSE and curve MSE, coefficient 1 per active output; no custom losses, graph,
crack-face addition, residualization, localization or auxiliary supervision.

| Setting | Field stage | Curve stage |
| --- | --- | --- |
| Width / heads / layers | 384 / 8 / 3 | 96 / 4 / 4 |
| Feed-forward multiplier | 4 | 6 |
| Activation / position | ReLU / sinusoidal | GELU / learned |
| Pooling | per-node output | masked mean, no CLS |
| Encoder dropout | 0.0423943 | 0.0168863 |
| Attention dropout | 0.1046886 | 0.2288386 |
| Head dropout / hidden multiplier | 0.2959779 / 2 | 0.0724220 / 0 |

AdamW; field LR 0.0002707084; curve LR factor 0.2979038; weight decay
0.000327321; batch 2; plateau factor 0.627342/patience14/threshold0.0001;
early-stop patience 52 from the anchor, delta 0.00001; maximum 450
epochs per fit. Seeds 42/43/44; split seed 42. Effective settings and anchor hash
are saved. Curve initialization is independently seeded so changing field-stage
sharing does not incidentally change its initial weights. Per-task block capacity
is matched, **not total parameter count**; record capacity and computational cost.
This is a controlled configuration comparison, not architecture-specific HPO or
proof that any architecture is globally optimal. If near-tied, tune the finalists
under equal search budgets before a definitive research claim.

Selection uses fixed physical MSE relative to a training-mean predictor, restricted
to the currently trained group's outputs. Final comparison reports all four
outputs, seed variability, local-motion/event metrics, FT inside/outside-region
errors and curve peak/work metrics. Keep RMSE/relative skill for comparability;
log training losses separately (log plots), never compare unequal custom objectives
as if they measured the same error. Accepted field non-inferiority margin is 2%
on BOTH tasks. Numerical curve/local-motion safeguards still require agreement
before declaring a winner. Gradient-norm/alignment logging is explicitly deferred.

### Preparation and submission

`resources/MLindependence.py` orchestrates the existing trainer; it does not
duplicate its optimization loop. The existing test runner accepts
`--independence-case 1i|1ii|2ai|2aii|2bi|2bii|3i|3ii` with `--experiment baseline`.
Bank preparation additionally uses `--prepare-field-bank` (1i or 3i); staged fits
require `--field-bank <archived-bank-directory>`. Group checkpoints and histories
are retained under `groups/`; each OOF fit under `folds/`. The final assembled
checkpoint reloads using the maintained dual loader and standard postprocessing.
Bank-only runs explicitly omit curve results because that stage is untrained.
Grouped history plots label independent fits and retain log scales.

```bash
# Non-submitting preview; run from the clean HPC checkout when authorised.
bash p2-DisorderML/HPC/DualOutputs/B4_Dual-independence.sh NEW-UNUSED-LABEL
# Once authorised and synchronized:
bash p2-DisorderML/HPC/DualOutputs/B4_Dual-independence.sh NEW-UNUSED-LABEL --submit
```

One 8h GPU gate exercises two small banks and all eight cases, including checkpoint
reload. Six reusable full-data banks and 24 case fits follow across three seeds.
Every full job requests 240h; B1 retains its existing one-GPU/12CPU/90,000MB policy
and home→scratch→archive collection. No extra CPU-per-task override is added.
Bank consumers require successful bank archiving; all jobs depend directly or
indirectly on the gate. Source is an immutable clean-Git snapshot; unique labels
are mandatory. Archive: `MULTI/Dual/Transformer/<suite>-<case>p-s<seed>` or
`<suite>-bank-{1i|3i}-s<seed>`. No exact mid-fit resume is claimed; failed jobs
retain B1 scratch and existing artifacts for inspection, not blind overwriting.

The active `test_independence.py` checks parameter/gradient isolation, frozen
field invariance, group counts, fold-only scalers/IDs, all eight CPU fits, checkpoint
round trips and non-submitting launcher behavior. Completed GPU execution and
validation results are recorded above; synthetic checks alone do not establish accuracy.

## Controlled architecture and interface experiments

### Sudden-motion weighting: isolated next experiment

Prepared, not submitted or accuracy-validated. The existing loss now has an
explicit `localization_mode="sudden"`; old `activity` and all model defaults stay
unchanged. `A0-HPC-Dual-test.py --experiment sudden --localization-gain 1` changes
only displacement-loss weights. The signed spatial/temporal loss coefficients
remain zero, so this first comparison isolates weighting from derivative losses.

For each specimen, after reconstructing physical displacements:

1. Compute each recorded-frame displacement increment, then its change between
   adjacent intervals. Steady translation or steady affine deformation gives
   zero temporal score; large displacement alone is not rewarded.
2. On the validated initial task graph, fit a local linear map from initial
   neighbour offsets to differences in displacement increments. Use the actual
   initial **disordered** coordinates for this fit, not the periodic positions
   used to reconstruct connectivity. The unexplained residual is the spatial
   score. Require at least three valid neighbours and rank two; skip an
   underdetermined spatial patch rather than inventing a deformation estimate.
3. Normalize each contribution with the existing task/component training-only
   jump RMS scales. Combine their squared scores equally and take the square
   root, giving nonnegative activity `q`. Distribute interval spatial activity
   to its two endpoint frames; assign temporal change to the middle frame.
4. Form raw weights `1 + gain*q/(1+q)`, then divide by their mean across that
   specimen's valid node/frame/component entries. Gain1 gives a maximum 2:1
   ratio between any two raw weights, with normalized weights between 0.5 and2.
   Retain whole-field supervision and detach weights from the gradient graph.

Weights are computed from each training target, not stored as permanent weights
for particular node IDs. Validation targets can define the same diagnostic loss;
checkpoint selection remains the unchanged unweighted four-task score. At
inference no targets, weights or extra inputs are needed. No fixed crack-region
prior is added. Loading transients or other nonlinear events can also activate
the rule: it is a kinematic proxy, not a fracture classifier or physical
acceleration. Missing frames are not bridged. No strain/damage exports needed.

Preview from the HPC directory, using a fresh suite label:

```bash
bash DualOutputs/B4_Dual-experiments.sh dual-sudden-261008 --variants baseline,sudden
```

Only add `--submit` after checking the source, anchor and existing manifests.
This selects a two-case 64-pair/one-epoch GPU gate, then two dependent full runs
through B1: 240 hours, one GPU, 12 CPUs, 90,000 MB, max450 epochs, anchor patience52,
seed/split42. Baseline and weighting use the same data and architecture. The
default thirteen-case suite is unchanged. CPU contracts (including a synthetic
800-node runner/save check) are in `test_sudden_weighting.py`; GPU execution and
scientific benefit remain unverified. This runner wires sample coordinates into
DUAL; single-mode/GNN trainers are not yet wired for this new weighting mode.

### Decision experiment: shared versus fully independent

Design awaiting diagram confirmation, not implemented/submitted: three training
seeds on one frozen paired split. Compare four separate fits; independent fields
feeding a dual curve fit; independent end-to-end UT/FT streams; current joint
DUAL; and, as a suggested extra, separately trained dual field and curve stages.
Four separate fits and stage-separated UT/FT streams are the same configuration
when their curve input source/schedule match: do not duplicate them. Detailed
training boundaries and deferred follow-ups are in the accuracy brief's current
decision suite. True-field curve oracles remain separate source controls.
Independence across UT/FT includes all stages, projections and optimizers; an
end-to-end stream may use one optimizer for its own two stages. The existing
`private` encoder-tail variant does NOT satisfy this control. Match field/curve
information, target masks, losses, normalization, per-task capacity, stopping
rules and training budget; report the independent pair's larger total parameter
count and runtime. Use predicted fields for end-to-end curve comparisons and
true fields only as a separately labelled oracle. Existing HPO populations differ
and cannot settle this comparison without matched retraining/evaluation.

Primary decision: both field RMSEs, plus local-jump and curve accuracy safeguards;
not the aggregate score alone. The user accepted a 2% field non-inferiority margin
on both tasks; numerical local-motion/curve safeguards remain to be specified.
Report per-seed results and paired specimen uncertainty,
not individual nodes/frames as independent replicates. Freeze acceptance criteria
before final locked-test evaluation. If DUAL consistently sacrifices a field
without a worthwhile practical benefit, retain independent models; this is a
decision for this dataset/budget, not a universal claim about multitask learning.
No new independent two-stage runner is implemented by this preparation.

### Corrected peak-loss follow-up

For fresh source-controlled fits, add
`--curve-loss-ablation combined_no_location` or `--curve-loss-ablation combined_soft`
to `--experiment curve_true` / `--experiment curve_predicted`. Pass the same
`--base-model-json` and frozen `--source-model-json` as the existing MSE controls.
The explicit switch replaces the anchor's curve loss, not its field loss,
architecture, task weights or fixed validation ranking. `mse` is also available
as an explicit control. Historical/default loss behavior remains unchanged.

Both combined variants retain MSE, zone, derivative, peak-height and energy
terms; only soft-location weight differs (0 versus default 0.02, beta20).
Both use the same soft target definition. Report hard peak position/height,
curve RMSE and work errors; soft centroids alone are not peak accuracy.
Combined-versus-MSE is a broader objective change, including normalization.

The proposed comparison is four full fits (two field sources × two combined
losses), reusing the already-completed matched MSE controls after config/split
verification. Run `A0-HPC-Dual-preflight.py --curve-loss-suite` first through B1:
64 pairs/one epoch for a baseline plus those four modes. Use a 4-hour gate and
240-hour dependent full jobs, 450 maximum epochs and the anchor's early stopping,
with ordinary home → scratch → archive collection. B4's default suite is
unchanged. Submission records follow; PROJECT_STATUS owns the latest verified
scheduler state. Prepared entry points alone are not execution evidence.

Submitted on 8 October as `dual-peak-261008`, immutable source `c29d77f`.
Preflight **30813256** checks five modes; full jobs require its successful exit
and archive collection (`afterok`, `kill-on-invalid-dep=yes`):

| Field source | Combined without location | Combined soft/soft location |
|---|---:|---:|
| True | 30813257 | 30813258 |
| Frozen predicted | 30813259 | 30813260 |

Every full job requests 240 hours, one GPU, 12 CPUs and 90,000 MB on
andrena/pilot_andrena. Preflight requests 4 hours. All use seed/split seed42,
450 maximum epochs and anchor early stopping52. The original MSE source controls
have matching anchor hashes/configurations and specimen splits and are not repeated.
Home manifest: `/data/home/exy053/p2/MULTI/Dual/Transformer/dual-peak-261008/jobs.tsv`.
Archive: `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/dual-peak-261008-<curve_true|curve_predicted>-<combined_no_location|combined_soft>`.
Scheduler resource/test-only checks passed; at submission the gate was waiting
for resources and full jobs for their dependency. Do not claim GPU success yet.

The four previously outstanding jobs (30367256/30367272,30326584,30326586) were
verified COMPLETED 0:0 and downloaded with B3 on 8 October, including checkpoints,
metrics, predictions and logs. A checksum dry-run found no differing files.
Property archive: `MULTI/FieldToProperty/Transformer/field-property-261004`
(and `field-property-preflight-261004`). Joint archives:
`MULTI/Dual/Transformer/dual-recovery-261004-residual` and
`dual-recovery-261004-true_field`. Both are relative to the same p2 archive root
and external Windows `Z:/p2/` (other hosts require an explicit external root). Read the Z: checkout's `samples/accuracy-continuation/README.md` for
the ten-run comparison and property/thickness reports; the debug gate is not an
accuracy result. `true_field` adds auxiliary supervision, not oracle inference.

The existing `A0-HPC-Dual-test.py` accepts `--experiment`; no extra trainer is
introduced. All switches are opt-in. The first comparison is baseline vs
crack-face-only vs local-graph-only; do not combine changes before measuring them.

| Experiment | One change from the anchor |
|---|---|
| `baseline` | unchanged shared field and curve stages |
| `crack_face` | existing optional 26-node FT feature, UT zero; 12 context columns |
| `local_graph` | one shared message MLP on separate initial UT/FT graphs before global attention; still 11 context columns |
| `partial` | last field block private to UT/FT by default; earlier field blocks shared |
| `private` | all field **encoder blocks** private; input projections and curve stage still shared |
| `true_field` | additional same-curve-network supervision on true fields, coefficient 0.5 |
| `detach` | curve losses no longer backpropagate into field predictions; field supervision retained |
| `residual` | per-node/frame/component training mean + residual scaling, floored at 10% of pooled frame/component scale |
| `localization` | existing target-activity displacement weights, gain 1; no spatial/temporal penalties added |
| `late_frame` | displacement MSE weighted by raw 1 + relative recorded load; same rule for both tasks/components |
| `ft_region` | displacement MSE weighted 2 inside the reference FT CrRegMESH node box, 1 outside; UT unchanged |
| `winner_probe` | no training: evaluate the frozen HPO winner on predicted and true fields |
| `curve_predicted` / `curve_true` | fresh shared UT/FT curve stage trained on frozen winner predictions / true fields respectively |

### Reproducible matched suite

B1 bootstraps the site's module shell and exports Bash's host identity before
loading Miniforge, so submissions through non-interactive SSH do not depend on
an interactive login's exported functions. Resource requests are unchanged.

On HPC, preview `bash DualOutputs/B4_Dual-experiments.sh dual-compare-260930`
from the repository HPC directory; append `--submit` to launch. Use a fresh name
for a new suite. It submits one **4-hour preflight** and thirteen dependent
**240-hour jobs**, all through the unchanged B1 resource policy (one GPU,
12 CPUs, 90,000 MB, andrena/pilot_andrena). `afterok` and
`--kill-on-invalid-dep=yes` prevent full training after a failed preflight.
The two fresh curve fits additionally depend on the full-data winner probe,
which checks the original checkpoint/data contract before releasing those fits.
For selective recovery, `--variants` accepts a comma-separated subset and keeps
dependency order (curve fits require winner_probe in the same selection).
`--preflight-node NODE` optionally tests the existing all-mode GPU preflight on
an available node involved in an environment failure; full jobs retain normal
placement. Never target or bypass an administratively drained node.
For example, preview only the seven previously unsuccessful experiments:

```bash
bash DualOutputs/B4_Dual-experiments.sh dual-recovery-261004 \
  --variants residual,ft_region,true_field,detach,winner_probe,curve_predicted,curve_true
```

Append `--submit` only after environment diagnosis and Git deployment; full
training remains gated by successful GPU preflight and archiving.
The reduced selection changes neither hyperparameters nor the 240-hour resource
requests, and never overwrites the earlier successful runs.
The preflight exercises all thirteen modes on 64 pairs/one epoch, including
checkpoints, true/predicted-source diagnostics and B1 archiving. Full experiments
use all pairs, seed/split seed 42 and at most 450 epochs with early stopping.
`winner_probe` is evaluation only. This is first-seed screening, not replicated
evidence; repeat promising comparisons on additional seeds later.

Home launch/log/manifest directory: `/data/home/exy053/p2/MULTI/Dual/Transformer/<suite>`.
Its `source/` is an immutable Git snapshot, including current and earlier test
scripts. B1 stages that snapshot's resources and selected entry point into
`/gpfs/scratch/exy053/<job-id>` and archives `mlruns/` to
`/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/<suite>-<variant>`.
`jobs.tsv` records IDs/dependencies/revision. Successful archive/log copying
permits scratch cleanup; failures retain scratch. Future repo edits cannot change
already queued jobs. No executable files or deployment bundles go under local data/.

For the submitted replacement suite, the manifest is
`/data/home/exy053/p2/MULTI/Dual/Transformer/dual-compare-260930-r1/jobs.tsv`.
It is a tab-separated submission ledger with columns `role`, `variant`,
`job_id`, `dependency`, and `source_revision`, not a metrics file or a complete
training configuration. Reproduction needs the source revision/snapshot **and**
the launch arguments (snapshot B4 script and job logs), saved effective model/run
configuration, matching data and split IDs, and environment versions. A script
filename or job ID alone is insufficient; seeds do not guarantee bitwise
reproduction across different hardware/software.

The `-test.py` filename does not impose a short run: the current dual runner
defaults to a maximum of 450 epochs. The original smoke run explicitly used
three epochs; this suite's preflight explicitly uses 64 pairs and one epoch per
mode. A preflight is a small end-to-end rehearsal of loading, GPU execution,
metrics, checkpointing and archive collection, not an accuracy comparison.
`test_dual_contract.py` instead contains local synthetic regression checks; it
does not launch research training or certify the scientific assumptions.

The suite intentionally excludes `localization`: current activity weights are
not the requested affine-departure detector. It includes all other variants
listed above separately, not combinations. Source HPO artifacts are never edited.

### Verified suite status and storage cleanup — 4 October 2026

Before recovery, no jobs remained queued/running. Full baseline, partial, private, crack_face,
local_graph and late_frame completed 0:0; their archive checkpoints, predictions,
metrics and logs exist. Five jobs failed before Python on sbg10 because the
miniforge module was unavailable: residual 29601768, ft_region 29601770,
true_field 29601771, detach 29601772 and winner_probe 29601773. The latter's
dependent curve fits 29601774/29601775 were cancelled. These are environment
failures, not model-accuracy results; diagnose before any authorised replacement.
Successful preflight results do not prove every execution node's module setup.

The original preflight 29598744 failed earlier with `module: command not found`.
Its and the five later failure logs printed `Data saved under:` unconditionally
from B1's EXIT handler, but rsync transferred an empty tree (total size 0).
No corresponding failed-run result directories existed in the conventional
MULTI/Dual/Transformer archive; no archive deletion was necessary.

With explicit user approval, removed the five empty scratch job directories
listed above (each contained only empty mlruns/) and their five home failure logs,
plus original preflight log `dual-compare-260930-preflight.o29598744`.
The user had already removed empty scratch 29598744. These log deletions were
permanent; no backup copies were made. Preserved all home scripts/manifests/source
snapshots, six successful full archives and thirteen successful preflight outputs.
After that cleanup only the hidden `.scratch-expiring-history` directory remained
under user scratch; it was left untouched. That cleanup did not resubmit jobs.
Failure evidence is summarised here; raw failed logs are deliberately not retained.

### Selective recovery — 4 October 2026

The renewed connection showed sbg10 administratively `IDLE+DRAIN`, reason
`modules`. The current B1 bootstrap loads the site's Miniforge and nf-ml-gpu
successfully on login-01; there is no evidence requiring a model-code change or
an alternative installation. Respect the drain and let Slurm select an available
node. This avoids the known failed node without claiming its configuration is
repaired or guaranteeing the environment on every other node.

Submitted `dual-recovery-261004` from immutable source `0b9215e` after synchronising
the clean HPC checkout. Home manifest:
`/data/home/exy053/p2/MULTI/Dual/Transformer/dual-recovery-261004/jobs.tsv`.

| Role / variant | Replacement job | Required successful jobs |
| --- | --- | --- |
| all-mode preflight | 30326583 | none |
| residual | 30326584 | 30326583 |
| ft_region | 30326585 | 30326583 |
| true_field | 30326586 | 30326583 |
| detach | 30326587 | 30326583 |
| winner_probe | 30326588 | 30326583 |
| curve_predicted | 30326589 | 30326583 and 30326588 |
| curve_true | 30326590 | 30326583 and 30326588 |

After SSH reconnection, preflight 30326583 was verified COMPLETED 0:0 in 9m05s
on sbg23; all thirteen archives contain checkpoint, predictions, metrics and
job log. Winner probe 30326588 completed 0:0 in 2m43s, with both field-source
diagnostics archived. At approximately 15:47 BST all six remaining training
jobs were RUNNING, with advancing epochs. No accuracy conclusions yet. The seven
full jobs request 240 hours each, one GPU, 12 CPUs and 90,000 MB on
andrena/pilot_andrena. The six successful original runs are not repeated.
Archive labels are `dual-recovery-261004-<variant>` in the conventional dual
tree above; compare these with the six `dual-compare-260930-r1-<variant>` runs,
not with their preflight counterparts. Hyperparameters, seeds, model code and
metric collection are unchanged. Do not submit duplicates.

SSH access reuses the user's persistent master and must leave it running after
commands/transfers. The observed `Broken pipe` did not identify its underlying
network cause. `ControlPersist=12h` controls idle lifetime; it cannot guarantee
network continuity. Optional `-o ServerAliveInterval=60 -o ServerAliveCountMax=5`
on the next master-start command detects unresponsive transport; it cannot
prevent disconnects during Mac sleep or network changes. No SSH settings were
changed and the reconnected master was verified alive across multiple commands.

### Weighting and frozen-source comparisons

For submitted job IDs, download commands and the existing notebook review
settings, read `../../handoffs/HANDOFF-ML-ACCURACY.md` → **Collect and visualise this suite**. Keep notebooks
in `p2-DisorderML/code/`; saved runs alone belong in `data/`. The source-comparison
plot uses explicit predicted-field evaluation tables when present, so a
true-field-trained model is not inadvertently compared against itself.
The accuracy brief distinguishes existing functionality from pending notebook
work and automated checks from researcher review. Accuracy owns the per-test
visual guide under `samples/` and agreed soft/soft peak correction. `../../handoffs/HANDOFFS.md`
is an index to the separate task briefs, not required reading for all chats;
damage, context, surrogate optimisation, tokenisation and physics/strain each
have their own brief in that directory.

Fixed weights are normalised to mean one over each specimen's valid values.
Late-frame raw weights are `1+(t-t_first)/(t_last-t_first)`, so on equally spaced
frames the effective weights span approximately 2/3 to 4/3. No adaptive error
feedback, component priority or additional spatial/temporal penalty is added.
FT's A1 box in cell-size-10 coordinates is x=96.6..173.6, y=54..136:
`xCrE=120-1.2*0.2*10=117.6`; subtract/add 2.1/5.6 cells horizontally and
4.1 cells vertically about y=95. It selects **125 retained reference nodes**.
This is a node prior inspired by the meshing box, not Abaqus edge-set membership
or the eventual crack path. Uniform unit/translation conversion is tested.
Every experiment reports global/inside/outside FT RMSE in `ft_region_metrics.json`;
frame/component diagnostics and unweighted selection remain unchanged.

Frozen-source runs require `--source-model-json`, checked against reconstructed
data IDs, scalers, features and frame coordinates. `winner_probe` copies the
winner unchanged into a new diagnostic run. The two `curve_*` runs copy only its
field stage, keep it frozen with dropout off, and train fresh identically seeded
curve stages. They use the winner's **four** curve blocks, not one; field depth
is three. They retain one shared UT/FT curve network with task conditioning.
No field supervision contributes in these curve-only fits, and checkpoint
selection averages the two unchanged curve baseline-relative scores (not four).
The curve architecture/loss/task weights/learning rate are otherwise held fixed.

Saved field arrays remain frozen field predictions even for `curve_true`;
standard curve results use the declared training source. Both source evaluations
are additionally saved as `true_field_curves.npz` / `predicted_field_curves.npz`
and labelled per-sample metric tables. The true-source curve is an oracle
diagnostic requiring true fields, not a deployable disorder-only inference result.
Its ordinary network forward still accepts predicted fields; use the recorded
source comparison tables for interpretation. Predicted training fields are
in-sample predictions of the existing field model, not cross-fitted predictions;
validation remains held out. This controls the interface comparison but is not
a guaranteed upper bound or an independently tuned curve HPO.

`private` is an encoder-sharing control, **not** complete task independence.
Small tokenizer/context projections and the entire curve stage still share
parameters. Unlike legacy MODEL's independent task optimizers/sequential fits,
every option keeps paired data, both stages, one optimizer, joint supervision and
one checkpoint. A fully independent end-to-end UT/FT comparison would require
separating those remaining components too; it is not silently substituted here.
Private tails increase total parameters at fixed per-task depth. Counts by stage
and a split hash are saved; do not attribute a gain exclusively to sharing before
capacity/seed checks. The local graph MLP shares weights across tasks: graph lists
are data, not trainable networks. Initial struts are not evolving damage labels.

Use the final HPO winner as the common **configuration** anchor, training fresh
weights. `--base-model-json` restores architecture, loss definitions/weights,
optimizer, LR groups, scheduling and recorded early-stop patience. It does not
resume weights or a study. Defaults remain 450 maximum epochs, fixed balanced
validation selection, and split seed 42 independent of training seed. CLI
overrides are recorded; keep them identical across variants. For a loss change,
use the existing loss-trial route separately, not both switches together.

```bash
# First run a one-epoch B1 GPU preflight; this is an example, NOT a submission record.
sbatch --time=02:00:00 -J dual-graph-preflight B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py \
  --experiment local_graph --base-model-json /data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best/model.json \
  --nsims 64 --epochs 1 --no-range-split --seed 42 --split-seed 42
# Once GPU/staging/archive checks pass, one unique full-data job per variant/seed:
sbatch --time=240:00:00 -J dual-baseline-s42 B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py \
  --experiment baseline --base-model-json /data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best/model.json \
  --seed 42 --split-seed 42
```

After the first baseline/crack/graph comparison, compare `partial` and `private`
against that same baseline, then interface/residual/localisation variants one at
a time. Repeat promising variants with training seeds 42/43/44 and fixed split
42. No claim of scientific benefit follows from synthetic unit tests.

Every experiment saves standard physical diagnostics, log-compatible loss history,
motion diagnostics, true-field curve substitution, and `design_diagnostics.json`
plus its readable Markdown summary. Design screening measures per-task objective
ranking/top-10% recovery and selection regret; it does not generate designs or
run a Pareto search. UT work uses its own post-peak 1% cutoff; FT full-domain work
is labelled a proxy, **not** fracture toughness. Missing cutoff events are counted.
`--minimum-ut-strength` enables false-feasibility counts only with a researcher-
chosen physical threshold. Future physical FT cutoff/normalised multi-objective
definitions remain an explicit scientific decision. Selection still uses the old
fixed four-output score, not these newly added diagnostics.

Worked arithmetic: `../../samples/dual-experiment-examples.md`. Accuracy
continuation: [HANDOFF-ML-ACCURACY.md](../../handoffs/HANDOFF-ML-ACCURACY.md); separate task
briefs: [HANDOFFS.md](../../handoffs/HANDOFFS.md). The latest explanatory diagrams and
read-only evidence are `../../samples/dual-clarification-evidence.md` and the
adjacent `dual-sharing-*.png`, `dual-peak-loss-explanation.png` and
`dual-field-difficulty.png`. These do not activate new training choices.
Current localisation weights measure neighbour/time differences, not departures
from smooth affine motion; they can still favour moving-boundary nodes. See the
handoff before treating them as fracture-anomaly weights. Temporal decoder,
attention bias and explainer work remain deferred. Do not interpret sparse field
frames as requiring temporal averaging of the 201-point target curves.

For independent GNNs, `MODEL(..., graph_semantics="fcc_initial_v1")` opts into
the validated crack cut and bidirectional edge_index. Default/old descriptors
retain `historical`, including their old graph semantics; never reinterpret an
archived GNN checkpoint as having been trained on the corrected graph. The new
dual graph block always uses the validated `reference_field_edges` graph.
The edge helper is checked for native and padded FT indexing. Corrected legacy
GNN **training** requires native 788-node FT data and rejects padded 800-node FT,
because its pooling does not implement absent-node masks. The dual graph route
supports canonical padding with masks. This explicit guard prevents ghost-node
contributions rather than silently changing legacy pooling.

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

Set the dual post-processing notebook's `RUN_DIR` to the study's `best/` directory. `code/ML-HPOpostProcess.ipynb` supports dual studies through its dual branch; legacy `MODEL` and B2 are not dual checkpoint consumers. Transfer with `B3_ML-transfer-mac.sh MULTI Dual Transformer HPO dual-joint-hpo1` (or the Windows counterpart).

Resume is **study-level**, not exact mid-epoch/optimizer continuation. A budget-interrupted configuration is queued for retraining from epoch one; completed trials remain. Median-pruned trials are not automatically rerun. A CUDA OOM records a failed trial and continues without secretly reducing batch size. Other unexpected errors stop the run so defects are not disguised as poor hyperparameters.

Only one worker may own the study. An archive-side `study.lock` records job ID/PID. If a hard kill leaves the lock, first verify that job is no longer active before removing that one lock. Do not run two jobs against the same archive. Live SQLite is backed up through the SQLite API, not copied mid-transaction. The 230-hour budget is checked at epoch boundaries and leaves a ten-hour reserve; it is not permission to exceed Slurm's limit.
