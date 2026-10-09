# Joint UT/FT runs

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
and local repo-root `data/`. Read `samples/accuracy-continuation/README.md` for
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
