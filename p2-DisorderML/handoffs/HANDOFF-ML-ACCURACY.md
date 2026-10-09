# Improve ML Accuracy — continuation brief

Read this brief alongside your existing chat's unfinished work and the user's
new annotations; reconcile overlaps rather than restarting or replacing them.
Do not load the sibling handoffs or the full source conversation by default.
Damage extraction, repository context optimisation, and surrogate design
optimisation are separate tasks. This brief owns model accuracy, loss machinery,
experiment review, regression-test examples and diagnostic notebooks.
It has not been dispatched automatically. New submissions require explicit scope.

## Start here: ownership and current state

### Current decision suite and imported deferred ideas (9 October)

This remains the active detailed checklist, not a completed/disposable brief.
Performance takes priority over sharing. The user accepted the proposed 2% field
RMSE non-inferiority margin on BOTH tasks, with local-motion and curve safeguards;
their numerical tolerances still need specifying before final model selection.
The expanded suite is a design for user confirmation, not implemented/submitted:

| Case | Separate training groups | Gradient boundary |
| --- | --- | --- |
| 1 / 3(ii) | UT field, FT field, UT curve, FT curve (four) | Each trains only on its own target; frozen fields feed the curve fits |
| 2 | UT field, FT field, dual curve (three) | Independent field fits; UT/FT share only the downstream curve fit |
| 3(i) | UT field→curve and FT field→curve (two) | Field+curve losses propagate within each stream, never across mechanical modes |
| 4 | Current joint DUAL (one) | Shared field and curve stages, all four losses jointly trained |
| Suggested extra | Dual field, then dual curve (two) | UT/FT sharing retained but curve gradients cannot update the frozen field stage |

Cases 1 and 3(ii) are identical if field sources/training schedules match; do not
submit duplicate jobs under different labels. Distinguish true-field-trained
curve oracles from predicted-source fits as source controls, not a fictitious
architectural difference. Proposed main staged fits use training-only out-of-fold
predictions; common independent field checkpoints/predictions can be reused by
cases 1 and 2. Evaluate deployable curves on predicted fields in every case.
Use one complete-pair split, identical feature information/masks/train-only scales,
three training seeds, matched per-task block capacity and training opportunities;
report total parameters/runtime and source mismatch. Fully independent means no
shared trainable projections, embeddings, normalisation parameters or optimizers,
not merely private encoder tails. Log gradient norms/cosines where gradients are
shared to investigate interference; mixed errors alone do not prove that cause.
Compare on validation; leave locked test untouched until selection is frozen.

After this comparison, isolate independent-model follow-ups: initial crack-face
feature; FT-only A1 fine-mesh-box weighting; taskwise residual fields; local graph
block; spatial/temporal/both losses; corrected curve loss; auxiliary true-field
curve supervision; and the new sudden/non-affine weighting. Existing independent
loss trials are completed evidence, not experiments never tried. New runs must
use the matched population/configuration. Detachment is covered by staged versus
end-to-end controls; an online stop-gradient fit is optional only if scheduling
needs separating from gradient effects. Do not combine all changes at once.

Correction to the earlier verdict: fixed FT crack-tip-box weighting is a valid,
user-approved physical region priority, not leakage/cheating or a rejected idea.
Keep outside-region supervision and whole-field/inside/outside reporting. Dynamic
sample-specific weighting is a separate comparison, not a compulsory replacement.
Gaussian frame weights are newly requested, not implemented: propose positive
1 + a*exp(-0.5*((q-0.5)/sigma)^2), mean-normalised over valid entries, with q the
verified normalised loading coordinate. Start widths sigma .15/.25 and amplitudes
a 1/3 against uniform weighting. Check training-side event timing before claiming
the midpoint is typical; retain positive tail weights and per-frame reporting.
Do not mix Gaussian/regional/sudden weights in the first comparisons.

Field correction remains after the independence verdict: a small residual
field-to-field model, initially separate by task, trained on held-out-fold field
predictions paired by specimen ID with true fields. Freeze the upstream model;
use field supervision first and assess both a fixed curve readout and subsequent
curve adaptation separately. Out-of-fold generators/scalers must not train on
their held-out specimens; final validation/test are never corrector training data.
No guarantee that missing fracture information can be reconstructed.

The user-supplied retirement briefs from **Assess unified ML model feasibility**,
**Summarize GNN chat changes**, and **Review cleanup folder** are incorporated;
those chats may be archived for the transferred scope. This does not certify a
separate audit of their full histories. No old code/notebooks need resurrecting.
Retain only these conditional alternatives, with no implementation/job authority:

- Separate field and true-field-curve pretraining followed by supervised joint
  fine-tuning; a true→predicted field curriculum is distinct from the existing
  auxiliary true-field loss, detached gradients and frozen-source fits. Consider
  only if the current interface evidence warrants it. Prefer source sampling
  over silently treating interpolated fields as physical states.
- Relative-distance/edge attention bias is already deferred. LocalFieldGraph
  supplies initial edge geometry before FIELD attention, not attention-logit bias
  and not curve-stage graph processing. An attention-specific comparison must
  add evidence beyond that control; no bundled GraphGPS/edge-token/damage redesign
  and no use of true future damage in disorder-only inference.
- Output compression only for a demonstrated cost/capacity need: train-only PCA
  first, held-out physical-curve reconstruction including peaks, ductility/work
  and worst cases. Legacy PCA helpers exist; this is not a verified dual reducer
  contract. Full curves and the field intermediary remain baseline. No input PCA
  before node tokenisation; motif discovery is separate. Consider an autoencoder
  only if PCA is insufficient, after verifying fit/inverse/checkpoint support.

**Reconciled ownership (8 October):** Improve ML Accuracy has now read the full
available user/answer history of Dual Model and this brief, reconciling them
with current code and the user's later decisions. Dual Model can be archived;
do not replay its commands. Keep this file until its detailed pending checks
and incoming references are consolidated elsewhere. PROJECT_STATUS's current
entries supersede the historical scheduler states and older proposals below.
In particular, the recovery/property runs are completed and reviewed, soft/soft
peak loss is implemented with comparison jobs submitted, and sudden/non-affine
weighting is implemented and CPU-tested but not yet GPU-tested/submitted.
None of those changes establishes a final accuracy winner.

Retained checks beyond the experiment queue: notebook controls and saved-report
dashboards listed below; controlled perturbed-input/pin-boundary illustrations
and scale-aware boundary-tolerance validation against the producer; keeping the
human test guide aligned with newly added tests. The historical task-embedding
diagnostic also remains a proposal: inspect gradients/updates and compare zeroed
or swapped embeddings with fixed checkpoint weights and validation IDs. Low
sensitivity alone is not a bug because task context and heads also identify the
task. These checks are not newly executed or authorised as extra HPC jobs by
this reconciliation. Newer user decisions retain priority; cancelled raw-damage
maximisation/direct-curve/Pareto-search proposals remain cancelled. Sibling
briefs were not loaded during this retirement audit.

Continue the Dual Model work in this thread; do not reimplement completed work.
Read repo AGENTS, applicable nested instructions and p2 PROJECT_STATUS first.
Canonical checkout: `/Users/niccoloforte/Desktop/Code/NF-PhD-gitRepo`.
The matched-suite implementation is `37350e9`; its batch bootstrap correction
is `6a3aee1`, published to both Git hosts and deployed before submission.
Consult PROJECT_STATUS for the latest verified scheduler state. Never
change a staged study or checkpoint to match newly edited source.

Preserve legacy DATA/MODEL/Transformer workflows and historical checkpoints.
The dirty .gitignore, FieldOutputs/test_field_loss.py and three notebook changes
pre-date the handoff: do not revert or commit them casually. Keep source notebooks
in p2/code, artifacts in ignored data/, human examples in ignored p2/samples.
Avoid duplicate trainers, loss implementations, notebook copies and classes.
Architecture remains owned by resources/MLdual.py; field losses by MLfield.py;
metrics/loaders by MLmetrics.py. This thread now coordinates their joint work.

Current user decisions: leave both data/ and samples/ ignore policies unchanged;
cross-computer data organisation is deferred. Soft-versus-soft peak-location
supervision is agreed in principle, with hard location retained for reporting
and the existing peak-height term retained. This documentation update does not
alter submitted jobs. Accuracy has now implemented explicit `peak_target_mode="soft"`
in CombinedCurveLoss / `target_mode="soft"` in SoftPeakLocationLoss, with saved
serialization and tests. Historical missing/default modes remain hard-target.
Corrected-loss submission evidence is recorded below. The user separately authorised
a true-field-to-property HPC test for UT Strength/Ductility/WoF and FT K_JIC.

8 October update: remaining property30367256/30367272 and joint30326584/30326586
verified COMPLETED 0:0, downloaded with B3 and checksum-checked. Do not repeat.
The ten-run comparison now includes residual (.837071) and true-field auxiliary
(.796952); neither improves every output. Property oracle R² values are .566/.760/
.963/.930 for strength/ductility/WoF/K_JIC on793 validation specimens, not predicted
fields. Reports/plots are in samples/accuracy-continuation/.
Corrected-loss follow-up submitted from c29d77f: gate30813256; full30813257–30813260,
two frozen sources × combined_no_location/combined_soft. Existing MSE controls
have matching source/configuration hashes and splits and were not repeated.
Full requests240h/oneGPU/12CPU/90GB; afterok gate preserves B1 archive workflow.
Last check: gate waiting for resources, full jobs for dependency. Inspect these
before any further submission. No new GPU success or corrected-loss accuracy yet.
Bounded thickness association uses29 train/validation specimens, no locked test:
weak simple correlations but broad batch-bootstrap intervals and batch confounding.
The user now parks further thickness auditing: retain provenance, but do not infer
no effect, alter features or regenerate data. The ten-run leader is provisional;
no configuration has been promoted to a default. Sudden/non-affine weighting is
now opt-in in StructuredFieldLoss and wired into DUAL; CPU-tested, not GPU-run or
accuracy-validated. The isolated B4 `--variants baseline,sudden` pair preserves
the anchor and fixed selection, with no derivative penalties. Do not repeat the
old magnitude/activity weighting as a substitute. See the runner README for the
formula, scope and proposed matched three-seed DUAL-versus-independent decision.
That independent control is not implemented/submitted; the user has now accepted
the 2% field margin and expanded the suite as specified above. Core reports lead with training-mean-relative
RMSE skill and new local-motion NRMSE; archived scores stay unchanged.
Recheck peak jobs after SSH is restored, without duplicates.
The user's latest annotations explicitly permit reconciling Tokenisation, Damage
and Optimisation briefs; these have been updated. Physics/strain has a separate
feasibility brief. No chat dispatched or physics/damage model implemented.

### Reconciliation from the Accuracy chat

Read current PROJECT_STATUS before treating earlier job states below as live.
Eleven completed matched-suite archives (six original, recovery detach/winner_probe/
curve_predicted/curve_true/ft_region) have been downloaded with B3. Residual/
true_field were still running at 18:25 BST on 4 October. Do not resubmit.
Eight joint runs have identical 7137/793/881 IDs. Joint scores: baseline .863022,
crack_face .807569, local_graph .902120, partial .841817, private .827576,
late_frame .865194, detach .866487, ft_region .809125. This is one-seed validation,
not a final winner. FT-region is a fixed geometric prior, not sudden-motion weighting.
Fresh curve_true fit, true input: UT RMSE1.639/FT3039.064; curve_predicted fit,
predicted input: UT4.360/FT6678.813. Source substitution degrades either fit.
Two-curve fit selection scores must not be ranked with four-output joint scores.

Property probe now submitted from immutable source47fabb9 through B1:
preflight30367256 (64 pairs/one epoch/4h), full30367272 (450 maximum epochs,
early-stop52,240h), afterok gate with kill-on-invalid-dep. Both PENDING at18:25 BST;
GPU execution still unverified. Both use 1GPU/12CPU/90000MB, andrena/pilot_andrena.
Home snapshot: /data/home/exy053/p2/MULTI/FieldToProperty/Transformer/field-property-261004/source.
Archive: /data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/FieldToProperty/Transformer,
labels field-property-preflight-261004 and field-property-261004. Inspect existing
IDs before acting; no duplicate submissions. It trains on true fields, joins
archived properties by unique specimen ID and leaves test targets unevaluated.

User reiterates sudden sample-specific motion, not large displacement, as the
weighting goal. Existing activity weights do not meet that distinction; keep
them excluded. No CNN, strain targets, damage extraction or PINN term added.
Keep stage two without an explicit disorder/latent bypass; such a bypass would
change the interpretability of the field bottleneck and requires a separate test.
Fixed Ti/FCC/BC channels and per-DOF boundary flags remain unnecessary.

The user-authorised bounded thickness provenance follow-up is complete: 33 paired
ML specimens, 66 INPs, 11 batches. All input coordinates match exactly and field
initial coordinates match within 5e-7 mm, with matching stems/source basenames.
Early sampled batches6538720/6538745 vary thickness per specimen/task; all nine
later sampled batches share a single UT/FT thickness across their three inspected
specimens. Sampled1000/1001 straddle the contrast. Archived conditional assignment
is consistent with first-UT thickness reuse, but batch6892743's B1 initial1901
disagrees with its INPs1001–1100. Merged Windows field paths cannot establish exact
response ODB provenance. Do not extrapolate to all IDs, infer density/error effects,
add features or regenerate records. Thickness/property association checks are now
authorised; use the verified INP-to-ID rows and training/validation properties,
report batch confounding and limited power. A nonsignificant trend does not
establish irrelevance. No thickness feature, filtering or FEA change is authorised.
Evidence: samples/accuracy-continuation/thickness-provenance.md and hashed JSON.

Tokenisation handoff was reconciled in handoffs/TOKENIZATION_NEXT_STEPS.md with actual
correlation-weighted PCA/KMeans code and read-only vault notes. Other handoff
briefs were not loaded during that original reconciliation; the newly authorised
follow-up now adds the retained concepts below to their dedicated briefs.
For later dedicated-task reconciliation, retain: damage is edge-specific, actual
element-to-strut membership, any failed constituent marks a broken strut, absent
FT edges separate from later breakage, missing deleted outputs not "intact",
irreversibility and visual validation. Nodal broken-fraction is only a summary;
shared displacement/damage heads do not enforce consistency. Optimisation should
retain curves initially, investigate field localisation/progressive-participation
proxies and direct properties, validate against real mechanical objectives, and
avoid force/energy reconstruction. No extra exports or optimisation code here.

### Validation review status

The 28 checks in test_dual_contract.py are automated software regression tests,
not 28 research models or a record of researcher sign-off. The user has reviewed
and corrected samples, but there is no evidence that they personally audited
every test assertion. Do not claim that approval or that passing tests prove
the entire scientific formulation correct.

Relevant assertions encode the corrected sample contract: UT800/FT788 nodes,
nominal tip (120,95) not a node, 26 retained crack-face nodes, UT2319/FT2259
initial struts, FT degree5 at (100,100), removal of its three downward links and
retention of (120,90)-(120,100). Tests use self-contained fixtures, not the ignored
sample generators. They also cover software-only behaviour such as gradients,
checkpoint reload and HPO locks which a geometry illustration cannot validate.
Owner: **Improve ML Accuracy**. The current guide is now in
`samples/accuracy-continuation/test-guide.md`, covering all 31 current tests,
with measured mask/gradient counterexamples and linked physical examples.
All 31 passed locally. Preserve the following maintenance requirements for that
human-readable guide
in `p2-DisorderML/samples/` covering EACH test in `test_dual_contract.py`, including
software checks, not just geometry assertions. Inventory the current tests by
name; 28 is the previously verified count, not a permanent expected count.
For every test show what it checks, why, how its inputs/manipulation/assertions
work, expected versus observed outcomes/tolerances, what failure it detects and
what it does NOT establish. Include small numerical examples where useful, but
prioritise labelled diagrams/plots and concise explanations over raw tables/code.
Examples: masked-node perturbation before/after; gradient paths that should be
active/blocked; predictions before/after checkpoint reload; a timeline for lock
and recovery tests. Related tests may share a figure, but each needs an explicit
entry. Link scientific assertions to corrected samples/producer evidence and a
deliberately broken counterexample that should fail. Reuse existing illustrations
where accurate; do not create notebook copies in data/ or make automated tests
depend on ignored samples. Request review of uncertain physical expectations,
not approval inferred from successful execution. Existing INP evidence is one
paired specimen, not an exhaustive archive audit. Preserve the documented
pin-boundary tolerance caveat. Keep it synchronized when tests change; it is not
an exhaustive archive or interactive-browser validation.

## Authorised matched suite and continuation

The existing runner now also implements late_frame, ft_region, winner_probe,
curve_predicted and curve_true. B4_Dual-experiments.sh submits thirteen isolated
variants through B1 from an immutable home snapshot, gated by its all-mode GPU
preflight. See ../HPC/DualOutputs/README.md for exact CLI, source/selection semantics,
weights and paths; PROJECT_STATUS is the submission record. Do not resubmit
existing labels or confuse a dependency queue entry with completed training.

Submitted suite: `dual-compare-260930-r1`, immutable source `6a3aee1`.
Preflight **29601762 completed 0:0 in 9m11s**, with all thirteen tiny archives
verified. Live audit on 4 October found no queued/running jobs: six full runs
completed (baseline, partial, private, crack_face, local_graph, late_frame).
Residual, ft_region, true_field, detach and winner_probe failed BEFORE Python
on sbg10: "Unable to locate a modulefile for 'miniforge'". The two curve fits
were cancelled through their dependency on winner_probe. Six successful archives
contain model.json/model.mdl, predictions, metrics and logs. No full-run accuracy
comparison was performed in that storage audit. Empty failed scratch is not
scientific evidence against those variants. Diagnose/validate the environment
before requesting scoped replacement submissions; never rerun successful jobs
by blindly submitting the whole suite. No storage-cleanup task is assigned here.
On 4 October the user authorised fixing this failure and resubmitting exactly
the five failed plus two cancelled experiments. Recovery is now submitted as
`dual-recovery-261004`, immutable source `0b9215e`: preflight 30326583;
residual 30326584, ft_region 30326585, true_field 30326586, detach 30326587,
winner_probe 30326588, curve_predicted 30326589 and curve_true 30326590.
All depend on the preflight; both curve fits also depend on winner_probe.
sbg10 is administratively drained with reason `modules`; the site bootstrap
works on login-01 and sbg23: after reconnection, preflight 30326583 was verified
COMPLETED 0:0 in 9m05s with all thirteen archives, and winner_probe 30326588
COMPLETED 0:0 in 2m43s with its field-source diagnostics. At approximately
15:47 BST the six training jobs were RUNNING with advancing epochs. No accuracy
comparison performed. No model/loss change was needed for this
node-specific failure. ../HPC/DualOutputs/README.md owns recovery IDs, paths and evidence;
consult PROJECT_STATUS before acting. Inspect existing jobs, do not resubmit.

| Variant | Job | Isolated question |
| --- | --- | --- |
| baseline | 29601763 | Fresh matched fully shared reference |
| partial | 29601764 | Private final field block |
| private | 29601765 | All field encoder blocks private |
| crack_face | 29601766 | Static crack-face input only |
| local_graph | 29601767 | Shared local messages with task-specific initial graphs |
| residual | 29601768 | Train-mean residuals and variation scaling |
| late_frame | 29601769 | Fixed positive late-load emphasis |
| ft_region | 29601770 | A-priori FT crack-tip neighbourhood emphasis |
| true_field | 29601771 | Auxiliary true-field curve supervision in joint training |
| detach | 29601772 | Stop curve gradients into field predictions |
| winner_probe | 29601773 | Frozen HPO winner, field-input substitution only |
| curve_predicted | 29601774 | Fresh curve stage trained on frozen predicted fields |
| curve_true | 29601775 | Fresh curve stage trained on true fields |

Preflight requests four hours; all thirteen dependents request 240 hours, one
GPU, 12 CPUs and 90,000 MB on andrena/pilot_andrena. The first attempt
`dual-compare-260930` failed before Python: module was absent in its batch
environment (29598744), so dependents 29598745–29598757 cancelled automatically.
The correction initialises site modules and exports HOSTNAME for their logging;
it did not prevent the distinct miniforge-availability failures above.

Use the HPO winner configuration (field THREE blocks, curve FOUR blocks), all
paired data, seed42/split42, 450 maximum epochs/early stopping. Baseline, partial,
private, crack-face, graph, residual, true-field auxiliary, detach, late-frame
and FT-region runs are independently compared. Old activity localisation is
excluded; its definition does not answer the user's local-anomaly question.
Repeat promising variants across seeds before interpreting sharing/capacity.

### Follow-up experiment priorities (not yet submitted)

First inspect the already-submitted suite. Subsequent candidates, owned by
Improve ML Accuracy and requiring explicit submission scope, are:

1. Repeat baseline and promising variants over TRAINING seeds (initialisation,
   minibatch shuffling, dropout), initially holding split seed42 and specimen IDs
   fixed. Use the same seed set across variants and report dispersion. Changing
   the dataset/split seed is a separate generalisation study, not this control.
2. Matched curve-loss comparison: MSE; combined loss with peak-location weight
   zero; combined loss with corrected soft/soft peak location. Keep other
   combined-loss terms/configuration fixed when isolating the peak term, retain
   peak-height supervision when enabled, and report hard location/height/work
   errors as well as curve MSE. Existing CombinedCurveLoss already includes
   PeakStressLoss via peak_weight. Distinguish the later optional field-source
   true/predicted × loss comparison from the current source-only suite.
   The explicit soft/soft correction is now implemented and regression-tested;
   select its saved option for this comparison. Preserve historical checkpoint
   semantics and do not modify the original suite. The user has now authorised
   this follow-up; jobs30813256–30813260 are submitted. Inspect before resubmitting.
3. Local non-affine displacement weighting, ONLY after the affine/local-separation
   examples and boundary/conditioning checks described below are validated.
4. Curve-stage sharing controls with fixed field sources if the interface
   results warrant them; do not change both stages simultaneously.
5. Capacity-matched field-sharing comparisons to distinguish extra parameters
   from the effect of sharing. This complements, not replaces, matched seeds.

Damage supervision still requires validated strut labels. Temporal decoding and
graph-relative attention biases remain deferred; no extra HPO or design search
is implicitly requested by this list.

winner_probe uses frozen winner weights with true-field substitution, no fit.
The two curve_* runs instead train fresh identically seeded shared UT/FT curve
networks, freezing the winner's field generator and switching only its source:
predicted fields versus true fields. Select on two curve scores; four-output
selection is unchanged elsewhere. Save both source evaluations for each curve
checkpoint. True-field results are oracle diagnostics, not disorder-only
deployment results. In-sample field predictions on training cases are not
cross-fitted; validation remains held out. No curve-depth HPO is implied.

Late-frame weights: raw 1+(t−first)/(last−first), then per-specimen valid-mean-one;
same across tasks/components, no dynamic error feedback. FT region: raw2 inside,
raw1 outside, UT1, same normalisation. Validated A1 dN=.2 reference box selects
125 retained nodes (x96.6..173.6,y54..136 in producer cell-size10 units), not an
Abaqus edge set. Global/inside/outside errors saved for every suite member.
Component weighting is parked: present scale-adjusted errors do not justify a
blanket U2 preference. This does not establish identical component difficulty.

Proposed affine-departure weighting is NOT implemented in this suite. For each
node/frame use task-specific initial neighbours and initial disordered edge
vectors. Fit a local displacement gradient A to (Uj−Ui)/length against
(xj−xi)/length, then calculate the RMS fit residual. Uniform translation and
affine displacement fields have zero residual, unlike current activity weights.
Validate neighbour rank/conditioning, masked targets and boundaries; at a node
with too few independent neighbours the residual is not informative. Fit a
robust positive residual scale on training data only, floor it, and use
1+g*q/(1+q), g initially1, normalised over valid values, detached from gradients.
This would weight the existing displacement error, not penalise true fracture
gradients into smoothness. Elastic non-affinity can also trigger it: it is a
kinematic localisation proxy, not a plasticity/damage label. Show affine and
local-separation samples before implementation. Temporal slope-change weighting
would be a different isolated test, not an implicit part of this proposal.
This is an adaptation of local best-affine-residual reasoning, not a validated
FCC damage classifier; background: Falk & Langer (1998),
https://doi.org/10.1103/PhysRevE.57.7192.

## Earlier implementation and evidence

The existing HPC/DualOutputs/A0-HPC-Dual-test.py accepts --experiment:
baseline, crack_face, local_graph, partial, private, true_field, detach, residual,
localization. Its --base-model-json anchors fresh experiments to saved dual
architecture/training/loss configuration, not saved weights. Use fixed split
seed 42 independently of training seed; record parameter counts and split hash.
All nine modes passed full-800-node synthetic train/save/diagnostic/reload checks;
23 dual tests and 14 field-loss tests passed. This is not GPU/accuracy validation.

- Baseline has shared field blocks and shared curve blocks, task embeddings,
  masks and separate small output heads. Task streams do not attend across tasks.
- Partial defaults to shared early field blocks and a private last field block.
- Private separates ALL FIELD ENCODER BLOCKS, but not projections or curve stage.
  It is not complete task independence. These tests isolate an intervention,
  not the cause of every downstream curve error. At winner depth 3, field
  parameters are 5.98M / 7.76M / 11.31M; capacity is a confound.
- LocalFieldGraph is one shared message MLP consuming distinct UT/FT initial
  edge lists and specimen-disordered edge vectors/lengths, before global
  attention. No additional tokens, hard adjacency-only attention or evolving
  fracture. Reuse reference_field_edges. Counts UT2319/FT2259; FT degree5 at
  (100,100); three downward connections removed, (120,90)-(120,100) retained.
- crack_face reuses the existing opt-in initial_crack_face feature: 26 retained
  FT nodes, UT zero. Compare baseline/crack-only/graph-only separately first.
- true_field adds supervision on true fields through the SAME curve network
  (default auxiliary coefficient .5). detach independently blocks curve-to-field
  gradients. Neither is a third model. Default joint gradients are unchanged.
- residual is optional: each node/frame/component learns (U−training mean)/scale,
  with per-location training variation floored at .1 times the original pooled
  frame/component scale. One saved affine inverse reconstructs physical U, and
  the curve input uses the same residual coordinates. No existing run was changed.
- localization reuses current bounded target-activity displacement weighting,
  with spatial/temporal penalty coefficients zero. See limitations below before
  interpreting this as anomaly or crack-region weighting.
- Legacy graph_semantics="fcc_initial_v1" explicitly opts into the corrected
  FT cut/bidirectional graph; historical defaults/loading remain unchanged.
  Corrected legacy GNN training requires native FT788 (unmasked legacy pooling),
  while the dual graph handles masked canonical800. Transformers are unaffected.

## Completed runs and what their evidence says

All requested transfers finished through B3. Full HPO is local at
data/MULTI/Dual/Transformer/HPO/dual-joint-hpo1, including DB and checkpoints.
Job26375307 completed in 9d14h05m: 45 complete,10 pruned,1 waiting retry; NOT 200
evaluations. Winner46, epoch58, fixed score .82661694 (earlier37: .83263056).
Validation RMSE skill vs training mean: UTfield4.20%, FTfield16.55%, UTcurve4.06%,
FTcurve12.14%. Better joint score does not mean all branches improved.
Full archive/missing-older-HPO inventory: samples/hpc-archive-review.md.
Shared ML-HPOpostProcess.ipynb supports dual; DualPostProcessing uses best/.

Twelve completed independent/dual loss trials and three preflights are local.
Names: loss-{UT|FT|DUAL}-{baseline|spatial|temporal|both}-260917, under their
UT/FT Field/Transformer or MULTI/Dual/Transformer parents. Do not resubmit them.
Independent field spatial RMSE is UT .96039 / FT .10852 versus final dual HPO
1.03037 / .12308. Independent validation counts799/804 versus dual793 and
different capacities/training mean this is not a controlled superiority claim.
The independent routes remain viable fallbacks, not abandoned code.

### Same-checkpoint true-field substitution already exists for four loss runs

Physical curve RMSE, predicted-field input → true-field input, same793 specimens:

| Run | UT | FT |
|---|---:|---:|
| baseline | 4.501 → 4.761 | 6672.490 → 7921.395 |
| spatial | 4.412 → 4.689 | 7538.217 → 7674.865 |
| temporal | 4.620 → 3.928 | 6493.434 → 8018.200 |
| both | 4.362 → 4.098 | 7031.547 → 7719.865 |

These are saved true_field_curves.npz, not newly trained oracles. FT worsens for
all substitutions; UT improves for temporal/both. Distribution mismatch or
co-adaptation is plausible, not proved. Do not infer that inaccurate fields are
physically preferable or that field errors cannot cause curve errors.
Trial1 still lacks this saved diagnostic. The authorised winner_probe generates
the winner comparison with reconstructed original DUAL_DATA/context;
its model.json stores reference features, not every sample's pin memberships.
Do not silently replace sample-specific context with the reference flags.

Review end-to-end curves, same-checkpoint substitutions, and true-field-trained
curve baselines separately. Field sharing alone cannot locate curve-stage
interference. Curve-only sharing controls with fixed field sources may be a
subsequent experiment; avoid simultaneously changing both stages before these
diagnostics. Keep matched samples/transforms and distinguish frozen versus
retrained downstream networks explicitly.

### Loss provenance: correct the user's understandable assumption

Final HPO winner fields use MaskedFieldMSELoss; curves use MSELoss. Trial1 also
uses MSE. HPO allowed curve mse OR CombinedCurveLoss; winner selected mse by the
fixed selection score. The later custom spatial/temporal field losses were NOT
in that HPO. They ran in separate loss trials. Do not claim the final HPO used
all custom losses or that MSE is universally best; studies/seeds are limited.
New architecture runs anchored to this winner inherit those loss choices.
Explicitly state active loss classes and nonzero coefficients before future jobs.

The saved study contains 34 completed MSE trials (best fixed score .82661694)
and 11 completed combined-loss trials (best .86242156). This is not a matched
loss ablation: architecture, coefficients and optimisation also vary. The fixed
score rewards baseline-relative MSE, not peak/work/ranking quality. Combined loss
also changes per-specimen scaling (target-range normalisation versus train-fitted
task scaling), as well as adding derivative/feature objectives. Predicted-field
error could interact with these objectives, but is not established as the cause.
The submitted source tests do NOT isolate loss choice. A later matched
true/predicted-source × MSE/corrected-combined comparison could do so.

## Latest user clarifications and the next decisions

The earlier sample-review brief remains applicable: spatial/temporal supervision
already exists, so do not present it as a new method. Crack-face-only is a small
interpretable boundary-cue ablation, not a promised large gain: coordinates already
encode much of the location. Do not bundle degree, orientation counts and incident
length statistics into a new feature set; degree is only a possible later comparator.
Architecture ownership has moved from Dual Model into this Accuracy continuation.
Keep architectural and loss changes separate; inspect all four outputs, regional
errors, signed displacement jumps, true/predicted-field curve behaviour and seed
stability on validation, not repeated locked-test inspection.

Review the controlled input gallery in samples/: show actual perturbed coordinates
and offsets, label the deliberately moved node, and zoom the relevant pin boundary.
Keep reference coordinates invariant across specimens; displayed sample geometry
must not conceal the perturbation. Define/test a scale-aware circle-boundary
tolerance against the producer's intended inclusion rule, using inside, exactly
on-boundary and outside cases under rescaling/translation. Do not use a tolerance
large enough to erase genuine disorder-induced pin-membership changes. This issue
is documented, not already fixed. Damage and strain targets need separate explicit
data contracts; do not relabel initial crack-face membership as evolving damage or
perform additional Abaqus exports through this brief. The old draft's "no new
jobs" sentence is superseded only by explicitly authorised submissions.

1. **Weighting needs correction in interpretation, not a silent code change.**
   Current q combines mean squared scaled neighbour displacement differences
   and adjacent temporal first differences; then sqrt/component averaging,
   bounded 1+gain*q/(1+q), and specimen-mean-one normalisation. It is not |U|,
   but smooth affine loading has nonzero gradients and upper nodes can have
   larger temporal increments. It is NOT sudden-change/acceleration or local-
   anomaly detection. The user's concern is valid. Compare illustrations of
   rigid translation, affine loading and local separation before proposing a
   new weight. Consider deviations from a local affine motion fit/training-only
   expected gradients, and changes in load-normalised temporal slope. A raw
   neighbour mean is boundary-biased; avoid claiming it solves the issue.
2. **Component and frame weights:** user wants evidence-led U2 and late-frame
   emphasis. Final HPO physical U1/U2 RMSE: UT .7869/1.2264, FT .0639/.1619.
   Divide each error by its saved training frame/component scale: UT .3871/.3363,
   FT .0503/.0393. U2 has larger physical error but U1 larger standardised error.
   These are pooled scale comparisons, not location-wise specimen-variation skill.
   Standardised first→last frame RMSE grows UT .0898→.5381, FT .0257→.0716.
   A modest late-frame experiment is justified; blanket U2 priority depends on
   mechanical objectives, not raw magnitude alone. Quantify per-component/frame
   baseline skill and curve sensitivity; retain positive weights and fixed
   unweighted validation reporting. Late-frame weighting is now part of the
   authorised suite above; component weighting is parked.
3. **FT region prior:** user means an a-priori crack-tip neighbourhood, NOT the
   actual realised crack path. This is a reasonable independent experiment.
   FCC A1 CrRegMESH = [xCrE−2.1a,xCrE+5.6a,H/2−4.1a,H/2+4.1a], a=unitCellSize;
   CrRegSTAT is a different smaller box. Meshing selects edges in the former.
   Validate task units, xCrE vs nominal tip, initial/reference coordinates and
   node/edge membership before reuse. Show a labelled mask; choose modest
   FT-only positive regional weights and retain outside-region supervision.
   The suite now implements that bounded prior with whole-field/inside/outside
   and curve reporting; it does not classify damage.
4. **Damage:** local graph gives useful endpoints/representations for a future
   strut head, but does not itself predict damage. A head can read endpoint
   embeddings plus edge geometry to predict strut histories without new
   Transformer tokens. Handle endpoint-order invariance, masks/absent struts,
   loss imbalance and valid frame alignment. Do not remove edges based on true
   future damage during inference. Validated labels must come from the separate Damage variable processing task FIRST.
5. **Residual model:** optional experiment, NOT a change to all predictions.
   Mean is task-specific, node-specific, frame-specific, component-specific and
   training-only. Compare physical-unit errors/diversity/curves; floor small
   variation. Work through the numerical example in samples before running.
6. **Design diagnostics are not HPO:** for a fixed model, ask whether predicted
   good designs are actually good using held-out known FEA cases. Top10 recovery,
   objective ranks/regret/false feasibility do not optimise hyperparameters,
   generate designs or train the model. Selection rule remains unchanged.

## Deferred ideas: retain explicitly, do not silently launch

- Temporal/load-conditioned decoder: no need to average201 curve points to20
  fields. First audit whether relevant events are observed; interpolation does
  not recover missing event physics. Higher-frequency export needs approval.
- Graph-relative attention bias: later comparison, separate from local messages;
  keep long-range attention. No hard adjacency-only Transformer.
- GNNExplainer: defer until graph benefit/model reliability are established;
  no separate explainer handoff was sent.
- NN derivatives/saliency: diagnostic first, not model-derived importance weights
  that may reinforce current blind spots. No sensitivity-based weighting added.
- Curve integrals: current UT diagnostic uses each curve's own postpeak1%
  crossing and reports missing crossings. FT full-domain work is a proxy, not
  fracture toughness or initiation work. Validate physical cutoff definition
  before changing objectives. A weighted MOO sum needs fixed reference/train-
  only normalisation and several trade-off weights while retaining both axes;
  it is not equivalent to recovering an entire Pareto front.
- Soft peak-location mismatch: current SoftPeakLocationLoss compares softmax-
  weighted predicted x with hard target argmax x. Exact prediction may have
  nonzero loss. Illustrative x=(0,1,2),y=(0,1,.99),beta20: hard1,soft1.450,
  normalized squared penalty .05066. This is not curve smoothing. Consider an
  explicitly versioned soft-versus-soft term (now agreed in principle) with the same scale/beta and
  exact-match zero/finite-gradient tests; retain hard peak-position diagnostics
  since soft centroid is not always the physical maximum. Disable only this
  term in a new combined-loss trial if necessary; do not alter archived losses
  or assume higher beta alone fixes consistency. Final winner MSE is unaffected.
  Precisely: leave the prediction's soft centroid unchanged, replace only the
  hard target argmax by the target's soft centroid using the SAME beta and
  target amplitude scale. Equal curves then produce equal centroids/zero loss.
  Keep all other combined-loss terms fixed in an isolated comparison; test
  exact equality, shifted peaks, broad/double peaks and finite gradients.
  No corrected peak-loss implementation or training is included in this suite.
- Drop raw sum-of-element-failure maximisation, standalone direct-curve benchmark
  and standalone Pareto-specimen search per the user's earlier cancellation.
  Retain design-screening diagnostics here; curve/field-only design objectives belong to the separate Surrogate optimisation task.

## Review material and continuation order

### Collect and visualise this suite

1. Read the home manifest and scheduler status before downloading. Launch and
   logs: `/data/home/exy053/p2/MULTI/Dual/Transformer/dual-compare-260930-r1/`.
   `jobs.tsv` is authoritative; `source/` is the immutable submission snapshot,
   including the earlier test/trial/loss scripts. The maintained scripts remain
   in the home Git checkout's `p2-DisorderML/HPC/` directories.
2. B1 stages to `/gpfs/scratch/exy053/<job-id>`, writes `mlruns/`, copies outputs
   and logs into archive `p2/MULTI/Dual/Transformer/<suite>-<variant>`, then
   removes scratch only after success. Failed scratch is retained. Check exit
   status AND archived model/results/logs; a queue entry alone is not a pass.
3. Download each completed variant with the existing macOS script, from repo
   root, for example:

   ```bash
   bash p2-DisorderML/HPC/B3_ML-transfer-mac.sh MULTI Dual Transformer dual-compare-260930-r1-baseline
   ```

   This creates only artifacts under ignored `data/`. Do not copy notebooks or
   processing scripts there. The preflight's thirteen tiny outputs have names
   `<suite>-preflight-<variant>` and demonstrate execution, not predictive skill.
4. In maintained `code/ML-DualOutputs.ipynb`, set `LOAD_RUN` to that local run,
   `LOAD_MODEL=True`, `LOAD_DATA=False`, `RUN_HPO=False`, `RUN_TRAINING=False`.
   This inspects a checkpoint without raw MLdata or accidental retraining.
   Reconstructing source-conditioned predictions needs matching DUAL_DATA;
   plotting existing saved arrays does not.
5. In `code/ML-DualPostProcessing.ipynb`, change `RUN_DIR` to the same run and
   rerun from the top. Both UT/FT curve and field dashboards, component/frame
   errors, diversity, paired errors, live field viewers and log-loss plots
   already consume the saved contract. Keep one maintained notebook in code/.
6. Its source-comparison helper uses explicit `*_predicted_field_curve_sample_metrics.csv`
   when available and `*_true_field_curve_sample_metrics.csv`, joined by ID.
   For curve_true, standard curve outputs also use true fields: they must NOT
   be labelled predicted-field inference. Missing explicit predicted tables in
   this case raise an error. For ordinary runs, standard predictions are the
   valid fallback. winner_probe preserves the source winner's history; it has
   no NEW training epochs. Do not interpret that inherited loss plot as a refit.
7. Read `results/design_diagnostics.md` for readable ranking/top10/regret
   reports; the JSON is for cross-run aggregation. These are implemented in
   MLmetrics and the experiment runner, not a dedicated existing notebook
   dashboard or a retroactive addition to old runs. `ft_region_metrics.json`
   provides inside/outside/global errors for the FT-region test. Interpret
   these alongside the existing spatial viewer, not raw weighted loss alone.
8. Compare fixed validation scores only within comparable training modes:
   four-output joint fits versus baseline; two-curve fits against each other.
   Compare physical per-task metrics and train-mean skill across all modes,
   joined by identical sample IDs/splits. Inspect oracle/deployable sources,
   epochs and parameter counts. Use ML-HPOpostProcess only for the original
   Optuna study: this fixed ablation suite is NOT another HPO database.

### Notebook backlog — owned by Improve ML Accuracy

Existing DualPostProcessing supports saved UT/FT curve/field diagnostics, live
field viewers, log-scale losses and same-checkpoint field-source comparisons.
Shared ML-HPOpostProcess supports dual Optuna studies. These capabilities do NOT
mean that every new experiment switch or diagnostic already has notebook UI.

Pending targeted work, preserving current cells/user edits and familiar syntax:

- Expose design ranking/top-recovery/false-elite/regret from the saved reports;
  distinguish UT cutoff work from the FT full-domain work proxy.
- Display dedicated FT inside/outside/global region comparisons from saved
  ft_region_metrics.json, alongside the existing spatial viewer.
- Expose fresh-run configuration for implemented sharing, graph/crack-face,
  residual, interface and fixed-weight options in DualOutputs, through shared
  framework methods rather than duplicate HPC subprocess/training logic.
- Provide a matched suite-level comparison of all four outputs, source mode,
  splits, seeds, parameter counts and epochs. Four-output joint selection scores
  and two-curve fit scores are not interchangeable.

Prefer existing notebooks for single-run controls/diagnostics. A small dedicated
ablation-comparison notebook under p2/code may be justified for repeated multi-run
review; propose it before creation, and do not create one notebook per variant
or any notebooks in data/. Broad notebook rewrites require a scoped plan. Missing
new artifacts must be clearly reported while older runs remain reviewable.
All loss axes remain logarithmic; worked feature/weight calculations stay in
samples/. This backlog is documented, not already implemented by this handoff.

### Earlier visual explanations

Generated on29Sep, ignored under p2/samples: dual-sharing-{baseline,partial,private}.png,
dual-peak-loss-explanation.png, dual-field-difficulty.png,
dual-clarification-evidence.md; reproducible via explain_dual_sharing.py.
Existing dual-experiment-examples.md and hpc-archive-review.md remain useful.
The user's reading-position question concerns Improve ML Accuracy, not this
thread. Its latest annotation references the earlier implementation report
beginning "The implementation and slides are complete for this phase". Resume
conceptual reading with the subsequent response beginning "I checked the current
code. No changes made. Two corrections to my earlier explanation are important";
then the status response "All twelve trials completed successfully". Review the
preceding implementation report too if only its job table has been read. Actual
read status cannot be observed; these are verified message anchors only.
The user will finish reading/annotating from the quoted correction response and
then paste this handoff into Improve ML Accuracy. Await that user-led transfer;
do not automatically dispatch it or treat the unread material as approved.

Next address the recorded incomplete experiments and compare matched
validation results. Keep baseline/crack-only/graph-only and encoder controls
separate. Use the same home→scratch→archive workflow, complete diagnostics,
source provenance and up-to-date Git. Keep locked test unused for development.
Retain the full201 curve intermediary and separate-model fallback. Do not let
this handoff expand into all deferred projects automatically.
