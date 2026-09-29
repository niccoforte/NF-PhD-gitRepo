# Consolidated handoff to Improve ML Accuracy

This is a copy/paste task brief, not an executable script. The user will attach
it to their annotations in the existing **Improve ML Accuracy** thread
(`01a08760-1e96-7cd3-893d-7d02ead914d7`). It has NOT been sent automatically.
No new jobs, ODB exports or optimisation searches are authorised by this handoff.
The later user message/annotations determine what to execute next.

## Start here: ownership and current state

Continue the Dual Model work in this thread; do not reimplement completed work.
Read repo AGENTS, applicable nested instructions and p2 PROJECT_STATUS first.
Canonical checkout: `/Users/niccoloforte/Desktop/Code/NF-PhD-gitRepo`.
Architecture/loss experiment implementation is commit `017be58`, published to
QMUL and GitHub.com. HPC checkout synchronisation was still pending after the
SSH master broke; inspect/reconcile before any authorised deployment. Never
change a staged study or checkpoint to match newly edited source.

Preserve legacy DATA/MODEL/Transformer workflows and historical checkpoints.
The dirty .gitignore, FieldOutputs/test_field_loss.py and three notebook changes
pre-date the handoff: do not revert or commit them casually. Keep source notebooks
in p2/code, artifacts in ignored data/, human examples in ignored p2/samples.
Avoid duplicate trainers, loss implementations, notebook copies and classes.
Architecture remains owned by resources/MLdual.py; field losses by MLfield.py;
metrics/loaders by MLmetrics.py. This thread now coordinates their joint work.

## Already implemented, but NOT yet trained on real data

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
Trial1 and final HPO winner do NOT have this saved diagnostic. Generate the
winner comparison with reconstructed original DUAL_DATA/context when authorised;
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

## Latest user clarifications and the next decisions

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
   unweighted validation reporting. No new weighting was implemented on29Sep.
3. **FT region prior:** user means an a-priori crack-tip neighbourhood, NOT the
   actual realised crack path. This is a reasonable independent experiment.
   FCC A1 CrRegMESH = [xCrE−2.1a,xCrE+5.6a,H/2−4.1a,H/2+4.1a], a=unitCellSize;
   CrRegSTAT is a different smaller box. Meshing selects edges in the former.
   Validate task units, xCrE vs nominal tip, initial/reference coordinates and
   node/edge membership before reuse. Show a labelled mask; choose modest
   FT-only positive regional weights and retain outside-region supervision.
   Report whole-field, inside/outside and curve errors. Not implemented yet.
4. **Damage:** local graph gives useful endpoints/representations for a future
   strut head, but does not itself predict damage. A head can read endpoint
   embeddings plus edge geometry to predict strut histories without new
   Transformer tokens. Handle endpoint-order invariance, masks/absent struts,
   loss imbalance and valid frame alignment. Do not remove edges based on true
   future damage during inference. Follow the extraction brief below FIRST.
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
  explicitly versioned soft-versus-soft term with the same scale/beta and
  exact-match zero/finite-gradient tests; retain hard peak-position diagnostics
  since soft centroid is not always the physical maximum. Disable only this
  term in a new combined-loss trial if necessary; do not alter archived losses
  or assume higher beta alone fixes consistency. Final winner MSE is unaffected.
- Drop raw sum-of-element-failure maximisation, standalone direct-curve benchmark
  and standalone Pareto-specimen search per the user's earlier cancellation.
  Retain design-screening diagnostics and field-only feasibility below.

## Review material and continuation order

Generated on29Sep, ignored under p2/samples: dual-sharing-{baseline,partial,private}.png,
dual-peak-loss-explanation.png, dual-field-difficulty.png,
dual-clarification-evidence.md; reproducible via explain_dual_sharing.py.
Existing dual-experiment-examples.md and hpc-archive-review.md remain useful.
All thirteen annotations on29Sep refer to the response opening "The implementation
and downloads are complete." The preceding substantive recommendation began
"I would now test partial sharing and a no-sharing control"; a short pending-
work summary lay between them. Reading status is not known: use these anchors,
not claims that the user has read a response.

First answer the user's appended annotations and agree on weighting/diagnostic
definitions. Then inspect the winner interface and completed loss comparisons;
choose a small isolated experiment sequence, preserving baseline/crack-only/
graph-only comparison and encoder controls. New jobs require explicit permission
and a B1 GPU preflight, same home→scratch→archive workflow, complete diagnostics,
source provenance and up-to-date Git. Keep locked test unused for development.
Retain the full201 curve intermediary and separate-model fallback. Do not let
this handoff expand into all deferred projects automatically.

---

The following two briefs are retained here, not separately dispatched. Coordinate
them within Improve ML Accuracy only when the user chooses to begin them.

## Damage export: element → strut → loading history

Continue p1 damage processing for the paired FCC UT/FT simulations in the p1 HPC
archive. First inspect a small authorised sample and the actual ODB field/history
names to establish which damage variable is available (do not assume SDEG, STATUS
or a particular SDV are interchangeable). Preserve ODBs and existing displacement
exports. Use Abaqus Python and the existing p1 scratch/archive workflow; do not
upgrade or scan thousands of ODBs by default.

The output entities are **struts, not lattice nodes**. Reconstruct an explicit
element-label-to-parent-strut map from the original INP connectivity/producer
metadata, with instance names and beam subdivisions handled correctly. Do not use
consecutive labels or an assumed fixed number of elements as the mapping rule.
The researcher expects five or ten elements per strut, but the prior audit found
variable subdivision counts: measure and report them instead of assuming either.

At each recorded frame, if any constituent element has verified failure D=1,
assign D=1 to its whole parent strut as requested. For partial damage, propose
and obtain approval for the aggregation rule (e.g. maximum), while retaining the
underlying element values for audit. Handle deleted elements/missing output:
missing does not automatically mean undamaged or failed. Identify failure through
the validated damage/status convention and retain absorbing failure only when
the simulation definition supports it. Distinguish first failure from average
strut degradation. Do not maximise a raw sum of element failures.

Produce small, labelled examples under the existing `p2-DisorderML/samples/`:
an intact strut, a partially damaged strut, and a strut with one failed element;
show endpoints, strut identity, constituent element labels, raw values, aggregation
arithmetic and the resulting strut value. Include a crack-adjacent FT example and
different subdivision counts. Plot a highlighted parent strut and its elements
at several frames, and D against explicitly defined global loading/strain.
Ask for researcher confirmation before scaling up.

Propose a versioned edge/strut NPZ contract with stable specimen IDs, endpoint
mapping, frame/load values, valid masks, source provenance and variable names.
Keep it separate from the node-displacement arrays; never broadcast strut labels
onto nodes silently. No architecture change is part of this task. The dual local
graph uses **initial** connectivity and is not already a damage-evolution model.

## Field-only optimisation: feasibility and validation first

Investigate whether displacement-derived field descriptors can supplement the
existing curve objectives for FCC disorder design. Do not remove the curve stage,
launch an optimisation search, or claim displacement activity is damage.

Use aligned train/validation specimens and existing diagnostics to compare
candidate descriptors (distributed displacement differences, localisation,
deformation extent) against the established UT/FT mechanical quantities while
controlling for strength and loading level. Report counterexamples where large
motion corresponds to weak or already failed structures. Keep the locked test
set untouched. This is not the cancelled standalone Pareto-specimen search.

Provide readable worked examples and maps in `p2-DisorderML/samples/`, reuse
`resources/MLfield.py` and `MLmetrics.py`, and avoid duplicate loss machinery.
Only propose a field-based objective if the evidence links it to the desired
mechanics; then ask for approval of its definition/constraints before optimisation
or FEA interventions. State explicitly if no reliable field proxy is found.
