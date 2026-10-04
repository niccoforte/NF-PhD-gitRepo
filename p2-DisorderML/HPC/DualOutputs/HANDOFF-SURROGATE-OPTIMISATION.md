# Surrogate optimisation — curve and field-only task brief

This separate task owns FCC disorder DESIGN optimisation using UT/FT surrogates,
not model HPO, architecture or training-loss development. It includes both curve
objectives and field-only alternatives; do not send it back to Improve ML Accuracy.
Read applicable repo/resources/p2 code instructions and project status first, then
inspect the existing optimisation code and relevant research notes on demand.
Do not load sibling handoffs or the full previous chat by default.

## Starting point and boundaries

The dual surrogate has one geometry-to-field stage followed by a field-to-curve
stage, both with task-conditioned UT/FT streams. Canonical UT inputs feed both
branches; FT has absent-node masks. Preserve the working independent DATA/MODEL
fallback. Existing checkpoints and the four-output training contract are not to
be changed by this task. Optimisation should consume a frozen, validated model.

Accuracy is not yet established for design search: the final dual HPO winner46
improved validation RMSE over the train-mean predictor by about 4.20% UT field,
16.55% FT field, 4.06% UT curve and 12.14% FT curve. These are reconstruction
scores, not proof that exceptional designs are ranked correctly. Other ablations
are under review by Improve ML Accuracy; ask for its verified model/configuration
and validation manifest instead of assuming the most recent checkpoint is best.

First deliver an objective/constraint proposal and small known-FEA examples.
Do not launch a design search, retrain models, process bulk ODBs, or submit FEA
without explicit approval of that stage. Prior standalone Pareto searching and
raw sum-of-element-failure maximisation were cancelled; this new brief authorises
investigating a coherent optimisation plan, not reviving those jobs automatically.

## Curve-based route

- Inspect existing integration and cutoff code before adding a duplicate helper.
  Current UT diagnostics integrate to each curve's own post-peak 1% crossing and
  report missing crossings. Validate the intended event, noisy/multiple crossings,
  endpoint handling, units and predicted versus true cutoff error with labelled
  examples. Do not silently replace that definition.
- FT full-domain force-displacement area is a work proxy, NOT automatically
  fracture toughness or crack-initiation energy. Agree the physical FT objective
  and event/cutoff using existing mechanics evidence before optimising it.
- Evaluate UT/FT objectives separately. For a normalised weighted sum, fix reference
  scales from training/physical references, state units/signs and inspect several
  trade-off weights. A weighted sum is not guaranteed to recover all Pareto regions.
  Do not fit normalisation from candidate extremes or the locked test set.
- Define manufacturable disorder bounds, geometry/connectivity validity and any
  researcher-approved strength/feasibility constraints. Preserve UT/FT pairing.

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

## Trustworthiness and evaluation

Reuse existing MLmetrics/runner design diagnostics: ranking, top-10% recovery,
false excellent predictions, selection regret and optional strength feasibility.
These evaluate fixed-model choices; they are not HPO or a design generator.
Improve ML Accuracy owns their existing notebook exposure and model diagnostics;
this task owns objective definitions/search analysis, avoiding concurrent edits to
the same helper. Propose additions through that interface before shared changes.

Use aligned known FEA validation specimens to test whether curve and field-only
objectives reward intended mechanics, not merely large deformation or early
failure. Keep locked-test data out of development. Assess out-of-distribution
candidates and surrogate exploitation. Propose uncertainty/ensemble checks and
small, targeted FEA confirmation of promising/uncertain designs before any claim
of optimisation benefit; these are later approval gates, not automatic launches.
Field motion/gradients are not damage; actual strut labels belong to the separate
Damage variable processing task and must be validated before use.

Deliver readable samples under p2-DisorderML/samples/, a curve-versus-field decision,
objective/constraint definitions, validation criteria and a staged implementation
plan reusing existing code. Keep maintained notebooks under p2-DisorderML/code/;
no scripts/notebooks in data/. Report if field-only optimisation lacks a defensible
proxy. Record decisions in this task's appropriate docs, not duplicate loss code.
