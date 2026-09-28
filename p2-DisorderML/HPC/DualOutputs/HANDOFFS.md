# Focused follow-up prompts

These are copy/paste task briefs, not executable scripts. No remote processing or
new jobs are authorised by their existence. Read current repo instructions and
status before work; coordinate shared-file edits with the architecture task.

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

## Coordination with Improve ML Accuracy

Architecture changes live in MLdual; spatial/temporal/localisation loss ownership
stays in MLfield. New experiment runs retain the old fixed validation selection
score and save motion, true-field substitution and design-screening diagnostics.
Any proposed change to objective definitions, cutoff treatment, loss gradients or
checkpoint selection must remain a separate experiment, not a silent baseline
update. Local graph attention bias, temporal decoding and GNNExplainer are deferred.
