# Damage variable processing — task brief

Location: p2-DisorderML/handoffs/. Source paths below are repository-relative unless stated otherwise.

This is a separate task, not the Improve ML Accuracy continuation. Read applicable
repo/p1 simulation and p2 samples instructions plus relevant progress notes first.
Do not load the other handoff files or full previous chat by default. Start by
inspecting existing producer/export code and proposing a bounded sample audit;
confirm remote access and scope before ODB processing or job submissions.
Preserve the existing displacement/curve pipeline and its artifacts.

Current implementation: p1 `SIMscripts/A2_DamageOUTpostProcess.py` and shared
`resources/damage.py`; review consumer p2 `code/review_damage.py`. See
`p2-DisorderML/samples/damage-audit/README.md` and the damage status entry.
The researcher authorised the local periodic UT/FT ODB pair. Both have 21 actual
frames, stored bottom/top section SDEG and whole-element STATUS; input meshes
match and hashes remain unchanged. STATUS=0 establishes failure even though stored
SDEG reaches only 0.99999. Separate `strut_damage_v1` products retain raw data,
body-only UT membership and masks. The researcher has clarified the target as
binary fracture from STATUS: any failed member gives `D_fracture=1`, all known
active members give 0, otherwise unknown. Use existing `strut_failed` under
`strut_failure_known`; no re-extraction or schema change is needed. SDEG remains
an optional raw diagnostic. Maximum partial severity is unselected and no longer
blocks this binary target; the saved failure-only numeric-severity arrays still
leave active surface damage as NaN. No new model, bulk export, disordered
processing or job is authorised by these periodic tests.

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

At each recorded frame, if any constituent element has verified STATUS=0,
assign binary fracture 1 to its whole parent strut. If all constituent statuses
are known and active, assign 0 regardless of SDEG. Keep raw SDEG for audit;
continuous severity aggregation is optional and would require an explicit
researcher choice, but is unnecessary for this target. Handle deleted elements
and missing output: missing does not automatically mean undamaged or failed. Identify failure through
the validated damage/status convention and retain absorbing failure only when
the simulation definition supports it. Distinguish first failure from average
strut degradation. Do not maximise a raw sum of element failures.

Produce small, labelled examples under the existing `p2-DisorderML/samples/`:
an intact strut, an active surface-damaged strut, and a strut with one failed
element; show endpoints, strut identity, constituent element labels, raw values, aggregation
arithmetic and the resulting binary fracture value. Full-body and example maps
use purple 0/red 1, with black-dashed unknown observations and no grey active
state. Keep local SDEG diagnostics distinct. Include a crack-adjacent FT example
and different subdivision counts. Plot a highlighted parent strut and its
elements at several frames, and binary fracture against explicitly defined
global loading/strain.
Ask for researcher confirmation before scaling up.

Propose a versioned edge/strut NPZ contract with stable specimen IDs, endpoint
mapping, frame/load values, valid masks, source provenance and variable names.
Keep it separate from the node-displacement arrays; never broadcast strut labels
onto nodes silently. No architecture change is part of this task. The dual local
graph uses **initial** connectivity and is not already a damage-evolution model.

## Interface to model work

Deliver the validated label contract, readable examples and unresolved decisions
to the user for transfer to Improve ML Accuracy. That task owns model integration:
a future strut head may read endpoint-node embeddings plus edge geometry, without
new Transformer tokens. It must handle endpoint-order invariance, task/absent-strut
masks, class/loss imbalance and frame alignment. A local graph can help but is not
a prerequisite; no damage head exists yet. Initial adjacency is not survival, and
true future damage must never determine inference-time edges. Do not independently
edit architecture/loss modules. Reuse validated reference_field_edges as an initial
topology reference; the original INP owns FE-element-to-strut provenance.

A node's fraction of incident struts broken is an optional summary, never a
replacement for edge histories. Shared displacement/damage representations can
still produce inconsistent heads: first test auxiliary supervision without
feeding predicted damage back into displacement or removing predicted edges.
Avoid feedback until the labels/head are independently validated.

Completed work goes into the appropriate producer documentation/progress notes;
keep this brief current until its actions have migrated. No bulk export, new FEA
or damage training is authorised merely by reading this file.
