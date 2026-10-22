# Locality study — preparation only

Goal: improve independent Case1(i) **field** prediction with one change at a
time. There is no implemented locality study runner or submitted job yet.
The current loss study continues unchanged. Locality has no logical dependency
on its completion: use the unchanged masked-MSE baseline to isolate architecture,
then test combinations only after each intervention has evidence.

Common controls: native UT/FT topology on canonical nodes, exact saved specimen
split and train-only scaling, width384 / eight heads / three layers, original
11 context channels, full U1/U2 histories. Separate UT and FT training, seed42,
up to450 epochs/patience52. No OOF banks or curve retraining in this first screen.
Confirm promising variants across42/43/44; report parameter count and cost.

| Variant | Change from control |
| --- | --- |
| Control | Existing independent field Transformer, no local block |
| Degree only | One task-topology context channel, degree/8; FT absent nodes masked |
| Existing GNN block | One `LocalFieldGraph` before field attention, initial disordered edge direction/length |
| Gated GNN | Same block with a small learned residual gate; compare directly to existing block |
| Adjacency attention | Add learned per-head bias for initially connected pairs; retain global non-edge attention |
| Geometry attention | Adjacency bias plus projected relative direction/length; compare directly to adjacency-only |
| Masked CNN, one block | Canonical reference-grid 3×3 residual block; occupancy/disorder retained, absent sites masked |
| Masked CNN, two blocks | Same grid/channel contract, two blocks instead of one |

CNN site mapping must be a verified one-to-one reference-coordinate embedding,
not reshaping arbitrary node indices or filling the FT cut as material. A grid
neighbour is not automatically a physical strut. Edge-biased attention uses
pairwise attention scores, **not extra input nodes** or an800-column node feature.
No true future damage/topology is available to a disorder-only predictor.

Degree describes four/eight-connected coordination but adds no specimen-specific
fracture information by itself. Current spatial loss degree balancing does not
mean degree is an input or attention is graph-aware. Pre-attention graph messages,
attention-score bias and CNNs are distinct experiments; do not bundle them.

Before submission: implement opt-in modules/serialization, validate topology,
masking and isolated toggles, CPU regressions and one disposable GPU gate. Then
use normal B1 resources/staging. No blanket HPO, combined redesign or gradient
instrumentation is implied. Selection prioritizes field fracture-motion accuracy,
with global-RMSE safeguards; curves are checked only for retained candidates.
