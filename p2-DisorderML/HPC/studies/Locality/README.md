# Locality study — isolated field screen

Goal: improve independent Case1(i) **field** prediction with one change at a
time. `A0-Locality.py` and `B4_Locality.sh` implement the screen; PROJECT_STATUS
records actual submission/execution, never infer it from this guide.
Locality is independent of the completed loss comparison: use the unchanged
masked-MSE baseline to isolate architecture,
then test combinations only after each intervention has evidence.

Common controls: native UT/FT topology on canonical nodes, exact saved specimen
split and train-only scaling, width384 / eight heads / three layers, original
11 context channels, full U1/U2 histories. Separate UT and FT training, seed42,
up to450 epochs/patience52. No OOF banks or curve retraining in this first screen.
Confirm promising variants across42/43/44; report parameter count and cost.
The fresh control is fitted in the same field-only protocol for matched runtime
and initialization. Eight variants × two tasks =16 final-generator fits, not
sixteen cross-fitted banks. Existing studies are not rerun or altered.

| Variant | Change from control |
| --- | --- |
| Control | Existing independent field Transformer, no local block |
| Degree only | One task-topology context channel, degree/8; FT absent nodes masked |
| Existing GNN block | One `LocalFieldGraph` before field attention, initial disordered edge direction/length |
| Gated GNN | Same block with a small learned residual gate; compare directly to existing block |
| Adjacency attention | Add learned per-head bias for initially connected pairs; retain global non-edge attention |
| Geometry attention | Adjacency bias plus projected relative direction/length; compare directly to adjacency-only |
| Masked CNN, one block | Canonical reference-grid depthwise3×3 + pointwise residual block; absent sites masked |
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

Plain encoder weights initialize identically before the intervention. Degree's
new context column starts with zero projection; other common parameters are
unchanged. Gated graph starts at sigmoid(-3), about0.047. Adjacency and geometry
bias start at0; global non-edge attention remains available. Geometry bias acts
on actual initial struts, with differentiably reconstructed disordered vectors.
CNN updates are occupancy-masked and normalized by valid-neighbour count. A grid
convolution is not a graph convolution and can mix sites across a geometric gap;
it never claims missing material or load-bearing connections exist there.
CNN refinement is before global attention in this screen, like the graph block;
this is not a displacement-output CNN decoder. Additional block parameters are
reported; common-base capacity is held fixed, not total parameter count. An
unhelpful result at these presets does not reject an entire architecture family.

`DUAL_MODEL(field_only_modes=(TASK,))` executes/trains only the selected field
network. The saved standard container still has frozen unused stages for reload,
but **no curve or other-task predictions/metrics are exported**. Checkpoint
selection is the same task's physical validation-MSE/train-mean-MSE ratio. Save
field/motion metrics, truths/predictions/masks, training time, active parameter
count, peak GPU memory, log losses and UT/full or FT/crack-tip examples with
signed error. There is no scalar combined curve score in this screen.

Preview: `bash p2-DisorderML/HPC/studies/Locality/B4_Locality.sh SUITE`.
Explicit `--submit` snapshots clean source and queues one disposable8h GPU gate
plus16 full240h/1GPU/12CPU/90000MB jobs through B1 with afterok dependencies.
The gate first checks full-data IDs/scalers, then all16 one-epoch/64-specimen
training/diagnostic/reload cases. Its successful scratch is removed, no archive
is kept, and its successful home log remains. `test_locality.py` protects opt-ins,
mask invariance, common weights, gradients, reload and field-only scope. CPU
contracts do not demonstrate GPU success or improved real-data accuracy.

The 22 October gate failed before training: forcing feature extrema into the
64-specimen debug training partition exhausted its holdouts. Bounded runs now
disable that forcing; the full-data gate and production retain the exact saved
extrema-aware split. The regression exercises both paths. Restart with a fresh
suite label after checking the previous jobs; never alter their frozen source.

No blanket HPO, combined redesign or gradient
instrumentation is implied. Selection prioritizes field fracture-motion accuracy,
with global-RMSE safeguards; curves are checked only for retained candidates.

`review_locality.py` reads the completed16 saved field fits and checks common
IDs/scalers/masks/targets plus reproduced physical errors. It writes the canonical
`samples/locality-study/README.md`: per-task changes, local-jump safeguards,
training cost, logarithmic histories and saved UT/full or FT/zoom error examples.
One seed screens candidates; confirmation is required before default promotion.
Report TRAIN-mean-relative skill and output-variation ratios for every variant,
plus actual STATUS-onset endpoint/strut errors for control/CNN1/CNN2. The report
reconstructs the mean from train IDs only and verifies its global baseline.
Show train and validation losses together on log axes; selected best epochs,
not overfit tails, supply saved predictions. `run(...,configure=...)` also serves
the named FieldVariations entry point; the default Locality CLI is unchanged.

## Authorised confirmation and depth screen

`B4_Locality-confirmation.sh SUITE [--submit]` previews ten final fits: control
and CNN2 for UT/FT at seeds 43/44 (eight), plus CNN3 for UT/FT at seed 42 (two).
Reuse completed seed-42 control/CNN2 results; CNN3 is an isolated depth screen,
not an adopted improvement. Same reference, splits, MSE, 450-epoch ceiling and
patience; no OOF/curves/loss/feature changes. The six-case disposable gate uses
`--preflight-variants control cnn2 cnn3`, followed by 240h/1GPU/12CPU/90000MB fits.
Only successful home gate logs survive; production exports retain reload checks.

The maintained review now follows the visual-report route: global control/mean,
actual STATUS regions for every variant, four-column field galleries, costs and
validation/selected train-validation histories. Use `review_locality.py`.

`review_cnn_confirmation.py` reviews the ten completed confirmation/depth runs,
reusing original seed42 controls/CNN2. It verifies downloads, all IDs/masks/truths,
TRAIN means and same-seed controls, then updates the existing Locality README
with three-seed global/fracture/cost/variation evidence and a single-seed CNN3
depth check. It trains nothing and does not promote WINNER.md. Rebuilding the
initial screen preserves the completed follow-up section. Run from repo root:

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -B p2-DisorderML/HPC/studies/Locality/review_cnn_confirmation.py
```
