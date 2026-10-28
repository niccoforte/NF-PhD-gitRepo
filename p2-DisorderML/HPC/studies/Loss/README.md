# Custom-loss study

## Independent MSE versus MAE/Smooth-L1 field screen

`A0-RobustFieldLoss.py` / `B4_RobustFieldLoss.sh` fit final field generators only:
fresh masked MSE, masked MAE, Smooth-L1 beta0.1 and beta1, separately for UT/FT,
seed42: eight fits plus one disposable eight-case GPU gate. Capacity384/8/3,
original features/IDs/scalers,450 epochs/patience52 and fixed physical MSE
selection stay unchanged. Each production fit requests240h/1GPU/12CPU/90000MB.
Do not combine CNN, spatial, temporal, STATUS or dynamic weights in this screen.

Residuals remain in the accepted train-fitted standardized field coordinates.
Beta is a standardized residual magnitude, not mm. Smooth-L1 is e²/(2 beta)
inside beta and |e|−beta/2 outside, so it is Huber-shaped but has a beta-dependent
overall scale relative to the conventional Huber definition. There is no new
train-fitted threshold, target-dependent weight or loss-component coefficient.
Different weighted objective totals cannot rank methods. Log objective histories
and fixed validation errors, label them distinctly, and use the same masks.

Use existing Locality's field-only runner/reload through B1. Archives remain
`MULTI/Dual/Transformer/<suite>-<variant>-<mode>-s42`, recording the actually
trained task. `MaskedRobustFieldLoss` serializes kind/beta/reduction/eps; historical
MSE descriptors retain their class/default. Future promotion requires global,
confirmed-fracture, jump/sign/timing and TRAIN-mean/variation checks, then seed
confirmation before expensive OOF banks or curve training.

```bash
bash p2-DisorderML/HPC/studies/Loss/B4_RobustFieldLoss.sh robust-fields-261028 [--submit]
```

`A0-Loss.py` selects the Case1(i) isolated loss implementation;
`B4_Loss.sh SUITE [--submit]` is its production launcher. For the current policy:

```bash
LOSS_CURVE_SOURCE=mixed INPUT_SUITE=independent-inputs-261015 \
FIELD_BANK_SUITE=independent-losses-261018 \
bash p2-DisorderML/HPC/studies/Loss/B4_Loss.sh NEW_SUITE
```

This is a preview, NOT a request to repeat the current submitted suite
`independent-losses-261022-mixed`. Keep its twelve compatible field banks and
eighteen dependent curve fits. Do not edit their frozen sources.

| Isolated change | Fixed starting coefficient |
| --- | --- |
| Spatial signed neighbour-motion loss | 0.1 plus masked MSE |
| Temporal recorded-frame increment loss | 0.1 plus masked MSE |
| Both | 0.1 each plus masked MSE |
| Sudden-motion target weighting | Gain1, capped, specimen/valid-entry mean1 |
| Combined curve loss | No peak-location term |
| Same combined curve loss | Corrected soft/soft location0.02, beta20 |

All curves retain the selected Predicted+True source policy. Field changes get
new OOF banks and adapted curve fits; frozen curve readouts separate upstream
effects from adaptation. Curve-loss comparisons hold the raw field bank fixed.
Coefficients are controlled presets, not HPO-selected optima.

Judge fixed global and fracture-sensitive quantities, not differently weighted
training totals. Report normalized RMSE, local signed motion/jump timing,
curve peaks/postpeak/work and seed variability. Loss histories use log axes.
Dynamic weighting has been tested, not adopted as a universal improvement.
See `../../DualOutputs/README.md` for exact component presets and provenance.

`review_loss_study.py` reviews the completed three-seed Mixed continuation and
its reused banks, without training or locked-test evaluation. Run from the repo
root; defaults read external `Z:/p2/MULTI/Dual/Transformer` and write the ignored
`samples/loss-study/README.md` with inline figures. It verifies identities,
scalers, truths, masks, physical metrics, frozen curve-loss fields and train-only
OOF partitions. Report global/local/peak/work trade-offs and seed variability;
curve examples retain signed-error bars, fields truth/prediction/error with FT
crack-tip zooms, and grouped losses log axes. Tests protect matching and masking.

`review_fracture_regions.py` adds retrospective confirmed-STATUS onset windows:
exact specimen/native-edge/frame joins, endpoint errors and signed strut motion,
plus the larger FT box/tip region. True damage defines evaluation regions only;
it never becomes predictor input or training weights. Preserve event counts and
do not equate these specific fracture endpoints with every node in the FT box.
The single main report embeds narrow-versus-broad fracture and mean-field plots;
do not regenerate a separate fracture-region Markdown report. Global and
regional scores include RMSE skill against a training-only mean; reconstruct
that mean from allFIELD IDs and reproduce its saved global error first. Zero
skill means the average output, negative means worse. Keep variation/collapse
ratios separate: matching output spread does not establish correct predictions.
It follows the root visual-report order and keeps exhaustive numbers in CSV.
Field galleries include signed and percentage error with explicit tiny-truth
mask/colour caps; curves retain signed bars plus percentage-error views. Custom
objective values diagnose training, while fixed global/local/peak/work metrics
compare formulations. `review_fracture_regions.py` supplies CSV/figures/counts
to the main reader; it is no longer a second report writer.

`review_true_curve_losses.py` checks the three historical true-field shared-curve
fits (MSE/no-location/soft-location) for identical architecture, optimizer/source
settings, splits, normalization and saved truths. Report true-field RMSE and
peak/work trade-offs separately from independent Mixed deployment; do not call
these historical shared-encoder results a completed Case1(i) custom-loss oracle.

## Authorised FT spatial balance

`B4_SpatialBalance.sh SUITE [--submit]` submits `A0-SpatialBalance.py`: five
FT-only final fields with spatial coefficients 0/.001/.003/.01/.1, seed 42,
unchanged independent plain Transformer and masked displacement MSE coefficient
1. Temporal/localisation weights are zero. Physical train-fitted jump scales,
valid connected pairs and degree averaging are unchanged. Fresh 0/.1 controls
bridge the older OOF-bank trainer. The gate checks full IDs then all five tiny
train/export/reload paths; production uses 450 epochs and 240h B1 dependents.
Report global, actual fracture endpoint/strut and broader FT-box errors plus
TRAIN-mean skill, raw components and cost. Temporal work is deferred until this
spatial balance question is resolved. Regional spatial-only supervision inside
the FT box is feasible but remains a separate proposed ablation; it is not
included in the coefficient screen or silently combined with regional MSE.

The separate `historical-shared-true-field/README.md` is a distinct historical shared-encoder,
true-training/true-evaluation study. Its expanded reader verifies physical CSV
truths and TRAIN means, reports target-defined peak/pre/post-peak windows, hard
peak/work metrics, large error/percentage examples, costs and log histories.
These curve regions are not spatial STATUS-labelled nodes. Bundled presets do
not identify each loss term: one-term attribution remains deferred.
# Independent true-field curve-term attribution (28 October)

`A0-CurveTerms.py` and `B4_CurveTerms.sh` fit **one independent UT or FT curve
Transformer per job**, using true fields in both training and validation. No
field-generator optimization, Mixed source, shared UT/FT encoder or locked-test
evaluation is involved. The current accepted descriptor supplies architecture,
IDs, feature/scaler contract and optimizer/budget, not pretrained weights.

Eight arms × two modes × seeds42/43/44 = **48 final fits**, each up to450 epochs,
patience52,240h/1GPU/12CPU/90000MB. One disposable16-case GPU execution gate
checks all arms and both modes, including full-data identity/scaler checks and
checkpoint reload. Jobs remain dependent on its success.

| Arm | Objective change from normalized MSE |
| --- | --- |
| mse | None; fresh matched control |
| range_mse | Replace only the base by historical per-target-range MSE |
| zone | Add0.5 × range-normalized zone MSE; zone weights1/5/0.2 |
| derivative | Add0.05 × existing first-derivative term |
| peak_height | Add0.25 × existing normalized peak-height term |
| work | Add0.1 × existing normalized area/work term |
| peak_location | Add0.02 × soft-versus-soft location term, beta20 |
| combined | Add all five terms; interaction diagnostic, not single-term attribution |

Except `range_mse`, the base is unchanged. Additional terms reconstruct physical
curves differentiably; UT/FT zone boundaries use their existing definitions.
These fixed coefficients are screening settings, not HPO optima. A negative
result at one coefficient does not establish that a term is intrinsically bad.
All arms select by the same physical validation MSE divided by the fixed error
of the TRAIN-mean predictor; differing weighted loss totals never rank arms.

Archives use `{UT|FT}/FieldToCurve/Transformer/<suite>-<arm>-<mode>-s<seed>`.
The explicit `independent-curve-loss-terms` checkpoint kind loads with
`resources.MLcurve_terms.load_curve_term_model`, not `DUAL_MODEL` or legacy
`MODEL`. It contains only the active curve model, data contract and provenance.
Predictions, per-specimen peak/work diagnostics, peak/pre/post-peak regions and
loss components are saved for the requested visual report. The historical
shared-encoder report lives in
`samples/loss-study/historical-shared-true-field/README.md`; do not mix populations.

## FT-box spatial supervision

`B4_SpatialBox.sh` uses `A0-SpatialBalance.py --spatial-region ft_box`:
four seed42 final FT fits at0.001/0.003/0.01/0.1, reusing the zero coefficient
control in the all-edge balance suite. Architecture, full-field MSE and TRAIN
normalizers/jump scales stay fixed. Only initial struts with **both endpoints**
inside the existing geometric crack-tip box contribute to the added spatial
term. No STATUS labels define training regions. Endpoint validity and
degree-balanced averaging remain active. Global attention/shared parameters
can still change predictions outside the box. A separate five-case GPU gate
checks this region implementation; existing all-edge jobs remain untouched.

Preview/submit from a clean synchronized repository:

```bash
bash p2-DisorderML/HPC/studies/Loss/B4_CurveTerms.sh curve-terms-261028 [--submit]
bash p2-DisorderML/HPC/studies/Loss/B4_SpatialBox.sh spatial-box-261028 [--submit]
```
