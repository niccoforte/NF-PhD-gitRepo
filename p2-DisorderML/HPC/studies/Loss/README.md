# Custom-loss study

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
