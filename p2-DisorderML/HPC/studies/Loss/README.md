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
Dynamic weighting is implemented and being tested, not an adopted improvement.
See `../../DualOutputs/README.md` for exact component presets and provenance.
