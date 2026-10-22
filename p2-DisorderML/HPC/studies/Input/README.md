# Field-input study

`A0-Input.py` is the production Case1(i) source-study entry point;
`B4_Input.sh SUITE [--submit]` previews/submits through the shared B1 workflow.
The completed `independent-inputs-261015` study compares predicted, true,
Predicted+True, corrected, and Corrected+True curve-training fields.
Deployment uses predicted information only; true-field substitutions are
diagnostics. Current practical selection is **Predicted+True**, without a
corrector. Do not resubmit the completed comparison.

`review_field_inputs.py` verifies IDs/scalers/masks/truths/RMSE/folds and creates
the human report. `--cost-evidence FILE` adds paired cost review from saved
results and scheduler snapshots, without SSH/training/inference. Separate
allocation cost from latency and cost-aware recommendations from original
accuracy ranking. `test_field_input_review.py` uses synthetic artifacts.
The generated report is canonical `samples/field-input-study/README.md`;
artifacts are external `Z:/p2/MULTI/Dual/Transformer/`.

## Current practical configuration

The complete maintained winner record is [WINNER.md](../../../WINNER.md).
This overview must remain consistent with it; unreviewed studies do not promote a winner.

Verified from `independent-inputs-261015-mixed-s42/model.json`, not inferred from
the earlier Trial1 deck. UT/FT parameters and field/curve training are independent.

| Setting | Geometry to field | Field to curve |
| --- | --- | --- |
| Width / heads / layers | 384 / 8 / 3 | 96 / 4 / 4 |
| Feed-forward multiplier | 4 | 6 |
| Activation / positional encoding | ReLU / sinusoidal | GELU / learned |
| Output / pooling | 40 values per node | 201 points / masked mean |
| Baseline objective | Masked displacement MSE | Curve MSE |

AdamW: learning rate0.0002707084, weight decay0.000327321; curve learning-rate
factor0.297904. Batch2, maximum450 epochs, stopping52; plateau factor0.627342,
patience14. Three training seeds42/43/44, split seed42;7137/793/881 train/val/test.
Three training-only OOF folds provide realistic curve-training predictions.
Curve source is50/50 complete predicted **or** true history per specimen/update;
validation/deployment uses predicted fields. No corrector or latent bypass.

Node input: geometry stage separately projects `dx,dy`; both stages project the
same11-column context. Shared physical columns are reference `x0,y0,designable`;
`present` is task-specific. UT adds `bottom_interface,top_interface`; FT adds
`coupled_fixity,coupled_load,tip_dx_ref,tip_dy_ref,tip_distance_ref`. Inapplicable
task columns are zero. Pin memberships use each specimen's initial disordered
coordinates. These equal/overlapping values do **not** mean shared model weights.
Curve tokens contain the task's40 U1/U2 history channels plus context, not explicit
disorder or field-stage latent embeddings. Learned task vectors and positional
vectors are added in embedding space, not appended scalar context columns.

Canonical800 nodes: UT all present, FT788 present with12 absent slots masked;
separate field-validity masks also protect missing entries/frames. All scaling
fits train only. Full curves, no PCA. Degree, crack-face, local graph/CNN/attention
bias, residual field targets, Gaussian/region/dynamic weights and damage heads
are not enabled in this baseline. Some are implemented opt-ins or tested prior
DUAL variants, not adopted independent-model improvements.

Full source/cross-fitting and loss-policy contracts remain documented in
`../../DualOutputs/README.md`. The CLI lives here; reusable fitting logic is
`resources/MLfield_sources.py`, and source controls stay explicit.
