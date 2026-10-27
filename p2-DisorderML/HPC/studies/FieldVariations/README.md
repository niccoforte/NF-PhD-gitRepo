# Field variations: one change per independent generator

`A0-FieldVariations.py` / `B4_FieldVariations.sh` implement the authorised
feature/representation/weighting screen. Thirteen new final-generator fits,
seed42; reuse `locality-261026-r1-control-UT-s42` and `...-FT-s42` as matched
controls. No CNN, degree, graph, derivative, dynamic weight or damage head is
enabled. These compare to the accepted plain architecture, not a promoted CNN.

| Variant | Tasks / fits | Only change |
|---|---|---|
| crack_face | FT /1 | Append initial crack-face membership, new projection column starts0;26 retained FT nodes. No pointless all-zero UT run. |
| residual | UT + FT /2 | Train-only per-node/frame/component mean/variation coordinates,0.1 pooled-scale floor; inverse to physical displacements for every accuracy metric. |
| box2, box4 | FT /2 | A1 CrRegMESH crack-tip region: raw nodal weights2 or4 inside,1 outside; valid-entry mean-one normalization per specimen. |
| gaussian_a1_w0.15, gaussian_a1_w0.25, gaussian_a3_w0.15, gaussian_a3_w0.25 | UT + FT /8 | Temporal frame weights only:1+a exp[−0.5((q−0.5)/width)²], q=frame index/(20−1). Same weights at all nodes/components before valid-entry normalization. |

Gaussian emphasis is highest in the two middle retained frames, symmetric and
positive at the start/end. `q` is the recorded-frame coordinate, not x/y, a
specimen-exact physical time derivative or a target-derived fracture onset.
This tests the requested mid-simulation prior without claiming every specimen
fractures there. The baseline1 retains whole-history supervision. Regional and
temporal weights are separate fits; no interaction is bundled.

All use the saved Mixed control's exact IDs, train-fitted scales, geometry,
initialization, field384/8/3, AdamW, batch2,450 epochs/patience52. Residual changes
only the active task's target affine coordinates; split/mask/physical truth stay
fixed. Selection remains the same physical validation-MSE/TRAIN-mean-MSE ratio,
not a weighted objective. No locked-test evaluation. No OOF/curve fitting yet.

The named runner reuses Locality's final-generator training/export routine;
`resources/MLfield_variations.py` contains only imported intervention functions.
B1 stages the common runner. Preserve default Locality behaviour and old
checkpoints. Standard dual containers carry field-only scope, metadata and exact
loss configs for reload; export trained fields only.

Preview from repository root:
`bash p2-DisorderML/HPC/studies/FieldVariations/B4_FieldVariations.sh SUITE`.
Explicit `--submit` snapshots clean immutable source, queues one disposable8h
full-contract/13-case GPU train/export/reload gate, then13 B1 afterok dependents.
Full requests:240h,1GPU,12CPUs,90000MB. Existing STATUS jobs remain unchanged.
Home/archive layout: `MULTI/Dual/Transformer/<suite>-<variant>-<UT|FT>-s42`;
`PROJECT_STATUS.md` records actual IDs, not this protocol.

Keep only the successful gate home log, no gate archive, remove successful
scratch. Diagnose failures then clean obsolete failed evidence after correction.
Save physical predictions/masks, mean-relative skill, local motion metrics,
training/runtime metadata, log losses and truth/prediction/error views (UT full,
FT crack-tip). Retrospective STATUS-event review follows completion. Confirm
promising variants across seeds43/44 before combining or generating OOF banks.
`review_field_variations.py --jobs PATH --execution-note TEXT` renders the
preliminary protocol, actual submission receipt and temporal-weight profiles
to canonical `samples/field-variations/`; it does not invent completed results.
