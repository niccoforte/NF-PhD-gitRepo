# Current practical ML winner

**Case 1(i), Predicted + True (`mixed`), without a correction model.**
Verified against the three saved input-study models on 22 October 2026. This is
the accepted cost/accuracy configuration, not a claim of satisfactory fracture
prediction or a global optimum. New experiments do not change this record until
their results are reviewed and a replacement is accepted.

## Architecture and training separation

```text
canonical disorder → independent UT field Transformer → independent UT curve Transformer
                   → independent FT field Transformer → independent FT curve Transformer
```

Four separately fitted networks: no shared UT/FT parameters, optimizer state,
loss or stopping; no curve gradients enter the field generators. The saved
`DualStageTransformer` container and `DUAL_DATA` adapter do not imply joint
training. Each stage retains node tokens and global self-attention.

| Parameter | Disorder → field (each task) | Field → curve (each task) |
| --- | --- | --- |
| Input width before context | 2 (`dx,dy`) | 40 (20 retained frames × U1/U2) |
| Context width | 11, separately projected | Same11 channels |
| Sequence positions | 800 canonical nodes | Same800 |
| Embedding / heads / layers | 384 / 8 / 3 | 96 / 4 / 4 |
| Feed-forward multiplier / width | 4 / 1536 | 6 / 576 |
| Activation | ReLU | GELU |
| Position encoding | Sinusoidal | Learned |
| Encoder dropout | 0.04239430192250655 | 0.016886313997489938 |
| Attention dropout | 0.10468863618228602 | 0.22883857145164282 |
| Output-head dropout | 0.2959779026614149 | 0.07242202597111595 |
| Output-head hidden multiplier | 2 (768 hidden units) | 0 (direct projection) |
| Normalization placement | Post-norm (`norm_first=false`) | Same |
| Linear bias | true | true |
| Pooling | Per node | Presence-masked mean |
| CLS token | false | false |
| Output | 40 displacement channels/node | Full201-point ordered curve |
| Private encoder tail | 0; complete networks are already independent | 0 |
| Local graph module | None | None |

Each task also has its own learned task vector. Neither a task vector nor a
position vector is an extra scalar physical feature.

## Inputs, geometry and masks

Dataset family: Ti, FCC, `nnx=20`, `disNodes`, disorder amplitude `dN=0.2`,
`d_data="in"`, `freq=false`. The material/loading family is fixed, not varied
as model inputs. UT body only; no extra grip nodes or new Abaqus exports.

The canonical UT body contains800 reference nodes over200×190 coordinate units.
FT retains788 of those;12 initial-cut positions are absent. Reference cell10;
nominal FT tip(120,95), fixed/moving pin centres(40,35)/(40,155), radius15.
These are the checked `fcc_ti` geometry profile, not inferred SI units.

| Context columns, in saved order | Meaning |
| --- | --- |
| `x0,y0` | Reference coordinates normalized by specimen span |
| `designable` | Shared canonical disorder/design eligibility; not task presence |
| `present` | UT all present; FT masks its initial cut |
| `bottom_interface,top_interface` | UT body/grip interfaces; zero for FT |
| `coupled_fixity,coupled_load` | FT pin-region membership; zero for UT |
| `tip_dx_ref,tip_dy_ref,tip_distance_ref` | FT reference tip offsets/distance; zero for UT |

Geometry stage: normalized `dx,dy` plus context. Pin memberships use each
specimen's initial disordered positions before standardization, not displaced
response coordinates. Coupled nodes are not falsely labelled independently fixed.
Curve stage: predicted/true displacement history plus the same task context;
no explicit disorder or first-stage latent bypass. Other-task context columns
are zero; overlapping physical values do not share network weights.

FT absent positions remain padded slots, not material. Geometry is zeroed there,
they cannot be attention keys/values, and outputs/loss entries are masked.
`designable=1` at an absent canonical position cannot override `present=0`.
Separate entry/frame-validity masks protect missing values in either task.
Unloaded frame0 is dropped; the20 saved frame values are approximately0.005 to
0.100, not asserted velocities or specimen-exact physical times. Full exact
arrays, node maps, IDs, feature values and affine normalizers are in model JSON.

## Optimization, losses and field-source policy

| Setting | Current value |
| --- | --- |
| Optimizer | AdamW |
| Base learning rate | 0.0002707083854281063 |
| Curve learning-rate multiplier | 0.29790381254971326 |
| Weight decay | 0.00032732095157157824 |
| Batch size / accumulation | 2 specimens / no accumulation |
| Epoch ceiling | 450 per independent fit; not necessarily epochs actually run |
| Early-stopping patience / minimum delta | 52 / 0.00001 |
| LR scheduler | Reduce-on-plateau, min, factor0.6273419968416533 |
| Scheduler patience / threshold | 14 / 0.0001 |
| Gradient clipping | None |
| Data-loader workers | 0 |
| Field loss | Masked MSE, mean reduction, epsilon1e-8 |
| Curve loss | MSE, mean reduction |
| Active-output weights/scales | 1/1; other groups have zero training weight |
| Auxiliary true-curve loss coefficient | 0; source mixing is used instead |
| True/predicted source mixture | 50/50 whole histories per specimen/update |
| Curve validation/deployment source | Predicted fields only |

The mixture is not field interpolation or two optimizer updates. TRAIN inputs
use3-fold out-of-fold field predictions with fold-local scalers and disjoint
inner stopping specimens. Final full-training generators supply validation
fields. A true-input substitution is a separate diagnostic, not deployment.

All geometry/field/curve normalization is train-fitted affine standardization;
one per-task field transform is used for both target and curve input. Geometry
scaling is shared across nodes; field scaling is per frame/component across
nodes. Invalid entries do not fit field statistics. No PCA or residual-field
coordinates. Split seed42, training seeds42/43/44;7137 train /793 validation /
881 locked test. Test remains unevaluated. These seeds are repeats, not an
adopted prediction ensemble. There is no declared best individual seed.

## Provenance and evidence

Architecture, batch2, optimization, scheduler and stopping patience were inherited
from **dual-joint-hpo1** (`best_params.json`), not newly optimized for Case1(i).
The independence study selected the sharing arrangement; the input study and
cost review selected Mixed. Equal independent MSE weights, source policy,
three-fold cross-fitting, seed set and450-epoch ceiling are controlled study
choices, not all HPO-discovered optima. Small batch size is genuinely HPO-derived
here; it is not proof that2 remains optimal for independent field-only training.

- Saved runs: `independent-inputs-261015-mixed-s42`, `-s43`, `-s44`, jobs
  31869466/31869471/31869476; input-study source `c0ab41a5f17a4813c0ccf29aa417b5dc15d9ee91`.
- Frozen generators/banks: `independence-261013-r1-bank-1i-s42/43/44`.
- Saved split-manifest hash: `a4430317305c62a27192930f8df28cf31be819c770fbda6adecefaf8d3804e9f`;
  seed42 field-bank data fingerprint:
  `474f103de2a2f042a3b20a40ea2be9b3699dd5803fa5711a36a5f73a09fa6ae0`.
- Local root: `Z:/p2/MULTI/Dual/Transformer/`; HPC root:
  `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/`.
- Representative exact [model descriptor](Z:/p2/MULTI/Dual/Transformer/independent-inputs-261015-mixed-s42/model.json)
  and [HPO parameters](Z:/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1/best_params.json).
- [Input comparison](samples/field-input-study/README.md),
  [cost decision](samples/field-input-study/correction-cost.md),
  [independence comparison](samples/independence-review/README.md).

Corrected+True remains the original numerical four-output-score leader; Mixed
is the accepted practical choice because correction adds substantial cost for
small/mixed benefits. Do not rewrite historical selection.json to imply otherwise.
Raw checkpoint metrics remain authoritative; reports lead with normalized RMSE
and fixed fracture-motion/curve-feature measures, never incomparable weighted
training totals. Current loss-study results have not promoted a new winner.

## Not enabled / update rule

Current validation evidence (mean ± sample SD across three seeds; RMSE reduction
against the SAME training-mean predictor, higher is better):

| Output | RMSE skill |
| --- | --- |
| UT field | 6.11 ± 0.19% |
| FT field | 24.20 ± 0.82% |
| UT curve from predicted fields | 5.34 ± 0.32% |
| FT curve from predicted fields | 19.06 ± 0.96% |

These are not confidence intervals or percentages of maximum load/displacement.
The fixed four-output normalized-MSE score is 0.751884. UT curves remain close
to the mean predictor; the configuration is a baseline for improvement, not
evidence that fracture prediction is solved. No locked-test performance is claimed.

No degree/crack-face addition, GNN/CNN/attention bias, dynamic or Gaussian/region
weighting, derivative field loss, residual target, correction or damage head is
enabled in this configuration. Available code and submitted studies are not
accepted accuracy gains. Damage-to-curve inputs remain a later separate TODO.

Keep this file current whenever an accepted winner changes: verify actual saved
configs and results; update architecture, every active feature/mask, transforms,
losses, source route, training budget, split/seed protocol, model/run provenance,
validation evidence and limitations together. Distinguish inherited HPO settings,
controlled choices and new tuned values. Update PROJECT_STATUS and linked study
guides at the same time; do not promote an unreviewed trial or a proposed feature.
