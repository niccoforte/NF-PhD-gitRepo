# P1-To-P2 Data Contract Reference

Use only the section relevant to the changed boundary. Confirm details in code before editing.

## Pipeline ownership

| Stage | Producer or consumer |
| --- | --- |
| Model/job identity | p1 `SIMscripts/A1_*.py` |
| Raw input, curve, and field exports | p1 `SIMscripts/A2_*.py` |
| Transfer-to-processed conversion | `resources/data_processing.py` and p1 processing notebooks |
| ML-ready loading and splits | `resources/MLdata.py` |
| Training and saved-run layouts | p2 notebooks/HPC scripts and `resources/MLmodels.py` |
| Joint serial UT/FT training | `resources/MLdual.py` |
| Diagnostics and run discovery | `resources/MLmetrics.py` |
| Remote/local archive paths | p2 `HPC/B3_ML-transfer-mac.sh` and `HPC/B3_ML-transfer-windows.sh` |
| Full p1 archive mirror | p1 `SIMscripts/B3_ABAQUS-archive.sh`: complete remote p1 contents into external `Z:/p1/HPCarchive`, preserving archive-relative paths; separate from consolidated transfer/MLdata and existing consumers. |

For archived raw/execution records, map `/data/SEMS-TaoLab/Niccolo-Forte/p1/<relative>`
to `Z:/p1/HPCarchive/<relative>`. Do not replace existing consolidated local MLdata
paths or change HPC p2/MLdata. Keep original source paths in immutable provenance.
If remote p1 is removed, restore required raw subsets before any future HPC
re-export; existing producer output defaults are not current read dependencies.

## Stable identity and naming

- `Ductile`/`UT`: uniaxial or ductile branch.
- `Fracture`/`FT`: compact-tension or fracture branch.
- `MULTI`/`both`: aligned UT and FT samples.
- `per`: periodic/reference case, represented downstream by simulation id `0`.
- `disNodes` and `disStruts`: nodal and strut-thickness disorder.
- Raw families: `IN-n`, `IN-s`, `IN-f`, `OUT-Ductile`, `OUT-Fracture`, `FIELDu` and separate binary-fracture `FIELDd` (lowercase d).
- Saved output-layout tokens: `Curve`, `Field`, and `FieldToCurve`. `FieldToCurve` uses field inputs and curve targets but must not be stored as an ordinary `Curve` run.

## Alignment invariants

- Input, output, frequency, field, and manifest records must refer to the same integer simulation id.
- Filtering a failed or missing output must remove or mark the matching records without shifting row correspondence.
- Preserve the periodic/reference row with id `0` unless the task explicitly changes that scientific baseline.
- For `MULTI`, UT and FT train/validation/test membership must stay aligned where the data supports paired samples.

## Curve and manifest boundary

- Processed curve products and manifests are written beside the selected p1 data root.
- Manifests must retain missing-input/output, frequency, NaN, failure-index, and final-inclusion evidence.
- Any filename, column, index, or filtering change must be traced through `data_processing.py`, `MLdata.py`, the relevant p1 notebook, and every p2 loader using it.
- The opt-in p1 `validation_*.py` route reads indexed native ODBs into
  `Z:/p1/sims/Ti/Validation/audit`, preserving actual times/validity and originals.
  `native_stored_energy.csv` is a separate optional diagnostic, with unavailable
  channels left missing; it does not extend/rewrite processed OUT or MLdata.
  Exact-source RVE moduli, modal cutoffs and initiation brackets do not propagate
  into shared calcFT defaults or saved labels. FCC foundation repairs preserve
  the canonical 10 mm UT/FT edges used by existing graph/loss consumers.
  Selected validation CSVs/plots remain external audit products, not training
  labels. They distinguish native dynamic histories from damage-off QS controls
  and Hsieh's initial-triangle convention from dissipation/nonlinear-recovery
  eta-J estimates. Source hashes, event brackets and invalid negative areas
  survive this comparison boundary. An alternate work estimate is not ASTM J.

## Field boundary

- Raw field NPZ metadata may include `sample_id`, `frame_values`, `node_labels`, `node_coords`/`coords0`, `components`, `valid_mask`, and source paths.
- The processing layout is `[sample, frame, node, component]`; model loaders may flatten frame/component axes and diagnostics must reconstruct them using metadata.
- `valid_mask` must align with the sample/frame/node axes and distinguish padding or missing frames from valid values.
- UT body-node filtering removes grip nodes to match input-node conventions; FT does not automatically use that crop.
- Trace field changes through the A2 field exporter, `create_fieldNPZ`, MLdata save/load helpers, field/field-to-curve entry points, saved metadata, and diagnostics.

## Separate strut-damage boundary

- The authorised FCC dataset route extends the existing p1 exporter with `--status-only` (`strut_status_v1`) and B2 with an explicit damage-only manifest route. `A2_DamageBatch.py` restricts discovery to the reviewed Ti FCC disNodes0.2 archive and separate periodic baseline, pins code/source evidence, stages bounded scratch copies and publishes `FIELDd-<stem>.npz` plus JSON beside existing exports. Other lattices and new FEA remain outside scope. Existing FIELDu/curve products are not rewritten.
- `resources/damage_data.py` prepares additive `{UT-UT,FT-FT,MULTI-UT,MULTI-FT}-disNodes-allD.npz` banks (`strut_status_bank_v1`) using the exact IDs/order of the corresponding allFIELD products. Native task struts stay separate from node tensors; explicitly remap canonical endpoints to saved node order and verify actual specimen geometry. Retain periodic ID0 and existing filtering/membership. Unknown fracture stays masked; initial FT absence is not failure. No DATA/MODEL training integration is implied.
- Build NPZ/JSON in separate staging outside MLdata and its descendants. MLdata holds only files directly consumed by ML training/testing; publish the NPZ banks there. Retain optional info/audits/reports elsewhere only when needed, in the upstream FCC archive's existing `<processing-run>/` layout without a provenance tier. Raw FIELDd NPZ/JSON pairs stay together in their original transfer folders. No training consumer requires the bank JSON report.
- Align STATUS/raw FIELDu by recorded frame index/order and count. Bank stacking follows existing fields: retain shorter raw prefixes, pad only unavailable trailing frames with D storage0/valid_mask=false and frame_time=NaN, and preserve sample_frame_count/frame_present. Reject longer sources. Unknown padding is not an observed active label or reactivation; retain last-observed fracture counts separately from the final bank frame. Retain initial frame0 and actual recorded times; do not interpolate binary states, carry failure or require identical specimen timestamps. Local raw products consolidate into the separate periodic/disordered transfer folders; MLdata banks go to local `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata` and HPC `/data/SEMS-TaoLab/Niccolo-Forte/p2/MLdata`, with checksum verification.

- `p2/HPC/DualOutputs/A0-HPC-Damage-contract.py` is a bounded fixture execution check, not a DATA/MODEL loader: pinned NPZ/INP/consumer hashes, endpoint remapping, presence/binary-known masks, frame0-only removal, exact shared-time differences and NPZ/Torch reload. No specimen join, interpolation, prediction evaluation or training follows from a layout match. Scalar-severity coverage is separate from binary-target coverage.

- Producer: p1 `SIMscripts/A2_DamageOUTpostProcess.py`; pure mapping/reduction/validation: `resources/damage.py`; review consumer: p2 `HPC/studies/Damage/review_damage.py`. No existing DATA/MODEL loader consumes this schema.
- One explicit supported FCC20 specimen/task per `strut_damage_v1` NPZ: periodic simulation ID0 or the actual disNodes0.2/lhs/all numeric ID; retain source stem/role separately so same-ID maxB/maxK files are not overwritten. Native struts use instance-scoped original element/endpoint identities and actual INP connectivity; canonical endpoints refer to the audit's sorted FCC reference coordinates, not saved ML order; endpoint_xy0 stores actual disordered geometry. Disordered degree-based original joints, unique producer-bounded reference identity and full canonical-edge equality are all required. UT BodySet excludes grips; initial FT absence is not failure.
- Retain raw section/IP SDEG, DUCTCRT initiation and whole-element STATUS, actual recorded frames including frame 0, raw nodal load histories/time, source/code fingerprints, source-match evidence, masks and reduction rules. Never broadcast strut labels to nodes or silently stack unlike frame/load grids.
- The binary fracture target uses `strut_failed` with `strut_failure_known` (and the element equivalents), not optional scalar-severity `strut_damage`: 1 if any member has STATUS=0, 0 if all members are known active, unknown otherwise. The review colours this binary variable only; raw SDEG and observed active-damage fates are separate diagnostics. Historical scalar-severity NaNs do not imply unknown fracture.
- STATUS=0 establishes element failure; a verified failed member overrides the parent to D=1. Maximum reported section then constituent damage is an explicit severity rule, not average degradation. Preserve incomplete observations as unknown, with independent failure-known/coverage masks. Optional irreversible carry has a separate mask and rejects reactivation; it is disabled in the initial periodic tests.
- Run dedicated synthetic tests and real Abaqus read-only checks, including parent-known masks. The original six-file audit remains diagnostic evidence. The researcher additionally authorised the existing periodic and 20% nodal-disorder FCC archive only: preflight before resumable batches, then verified local/HPC MLdata additions. Existing synthetic CSV/field fixtures do not establish native ODB correctness; ML model integration remains separate.

## Dual serial consumer boundary

- `DUAL_DATA` consumes one existing unscaled, unreduced `DATA(input_kind="field", output_kind="curve", mechMode="MULTI")` object; it does not introduce a new p1 file family.
- UT geometry is the canonical input. UT and FT fields are aligned into the canonical node axis by reference coordinates; the native FT-node subset becomes a boolean presence/attention/loss mask.
- The `fcc_ti` profile validates the 800-node reference body and twelve-node FT exclusion. x0/y0/designable are shared; presence, UT interfaces, sample-specific FT pin/coupling and nominal-tip features are task-specific. Derive pin membership on raw initial geometry before scaling. Input-audit NPZ/Markdown/plots use those same stored decisions; no new p1 export family is introduced.
- A dual split must contain identical ordered UT/FT sample ids and complete geometry, field, and curve targets. Do not silently fall back to independently split task data.
- One train-fitted field normalizer per task transforms the field supervision target. The predicted normalized field is passed directly to the curve stage, so no second field-input scaler may be fitted.
- Dual `crack_face=True` is an explicit fresh-model feature ablation: it appends `initial_crack_face` (retained FT nodes with reduced initial coordination; UT=0) to the default 11-channel context. Feature names/values and definition are saved in the existing descriptor. Preserve old descriptors and do not resume an old checkpoint/study with the changed input schema. This is not a damage target or an attention bias.
- Initial dual targets are full curves. PCA output reduction requires an explicit differentiable reconstruction path before curve-space losses can backpropagate through PCA coordinates.
- The dual runner applies `CombinedCurveLoss` after a fixed Torch inverse-affine transform of both normalized prediction and target; normalized MSE is an explicit alternative. Saved prediction fields remain canonical with masks, but physical-unit diagnostic tables contain only present task nodes. Train-only mean baselines are reused from existing metrics.

## Saved-run boundary

- FieldVariations reuses field-only dual containers and standard result keys.
  FT crack-face adds a zero-initialized fresh context column; residual coordinates
  fit one task's train-only node/frame/component mean/variation with0.1 scale
  floor and retain explicit affine inverse. Frame priors use recorded index,
  fixed weights have specimen-valid mean1. Keep physical selection unchanged;
  compare physical reconstructions within floating precision, not normalized
  values or byte-identical arrays after a changed affine inverse. No p1 exports.

- Opt-in locality screens save the standard dual descriptor/container with
  `training.field_only_modes`; MLmetrics restores that scope. Only the trained
  field task executes and only its field arrays/metrics are exported; results
  record `prediction_scope`. There is no trained curve or second-task result.
  Old descriptors/defaults are unchanged. Locality/gated-graph configs serialize
  initial topology/geometry/grid buffers, not future damage. Degree adds one
  fresh-model context channel; never attach it to an old checkpoint.

- STATUS screen: `MLdamage.align_status_bank` maps native `edge_index` through
  saved field-to-canonical indices, verifies exact IDs/initial coordinates/frame
  schedules, and slices off frame0 from D and masks together. No test fitting or
  unknown-state imputation. Version1 `independent-status-field` checkpoints use
  `load_status_model`; field-result arrays retain existing MLmetrics keys. New
  runs live at `MULTI/Damage/Transformer`, separate from shared DUAL checkpoints.
- Local RUN_ROOT is external `Z:/p2`; other hosts must set an external root explicitly. No checkout may contain `data/`. Generated p2 reports/samples live only in `Z:/00-PhD-gitRepo/p2-DisorderML/samples/`. Preserve archive-relative paths and saved scientific metadata; update operational links, never bulk-rewrite original checkpoint/source provenance.

- Case1(i) follow-ups retain standard prediction keys. Version1 `correction_bank.json/npz` records nested TRAIN-only IDs, data/scaler fingerprint, seed and source-bank hash. `FieldCorrectionStage` saves `base_stage` plus `correction_stage` configs and reloads through `_task_stage`; old configs omit them. Standard results use raw/corrected predicted deployment fields; separate physical `*_source_curves`/`fixed_readout_curves` probes never replace them. Field-bank manifests additionally record active loss variant/coefficients; absent fields preserve old MSE banks. No p1/export/filter contract changes.
- Frozen-source dual curve experiments record `training.curve_only_source` and
  restore it through the dual loader. Standard field arrays are still frozen
  predictions; standard curve arrays use the declared source. Additional true/
  predicted-field curve NPZ/metric tables distinguish substitution from fresh
  training. These fits use two-curve fixed selection; default joint runs/HPO
  remain four-output. Optional fixed field-loss weights are saved in loss config;
  old descriptors omit them and retain identical behaviour. No upstream export
  or sample filtering change is involved.

- Opt-in dual experiments preserve prediction keys/layouts. Architecture descriptors additionally record private encoder depth, local graph buffers/initial-geometry affine reconstruction and interface detachment. Residual-field normalization stores node-resolved train-only mean/scales with the floor definition in data metadata. True-field auxiliary weight is training metadata. Design diagnostics are validation-only additions; FT work is not a toughness label. Legacy GNN descriptors explicitly record historical vs validated FCC initial graph semantics; missing keys retain historical interpretation.

- Opt-in structured field trials add `field_loss_history.csv` for independent runs, raw field-component columns to dual history, `{mode}_{split}_field_motion.csv`, and `field_motion_definitions.json`. Existing prediction keys/layouts do not change. Frozen curve comparisons add paired curve metric tables and an auxiliary NPZ; single-task comparisons use only the intersection held out from both stages. They do not redefine old HPO metrics or overwrite source checkpoints. Physical jump losses reconstruct the existing affine normalization, without refitting the serial bridge.

- Standard: `{RUN_ROOT}/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/{Model}/{Run}`.
- Dual single runs: `{RUN_ROOT}/MULTI/Dual/Transformer/{Run}`. Standard staging/archive/transfer support this layout; reconstruct architecture from `DualStageTransformer.from_config` and load weights through `DUAL_MODEL.load`. Legacy `MODEL` reload and single-task notebook loaders are not dual consumers. Dual HPO: `{RUN_ROOT}/MULTI/Dual/Transformer/HPO/{Study}`, with winner in `best/` and diagnostics in `best/results/`; use the dual notebook and B1 `--resume`, not B2. A fixed equal-task physical validation-MSE ratio selects trials independently of tunable training-loss weights. Code/data/split fingerprints reject mismatched resume.
- Model-specific HPO: `{RUN_ROOT}/{Task}/{OutputKind}/{Model}/HPO/{Study}`.
- Cross-model HPO: `{RUN_ROOT}/{Task}/{OutputKind}/HPO/{Study}/{Model}`.
- Dual notebook consumers: Outputs constructs DUAL_DATA/DUAL_MODEL or loads saved artifacts through `postprocess_load_dual_run`; the existing `ML-HPOpostProcess.ipynb` reviews the dual study and its `best/` winner. No new saved layout or separate HPO notebook is required. Dual post-processing joins specimen errors by sample ID and displays both task fields with their own masks. Keep all executable notebooks under p2 `code/`, not `data/`.
- If an output-layout token or metadata key changes, update model save paths, `run_layout`, run listing, HPO resolution, output-kind inference, diagnostics, and transfer helpers together.

## Synthetic boundary check

The explicitly non-scientific fixture lives at `.agents/skills/validate-repo-change/fixtures/synthetic_contract/contract.json`. Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope contract` to check aligned ids, reference id `0`, field dimensions/metadata, CSV round-trip identity, and NPZ round-trip shapes when NumPy is available. It does not validate real values, Abaqus behavior, scientific assumptions, or HPC execution.
# Independent curve-term checkpoints

`independent-curve-loss-terms`,version1 is a curve-only saved kind, loaded by
resources.MLcurve_terms.load_curve_term_model. It stores one task Transformer,
train-fitted DUAL_DATA contract, true-field source metadata and validation-only
curve outputs. No placeholder field predictions are exported. It is distinct
from both legacy MODEL and the assembled independent DUAL_MODEL container.
