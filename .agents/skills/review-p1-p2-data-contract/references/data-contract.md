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

## Stable identity and naming

- `Ductile`/`UT`: uniaxial or ductile branch.
- `Fracture`/`FT`: compact-tension or fracture branch.
- `MULTI`/`both`: aligned UT and FT samples.
- `per`: periodic/reference case, represented downstream by simulation id `0`.
- `disNodes` and `disStruts`: nodal and strut-thickness disorder.
- Raw families: `IN-n`, `IN-s`, `IN-f`, `OUT-Ductile`, `OUT-Fracture`, and `FIELDu`.
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

## Field boundary

- Raw field NPZ metadata may include `sample_id`, `frame_values`, `node_labels`, `node_coords`/`coords0`, `components`, `valid_mask`, and source paths.
- The processing layout is `[sample, frame, node, component]`; model loaders may flatten frame/component axes and diagnostics must reconstruct them using metadata.
- `valid_mask` must align with the sample/frame/node axes and distinguish padding or missing frames from valid values.
- UT body-node filtering removes grip nodes to match input-node conventions; FT does not automatically use that crop.
- Trace field changes through the A2 field exporter, `create_fieldNPZ`, MLdata save/load helpers, field/field-to-curve entry points, saved metadata, and diagnostics.

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
