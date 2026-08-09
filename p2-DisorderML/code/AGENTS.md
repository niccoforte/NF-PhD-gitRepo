# p2 Notebook Agent Context

This directory contains the local notebook layer for the p2 disorder-to-response ML framework. The notebooks call shared helpers in `resources/`, so understand both the notebook and the helper function before changing behavior.

## Notebook Roles

- `ML-CurveOutputs.ipynb` is the main local curve-output training/HPO notebook. It covers MLP, GCN/GAT, and Transformer stress-strain/force-displacement surrogates.
- `ML-FieldOutputs.ipynb` is the main local field-output training/HPO notebook. It covers GCN/GAT and Transformer nodal displacement-field surrogates.
- `ML-FieldToCurveOutputs.ipynb` is the exploratory field-input to curve-output notebook. It should mirror the HPC smoke path rather than becoming a separate framework.
- `ML-CurvePostProcessing.ipynb` inspects one saved curve run.
- `ML-FieldPostProcessing.ipynb` inspects one saved field run and is the current field visualization/refactor focus.
- `ML-HPOpostProcess.ipynb` inspects model-specific and cross-model HPO studies and can then do a light best-model diagnostic check.
- `Tokenization.ipynb` and `TOKENIZATION_NEXT_STEPS.md` define the output-informed tokenization prototype for recurring disorder motifs.
- `DimensionalityReduction.ipynb`, `GPR.ipynb`, `ML-DisorderDistribution.ipynb`, `Optimization.ipynb`, and `AK-ML-StressStrain.ipynb` are exploratory/prototype notebooks. Treat them as research history unless the user explicitly makes one active.

## Current Research Context

- The framework maps disordered 2D FCC lattice node coordinates to mechanical response.
- Curve-output models predict macroscopic stress-strain or force-displacement curves.
- Field-output models predict per-node displacement fields over Abaqus field-output frames.
- Field-to-curve models use field displacement histories as node-token inputs and curve data as outputs. They use `DATA(input_kind="field", output_kind="curve")`, but must remain distinct from ordinary geometry-to-curve runs through the saved output-layout token `FieldToCurve`.
- Current field target is displacement `U` only, usually components `U1` and `U2`.
- Field data is stored in final ML-ready `.npz` files named with the existing input/output convention, ending in `allFIELD.npz`.
- UT field outputs must use main-body nodes only. Grip-section nodes are dropped to match the input-processing convention.
- FT field outputs do not use the UT grip-node crop. Current final `disNodes` FT field node-count sanity checks are FCC `788`, hex `902`, kagome `1278`, and tri `782`.
- UT field outputs are body-node filtered. Current final `disNodes` UT raw-to-body sanity checks are FCC `1086 -> 800`, hex `1286 -> 902`, kagome `1667 -> 1289`, and tri `1031 -> 791`.
- Some field outputs can have fewer saved frames because of premature termination or missing Abaqus field frames. Treat this as expected when `valid_mask`/padding handles it, not automatically as corruption.

## Important Paths

- Local field ML data is expected at:
  `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`
- Local run archive root is expected at:
  `Z:/p2`
- Current downloaded field Transformer run:
  `Z:/p2/UT/Field/Transformer/TR-Field-UT-1`
- Main notebook under active refactor:
  `p2-DisorderML/code/ML-FieldPostProcessing.ipynb`
- Main ML helper files:
  `resources/MLdata.py`, `resources/MLfunc.py`, `resources/MLmodels.py`, and `resources/MLmetrics.py`

## Shared Helper Ownership

- `resources/MLdata.py` owns `DATA`, path resolution, curve/field loading, train/validation/test split construction, scaling, node filtering, and field shape/component metadata.
- `resources/MLmetrics.py` owns curve/field diagnostics, diagnostic plotting, saved-run discovery/loading, HPO post-processing, and saved-run visualization helpers.
- `resources/MLfunc.py` owns ML training helpers, loss functions, HPO utilities, activation diagnostics, and older general ML plotting helpers.
- `resources/MLmodels.py` owns model classes and model lifecycle behavior such as training, evaluation, saving, loading, and result artifact writing.
- `resources/tokenization.py` owns the tokenization prototype used by `Tokenization.ipynb`.

## Training Notebook Conventions

- Use `DATA(..., output_kind="curve")` for geometry-to-curve targets and `DATA(..., output_kind="field", field_config=...)` for geometry-to-field targets.
- Use `DATA(..., input_kind="field", output_kind="curve", field_input_config=...)` for field-to-curve runs. Keep the notebook aligned with `FieldToCurveOutputs/A0-HPC_FieldToCurve-test.py`: Transformer first, `pool="mean"`, PCA output reduction, and no MLP fallback unless the framework contract changes.
- Use `model="MLP"` for flattened curve inputs, and `model="GNN"`, `GCN`, `GAT`, or `TR` for node-shaped inputs.
- For graph and Transformer curve models, preserve `geom_feats=(True, True)` when node coordinates and boundary flags are intended as features.
- Field models should use node-compatible models only, with `pool="node"` and `MaskedFieldMSELoss`.
- Curve HPO can compare MLP, GCN, GAT, and Transformer; field HPO currently compares GCN, GAT, and Transformer.
- Keep HPO search spaces explicit in notebooks. Do not silently broaden trial counts, epochs, or model families.
- `CombinedCurveLoss` is the current curve-aware loss wrapper. Build future physics/constraint losses on it or adjacent helpers rather than reviving old PINN-style code blindly.

## Post-Processing Notebook Separation

- `ML-CurvePostProcessing.ipynb` should focus on one saved curve model/run.
- `ML-FieldPostProcessing.ipynb` should focus on one saved field model/run.
- HPO study comparison should live in `ML-HPOpostProcess.ipynb`.
- Do not add HPO comparison sections into the curve or field post-processing notebooks unless explicitly requested.
- Shared post-processing logic belongs in `resources/MLmetrics.py`.
- Post-processing should read existing saved metrics and diagnostic artifacts when present: `metrics.json`, `diagnostics_summary.json`, `*_sample_metrics.csv`, `*_point_metrics.csv`, and `*_zone_metrics.csv`. Only save recomputed tables when the original run is missing them or when diagnostic settings changed.
- Post-processing figure outputs should live inside the relevant results folder: `results/postProcessing/`, or for HPO best models, `best_model_results/postProcessing/`. Do not create timestamped post-processing folders by default.
- Field-to-curve runs should use curve diagnostics and saved curve prediction tables, but path/layout helpers must preserve `FieldToCurve` rather than treating them as ordinary `Curve` runs.
- For HPO runs, strongly prefer `best_model.*` and `best_model_results/` when present, but keep loaders flexible for missing or varied artifacts.
- Treat `predictions.npz` as saved prediction/truth arrays for visualization and diagnostics, not model weights. Use it to inspect saved runs without a model forward pass whenever possible.

## Field Refactor Priorities

The user wants a guided, cell-by-cell cleanup. Do not jump ahead and rewrite large sections without agreeing on the cell being changed.

Core visual goals:

1. Load all saved field outputs and choose sample, frame, and component directly in the visualization cell.
2. Support selected sample index, best sample, worst sample, and random sample views.
3. Make the field visualization interactive enough to justify using a notebook.
4. Show loss evolution over training for HPC runs.
5. Improve field maps beyond colored points, ideally with a toggle between point maps and continuous-style plots like the continuum plots.

Keep these goals active throughout the field post-processing refactor.

## Configuration Cell Direction

The top field post-processing configuration cells should only contain durable run/loading settings:

- `RUN_ROOT`
- `mechMode`
- `model`
- `run_name`
- `VIEW_MODE`
- `run_type`
- `RUN_PATH_OVERRIDE`
- `LOAD_DATA`
- `LOAD_MODEL`
- `DEVICE`
- `DATA_PATH_OVERRIDE`
- `ACTIVE_SPLIT`

Viewing controls should not live in the global configuration cell:

- `FIELD_VIEW_FRAME`
- `FIELD_VIEW_COMPONENT`
- `FIELD_SELECTED_SAMPLES`
- `FIELD_RANDOM_COUNT`
- `FIELD_RANKING_METRIC`
- `FIELD_PLOT_STYLE`
- `NODE_METRIC`

Those belong in the relevant visualization cells.

## Helper-Code Rules

- Keep helper changes minimal and focused.
- Treat cleanup as part of the task, not polish after the task. Active notebooks should not keep stale cells, stale outputs, copied fallback versions, or superseded implementation paths.
- Do not add module-level path constants or hidden global variables to helper scripts.
- Function parameters/defaults are acceptable when a default is needed.
- The default data path for post-processing should live as a function parameter/default, not as a module-level variable.
- `postprocess_load_data()` should treat `None`, `""`, and `"auto"` as using its default data path.
- If the path points directly to an `MLdata` folder, normalize it to the parent folder before passing to `DATA`, because `DATA` appends `MLdata`.
- Avoid stale helpers, stale imports, and old notebook cells as the refactor proceeds.
- Do not leave dated, legacy, stale, broken, duplicated, or unused code anywhere. Remove it when it is clearly obsolete, or move it into the module where it belongs if it is still active.
- Do not keep old notebook metadata snapshots or copied fallback versions of cells after a working implementation has replaced them.
- Prefer direct readable code over defensive safety-net layers unless the user has explicitly asked for that recovery path.

## Known Recent Fixes

- `postprocess_load_data()` in `resources/MLmetrics.py` now defaults to:
  `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`
- It normalizes an `MLdata` folder path to its parent before constructing `DATA`.
- `postprocess_build_diagnostics()` can now recover node coordinates from saved node-metrics CSVs when `DATA` is unavailable, using `x` and `y` columns.
- Current saved runs can include `loss_history.csv`; post-processing should use that file when present rather than parsing Slurm logs.
- The current run has saved node metrics with coordinates, so field maps should not depend strictly on a loaded `DATA` object.
- Saved ML run directories are separated by output kind:
  `Z:/p2/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/{model}/{run_name}`.
- Model-specific HPO studies use:
  `Z:/p2/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/{model}/HPO/{run_name}`.
- Cross-model HPO studies use:
  `Z:/p2/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/HPO/{run_name}/{model}`.
- When changing save/load or post-processing path logic, update `FieldToCurve` support together with `Curve` and `Field` support, including run listing, HPO path resolution, saved output-kind inference, and transfer examples.

## Current Run Artifacts

For `Z:/p2/UT/Field/Transformer/TR-Field-UT-1`:

- `predictions.npz` contains saved truth and prediction arrays.
- Current prediction and truth shapes observed earlier:
  `(889, 800, 40)`
- This corresponds to flattened field targets that are reshaped by diagnostics into sample/frame/node/component form.
- `model.json` has summary metrics, but not full loss history.
- Full training loss evolution was not saved for the current run except sparse values printed in the Slurm log.
- Future HPC runs should save loss history explicitly as `loss_history.csv`, so post-processing can plot training and validation loss.

## Tokenization Notes

- The tokenization prototype uses existing UT/FT or MULTI simulation data, not generated samples.
- It builds a normalized score from selected mechanical properties, extracts local node-patch features, learns supervised PCA-like embeddings, clusters them with KMeans, and saves token IDs/diagnostics.
- Success criteria are enriched tokens in high-performing samples, robust enrichment signs across seeds/splits, and later FEA intervention tests.
- If diagnostics show near-uniform tokens, weak enrichment, or unstable enrichment, do not keep elaborating the discrete-token path without discussing it.

## Tone And Workflow

- The user wants to work through notebook refactors interactively.
- Explain each configuration variable before changing it.
- Ask before making broad notebook edits.
- When editing, keep changes scoped to the agreed cell or helper.
- If a cell or helper is stale, point it out and propose removing it before doing so.
