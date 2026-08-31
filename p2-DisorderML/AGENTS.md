# p2-DisorderML Agent Context

This paper folder is the machine-learning and optimization layer of the PhD repository. It uses ML surrogates to map nodal disorder in 2D FCC lattice geometries to mechanical response, then uses those surrogates and diagnostics to guide design decisions. Keep edits tied to that workflow.

## Repository Relationship

- `p2-DisorderML/code/` contains notebooks for local exploration, training, HPO post-processing, saved-run diagnostics, tokenization, and optimization.
- `p2-DisorderML/HPC/` contains Slurm/bash submit helpers and run-specific Python entry points for cluster training and HPO.
- `resources/` is the shared package used by this folder. Prefer changing shared behavior there only when the change belongs to the reusable framework, not when it is just a notebook-specific adjustment.
- `p1-DisorderLatticeProperties/` owns most Abaqus simulation and ML-ready data generation logic. Treat p2 data files as outputs of that upstream FEA/data-processing workflow unless the task explicitly asks to change data generation.
- `p3-DisorderIcingMitigation/` is a separate paper folder. Do not pull p3 conventions into p2 unless the user explicitly asks.

## Research Vocabulary

- `UT` is the uniaxial/ductile branch. Current UT curve metrics include ductility, strength, stiffness, and work of fracture.
- `FT` is the fracture-toughness branch. Current FT metrics include `K_JIC`, `K_IC`, force, and displacement.
- `MULTI` or `both` means aligned UT and FT samples, with shared train/validation/test indices where the data supports it.
- Curve-output models predict macroscopic stress-strain or force-displacement curves.
- Field-output models predict per-node displacement fields over Abaqus field-output frames. Current field work is displacement `U`, usually `U1` and `U2`.
- Field-to-curve models use field displacement histories as node-token inputs and curve data as outputs. In the framework this is `DATA(input_kind="field", output_kind="curve")`, saved under the distinct output-layout token `FieldToCurve`.
- Nodal disorder inputs are stored as paired x/y node coordinates or displacements. Graph and Transformer models use node-token input shapes; MLP models use flattened inputs.

## Core Framework Ownership

- `resources/MLdata.py` owns `DATA`, path resolution, UT/FT/MULTI loading, splitting, scaling, node filtering, field `.npz` loading, field component selection, and reconstruction metadata.
- `resources/MLmodels.py` owns `MODEL`, model classes, dataloaders, training/evaluation orchestration, checkpoint metadata, result saving, and model reload behavior.
- `resources/MLfunc.py` owns training loops, HPO helpers, loss functions, activation diagnostics, and older general ML plotting helpers.
- `resources/MLmetrics.py` owns curve/field diagnostics, post-processing loaders, saved-run artifact discovery, diagnostic plotting, HPO summaries, and saved-run visualization helpers.
- `resources/tokenization.py` owns the output-informed tokenization prototype.

## Data And Run Paths

- Local ML-ready data used by current notebooks is generally under `Z:/p1/data/Ti/disNodes/0.2/FCC/MLdata`.
- Local saved ML runs are generally under `Z:/p2`.
- On the cluster, ML data and archives are currently routed through `/data/SEMS-TaoLab/Niccolo-Forte/p2`.
- `DATA(path=1, ...)` resolves to the local Ti/FCC data tree. `DATA(path="HPC", ...)` resolves to the cluster p2 data root.
- When passing a data override into post-processing, an `MLdata` folder path is normalized to its parent because `DATA` appends `MLdata`.
- Avoid committing generated data, checkpoints, results directories, Slurm logs, or notebook output bulk unless the user explicitly asks.

## Modeling Conventions

- Use `output_kind="curve"` for curve targets and `output_kind="field"` for nodal field targets.
- Use `input_kind="field", output_kind="curve"` for field-to-curve runs. Keep this separate from ordinary geometry-to-curve runs in saving, loading, run listing, transfer, and post-processing path helpers.
- Field-output `DATA` currently requires node-compatible models: `GNN`, `GCN`, `GAT`, `TR`, or `Transformer`. MLP is curve-only in the current field/HPO path.
- For graph and Transformer inputs, preserve node structure. Do not apply input PCA/reduction before node tokenization.
- Field-to-curve inputs are also node-token inputs. The smoke route is Transformer with `pool="mean"` and PCA-reduced curve targets; cross-model HPO may compare GCN, GAT, and Transformer. MLP is not a supported fallback for this input contract.
- Field targets are flattened as `[samples, nodes, frames * components]` for model training, then reconstructed for diagnostics.
- Field `.npz` files are expected to contain arrays such as `Y`, `U`, `field`, or `values`, plus useful metadata keys when available (`frame_values`, `node_labels`, `coords0`, `components`, `valid_mask`, `sample_ids`).
- UT field outputs must use main-body nodes only. Grip-section nodes are cropped to match the input-processing convention.

## Target Dual-Output Architecture

- The intended unified surrogate has one nodal disorder/geometry input and separate UT and FT branches.
- Each branch is serial: a disorder-to-field Transformer predicts `u(x,y,t)` and `v(x,y,t)`, then a field-to-curve Transformer predicts the corresponding global response curve.
- UT produces a displacement field followed by a stress-strain curve. FT produces a displacement field followed by a force-displacement curve.
- Do not replace this with parallel field and curve readouts from the same latent representation. The learned displacement field is the required intermediary.
- Direct disorder-to-curve models have been tried and have not learned the relationship adequately.
- The main unresolved implementation problem is joint training with two objectives acting at different depths. The field loss directly supervises the first Transformer; the curve loss supervises the second Transformer and backpropagates through both stages.
- Conceptually, the objective is `L_total = sum_m(lambda_field,m * L_field,m + lambda_curve,m * L_curve,m)` for `m in {UT, FT}`.
- `MaskedFieldMSELoss` is the current pointwise field baseline and needs future development. `CombinedCurveLoss` is the current full-curve objective. Loss weighting/balancing remains undecided.
- Treat this as a documented research target rather than implemented behavior until the data are accessible and the user explicitly resumes model development.

## Saved Artifacts

- `MODEL.save()` writes a model checkpoint, model JSON, and DATA sidecar JSON.
- `MODEL.save_results()` writes `metrics.json`, `predictions.npz`, `loss_history.csv` when training logs exist, diagnostic CSVs, and `diagnostics_summary.json`.
- Treat `predictions.npz` as saved prediction/truth arrays for visualization and diagnostics, not as model weights. Prefer using it for post-processing so saved runs can be inspected without a model forward pass.
- HPO runs use Optuna `full_study.db` plus `best_params.json`, `best_trial_user_attrs.json`, and, when enabled, `best_model.*` with `best_model_results/`.
- HPO post-processing should strongly prefer `best_model.*` and `best_model_results/` when present, while remaining flexible to missing or varied artifacts.
- Standard run layout is `RUN_ROOT/{UT|FT|MULTI}/{Curve|Field|FieldToCurve}/{Model}/{run_descriptor}`.
- Model-specific HPO layout is `RUN_ROOT/{Task}/{Curve|Field|FieldToCurve}/{Model}/HPO/{run_descriptor}`.
- Cross-model HPO layout is `RUN_ROOT/{Task}/{Curve|Field|FieldToCurve}/HPO/{run_descriptor}/{Model}`.
- If adding or changing an output-layout token, update `MODEL` save paths, model JSON `run_layout`, `MLmetrics` run listing/HPO path parsers, post-processing saved-output-kind inference, and `B3_ML-transfer.sh` examples/prompts together.

## How To Work Here

- Read the relevant notebook/script and the shared helper it calls before editing. Most p2 behavior is controlled by helper functions, not by notebook cells alone.
- Preserve notebook separation: training notebooks train, post-processing notebooks inspect one saved run, and `ML-HPOpostProcess.ipynb` compares HPO studies.
- For broad notebook edits, ask before rewriting multiple cells. The user prefers guided, cell-by-cell cleanup when notebooks are under active refactor.
- Keep active notebooks, scripts, and helpers clean and current. Remove stale code paths, old workaround helpers, failed branches, duplicated cells, and superseded experiment scaffolding as part of the change.
- Keep path, model, run, split, and visualization controls explicit in notebooks. Avoid hidden module-level globals in `resources`.
- Prefer improving shared post-processing helpers in `resources/MLmetrics.py` when several notebooks need the same diagnostic behavior.
- When changing training or HPO scripts, keep local/debug `--allow-cpu` behavior separate from production GPU/HPC behavior.
- Validate syntax for Python and shell scripts after edits. Notebook validation can use import checks or targeted cell inspection when full execution requires data or HPC resources.
