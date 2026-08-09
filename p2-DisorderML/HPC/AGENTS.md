# HPC Agent Context

This directory contains the cluster-facing training workflow for p2 ML models. The scripts are designed for QMUL HPC/Slurm usage and should stay robust under scratch execution, GPU allocation, and result archiving.

## Directory Roles

- `B0_ML-env-setup.sh` creates or refreshes the `nf-ml-gpu` conda environment using Miniforge, CUDA PyTorch wheels, PyTorch Geometric, Optuna, GPytorch, BoTorch, and related ML dependencies.
- `B1_ML-new.sh` is the main Slurm submit wrapper. It selects a Python run script, copies `resources/` and that script to scratch, runs on scratch, and rsyncs outputs to the archive root.
- `B3_ML-transfer.sh` copies saved run folders from the remote HPC archive to the local `Z:/p2` tree or a fallback local folder.
- `CurveOutputs/` contains curve-output single-run training and cross-model HPO entry points.
- `FieldOutputs/` contains nodal displacement field single-run training and cross-model HPO entry points.
- `FieldToCurve/` contains field-input to curve-output single-run training and cross-model HPO entry points. These runs save under the `FieldToCurve` output-layout token, not under `Curve`.

## Default Run Policy

- Treat every active HPC Python entry point as a final, production-oriented script. Defaults must be suitable for a real cluster run, not a smoke test.
- Default dataset size must be full data, normally `--nsims all`. Do not set limited-sample defaults such as `64`, `128`, or other debug subsets in active HPC scripts.
- Default training budgets must be production-scale for that output family: full epoch counts, realistic scheduler patience, and realistic early-stopping patience. Short runs belong only in explicit command-line overrides.
- Smoke tests, quick checks, reduced datasets, shortened epochs, tiny HPO trials, and local debug runs are opt-in behaviors supplied by the user at submit time, for example with `--nsims 64`, `--epochs 3`, `--n-trials-per-typ 2`, or `--allow-cpu`.
- Do not describe active HPC Python scripts, Slurm examples, or transfer examples as "smoke" scripts/runs. If a smoke example is useful, it must visibly include the command-line args that make it a smoke test.
- When adding a new HPC Python script, start from full-run defaults first, then expose CLI arguments for smaller debug runs. Never invert that relationship.

## Cluster Workflow

- `B1_ML-new.sh` expects to be submitted from a task/output/model run directory, for example `/data/home/exy053/p2/UT/Curve/MLP`, but copies Python code from `REPO_ROOT/p2-DisorderML/HPC`.
- `ML_SCRIPT` can be a filename under `HPC`, a relative path such as `CurveOutputs/A0-HPC_Curve-test.py`, a repo-relative path, or an absolute path.
- `DATA_ROOT` must be the parent directory containing `MLdata`; do not point it directly at `MLdata`.
- `ARCHIVE_ROOT` is where final runs are rsynced. `ML_RUN_ROOT` points to scratch during the job, and `ML_ARCHIVE_ROOT` records the archive mapping in saved metadata.
- `B1_ML-new.sh` uses the Slurm `-J` name as the default archive folder and exports it as `ML_JOB_NAME`; HPO scripts may use `ML_JOB_NAME` as their default study/folder name. For single-run scripts that expose `--run-label`, `B1_ML-new.sh` forwards the Slurm `-J` name as `--run-label` unless `RUN_LABEL` or script-level `--run-label` is explicitly provided. Explicit `ARCHIVE_ROOT`, `ML_JOB_NAME`, `RUN_LABEL`, `--run-label`, or `--study-name` overrides take precedence.
- Saved run paths use `{Task}/{Curve|Field|FieldToCurve}/{Model}/{Run}`. `B3_ML-transfer.sh` should accept and document all three output-layout tokens.
- For cross-model HPO transfer, `B3_ML-transfer.sh` should download the whole comparison folder `{Task}/{OutputKind}/HPO/{Study}` by default. A single model subfolder can still be transferred by raw path or explicit CLI form, but the interactive `compare-hpo` flow should not require a model name.
- `ML_RUN_CONTEXT=HPC` is exported so scripts can record/handle HPC context, but it must not alter run names. Current Slurm `-J`/`RUN_LABEL` names should be preserved exactly; do not add any cluster/context prefix.
- `MPLBACKEND=Agg` is set for non-interactive plotting. Do not introduce GUI-only plotting requirements in HPC scripts.
- Scratch cleanup is guarded by an explicit path check. Preserve that safety check if changing scratch behavior.

## Current Slurm Assumptions

- The current submit script loads `miniforge`, activates `nf-ml-gpu` by default, and uses GPU Slurm settings.
- The file currently includes active `andrena`/`pilot_andrena` directives and commented alternatives for other GPU partitions. Do not change partition/account/time directives unless the user asks or provides the target cluster policy.
- The `sae` GPU partition requires the matching billing account when enabled: `#SBATCH -p sae` with `#SBATCH -A pilot_sae_gpu`.
- Keep CPU thread exports (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`) aligned with Slurm CPU allocation.

## Curve Entry Points

- `CurveOutputs/A0-HPC_Curve-test.py` is a single-run training entry point for MLP, GCN, GAT, GNN, and Transformer curve models. It defaults to all simulations and production-scale epochs; pass smaller `--nsims`/`--epochs` explicitly for debug runs.
- `CurveOutputs/A0-HPC_Curve-CrossModelHPO.py` runs cross-model curve HPO over MLP, GCN, GAT, and Transformer by default.
- `CurveOutputs/A0-HPC_Curve-CrossModelHPO.py` supports UT and FT full-curve HPO by default, and PCA-reduced curve-output HPO with `--output-reduction pca` plus either `--pca-components` or `--pca-accuracy`.
- Curve data uses `output_kind="curve"`, `scale=("symm", "inout")`, `d_data="in"`, and commonly `mechMode="UT"`.
- MLP uses flattened inputs. GCN/GAT/Transformer use node-shaped inputs with `geom_feats` enabled by default for node models.
- Curve losses may use plain MSE or `CombinedCurveLoss` terms for pointwise, weighted-zone, derivative, peak, energy, and soft peak-location behavior.
- PCA-reduced curve-output HPO should use latent-space MSE only; curve-aware combined losses should stay on full curve outputs.
- Default curve zone weighting uses UT boundaries `(65, 130)`, FT boundaries `(85, 160)`, and weights `(1.0, 5.0, 0.2)` unless a script-specific config overrides it.

## Field Entry Points

- `FieldOutputs/A0-HPC_Field-test.py` is a single-run training entry point for GCN, GAT, GNN, and Transformer field models. It defaults to all simulations and production-scale epochs; pass smaller `--nsims`/`--epochs` explicitly for debug runs.
- `FieldOutputs/A0-HPC_Field-CrossModelHPO.py` runs cross-model field HPO over GCN, GAT, and Transformer by default.
- Field data uses `output_kind="field"`, `field_config={"components": ..., "drop_frame0": ..., "layout": "auto"}`, and `geom_feats=(True, True)` by default.
- MLP is not field-compatible in the current framework. Do not add it to field HPO without changing the underlying data/model contract.
- Field models must use node-level outputs: GNN `pool="node"` and Transformer `pool="node"`.
- Use `MaskedFieldMSELoss` for field targets, especially when targets contain NaNs or invalid-mask entries.
- `--components` supports values such as `U1,U2`, a single component, or `Umag` if both `U1` and `U2` are available.
- `--keep-frame0` controls whether the unloaded frame is retained. The default is to drop frame 0.
- FT field HPO currently uses the same node/geometry feature contract as the rest of the field framework. No explicit crack-tip distance, notch-region flag, ligament coordinate, or other FT-specific input feature has been added yet.

## Field-To-Curve Entry Points

- `FieldToCurve/A0-HPC_FieldToCurve-test.py` is the Transformer single-run training entry point for mapping UT or FT field inputs to curve outputs. It defaults to all simulations and production-scale epochs; pass smaller `--nsims`/`--epochs` explicitly for debug runs.
- `FieldToCurve/A0-HPC_FieldToCurve-CrossModelHPO.py` runs cross-model field-to-curve HPO over GCN, GAT, and Transformer for UT or FT separately. MULTI is not implemented in this entry point yet.
- Field-to-curve data uses `input_kind="field"`, `output_kind="curve"`, `field_input_config={"components": ..., "drop_frame0": ..., "layout": "auto"}`, and `scale=("symm", "inout")`. The single-run script can use full curves with `--output-reduction none`; PCA output runs use `reduce_dim=("PCA", "out", None, 16, True)` when PCA is enabled without an explicit component/accuracy override.
- Field-to-curve runs should save under `{Task}/FieldToCurve/{Model}/{Run}` even though their `DATA.output_kind` is `"curve"`. Keep `MODEL` save layout, saved model `run_layout`, post-processing path parsers, HPO path resolvers, and transfer helpers in sync with this token.
- The field-to-curve single-run path is Transformer-first, using node-token input shape `[samples, nodes, features]` and `pool="mean"`. PCA outputs use latent-space MSE; full-curve outputs may use curve-aware losses. Cross-model HPO may compare GCN, GAT, and Transformer because all three preserve node-token inputs and then globally pool to the curve latent. Do not add MLP support unless the data/model contract is deliberately redesigned.
- Field-to-curve HPO study names should preserve the Slurm `-J`/`ML_JOB_NAME` value exactly unless the user passes `--study-name`.

## HPO Rules

- `hOpt_model()` saves model-specific HPO under `{Task}/{Curve|Field|FieldToCurve}/{Model}/HPO/{Study}`.
- `hOpt_compare()` saves cross-model HPO under `{Task}/{Curve|Field|FieldToCurve}/HPO/{Study}/{Model}`.
- For `B1_ML-new.sh` HPO submissions, prefer setting `-J` to the intended study descriptor and omit `--study-name` unless the HPO folder should differ from the Slurm/archive name.
- HPO scripts should write serializable run metadata when `ML_RUN_METADATA` is set.
- Keep HPO search spaces explicit and conservative for cluster jobs. Batch sizes for field runs are intentionally small because field targets are large.
- Use `--allow-cpu` only for local/debug runs. Production cluster runs should fail loudly if CUDA is unavailable.

## Editing And Validation

- Keep command-line arguments stable unless there is a clear migration path. These scripts are launched through `sbatch B1_ML-new.sh ...`.
- Keep active HPC scripts lean and current. Remove abandoned debug paths, stale run-script branches, and unused compatibility scaffolding unless a current submit or transfer workflow still needs them.
- If changing path logic, test representative `ML_SCRIPT` forms: subfolder path, repo-relative path, and absolute path if feasible.
- If changing Python entry points, run a syntax check such as `python -m py_compile` on the modified scripts.
- If changing shell scripts, review with `bash -n` where available.
- Do not introduce local Windows paths into HPC scripts except in comments describing transfer destinations.
