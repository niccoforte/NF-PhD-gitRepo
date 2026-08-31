# P2 HPC Workflow Reference

Read only the section relevant to the task and confirm it against the actual scripts.

## Directory roles

- `B0_ML-env-setup.sh`: creates or refreshes the `nf-ml-gpu` environment.
- `B1_ML-new.sh`: stages `resources/` plus a selected Python entry point to scratch, runs it, and archives outputs.
- `B2_ML-resumeHPO.sh`: resumes archived cross-model Optuna studies; use `--dry-run` before launch.
- `B3_ML-transfer.sh`: copies saved runs from the HPC archive to `Z:/p2` or a fallback local folder.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/`: single-run and cross-model HPO entry points for each output family.

## Production and debug policy

- Active entry points default to full data (`--nsims all`) and production-scale epochs, scheduler patience, and early stopping.
- Reduced samples, epochs, trials, or CPU execution are opt-in CLI overrides such as `--nsims 64`, `--epochs 3`, `--n-trials-per-typ 2`, or `--allow-cpu`.
- Do not label active scripts or default examples as smoke runs. A smoke example must show the arguments that make it small.
- Production cluster jobs should fail if CUDA is unavailable; `--allow-cpu` is for local/debug use.

## Submit, scratch, and archive contract

- Submit `B1_ML-new.sh` from the intended task/output/model directory; it resolves `ML_SCRIPT` from an HPC filename, HPC-relative path, repository-relative path, or absolute path.
- `DATA_ROOT` is the parent containing `MLdata`, not the `MLdata` directory itself.
- `ML_RUN_ROOT` is scratch; `ARCHIVE_ROOT` receives final rsync output; `ML_ARCHIVE_ROOT` records that mapping in metadata.
- The Slurm `-J` value becomes the default `ML_JOB_NAME` and archive/run label. Explicit `ARCHIVE_ROOT`, `ML_JOB_NAME`, `RUN_LABEL`, `--run-label`, or `--study-name` overrides take precedence.
- `ML_RUN_CONTEXT=HPC` records context but must not prefix or otherwise change run names.
- Preserve the explicit scratch-path cleanup guard and `MPLBACKEND=Agg` non-interactive behavior.

## Slurm assumptions

- The submit wrapper currently loads `miniforge`, activates `nf-ml-gpu`, and requests GPU resources.
- Active directives currently target `andrena`/`pilot_andrena`; alternatives are commented. Change partition, account, time, CPU, memory, or license settings only with a supplied target policy.
- Enabling `sae` requires the matching `pilot_sae_gpu` account.
- Keep `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` aligned with the Slurm CPU allocation.

## Curve entry points

- `CurveOutputs/A0-HPC_Curve-test.py`: production-default single run for MLP, graph models, or Transformer.
- `CurveOutputs/A0-HPC_Curve-CrossModelHPO.py`: cross-model UT/FT HPO, with optional PCA output reduction.
- MLP uses flattened inputs; graph/Transformer models preserve node-shaped inputs and geometry features when configured.
- Full curves may use MSE or `CombinedCurveLoss`; PCA-reduced targets use latent-space MSE.
- Script defaults currently define curve zone boundaries/weights. Treat changes as scientific unless authority is established.

## Field entry points

- `FieldOutputs/A0-HPC_Field-test.py`: production-default single run for GCN, GAT, GNN, or Transformer.
- `FieldOutputs/A0-HPC_Field-CrossModelHPO.py`: cross-model GCN/GAT/Transformer HPO.
- Field models use node-level output and `MaskedFieldMSELoss`; MLP is not compatible with this contract.
- Component selection and unloaded-frame retention are explicit CLI/config choices.
- No FT-specific crack-tip, notch, ligament, or similar input feature is currently established.

## Field-to-curve entry points

- `FieldToCurve/A0-HPC_FieldToCurve-test.py`: Transformer-first UT/FT single run with node-token inputs and mean pooling.
- `FieldToCurve/A0-HPC_FieldToCurve-CrossModelHPO.py`: GCN/GAT/Transformer comparison; MULTI is not implemented.
- Runs use field inputs and curve targets but save under `FieldToCurve`. Keep this token aligned across model metadata, HPO resolution, diagnostics, and transfer.
- PCA targets use latent MSE; full curves may use curve-aware losses. MLP requires a deliberate contract redesign before support.

## HPO, resume, and transfer

- Prefer the Slurm `-J` value as the HPO study descriptor unless the study folder intentionally differs.
- Model-specific HPO and cross-model HPO layouts are defined in the data-contract reference.
- Write serializable metadata when `ML_RUN_METADATA` is provided.
- Keep HPO spaces explicit and conservative; field batches are intentionally small.
- Cross-model transfer defaults to the whole comparison folder. A single model subfolder requires an explicit raw path or CLI selection.
- Before resume or transfer, verify task, output token, study/run name, model set, remaining trials, source, and destination.
