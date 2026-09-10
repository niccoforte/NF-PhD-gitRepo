# P2 HPC Workflow Reference

Read only the section relevant to the task and confirm it against the actual scripts.

## Directory roles

- `B0_ML-env-setup.sh`: creates or refreshes the `nf-ml-gpu` environment.
- `B1_ML-new.sh`: stages `resources/` plus a selected Python entry point to scratch, runs it, and archives outputs.
- `B2_ML-resumeHPO.sh`: resumes archived cross-model Optuna studies; use `--dry-run` before launch.
- `B3_ML-transfer-windows.sh`: preserves the Windows/Git Bash download path to `Z:/p2` or a fallback. `B3_ML-transfer-mac.sh`: macOS Bash 3.2/rsync download to repo-root `data/`, with `--dry-run` and optional `SSH_CONTROL_PATH`. Both keep the saved-run relative tree.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/`: single-run and cross-model HPO entry points for each output family.
- `DualOutputs/`: joint UT/FT two-stage single-run entry point, thin `A0-HPC-Dual-trial1.py` preset and active synthetic contract tests; no dual HPO entry point yet. The trial preset reuses `A0-HPC-Dual-test.py`, so B1 stages that companion script too. See the adjacent README for HPO provenance and parameter choices.

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
- `ML_SOURCE_REVISION` records the staged checkout's Git revision. The dual runner additionally hashes its script and model module because uncommitted edits are not represented by the revision alone.
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

For dual single runs use `B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py`. Defaults are all samples, CUDA, 450 epochs, validation diagnostics, full curves, and a joint normalized-field/physical-curve objective. Explicit smoke overrides are `--nsims 64 --epochs 3 --batch 2 --no-range-split`; `--allow-cpu` is local/debug only. Outputs follow `MULTI/Dual/Transformer/<run-label>` and include one `model.mdl`, JSON metadata, loss history, physical predictions/masks and existing field/curve diagnostic CSVs. Best checkpoints and history are written during training so the launcher can collect partial results on failure. Do not route these checkpoints through legacy `MODEL`, existing single-task notebooks, or `B2_ML-resumeHPO.sh`; dual HPO/resume remains future work. Transfer uses the standard `B3_ML-transfer-windows.sh MULTI Dual Transformer <run-label>` path.

- Prefer the Slurm `-J` value as the HPO study descriptor unless the study folder intentionally differs.
- Model-specific HPO and cross-model HPO layouts are defined in the data-contract reference.
- Write serializable metadata when `ML_RUN_METADATA` is provided.
- Keep HPO spaces explicit and conservative; field batches are intentionally small.
- Cross-model transfer defaults to the whole comparison folder. A single model subfolder requires an explicit raw path or CLI selection.
- Before resume or transfer, verify task, output token, study/run name, model set, remaining trials, source, and destination.
