# P2 HPC Workflow Reference

Read only the section relevant to the task and confirm it against the actual scripts.

## Directory roles

- `B0_ML-env-setup.sh`: creates or refreshes the `nf-ml-gpu` environment.
- `B1_ML-new.sh`: stages `resources/` plus a selected Python entry point to scratch, runs it, and archives outputs.
- `B2_ML-resumeHPO.sh`: resumes archived cross-model Optuna studies; use `--dry-run` before launch.
- `B3_ML-transfer-windows.sh`: preserves the Windows/Git Bash download path to `Z:/p2` or a fallback. `B3_ML-transfer-mac.sh`: macOS Bash 3.2/rsync download to repo-root `data/`, with `--dry-run` and optional `SSH_CONTROL_PATH`. Both keep the saved-run relative tree.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/`: single-run and cross-model HPO entry points for each output family.
- `DualOutputs/`: joint single-run entry point, thin Trial 1 preset, active contract tests, and `A0-HPC-Dual-HPO.py`. Dual HPO runs/resumes through B1 using `--study-name` and explicit `--resume`, not legacy B2. It uses model-specific `MULTI/Dual/Transformer/HPO/<study>` storage, one archive lock, SQLite backups and fixed validation ranking. See the adjacent README for ranges, artifacts and recovery limitations.

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
- For new loss-trial routes, complete a GPU preflight through the same B1 diagnostics/archive path. Full runs may be submitted with `--dependency=afterok:<matching-preflight-id> --kill-on-invalid-dep=yes`; record dependencies and do not describe scheduler acceptance as GPU validation. If the preflight fails, inspect retained scratch/logs before any replacement submission.

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

- Dual architecture comparisons use the existing dual test runner's `--experiment`
  and optional `--base-model-json`. `DualOutputs/B4_Dual-experiments.sh` is a
  preview-first suite submitter, not a replacement staging/training workflow.
  Explicit `--submit` snapshots the clean checkout under the conventional home
  launch directory and queues a 4-hour all-mode GPU preflight plus thirteen
  240-hour B1 jobs with afterok/kill-on-invalid-dep. Keep each change
  separate, use unique labels and fixed split seed, retain all result/motion/design
  diagnostics and gate production runs on an authorised GPU preflight. See its
  README; private encoders do not mean complete task independence.

- `FieldOutputs/A0-HPC_Field-test.py`: production-default single run for GCN, GAT, GNN, or Transformer.
- `FieldOutputs/A0-HPC_Field-CrossModelHPO.py`: cross-model GCN/GAT/Transformer HPO.
- Field models use node-level output and `MaskedFieldMSELoss`; MLP is not compatible with this contract.
- Component selection and unloaded-frame retention are explicit CLI/config choices.
- No FT-specific crack-tip, notch, ligament, or similar input feature is currently established.
- `FieldOutputs/A0-HPC_Field-lossTrial.py` reuses the single-run runner with exact archived Transformer presets and fixed MSE checkpoint selection. `DualOutputs/A0-HPC-Dual-lossTrial.py` reuses Trial 1 with the fixed dual score. Select one loss variant per unique B1 job. B1 stages companion runners. Collect validation motion metrics and true-field curve comparisons as well as standard predictions; localisation weighting follows the first four variants, not a simultaneous new HPO. Retain existing scripts.

## Field-to-curve entry points

- `FieldToCurve/A0-HPC_FieldToCurve-test.py`: Transformer-first UT/FT single run with node-token inputs and mean pooling.
- `FieldToCurve/A0-HPC_FieldToCurve-CrossModelHPO.py`: GCN/GAT/Transformer comparison; MULTI is not implemented.
- Runs use field inputs and curve targets but save under `FieldToCurve`. Keep this token aligned across model metadata, HPO resolution, diagnostics, and transfer.
- PCA targets use latent MSE; full curves may use curve-aware losses. MLP requires a deliberate contract redesign before support.

## HPO, resume, and transfer

For dual single runs use `B1_ML-new.sh DualOutputs/A0-HPC-Dual-test.py`. Defaults are all samples, CUDA, 450 epochs, validation diagnostics, full curves, and a joint normalized-field/physical-curve objective. Explicit smoke overrides are `--nsims 64 --epochs 3 --batch 2 --no-range-split`; `--allow-cpu` is local/debug only. Outputs follow `MULTI/Dual/Transformer/<run-label>`. Do not route checkpoints through legacy `MODEL` or B2. Dual HPO uses B1 with `DualOutputs/A0-HPC-Dual-HPO.py`; `--resume` restores completed study history and retries an interrupted configuration from epoch one. A 230-hour budget leaves margin below the unchanged ten-day limit. Best physical diagnostics are under `HPO/<study>/best/results/`; the dual notebook consumes that directory. Transfer a study using the platform B3 script with `MULTI Dual Transformer HPO <study>`.

- Prefer the Slurm `-J` value as the HPO study descriptor unless the study folder intentionally differs.
- Model-specific HPO and cross-model HPO layouts are defined in the data-contract reference.
- Write serializable metadata when `ML_RUN_METADATA` is provided.
- Keep HPO spaces explicit and conservative; field batches are intentionally small.
- Cross-model transfer defaults to the whole comparison folder. A single model subfolder requires an explicit raw path or CLI selection.
- Before resume or transfer, verify task, output token, study/run name, model set, remaining trials, source, and destination.
