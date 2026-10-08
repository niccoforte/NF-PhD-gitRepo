# P2 HPC Workflow Reference

Read only the section relevant to the task and confirm it against the actual scripts.

## Directory roles

- Task briefs live in `p2-DisorderML/handoffs/`, outside executable HPC directories. Read only the selected brief; `HANDOFFS.md` is the routing index. Moving a brief never moves an entry point, archive or immutable job snapshot.
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

- Reuse the user's SSH control socket for commands/transfers and leave its master running. `ssh -O check` does not prove the remote transport is alive. `ControlPersist` is an idle-lifetime setting, not a network guarantee; optional `ServerAliveInterval=60` and `ServerAliveCountMax=5` belong on the user's next master-start command, not individual multiplexed clients. They detect stale transport but cannot survive Mac sleep/network loss. Do not restart a healthy master or change SSH config without permission.
- Submit `B1_ML-new.sh` from the intended task/output/model directory; it resolves `ML_SCRIPT` from an HPC filename, HPC-relative path, repository-relative path, or absolute path.
- B1 initialises `/etc/profile.d/modules.sh` when `module` is absent and exports Bash's `HOSTNAME` for site module logging. Verify this bootstrap for non-interactive SSH; do not assume login-shell functions reach Slurm.
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
  240-hour B1 jobs with afterok/kill-on-invalid-dep. Optional `--variants CSV`
  selects only requested recovery jobs, with winner_probe required before either
  curve fit. `--preflight-node NODE` tests an available affected execution node
  without changing full-job placement. Inspect node state/reason first; never
  bypass an administrative drain. A successful gate elsewhere does not prove
  that node is repaired. Inspect preview and prior IDs before submitting;
  do not repeat successful variants during recovery. Keep each change
  separate, use unique labels and fixed split seed, retain all result/motion/design
  diagnostics and gate production runs on an authorised GPU preflight. See its
  README; private encoders do not mean complete task independence.

  For the isolated sudden-motion experiment use `--variants baseline,sudden`:
  B4 selects the two-case `--sudden-suite` gate, then two 240-hour runs. Gain1,
  fixed split42 and the same anchor isolate weighting from derivative penalties.
  This is not part of the default thirteen cases. Check the prepared CPU tests,
  source revision and existing job manifest before submission; do not infer GPU
  execution or improved accuracy from preparation alone.

- `FieldOutputs/A0-HPC_Field-test.py`: production-default single run for GCN, GAT, GNN, or Transformer.
- `FieldOutputs/A0-HPC_Field-CrossModelHPO.py`: cross-model GCN/GAT/Transformer HPO.
- Field models use node-level output and `MaskedFieldMSELoss`; MLP is not compatible with this contract.
- Component selection and unloaded-frame retention are explicit CLI/config choices.
- No FT-specific crack-tip, notch, ligament, or similar input feature is currently established.
- `FieldOutputs/A0-HPC_Field-lossTrial.py` reuses the single-run runner with exact archived Transformer presets and fixed MSE checkpoint selection. `DualOutputs/A0-HPC-Dual-lossTrial.py` reuses Trial 1 with the fixed dual score. Select one loss variant per unique B1 job. B1 stages companion runners. Collect validation motion metrics and true-field curve comparisons as well as standard predictions; localisation weighting follows the first four variants, not a simultaneous new HPO. Retain existing scripts.

## Field-to-curve entry points

- Corrected dual curve-loss comparisons reuse `DualOutputs/A0-HPC-Dual-test.py`
  with `--experiment curve_true|curve_predicted` and explicit
  `--curve-loss-ablation combined_no_location|combined_soft`. Use the same frozen
  source/configuration anchor and split as the existing MSE controls; do not
  repeat them without a reason. Gate the four full fits with
  `A0-HPC-Dual-preflight.py --curve-loss-suite` through B1 (five one-epoch cases).
  B4's default thirteen-mode suite is unchanged and does not submit this new
  comparison. Record explicit arguments, dependencies and revision for each
  new label. Combined-versus-MSE changes more than the peak term; only the two
  combined variants isolate its contribution. Keep hard peak reporting.

- `FieldToCurve/A0-HPC_FieldToProperty-test.py` is an oracle property diagnostic,
  not a curve runner. It reuses the existing encoder/trainer and writes
  `MULTI/FieldToProperty/Transformer/<label>`. B1 discovers its `.mdl` for log
  collection; B3 supports its relative path. Preserve explicit split/config
  anchors, unique-ID property joins, train-only normalization and validation-only
  selection. A 64-pair one-epoch preflight gates any authorised full run; neither
  scheduler acceptance nor a preflight establishes predictive accuracy.

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
