# HPC Agent Context

This directory is the QMUL HPC/Slurm side of p2 training, HPO, resume, archive, and transfer. Read `../PROJECT_STATUS.md` for the current p2 objective. Use the `operate-p2-hpc` skill for any edit, review, planned run, resume, transfer, or diagnosis here; its reference holds the detailed entry-point and path mappings on demand.

## Ownership

- `B0_ML-env-setup.sh` manages the GPU environment.
- `B1_ML-new.sh` stages and submits work through scratch to the archive.
- `B2_ML-resumeHPO.sh` resumes archived cross-model studies.
- `B3_ML-transfer-windows.sh` preserves the Windows/Git Bash download path. `B3_ML-transfer-mac.sh` supports macOS Bash 3.2, rsync/ControlMaster, `--dry-run`, and defaults to ignored repo-root `data/`. Neither transfer script deletes source data.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/` own their respective single-run and cross-model HPO entry points.
- `DualOutputs/A0-HPC-Dual-test.py` owns the joint UT/FT single run. `A0-HPC-Dual-trial1.py` is a thin preset calling it; B1 stages both. `A0-HPC-Dual-HPO.py` calls `resources/MLdualHPO.py` for single-worker full-data HPO and explicit `--resume` through B1. The adjacent README owns search ranges, ranking and recovery instructions. `test_dual_contract.py` protects active invariants, including HPO and saved-review adapters.

## Durable Guardrails

- Active entry points are production-oriented: full data and realistic training budgets by default. Small datasets, short epochs/trials, and CPU execution must be explicit debug overrides.
- Do not change Slurm partition, account, license, CPU, memory, or time policy without the user's target cluster policy.
- Preserve explicit scratch cleanup guards and verify every archive, resume, transfer, or cleanup target before use.
- B1 must retain scratch and return failure if copying run outputs or job logs to the archive fails, even when Python training succeeded.
- B1 initialises `/etc/profile.d/modules.sh` if non-interactive submission has not supplied `module`; do not rely on an interactive login environment.
- Slurm job/run names, `Curve`/`Field`/`FieldToCurve`/`Dual` layouts, metadata, diagnostics, and transfer paths form a shared saved-run contract. Use `review-p1-p2-data-contract` when they change. Dual runs use `MULTI/Dual/Transformer/<run-label>` and the existing staging/archive/transfer scripts, but their checkpoint loader is `DUAL_MODEL`, not legacy `MODEL`.
- Keep CLI arguments stable or provide a clear migration. Do not introduce local Windows paths except as documented transfer destinations.
- Do not submit Slurm jobs, resume studies, transfer archives, or alter external environments unless the user requested that operation.
- Keep active scripts lean; remove abandoned debug branches and obsolete compatibility paths.
- Dual HPO uses `MULTI/Dual/Transformer/HPO/<study>` with a scratch SQLite DB, consistent archive backups, an exclusive archive lock and explicit study-level resume. Do not pass it through legacy B2. Never clear a stale lock without verifying the recorded job stopped. Resume retries an interrupted configuration from epoch one; it is not optimizer-state continuation.
- Rank dual HPO by fixed equal-task, physical, train-mean-baseline-relative validation MSE; tuned loss weights must not change ranking. Keep full paired data, splits, normalization, masks and all four positive supervision weights. Default 230-hour Python budget leaves margin under the unchanged ten-day Slurm limit; checkpoints and DB backups protect unexpected termination.

## Validation

- `DualOutputs/HANDOFFS.md` owns task routing, pending notebook/experiment work, per-test visual review and the read-only storage audit; passing automated checks is not researcher sign-off. Improve ML Accuracy owns the agreed, not yet implemented soft/soft peak correction. Old localisation weighting is target activity, not affine-invariant anomaly detection; regional FT weighting is a separate control. Confirm saved loss classes/nonzero coefficients before claiming a run used custom losses.

- The dual test runner's `--experiment` prepares one-change baseline/crack-face/local-graph/encoder-sharing/interface/residual/localisation comparisons. `--base-model-json` uses a shared non-graph dual configuration anchor, not its weights; fixed split seed 42 and unchanged balanced selection make variants comparable. Parameter counts and split hashes are mandatory. Run local checks first, then an authorised B1 GPU preflight before full jobs; preparation is not submission. See DualOutputs README for exact semantics and deferred work.
- `DualOutputs/B4_Dual-experiments.sh` previews by default; explicit `--submit` creates an immutable home Git snapshot and submits the thirteen-mode matched suite through B1. `A0-HPC-Dual-preflight.py` runs explicit 64-pair/one-epoch GPU checks for every mode; all full jobs depend on its successful archiving. Fresh labels and a clean remote checkout are required. `jobs.tsv` records dependencies. The suite adds late-frame and FT-region weighting, winner substitution, and fresh frozen-source curve fits; it excludes old activity localisation. Curve-only fits select using two curve scores, all other training retains four-output selection. No test-set development or source-HPO changes.

- Preserve existing approved trial, test and HPO entry points unless the user explicitly requests their removal. Loss ablations use `FieldOutputs/A0-HPC_Field-lossTrial.py` (exact archived UT/FT architecture/training presets) and `DualOutputs/A0-HPC-Dual-lossTrial.py` (Trial 1 capacity). B1 stages their existing runner dependencies. Choose `--field-loss-variant baseline|spatial|temporal|both`; `weighted` additionally requires an explicit positive gain and follows evidence-based variant selection. Defaults remain full data/450 epochs, validation-only selection, and the existing resource policy. No damage/strain export or architecture/HPO extension is implied.
- Independent ablations select checkpoints by unchanged normalised validation MSE; dual ablations use the unchanged balanced physical validation score. Always use unique run labels. Save field-loss components, motion diagnostics and paired true/predicted-field curve checks alongside standard results. Different single/dual populations must not be treated as a matched comparison.
- Gate full loss trials on a successful real-data/GPU B1 preflight for their family, including diagnostics and archiving. Slurm `afterok` with `--kill-on-invalid-dep=yes` may queue the full trials safely; record preflight and dependent job IDs separately. A queued/started preflight is not proof of success.

- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` from the repository root. It performs Python and shell syntax checks where available but does not execute Slurm, GPUs, transfers, or research workloads.
- For path changes, add focused dry-run checks for representative script forms and report any unavailable Bash/HPC checks.
