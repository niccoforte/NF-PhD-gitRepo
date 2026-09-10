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
- Slurm job/run names, `Curve`/`Field`/`FieldToCurve`/`Dual` layouts, metadata, diagnostics, and transfer paths form a shared saved-run contract. Use `review-p1-p2-data-contract` when they change. Dual runs use `MULTI/Dual/Transformer/<run-label>` and the existing staging/archive/transfer scripts, but their checkpoint loader is `DUAL_MODEL`, not legacy `MODEL`.
- Keep CLI arguments stable or provide a clear migration. Do not introduce local Windows paths except as documented transfer destinations.
- Do not submit Slurm jobs, resume studies, transfer archives, or alter external environments unless the user requested that operation.
- Keep active scripts lean; remove abandoned debug branches and obsolete compatibility paths.
- Dual HPO uses `MULTI/Dual/Transformer/HPO/<study>` with a scratch SQLite DB, consistent archive backups, an exclusive archive lock and explicit study-level resume. Do not pass it through legacy B2. Never clear a stale lock without verifying the recorded job stopped. Resume retries an interrupted configuration from epoch one; it is not optimizer-state continuation.
- Rank dual HPO by fixed equal-task, physical, train-mean-baseline-relative validation MSE; tuned loss weights must not change ranking. Keep full paired data, splits, normalization, masks and all four positive supervision weights. Default 230-hour Python budget leaves margin under the unchanged ten-day Slurm limit; checkpoints and DB backups protect unexpected termination.

## Validation

- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` from the repository root. It performs Python and shell syntax checks where available but does not execute Slurm, GPUs, transfers, or research workloads.
- For path changes, add focused dry-run checks for representative script forms and report any unavailable Bash/HPC checks.
