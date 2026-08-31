# HPC Agent Context

This directory is the QMUL HPC/Slurm side of p2 training, HPO, resume, archive, and transfer. Read `../PROJECT_STATUS.md` for the current p2 objective. Use the `operate-p2-hpc` skill for any edit, review, planned run, resume, transfer, or diagnosis here; its reference holds the detailed entry-point and path mappings on demand.

## Ownership

- `B0_ML-env-setup.sh` manages the GPU environment.
- `B1_ML-new.sh` stages and submits work through scratch to the archive.
- `B2_ML-resumeHPO.sh` resumes archived cross-model studies.
- `B3_ML-transfer.sh` downloads saved runs.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/` own their respective single-run and cross-model HPO entry points.

## Durable Guardrails

- Active entry points are production-oriented: full data and realistic training budgets by default. Small datasets, short epochs/trials, and CPU execution must be explicit debug overrides.
- Do not change Slurm partition, account, license, CPU, memory, or time policy without the user's target cluster policy.
- Preserve explicit scratch cleanup guards and verify every archive, resume, transfer, or cleanup target before use.
- Slurm job/run names, `Curve`/`Field`/`FieldToCurve` layouts, metadata, diagnostics, and transfer paths form a shared saved-run contract. Use `review-p1-p2-data-contract` when they change.
- Keep CLI arguments stable or provide a clear migration. Do not introduce local Windows paths except as documented transfer destinations.
- Do not submit Slurm jobs, resume studies, transfer archives, or alter external environments unless the user requested that operation.
- Keep active scripts lean; remove abandoned debug branches and obsolete compatibility paths.

## Validation

- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` from the repository root. It performs Python and shell syntax checks where available but does not execute Slurm, GPUs, transfers, or research workloads.
- For path changes, add focused dry-run checks for representative script forms and report any unavailable Bash/HPC checks.
