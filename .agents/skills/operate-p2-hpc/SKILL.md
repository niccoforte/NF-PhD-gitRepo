---
name: operate-p2-hpc
description: Safely edit, review, prepare, resume, transfer, or diagnose the p2 QMUL HPC and Slurm workflow. Use for p2-DisorderML/HPC shell scripts, Python entry points, production versus debug defaults, scratch/archive paths, Slurm names, HPO layouts, resume behavior, saved-run transfer, or cluster validation.
---

# Operate P2 HPC Workflows

Use this procedure for cluster-facing p2 work. Do not submit, resume, transfer, or clean up a job unless the user has requested that external operation and every target is confirmed.

1. Read `p2-DisorderML/HPC/AGENTS.md` and the relevant section of [references/hpc-workflow.md](references/hpc-workflow.md).
2. Inspect the actual wrapper and Python entry point; the reference is routing context, not proof of current code.
3. Preserve production defaults. Express smoke/debug behavior only through explicit CLI overrides.
4. For path or naming changes, trace scratch, archive, metadata, resume, diagnostics, and transfer consumers. Invoke `review-p1-p2-data-contract` when data or saved-run contracts are affected.
5. Use dry-run modes where available before any authorized external operation.
6. Run `python tools/validate_repo.py --changed`; add focused argument/path checks and report Bash, Slurm, GPU, data, and job execution separately.
7. Invoke `maintain-repo-guidance` if the workflow, reference, commands, or current handoff state changes.
