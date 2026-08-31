---
name: validate-repo-change
description: Select and run proportionate validation for changes in this PhD repository. Use whenever Codex edits or reviews Python, Abaqus scripts, notebooks, shell/Slurm scripts, setup files, shared resources, data-processing contracts, ML training or post-processing code, or repository guidance; and before reporting that a change is complete.
---

# Validate Repository Changes

Use the repository entry point first:

```powershell
python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed
python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope root
python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope resources
python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope p1
python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope p2
python .agents/skills/validate-repo-change/scripts/validate_repo.py --scope contract
```

Run these commands from the repository root. The colocated script is non-destructive. It checks diff whitespace and guidance coverage; compiles selected Python without running modules; imports relevant shared modules in fresh processes; validates notebook JSON; runs `bash -n` and PowerShell parse checks when available; and validates the explicitly synthetic fixture under this skill's `fixtures/` directory. It reports environment setup/removal, Abaqus, HPC, external-data, and scientific execution as skipped rather than implying coverage.

## Add focused validation

1. Inspect `git status --short` and the complete diff.
2. Add the narrowest test or import that exercises the changed behavior without requiring unapproved external operations.
3. For data-contract changes, invoke `review-p1-p2-data-contract` and run the contract scope.
4. Use Abaqus Python for real model/ODB behavior only when the environment and task authorize it.
5. Do not install/remove Python, Abaqus, or Conda environments; submit Abaqus or Slurm jobs; transfer archives; upgrade ODBs; rename research files; or delete artifacts merely to claim validation.

Report passed, failed, syntax-only, and skipped checks separately. Local or synthetic validation is not scientific validation on real data.
