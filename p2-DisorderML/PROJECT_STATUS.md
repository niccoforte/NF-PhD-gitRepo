# p2 Implementation Status

This is a compact repository handoff, not a second research plan or an Obsidian mirror. Scientific objectives, manuscript reasoning and supervisor decisions remain in the vault's `Writing/p2/` and `Writing/Research Plans/`. Durable contracts belong in `AGENTS.md`; completed changes belong in Git history.

## Active implementation

- Opt-in joint two-stage UT/FT model: `resources/MLdual.py`. Each stage uses one shared encoder call with task streams stacked along the batch dimension. Field and curve losses train both stages through one optimizer.
- Legacy `DATA`, `MODEL`, separate-task networks and notebooks retain their existing paths.
- Real-data runner and HPO-informed trial preset: `HPC/DualOutputs/`. The adjacent README owns run commands, exact HPO sources and the configuration rationale.
- Dual notebooks: `code/ML-DualOutputs.ipynb` follows data/HPO/model configuration with opt-in execution; `code/ML-DualPostProcessing.ipynb` contains full curve and field diagnostic sections, interactive field viewing and log-scale task losses. Both default to the user's renamed local Trial 1 folder; worked inputs belong in `samples/`.
- Human review material: `samples/README.md`, including controlled examples and three actual HPC input reports. Shared reference features and task-specific body-interface/pin/crack features are active. Degree and graph-relative attention remain explanatory examples, not active inputs.
- Existing single-task/HPO notebooks and `HPC/FieldOutputs/`, `HPC/FieldToCurve/` remain the independent baseline surfaces. `code/TOKENIZATION_NEXT_STEPS.md` owns the separate exploratory tokenisation handoff.

## Verified on 7 September 2026

- Remote MULTI headers contain UT `(8811,21,800,2)` and FT `(8811,21,788,2)`, aligned by sample ID. The adapter uses reference-coordinate mapping, preserves the canonical order and masks twelve absent FT nodes. Header counts alone do not establish final complete-case/split counts.
- Slurm job **25868425**, `dual-MULTI-test-260907`, completed successfully: 64 selected pairs, three epochs, one A100, validation diagnostics. Home staging, scratch execution and archive collection all ran. See `samples/hpc-test-report.md` for evidence and limitations.
- The archive is `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/dual-MULTI-test-260907/`. An isolated home code snapshot was used; the remote Git checkout and ML source data were not edited.
- Nine synthetic contract tests passed, covering encoder counts/calls, masks, joint gradients, affine loss reconstruction, checkpoints, reordered FT mapping, pin calculations, trial configuration and saved artifacts.
- Independent UT/FT field and full-201-point field-to-curve HPO parameter records were recovered. Trial 1 is a transparent capacity/lr compromise, not a dual-HPO optimum.
- Three-epoch smoke predictions remain worse than training-mean baselines. Do not cite the execution test as accuracy, negative-transfer or optimisation evidence.

## Deployment on 10 September 2026

- Full-data Trial 1, Slurm **26267130**, `dual-MULTI-trial1-260910`, completed successfully in 3 h 37 min. Downloaded to repo-root `data/`; the user renamed its local folder to `dual-MULTI-trial1`. Remote metadata retains the original dated name.
- Training implementation: Git `7da7e72`. Splits: 7,137 train / 793 validation / 881 locked test; 149 epochs logged, best checkpoint 74. Validation RMSE improvements over the training mean: UT field +5.47%, FT field -23.58%, UT curve +3.42%, FT curve +14.53%. See the dual HPC README for numerical context and limitations; raw weighted-loss sizes alone do not identify the weakest task.
- Nine dual contract tests passed. Repository contract validation finished with 64 non-failing checks and zero failures. Both saved-run review notebooks executed successfully; visible-output copies and figures are in the downloaded smoke run's `results/postProcessing/` directory.

- The completed smoke archive is also downloaded (locally renamed `dual-MULTI-test`); it remains execution evidence, not the default accuracy review. macOS transfer uses `HPC/B3_ML-transfer-mac.sh`; `-windows.sh` preserves the Windows route.
- Generated `data/` and `samples/` content remains ignored. Only `samples/AGENTS.md` is exempted to keep required guidance tracked. Tests no longer depend on ignored sample generators.
- Both Git hosting destinations received the implementation commit, and the clean HPC checkout was fast-forwarded through a verified incremental Git bundle because direct Git SSH authentication was unavailable. Its obsolete origin URL was corrected. No credentials were copied.

## Next evidence

- `resources/MLdualHPO.py` and `HPC/DualOutputs/A0-HPC-Dual-HPO.py` prepare a broad conditional, single-worker study with four positive task weights, fixed balanced validation ranking, periodic backups and study-level resume through B1. Default budget: 200 completed/pruned evaluations, up to 450 epochs, 230 hours. No HPO job has been submitted. Review the scope and confirm launch before any remote execution.
- Review the labelled pin/interface examples against the intended production geometry. The current profile uses the A1 Ti/Al proportions and the sample's initial disordered coordinates.
- Validate reconstructed lattice connectivity against one original producer/INP artifact before claiming agreement with archived meshes. Do not process thousands of INPs by default.
- HPO execution, repeated-seed finalist confirmation, locked-test evaluation, full-dimensional optimisation and new FEA verification remain outstanding. Study resume does not continue a partially trained optimizer; interrupted configurations restart from epoch one.
- Local checks cover the expanded architecture, strict reload of the actual Trial 1 checkpoint, HPO archive-to-new-scratch resume, exclusive-lock refusal and saved diagnostic adapters. Both full review routes (UT and FT field selection) executed on Trial 1 with inline plots; copies are in its ignored `results/postProcessing/`. Repository contract validation passed (67 non-failing, zero failures). No new GPU/HPO job was run.
- HPC code synchronization currently requires renewal of the user's SSH ControlMaster connection; a read-only check found the old connection broken. Do not claim this iteration is deployed remotely until the checkout hash is verified.
- Keep the field intermediary and full 201-point curves. User-reported true-field FT full-curve performance is promising; numerical manuscript claims still require the saved evidence.
- SSH access uses the user's active ControlMaster socket. Access availability is temporary and does not itself authorize arbitrary remote changes. Never store passwords in chat or repo files.
