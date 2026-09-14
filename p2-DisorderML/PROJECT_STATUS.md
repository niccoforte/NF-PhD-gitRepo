# p2 Implementation Status

This is a compact repository handoff, not a second research plan or an Obsidian mirror. Scientific objectives, manuscript reasoning and supervisor decisions remain in the vault's `Writing/p2/` and `Writing/Research Plans/`. Durable contracts belong in `AGENTS.md`; completed changes belong in Git history.

## Active implementation

- Opt-in joint two-stage UT/FT model: `resources/MLdual.py`. Each stage uses one shared encoder call with task streams stacked along the batch dimension. Field and curve losses train both stages through one optimizer.
- Legacy `DATA`, `MODEL`, separate-task networks and notebooks retain their existing paths.
- Real-data runner and HPO-informed trial preset: `HPC/DualOutputs/`. The adjacent README owns run commands, exact HPO sources and the configuration rationale.
- Dual notebooks: `code/ML-DualOutputs.ipynb` directly constructs DAT/TR_DUAL or loads the saved checkpoint/results, with explicit fresh HPO and train/save/predict cells. `code/ML-DualPostProcessing.ipynb` displays both UT/FT field and curve branches, two live field viewers, log losses and the four-pair 2×2 error plot. `code/ML-HPOpostProcess.ipynb` is the shared single-task/cross-model/dual study review surface; no separate dual HPO notebook. Trial 1 remains the saved-run default; worked inputs belong in `samples/`.
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
- Nine dual contract tests passed at deployment. Repository contract validation finished with 64 non-failing checks and zero failures. Both saved-run review notebooks executed successfully; generated figures remain in the downloaded smoke run's `results/postProcessing/` directory.

- The completed smoke archive is also downloaded (locally renamed `dual-MULTI-test`); it remains execution evidence, not the default accuracy review. macOS transfer uses `HPC/B3_ML-transfer-mac.sh`; `-windows.sh` preserves the Windows route.
- Generated `data/` and `samples/` content remains ignored. Only `samples/AGENTS.md` is exempted to keep required guidance tracked. Tests no longer depend on ignored sample generators.
- Both Git hosting destinations received the implementation commit, and the clean HPC checkout was fast-forwarded through a verified incremental Git bundle because direct Git SSH authentication was unavailable. Its obsolete origin URL was corrected. No credentials were copied.

## Next evidence

- Notebook corrections verified on 14 September: all 15 dual contract tests passed, including fresh notebook build/train/save/predict, direct HPO dispatch, saved-model reconstruction, sample-ID alignment and coexistence of field viewers. Trial 1 Outputs executed all 14 cells successfully; DualPostProcessing executed all 40 cells with 41 inline plots and no errors. Validation used memory/outside-repository temporary files, not new notebook copies under data.
- Shared HPO notebook executed all 29 cells against an explicitly synthetic Optuna study and saved dual winner: 12 inline plots, no cell errors. The live HPC study was not downloaded or modified. Changed-surface validation passed with 13 non-failing checks; contract validation passed with 67 non-failing checks and zero failures. Abaqus/GPU execution remains outside these local checks.
- The five duplicate notebooks in `data/` were moved to macOS Trash with explicit user approval: `/Users/niccoloforte/.Trash/p2-duplicate-notebooks-20260912/`. Trial 1 copies were full executed versions differing by field selection; smoke copies were older shorter versions. Results, checkpoints and images were preserved. Root instructions now explicitly prohibit surprise scope expansion or executable/notebook copies in data; maintained notebooks stay in p2 code.
- Full-data dual HPO **26375307**, `dual-joint-hpo1`, was submitted on 10 September 2026 through B1 and started at 16:52:57 BST on `sbg9`. Slurm confirms `andrena` / `pilot_andrena`, one A100 40 GB GPU, 12 CPUs, 90,000 MB RAM and **240 hours**. The Python budget is 230 hours, targeting 200 completed/pruned evaluations with up to 450 epochs each. Submission checkout: `8061a43`; `resources/MLdualHPO.py` owns fixed balanced validation ranking, four positive task weights, periodic backups and study-level resume.
- Submit directory: `/data/home/exy053/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1`; log: `dual-joint-hpo1.o26375307`. B1 staged to `/gpfs/scratch/exy053/26375307` and launched Python successfully; study backups/results go to `/data/SEMS-TaoLab/Niccolo-Forte/p2/MULTI/Dual/Transformer/HPO/dual-joint-hpo1`. Inspect this job before submitting a duplicate or attempting resume. Last verified scheduler state on 10 September was RUNNING; current state and completed training epochs/trials have not been checked in this notebook task.
- Review the labelled pin/interface examples against the intended production geometry. The current profile uses the A1 Ti/Al proportions and the sample's initial disordered coordinates.
- Validate reconstructed lattice connectivity against one original producer/INP artifact before claiming agreement with archived meshes. Do not process thousands of INPs by default.
- HPO completion, repeated-seed finalist confirmation, locked-test evaluation, full-dimensional optimisation and new FEA verification remain outstanding. Study resume does not continue a partially trained optimizer; interrupted configurations restart from epoch one.
- Local checks cover the expanded architecture, strict reload of the actual Trial 1 checkpoint, HPO archive-to-new-scratch resume, exclusive-lock refusal and saved diagnostic adapters. Repository contract validation previously passed (67 non-failing, zero failures). Remote HPO CLI help, wrapper Bash syntax and Slurm `--test-only` resource validation passed before submission; these do not establish GPU training success.
- SSH access was renewed and the clean HPC checkout fast-forwarded through a verified incremental Git bundle to `8061a43` before submission. Both hosting destinations already contained that implementation. Temporary deployment bundles are outside the repository, not under `data/`.
- Keep the field intermediary and full 201-point curves. User-reported true-field FT full-curve performance is promising; numerical manuscript claims still require the saved evidence.
- SSH access uses the user's active ControlMaster socket. Access availability is temporary and does not itself authorize arbitrary remote changes. Never store passwords in chat or repo files.
- HPC synchronization of these notebook corrections requires renewed SSH access: the socket expired and a non-interactive connection was denied on 14 September. No running job or staged study source was changed. Training/HPO fingerprinted modules remain unchanged by the notebook corrections; only shared result-viewing helpers changed.
