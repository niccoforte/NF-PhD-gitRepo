# HPC Agent Context

This directory owns p2 QMUL Slurm training, HPO, resume, archive and transfer. Read `../PROJECT_STATUS.md` for the objective; use `operate-p2-hpc` for work here and its reference for entry-point/path mappings.

## Ownership

- `B0_ML-env-setup.sh` manages the GPU environment.
- `B1_ML-new.sh` stages and submits work through scratch to the archive.
- `B2_ML-resumeHPO.sh` resumes archived cross-model studies.
- B3 Windows defaults to external `Z:/p2`; Mac requires explicit external `LOCAL_ROOT`. Both reject Git-contained destinations and preserve relative paths; neither deletes sources. Mac supports Bash3.2, rsync/ControlMaster and `--dry-run`.
- Existing `CurveOutputs/`, `FieldOutputs/`, `FieldToCurve/` and `DualOutputs/` retain their run/HPO interfaces. New study entry points, launchers and reviews belong in `studies/<Study>/`; see `studies/README.md`. Imported training helpers stay in `resources/`. Do not copy trainers or rename immutable submissions.
- `DualOutputs/A0-HPC-Dual-test.py` owns dual runs; Trial1 is a thin preset staged with it by B1. Its HPO entry calls `MLdualHPO` for single-worker full-data search/explicit resume through B1. README owns ranges/ranking/recovery; `test_dual_contract.py` protects architecture, HPO and review contracts.
- `FieldToProperty/A0-HPC_FieldToProperty.py` is the true-field property oracle, not a curve run. Its guide owns targets and controls; preserve ID joins, train-only scaling and locked test. Do not repeat it because its source moved.
- `studies/Damage/A0-Damage.py` / `B4_Damage.sh` own the isolated eight-fit STATUS
  screen in `MULTI/Damage/Transformer`; the guide specifies arms, masks, balanced
  loss and selection. Its disposable GPU gate verifies full data alignment first.
  STATUS uses an explicit checkpoint reader, not DUAL reload.

## Durable Guardrails

- Preflights are disposable execution tests: B1 recognizes `preflight` in the
  script/job/run label, skips archive copies and retains only the successful home
  log; successful scratch is deleted. Never use these test outputs as production
  dependencies. Full-data field banks are production artifacts, not preflights.
  Temporary local replay fixtures/exports and migration staging are also
  disposable after completion; do not require permanent fixture bundles.
- Failure evidence is temporary. After diagnosis/correction, cleanup is HIGH PRIORITY:
  confirm terminal jobs and exact paths, preserve resumable production results,
  then remove failed scratch/logs/useless archives. Never delete active jobs or
  unidentified data. Completed home keeps successful logs only; archive code,
  manifests, needed audits and log copies beside results in the existing run layout.

- Active entry points are production-oriented: full data and realistic training budgets by default. Small datasets, short epochs/trials, and CPU execution must be explicit debug overrides.
- Locality's bounded debug subsets disable extrema-forced splitting to keep
  holdouts nonempty; full-data identity checks and production retain the saved
  extrema-aware split. Do not change production splitting to repair a smoke run.
- Do not change Slurm partition, account, license, CPU, memory, or time policy without the user's target cluster policy.
- Preserve explicit scratch cleanup guards and verify every archive, resume, transfer, or cleanup target before use.
- B1 must retain scratch and return failure if copying run outputs or job logs to the archive fails, even when Python training succeeded.
- B1 initialises `/etc/profile.d/modules.sh` if non-interactive submission has not supplied `module`; do not rely on an interactive login environment.
- For node-specific module failures, inspect scheduler node state/reason before changing code or targeting a preflight node. Respect administrative drains; a successful gate on another node does not establish that the affected node is repaired.
- Slurm job/run names, `Curve`/`Field`/`FieldToCurve`/`Dual` layouts, metadata, diagnostics, and transfer paths form a shared saved-run contract. Use `review-p1-p2-data-contract` when they change. Dual runs use `MULTI/Dual/Transformer/<run-label>` and the existing staging/archive/transfer scripts, but their checkpoint loader is `DUAL_MODEL`, not legacy `MODEL`.
- Keep CLI arguments stable or provide a clear migration. Do not introduce local Windows paths except as documented transfer destinations.
- Do not submit Slurm jobs, resume studies, transfer archives, or alter external environments unless the user requested that operation.
- Reuse the user's SSH ControlMaster; do not close it after commands or transfers. `-O check` checks the local master only; verify remote liveness with a harmless command. Never send `-O exit`, kill the master or remove its socket without explicit permission.
- Windows: reuse PuTTY `HPC` via `plink -batch -share -load HPC -l exy053`; verify `-shareexists` and remote liveness. Transfer via `pscp -share` to `Z:/p2`. Leave the session open; reports use canonical Z: samples per root guidance.
- Keep active scripts lean; remove abandoned debug branches and obsolete compatibility paths.
- Dual HPO uses `MULTI/Dual/Transformer/HPO/<study>` with a scratch SQLite DB, consistent archive backups, an exclusive archive lock and explicit study-level resume. Do not pass it through legacy B2. Never clear a stale lock without verifying the recorded job stopped. Resume retries an interrupted configuration from epoch one; it is not optimizer-state continuation.
- Rank dual HPO by fixed equal-task, physical, train-mean-baseline-relative validation MSE; tuned loss weights must not change ranking. Keep full paired data, splits, normalization, masks and all four positive supervision weights. Default 230-hour Python budget leaves margin under the unchanged ten-day Slurm limit; checkpoints and DB backups protect unexpected termination.

## Validation

- `studies/Locality/` screens degree/GNN/CNN/attention one change at a time, field-only. Preserve exact control IDs/scalers, MSE selection and no curve exports; its guide owns the disposable16-case GPU gate and production budget. Read WINNER.md before choosing the control.

- Named study wrappers preview unless `--submit`. Keep strict bank/seed/scaler/source checks and afterok gates; `studies/README.md` owns entry paths and each study guide owns cases/resources. Mixed loss continuation reuses12 compatible banks; do not resubmit them or mix new features into frozen comparisons.
- `../handoffs/HANDOFFS.md` routes separate tasks; read only the relevant brief. Passing tests is not researcher sign-off. Verify saved loss classes/nonzero coefficients. Fresh `--curve-loss-ablation` baseline/source fits use the five-case `--curve-loss-suite` gate; combined variants have soft targets and differ only in location weight. Preserve fixed selection, MSE controls and historical runs. PROJECT_STATUS owns submission evidence.
- Do not retain verified useless failed-run logs/empty outputs in the results archive. Audit and obtain exact-target cleanup authority first; a failed HPO can contain valuable resumable checkpoints and must not be erased merely for its exit status. The 4 October failed-bootstrap cleanup is complete, not a handoff task; see the adjacent DualOutputs README.

- Historical dual experiments retain split42, fixed selection, parameter/split
  records and B1 gates. `--base-model-json` anchors configuration, not weights;
  consult DualOutputs README before using its experiment/recovery modes.
- `B4_Dual-experiments.sh` previews unless `--submit`; recovery must not repeat
  successes. Keep immutable sources and jobs.tsv. Curve-only selection uses two
  curves, joint selection four; do not change source HPOs or develop on test.
- Optional `baseline,sudden` uses its own two-case gate: gain1 changes only
  target weighting, derivative penalties stay zero. Preserve the default suite,
  CPU contracts, source snapshots and240h full requests; see DualOutputs README.

- Preserve existing approved trial, test and HPO entry points unless the user explicitly requests their removal. Loss ablations use `FieldOutputs/A0-HPC_Field-lossTrial.py` (exact archived UT/FT architecture/training presets) and `DualOutputs/A0-HPC-Dual-lossTrial.py` (Trial 1 capacity). B1 stages their existing runner dependencies. Choose `--field-loss-variant baseline|spatial|temporal|both`; `weighted` additionally requires an explicit positive gain and follows evidence-based variant selection. Defaults remain full data/450 epochs, validation-only selection, and the existing resource policy. No damage/strain export or architecture/HPO extension is implied.
- Independent ablations select checkpoints by unchanged normalised validation MSE; dual ablations use the unchanged balanced physical validation score. Always use unique run labels. Save field-loss components, motion diagnostics and paired true/predicted-field curve checks alongside standard results. Different single/dual populations must not be treated as a matched comparison.
- Gate full loss trials on a successful real-data/GPU B1 preflight for their family, including diagnostics and archiving. Slurm `afterok` with `--kill-on-invalid-dep=yes` may queue the full trials safely; record preflight and dependent job IDs separately. A queued/started preflight is not proof of success.

- Run root changed-surface validation and focused argument/path tests; distinguish syntax checks from live GPU/Slurm execution.
