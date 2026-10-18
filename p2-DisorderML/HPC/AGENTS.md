# HPC Agent Context

This directory owns p2 QMUL Slurm training, HPO, resume, archive and transfer. Read `../PROJECT_STATUS.md` for the objective; use `operate-p2-hpc` for work here and its reference for entry-point/path mappings.

## Ownership

- `B0_ML-env-setup.sh` manages the GPU environment.
- `B1_ML-new.sh` stages and submits work through scratch to the archive.
- `B2_ML-resumeHPO.sh` resumes archived cross-model studies.
- B3 Windows defaults to external `Z:/p2`; Mac requires explicit external `LOCAL_ROOT`. Both reject Git-contained destinations and preserve relative paths; neither deletes sources. Mac supports Bash3.2, rsync/ControlMaster and `--dry-run`.
- `CurveOutputs/`, `FieldOutputs/`, and `FieldToCurve/` own their respective single-run and cross-model HPO entry points.
- `DualOutputs/A0-HPC-Dual-test.py` owns dual runs; Trial1 is a thin preset staged with it by B1. Its HPO entry calls `MLdualHPO` for single-worker full-data search/explicit resume through B1. README owns ranges/ranking/recovery; `test_dual_contract.py` protects architecture, HPO and review contracts.
- `FieldToCurve/A0-HPC_FieldToProperty-test.py` is the separate true-field oracle diagnostic for archived UT Strength/Ductility/WoF and FT K_JIC. It reuses `DUAL_DATA`, `DualTaskTransformer` and `train_model`, with no disorder/context input. Store under `MULTI/FieldToProperty/Transformer/<label>` and review its property-specific JSON/plots, not curve loaders. Full runs require exact frozen split IDs; properties join by unique ID, scalers fit on train, validation selects and test targets remain unevaluated. Debug subsets require `--nsims`; run a B1 GPU preflight before a dependent full test. No property definitions, data filters or FEA exports are changed.

## Durable Guardrails

- Preflights are disposable execution tests: B1 recognizes `preflight` in the
  script/job/run label, skips archive copies and retains only the successful home
  log; successful scratch is deleted. Never use these test outputs as production
  dependencies. Full-data field banks are production artifacts, not preflights.
- Failure evidence is temporary. After diagnosis/correction, cleanup is HIGH PRIORITY:
  confirm terminal jobs and exact paths, preserve resumable production results,
  then remove failed scratch/logs/useless archives. Never delete active jobs or
  unidentified data. Completed home keeps successful logs only; archive code,
  manifests, needed audits and log copies beside results in the existing run layout.

- Active entry points are production-oriented: full data and realistic training budgets by default. Small datasets, short epochs/trials, and CPU execution must be explicit debug overrides.
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

- `DualOutputs/A0-HPC-Damage-contract.py` replays only the six handoff fixtures: STATUS masks, INP identity, endpoint/frame layout and NumPy/Torch reload, not training. Use disposable B1 retention; see README. Layout matches do not prove specimen joins.

- `DualOutputs/B4_Independent-followups.sh` keeps Case1(i) losses on the selected input policy. Corrected policies need42 full jobs: field banks, newly nested-cross-fitted corrections, adapted curves and frozen-pipeline curve-loss fits. Nested generators inherit fold-local losses; correctors retain MSE. Preserve fixed selection, source hashes and disposable gates; see README.
- `DualOutputs/B4_Dual-independence.sh` previews eight sharing cases: one8h gate, six OOF banks,24 full240h fits across seeds42/43/44 through B1. Explicit `--submit` requires user authority. The adjacent README owns cases, isolation, bank contracts and safeguards. Do not mix deferred losses/features or gradient diagnostics into this suite.
- `../handoffs/HANDOFFS.md` routes separate tasks; read only the relevant brief. Passing tests is not researcher sign-off. Verify saved loss classes/nonzero coefficients. Fresh `--curve-loss-ablation` baseline/source fits use the five-case `--curve-loss-suite` gate; combined variants have soft targets and differ only in location weight. Preserve fixed selection, MSE controls and historical runs. PROJECT_STATUS owns submission evidence.
- Do not retain verified useless failed-run logs/empty outputs in the results archive. Audit and obtain exact-target cleanup authority first; a failed HPO can contain valuable resumable checkpoints and must not be erased merely for its exit status. The 4 October failed-bootstrap cleanup is complete, not a handoff task; see the adjacent DualOutputs README.

- The dual test runner's `--experiment` prepares one-change baseline/crack-face/local-graph/encoder-sharing/interface/residual/localisation comparisons. `--base-model-json` uses a shared non-graph dual configuration anchor, not its weights; fixed split seed 42 and unchanged balanced selection make variants comparable. Parameter counts and split hashes are mandatory. Run local checks first, then an authorised B1 GPU preflight before full jobs; preparation is not submission. See DualOutputs README for exact semantics and deferred work.
- `DualOutputs/B4_Dual-experiments.sh` previews unless `--submit`; use fresh labels and immutable clean sources through B1. Defaults cover13 modes; recovery `--variants` must not repeat successes and curve fits require winner_probe. See the workflow reference for node-targeted gates. Record jobs.tsv dependencies; curve-only selection uses two curves, joint selection four. No test development or source-HPO changes.
- The separate opt-in `--variants baseline,sudden` pair uses a two-case `--sudden-suite` GPU gate. Gain1 changes only specimen-specific displacement weighting; spatial/temporal penalties stay zero, selection stays fixed. Default thirteen-mode suite is unchanged. CPU contracts are in `DualOutputs/test_sudden_weighting.py`; passing them is not GPU or accuracy evidence. Preserve source snapshots, 240-hour full requests and B1 collection.

- Preserve existing approved trial, test and HPO entry points unless the user explicitly requests their removal. Loss ablations use `FieldOutputs/A0-HPC_Field-lossTrial.py` (exact archived UT/FT architecture/training presets) and `DualOutputs/A0-HPC-Dual-lossTrial.py` (Trial 1 capacity). B1 stages their existing runner dependencies. Choose `--field-loss-variant baseline|spatial|temporal|both`; `weighted` additionally requires an explicit positive gain and follows evidence-based variant selection. Defaults remain full data/450 epochs, validation-only selection, and the existing resource policy. No damage/strain export or architecture/HPO extension is implied.
- Independent ablations select checkpoints by unchanged normalised validation MSE; dual ablations use the unchanged balanced physical validation score. Always use unique run labels. Save field-loss components, motion diagnostics and paired true/predicted-field curve checks alongside standard results. Different single/dual populations must not be treated as a matched comparison.
- Gate full loss trials on a successful real-data/GPU B1 preflight for their family, including diagnostics and archiving. Slurm `afterok` with `--kill-on-invalid-dep=yes` may queue the full trials safely; record preflight and dependent job IDs separately. A queued/started preflight is not proof of success.

- Run root changed-surface validation and focused argument/path tests; distinguish syntax checks from live GPU/Slurm execution.
