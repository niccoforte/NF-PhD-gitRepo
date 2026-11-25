# SimCheck cleanup checklist

This is the separate cleanup checklist for this research task. Keep it current
while the work progresses. Cleanup begins only after the user reviews the
completed research checklist and explicitly approves the cleanup scope. Do not
delete evidence or change approved final figures while the calculation and
manuscript work continues.

The [research checklist](CHECKLIST.md) tracks scientific work. The
[machine-readable register](Z:/p1/sims/Ti/Validation/audit/cleanup-review.json) and
[file inventory](Z:/p1/sims/Ti/Validation/audit/file-inventory.csv) identify paths,
sizes, source dependencies and proposed dispositions. They are not deletion
instructions. Refresh them before the final review.

- [ ] Confirm one accepted calculation implementation and its notebook callers;
  replace superseded active code, duplicate methods and obsolete notebook outputs.
- [ ] Reconcile Excel values, formulas, audit additions and chart links with the
  accepted results. Retain the original literature digitisation. Remove redundant
  audit sheets and backups only after verified recovery and manuscript integration.
- [ ] Retain the user-approved validation figures and their frozen input recipe
  in `Z:/p1/sims/Ti/Validation/audit/paper/approved-figure-recipe.json`. Final figures
  may change only after explicit user approval; diagnostic recalculation is separate.
- [x] Retire4 superseded Figure10 renders after verified author selection16November;
  the 16 November release recorded the exact paths and hashes. Keep original scientific
  sources, calculation provenance and all other current figures. The obsolete
  density-chart renderer is removed so default refreshes cannot recreate it.
- [ ] Review `audit/figure10-fixed-cutoffs-20261123` after the manuscript handoff.
  The canonical PNG, PDF and SVG replaced the previous renders in place and
  the temporary render copies were removed. The folder holds the staged review,
  rollback metadata and data, and the Matplotlib cache. Keep only evidence still
  needed to reproduce or check the `audit/paper` release.
- [ ] Review `Z:/codex-p1-figure10-20261116` alongside the release references.
  The machine-readable register identifies it as the earlier staging root.
  Check the retained release and rollback records for dependencies on its
  scripts and receipts before proposing any disposition.
- [ ] Review all audit outputs, including filtering galleries, stiffness history,
  axis examples, modal controls, FCC feasibility controls and published benchmarks.
  Keep the minimum reproducible evidence; remove superseded renders and bulk.
- [ ] Review the current snub-square source downloads,3 tensor/3 CT runs and
  validation overlays under `audit/omidi`; retain author-source identities and
  accepted evidence, remove unsubmitted separate plans if superseded by the
  combined serial manifest. Never remove active queue files.
- [ ] Review `accepted-filter-study`, its64-curve galleries, frozen9-holdout
  controls and dense FCC replay; retain one accepted filter/stiffness API and
  minimal qualification evidence. Review older sampling-unresolved galleries
  and report-history copies for removal only after links are reconciled.
  Include flagged-hex-curves, its nine inspection CSVs, the filter-envelope
  diagnostic, and output-filter-replays deliveries/manifest. Preserve source
  identity and qualification evidence; never remove a running queue.
  Include the separately identified double-precision control and original
  single-precision warning/termination/dead-worker receipts. Reconcile future
  collector dependencies before retiring either attempt; no silent overwrite.
  Include `fixed-cutoff-pilot`, whose case-level results depend on its `events`
  index and source-bound Abaqus extraction receipts.
- [ ] Review FCC initial-event and continuation deliveries, dense-event exports,
  same-interval verification and old subtraction-sensitivity outputs. Preserve
  partial support/failure provenance and the chosen final method, not every plot.
  Include calibration-review native energy histories, train/holdout metadata and
  provisional coefficient plots. A conditional fitted coefficient is not accepted
  production output; retain only the final chosen calibration evidence.
  Include original-flow-strip and Z:/codex-p1-fcc-strip-20261118. Keep the
  deterministic mesh generator, source manifests and verified observations until
  the domain and C(T) transfer checks are resolved.
- [ ] Review the isotropic and FCC stationary-notch extension deliveries in
  `audit/fcc-calibration/eta-verification`. Retain the exact 1 N controls, case
  index, solver and extraction receipts, and the ordered FCC cutoff and existing
  history comparisons. Keep the hexagonal energy receipt, the serial handoff
  and the H190 failed-prefix receipt with their sources. Include the ordered
  80 mm FCC rerun, the supervision report and its earlier figure, and
  the larger PSC preparation, failed first launch, verified onset records,
  and settled [serial queue](Z:/p1/sims/Ti/Validation/audit/psc-eta-q/state.json).
  The failed triangular reference delivery supports its qualified onset prefix.
  Include the 40-cell FCC stationary control [study directory](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n40)
  and [owned queue](Z:/p1/sims/Ti/Validation/audit/fcc40-eta-q), the 30-cell
  [study directory](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n30)
  and [owned queue](Z:/p1/sims/Ti/Validation/audit/fcc30-eta-q). Both original
  linear controls and dense references remain linked in place. Retain the
  180 mm late cutback and timestamp-provenance receipts with the qualified fit.
  Include the 26- and 36-cell stationary case directories, the single
  [archive queue](Z:/p1/sims/Ti/Validation/audit/fcc-eta-archive-q), and the
  PSC-matched [50-cell source-build directory](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n50/source-build).
  Include the 158 mm [refinement study](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n36/middle-notch-refined)
  and its [serial queue](Z:/p1/sims/Ti/Validation/audit/fcc36-refined-q).
  Include the 50-cell postprocessing receipts and the qualified geometry-only
  [46-cell source](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n46/source-build)
  and [56-cell source](Z:/p1/sims/Ti/Validation/audit/fcc-calibration/eta-verification/fcc-size/load-separation-n56/source-build).
  Refresh the machine-readable register before any final disposition.
- [ ] Retain the accepted hex stiffness catalogue and nine source-exact controls;
  reconcile it with saved Props in the population pass. Retire superseded filter
  galleries/conditional CLI routes after the selected capped rule is integrated.
- [ ] Review abandoned Omidi polymer/pin-controlled CT attempts, ring-rotation
  diagnostic, original angular-branch inputs (unsubmitted), released-notch-phase
  controls and unsubmitted refinement/domain plans. Keep only essential rejection
  evidence plus the eventual accepted benchmark and its numerical checks.
  Neither released-source nor Figure2c phase is currently qualified. Reconcile
  the shared Abaqus inspection index with retained INP/ODB paths before deletion.
- [ ] Retain original research INP/ODB/transfer data and useful new simulations.
  Review auxiliary ODBs, logs, prepared-but-unused decks and retry lineages separately.
- [ ] Review `Z:/p1/sims/Ti/Validation/audit/fcc-calibration/tensor`, which was
  prepared but never submitted and is not used. Preserve the actual source-CT
  periodic tensor instead. No removal is attempted during the current work.
- [ ] Verify every serial queue is settled and every retained solver delivery is
  checked before removing its owned scratch. Historical queued/failed receipts
  must be reconciled with their continuation records, not silently rewritten.
- [ ] Review the fingerprinted before-state under `audit/final/before-periodic-filter`
  after final integration. It is currently also the approved figure data source;
  do not delete it while the frozen figure recipe depends on it.
- [ ] Review named temporary work under
  `C:/Users/exy053/AppData/Local/Temp/codex-p1-*`, including scripts, extraction
  intermediates, PDF renders and temporary validation worktrees. Verify dependencies
  and remove registered Git worktrees through Git before deleting their directories.
- [ ] Check `C:/temp` and all task-owned scratch locations. Do not sweep unrelated
  temporary files, other projects or original source data.
- [ ] Resolve ownership of the three adjacent recovery files reported at the
  drive root: `Z:/codex-handbook-recovery.tmp`,
  `Z:/codex-memory-summary-recovery.tmp` and
  `Z:/codex-raw-memories-recovery.tmp`. They appear to contain older Codex
  memory material. Their creator, relationship to SimCheck, and recovery
  value are unverified. Keep them pending a separate owner decision.
- [ ] Resolve pending reference intake and report links before deleting supporting
  extracts or captures. Do not modify the personal memory store as part of cleanup.
- [ ] Perform the final cross-drive inventory and verify that only accepted code,
  research data, figures and necessary provenance remain. Report retained paths,
  removed paths and any residual dependencies to the user.

No automatic deletion is enabled.
