# SimCheck cleanup checklist

This is the separate cleanup checklist for this research task. Cleanup is deferred
until the user reviews it at the end. Do not delete evidence or change approved
final figures while the calculation and manuscript work continues.

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
- [ ] Review all audit outputs, including filtering galleries, stiffness history,
  axis examples, modal controls, FCC feasibility controls and published benchmarks.
  Keep the minimum reproducible evidence; remove superseded renders and bulk.
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
- [ ] Resolve pending reference intake and report links before deleting supporting
  extracts or captures. Do not modify the personal memory store as part of cleanup.
- [ ] Perform the final cross-drive inventory and verify that only accepted code,
  research data, figures and necessary provenance remain. Report retained paths,
  removed paths and any residual dependencies to the user.

No automatic deletion is enabled. Final disposition is reviewed with the user.
