# p1 Project Status

Read this file for continuation, planning, or handoff work. Durable workflow and safety rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- The full p1 archive was downloaded to `Z:/p1/HPCarchive`. Read-only comparison
  confirmed all 212,198 files / 995,745,698,879 bytes and directory/type/size
  agreement with `/data/SEMS-TaoLab/Niccolo-Forte/p1` on 23 October 2026.
  The requested quick content check matched SHA-256 for 160 complete files and
  1 MiB head/middle/tail chunks from 36 larger files: zero mismatches/errors,
  371,540,157 bytes read per copy. Deterministic seed 20261023 covered all 156
  defined disorder/lattice/UT-FT/file-type strata; it is not a full-file checksum
  proof for the whole archive. Source files were unchanged during their reads.
  The checked live p2 jobs use p2 MLdata/field banks, not the p1 raw archive;
  MLdata contains no symlinks into p1. No active p1 job was listed.
  Z: still reports filesystem Warning/Full Repair Needed and DirtyBitSet=True;
  the physical USB disk reports Healthy/Online. No independent second full
  backup was verified, no filesystem repair was run and no HPC data was deleted.
  Transfer confidence, filesystem integrity and single-drive loss remain separate
  questions. The researcher retains the deletion decision; the sample is not
  automatic cleanup authority. Existing loaders and selective B3 paths stay unchanged.
- Ti validation is consolidated in `Z:/p1/sims/Ti/Validation`, with101 ODBs,
  740 retained original files and full relocation hashes. The former HPC
  validation folder was deleted by the researcher. Forty-eight retained
  density-campaign tri/kagome/hex fracture histories are recovered with8,788
  genuine samples and19,576,675 monitored STATUS records. Source fingerprints
  are unchanged at read-only extraction. Fifteen histories are short; one has
  a genuine NaN post-deletion tail. The audit preserves times/missingness and
  bypasses the exporter's discarded-last-sample/artificial-zero-tail behavior.
- The validation calculation route now brackets first tensile LE11=.1105 in
  all48 cases and keeps whole-element deletion separate. Saved outer-section
  damage supports reusing preceding prefixes; unsaved interior states and the
  source's unpublished VUSFLD remain explicit limits. Selected pre-crossing
  KE/IE is at most.2854%, not a claim about startup/post-fracture dynamics.
- All48 exact-source periodic B21 conversion controls are available and use
  modified-material2D conversion1/S11 once, without double plane-strain mapping.
  All48 exact elastic compliance controls are complete (4 reused,44 new).
  All48 native cases are recalculated under the initial-triangle convention
  alongside distinct dissipation work estimates. Eleven benchmark cases provide
  37 current comparison rows, including10 converged QS loading prefixes and
  5 eligible matched-recovery estimates. Exact-source QS load/unload checks
  remain separate counterfactual histories. The tri10/.16 loading original and
  finer numerical retry failed before the criterion and are not extrapolated.
  A failed tri10/.13 load has a reviewed criterion-containing converged prefix.
  The B21 Explicit plastic axial/bending probe recorded zero artificial energy.
- Hsieh AppendixD explicitly subtracts the initial-slope triangle. That
  calculation gives negative area even for clean pre-criterion QS hexagonal
  states. Measured nonlinear recovery fixes recoverable-work accounting but
  does not reproduce the published work definition; IE-SE dissipation and
  recovery eta-J estimates remain separate from source-convention calculations.
  Selected plots read `audit/final/validation-results.csv`; no source/paper
  values are replaced. Kagome retains a material residual discrepancy.
- Full Figs8a/8b/10 overlays and dimensional comparisons are integrated into
  ValConvPlots. Fig10 provides18 digitised points with graphical bounds; only
  3mm comparisons are direct, while10mm values require declared sqrt(l) scaling.
  Source Fig8 normalisation and printed Fig10 density ambiguity remain visible.
  Original digitisation is preserved. The workbook retains its additive audit
  sheet and Deshpande!A50:L73 mirror with verified backups and no further edits.
- All20 Review lattices.py findings are reconciled. Nine bounded foundation
  repairs preserve canonical tri/kagome/hex/FCC properties and graph conventions.
  Full/disordered stiffness geometry remains unresolved; exact archived inputs
  bypass it here. Square/45square belongs to later orthotropic work. Shared
  exporters/calculation defaults, p2 datasets and paper/PSC values are unchanged.
  PSC and supervisor-agreed sizes/criteria remain deferred. No user code
  clarification blocks current validation. All queued auxiliary jobs are now
  settled;7 of10 matched unloads completed,5 pass the numerical closure screen,
  and no solver remains running/queued. Failed controls remain explicitly failed.
  Triangular native comparisons are+0.66 to+5.38%, kagome+9.16 to+18.66% versus
  dimensional Fig10 (10mm requires declared sqrt(length) inference). Both3mm
  hexagonal initial-triangle values remain invalid; separate recovery estimates
  differ by+9.17%/.16 and+13.68%/.20 from direct references. The calculation
  audit is reproducible; universal source-method agreement is not established.
  Clean initiation-only paper exports now separate original-simulation results
  from diagnostic energy/unloading comparisons; they use nominal relative
  density to match Hsieh's. The underlying original-simulation calculation
  convention is unchanged. FTval/shared calcFT still retain the historical path;
  the corrected values are in external CSVs read by ValConvPlots. The file
  inventory and cleanup-review register cover code, workbook additions/legacy
  cells, raw/derived audit evidence, early reviews and named temporary work.
  Cleanup remains deferred until the final method and integration are accepted.
- The authorised existing Ti FCC periodic/20% nodal-disorder STATUS dataset
  processing and local/HPC archive delivery are complete. The generated labels,
  aligned banks and visual report are ready for the separate ML workflow.
  Other lattices, new FEA and ML architecture/training integration remain outside
  this completed data-processing task.
- Broader p1 research objective and phase remain `To be confirmed`.

## Authoritative Working Surfaces

- Abaqus generation and export: `SIMscripts/A1_FractureToughness-Ductility.py` and the active `SIMscripts/A2_*.py` exporters.
- Opt-in damage route: `SIMscripts/A2_DamageOUTpostProcess.py`,
  `resources/damage.py` and p2 `HPC/studies/Damage/review_damage.py`; readable products under
  `p2-DisorderML/samples/damage-audit/` (periodic) and the separate
  `p2-DisorderML/samples/damage-disordered/` comparison.
- ML-ready processing: `code/DataProcessing.ipynb`, `code/InputsOutputs.ipynb`, `resources/data_processing.py`, and `resources/MLdata.py`.
- Current validation plots are in `code/ValConvPlots.ipynb`; other mechanics
  notebooks and their roles are identified in `code/AGENTS.md`.
- Ti validation inputs/results: `Z:/p1/sims/Ti/Validation/README.md` and
  `case-index.csv`; use `file-manifest.json` for historical-path resolution.
- Validation audit: `code/validation_history_audit.py`, `validation_modal.py`,
  `validation_calculations.py`, `validation_benchmark.py`, `validation_comparison.py`,
  `validation_diagnostics.py`, `validation_failure_audit.py`, `validation_controls.py`,
  `validation_work_partition.py`, `validation_results.py`, `validation_inventory.py`
  and serial `validation_queue.py`; evidence under the external
  collection's `audit/`; readable report in `p2-DisorderML/samples/p1-validation/`.
  The canonical OneDrive workbook's Validation audit sheet is a sensitivity view,
  not replacement paper values. Resolve work/initiation limits before promotion.

## Last Verified State

- Repository structure and static guidance were reviewed on 2026-08-09.
- The original August review did not execute Abaqus/data workflows. The periodic
  damage pass opened the two selected ODBs in Abaqus 2024 Python:
  21 recorded frames each, original mesh/connectivity agreement, unchanged source
  SHA-256 hashes, zero missing expected SDEG/STATUS records. Final observed failed
  body parents: UT 83/2319, FT 7/2259. Twelve synthetic damage tests passed; full
  contract validation passed (78 non-failing, zero failed). This does not validate
  the complete archive, unreported beam section points or any new FEA.
- The four authorised disordered INPs validate independently: graph-degree
  original joints, producer-derived +/-sqrt(2) mm coordinate bounds, straight
  original FE chains and exact canonical edge sets. UT 397/747 have 9,463/9,461
  body elements and 2,319 parents each, excluding 830 grips each. Both FT files
  have 4,429 elements/2,259 parents. The maxB UT/FT pair shares exactly matching
  coordinates for all 788 common original joints. FT maxB/maxK INPs are
  byte-identical; exported STATUS/SDEG, displacements, reaction/loading and raw
  histories also match exactly, so they are duplicate exported responses, not
  independent specimens. All four exports contain 21 actual frames, zero missing
  expected STATUS/SDEG observations and no reactivation; source hashes remain
  unchanged. Final failed parents/elements: UT 397 95/97; UT 747 92/93; each FT
  9/9. Sixteen synthetic mapping/reduction tests pass. Independent live-ODB
  verification matches all 583,422 STATUS values, all 192,276 parent/frame
  failure and known masks, and 277,820 section SDEG values at five frames/file.
  The separate visual report retains all four file results and their provenance.

## External Requirements

- Abaqus Python/CAE and a valid license for model or ODB behavior.
- The relevant local `Z:/p1/data/Ti/...` data or QMUL HPC archive/scratch paths.
- The exact target lattice, disorder, mechanics mode, and scientific validation authority.

## Decisions And Next Task

- The researcher selected binary fracture: any constituent whole-element
  STATUS=0 gives parent 1; all known active constituents give 0; otherwise unknown.
  SDEG remains diagnostic.
- Production array 32085486 completed all 91 batches/18,002 records with zero
  errors. All 18,002 FIELDd NPZ/JSON pairs are checksum-verified locally:
  18,000 disordered plus the separate periodic pair, reconciled by 91 receipts.
  Corrected bank job 32095821 completed 0:0 in 1 h 13 min using bank-code-4c435cf.
  Original bank 32085487 was cancelled before execution and remains superseded.
- Four allD NPZ banks are verified in HPC p2/MLdata and local FCC MLdata,
  with exact existing ordered UT 8882, FT 8928 and MULTI 8811 populations and ID 0.
  Full bank byte hashes match; geometry, source hashes and saved field layouts
  reconcile. The frozen builder validated binary arrays, masks, onset and exact
  serialization round trips; the separate delivery proof checks byte identity,
  metadata/layout/coverage and D/mask headers without loading full label tensors.
- Recorded corpus checks cover 2,622,439,219 raw STATUS positions and 865,222,188
  parent/frame labels, with zero missing observations, unknown observed parents
  or reactivation. Independent native ODB comparison remains bounded to eight
  preflight products: 1,171,296 raw positions and 384,552 parent/frame labels.
- Eleven short UT histories preserve their actual prefixes. Each UT bank has
  50 absent sample frames/115,950 masked parent-frame entries; FT has none.
  Frame counts/presence and NaN unavailable times distinguish these from unknown
  observed STATUS; no interpolation or failure carry is applied. The real
  periodic0/UT601 prefix check, 13 bank tests and 13 report tests passed. Earlier
  contract validation recorded 85 non-failing/zero failed checks.
- The complete visual report is p2 samples/damage-dataset/README.md. Its
  execution/ folder retains per-file results/receipts, four bank proofs, source
  equivalence/padding evidence, HPC layout and directory-level source inventory.
  Frozen code, manifests and result records are checksum-verified in the remote
  FCC archive's damage-261016/execution.tar.gz; four bank JSON audit reports
  are in its bank-reports/ folder. MLdata retains its NPZ/CSV training/testing files.
- Home p1/Ti/disNodes/0.2/FCC/damage-261016 now retains only 94 successful logs.
  Copies also accompany the completed job's archived results.
  All 120 non-log launch files and four JSON reports from each local/HPC MLdata
  folder were removed after verifying the archived copies. NPZ hashes, arrays
  and all 19 remaining MLdata files are preserved. The processing archive is
  /data/SEMS-TaoLab/Niccolo-Forte/p1/Ti/data/disNodes/0.2/FCC/damage-261016.
- All 91 production scratch roots and the task-owned local staging/helpers are
  removed. Original ODB/INP/FIELDu/curves are preserved. The scheduled follow-up
  was deleted at the researcher's request and was not recreated.
