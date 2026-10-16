# p1 Project Status

Read this file for continuation, planning, or handoff work. Durable workflow and safety rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- Current authorised implementation: process existing Ti FCC periodic and 20%
  nodal-disorder UT/FT archives for binary strut fracture from whole-element
  STATUS=0. Add lowercase FIELDd products in the original job transfer folders,
  consolidate local transfers and add aligned banks to both MLdata archives.
  Preflight precedes resumable Slurm batches. Other lattices, new FEA and ML
  architecture/training integration remain outside scope.
- Broader p1 research objective and phase remain `To be confirmed`.

## Authoritative Working Surfaces

- Abaqus generation and export: `SIMscripts/A1_FractureToughness-Ductility.py` and the active `SIMscripts/A2_*.py` exporters.
- Opt-in damage route: `SIMscripts/A2_DamageOUTpostProcess.py`,
  `resources/damage.py` and p2 `code/review_damage.py`; readable products under
  `p2-DisorderML/samples/damage-audit/` (periodic) and the separate
  `p2-DisorderML/samples/damage-disordered/` comparison.
- ML-ready processing: `code/DataProcessing.ipynb`, `code/InputsOutputs.ipynb`, `resources/data_processing.py`, and `resources/MLdata.py`.
- Validation and mechanics notebooks are identified in `code/AGENTS.md`; which one is current is `To be confirmed`.

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

- The researcher selected binary fracture: any failed constituent gives parent1;
  all known active constituents give0; otherwise unknown. SDEG stays diagnostic.
  Production processing is authorised for the existing FCC archive only:90 job
  groups,9,000 UT plus9,000 FT ODBs, and the separate periodic pair. Each ODB has
  matching INP/FIELDu filenames. Eight preflight cases passed native STATUS
  verification; whole-corpus validation remains in progress. The existing
  exporter/B2 route now preserves frame-index alignment, periodic ID0 and exact
  existing MLdata populations. Local/HPC paths and contracts are in
  SIMscripts/DAMAGE.md. The lean route matches every raw STATUS/parent mask in
  both local periodic Abaqus exports (UT83/FT7 final failed parents;6.85s/3.66s
  export time). Archived periodic INPs match those sources byte-for-byte; ODB
  hashes differ and require direct response comparison in the cluster preflight.
  Corrected cluster export preflight32084741 and independent native verification
  32084803 COMPLETED0:0: eight products,1,171,296 raw STATUS positions and384,552
  parent/frame labels agree exactly, zero missing STATUS. Earlier instance-case
  and verifier-launch import-path failures were corrected; their failed scratch
  and unused launch artifacts were removed after concise diagnosis was retained.
  Production array32085486 tasks0–90 is RUNNING, throttled to four concurrent
  batches; corrected CPU MLdata job32095821 waits afterok on that array, using
  immutable bank-code-4c435cf. Original bank job32085487 was cancelled before running.
  Required UT601 has20 raw frames; existing allFIELD preserves that exact prefix
  and masks its21st frame. The corrected builder follows this existing padding
  rule with explicit frame presence/counts; a real periodic0/601 NumPy check passes.
  Thirteen bank and13 report tests pass; contract validation is85 non-failing,
  zero failed checks. Report execution records retain the replacement job evidence.
  The extraction array is unaffected. Frozen e92e899 source and
  dataset-v2 manifest live in home p1/Ti/disNodes/0.2/FCC/damage-261016. Eight
  checked preflight products are hash-verified in the canonical local transfer folders;
  periodic raw STATUS/masks/times exactly match the earlier audit despite
  different ODB hashes.49 focused tests pass. The complete corpus and both
  MLdata archive additions are still in progress; a30-minute chat follow-up
  monitors completion and performs the remaining verified local transfer/report.
  Timestamped batch counts, per-file results and incremental local transfer
  receipts are retained under ignored p2 samples/damage-dataset/execution/.
