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
  matching INP/FIELDu filenames; native STATUS availability still needs preflight.
  Extend the existing exporter/B2 route, keep frame-index alignment, periodic ID0
  and exact existing MLdata populations. Local/HPC paths and contracts are in
  SIMscripts/DAMAGE.md. The lean route matches every raw STATUS/parent mask in
  both local periodic Abaqus exports (UT83/FT7 final failed parents;6.85s/3.66s
  export time). Archived periodic INPs match those sources byte-for-byte; ODB
  hashes differ and require direct response comparison in the cluster preflight.
  No jobs have yet been submitted for this production route.
