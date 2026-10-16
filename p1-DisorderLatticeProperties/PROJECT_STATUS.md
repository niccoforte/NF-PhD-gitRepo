# p1 Project Status

Read this file for continuation, planning, or handoff work. Durable workflow and safety rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- Current bounded implementation: read-only periodic and four explicitly selected
  20% nodal-disorder Ti FCC UT/FT element-to-strut fracture processing for Damage
  Modeling. The target is binary fracture from whole-element STATUS=0. No new
  FEA, architecture changes or archive-wide extraction.
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

- The researcher selected binary fracture: any failed constituent gives parent
  1; all known active constituents give 0; missing without a failed member is
  unknown. SDEG stays diagnostic; no partial aggregation decision blocks this
  target. The four local disordered cases (UT 397 maxD, UT 747 maxB, FT 747 maxB,
  FT 747 maxK) are authorised. Broader extraction, model integration and jobs are
  outside this scope.
