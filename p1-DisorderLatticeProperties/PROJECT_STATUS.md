# p1 Project Status

Read this file for continuation, planning, or handoff work. Durable workflow and safety rules remain in `AGENTS.md`; Git history records completed changes.

## Current Objective

- Current bounded implementation: read-only periodic Ti FCC UT/FT element-to-strut
  damage processing for Damage Modeling. No new FEA or archive-wide extraction.
- Broader p1 research objective and phase remain `To be confirmed`.

## Authoritative Working Surfaces

- Abaqus generation and export: `SIMscripts/A1_FractureToughness-Ductility.py` and the active `SIMscripts/A2_*.py` exporters.
- Opt-in damage route: `SIMscripts/A2_DamageOUTpostProcess.py`,
  `resources/damage.py` and p2 `code/review_damage.py`; readable products under
  `p2-DisorderML/samples/damage-audit/`.
- ML-ready processing: `code/DataProcessing.ipynb`, `code/InputsOutputs.ipynb`, `resources/data_processing.py`, and `resources/MLdata.py`.
- Validation and mechanics notebooks are identified in `code/AGENTS.md`; which one is current is `To be confirmed`.

## Last Verified State

- Repository structure and static guidance were reviewed on 2026-08-09.
- The original August review did not execute Abaqus/data workflows. The current
  damage pass opened only the two selected periodic ODBs in Abaqus 2024 Python:
  21 recorded frames each, original mesh/connectivity agreement, unchanged source
  SHA-256 hashes, zero missing expected SDEG/STATUS records. Final observed failed
  body parents: UT 83/2319, FT 7/2259. Twelve synthetic damage tests passed; full
  contract validation passed (78 non-failing, zero failed). This does not validate
  the complete archive, unreported beam section points or any new FEA.

## External Requirements

- Abaqus Python/CAE and a valid license for model or ODB behavior.
- The relevant local `Z:/p1/data/Ti/...` data or QMUL HPC archive/scratch paths.
- The exact target lattice, disorder, mechanics mode, and scientific validation authority.

## Decisions And Next Task

- Periodic sample scope is authorised. Partial severity reduction is explicitly
  configurable; researcher selection of the reported-section/parent maximum is
  separate from verified whole-element STATUS failure. No scaling or model
  integration is authorised by this test.
