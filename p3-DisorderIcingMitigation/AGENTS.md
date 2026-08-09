# p3 Disorder Icing Mitigation Agent Context

This paper folder is provisional. It currently contains one tracked Abaqus CAE script, `SIMscripts/A1_IcingModels.py`. Repository evidence does not yet establish a validated icing-interface, adhesion, thermal, cohesive, or de-icing workflow.

Read `PROJECT_STATUS.md` for changing objectives, evidence, decisions, and next work. Do not present the script or its results as a validated icing-mitigation model without explicit user confirmation and supporting evidence.

## Current Script

- `A1_IcingModels.py` is a large monolithic Abaqus script with local `node`, `connectivity`, `Geometry`, material, model-building, and job-running logic.
- It imports Abaqus CAE modules and cannot be behaviorally validated with standard Python.
- Its default working directory is `C:\temp`; inactive branches contain older `Z:\p1\...` and user-specific paths. Treat every path as provisional and confirm it before execution.
- It reads a positional command sequence from `sys.argv[8:]` and can write, submit, and wait for Abaqus jobs depending on its run settings.
- Its current job names and branches use p1-style `Ductile` and `Fracture` terminology. That naming does not establish p3 scientific meaning.

## How To Work Here

- Ask for clarification before changing the scientific model, geometry, disorder definition, material law, boundary conditions, contact/interface behavior, output requests, or validation criteria.
- Inspect the entire affected branch of the monolithic script before editing; duplicated ductile/fracture sections may require paired changes.
- Do not run the script until the Abaqus environment, target working directory, expected artifacts, and submit/write-only behavior are confirmed.
- Use standard Python only for syntax-oriented inspection, and state clearly that it does not validate Abaqus APIs or model behavior.
- Keep p3 changes isolated from p1 and `resources/` unless the user explicitly approves a shared-code migration.
- Do not import p1 scientific assumptions merely because the script shares p1 code patterns or terminology.
- Do not refactor the monolith solely for tidiness. First confirm which parts are active and scientifically authoritative.
- Keep this file, `PROJECT_STATUS.md`, and the p3 section of the root `README.md` current at their respective durable, current-state, and human-facing levels whenever p3 gains confirmed information.
