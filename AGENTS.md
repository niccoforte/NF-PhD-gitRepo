# Repository Agent Context

Work is grouped by PhD paper with shared Python resources. Read this file and the
nearest nested `AGENTS.md` before editing a paper folder.

For planning/continuation/handoffs, read the paper's `PROJECT_STATUS.md` for changing
state. Durable rules belong in `AGENTS.md`; reusable procedures in `.agents/skills/`.

## Repository Map

- `p1-DisorderLatticeProperties/` is the Abaqus FEA, post-processing, and data-generation workflow for quasi-disordered 2D lattices under uniaxial/ductile and compact-tension/fracture loading.
- `p2-DisorderML/` is the machine-learning and surrogate-modeling workflow built on p1 outputs. It trains and diagnoses curve and field surrogates and contains cluster training/HPO scripts.
- `p3-DisorderIcingMitigation/` contains early, provisional Abaqus work. Read its `AGENTS.md`; its icing-specific scientific scope is still `To be confirmed`, so do not borrow p1 assumptions or present the current script as a validated icing model.
- `resources/` is the shared package used by p1 and p2. Changes there can affect Abaqus scripts, notebooks, ML training, post-processing, and saved artifact loading.

## Environment And Imports

- `pyproject.toml` exposes `resources` as the editable package `phd-shared-resources`.
- Windows: activate `.venv311`, then run `setup-Windows.ps1 -OnlyPython`; omit the selector to also set up Abaqus. macOS: `setup-macOS.sh` maintains `nf-phd`. Both scripts support `-remove`.
- Dependency changes must consider all three environments: Windows standard Python (`requirements.txt` plus Windows-specific compatibility pins), Abaqus Python (`requirements-abaqus.txt`), and the package list/platform pins embedded in `setup-macOS.sh`.
- Setup and removal mutate external Python, Abaqus, or Conda environments. Do not run either script merely for validation; use parse/syntax checks unless the user explicitly authorizes the install or removal.
- Use imports like `from resources.MLdata import DATA` or `from resources.lattices import Geometry`; avoid local `sys.path` hacks unless an Abaqus/HPC launch path truly requires them.
- Standard Python and Abaqus Python are separate environments. Code that touches Abaqus model/session/ODB APIs must be validated in the Abaqus interpreter or by syntax review only when Abaqus is unavailable.

## Data And Path Conventions

- Primary Windows checkout: `Z:/00-PhD-gitRepo`; use it for project sources, commands, and import hooks.
- Local p1 data is commonly rooted at `Z:/p1/data/Ti/...`.
- p2 saved runs belong under `Z:/p2`, outside Git, preserving archive-relative paths. No checkout may contain `data/`; other hosts require an explicit external root.
- Reports/examples use only `Z:/00-PhD-gitRepo/p2-DisorderML/samples/`; other checkouts keep tracked guidance only. MLdata and its subfolders contain only files directly consumed by ML training/testing; retain optional info/audits/reports elsewhere only when needed, following the existing archive structure without an extra provenance tier.
- QMUL HPC archive paths commonly use `/data/SEMS-TaoLab/Niccolo-Forte/...`; scratch paths may use `/gpfs/scratch/...`.
- Generated Abaqus files, transfer CSVs/NPZs, MLdata products, model checkpoints, HPO databases, Slurm logs, and notebook output bulk are research artifacts. Do not delete, rename, or commit them unless the task explicitly asks for that.
- `AGENTS.md` files and repo skills under `.agents/skills/` are tracked project infrastructure. Keep them synchronized with the implementation.

## Cross-Folder Workflow

- p1 `SIMscripts/` creates Abaqus `.inp` and `.odb` files and exports `transfer/` CSV/NPZ files.
- p1 `code/` notebooks turn `transfer/` files into ML-ready CSV/NPZ data and inspect mechanics/validation results.
- `resources/data_processing.py`, `resources/MLdata.py`, and `resources/MLmetrics.py` form the main bridge from p1 outputs to p2 training and diagnostics.
- p2 notebooks and HPC scripts should treat p1 data as upstream unless the user explicitly asks to change data generation.

## Editing Rules

- HIGH PRIORITY: study entry points, launchers and study-specific review scripts belong in `p2-DisorderML/HPC/studies/<Study>/`; `resources/` contains imported functions/classes, not job CLIs. Use descriptive study names; production scripts must not be newly labelled `-test`. Preserve established unrelated entry points and active immutable submissions.
- Read the script/notebook plus the shared helper it calls before changing behavior.
- Preserve file naming and index conventions between p1 and p2. In particular, `Ductile`, `Fracture`, `UT`, `FT`, `MULTI`, `per`, `disNodes`, `disStruts`, and simulation id `0` for the periodic reference have downstream meaning.
- Keep path overrides explicit in notebooks and scripts. Do not hide new global path constants inside shared modules.
- Ask for clarification before changing a scientific assumption, simulation setup, lattice definition, material law, loss definition, or data-filtering rule that cannot be verified from local context.
- For notebooks, prefer targeted cell/helper edits. Running full notebooks may require large data, Abaqus artifacts, or HPC resources.
- `update-repo.ps1` is a legacy, side-effectful multi-checkout script that runs pull, broad staging, commit, push, and SSH commands against hard-coded locations. Do not run it or treat it as the normal Git workflow unless the user explicitly requests that exact operation and the targets have been re-verified.
- The intended `origin` setup fetches from QMUL and has both QMUL and GitHub.com push URLs, so one `git push` updates both repositories. This is clone-local configuration: follow the README setup for a fresh checkout and inspect `git remote get-url --all --push origin` before relying on it.
- Multi-destination pushes are not atomic. If either destination fails, inspect both remote branch tips and reconcile the lagging destination without rewriting shared history.
- Always validate, commit and push completed work to every configured remote, per standing user authorization. Verify clean Git status and matching tips; report blockers.

## Living Documentation Requirement

Keeping human and agent guidance current is a completion requirement for every repository-changing task.

- Before editing, read this file, every applicable `AGENTS.md` down to the affected directory, and any matching repo skill under `.agents/skills/`.
- Use this file's map/workflow as minimum context. Read relevant README sections for unfamiliar, cross-paper/shared-resource work or changed human workflows; read the full README for repository-wide onboarding/restructuring or unclear scope.
- Read `PROJECT_STATUS.md` for planning, continuation, or handoff tasks; update it only when the current objective, evidence, blocker, decision, or next task changes.
- Update the nearest `AGENTS.md` for changed behavior, structure, paths, commands, dependencies, science, data contracts, outputs, risks or validation.
- Update `README.md` for changed human setup, navigation, execution, interpretation or maintenance.
- Update the relevant `SKILL.md`, scripts, references, and `agents/openai.yaml` whenever a reusable workflow or its discovery metadata changes.
- Add an `AGENTS.md` when a new subtree develops a distinct workflow, risk boundary, or validation contract.
- Edit current guidance in place. Do not leave stale statements or append routine dated change logs; Git history is the change log.
- If guidance already remains exact after a non-durable implementation change, explicitly verify that conclusion rather than silently skipping the documentation review.

Use the `maintain-repo-guidance` skill for every repository change and run its checker before completion.

Use `validate-repo-change` to select proportionate checks, `review-p1-p2-data-contract` whenever a change can affect the p1-to-p2 producer-consumer boundary, and `operate-p2-hpc` for cluster-facing p2 work.

`.github/workflows/guidance-integrity.yml` runs the changed-surface validator on GitHub.com pushes and pull requests. Its job is skipped before runner allocation on the QMUL Enterprise mirror, whose runner-backed jobs are currently cancelled. Keep local validation authoritative when remote Actions are unavailable.

Checkout and Python setup use Node 24-based v7 actions to avoid deprecated action runtimes.

## Research References

Capture verified sources used in retained work before handoff/archive, following
[README](README.md#retained-research-references) and ZoteroLibraryManager's workflow.
Import eligible sources or persist pending writes in its private intake; exclude
speculative/abandoned reading. The manager owns sorting. Keep private state out of Git.

## Analysis Deliverables

Per study, deliver ignored `samples/<suite>/README.md`: conclusions, labelled figures, controls, relative errors, uncertainty, limits, tables and provenance. Reports stay Markdown-only unless another format is explicitly requested. Use inline image syntax with absolute local paths, not PNG download links. Link the report. Raw runs stay in data stores; analysis stays in maintained code, not copied notebooks. Applies beyond HPC.

## Cleanup Constraint

Treat the user's cleanup preference as a high-priority engineering constraint, not a cosmetic preference.

- **High priority: no surprises.** Do not expand scope, relocate files or create alternate notebook copies without explicit authorization. Keep p2 notebooks in `p2-DisorderML/code/`. External data stores hold datasets, results and requested figures—not executable notebooks, scripts or deployment bundles. Use outside-repository temporary directories for validation intermediates.
- Keep active scripts, notebooks, and helpers compact, current, and intentional. Remove abandoned approaches, stale notebook cells and outputs, duplicated or superseded code, obsolete workaround helpers, and unnecessary compatibility or fallback scaffolding as part of the change.
- Prefer direct code and compact helpers: inline one-off helpers that do not clarify the flow, and move genuinely reused logic into the appropriate `resources/` module.
- Important exception: archived or explicitly historical folders, such as `p1-DisorderLatticeProperties/SIMscripts/OldScriptVersions/`, can preserve old versions. Active/current scripts and notebooks should not accumulate old experiments.

## Validation

- Run `python .agents/skills/validate-repo-change/scripts/validate_repo.py --changed` before claiming completion. Use a named `--scope` for broader review.
- Treat `SYNTAX-ONLY` and `SKIP` as explicit limits, especially for Abaqus, HPC, external data, and scientific behavior.
- Add focused checks for the changed behavior when the general validator cannot exercise it.
