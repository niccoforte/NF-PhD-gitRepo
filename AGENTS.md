# Repository Agent Context

This repository is organized by PhD paper folder with one shared Python package. Start with this file, then read the nearest nested `AGENTS.md` before editing inside a paper folder.

For planning, continuation, or handoff work, also read that paper's `PROJECT_STATUS.md`. It contains changing work state; durable rules belong in `AGENTS.md` and reusable procedures belong in `.agents/skills/`.

## Repository Map

- `p1-DisorderLatticeProperties/` is the Abaqus FEA, post-processing, and data-generation workflow for quasi-disordered 2D lattices under uniaxial/ductile and compact-tension/fracture loading.
- `p2-DisorderML/` is the machine-learning and surrogate-modeling workflow built on p1 outputs. It trains and diagnoses curve and field surrogates and contains cluster training/HPO scripts.
- `p3-DisorderIcingMitigation/` contains early, provisional Abaqus work. Read its `AGENTS.md`; its icing-specific scientific scope is still `To be confirmed`, so do not borrow p1 assumptions or present the current script as a validated icing model.
- `resources/` is the shared package used by p1 and p2. Changes there can affect Abaqus scripts, notebooks, ML training, post-processing, and saved artifact loading.

## Environment And Imports

- `pyproject.toml` exposes `resources` as the editable package `phd-shared-resources`.
- `setup.ps1` installs the shared package for standard Python and Abaqus Python; `remove-setup.ps1` reverses that setup.
- Use imports like `from resources.MLdata import DATA` or `from resources.lattices import Geometry`; avoid local `sys.path` hacks unless an Abaqus/HPC launch path truly requires them.
- Standard Python and Abaqus Python are separate environments. Code that touches Abaqus model/session/ODB APIs must be validated in the Abaqus interpreter or by syntax review only when Abaqus is unavailable.

## Data And Path Conventions

- Local p1 data is commonly rooted at `Z:/p1/data/Ti/...`.
- Local p2 saved runs are commonly rooted at `Z:/p2`.
- QMUL HPC archive paths commonly use `/data/SEMS-TaoLab/Niccolo-Forte/...`; scratch paths may use `/gpfs/scratch/...`.
- Generated Abaqus files, transfer CSVs/NPZs, MLdata products, model checkpoints, HPO databases, Slurm logs, and notebook output bulk are research artifacts. Do not delete, rename, or commit them unless the task explicitly asks for that.
- `AGENTS.md` files and repo skills under `.agents/skills/` are tracked project infrastructure. Keep them synchronized with the implementation.

## Cross-Folder Workflow

- p1 `SIMscripts/` creates Abaqus `.inp` and `.odb` files and exports `transfer/` CSV/NPZ files.
- p1 `code/` notebooks turn `transfer/` files into ML-ready CSV/NPZ data and inspect mechanics/validation results.
- `resources/data_processing.py`, `resources/MLdata.py`, and `resources/MLmetrics.py` form the main bridge from p1 outputs to p2 training and diagnostics.
- p2 notebooks and HPC scripts should treat p1 data as upstream unless the user explicitly asks to change data generation.

## Editing Rules

- Read the script/notebook plus the shared helper it calls before changing behavior.
- Preserve file naming and index conventions between p1 and p2. In particular, `Ductile`, `Fracture`, `UT`, `FT`, `MULTI`, `per`, `disNodes`, `disStruts`, and simulation id `0` for the periodic reference have downstream meaning.
- Keep path overrides explicit in notebooks and scripts. Do not hide new global path constants inside shared modules.
- Ask for clarification before changing a scientific assumption, simulation setup, lattice definition, material law, loss definition, or data-filtering rule that cannot be verified from local context.
- For notebooks, prefer targeted cell/helper edits. Running full notebooks may require large data, Abaqus artifacts, or HPC resources.
- `update-repo.ps1` is a legacy, side-effectful multi-checkout script that runs pull, broad staging, commit, push, and SSH commands against hard-coded locations. Do not run it or treat it as the normal Git workflow unless the user explicitly requests that exact operation and the targets have been re-verified.
- The intended `origin` setup fetches from QMUL and has both QMUL and GitHub.com push URLs, so one `git push` updates both repositories. This is clone-local configuration: follow the README setup for a fresh checkout and inspect `git remote get-url --all --push origin` before relying on it.
- Multi-destination pushes are not atomic. If either destination fails, inspect both remote branch tips and reconcile the lagging destination without rewriting shared history.

## Living Documentation Requirement

Keeping human and agent guidance current is a completion requirement for every repository-changing task.

- Before editing, read the root `README.md`, this file, the closest nested `AGENTS.md`, and any matching repo skill under `.agents/skills/`.
- Read `PROJECT_STATUS.md` for planning, continuation, or handoff tasks; update it only when the current objective, evidence, blocker, decision, or next task changes.
- During the same change, update the closest relevant `AGENTS.md` whenever behavior, structure, paths, commands, dependencies, scientific conventions, data contracts, outputs, risks, or validation expectations change.
- Update `README.md` whenever the changed fact affects human setup, navigation, execution, interpretation, or maintenance.
- Update the relevant `SKILL.md`, scripts, references, and `agents/openai.yaml` whenever a reusable workflow or its discovery metadata changes.
- Add an `AGENTS.md` when a new subtree develops a distinct workflow, risk boundary, or validation contract.
- Edit current guidance in place. Do not leave stale statements or append routine dated change logs; Git history is the change log.
- If guidance already remains exact after a non-durable implementation change, explicitly verify that conclusion rather than silently skipping the documentation review.

Use the `maintain-repo-guidance` skill for every repository change and run its checker before completion.

Use `validate-repo-change` to select proportionate checks, `review-p1-p2-data-contract` whenever a change can affect the p1-to-p2 producer-consumer boundary, and `operate-p2-hpc` for cluster-facing p2 work.

`.github/workflows/guidance-integrity.yml` runs the changed-surface validator on pushes and pull requests. Keep local validation authoritative when remote Actions are unavailable.

## Cleanup Constraint

Treat the user's cleanup preference as a high-priority engineering constraint, not a cosmetic preference.

- Active scripts, notebooks, and helper modules should stay clean, compact, current, and intentional.
- Do not leave behind dated, stale, legacy, failed, partial, duplicated, or superseded code in active work areas.
- If an attempted approach is abandoned, remove it. If a helper exists only because of an old workaround and is no longer needed, remove it.
- If notebook cells are replaced by helper functions, remove the stale code paths and clear stale outputs instead of preserving them "just in case".
- Avoid verbose safety-net layers, fallback branches, legacy compatibility scaffolding, and overly defensive helper functions unless there is a concrete current need.
- Prefer direct, readable code and compact helpers. If a helper is used once and does not clarify the main flow, consider inlining it. If logic is reused or makes notebooks cleaner, move it into the appropriate `resources/` module.
- Important exception: archived or explicitly historical folders, such as `p1-DisorderLatticeProperties/SIMscripts/OldScriptVersions/`, can preserve old versions. Active/current scripts and notebooks should not accumulate old experiments.
- In short: clean up as you go.

## Validation

- Run `python tools/validate_repo.py --changed` before claiming completion. Use a named `--scope` for broader review.
- Treat `SYNTAX-ONLY` and `SKIP` as explicit limits, especially for Abaqus, HPC, external data, and scientific behavior.
- Add focused checks for the changed behavior when the general validator cannot exercise it.
