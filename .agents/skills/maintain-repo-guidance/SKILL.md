---
name: maintain-repo-guidance
description: Keep human and agent-facing repository guidance synchronized with repository changes. Use whenever Codex creates, edits, moves, renames, or deletes repository files; changes behavior, workflows, commands, configuration, paths, dependencies, data contracts, scientific conventions, project structure, status, or validation; reviews a diff; or prepares a commit. Review and update the relevant AGENTS.md, README.md, PROJECT_STATUS.md, and .agents/skills content in the same change.
---

# Maintain Repository Guidance

Treat current human and agent instructions as part of the implementation, not as optional follow-up documentation.

## Before editing

1. Read the root `AGENTS.md` and `README.md`.
2. Read every `AGENTS.md` from the repository root down to the affected directory.
3. Inspect `.agents/skills/` for a workflow relevant to the task.
4. For planning, continuation, or handoff work, read the affected paper's `PROJECT_STATUS.md`.
5. Identify which current facts may change: structure, entry points, commands, paths, schemas, scientific assumptions, dependencies, outputs, safety constraints, validation expectations, or active handoff state.

## Keep the right surfaces current

- Update the closest `AGENTS.md` when future agents need the changed fact or rule.
- Update the paper-level or root `AGENTS.md` when the change crosses folders or affects repository-wide behavior.
- Update `README.md` when a human needs the changed setup, structure, command, workflow, file role, or limitation.
- Update the relevant skill when its trigger, procedure, script, reference, validation, or dependency changes.
- Update `PROJECT_STATUS.md` only when the current objective, authoritative working surface, verified state, blocker, decision, external requirement, or next task changes. Do not copy durable rules into it.
- Add an `AGENTS.md` when a new subtree has a distinct workflow, risk boundary, or validation contract.
- Keep scientific claims evidence-led. Use `Decision required` or `To be confirmed` when the repository does not establish the answer.

Edit current guidance in place. Do not append routine dated change logs or duplicate Git history. Remove stale statements and superseded instructions as part of the same change.

## Completion gate

1. Review the final diff file by file and map each implementation change to its affected guidance.
2. Run:

   ```powershell
   python .agents/skills/maintain-repo-guidance/scripts/check_guidance.py
   ```

3. Resolve every missing, ignored, stale, oversized, or uncovered guidance error before claiming completion.
4. Keep `.github/workflows/guidance-integrity.yml` aligned with the checker when its CLI or required runtime changes.
5. If a repository change genuinely leaves all current guidance exact, run the checker with an explicit reason:

   ```powershell
   python .agents/skills/maintain-repo-guidance/scripts/check_guidance.py --acknowledge-current-guidance "Why no instruction text changed"
   ```

   Use this exception only for a change that alters no durable fact or workflow. Report the reason in the final response.
6. State which `AGENTS.md`, `README.md`, and skill files were updated. Do not claim the task is complete from code validation alone.
