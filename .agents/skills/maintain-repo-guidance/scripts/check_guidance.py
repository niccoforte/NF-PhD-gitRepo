#!/usr/bin/env python3
"""Validate repository guidance structure and changed-file coverage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath


MAX_INSTRUCTION_BYTES = 32 * 1024
SOFT_INSTRUCTION_BYTES = 24 * 1024
REQUIRED_FILES = (
    "AGENTS.md",
    "README.md",
    "setup-Windows.ps1",
    "setup-macOS.sh",
    "p1-DisorderLatticeProperties/AGENTS.md",
    "p1-DisorderLatticeProperties/PROJECT_STATUS.md",
    "p1-DisorderLatticeProperties/SIMscripts/AGENTS.md",
    "p1-DisorderLatticeProperties/code/AGENTS.md",
    "p2-DisorderML/AGENTS.md",
    "p2-DisorderML/PROJECT_STATUS.md",
    "p2-DisorderML/HPC/AGENTS.md",
    "p2-DisorderML/code/AGENTS.md",
    "p3-DisorderIcingMitigation/AGENTS.md",
    "p3-DisorderIcingMitigation/PROJECT_STATUS.md",
    "resources/AGENTS.md",
    ".agents/skills/maintain-repo-guidance/SKILL.md",
    ".agents/skills/maintain-repo-guidance/agents/openai.yaml",
    ".agents/skills/maintain-repo-guidance/scripts/check_guidance.py",
    ".agents/skills/review-p1-p2-data-contract/references/data-contract.md",
    ".agents/skills/validate-repo-change/scripts/validate_repo.py",
    ".agents/skills/validate-repo-change/fixtures/synthetic_contract/contract.json",
    ".github/workflows/guidance-integrity.yml",
)
STALE_TEXT = (
    "repository currently ignores `*AGENTS.md`",
    "local guidance unless force-added",
    "`setup.ps1`",
    "`remove-setup.ps1`",
)


def find_repo_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        start = candidate if candidate.is_dir() else candidate.parent
        for directory in (start, *start.parents):
            if (directory / ".git").exists():
                return directory
    raise RuntimeError("Could not find the Git repository root.")


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def changed_paths(root: Path, base_ref: str | None = None) -> set[str]:
    diff_range = f"{base_ref}...HEAD" if base_ref else "HEAD"
    changed = run_git(root, "diff", "--name-only", diff_range, "--")
    untracked = run_git(root, "ls-files", "--others", "--exclude-standard")
    if changed.returncode != 0:
        raise RuntimeError(changed.stderr.strip() or "git diff failed")
    if untracked.returncode != 0:
        raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
    paths = {
        line.strip().replace("\\", "/")
        for line in (changed.stdout + "\n" + untracked.stdout).splitlines()
        if line.strip()
    }
    return paths


def is_guidance_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        pure.name in {"AGENTS.md", "AGENTS.override.md", "README.md", "PROJECT_STATUS.md"}
        or path.startswith(".agents/skills/")
    )


def guidance_candidates(path: str) -> tuple[str, ...]:
    if path.startswith("p1-DisorderLatticeProperties/SIMscripts/"):
        return (
            "p1-DisorderLatticeProperties/SIMscripts/AGENTS.md",
            "p1-DisorderLatticeProperties/AGENTS.md",
            "p1-DisorderLatticeProperties/PROJECT_STATUS.md",
            "README.md",
        )
    if path.startswith("p1-DisorderLatticeProperties/code/"):
        return (
            "p1-DisorderLatticeProperties/code/AGENTS.md",
            "p1-DisorderLatticeProperties/AGENTS.md",
            "p1-DisorderLatticeProperties/PROJECT_STATUS.md",
            "README.md",
        )
    if path.startswith("p1-DisorderLatticeProperties/"):
        return ("p1-DisorderLatticeProperties/AGENTS.md", "p1-DisorderLatticeProperties/PROJECT_STATUS.md", "README.md")
    if path.startswith("p2-DisorderML/HPC/"):
        return (
            "p2-DisorderML/HPC/AGENTS.md",
            "p2-DisorderML/AGENTS.md",
            "p2-DisorderML/PROJECT_STATUS.md",
            "README.md",
        )
    if path.startswith("p2-DisorderML/code/"):
        return (
            "p2-DisorderML/code/AGENTS.md",
            "p2-DisorderML/AGENTS.md",
            "p2-DisorderML/PROJECT_STATUS.md",
            "README.md",
        )
    if path.startswith("p2-DisorderML/"):
        return ("p2-DisorderML/AGENTS.md", "p2-DisorderML/PROJECT_STATUS.md", "README.md")
    if path.startswith("p3-DisorderIcingMitigation/"):
        return ("p3-DisorderIcingMitigation/AGENTS.md", "p3-DisorderIcingMitigation/PROJECT_STATUS.md", "README.md")
    if path.startswith("resources/"):
        return ("resources/AGENTS.md", "AGENTS.md", "README.md")
    return ("AGENTS.md", "README.md")


def instruction_chain(root: Path, agent_file: Path) -> list[Path]:
    relative_dir = agent_file.relative_to(root).parent
    chain = [root / "AGENTS.md"]
    current = root
    for part in relative_dir.parts:
        current /= part
        candidate = current / "AGENTS.md"
        if candidate != root / "AGENTS.md" and candidate.exists():
            chain.append(candidate)
    return chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="Compare committed changes from this Git revision through HEAD, for CI use.",
    )
    parser.add_argument(
        "--acknowledge-current-guidance",
        metavar="REASON",
        help="Allow implementation changes without a guidance-file diff when current guidance remains exact.",
    )
    parser.add_argument(
        "--no-diff-check",
        action="store_true",
        help="Run structural checks without checking changed-file guidance coverage.",
    )
    args = parser.parse_args()

    root = find_repo_root()
    errors: list[str] = []
    notes: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"Missing required guidance file: {relative}")

    for relative in ("AGENTS.md", "README.md"):
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for stale in STALE_TEXT:
            if stale.casefold() in text.casefold():
                errors.append(f"Stale guidance in {relative}: {stale}")

    root_agents = (root / "AGENTS.md").read_text(encoding="utf-8") if (root / "AGENTS.md").exists() else ""
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    for token, relative in (
        ("maintain-repo-guidance", "AGENTS.md"),
        ("validate-repo-change", "AGENTS.md"),
        ("review-p1-p2-data-contract", "AGENTS.md"),
        ("operate-p2-hpc", "AGENTS.md"),
        (".agents/skills", "README.md"),
        ("maintain-repo-guidance", "README.md"),
        ("validate-repo-change", "README.md"),
        ("review-p1-p2-data-contract", "README.md"),
        ("operate-p2-hpc", "README.md"),
        (".github/workflows/guidance-integrity.yml", "README.md"),
        ("p3-DisorderIcingMitigation/AGENTS.md", "README.md"),
        ("PROJECT_STATUS.md", "README.md"),
        (".agents/skills/validate-repo-change/scripts/validate_repo.py", "README.md"),
        ("setup-Windows.ps1", "AGENTS.md"),
        ("setup-macOS.sh", "AGENTS.md"),
        ("setup-Windows.ps1", "README.md"),
        ("setup-macOS.sh", "README.md"),
    ):
        content = root_agents if relative == "AGENTS.md" else readme
        if token not in content:
            errors.append(f"{relative} does not mention required guidance token: {token}")

    skill_root = root / ".agents/skills"
    if skill_root.exists():
        for skill_path in sorted(skill_root.glob("*/SKILL.md")):
            skill_text = skill_path.read_text(encoding="utf-8")
            skill_name = skill_path.parent.name
            if "TODO" in skill_text or f"name: {skill_name}" not in skill_text:
                errors.append(f"Skill is incomplete or has invalid metadata: {skill_path.relative_to(root)}")
            if not (skill_path.parent / "agents/openai.yaml").is_file():
                errors.append(f"Skill is missing agents/openai.yaml: {skill_name}")

    important_paths = {"resources/abaqus.py", *REQUIRED_FILES}
    important_paths.update(
        path.relative_to(root).as_posix()
        for path in root.rglob("AGENTS.md")
        if ".git" not in path.parts
    )
    important_paths.update(
        path.relative_to(root).as_posix()
        for path in root.rglob("PROJECT_STATUS.md")
        if ".git" not in path.parts
    )
    if skill_root.exists():
        important_paths.update(
            path.relative_to(root).as_posix()
            for path in skill_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    for relative in sorted(important_paths):
        ignored = run_git(root, "check-ignore", "--no-index", "--quiet", "--", relative)
        if ignored.returncode == 0:
            errors.append(f"Important guidance/source path is ignored: {relative}")

    for agent_file in sorted(root.rglob("AGENTS.md")):
        if ".git" in agent_file.parts:
            continue
        chain = instruction_chain(root, agent_file)
        size = sum(path.stat().st_size for path in chain)
        relative = agent_file.relative_to(root).as_posix()
        notes.append(f"Instruction chain {relative}: {size} bytes")
        if size > MAX_INSTRUCTION_BYTES:
            errors.append(
                f"Instruction chain for {relative} is {size} bytes; "
                f"default limit is {MAX_INSTRUCTION_BYTES}."
            )
        elif size > SOFT_INSTRUCTION_BYTES:
            warnings.append(
                f"Instruction chain for {relative} is {size} bytes; review for "
                f"duplication before it reaches the {MAX_INSTRUCTION_BYTES}-byte hard limit."
            )

    if not args.no_diff_check:
        changed = changed_paths(root, args.base_ref)
        guidance_changed = {path for path in changed if is_guidance_path(path)}
        uncovered: list[tuple[str, tuple[str, ...]]] = []
        for path in sorted(changed - guidance_changed):
            candidates = guidance_candidates(path)
            if not guidance_changed.intersection(candidates):
                uncovered.append((path, candidates))
        if uncovered and not args.acknowledge_current_guidance:
            for path, candidates in uncovered:
                errors.append(
                    f"No relevant guidance changed for {path}; review one of: "
                    + ", ".join(candidates)
                )
        elif uncovered:
            reason = args.acknowledge_current_guidance.strip()
            if not reason:
                errors.append("The current-guidance acknowledgement reason is empty.")
            else:
                notes.append(f"Guidance-current acknowledgement: {reason}")

    for note in notes:
        print(f"NOTE: {note}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Guidance validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
