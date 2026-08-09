#!/usr/bin/env python3
"""Run deterministic, non-destructive validation for this repository."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ABAQUS_PATHS = (
    "p1-DisorderLatticeProperties/SIMscripts/",
    "p3-DisorderIcingMitigation/SIMscripts/",
    "resources/abaqus.py",
)
CONTRACT_PREFIXES = (
    "p1-DisorderLatticeProperties/SIMscripts/A1_",
    "p1-DisorderLatticeProperties/SIMscripts/A2_",
    "p1-DisorderLatticeProperties/code/",
    "p2-DisorderML/",
    "resources/data_processing.py",
    "resources/MLdata.py",
    "resources/MLmodels.py",
    "resources/MLmetrics.py",
    ".agents/skills/review-p1-p2-data-contract/",
    "tests/fixtures/synthetic_contract/",
)
IGNORED_PARTS = {".git", ".venv", "__pycache__", "OldScriptVersions"}
SCOPES = {
    "root": ("AGENTS.md", "README.md", ".agents/", ".github/", "tools/", "tests/"),
    "resources": ("resources/",),
    "p1": ("p1-DisorderLatticeProperties/",),
    "p2": ("p2-DisorderML/",),
    "p3": ("p3-DisorderIcingMitigation/",),
    "contract": CONTRACT_PREFIXES,
    "all": ("",),
}


@dataclass(frozen=True)
class Result:
    status: str
    check: str
    detail: str


def find_repo_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / ".git").exists():
                return directory
    raise RuntimeError("Could not find the Git repository root.")


def run(command: list[str], root: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def git_paths(root: Path, base_ref: str | None) -> set[str]:
    diff_range = f"{base_ref}...HEAD" if base_ref else "HEAD"
    changed = run(["git", "diff", "--name-only", diff_range, "--"], root)
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"], root)
    if changed.returncode or untracked.returncode:
        message = changed.stderr.strip() or untracked.stderr.strip() or "Git path discovery failed."
        raise RuntimeError(message)
    return {
        line.strip().replace("\\", "/")
        for line in (changed.stdout + "\n" + untracked.stdout).splitlines()
        if line.strip()
    }


def repository_paths(root: Path) -> set[str]:
    listed = run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], root)
    if listed.returncode:
        raise RuntimeError(listed.stderr.strip() or "Git path discovery failed.")
    return {line.strip().replace("\\", "/") for line in listed.stdout.splitlines() if line.strip()}


def in_scope(path: str, scope: str) -> bool:
    if any(part in IGNORED_PARTS for part in PurePosixPath(path).parts):
        return False
    return any(path == prefix or path.startswith(prefix) for prefix in SCOPES[scope])


def check_diff(root: Path, base_ref: str | None) -> Result:
    command = ["git", "diff", "--check"]
    if base_ref:
        command.append(f"{base_ref}...HEAD")
    completed = run(command, root)
    if completed.returncode:
        return Result("FAIL", "git diff --check", completed.stdout.strip() or completed.stderr.strip())
    return Result("PASS", "git diff --check", "No whitespace errors detected.")


def check_guidance(root: Path, base_ref: str | None) -> Result:
    command = [
        sys.executable,
        str(root / ".agents/skills/maintain-repo-guidance/scripts/check_guidance.py"),
    ]
    if base_ref:
        command.extend(["--base-ref", base_ref])
    completed = run(command, root)
    output = completed.stdout.strip() or completed.stderr.strip()
    status = "PASS" if completed.returncode == 0 else "FAIL"
    return Result(status, "repository guidance", output.splitlines()[-1] if output else "No output.")


def is_abaqus_path(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in ABAQUS_PATHS)


def check_python(root: Path, paths: list[str]) -> list[Result]:
    results: list[Result] = []
    for relative in paths:
        path = root / relative
        try:
            compile(path.read_text(encoding="utf-8-sig"), relative, "exec")
        except (OSError, SyntaxError, UnicodeError) as exc:
            results.append(Result("FAIL", f"Python syntax: {relative}", str(exc)))
            continue
        status = "SYNTAX-ONLY" if is_abaqus_path(relative) else "PASS"
        detail = "Compiled without executing the module."
        if status == "SYNTAX-ONLY":
            detail += " Abaqus API behavior was not exercised."
        results.append(Result(status, f"Python syntax: {relative}", detail))
    return results


def import_targets(paths: list[str], scope: str | None) -> list[str]:
    targets = {"resources"}
    for relative in paths:
        if not relative.startswith("resources/") or not relative.endswith(".py"):
            continue
        if relative in {"resources/__init__.py", "resources/abaqus.py", "resources/imports.py"}:
            continue
        targets.add(relative[:-3].replace("/", "."))
    if scope in {"resources", "all", "contract"}:
        targets.update(
            {
                "resources.calculations",
                "resources.data_processing",
                "resources.lattices",
            }
        )
    return sorted(targets)


def check_imports(root: Path, targets: list[str]) -> list[Result]:
    results: list[Result] = []
    for module in targets:
        completed = run([sys.executable, "-c", f"import {module}"], root, timeout=90)
        if completed.returncode == 0:
            results.append(Result("PASS", f"Import: {module}", "Imported in a fresh Python process."))
            continue
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "Import failed."
        status = "SKIP" if "ModuleNotFoundError" in completed.stderr else "FAIL"
        results.append(Result(status, f"Import: {module}", detail))
    return results


def check_notebooks(root: Path, paths: list[str]) -> list[Result]:
    results: list[Result] = []
    for relative in paths:
        try:
            notebook = json.loads((root / relative).read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                raise ValueError("top-level cells must be a list")
            if not isinstance(notebook.get("metadata", {}), dict):
                raise ValueError("top-level metadata must be an object")
            if not isinstance(notebook.get("nbformat"), int):
                raise ValueError("nbformat must be an integer")
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") not in {"code", "markdown", "raw"}:
                    raise ValueError(f"cell {index} has an invalid cell_type")
                if not isinstance(cell.get("source", []), (str, list)):
                    raise ValueError(f"cell {index} source must be text or a list")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            results.append(Result("FAIL", f"Notebook structure: {relative}", str(exc)))
        else:
            results.append(Result("PASS", f"Notebook structure: {relative}", "Valid JSON and required notebook structure."))
    return results


def check_shell(root: Path, paths: list[str]) -> list[Result]:
    if not paths:
        return []
    bash = shutil.which("bash")
    if not bash:
        return [Result("SKIP", "Shell syntax", "Bash is unavailable; no shell scripts were executed.")]
    results: list[Result] = []
    for relative in paths:
        completed = run([bash, "-n", str(root / relative)], root)
        detail = completed.stderr.strip() or "bash -n passed."
        results.append(Result("PASS" if completed.returncode == 0 else "FAIL", f"Shell syntax: {relative}", detail))
    return results


def check_powershell(root: Path, paths: list[str]) -> list[Result]:
    if not paths:
        return []
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if not executable:
        return [Result("SKIP", "PowerShell syntax", "PowerShell is unavailable.")]
    results: list[Result] = []
    for relative in paths:
        literal = str(root / relative).replace("'", "''")
        script = (
            f"$e=$null; [System.Management.Automation.Language.Parser]::ParseFile('{literal}', "
            "[ref]$null, [ref]$e) > $null; if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }"
        )
        completed = run([executable, "-NoProfile", "-Command", script], root)
        detail = completed.stdout.strip() or completed.stderr.strip() or "PowerShell parser passed."
        results.append(Result("PASS" if completed.returncode == 0 else "FAIL", f"PowerShell syntax: {relative}", detail))
    return results


def check_contract_fixture(root: Path) -> list[Result]:
    path = root / "tests/fixtures/synthetic_contract/contract.json"
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("fixture_kind") != "synthetic_non_scientific_contract":
            raise ValueError("fixture must identify itself as synthetic and non-scientific")
        sample_ids = fixture["sample_ids"]
        aligned = [
            fixture["inputs"]["sample_ids"],
            fixture["curve_outputs"]["sample_ids"],
            fixture["field_outputs"]["sample_ids"],
            fixture["manifest"]["sample_ids"],
        ]
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_ids must be non-empty and unique")
        if any(ids != sample_ids for ids in aligned):
            raise ValueError("input, output, field, and manifest sample IDs are not aligned")
        if fixture["periodic_sample_id"] != 0 or 0 not in sample_ids:
            raise ValueError("periodic/reference sample id 0 is missing")
        field = fixture["field_outputs"]
        shape = field["value_shape"]
        expected = [
            len(sample_ids),
            len(field["frame_values"]),
            len(field["node_labels"]),
            len(field["components"]),
        ]
        if shape != expected:
            raise ValueError(f"field value_shape {shape} does not match metadata {expected}")
        if field["valid_mask_shape"] != shape[:3]:
            raise ValueError("valid_mask_shape must match sample/frame/node axes")
        if field["node_coordinates_shape"] != [shape[2], 2]:
            raise ValueError("node coordinate shape must be [node, 2]")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [Result("FAIL", "Synthetic contract fixture", str(exc))]

    results = [Result("PASS", "Synthetic contract fixture", "IDs, periodic reference, shapes, and metadata are aligned.")]
    with tempfile.TemporaryDirectory() as temporary:
        csv_path = Path(temporary) / "manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "included"])
            writer.writerows((sample_id, True) for sample_id in sample_ids)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            roundtrip = [int(row["sample_id"]) for row in csv.DictReader(handle)]
        if roundtrip != sample_ids:
            results.append(Result("FAIL", "Synthetic CSV round trip", "Sample IDs changed during serialization."))
        else:
            results.append(Result("PASS", "Synthetic CSV round trip", "Aligned sample IDs survived serialization."))

        if importlib.util.find_spec("numpy") is None:
            results.append(Result("SKIP", "Synthetic NPZ round trip", "NumPy is unavailable."))
        else:
            import numpy as np

            npz_path = Path(temporary) / "field.npz"
            np.savez(
                npz_path,
                sample_ids=np.asarray(sample_ids),
                field=np.zeros(field["value_shape"], dtype=float),
                valid_mask=np.ones(field["valid_mask_shape"], dtype=bool),
                frame_values=np.asarray(field["frame_values"], dtype=float),
                node_labels=np.asarray(field["node_labels"]),
                coords0=np.zeros(field["node_coordinates_shape"], dtype=float),
                components=np.asarray(field["components"]),
            )
            with np.load(npz_path) as loaded:
                aligned_npz = loaded["sample_ids"].tolist() == sample_ids
                shape_npz = list(loaded["field"].shape) == field["value_shape"]
                mask_npz = list(loaded["valid_mask"].shape) == field["valid_mask_shape"]
            status = "PASS" if aligned_npz and shape_npz and mask_npz else "FAIL"
            detail = "Field IDs and shapes survived NPZ serialization." if status == "PASS" else "NPZ round trip changed IDs or shapes."
            results.append(Result(status, "Synthetic NPZ round trip", detail))
    return results


def external_limitations(paths: list[str]) -> list[Result]:
    results: list[Result] = []
    if any(path.endswith(".py") and is_abaqus_path(path) for path in paths):
        results.append(Result("SKIP", "Abaqus behavior", "No CAE/ODB job or Abaqus API execution was attempted."))
    if any(path.endswith(".sh") and ("HPC/" in path or "/SIMscripts/" in path) for path in paths):
        results.append(Result("SKIP", "HPC behavior", "No Slurm job, archive transfer, or scratch operation was attempted."))
    if any(
        path.endswith(".ipynb")
        or (path.endswith((".py", ".sh")) and any(path.startswith(prefix) for prefix in CONTRACT_PREFIXES))
        for path in paths
    ):
        results.append(Result("SKIP", "Research-data behavior", "No Z: data, saved model, ODB, or full scientific workflow was executed."))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--changed", action="store_true", help="Validate working-tree changes (default).")
    selection.add_argument("--scope", choices=sorted(SCOPES), help="Validate one repository scope.")
    parser.add_argument("--base-ref", help="Validate committed changes since this Git revision, for CI use.")
    args = parser.parse_args()

    root = find_repo_root()
    scope = args.scope
    paths = git_paths(root, args.base_ref) if scope is None else repository_paths(root)
    if scope is not None:
        paths = {path for path in paths if in_scope(path, scope)}
    paths = {path for path in paths if (root / path).is_file() and not any(part in IGNORED_PARTS for part in PurePosixPath(path).parts)}
    selected = sorted(paths)

    results = [check_diff(root, args.base_ref), check_guidance(root, args.base_ref)]
    python_paths = [path for path in selected if path.endswith(".py")]
    notebook_paths = [path for path in selected if path.endswith(".ipynb")]
    shell_paths = [path for path in selected if path.endswith(".sh")]
    powershell_paths = [path for path in selected if path.endswith(".ps1")]
    results.extend(check_python(root, python_paths))
    results.extend(check_imports(root, import_targets(python_paths, scope)))
    results.extend(check_notebooks(root, notebook_paths))
    results.extend(check_shell(root, shell_paths))
    results.extend(check_powershell(root, powershell_paths))
    results.extend(check_contract_fixture(root))
    results.extend(external_limitations(selected))

    if not selected:
        results.append(Result("PASS", "File selection", "No changed files matched file-specific validators."))
    for result in results:
        print(f"{result.status}: {result.check} - {result.detail}")
    failures = sum(result.status == "FAIL" for result in results)
    print(f"SUMMARY: {len(results) - failures} non-failing, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
