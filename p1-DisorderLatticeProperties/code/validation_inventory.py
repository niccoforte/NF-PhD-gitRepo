"""Catalogue p1 validation audit products and known task scratch; never delete.

Original research sources, maintained code and reproducible derived products
are distinguished from named task-owned temporary work. Unknown temporary
directories are deliberately not swept or designated for deletion.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


REPO = Path("Z:/00-PhD-gitRepo")
ROOT = Path("Z:/p1/sims/Ti/Validation")
TEMP = Path(os.environ.get("TEMP", "C:/Users/exy053/AppData/Local/Temp"))
WORKBOOK = Path("C:/Users/exy053/OneDrive - Queen Mary, University of London/Documents/Research/p1-Lattices/Ductility-FractureToughness.xlsx")
LITERATURE = WORKBOOK.parent.parent / "00-Literature"
REFERENCES = (LITERATURE / "01_Foundations_Fracture_And_Solid_Mechanics/Crack_Tip_Fields_And_R_Curves/Hsieh2020-Deshpande-FT&RcurveCellularMaterials.pdf",
              LITERATURE / "08_Standards_Test_Methods_And_Data_Reduction/ASTM_Fracture/ASTMe1820.pdf")


def files(root):
    """Do not traverse junctions/symlinks into unrelated repositories or runtimes."""
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if (Path(parent) / name).exists()
                   and not (Path(parent) / name).is_symlink()
                   and not (getattr((Path(parent) / name).lstat(), "st_file_attributes", 0) & 1024)]
        for name in names:
            path = Path(parent) / name
            if not path.is_symlink():
                yield path


def record(path, group, retention, reason):
    stat = path.stat()
    digest = "see source/solver receipt or original manifest"
    if stat.st_size <= 1024 * 1024:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except PermissionError:
            digest = "locked during active use; refresh after the queue settles"
    return {"path": str(path.resolve()), "group": group, "bytes": stat.st_size,
            "mtime_UTC": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "retention": retention, "reason": reason,
            "sha256_small_file": digest}


def queue_ledger(audit):
    """Resolve queued entries transferred between manifests without double counts."""
    occurrences = {}
    priority = {"running": 5, "completed": 4, "failed": 3, "queued": 1, "queued_after_parent": 0}
    for path in audit.rglob("state.json"):
        if "scratch" in path.parts:
            continue
        state = json.loads(path.read_text())
        for job in state.get("jobs", []):
            occurrences.setdefault(job["job"], []).append({
                "queue_state": str(path), "status": job["status"],
                "queue_phase": state.get("phase"), "job": job["job"]})
    for plan_path in audit.glob("serial-continuation-*.json"):
        plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
        for entry in plan["following_manifests"]:
            manifest_path = Path(entry["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for job in manifest["jobs"]:
                occurrences.setdefault(job["job"], []).append({
                    "job": job["job"], "status": "queued_after_parent", "manifest": str(manifest_path),
                    "queue_state": str(Path(manifest["queue_root"])/"state.json"),
                    "queue_phase": "planned_serial_continuation", "parent_queue": plan["parent_queue"]})
    jobs = []
    for name, records in sorted(occurrences.items()):
        chosen = max(records, key=lambda row: priority.get(row["status"], 0))
        jobs.append(dict(chosen, occurrences=records,
                         transferred_queued_entries=sum(row["status"] == "queued" for row in records)
                         if chosen["status"] != "queued" else max(0, len(records)-1)))
    result = {"generated_UTC": datetime.now(timezone.utc).isoformat(), "jobs": jobs,
              "counts": {status: sum(row["status"] == status for row in jobs)
                         for status in sorted({row["status"] for row in jobs})},
              "limit": "Unique queued/running job names only; failed solver states remain failed. Source result receipts, not this index, establish extraction eligibility."}
    (audit / "queue-ledger.json").write_text(json.dumps(result, indent=2))
    return result


def run(root=ROOT, repo=REPO, temp=TEMP):
    audit = Path(root) / "audit"
    rows = []
    destination = audit / "file-inventory.csv"
    summary_path = audit / "file-inventory.json"
    queues = queue_ledger(audit)
    vanished = []
    for path in files(audit):
        if path in (destination, summary_path):
            continue
        if "scratch" in path.relative_to(audit).parts:
            group, retention = "solver_scratch", "temporary_after_solver_and_delivery_verified"
            reason = "Review active queue and delivery receipt before cleanup; never delete a running solver's files"
        elif "workbook-before" in path.name:
            group, retention = "workbook_backup", "retain_until_paper_values_and_integration_accepted"
            reason = "Recover exact source bytes; not a disposable duplicate"
        else:
            group, retention = "audit_evidence_or_derived_product", "retain_or_rebuild_only_after_final_review"
            reason = "Source/solver receipts distinguish retained evidence from reproducible plots and tables"
        try:
            rows.append(record(path, group, retention, reason))
        except FileNotFoundError:
            vanished.append(str(path))  # Atomic queue/solver temporary files may settle during a snapshot.
    code = Path(repo) / "p1-DisorderLatticeProperties/code"
    maintained = list(code.glob("validation_*.py")) + list(code.glob("test_validation_*.py"))
    maintained += [code / "validation_workbook_fig10.mjs", code / "ValConvPlots.ipynb",
                   Path(repo) / "resources/lattices.py", Path(repo) / "resources/test_lattice_validation.py",
                   Path(repo) / "p2-DisorderML/samples/p1-validation/README.md"]
    for path in maintained:
        if path.exists():
            rows.append(record(path, "maintained_code_or_report", "retain", "Reproduction entry point or consolidated study report"))
    if WORKBOOK.exists():
        rows.append(record(WORKBOOK, "canonical_workbook", "retain", "Paper source with additive audit/digitisation; preserve original values"))
    for reference in REFERENCES:
        if reference.exists():
            rows.append(record(reference, "published_source", "retain", "Source equations, failure criterion and specimen calibration"))
    scratch_roots = sorted({path for pattern in ("codex-p1-review-20261026", "codex-p1-final-check-20261028-*", "codex-p1-validation-*")
                            for path in Path(temp).glob(pattern) if path.is_dir()})
    for directory in scratch_roots:
        for path in files(directory):
            try:
                rows.append(record(path, "known_task_temporary", "cleanup_candidate_after_final_review",
                                   "Named validation scratch only; verify unique scripts/evidence are preserved before deletion"))
            except FileNotFoundError:
                vanished.append(str(path))
    rows.sort(key=lambda row: row["path"].lower())
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"generated_UTC": datetime.now(timezone.utc).isoformat(), "files": len(rows),
               "bytes": sum(row["bytes"] for row in rows), "inventory": str(destination),
               "known_temporary_roots": [str(path) for path in scratch_roots],
               "queue_counts_unique_job_names": queues["counts"],
               "transient_files_vanished_during_snapshot": vanished,
               "groups": {group: {"files": sum(row["group"] == group for row in rows),
                                    "bytes": sum(row["bytes"] for row in rows if row["group"] == group)}
                          for group in sorted({row["group"] for row in rows})},
               "original_sources": {"case_index": str(Path(root) / "case-index.csv"),
                                    "full_original_manifest": str(Path(root) / "file-manifest.json")},
               "limits": "Read-only inventory, not deletion authority. No traversal of linked runtime/repository roots; no generic Temp sweep. Original INP/ODB manifests remain authoritative. Refresh before final cleanup."}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--temp", type=Path, default=TEMP)
    args = parser.parse_args()
    run(args.root, args.repo, args.temp)
