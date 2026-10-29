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
import xml.etree.ElementTree as ET
from zipfile import ZipFile


REPO = Path("Z:/00-PhD-gitRepo")
ROOT = Path("Z:/p1/sims/Ti/Validation")
TEMP = Path(os.environ.get("TEMP", "C:/Users/exy053/AppData/Local/Temp"))
WORKBOOK = Path("C:/Users/exy053/OneDrive - Queen Mary, University of London/Documents/Research/p1-Lattices/Ductility-FractureToughness.xlsx")
LITERATURE = WORKBOOK.parent.parent / "00-Literature"
REFERENCES = (LITERATURE / "01_Foundations_Fracture_And_Solid_Mechanics/Crack_Tip_Fields_And_R_Curves/Hsieh2020-Deshpande-FT&RcurveCellularMaterials.pdf",
              LITERATURE / "08_Standards_Test_Methods_And_Data_Reduction/ASTM_Fracture/ASTMe1820.pdf")
REFERENCE_INTAKE = Path("Z:/ZoteroLibraryManager/ReferenceIntake/p1-validation-20261028.json")
EARLY_REVIEW = Path("C:/Users/exy053/OneDrive - Queen Mary, University of London/Documents/ChatGPT/p1 Lattices/p1-validation-preliminary-review.md")


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


def cleanup_review(audit, repo, scratch_roots):
    """Keep a reviewable cross-file cleanup plan; never perform its actions."""
    workbook = {"path": str(WORKBOOK), "edited_this_pass": False}
    if WORKBOOK.exists():
        workbook["sha256"] = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest()
        with ZipFile(WORKBOOK) as package:
            tree = ET.fromstring(package.read("xl/workbook.xml"))
            workbook["sheets"] = [sheet.attrib["name"] for sheet in
                                  tree.findall("{*}sheets/{*}sheet")]
    components = [
        {"component": "Original simulations and source digitisation", "action": "retain",
         "paths": [str(audit.parent / "case-index.csv"), str(audit.parent / "file-manifest.json"),
                   str(audit / "benchmark/fig8-original-digitisation.csv"), str(audit / "benchmark/fig10-digitised.csv")],
         "reason": "Original INP/ODB/export identities and both initial/final literature points remain source evidence even if only initiation is plotted."},
        {"component": "Paper outputs and calculation source", "action": "retain_one_current_workflow",
         "paths": [str(audit / "paper"), str(audit / "final"),
                   str(repo / "p1-DisorderLatticeProperties/code/ValConvPlots.ipynb")],
         "reason": "Keep the agreed calculation CSV, clean initiation figures, captions and notebook caller; avoid independent spreadsheet/Python copies of final values."},
        {"component": "Workbook legacy results", "action": "reconcile_after_method_is_accepted",
         "paths": [str(WORKBOOK)], "areas": ["FTval", "Deshpande", "Validation audit"],
         "reason": "FTval still holds historical values/placeholders. The added Validation audit sheet contains sensitivity controls, not final results. Decide its replacement/removal and update affected notebook/chart links only after the final method is settled. Preserve original reference cells and Deshpande!A50:L73 digitisation."},
        {"component": "Workbook backups", "action": "retain_until_integration_and_original_recovery_are_verified",
         "paths": [str(path) for path in sorted(audit.glob("workbook-before*"))],
         "reason": "Do not discard the only verified before-state while paper/workbook reconciliation remains open."},
        {"component": "Diagnostic plots, tables and reports", "action": "consolidate_after_final_review",
         "paths": [str(audit / name) for name in ("calculations", "modal", "diagnostics", "work-partition", "report-history", "benchmark-review.md")] + [str(EARLY_REVIEW)],
         "reason": "Retain the minimal evidence supporting event, modulus and negative-work decisions; remove superseded renders/tables after checking all report and notebook dependencies."},
        {"component": "Auxiliary solver results", "action": "review_minimal_retention_after_acceptance",
         "paths": [str(audit / "controls"), str(audit / "work-partition")],
         "reason": "Retain source-matched INPs, essential numerical outputs and receipts. Review bulky auxiliary ODB/log retention separately from original research ODBs; failed controls are evidence, not successful validation."},
        {"component": "Maintained code and guidance", "action": "consolidate_without_breaking_callers",
         "paths": [str(repo / "p1-DisorderLatticeProperties/code"), str(repo / "resources/lattices.py")],
         "reason": "Keep verified shared foundation fixes and regression checks. Fold truly superseded audit helpers/cells into the accepted workflow, checking p1/p2 consumers; do not blanket-delete all validation scripts."},
        {"component": "Named task temporary work", "action": "cleanup_candidate_after_unique_evidence_is_preserved",
         "paths": [str(path) for path in scratch_roots],
         "reason": "Includes verification checkouts, rendering intermediates and retired serial-worker scripts. Check absolute ownership and settled jobs before deletion; never sweep generic Temp."},
        {"component": "Temporary workbook extracts referenced by retained evidence", "action": "preserve_or_replace_links_before_temp_cleanup",
         "paths": [str(TEMP / "codex-p1-review-20261026" / name) for name in ("Deshpande-cells.txt", "FTval-cells.txt")],
         "reason": "benchmark-review.md still cites these extracts. Preserve unique before-state evidence or replace those links with verified retained workbook/CSV evidence before deleting the owning temporary directory."},
        {"component": "Pending reference capture", "action": "retain_until_library_management_finishes",
         "paths": [str(REFERENCE_INTAKE), str(audit / "reference-use.json")],
         "reason": "Private intake is pending library verification, not disposable until its owning workflow closes it."},
    ]
    result = {"generated_UTC": datetime.now(timezone.utc).isoformat(),
              "status": "cleanup_deferred_user_requested_end_of_work",
              "inventory": str(audit / "file-inventory.csv"), "canonical_workbook": workbook,
              "components": components, "automatic_deletion": False,
              "scope": "All validation audit outputs, new/modified code and notebook cells, workbook additions/legacy values, auxiliary solver evidence and named temporary work. PSC and orthotropic work remain deferred."}
    (audit / "cleanup-review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


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
        elif path.relative_to(audit).parts[0] == "paper":
            group, retention = "paper_output_or_caption", "retain_current_verified_export"
            reason = "Clean initiation-only publication layout; numerical eligibility and source ambiguity remain in its caption/data receipt"
        elif path.relative_to(audit).parts[0] == "final":
            group, retention = "calculation_source_or_receipt", "retain_current_verified_calculation"
            reason = "Single source for paper/diagnostic rendering; method and unavailable-area records remain explicit"
        elif path.name.startswith(("fig10-", "fig8-")) and path.suffix in (".csv", ".json", ".jpg"):
            group, retention = "literature_digitisation_or_provenance", "retain_source_values"
            reason = "Preserve source coordinates and original initial/final values independently of paper selection"
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
                   Path(repo) / "p2-DisorderML/samples/p1-validation/README.md",
                   Path(repo) / "README.md", Path(repo) / "p1-DisorderLatticeProperties/PROJECT_STATUS.md",
                   Path(repo) / "p1-DisorderLatticeProperties/AGENTS.md", code / "AGENTS.md",
                   Path(repo) / ".agents/skills/review-p1-p2-data-contract/references/data-contract.md"]
    for path in maintained:
        if path.exists():
            rows.append(record(path, "maintained_code_or_report", "retain", "Reproduction entry point or consolidated study report"))
    for name in ("README.md", "case-index.csv", "file-manifest.json", "consolidation-receipt.json", "hpc-selection.json"):
        path = Path(root) / name
        if path.exists():
            rows.append(record(path, "canonical_source_catalogue", "retain", "Authoritative retained-source identities, duplicate mapping and relocation provenance"))
    if EARLY_REVIEW.exists():
        rows.append(record(EARLY_REVIEW, "earlier_review", "consolidate_after_final_review",
                           "Earlier review can be removed only after unique findings/links are covered by the maintained study report"))
    if WORKBOOK.exists():
        rows.append(record(WORKBOOK, "canonical_workbook", "retain", "Paper source with additive audit/digitisation; preserve original values"))
    for reference in REFERENCES:
        if reference.exists():
            rows.append(record(reference, "published_source", "retain", "Source equations, failure criterion and specimen calibration"))
    scratch_roots = sorted({path for pattern in ("codex-p1-review-20261026", "codex-p1-final-check-20261028-*", "codex-p1-validation-*")
                            for path in Path(temp).glob(pattern) if path.is_dir()})
    cleanup_review(audit, Path(repo), scratch_roots)
    if REFERENCE_INTAKE.exists():
        rows.append(record(REFERENCE_INTAKE, "pending_reference_capture", "retain_until_library_management_finishes",
                           "Private cross-repository intake; no library write or cleanup is implied"))
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
