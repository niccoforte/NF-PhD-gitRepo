"""Run an explicit local validation manifest in series, retaining verified results.

One validation-wide lock prevents separate queues from overlapping. A second
lock protects each queue's state. No geometry is generated and no original INP
is edited. Solver scratch is confined to a marker-owned directory below the
manifest's queue root; useful outputs are checksum-verified before its removal.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time


VALIDATION_ROOT = Path("Z:/p1/sims/Ti/Validation")
RETAIN_SUFFIXES = {".odb", ".inp", ".dat", ".msg", ".sta", ".log", ".prt"}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # Windows scanners/readers can briefly hold a file being atomically replaced.
    # Keep the previous complete state and fail visibly if that lock persists.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.025 * 2**attempt)


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    path.relative_to(root)
    if path == root:
        raise ValueError("Expected a child path, not the root itself")
    return path


def owned_subroot(root, name):
    candidate = Path(root) / name
    if candidate.exists() and (candidate.is_symlink() or
            getattr(candidate.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError("Linked queue subroot is not owned scratch: " + str(candidate))
    return within(candidate, root)


@contextmanager
def worker_lock(root, name="worker.lock"):
    """An OS-held byte lock releases on process exit, including abnormal exit."""
    import msvcrt

    with (root / name).open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another worker owns this queue") from error
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def validate_manifest(manifest_path, allowed_root=VALIDATION_ROOT):
    manifest_path = within(manifest_path, allowed_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = within(manifest["queue_root"], allowed_root)
    owned_subroot(root, "scratch")
    owned_subroot(root, "runs")
    if not manifest.get("jobs"):
        raise ValueError("No explicitly listed jobs")
    names = set()
    for job in manifest["jobs"]:
        name = job["job"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,69}", name) or name in names:
            raise ValueError("Unsafe or duplicate Abaqus job name: " + name)
        names.add(name)
        source = within(job["input"], allowed_root)
        if source.suffix.lower() != ".inp" or sha256(source) != job["input_sha256"]:
            raise ValueError("Missing/changed source input: " + str(source))
        cpus = job.get("cpus", 8)
        if type(cpus) is not int or not 1 <= cpus <= 12:
            raise ValueError("Local queue CPU allocation must be explicit and bounded")
        required = job.get("required_outputs", [".odb"])
        if not required or any(suffix not in RETAIN_SUFFIXES for suffix in required):
            raise ValueError("Required outputs must be retained artifact suffixes")
    return manifest, root


def execute_abaqus(abaqus, job, work, on_started):
    scratch = work / "solver-scratch"
    scratch.mkdir()
    command = [abaqus, "job=" + job["job"], "input=" + job["job"] + ".inp",
               "cpus=" + str(job.get("cpus", 8)), "memory=25%", "interactive",
               "scratch=" + str(scratch)]
    for option in job.get("execution_options", []):
        if option not in ("double=both", "double=explicit", "output_precision=full"):
            raise ValueError("Unreviewed execution option: " + option)
        command.append(option)
    with (work / "launcher.txt").open("w", encoding="utf-8") as launcher:
        process = subprocess.Popen(command, cwd=str(work), stdout=launcher, stderr=subprocess.STDOUT)
        try:
            on_started(process.pid, command)
        finally:
            # Retain the serial lock until this child exits even if recording
            # its PID fails. A callback error must not orphan a live solver.
            code = process.wait()
        return code


def deliver_and_clean(work, destination, job):
    """Only our exact marker-owned scratch may be removed after verified delivery."""
    queue_root = destination.parent.parent.resolve()
    work = within(work, owned_subroot(queue_root, "scratch"))
    destination = within(destination, owned_subroot(queue_root, "runs"))
    marker = json.loads((work / "scratch-owner.json").read_text(encoding="utf-8"))
    if marker != {"job": job["job"], "input_sha256": job["input_sha256"]}:
        raise ValueError("Scratch ownership mismatch; retained without cleanup")
    if any(path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) &
           stat.FILE_ATTRIBUTE_REPARSE_POINT for path in work.rglob("*")):
        raise ValueError("Linked scratch content needs explicit review")
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sorted(work.iterdir()):
        if not source.is_file() or (source.suffix.lower() not in RETAIN_SUFFIXES
                                    and source.name not in ("launcher.txt", "scratch-owner.json")):
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        before, after = sha256(source), sha256(target)
        if before != after:
            raise ValueError("Delivery hash mismatch; original scratch retained")
        records.append({"file": source.name, "bytes": source.stat().st_size, "sha256": after})
    copied = {record["file"] for record in records if record["bytes"] > 0}
    missing = [suffix for suffix in job.get("required_outputs", [".odb"])
               if job["job"] + suffix not in copied]
    receipt = {"job": job["job"], "files": records, "required_outputs_missing": missing,
               "delivered_at": timestamp(), "scratch_cleanup": "not yet attempted"}
    write_json(destination / "delivery.json", receipt)
    # Solver failure may still yield a scientifically useful finite prefix. Its
    # copied files are retained, but missing required artifacts block cleanup.
    if missing:
        return receipt
    shutil.rmtree(work)
    receipt["scratch_cleanup"] = "verified marker-owned directory removed"
    write_json(destination / "delivery.json", receipt)
    return receipt


def run_queue(manifest_path, execute=execute_abaqus, allowed_root=VALIDATION_ROOT):
    manifest, root = validate_manifest(manifest_path, allowed_root)
    root.mkdir(parents=True, exist_ok=True)
    global_root = owned_subroot(Path(allowed_root).resolve(), "audit")
    global_root.mkdir(exist_ok=True)
    owner_path = global_root / "serial-owner.json"
    state_path = root / "state.json"
    manifest_hash = sha256(manifest_path)
    with worker_lock(global_root, "serial-worker.lock"), worker_lock(root):
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["manifest_sha256"] != manifest_hash:
                raise ValueError("Queue manifest changed since its first execution")
            if state["phase"] == "completed":
                return state
            if any(row["status"] in ("running", "failed", "interrupted") for row in state["jobs"]):
                raise ValueError("Interrupted/failed queue requires explicit review; no implicit rerun")
        else:
            state = {"manifest": str(Path(manifest_path).resolve()), "manifest_sha256": manifest_hash,
                     "phase": "prepared", "created_at": timestamp(), "jobs": [
                         {"job": job["job"], "kind": job.get("kind"), "purpose": job.get("purpose"),
                          "cpus": int(job.get("cpus", 8)), "status": "queued"} for job in manifest["jobs"]]}
        if owner_path.exists() and json.loads(owner_path.read_text()).get("phase") != "settled":
            raise RuntimeError("A previous validation worker did not settle; review its solver processes and owner record before another queue")
        owner = {"phase": "active", "queue_root": str(root), "worker_pid": os.getpid(),
                 "manifest_sha256": manifest_hash, "started_at": timestamp()}
        # This durable gate precedes every Popen. It survives worker death,
        # unlike an OS lock, and requires explicit review after a crash.
        write_json(owner_path, owner)
        state.update(phase="running", worker_pid=os.getpid(), updated_at=timestamp())
        state["runner_sha256"] = sha256(__file__)
        write_json(state_path, state)
        scratch_root = owned_subroot(root, "scratch")
        runs_root = owned_subroot(root, "runs")
        scratch_root.mkdir(exist_ok=True)
        runs_root.mkdir(exist_ok=True)
        for job, row in zip(manifest["jobs"], state["jobs"]):
            if row["status"] == "completed":
                continue
            work = within(scratch_root / job["job"], scratch_root)
            if work.exists():
                raise ValueError("Existing scratch is retained for review: " + str(work))
            if sha256(job["input"]) != job["input_sha256"]:
                raise ValueError("Input changed before its turn in the queue")
            work.mkdir()
            write_json(work / "scratch-owner.json", {"job": job["job"], "input_sha256": job["input_sha256"]})
            staged_input = work / (job["job"] + ".inp")
            shutil.copy2(job["input"], staged_input)
            if sha256(staged_input) != job["input_sha256"]:
                raise ValueError("Staged input differs from the reviewed source")
            row.update(status="running", started_at=timestamp(), scratch=str(work))
            state["current_job"] = job["job"]
            write_json(state_path, state)
            def started(pid, command):
                row.update(solver_launcher_pid=pid, command=command)
                write_json(state_path, state)
            try:
                code = execute(manifest["abaqus"], job, work, started)
                receipt = deliver_and_clean(work, root / "runs" / job["job"], job)
                launcher = (root / "runs" / job["job"] / "launcher.txt").read_text(encoding="utf-8", errors="replace")
                completed = re.search(r"\bAbaqus JOB " + re.escape(job["job"]) + r" COMPLETED\b",
                                      launcher, re.IGNORECASE)
                success = code == 0 and bool(completed) and not receipt["required_outputs_missing"]
                row.update(status="completed" if success else "failed", solver_returncode=code,
                           finished_at=timestamp(), result=str(root / "runs" / job["job"]))
            except Exception as error:
                row.update(status="failed", finished_at=timestamp(), error=str(error))
                success = False
            state["updated_at"] = timestamp()
            write_json(state_path, state)
            if not success and not job.get("allow_solver_failure", False):
                state.update(phase="stopped_after_failure", current_job=None)
                write_json(state_path, state)
                write_json(owner_path, dict(owner, phase="settled", finished_at=timestamp()))
                return state
        state.update(phase="completed", current_job=None, finished_at=timestamp())
        write_json(state_path, state)
        write_json(owner_path, dict(owner, phase="settled", finished_at=timestamp()))
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "status"))
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.action == "run":
        result = run_queue(args.manifest)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["phase"] == "completed" else 1)
    manifest, root = validate_manifest(args.manifest)
    if args.action == "status" and (root / "state.json").exists():
        print((root / "state.json").read_text(encoding="utf-8"))
    else:
        print(json.dumps({"phase": "prepared", "queue_root": str(root),
                          "jobs": [{key: job.get(key) for key in ("job", "kind", "cpus", "purpose")}
                                   for job in manifest["jobs"]]}, indent=2))
