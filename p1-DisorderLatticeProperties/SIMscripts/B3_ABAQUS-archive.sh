#!/usr/bin/env bash
# Windows/Git Bash: download the complete p1 archive through the shared PuTTY session.
# No upload, extraction, remote deletion, Slurm submission or archive flattening.
set -euo pipefail

exec python -u - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

SOURCE = "/data/SEMS-TaoLab/Niccolo-Forte/p1"


def putty_client(name):
    path = shutil.which(name) or str(Path("C:/Program Files/PuTTY") / (name + ".exe"))
    if not Path(path).is_file():
        raise RuntimeError(f"PuTTY client not found: {name}")
    return path


def inventory(connection):
    # NUL-delimited records preserve spaces and dotfiles. This only reads metadata.
    command = "cd " + shlex.quote(SOURCE) + " && find . -printf '%y\\t%s\\t%T@\\t%P\\0'"
    raw = subprocess.check_output(connection + [command])
    entries = {}
    folded = set()
    for record in raw.decode("utf-8").split("\0"):
        if not record:
            continue
        kind, size, modified, name = record.split("\t", 3)
        if not name:
            continue
        parts = PurePosixPath(name).parts
        if (kind not in ("d", "f") or name.startswith("/") or ".." in parts
                or name.casefold() in folded or any(
                    re.search(r'[<>:"\\|?*\x00-\x1f]', part)
                    or part.endswith((".", " "))
                    or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]",
                                    part.split(".")[0], re.IGNORECASE)
                    for part in parts)):
            raise RuntimeError(f"Cannot mirror this entry exactly on Windows/exFAT: {name!r} ({kind})")
        entries[name] = [kind, int(size), modified]
        folded.add(name.casefold())
    if not entries:
        raise RuntimeError("Remote archive inventory is empty; refusing to report success.")
    return entries


def local_plan(destination, entries):
    pending = []
    remaining = 0
    if destination.exists():
        for parent, dirs, files in os.walk(destination):
            for name in dirs + files:
                path = Path(parent) / name
                relative = path.relative_to(destination).as_posix()
                if path.is_symlink() or relative not in entries:
                    raise RuntimeError(f"Unexpected local entry; nothing will be deleted: {path}")
    for relative, (kind, size, _) in entries.items():
        path = destination / relative
        if len(str(path)) >= 260:
            raise RuntimeError(f"Path exceeds the conservative Windows client limit: {path}")
        if kind == "d":
            if path.exists() and not path.is_dir():
                raise RuntimeError(f"Expected a directory: {path}")
            continue
        if path.exists() and not path.is_file():
            raise RuntimeError(f"Expected a regular file: {path}")
        actual = path.stat().st_size if path.exists() else 0
        if actual > size:
            raise RuntimeError(f"Local file is larger than its source; refusing to resume: {path}")
        if not path.exists() or actual != size:
            pending.append(relative)
            remaining += size - actual
    return pending, remaining


def download(command, files, remaining):
    """Run one SFTP batch and report actual local byte progress every five seconds."""
    process = subprocess.Popen(command)
    finished = completed = 0
    started = time.monotonic()
    try:
        while True:
            returncode = process.poll()
            current = 0
            while finished < len(files):
                path, size, offset = files[finished]
                actual = path.stat().st_size if path.is_file() else None
                if actual is None or actual < size:
                    current = max(0, (actual or 0) - offset)
                    break
                completed += size - offset
                finished += 1
            received = completed + current
            percent = 100 * received / remaining if remaining else 100 * finished / len(files)
            rate = received / max(time.monotonic() - started, 0.001) / 1e6
            print(f"Progress: {percent:.2f}% of remaining bytes; "
                  f"{received / 1e9:.3f}/{remaining / 1e9:.3f} GB; "
                  f"{finished:,}/{len(files):,} files complete; {rate:.2f} MB/s average.", flush=True)
            if finished < len(files):
                path, size, offset = files[finished]
                print(f"Current file: {path} ({(offset + current) / 1e6:.2f}/{size / 1e6:.2f} MB)",
                      flush=True)
            if returncode is not None:
                if returncode:
                    raise subprocess.CalledProcessError(returncode, command)
                return
            time.sleep(5)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Copy the entire HPC p1 tree into Z:/p1/HPCarchive, without an extra p1 folder. "
        "Requires Git Bash, Python and an authenticated shared PuTTY HPC session. "
        "Keep the source unchanged throughout the download and any resume. "
        "Verification checks paths/types/sizes, NOT content hashes or Unix permissions."))
    parser.add_argument("--destination", type=Path, default=Path("Z:/p1/HPCarchive"))
    parser.add_argument("--session", default="HPC", help="Authenticated PuTTY sharing profile")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print the plan; no connections or writes")
    action.add_argument("--check-only", action="store_true", help="Read remote/local metadata; do not download")
    args = parser.parse_args(argv)
    destination = args.destination.resolve()
    if destination == Path(destination.anchor) or any(
            (parent / ".git").exists() for parent in (destination, *destination.parents)):
        raise RuntimeError("Destination must be a dedicated directory outside every Git checkout.")
    print(f"Source:      {SOURCE}/", flush=True)
    print(f"Destination: {destination}", flush=True)
    print("Download only; all files/subdirectories, including hidden entries; no extraction.", flush=True)
    if args.dry_run:
        print("Would inventory, validate names/space, resume individual files, then check paths and sizes.")
        return

    plink = putty_client("plink")
    psftp = putty_client("psftp")
    print("Checking the shared HPC session, then reading the remote file inventory (may take minutes)...",
          flush=True)
    subprocess.run([plink, "-shareexists", "-load", args.session, "-l", "exy053"], check=True)
    connection = [plink, "-batch", "-share", "-load", args.session, "-l", "exy053"]
    entries = inventory(connection)
    print("Remote inventory received; checking existing local files...", flush=True)
    pending, remaining = local_plan(destination, entries)
    count = sum(value[0] == "f" for value in entries.values())
    total = sum(value[1] for value in entries.values() if value[0] == "f")
    print(f"Inventory: {count:,} files, {total / 1e9:.2f} GB; {len(pending):,} files need transfer; "
          f"{remaining / 1e9:.2f} GB remaining.", flush=True)
    if args.check_only:
        missing_dirs = [name for name, value in entries.items()
                        if value[0] == "d" and not (destination / name).is_dir()]
        if pending or missing_dirs:
            raise RuntimeError(f"Incomplete copy: {len(pending)} files and {len(missing_dirs)} directories.")
        print("Paths/types/sizes match. This is not checksum verification or permission to delete HPC data.")
        return

    # Only this small resume marker is persistent while a transfer is incomplete.
    # It is outside the mirrored tree and removed after successful verification.
    marker = destination.with_name(destination.name + ".resume.json")
    state = dict(source=SOURCE, destination=str(destination),
                 inventory_sha256=hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest())
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != state:
            raise RuntimeError("Source inventory changed since the interrupted transfer. Do not resume old prefixes.")
    elif destination.exists() and any(destination.iterdir()):
        raise RuntimeError("Nonempty destination has no matching resume marker; use --check-only or a fresh directory.")
    ancestor = next(parent for parent in (destination, *destination.parents) if parent.exists())
    if shutil.disk_usage(ancestor).free < remaining + 1024 ** 3:
        raise RuntimeError("Insufficient free space: need the remaining bytes plus 1 GiB safety margin.")
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("Preparing the local directory tree...", flush=True)
    for name, value in entries.items():
        if value[0] == "d":
            (destination / name).mkdir(parents=True, exist_ok=True)
    if pending:
        with tempfile.TemporaryDirectory(prefix="p1-archive-") as temporary:
            batch = Path(temporary) / "download.txt"
            # Per-file reget avoids recursive reget skipping a completed subtree.
            commands = []
            files = []
            for name in sorted(pending):
                remote = SOURCE + "/" + name
                remote = "".join("\\" + c if c in "[]*?\\" else c for c in remote)
                local = destination / name
                offset = local.stat().st_size if local.exists() else 0
                # PSFTP reget requires an existing local file; get starts missing/empty files.
                operation = "reget" if offset else "get"
                commands.append(f'{operation} "{remote}" "{local.as_posix()}"')
                files.append((local, entries[name][1], offset))
            batch.write_text("\n".join(commands) + "\nquit\n", encoding="utf-8")
            download([psftp, "-batch", "-share", "-load", args.session,
                      "-l", "exy053", "-b", str(batch)], files, remaining)
    print("Transfer finished; verifying the unchanged remote inventory and local completeness...", flush=True)
    if inventory(connection) != entries:
        raise RuntimeError("The remote archive changed during transfer. Keep HPC originals and investigate before resuming.")
    if local_plan(destination, entries)[0] or any(
            value[0] == "d" and not (destination / name).is_dir() for name, value in entries.items()):
        raise RuntimeError("Some local files are absent or incomplete; rerun with the same unchanged source.")
    marker.unlink()
    print("Download complete: relative paths and file sizes verified; temporary state removed.", flush=True)
    print("No HPC files were deleted. Verify file contents separately before any irreversible remote cleanup.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTransfer interrupted; partial files and the resume marker are retained. Rerun to resume.",
              file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
PY
