#!/usr/bin/env python3
"""
Transfer archived ML runs from the QMUL HPC to the local machine.

The matching B1_ML-new.sh archive layout is:
    /data/SEMS-TaoLab/Niccolo-Forte/p2/<path_extra>/<run_label>/<job_id>/zip/

Examples:
    python B3_ML.py baseline 123456 --path-extra UT/MLP
    python B3_ML.py baseline all --path-extra UT/MLP
    python B3_ML.py baseline 123456 --path-extra UT/MLP --models-only
    python B3_ML.py baseline 123456 --path-extra UT/MLP --local-root Z:/p2
"""

import argparse
import shlex
import subprocess
import tarfile
from pathlib import Path


DEFAULT_REMOTE = "exy053@login.hpc.qmul.ac.uk"
DEFAULT_REMOTE_ROOT = "/data/SEMS-TaoLab/Niccolo-Forte/p2"


def default_local_root() -> Path:
    z_drive = Path("Z:/")
    if z_drive.exists():
        return Path("Z:/p2")
    return Path.cwd() / "p2-MLruns"


def prompt(value, label, default=None):
    if value:
        return value
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{label}{suffix}: ").strip()
    return answer or default


def run_command(command, dry_run=False, capture=False):
    print("+ " + " ".join(str(part) for part in command))
    if dry_run:
        return ""
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout if capture else ""


def remote_run_base(remote_root, path_extra, run_label):
    parts = [remote_root.rstrip("/")]
    if path_extra:
        parts.append(path_extra.strip("/"))
    parts.append(run_label)
    return "/".join(parts)


def local_run_base(local_root, path_extra, run_label):
    base = Path(local_root)
    if path_extra:
        base = base / Path(path_extra)
    return base / run_label


def list_remote_jobs(remote, remote_root, path_extra, run_label, dry_run=False):
    remote_base = remote_run_base(remote_root, path_extra, run_label)
    remote_command = (
        f"for d in {shlex.quote(remote_base)}/*; do "
        f'[ -d "$d" ] && basename "$d"; '
        f"done"
    )
    if dry_run:
        run_command(["ssh", remote, remote_command], dry_run=True)
        return []
    output = run_command(["ssh", remote, remote_command], capture=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def extract_tgz_files(directory):
    for archive in directory.rglob("*.tgz"):
        target = archive.parent
        print(f"Extracting {archive} -> {target}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=target)


def transfer_job(args, job_id):
    remote_zip = f"{remote_run_base(args.remote_root, args.path_extra, args.run_label)}/{job_id}/zip"
    remote_source = f"{remote_zip}/models" if args.models_only else remote_zip
    local_job_dir = local_run_base(args.local_root, args.path_extra, args.run_label) / job_id
    if not args.dry_run:
        local_job_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "scp",
            "-r",
            f"{args.remote}:{remote_source}",
            str(local_job_dir),
        ],
        dry_run=args.dry_run,
    )

    if args.extract_tgz and not args.dry_run:
        extract_tgz_files(local_job_dir)

    verb = "Would save to" if args.dry_run else "Saved to"
    print(f"{verb}: {local_job_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer archived ML run outputs from QMUL HPC."
    )
    parser.add_argument("run_label", nargs="?", help="Run label used by B1_ML-new.sh.")
    parser.add_argument(
        "job_id",
        nargs="?",
        help="Slurm job ID to transfer, or 'all' to transfer every job under the label.",
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--local-root", type=Path, default=default_local_root())
    parser.add_argument(
        "--path-extra",
        default="",
        help="Archive subdirectory such as UT/MLP, FT/GNN, or MULTI/TR.",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Transfer only zip/models instead of the full zip directory.",
    )
    parser.add_argument(
        "--extract-tgz",
        action="store_true",
        help="Extract any .tgz archives after transfer.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_label = prompt(args.run_label, "Run label")
    args.job_id = prompt(args.job_id, "Job ID, or all", default="all")
    args.local_root = Path(args.local_root)

    if args.job_id.lower() == "all":
        job_ids = list_remote_jobs(
            args.remote,
            args.remote_root,
            args.path_extra,
            args.run_label,
            dry_run=args.dry_run,
        )
        if not job_ids and not args.dry_run:
            raise SystemExit(f"No remote jobs found for run label: {args.run_label}")
    else:
        job_ids = [args.job_id]

    for job_id in job_ids:
        transfer_job(args, job_id)


if __name__ == "__main__":
    main()
