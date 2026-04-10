import os
import re
import sys
import argparse
from typing import Tuple, List

run = "DeleteBackups"  # "BumpSimN"  # "Rename"  # "InpEdit"  # "NameConventionChange"  # 


def bump_simN(root_dir=os.getcwd(), bump=500):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            base, ext = os.path.splitext(fname)
            if ext.lower() not in ('.inp', '.odb', '.csv'):
                continue

            parts = base.split('-')
            number = int(parts[-1])

            new_number = number + bump
            parts[-1] = str(new_number)
            new_base = '-'.join(parts)
            new_fname = new_base + ext

            old_path = os.path.join(dirpath, fname)
            new_path = os.path.join(dirpath, new_fname)

            os.rename(old_path, new_path)

if run.lower() == "bumpsimn":
    ROOT_DIR = os.getcwd()  #"Z:\\p1\\data\\Ti\\20disNodes\\tri\\"
    BUMP     = 500

    bump_simN(
        root_dir=ROOT_DIR,
        bump=BUMP
    )


def rename(path, filetype, prefix=None, suffix=None, renumber=None):
    if prefix:
        for file in os.scandir(path):
            if file.name.endswith(filetype) and prefix not in file.name:
                os.rename(path+file.name, f'{path}{prefix}-{file.name}')
    if suffix:
        for file in os.scandir(path):
            if file.name.endswith(filetype):
                os.rename(path+file.name, f'{path}{file.name.split(".")[0]}.{suffix}')
    if renumber:
        for file in os.scandir(path):
            if file.name.endswith(filetype):
                rest, suf = file.name.split(".")[0], file.name.split(".")[-1]
                rest, num = rest.split("-")[:-1], int(rest.split("-")[-1])
                num = num + renumber
                name = "-".join(rest + num)
                os.rename(path+file.name, f'{path}{name}.{suf}')

if run.lower() == "rename":
    PATH      = os.getcwd() #"Z:\\p1\\data\\Ti\\20disNodes\\tri\\" 
    FILETYPE  = ".inp"
    PREFIX    = "IN-f"
    SUFFIX    = "csv"
    RENUMBER  = 500

    rename(
        path=PATH,
        filetype=FILETYPE,
        prefix=PREFIX,
        suffix=SUFFIX,
        renumber=RENUMBER
    )


def rename_NameConventionChange(root_dir, prefix, mechanical_model, lattice_type, size, disorder_type, disorder_magnitude, distribution, disorder_nodes, dry_run=False):
    pattern = re.compile(
        rf'^(?P<prefix>{prefix})'
        rf'{re.escape(mechanical_model)}-'
        rf'{re.escape(lattice_type)}-'
        rf'{re.escape(size)}-'
        rf'{re.escape(disorder_type)}-'
        r'(?P<simnum>\d+)\.(?P<ext>.+)$'
    )

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            m = pattern.match(fname)
            if not m:
                continue

            prefix_  = m.group('prefix')
            sim_num = m.group('simnum')
            ext     = m.group('ext')
            new_fname = (
                f"{prefix_}{mechanical_model}-{lattice_type}-{size}-"
                f"{disorder_magnitude}{disorder_type}-"
                f"{distribution}-{disorder_nodes}-"
                f"{sim_num}.{ext}"
            )

            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, new_fname)

            if os.path.exists(dst):
                print(f"[SKIP] target exists: {dst}")
                continue

            print(f"[RENAME] {src} → {dst}")
            if not dry_run:
                os.rename(src, dst)

if run.lower() == "nameconventionchange":
    ROOT_DIR         = os.getcwd() #"Z:\\p1\\data\\Ti\\20disNodes\\tri\\" 
    PREFIX           = r'(?:IN-f|IN-n|OUT-)?'
    MECHANICAL_MODEL = "Ductile"
    LATTICE_TYPE     = "kagome"
    SIZE             = "20"
    DISORDER_TYPE    = "disNodes"

    DISORDER_MAGNITUDE = "20"
    DISTRIBUTION       = "lhs" 
    DISORDER_NODES     = "all"

    DRY_RUN = False

    # Execute
    rename_NameConventionChange(
        root_dir=ROOT_DIR,
        prefix=PREFIX,
        mechanical_model=MECHANICAL_MODEL,
        lattice_type=LATTICE_TYPE,
        size=SIZE,
        disorder_type=DISORDER_TYPE,
        disorder_magnitude=DISORDER_MAGNITUDE,
        distribution=DISTRIBUTION,
        disorder_nodes=DISORDER_NODES,
        dry_run=DRY_RUN
    )


# ---------- Patterns ----------
BEAM_SECTION_PATTERN = re.compile(r'\*Beam\s+Section\b.*section\s*=\s*RECT', re.IGNORECASE)
THICKNESS = "10."

ELASTIC_PATTERN = re.compile(r'\*Elastic\b', re.IGNORECASE)
ELASTIC_NEW_LINE = "135165., 0.428571"

# ---------- Helpers ----------
def detect_line_ending(s: str) -> str:
    if s.endswith('\r\n'):
        return '\r\n'
    if s.endswith('\n'):
        return '\n'
    if s.endswith('\r'):
        return '\r'
    return '\n'

def next_data_line_index(lines: List[str], start_idx: int) -> int:
    """
    Return index of the first non-empty line after start_idx that does NOT start with '*'.
    If none found, returns len(lines).
    """
    i = start_idx + 1
    n = len(lines)
    while i < n and (lines[i].strip() == "" or lines[i].lstrip().startswith('*')):
        i += 1
    return i

def update_beam_sections(lines: List[str], forced_thickness: str) -> int:
    """
    For each *Beam Section ... section=RECT, set the first value on the next data line to forced_thickness.
    Returns number of sections updated.
    """
    updates = 0
    i = 0
    n = len(lines)
    while i < n:
        if BEAM_SECTION_PATTERN.search(lines[i]):
            j = next_data_line_index(lines, i)
            if j < n:
                dim_line = lines[j]
                le = detect_line_ending(dim_line)
                dim_text = dim_line.rstrip('\r\n')

                # Split by comma, keep only non-empty trimmed tokens
                parts = [p.strip() for p in dim_text.split(',') if p.strip() != '']

                if len(parts) == 0:
                    # Weird/empty data line; just write thickness
                    new_line = f"{forced_thickness}{le}"
                else:
                    # Force the FIRST value (out-of-plane thickness)
                    parts[0] = forced_thickness
                    new_line = ", ".join(parts) + le

                if new_line != dim_line:
                    lines[j] = new_line
                    updates += 1
        i += 1
    return updates

def update_elastic(lines: List[str]) -> int:
    """
    For each *Elastic, set the next data line to the specified elastic pair.
    Returns number of elastic updates.
    """
    updates = 0
    i = 0
    n = len(lines)
    while i < n:
        if ELASTIC_PATTERN.search(lines[i]):
            j = next_data_line_index(lines, i)
            if j < n:
                le = detect_line_ending(lines[j])
                new_line = ELASTIC_NEW_LINE + le
                if lines[j] != new_line:
                    lines[j] = new_line
                    updates += 1
        i += 1
    return updates

def process_file(path: str, forced_thickness: str, dry_run: bool = False) -> Tuple[bool, int, int]:
    """
    Returns (changed, beam_updates, elastic_updates)
    """
    with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
        original_lines = f.read().splitlines(keepends=True)

    lines = original_lines[:]

    b_updates = update_beam_sections(lines, forced_thickness)
    e_updates = update_elastic(lines)

    changed = (b_updates > 0) or (e_updates > 0)

    if changed and not dry_run:
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8', errors='replace', newline='') as b:
                b.writelines(original_lines)
        with open(path, 'w', encoding='utf-8', errors='strict', newline='') as f:
            f.writelines(lines)

    return changed, b_updates, e_updates

def matches_name(filename: str) -> bool:
    return filename.startswith("Ductile") and filename.endswith(".inp")

if run.lower() == "inpedit":
    parser = argparse.ArgumentParser(
        description="Update RECT *Beam Section out-of-plane thickness (first value) and *Elastic properties in Abaqus .inp files."
    )
    parser.add_argument("root", nargs="?", default=os.getcwd(), help="Root directory to walk (default: current working directory)")
    parser.add_argument("--thickness", default=THICKNESS, help="Out-of-plane thickness to set (default: 10.)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without modifying files")
    args = parser.parse_args()

    total_files = 0
    changed_files = 0
    total_beam_updates = 0
    total_elastic_updates = 0

    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if matches_name(fn):
                fullpath = os.path.join(dirpath, fn)
                total_files += 1
                try:
                    changed, b_updates, e_updates = process_file(fullpath, args.thickness, dry_run=args.dry_run)
                    if changed:
                        changed_files += 1
                        total_beam_updates += b_updates
                        total_elastic_updates += e_updates
                        action = "(dry-run) would modify" if args.dry_run else "modified"
                        print(f"{action}: {fullpath} — beam sections: {b_updates}, elastic blocks: {e_updates}")
                except Exception as e:
                    print(f"ERROR processing {fullpath}: {e}")

    print("\nSummary")
    print("-------")
    print(f"Matched files:             {total_files}")
    print(f"Files changed:             {changed_files}")
    print(f"Beam sections updated:     {total_beam_updates}")
    print(f"Elastic props updated:     {total_elastic_updates}")
    if args.dry_run:
        print("No files were modified (dry-run).")


def delete_backups(root: str, dry_run: bool = False) -> int:
    deleted = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".bak"):
                fullpath = os.path.join(dirpath, fn)
                if dry_run:
                    print(f"(dry-run) would delete: {fullpath}")
                else:
                    try:
                        os.remove(fullpath)
                        print(f"deleted: {fullpath}")
                        deleted += 1
                    except Exception as e:
                        print(f"ERROR deleting {fullpath}: {e}")
    return deleted

if run.lower() == "deletebackups":
    parser = argparse.ArgumentParser(
        description="Delete all .bak backup files created by previous modification scripts."
    )
    parser.add_argument(
        "root",
        nargs="?",                         # optional
        default=os.getcwd(),               # walk from current working directory by default
        help="Root directory to start search (default: current working directory)"
    )
    parser.add_argument("--dry-run", action="store_true", help="List .bak files without deleting them")
    args = parser.parse_args()

    print(f"Scanning from: {args.root}")
    deleted = delete_backups(args.root, dry_run=args.dry_run)

    print("\nSummary")
    print("-------")
    print(f"Total .bak files {'found' if args.dry_run else 'deleted'}: {deleted}")
    if args.dry_run:
        print("No files were deleted (dry-run).")

