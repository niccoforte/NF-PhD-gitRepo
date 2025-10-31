import os
import re
import sys


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

