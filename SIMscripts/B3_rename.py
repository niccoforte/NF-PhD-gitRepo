import os

root_dir = os.getcwd()
bump = 500

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
