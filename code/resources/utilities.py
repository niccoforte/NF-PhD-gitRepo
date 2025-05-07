import os


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