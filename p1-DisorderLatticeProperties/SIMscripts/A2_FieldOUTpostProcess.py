from __future__ import print_function

import os
import sys
from resources.abaqus import export_UfieldNPZ
from resources.lattices import Geometry


mode = "any"                             # "ductile", "fracture", "both", "any"
unitCellSize = 10.0
ffilter = None
pDir = r"C:\temp"
dtype = "float32"

cmdIN = sys.argv[8:]
if len(cmdIN) > 0:
    latticeType = str(cmdIN[0])
    nnx = int(cmdIN[1])
    unitCellSize = float(cmdIN[2])
    mode = str(cmdIN[3])
    userMaterial = str(cmdIN[4])
    relDensity = float(cmdIN[5])
    dis = str(cmdIN[6])
    fac = float(cmdIN[7])
    distribution = str(cmdIN[8])
    targeted_disorder = str(cmdIN[9])
    initialJob = int(cmdIN[10])
    numberOfRuns = int(cmdIN[11])
    cpus = int(cmdIN[12])
    FieldOut_frames = int(cmdIN[13])
    HistOut_frames = int(cmdIN[14])
    pDir = str(cmdIN[15])

if pDir in [None, "", "None", "none"]:
    pDir = os.getcwd()

def _mode_allowed(stem, selected_mode):
    low = stem.lower()
    selected_mode = str(selected_mode).lower()
    if selected_mode in ["any", "both"]:
        return low.startswith("ductile-") or low.startswith("fracture-")
    if selected_mode in ["ductile", "ut"]:
        return low.startswith("ductile-")
    if selected_mode in ["fracture", "ft"]:
        return low.startswith("fracture-")
    return False

def _job_mode(stem):
    if stem.lower().startswith("fracture-"):
        return "fracture"
    return "ductile"

def _job_geometry(stem, unit_cell_size):
    parts = stem.split("-")
    latticeType = parts[1]
    nnx = int(parts[2])
    geom = Geometry(latticeType, unit_cell_size, nnx)
    geom.nodeCount(mode=_job_mode(stem))
    return geom


def main():
    pDir_abs = os.path.abspath(pDir)
    os.chdir(pDir_abs)

    saved = []
    for curDirectory, folders, files in os.walk(pDir_abs):
        folders[:] = [folder for folder in folders if folder not in ["transfer", "__pycache__"]]
        odbs = [file for file in files if file.endswith(".odb")]
        if ffilter is not None:
            odbs = [file for file in odbs if ffilter in file]

        for odb in sorted(odbs):
            stem = odb[:-4]
            if not _mode_allowed(stem, mode):
                continue

            inpFile = os.path.join(curDirectory, stem + ".inp")
            if not os.path.exists(inpFile):
                print("Skipping {}, matching .inp not found.".format(odb))
                continue

            transferDir = os.path.join(curDirectory, "transfer")
            if not os.path.exists(transferDir):
                os.makedirs(transferDir)

            geom = _job_geometry(stem, unitCellSize)
            odbPath = os.path.join(curDirectory, odb)
            expFile = os.path.join(transferDir, "FIELDu-" + stem + ".npz")

            out = export_UfieldNPZ(
                odbPath,
                inpFile,
                expFile,
                totalNodes=geom.totalNodes,
                mode=_job_mode(stem),
                dtype=dtype,
            )
            saved.append(out["path"])
            print("Saved {} | Y shape {} | valid {:.3f}".format(out["path"], out["shape"], out["valid_fraction"]))

    if len(saved) == 0:
        raise RuntimeError("No FIELDu npz files were produced in {}.".format(pDir_abs))

    print("Done. Saved {} FIELDu file(s).".format(len(saved)))


if __name__ == "__main__":
    main()
