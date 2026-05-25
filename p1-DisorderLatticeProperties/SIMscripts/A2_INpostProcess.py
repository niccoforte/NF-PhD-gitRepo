import os
import numpy as np
import sys
from resources.abaqus import export_frequencies, export_nodes, export_struts

stiffMatrix = False
distribution = "lhs_uniform"
unitCellSize = 10.0

pDir = "C:\\temp"

if "--" in sys.argv:
    cmdIN = sys.argv[sys.argv.index("--") + 1:]
elif len(sys.argv) >= 26:
    cmdIN = sys.argv[10:]
else:
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
    beta = fac
    distribution = str(cmdIN[8])
    targeted_disorder = str(cmdIN[9])
    initialJob = int(cmdIN[10])
    numberOfRuns = int(cmdIN[11])
    cpus = int(cmdIN[12])
    FieldOut_frames = int(cmdIN[13])
    HistOut_frames = int(cmdIN[14])
    pDir = str(cmdIN[15])

    if "OptLoop" in cmdIN:
        sampleN = int(cmdIN[-1])
        opt_disorder = np.loadtxt(pDir+f"\\BO_sample{sampleN}.txt", delimiter=" ")
        if distribution.lower() == "opt-f":
            frequencies = opt_disorder
        else:
            opt_disorder = opt_disorder.reshape((len(opt_disorder)//2,2))
            opt_dis_x = opt_disorder[:,0]
            opt_dis_y = opt_disorder[:,1]
    
    stiffMatrix = False
    UTval = False
    
    finalRun = 'yes'
    
    if dis.lower() == 'per':
        nodeVar = 'no'
        sizeVar = 'no'
    elif dis.lower() == 'disnodes':
        nodeVar = 'yes'
        sizeVar = 'no'
    elif dis.lower() == 'disstruts':
        nodeVar = 'no'
        sizeVar = 'yes'
    else:
        raise Exception("Invalid disorder input.")

if stiffMatrix:
    pDir = "Z:\\p1\\sims\\Ti\\StiffMatrix"

if pDir in [None, "", "None", "none"]:
    pDir = os.getcwd()

os.chdir(pDir)


freq = False
if distribution.lower() == "frequency" or distribution.lower() == "opt-f":
    freq = True

for curDirectory, folders, files in os.walk(pDir):
    folders[:] = [folder for folder in folders if folder not in ["transfer", "__pycache__"]]
    if not os.path.exists(curDirectory + "/transfer"):
        os.makedirs(curDirectory + "/transfer")

    for file in files:
        if 'per' in file or 'disNodes' in file:
            if file.endswith('.inp'):
                inpFile = os.path.join(curDirectory, file)
                expFile_n = curDirectory + "/transfer/IN-n" + file[:-4].replace('_','-') + ".csv"
                expFile_f = curDirectory + "/transfer/IN-f" + file[:-4].replace('_','-') + ".csv"
                export_nodes(
                    inpFile,
                    expFile_n,
                    latticeType=file.split('-')[1],
                    unitCellSize=unitCellSize,
                    stiffMatrix=stiffMatrix,
                )
                if freq:
                    export_frequencies(inpFile, expFile_f)

        if 'per' in file or 'disStruts' in file:
            if file.endswith('.inp'):
                inpFile = os.path.join(curDirectory, file)
                expFile = curDirectory + "/transfer/IN-s" + file[:-4].replace('_','-') + ".csv"
                export_struts(inpFile, expFile)
