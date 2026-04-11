import os
import numpy as np
import math
import sys
from resources.abaqus import export_frequencies, export_nodes, export_struts
from resources.lattices import Geometry

stiffMatrix = False
distribution = "lhs_uniform"
unitCellSize = 10.0

pDir = "C:\\temp"

cmdIN = sys.argv[8:]
if len(cmdIN) > 0:
    latticeType = str(cmdIN[0])
    nnx = int(cmdIN[1])
    unitCellSize = float(cmdIN[2])
    MechanicalModel = str(cmdIN[3])
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

os.chdir(pDir)


freq = False
if distribution.lower() == "frequency" or distribution.lower() == "opt-f":
    freq = True

if not os.path.exists("transfer"):
    os.makedirs("transfer")

for file in os.scandir():
    if 'per' in file.name or 'disNodes' in file.name:
        if file.name.endswith('.inp'):
            expFile_n = "transfer/IN-n" + file.name[:-4].replace('_','-') + ".csv"
            expFile_f = "transfer/IN-f" + file.name[:-4].replace('_','-') + ".csv"
            export_nodes(
                file.name,
                expFile_n,
                latticeType=file.name.split('-')[1],
                unitCellSize=unitCellSize,
                stiffMatrix=stiffMatrix,
            )
            if freq:
                export_frequencies(file.name, expFile_f)

    if 'per' in file.name or 'disStruts' in file.name:
        if file.name.endswith('.inp'):
            expFile = "transfer/IN-s" + file.name[:-4].replace('_','-') + ".csv"
            export_struts(file.name, expFile)
