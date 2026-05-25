import numpy as np
import sys
import os
from resources.abaqus import get_DuctData, get_FracData
from resources.lattices import Geometry

unitCellSize = 10.0
Cmatrix = False
ffilter = None

LAT = "FCC"
DIS = "per"
nnx = 20
simN = 1
odbName = f"{LAT}-{nnx}-{DIS}-{simN}"

initial = 1
numberOfRuns = 1
HistOut_frames = 200

pDir = r"C:\\temp"

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
    Cmatrix = False
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

if Cmatrix:
    BCtype = "periodic"
    pDir = f"Z:\\p1\\sims\\Ti\\stiffMatrix\\Cmatrix-{BCtype}"

if pDir in [None, "", "None", "none"]:
    pDir = os.getcwd()

os.chdir(pDir)

for curDirectory, folders, files in os.walk(pDir):
    odbs = [f for f in files if f.endswith('.odb')]
    if ffilter is not None:
        odbs = [f for f in odbs if ffilter in f]
    for odb in odbs:
        odbPath = os.path.join(curDirectory, odb)
        if not os.path.exists(curDirectory + "/transfer"):
            os.makedirs(curDirectory + "/transfer")
        
        LAT = odb.split('-')[1]
        unitCellSize = unitCellSize
        nnx = int(odb.split('-')[2])
        if Cmatrix:
            case_Cmatrix = list(odb.split('-')[3])[-1].lower()
        geom = Geometry(LAT, unitCellSize, nnx) 
        H, L, B = geom.H, geom.L, geom.B
        if 'Ductile' in odb:
            Job = odbPath
            data = curDirectory + "/transfer/OUT-" + odb.split('.')[0]
            OUT = get_DuctData(
                Job,
                H,
                L,
                B,
                Cmatrix=Cmatrix,
                case_Cmatrix=(case_Cmatrix if Cmatrix else None),
                BCtype=(BCtype if Cmatrix else "periodic"),
                expected_steps=HistOut_frames+1,
            )
            for i, out in enumerate(OUT):
                if Cmatrix:
                    np.savetxt(data+f'-{i}.csv', out, delimiter=",")
                else:
                    np.savetxt(data+'.csv', out, delimiter=",")
        elif 'Fracture' in odb:
            Job = odbPath
            data = curDirectory + "/transfer/OUT-" + odb.split('.')[0]
            OUT = get_FracData(Job, expected_steps=HistOut_frames+1)
            np.savetxt(data+'.csv', OUT, delimiter=",")
