from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys

mode = "Ductile"
LAT = "FCC"
DIS = "per"
nnx = 30
unitCellSize = 10.0
initial = 1
numOfJobs = 1

os.chdir(r"C:\\Users\\exy053\\Documents\\validation\\10\\FCC") #\\" + LAT[:3])#\sims\Ti10\\" + LAT) # + "\\0.13")

if not os.path.exists("transfer"):
    os.makedirs("transfer")

if (LAT.lower() == 'fcc'):
    L = float(unitCellSize * nnx)
    H0 = 0.96 * L
    Hs = [unitCellSize*i for i in range(100)]
    H = min(Hs, key=lambda x:abs(x-H0))
    nny = H/unitCellSize
    if round(nny) % 2.0 == 0:
        if H/L >= 0.96:
            H = H - unitCellSize
            nny = H/unitCellSize
        elif H/L < 0.96:
            H = H + unitCellSize
            nny = H/unitCellSize
    nny = nny + 6
    H = unitCellSize * nny
    nny = int(round(nny))
elif (LAT.lower() == 'tri'):
    if nnx % 2.0 == 1.0:
        nnx = nnx - 1
    L = 0.5 * sqrt(3) * unitCellSize * nnx
    H0 = 0.96 * L
    Hs = [unitCellSize*i for i in range(100)]
    H = min(Hs, key=lambda x:abs(x-H0))
    nny = H/unitCellSize
    if round(nny) % 2.0 == 0:
        if H/L >= 0.96:
            H = H - unitCellSize
            nny = H/unitCellSize
        elif H/L < 0.96:
            H = H + unitCellSize
            nny = H/unitCellSize
    nny = nny + 6
    H = unitCellSize * nny
    nny = int(round(nny))
elif (LAT.lower() == 'kagome'):
    L = unitCellSize*(2.0*nnx - 1)
    H0 = 0.96 * L
    Hs = [(3**0.5)*unitCellSize*i for i in range(100)]
    H = min(Hs, key=lambda x:abs(x-H0))
    nny = H/((3**0.5)*unitCellSize)
    if round(nny) % 2.0 == 0:
        if H/L >= 0.96:
            H = H - ((3**0.5)*unitCellSize)
            nny = H/((3**0.5)*unitCellSize)
        elif H/L < 0.96:
            H = H + ((3**0.5)*unitCellSize)
            nny = H/((3**0.5)*unitCellSize)
    nny = nny + 6
    H = (3**0.5)*unitCellSize*nny
    nny = int(round(nny))
elif (LAT.lower() == 'hex'):
    L = sqrt(3)*unitCellSize*nnx
    H0 = 0.96 * L
    Hs = [(0.5*unitCellSize)+(1.5*unitCellSize*i) for i in range(100)]
    H = min(Hs, key=lambda x:abs(x-H0))
    nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
    if round(nny) % 2.0 == 0:
        if H/L >= 0.96:
            H = H - 1.5*unitCellSize
            nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
        elif H/L < 0.96:
            H = H + 1.5*unitCellSize
            nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
    nny = nny + 8
    H = (0.5*unitCellSize)+(1.5*unitCellSize*nny)
    nny = int(round(nny))

length = L
height = H

#print(L, H)

for kk in range(initial, initial+numOfJobs):
    Job = mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".odb"
    data = "transfer/OUT-" + mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".csv"

    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = ["U2", "RF2"]

    reg_load = 'Node '
    
    U2s = []
    RF2s = []
    print(odb.steps[step].historyRegions.keys())
    for reg in odb.steps[step].historyRegions.keys():
        if reg_load in reg:
            U2 = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[0]].data]
            RF2 = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[1]].data]
            U2s.append(U2)
            RF2s.append(RF2)
    
    U2s = np.transpose(U2s)
    RF2s = np.transpose(RF2s)
    
    numNodes = len(U2s[0])
    strain = []
    stress = []
    for Us_step, RFs_step in zip(U2s, RF2s):
        Usum = 0.0
        RFsum = 0.0
        for U, RF in zip(Us_step, RFs_step):
            Usum += U
            RFsum += RF
        e = Usum/numNodes/height
        s = RFsum/length
        strain.append(e)
        stress.append(s)
    OUT = np.transpose([strain, stress])
    
    np.savetxt(data, OUT, delimiter=",")
    odb.close()