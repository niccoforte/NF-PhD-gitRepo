from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np

mode = "Ductile"
LAT = "tri"
DIS = "per"
nnx = 11

os.chdir(r"C:\Users\exy053\Documents\ModelChanges\\" + LAT[:3])#\sims\Ti10\\" + LAT)# + "\\0.13")

if not os.path.exists("transfer"):
    os.makedirs("transfer")

length = 95.26
height = 90

initial = 1
numOfJobs = initial + 1

for kk in range(initial,numOfJobs):
    Job = mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".odb"
    data = "transfer/OUT-" + mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".csv"

    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = ["U2", "RF2"]

    reg_load = 'Node '
    
    U2s = []
    RF2s = []
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