from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys

mode = "Fracture"
LAT = "hex"
DIS = "per"
nnx = 40
initial = 1
numOfJobs = 1

#os.chdir(r"C:\\Users\\exy053\\Documents\\validation\\3\\0.13") #PerConv2\3\kagome") #C:\Users\exy053\Documents\ModelChanges\\" + LAT[:3])#\sims\Ti10\\" + LAT)# + "\\0.13")
os.chdir(r"C:\\Users\\exy053\\Documents\\PerSizeConv3\\3\\")
#os.chdir(r"C:\\Users\\exy053\\Documents\\ModelChanges")

if not os.path.exists("transfer"):
    os.makedirs("transfer")

for kk in range(initial, initial+numOfJobs):
    Job = mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".odb"
    data = "transfer/OUT-" + mode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".csv"

    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = ["U2", "RF2", "STATUS"]

    reg_load = 'Node ASSEMBLY.1'
    reg_cracktip = 'Element '

    U2 = [i[1] for i in odb.steps[step].historyRegions[reg_load].historyOutputs[variables[0]].data]
    RF2 = [i[1] for i in odb.steps[step].historyRegions[reg_load].historyOutputs[variables[1]].data]
    
    ALL_STATUS = []
    for reg in odb.steps[step].historyRegions.keys():
        if reg_cracktip in reg:
            STATUS = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[2]].data]
            ALL_STATUS.append(STATUS)
    
    ALL_STATUS = np.transpose(ALL_STATUS)
    
    STEPS_OUT = []
    for U, RF, STAT in zip(U2, RF2, ALL_STATUS):
        OUT = [U, RF]
        for el_STAT in STAT:
            OUT.append(el_STAT)
        STEPS_OUT.append(OUT)
    
    np.savetxt(data, STEPS_OUT, delimiter=",")
    odb.close()