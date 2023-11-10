import odbAccess
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import random
import math
import mesh
import interaction
import sys
import os
import odbAccess


initial = 1
numOfJobs = initial + 170

for kk in range(initial,numOfJobs):
    Job = 'Fracture_FCC_21X21_dis-' + str(kk) + '.odb'
    data = 'Fracture_dis-' + str(kk) + '.txt'

    res1= openOdb(path=Job) 
    files99 = open(data,"a+")
    # Create variables that refer to the first two steps.

    e=res1.rootAssembly.instances['PART-1-1'].elements 

    f1=res1.steps['Step-1'].frames[-1]
    step1 = res1.steps.values()[0]
    regS1 = res1.rootAssembly.nodeSets['LOAD']

    numNode=len(f1.fieldOutputs['RF'].getSubset(region=regS1).values)


    count = 0
    for j in res1.steps[step1.name].frames:
        count = count + 1
        RForce=j.fieldOutputs['RF']
        Disp=j.fieldOutputs['UT']
        FX = RForce.getSubset(region=regS1).values[0].data[1]
        UX = Disp.getSubset(region=regS1).values[0].data[1]
        files99.write("%i %5.10f %5.10f\n" % (int(count),UX,FX))

    files99.close()
    res1.close()