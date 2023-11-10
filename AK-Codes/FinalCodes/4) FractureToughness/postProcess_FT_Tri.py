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

num= sys.argv[-3]
#num = 1
Job= sys.argv[-2]
#Job = 'FCC_15X15_10%.odb'
fileName=sys.argv[-1]
#data = 'FCC_15X15_FractureToughness.txt'


os.chdir(r"F:\ANN_FractureToughness\DuctileMaterial\Triangular")

res1= openOdb(path=Job) 

files99 = open(fileName,"a+")
# Create variables that refer to the first two steps.

f1=res1.steps['Step-1'].frames[-1]
regS1 = res1.rootAssembly.nodeSets['LOAD']

numNode=len(f1.fieldOutputs['RF'].getSubset(region=regS1).values)

RForceX=f1.fieldOutputs['RF'].getSubset(region=regS1).values[-1].data[0]
RForceY=f1.fieldOutputs['RF'].getSubset(region=regS1).values[-1].data[1]

magnitute = sqrt(RForceX*RForceX+RForceY*RForceY)

files99.write("%i %5.10f\n" % (int(num),magnitute))

files99.close()
res1.close()