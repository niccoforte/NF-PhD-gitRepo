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


length = 210
height = 210

initial = 1
numOfJobs = initial + 170

for kk in range(initial,numOfJobs):
	Job = 'Ductile_FCC_21X21_dis-' + str(kk) + '.odb'
	data = 'Ductile_dis-' + str(kk) + '.txt'

	res1= openOdb(path=Job) 
	files99 = open(data,"a+")
	# Create variables that refer to the first two steps.

	e=res1.rootAssembly.instances['PART-1-1'].elements 

	f1=res1.steps['Step-1'].frames[-1]
	step1 = res1.steps.values()[0]
	regS1 = res1.rootAssembly.nodeSets['SET-TOP']

	numNode = len(f1.fieldOutputs['RF'].getSubset(region=regS1).values)

	Area = length

	count = 0
	for j in res1.steps[step1.name].frames:
		count = count + 1
		RForce = j.fieldOutputs['RF']
		Disp = j.fieldOutputs['UT']
		sumFX = 0.0
		tDisp = 0.0
		for ii in range(0,numNode):
            FX = RForce.getSubset(region=regS1).values[ii].data[1]
            UX = Disp.getSubset(region=regS1).values[ii].data[1]
            sumFX = sumFX + FX
            tDisp = tDisp + UX
		Stress = sumFX/(Area)
		Strain = tDisp/numNode/height
		files99.write("%i %5.10f %5.10f\n" % (int(count),Strain,Stress))

	files99.close()
	res1.close()