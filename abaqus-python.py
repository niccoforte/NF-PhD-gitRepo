###LESSON 1

## SHELL COMMANDS

#Run Abaqus CAE py scrypt without calling GUI.
abaqus cae noGUI=example.py

#Run Abaqus Viewer py scrypt without calling GUI.
abaqus viewer noGUI=example.py

#Run Abaqus using python interface - without license, limitations.
abaqus python example.py

## PYTHON SCRIPTING

#Important Modules
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup

#Create model
executeOnCaeStartup()
myModel = mdb.Model(name='Model A')

#Sketch
s = myModel.ConstrainedSketch(name='__profile__', sheetSize=200.)

#Draw model
s.Line(point1=(point1=(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))

#Create part object
p = myModel.Part(name='rect_beam', dimensionality=THREE_D, type=DEFORMABLE_BODY))

#Extruding sketch to make part
p.BaseSolidExtrude(sketch=s, depth=20.0)
s.unsetPrimaryObject()

#Set part to viewport
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mymodel.sketches['__profile__']

#ODB file
import visualization
myOdbName = 'beam_model.odb'

myViewpName = session.currentViewportName
myViewp = session.viewports[myViewpName]

#Open ODB
myodb = visualization.openODB(path=myOdbName)
myViewp.setValues(displayedObject=myodb)

#Step
mystep = myodb.step['Step-1']
frame1 = mystep.frames[-1]
frame2 = mystep.frames[-2]

#Access displacement & stress field outputs
u1 = frame1.fieldOutputs['U']
u2 = frame2.fieldOutputs['U']
s1 = frame1.fieldOutputs['S']
s2 = frame2.fieldOutputs['S']

du = u2 - u1
ds = s2 - s1

#Plot contours
myViewp.odbDisplay.setDeformedVaraible(du)
myViewp.odbDisplay.setPrimaryVariable(field=ds, outputPosition=INTEGRATION_POINT, refinement=(INVARIANT,'Mises'))
myViewp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF))

#Run on Abaqus python
from abaqusConstants import *
from odbAccess import *

myOdbName = ('beam_model.odb')
myodb = openOdb(path=myOdbName)

mystep = myodb.step['Step-1']

frame1 = mystep.frames[-1]
frame2 = mystep.frames[-2]

u1 = frame1.fieldOutputs['U']
u2 = frame2.fieldOutputs['U']
s1 = frame1.fieldOutputs['S']
s2 = frame2.fieldOutputs['S']

du = u2 - u1
ds = s2 - s1

fout = open('delta_displacement.dat','w')
for value in deltaDisp.values:
    fout.write('%8d, %15.8E, %15.8E, 15.8E,\N'%tuple([value.nodeLabel,]+ list(value.data)))

fout.close()


### LESSON 2

#Object Model Trees: odb, session, mdb 
from abaqus import *                        #import model trees

import part, materials, assembly, section, mesh, #...
#or just 
from caeModules import *

from abaqusConstants import *               #import abaqus constants

import visualization
import odbAccess
openOdb('path to .odb file')

#EX: odb -> steps -> Step -> frames -> Frame -> fieldOutputs -> FieldOutput 'S' -> MISES
odb.steps['Step-1'].frames[1].fieldOutputs['S'].getScalarField(invariant=MISES)

object.__members__   #get attributes - access object data
object.__methods__   #get methods - manipulate object data
print(object)        #prints attributes and their values
dir(object)          #get attributes and methods (inc. built-in)















