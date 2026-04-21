from abaqus import *
from abaqusConstants import *
from odbAccess import *
import os
from resources.abaqus import export_Udata, export_nodes
from resources.lattices import Geometry, connectivity

mode = "any"
unitCellSize = 10.0

LAT = "FCC"
DIS = "per"
nnx = 20

initial = 1
numOfJobs = 1

pDir = r"Z:\\p1\\data\\Ti\\disNodes\\"
os.chdir(pDir)

   

skipDirs = "Opt"
if mode.lower() == 'any':
    for curDirectory, folders, files in os.walk(pDir):
        folders[:] = [d for d in folders if skipDirs not in d]
        odbs = [f for f in files if f.endswith('.odb')]
        inps = [f for f in files if f.endswith('.inp')]
        for odb in odbs:
            odbPath = os.path.join(curDirectory, odb)
            if not os.path.exists(odbPath.split('.odb')[0]):
                os.makedirs(odbPath.split('.odb')[0])
            
            sim = odb.split('.')[0]
            MechMode = sim.split('-')[0]
            LAT = sim.split('-')[1]
            nnx = int(sim.split('-')[2])
            geom = Geometry(LAT, unitCellSize, nnx)
            geom.nodeCount(mode=MechMode)
            
            export_Udata(odbPath, geom.totalNodes, mode=MechMode)
            
        for inp in inps:
            inpPath = os.path.join(curDirectory, inp)
            if not os.path.exists(inpPath.split('.inp')[0]):
                os.makedirs(inpPath.split('.inp')[0])
                
            sim = inp.split('.')[0]
            MechMode = sim.split('-')[0]
            LAT = sim.split('-')[1]
            nnx = int(sim.split('-')[2])
            geom = Geometry(LAT, unitCellSize, nnx)
            geom.nodeCount(mode=MechMode)
            
            nodes = export_nodes(inpPath, totalNodes=geom.totalNodes)
            elems = connectivity(LAT, nodes, geom, job=inpPath)

if (mode.lower() == 'ductile' or mode.lower() == 'both'):
    MechMode = 'Ductile'
    for kk in range(initial, initial+numOfJobs):
        sim = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk)
        if not os.path.exists(sim):
            os.makedirs(sim)
        
        odbPath = sim + ".odb"
        inpPath = sim + ".inp"
        geom = Geometry(LAT, unitCellSize, nnx)
        geom.nodeCount(mode=MechMode)
        
        export_Udata(odbPath, geom.totalNodes, mode=MechMode)
        nodes = export_nodes(inpPath, totalNodes=geom.totalNodes)
        elems = connectivity(LAT, nodes, geom, job=inpPath)
        
if (mode.lower() == 'fracture' or mode.lower() == 'both'):
    MechMode = 'Fracture'
    for kk in range(initial, initial+numOfJobs):    
        sim = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk)    
        if not os.path.exists(sim):
            os.makedirs(sim)
        
        odbPath = sim + ".odb"
        inpPath = sim + ".inp"
        geom = Geometry(LAT, unitCellSize, nnx)
        geom.nodeCount(mode=MechMode)
        
        export_Udata(odbPath, geom.totalNodes, mode=MechMode)
        nodes = export_nodes(inpPath, totalNodes=geom.totalNodes)
        elems = connectivity(LAT, nodes, geom, job=inpPath)
