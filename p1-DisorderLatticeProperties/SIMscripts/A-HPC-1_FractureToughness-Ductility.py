from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np
import sys
import time
from resources.lattices import Geometry, connectivity, insidePoint, pStrainProperties
from resources.abaqus import node, in_circle
executeOnCaeStartup()

starttime = time.time()

############################################################################################
####################################### INPUT ##############################################
############################################################################################

unitCellSize = 10.0                             # Strut length
thickness = None                                # Strut thickness
outofPlaneThick = None                          # Out of plane thickness
latticeType = 'FCC'                             # 'FCC', 'tri', 'hex', 'kagome'
MechanicalModel = 'both'                        # 'fracture', 'ductile', 'both'
userMaterial = 'Ti'                             # 'al', 'sic', 'ti'
relDensity = 0.2                                # relative density
crossSection = 'rect'
if latticeType.lower() == "tri": nnx = 30
elif latticeType.lower() == "kagome": nnx = 20
elif latticeType.lower() == "hex": nnx = 20
elif latticeType.lower() == "fcc": nnx = 20
elif "square" in latticeType.lower(): nnx = 20
nnx = nnx                                       # number of Unit cells in X direction (Y automatic)

finalRun = 'no'
numberOfRuns = 1
initialJob = 1
cpus = 8
FieldOut_frames = 20
HistOut_frames = 200

distribution = 'lhs_uniform'                    # uniform, lhs_uniform, frequency, normal, exponential
targeted_disorder = "all"                       # all, X, nX, D, DD, DDD, v, h, o, oo, xs
nodeVar = 'no'                                  # distortion
sizeVar = 'no'
fac = 0.2
beta = fac

pStrainUT = True

stiffMatrix = False
Cmatrix_sim = False
UTval = False

global frequencies
frequencies = []

cmdIN = sys.argv[10:]
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
    Cmatrix_sim = False
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
    MechanicalModel = 'ductile'
    pDir = "Z:\\p1\\sims\\Ti\\StiffMatrix\\transfer"
    finalRun = 'inp'

if Cmatrix_sim:
    MechanicalModel = 'ductile'
    BCtype = "periodic"             # KUBC, periodic
    pDir = f"Z:\\p1\\sims\\Ti\\stiffMatrix\\Cmatrix-{BCtype}"
    cases_Cmatrix = ['a' , 'b', 'c']
    pStrainUT = False

if UTval:
    latticeType = "tri"
    nnx = 20
    unitCellSize = 10.0
    MechanicalModel = 'ductile'
    finalRun = 'no'
    userMaterial = 'al'
    nodeVar = 'no'
    sizeVar = 'no'
    pDir = "C:\\Users\\exy053\\Documents\\al\\new\\18-1.1"

# os.chdir(pDir)

STEP_TIME = 1E-1
sm_amp = False
AdaptiveTimeStepping = False
RayleighDampling = False
SevereDisplacementControl = False

if (latticeType.lower() == "kagome" or latticeType.lower() == "hex"):
    AdaptiveTimeStepping = True

## AMPLITUDE
if userMaterial.lower() == "ti":
    if latticeType.lower() == "fcc" or latticeType.lower() == '45square': # amplitude (uniax = strainAppUT * H; FT = stainAppFT * H)
        strainAppUT = 0.060                                               # FINAL 20 - 0.060
        strainAppFT = 0.080                                               # FINAL 20 - 0.080
    elif latticeType.lower() == "tri":
        strainAppUT = 0.100                                               # FINAL 30 - 0.100
        strainAppFT = 0.085                                               # FINAL 30 - 0.085
    elif latticeType.lower() == "kagome":
        strainAppUT = 0.070                                               # FINAL 20 - 0.070
        strainAppFT = 0.080                                               # FINAL 20 - 0.080
    elif latticeType.lower() == "hex":
        strainAppUT = 0.165                                               # FINAL 20 - 0.165
        strainAppFT = 0.200                                               # FINAL 20 - 0.200
elif userMaterial.lower() == "sic":
    if latticeType.lower() == "fcc":
        strainAppUT = 0.00125
        strainAppFT = 0.00125
    elif "square" in latticeType.lower():
        strainAppUT = 0.00
        strainAppFT = 0.01
    elif latticeType.lower() == "tri":
        strainAppUT = 0.00
        strainAppFT = 0.00
    elif latticeType.lower() == "kagome":
        strainAppUT = 0.00
        strainAppFT = 0.00
    elif latticeType.lower() == "hex":
        strainAppUT = 0.00
        strainAppFT = 0.00
elif userMaterial.lower() == "al":
    if latticeType.lower() == "fcc":
        strainAppUT = 0.035
        strainAppFT = 0.050
    elif latticeType.lower() == "tri":
        strainAppUT = 0.021
        strainAppFT = 0.03
    elif latticeType.lower() == "kagome":
        strainAppUT = 0.072
        strainAppFT = 0.067
    elif latticeType.lower() == "hex":
        strainAppUT = 0.100
        strainAppFT = 0.032

## MESH SIZING
if latticeType.lower() == "fcc" or "square" in latticeType.lower():
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/5.0
    FineElemSizeUT   = unitCellSize/5.0
    CoarseElemSizeFT = unitCellSize/2.0
    FineElemSizeFT   = unitCellSize/5.0
elif latticeType.lower() == "tri":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/2.0
    FineElemSizeUT   = unitCellSize/5.0
    CoarseElemSizeFT = unitCellSize/2.0
    FineElemSizeFT   = unitCellSize/5.0
elif latticeType.lower() == "kagome":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/1.0
    FineElemSizeUT   = unitCellSize/10.0
    CoarseElemSizeFT = unitCellSize/1.0
    FineElemSizeFT   = unitCellSize/10.0
elif latticeType.lower() == "hex":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/1.0
    FineElemSizeUT   = unitCellSize/15.0
    CoarseElemSizeFT = unitCellSize/1.0
    FineElemSizeFT   = unitCellSize/10.0
if Cmatrix_sim:
    CoarseElemSizeUT = unitCellSize/10.0
    FineElemSizeUT   = unitCellSize/10.0

############################################################################################
################################## START ###################################################
############################################################################################

## For PSC/DSC:
# for nnx in nnxs:

## For MeshConv:
# for CoarseElemSizeUT, CoarseElemSizeFT in zip(CoarseElemSizeUTs, CoarseElemSizeFTs):
#     for FineElemSizeUT, FineElemSizeFT in zip(FineElemSizeUTs, FineElemSizeFTs):
#         if not os.path.exists(pDir+"\\"+str(int(unitCellSize/BracketElemSize))+"-"+str(int(unitCellSize/CoarseElemSizeUT))+"-"+str(int(unitCellSize/FineElemSizeUT))):
#             os.makedirs(pDir+"\\"+str(int(unitCellSize/BracketElemSize))+"-"+str(int(unitCellSize/CoarseElemSizeUT))+"-"+str(int(unitCellSize/FineElemSizeUT)))
#         os.chdir(pDir+"\\"+str(int(unitCellSize/BracketElemSize))+"-"+str(int(unitCellSize/CoarseElemSizeUT))+"-"+str(int(unitCellSize/FineElemSizeUT)))

if (nodeVar == 'no' and sizeVar == 'no'):
    imper = 'per'
elif (nodeVar == 'yes' and sizeVar == 'no'):
    imper = 'disNodes'
elif (nodeVar == 'no' and sizeVar == 'yes'):
    imper = 'disStruts'
else:
    imper = 'disNodesStruts'

if (finalRun.lower() == 'yes' or finalRun.lower() == 'inp' or finalRun.lower() == 'input'):
    initial = initialJob
    numOfJobs = initial + numberOfRuns
elif (finalRun.lower() == 'no'):
    initial = initialJob
    numOfJobs = initial + 1

if nodeVar == "no":
    fac = 0.0

geom = Geometry(latticeType, unitCellSize, nnx)
if stiffMatrix:
    geom.stiffnessMatrix()
if UTval:
    geom.UTval()
nnx = geom.nnx
nny = geom.nny
L = geom.L
H = geom.H
W = geom.W
B = geom.B
a0 = geom.a0
ai = geom.ai
totalNodes = geom.totalNodes
totalBracketNodes = geom.totalBracketNodes
deltaNM = geom.deltaNM
delta = deltaNM * fac

if (distribution.lower() == 'lhs_uniform'):    
    if numberOfRuns == 1:
        distribution = 'uniform'

if (distribution.lower() == 'uniform'):
    fac = fac
    dist = "uni"
elif (distribution.lower() == 'lhs_uniform'):
    fac = fac
    dist = "lhs"
elif (distribution.lower() == 'frequency'):
    fac = fac
    dist = "freq"
elif (distribution.lower() == 'opt') or (distribution.lower() == 'opt-f'):
    fac = fac
    dist = "opt"
elif (distribution.lower() == 'normal'):
    fac = (2*fac)/np.sqrt(2*np.pi*np.exp(1))
    dist = "norm"
elif (distribution.lower() == 'exponential'):
    fac = np.exp(1)/(2*fac)
    dist = "exp"

for idNum in range(initial,numOfJobs):
    
    #data 	     = sys.argv[-1]
    PBC          = 'no'
    elemType     = B21
    units        = 'millimeter'    # mass = tonn, length = millimeter, stress = MPa
    
    nodes, nodesR, bracket_nodes = node(
        latticeType,
        L,
        H,
        nnx,
        nny,
        totalNodes,
        totalBracketNodes,
        delta,
        distribution,
        unitCellSize=unitCellSize,
        targeted_disorder=targeted_disorder,
        idNum=idNum,
        initialJob=initial,
        numberOfRuns=(numOfJobs - initial),
        frequencies=frequencies,
        opt_dis_x=globals().get("opt_dis_x"),
        opt_dis_y=globals().get("opt_dis_y"),
    )

    if (distribution.lower() == 'opt') or (distribution.lower() == 'opt-f'):
        idNum = sampleN

# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ###################################### Ductile Model #################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
        
    if  (MechanicalModel.lower() == 'ductile' or MechanicalModel.lower() == 'both'):
        if Cmatrix_sim: cases = cases_Cmatrix
        else: cases = ['none']
        for caseCmatrix in cases:
            ModelName = f"Ductile-{latticeType}-{int(nnx)}-{int(fac*100)}{imper}-{dist}-{targeted_disorder}-{idNum}"
            if imper == 'per':
                ModelName = f"Ductile-{latticeType}-{int(nnx)}-per-{idNum}"
            if stiffMatrix:
                ModelName = f"Ductile-{latticeType}-{int(nnx)}-stiffMatrix-{int(fac*100)}{imper}-{dist}-{targeted_disorder}-{idNum}"
                if latticeType.lower() == "tri":
                    ModelName = f"Ductile-{latticeType}-{int(nnx/2)}-stiffMatrix-{int(fac*100)}{imper}-{dist}-{targeted_disorder}-{idNum}"
                if imper == 'per':
                    ModelName = f"Ductile-{latticeType}-{int(nnx/2)}-stiffMatrix-per-{idNum}"
            if Cmatrix_sim:
                ModelName = f"Ductile-{latticeType}-{int(nnx)}-Cmatrix{caseCmatrix.upper()}-{int(fac*100)}{imper}-{dist}-{targeted_disorder}-{idNum}"
                if imper == 'per':
                    ModelName = f"Ductile-{latticeType}-{int(nnx)}-Cmatrix{caseCmatrix.upper()}-per-{idNum}"
            Job = ModelName

            # FIX STIFF MATRIX NAMING AND DIMENSIONS AND ETC

            #############################################################################################
            #################################### Brackets ###############################################
            #############################################################################################
            
            if stiffMatrix or Cmatrix_sim:
                nodes_duct = nodes
                nodesR_duct = nodesR
            else:
                nodes_duct = np.append(nodes, bracket_nodes, axis=0)
                nodesR_duct = np.append(nodesR, bracket_nodes, axis=0)
            
            #############################################################################################
            #################################### Strut Elements #########################################
            #############################################################################################
            
            element = connectivity(latticeType, nodes, geom)
            element_duct = connectivity(latticeType, nodes_duct, geom)
            
            #############################################################################################
            ################################ Radius Calculation #########################################
            #############################################################################################
            
            if outofPlaneThick is None:
                outofPlaneThick = 0.01
                if UTval:
                    outofPlaneThick = 2.0
                if pStrainUT:
                    outofPlaneThick = B
            
            length = np.zeros(shape=(len(element),1))
            for ik in range(0,len(element)-1):
                x1 = nodesR[int(element[ik][1]-1)][1]
                x2 = nodesR[int(element[ik][2]-1)][1]
                y1 = nodesR[int(element[ik][1]-1)][2]
                y2 = nodesR[int(element[ik][2]-1)][2]
                length[ik][0] = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

            if (crossSection.lower() == 'circ'):
                constants = [4 *relDensity, (L + H - np.pi * sum(length)), 4 * relDensity * (L * H)]
                dia_opt = np.roots(constants)
                dia_est = 2 * relDensity * 4 * L * H / (sum(length) * 2 * np.pi)
                diff_sqr = [(dia_opt[0] - dia_est) ** 2, (dia_opt[1] - dia_est) ** 2]
                index = np.argmin(diff_sqr)

                rad = dia_opt[index]/ 2
                Area = 3.14159*rad*rad
                
            elif (crossSection.lower() == 'rect'):
                thick_est = relDensity * L * H * outofPlaneThick / (sum(length) * outofPlaneThick)
                
                if thickness is None:
                    thickness = thick_est
                    if UTval:
                        thickness = 1.1
                Area = 1.0*thickness

            if (units.lower() == 'millimeter'):
                tol = 1e-3
            elif (units.lower() == 'meter'):
                tol = 1e-6

            ############################################################################################
            ####################################### Part Making ########################################
            ############################################################################################

            if (elemType == B32):
                dimenObject = THREE_D
            elif (elemType == B21):
                dimenObject = TWO_D_PLANAR
            elif (elemType == B22):
                dimenObject = TWO_D_PLANAR
            elif (elemType == B23):
                dimenObject = TWO_D_PLANAR
                
            mdb.Model(name=ModelName , modelType=STANDARD_EXPLICIT)

            s = mdb.models[ModelName].ConstrainedSketch(name='__profile__', 
                sheetSize=200.0)
            g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
            s.setPrimaryObject(option=STANDALONE)

            for ik in range(len(element_duct)):
                s.Line(point1=nodesR_duct[int(element_duct[ik][1]-1)][1:3], point2=nodesR_duct[int(element_duct[ik][2]-1)][1:3])
            p = mdb.models[ModelName].Part(name='Part-1', dimensionality=dimenObject, 
                type=DEFORMABLE_BODY)
            p = mdb.models[ModelName].parts['Part-1']
            p.BaseWire(sketch=s)
            s.unsetPrimaryObject()
            p = mdb.models[ModelName].parts['Part-1']
            session.viewports['Viewport: 1'].setValues(displayedObject=p)

            p = mdb.models[ModelName].parts['Part-1']
            e = p.edges
            edges = e.getByBoundingBox(-1000,-1000,0.0,1000,1000,0.0)
            p.Set(edges=edges, name='AllPartSet')
            
            p = mdb.models[ModelName].parts['Part-1']
            e = p.edges
            bodyElems = []
            bodyElem_idxs = []
            for numEdge in range(0,len(element)):
                x1 = nodesR_duct[int(element[numEdge][1]-1)][1]
                x2 = nodesR_duct[int(element[numEdge][2]-1)][1]
                y1 = nodesR_duct[int(element[numEdge][1]-1)][2]
                y2 = nodesR_duct[int(element[numEdge][2]-1)][2]
                xMid = (x1+x2)/2.0
                yMid = (y1+y2)/2.0
                edges = e.findAt(((xMid, yMid, 0.0), ))
                bodyElems.append(edges)
                for i in element_duct:
                    if i[1] == element[numEdge][1] and i[2] == element[numEdge][2]:
                        bodyElem_idxs.append(i[0]-1)
                if sizeVar.lower() == 'yes':
                    p.Set(edges=edges, name='edgeElem-'+str(numEdge))
            p.Set(edges=bodyElems, name='BodySet')
            
            brackElems = []
            elemBrackets = np.delete(element_duct, bodyElem_idxs, 0)
            for numEdge in range(len(element), len(element_duct)):
                x1 = nodesR_duct[int(elemBrackets[numEdge-len(element)][1]-1)][1]
                x2 = nodesR_duct[int(elemBrackets[numEdge-len(element)][2]-1)][1]
                y1 = nodesR_duct[int(elemBrackets[numEdge-len(element)][1]-1)][2]
                y2 = nodesR_duct[int(elemBrackets[numEdge-len(element)][2]-1)][2]
                xMid = (x1+x2)/2.0
                yMid = (y1+y2)/2.0
                edges = e.findAt(((xMid, yMid, 0.0), ))
                brackElems.append(edges)
            if stiffMatrix or Cmatrix_sim:
                brackElems = e.findAt(((0.0, 0.0, 0.0), ))
            p.Set(edges=brackElems, name='BracketSet')
            
            ############################################################################################
            ################################## Material Properties #####################################
            ############################################################################################
            
            if (userMaterial.lower() == 'al'):
                E, nu = 70000.0, 0.3
                if pStrainUT:
                    E, nu  = pStrainProperties(E, nu)
                mdb.models[ModelName].Material(name=userMaterial)
                mdb.models[ModelName].materials[userMaterial].Density(table=((2.7e-09, ), ))
                mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
                mdb.models[ModelName].materials[userMaterial].Plastic(table=((
                    134.0, 0.0), (134.3, 0.001996429), (134.5, 0.002835692), (135.0, 
                    0.004015369), (135.5, 0.005678497), (136.0, 0.008020233), (136.5, 
                    0.011313326), (137.0, 0.015938495), (137.5, 0.022426529), (138.0, 
                    0.031516535), (138.5, 0.044236464), (139.0, 0.062014286)))
                mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                    table=((0.027544286, 0.33333, 0.0), ))
                mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                    type=DISPLACEMENT, softening=TABULAR, table=(
                    (0.0, 0.0),
                    (0.0036, FineElemSizeUT*0.0021),
                    (0.0045, FineElemSizeUT*0.00422), 
                    (0.0104, FineElemSizeUT*0.0063),
                    (0.022,  FineElemSizeUT*0.0084),
                    (0.0336, FineElemSizeUT*0.0105),
                    (0.0466, FineElemSizeUT*0.0126),
                    (0.0599, FineElemSizeUT*0.0147),
                    (0.0761, FineElemSizeUT*0.0168),
                    (0.095,  FineElemSizeUT*0.01891),
                    (0.1173, FineElemSizeUT*0.02101),
                    (0.1443, FineElemSizeUT*0.02311),
                    (0.1761, FineElemSizeUT*0.02521),
                    (0.2144, FineElemSizeUT*0.02731),
                    (0.2578, FineElemSizeUT*0.02928), 
                    (0.3029, FineElemSizeUT*0.03098),
                    (0.3457, FineElemSizeUT*0.03236),
                    (0.3866, FineElemSizeUT*0.03335),
                    (0.4301, FineElemSizeUT*0.034),
                    (0.4665, FineElemSizeUT*0.03446),
                    (1.0,    FineElemSizeUT*0.03547)))

            elif (userMaterial.lower() == 'sic'):
                E, nu = 410000, 0.14
                if pStrainUT:
                    E, nu  = pStrainProperties(E, nu)
                mdb.models[ModelName].Material(name=userMaterial)
                mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
                mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
                mdb.models[ModelName].materials[userMaterial].Plastic(table=
                    ((550, 0.0),
                    (550.01, 0.00001)))
                mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                    table=((0.00001, 0.333333, 0.0), ))
                mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                    type=DISPLACEMENT, softening=TABULAR, table=(
                    (0.0, 0.0),
                    (0.2, FineElemSizeUT*0.00001),
                    (0.4, FineElemSizeUT*0.00002), 
                    (0.6, FineElemSizeUT*0.00003),
                    (0.8, FineElemSizeUT*0.00004),
                    (1.0, FineElemSizeUT*0.00005)))

            elif (userMaterial.lower() == 'ti'):
                E, nu = 123000, 0.3
                if pStrainUT:
                    E, nu  = pStrainProperties(E, nu)
                mdb.models[ModelName].Material(name=userMaterial)
                mdb.models[ModelName].materials[userMaterial].Density(table=((4.43e-09, ), ))
                mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
                mdb.models[ModelName].materials[userMaterial].Plastic(table=
                    ((932,         0),
                    (947.411802,    0.003453491),
                    (957.4512331, 0.006906981),
                    (966.1307689, 0.010360472),
                    (974.030469, 0.013813962),
                    (981.3967087, 0.017267453),
                    (988.3639577, 0.020720943),
                    (995.0160058, 0.024174434),
                    (1001.409616, 0.027627924),
                    (1007.585541, 0.031081415),
                    (1013.574312, 0.034534905),
                    (1019.399572, 0.037988396),
                    (1025.08011, 0.041441886),
                    (1030.631179, 0.044895377),
                    (1036.065381, 0.048348867),
                    (1041.393285, 0.051802358),
                    (1046.623866, 0.055255848),
                    (1051.764831, 0.058709339),
                    (1056.822861, 0.062162829),
                    (1061.803799, 0.06561632),
                    (1066.712789, 0.06906981),
                    (1071.554397, 0.072523301),
                    (1076.332693, 0.075976791),
                    (1081.05133, 0.079430282),
                    (1085.713601, 0.082883772),
                    (1090.322488, 0.086337263),
                    (1094.880702, 0.089790753),
                    (1099.39072, 0.093244244),
                    (1103.854808, 0.096697734),
                    (1108.275052, 0.100151225),
                    (1112.653372, 0.103604715)))
                mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                    table=((0.102268174, 0.333333, 0.0), ))
                mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                    type=DISPLACEMENT, softening=TABULAR, table=(
                    (0.0, 0.0),
                    (0.2, FineElemSizeUT*0.0001),
                    (0.4, FineElemSizeUT*0.0002), 
                    (0.6, FineElemSizeUT*0.0003),
                    (0.8, FineElemSizeUT*0.0004),
                    (1.0, FineElemSizeUT*0.0005)))

            if (sizeVar.lower() == 'yes'):
                relativeDensityUpdated = 10000
                while relativeDensityUpdated < (relDensity - 0.001) or relativeDensityUpdated > (relDensity + 0.001):

                    lowerLim = (1.0 - beta) * thickness
                    upperLim = (1.0 + beta) * thickness

                    thick = np.random.uniform(lowerLim, upperLim, len(element))
                    latticeVolume = np.dot(length[:].T, thick*outofPlaneThick)
                    
                    relativeDensityUpdated = latticeVolume/(L * H * outofPlaneThick)

                    if relativeDensityUpdated > relDensity:
                        thickness = thickness - 0.001
                    else:
                        thickness = thickness + 0.001
                
                mdb.models[ModelName].RectangularProfile(name='RectBracket', a=outofPlaneThick, b=2.0*thickness)
                mdb.models[ModelName].BeamSection(name='BeamSecBracket', integration=DURING_ANALYSIS, 
                    poissonRatio=0.0, profile='RectBracket', material=userMaterial, 
                    temperatureVar=LINEAR, consistentMassMatrix=False)
                
                region = p.sets['BracketSet']
                p.SectionAssignment(region=region, sectionName='BeamSecBracket', offset=0.0, 
                    offsetType=MIDDLE_SURFACE, offsetField='', 
                    thicknessAssignment=FROM_SECTION)
                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
                
                for numEdge in range(0,len(element)):       
                    mdb.models[ModelName].RectangularProfile(name='Rect'+str(numEdge), a=outofPlaneThick, b=thick[numEdge])

                    mdb.models[ModelName].BeamSection(name='BeamSec'+str(numEdge), integration=DURING_ANALYSIS, 
                        poissonRatio=0.0, profile='Rect'+str(numEdge), material=userMaterial, 
                        temperatureVar=LINEAR, consistentMassMatrix=False)

                    p = mdb.models[ModelName].parts['Part-1']
                    region = p.sets['edgeElem-'+str(numEdge)]
                    p.SectionAssignment(region=region, sectionName='BeamSec'+str(numEdge), offset=0.0, 
                        offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)
                    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))

            if (sizeVar.lower() == 'no'):
                if (crossSection.lower() == 'rect'):
                    mdb.models[ModelName].RectangularProfile(name='RectBody', a=outofPlaneThick, b=thickness)
                    mdb.models[ModelName].RectangularProfile(name='RectBracket', a=outofPlaneThick, b=2.0*thickness)
                    
                    mdb.models[ModelName].BeamSection(name='BeamSecBody', integration=DURING_ANALYSIS, 
                        poissonRatio=0.0, profile='RectBody', material=userMaterial, 
                        temperatureVar=LINEAR, consistentMassMatrix=False)
                    mdb.models[ModelName].BeamSection(name='BeamSecBracket', integration=DURING_ANALYSIS, 
                        poissonRatio=0.0, profile='RectBracket', material=userMaterial, 
                        temperatureVar=LINEAR, consistentMassMatrix=False)
                    
                    region = p.sets['BracketSet']
                    p.SectionAssignment(region=region, sectionName='BeamSecBracket', offset=0.0, 
                        offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)
                    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
                    region = p.sets['BodySet']
                    p.SectionAssignment(region=region, sectionName='BeamSecBody', offset=0.0, 
                        offsetType=MIDDLE_SURFACE, offsetField='', 
                        thicknessAssignment=FROM_SECTION)
                    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
               
                elif (crossSection.lower() == 'circ'):
               
                   mdb.models[ModelName].CircularProfile(name='Circ', r=rad)
    
                   mdb.models[ModelName].BeamSection(name='BeamSec', integration=DURING_ANALYSIS, 
                       poissonRatio=0.0, profile='Circ', material=userMaterial, 
                       temperatureVar=LINEAR, consistentMassMatrix=False)
    
                   p = mdb.models[ModelName].parts['Part-1']
                   region = p.sets['AllPartSet']
                   p.SectionAssignment(region=region, sectionName='BeamSec', offset=0.0, 
                       offsetType=MIDDLE_SURFACE, offsetField='', 
                       thicknessAssignment=FROM_SECTION)
    
                   p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))

            ###########################################################################################
            ####################################### Assembly ##########################################
            ###########################################################################################

            a = mdb.models[ModelName].rootAssembly
            a.DatumCsysByDefault(CARTESIAN)
            p = mdb.models[ModelName].parts['Part-1']
            a.Instance(name='Part-1-1', part=p, dependent=OFF)

            ############################################################################################
            ######################################## Step ##############################################
            ############################################################################################
            
            if AdaptiveTimeStepping:
                mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=STEP_TIME, 
                                                        improvedDtMethod=ON, scaleFactor=0.95)
            
            if RayleighDampling:
                mdb.models[ModelName].materials[userMaterial].Damping(alpha=0.0, beta=1e-6)
                mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=STEP_TIME)
            
            if SevereDisplacementControl:
                mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=STEP_TIME,
                                                        linearBulkViscosity=0.06, quadBulkViscosity=1.2)
            
            if AdaptiveTimeStepping == False and RayleighDampling == False and SevereDisplacementControl == False:
                mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=STEP_TIME)
            
            # ############################################################################################
            # ######################################## Mesh ##############################################
            # ############################################################################################
            
            a = mdb.models[ModelName].rootAssembly
            e = a.instances['Part-1-1'].edges
            edges = e.getByBoundingBox(-L,-H,0.0,2*L,2*H,0.0)
            elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
            pickedRegions =(edges, )
            a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
            partInstances =(a.instances['Part-1-1'], )
            a.seedPartInstance(regions=partInstances, size=BracketElemSize, deviationFactor=0.1, 
                minSizeFactor=0.1)
            a = mdb.models[ModelName].rootAssembly
            partInstances =(a.instances['Part-1-1'], )
            a.generateMesh(regions=partInstances)
            
            a = mdb.models[ModelName].rootAssembly
            e = a.instances['Part-1-1'].edges
            edges = e.getByBoundingBox(0.0,0.0,0.0,L,H,0.0)
            elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
            pickedRegions =(edges, )
            a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
            partInstances =(a.instances['Part-1-1'], )
            a.seedEdgeBySize(edges=edges, size=CoarseElemSizeUT, deviationFactor=0.1, 
                minSizeFactor=0.1)
            a = mdb.models[ModelName].rootAssembly
            partInstances =(a.instances['Part-1-1'], )
            a.generateMesh(regions=partInstances)
            
            a = mdb.models[ModelName].rootAssembly
            e = a.instances['Part-1-1'].edges
            edges = e.getByBoundingBox(-L, H/6, 0.0, 2*L, H-(H/6), 0.0)
            elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
            pickedRegions =(edges, )
            a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
            partInstances = (a.instances['Part-1-1'], )
            a.seedEdgeBySize(edges=edges, size=FineElemSizeUT, deviationFactor=0.1, 
                minSizeFactor=0.1, constraint=FINER)
            a = mdb.models[ModelName].rootAssembly
            partInstances =(a.instances['Part-1-1'], )
            a.generateMesh(regions=partInstances)
            
            all_nodes = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].nodes
            
            left_nodes = []
            bottom_nodes = []
            right_nodes = []
            top_nodes = []
            topBody_nodes = []
            bottomBody_nodes = []
            if latticeType.lower() == 'tri':
                for n in all_nodes:
                    xcoord = n.coordinates[0]
                    ycoord = n.coordinates[1]
                    if xcoord > -tol and xcoord < +tol and ycoord > -tol and ycoord < (H+tol):
                        left_nodes.append(n)
                    if ycoord > (-3*unitCellSize)-tol and ycoord < (-3*unitCellSize)+tol:
                        bottom_nodes.append(n)
                    if xcoord > L-tol and xcoord < L+tol and ycoord > -tol and ycoord < (H+tol):
                        right_nodes.append(n)
                    if ycoord > (H+3*unitCellSize)-tol and ycoord < (H+3*unitCellSize)+tol:
                        top_nodes.append(n)
                    if ycoord > (H-tol) and ycoord < (H+tol) and xcoord > -tol and xcoord < (L+tol):
                        topBody_nodes.append(n)
                    if ycoord > -tol and ycoord < tol and xcoord > -tol and xcoord < (L+tol):
                        bottomBody_nodes.append(n)
            elif latticeType.lower() == 'hex':
                for n in all_nodes:
                    xcoord = n.coordinates[0]
                    ycoord = n.coordinates[1]
                    if xcoord > -tol and xcoord < +tol and ycoord > -tol and ycoord < (H+tol):
                        left_nodes.append(n)
                    if ycoord > (-6*unitCellSize)-tol and ycoord < (-6*unitCellSize)+tol:
                        bottom_nodes.append(n)
                    if xcoord > L-tol and xcoord < L+tol and ycoord > -tol and ycoord < (H+tol):
                        right_nodes.append(n)
                    if ycoord > (H+6*unitCellSize)-tol and ycoord < (H+6*unitCellSize)+tol:
                        top_nodes.append(n)
                    if ycoord > (H-tol) and ycoord < (H+tol) and xcoord > -tol and xcoord < (L+tol):
                        topBody_nodes.append(n)
                    if ycoord > -tol and ycoord < tol and xcoord > -tol and xcoord < (L+tol):
                        bottomBody_nodes.append(n)
            elif latticeType.lower() == 'kagome':
                rfNodes = []
                val = -2.0*unitCellSize
                for kk in range(0,2*nnx+5):
                    rfNodes.append(val)
                    val = val + unitCellSize
                for n in all_nodes:
                    xcoord = n.coordinates[0]
                    ycoord = n.coordinates[1]
                    if xcoord > -tol and xcoord < +tol and ycoord > -tol and ycoord < (H+tol):
                        left_nodes.append(n)
                    if ycoord > (-3*np.sqrt(3)*unitCellSize)-tol and ycoord < (-3*np.sqrt(3)*unitCellSize)+tol:
                        bottom_nodes.append(n)
                    if xcoord > L-tol and xcoord < L+tol and ycoord > -tol and ycoord < (H+tol):
                        right_nodes.append(n)
                    if ycoord > (H+3*np.sqrt(3)*unitCellSize)-tol and ycoord < (H+3*np.sqrt(3)*unitCellSize)+tol:
                        exist = xcoord in rfNodes
                        if exist:
                            top_nodes.append(n)
                    if ycoord > (H-tol) and ycoord < (H+tol) and xcoord > -tol and xcoord < (L+tol):
                        topBody_nodes.append(n)
                    if ycoord > -tol and ycoord < tol and xcoord > -tol and xcoord < (L+tol):
                        bottomBody_nodes.append(n)
            elif latticeType.lower() == 'fcc':
                rfNodes = []
                val = -2.0*unitCellSize
                for kk in range(0,nnx+5):
                    rfNodes.append(val)
                    val = val + unitCellSize
                for n in all_nodes:
                    xcoord = n.coordinates[0]
                    ycoord = n.coordinates[1]
                    if xcoord > -tol and xcoord < +tol and ycoord > -tol and ycoord < (H+tol):
                        left_nodes.append(n)
                    if ycoord > (-3*unitCellSize)-tol and ycoord < (-3*unitCellSize)+tol:
                        bottom_nodes.append(n)
                    if xcoord > L-tol and xcoord < L+tol and ycoord > -tol and ycoord < (H+tol):
                        right_nodes.append(n)
                    if ycoord > (H+3*unitCellSize)-tol and ycoord < (H+3*unitCellSize)+tol:
                        exist = xcoord in rfNodes
                        if exist:
                            top_nodes.append(n)
                    if ycoord > (H-tol) and ycoord < (H+tol) and xcoord > -tol and xcoord < (L+tol):
                        topBody_nodes.append(n)
                    if ycoord > -tol and ycoord < tol and xcoord > -tol and xcoord < (L+tol):
                        bottomBody_nodes.append(n)
            if stiffMatrix:
                bottom_nodes.append(all_nodes[0])
                top_nodes.append(all_nodes[0])

            if not Cmatrix_sim:
                ln = mesh.MeshNodeArray(bottom_nodes)
                mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-bottom')

                ln = mesh.MeshNodeArray(top_nodes)
                mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-top')

            ln = mesh.MeshNodeArray(left_nodes)
            mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-left')

            ln = mesh.MeshNodeArray(right_nodes)
            mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-right')
            
            ln = mesh.MeshNodeArray(topBody_nodes)
            mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-topBody')
            
            ln = mesh.MeshNodeArray(bottomBody_nodes)
            mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-bottomBody')
            
            # ############################################################################################
            # ################################## Output Requests #########################################
            # ############################################################################################
            if (userMaterial.lower() == 'material-1'):    
                mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                    'S', 'PE', 'LE', 'UT', 'RF', 'EVOL', 'SDV'), numIntervals=FieldOut_frames)

                regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
                mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                    'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                    rebar=EXCLUDE)

            else:
                regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
                mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                    'ALLIE', 'ALLKE', 'ALLSE'), numIntervals=HistOut_frames, region=regionDef, sectionPoints=DEFAULT, 
                    rebar=EXCLUDE)
                
                if Cmatrix_sim:
                    if BCtype.lower() == "kubc" and caseCmatrix.lower() == "a" or caseCmatrix.lower() == "b":
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-right']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-2', createStepName='Step-1', 
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-topBody']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-3', createStepName='Step-1', 
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)
                    elif BCtype.lower() == "kubc" and caseCmatrix.lower() == "c":
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-right']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-2', createStepName='Step-1', 
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-left']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-3', createStepName='Step-1', 
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-topBody']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-4', createStepName='Step-1',
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)
                        regionDef=mdb.models[ModelName].rootAssembly.sets['Set-bottomBody']
                        mdb.models[ModelName].HistoryOutputRequest(name='H-Output-5', createStepName='Step-1',
                            variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                            sectionPoints=DEFAULT, rebar=EXCLUDE)      
                
                else:
                    mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                        'S', 'PE', 'LE', 'UT', 'RF', 'SDEG', 'DMICRT', 'STATUS'), numIntervals=FieldOut_frames)
                        
                    
                    regionDef=mdb.models[ModelName].rootAssembly.sets['Set-top']
                    mdb.models[ModelName].HistoryOutputRequest(name='H-Output-2', createStepName='Step-1', 
                        variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                        sectionPoints=DEFAULT, rebar=EXCLUDE)
            
            # ############################################################################################
            # #################################### Loading ###############################################
            # ############################################################################################
            
            if sm_amp:
                mdb.models[ModelName].SmoothStepAmplitude(name='Amp-1', 
                    timeSpan=STEP, data=((0.0, 0.0), (STEP_TIME, strainAppUT*H)))
            else:
                mdb.models[ModelName].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
                    smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (STEP_TIME, strainAppUT*H)))
            
            if Cmatrix_sim:
                if BCtype.lower() == "kubc":
                    mdb.models[ModelName].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
                    smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (STEP_TIME, 1)))

                    if caseCmatrix.lower() == "a":
                        a = mdb.models[ModelName].rootAssembly
                        region = a.sets['Set-right']
                        mdb.models[ModelName].DisplacementBC(name='BC-1', createStepName='Step-1', 
                            region=region, u1=1.0, u2=0.0, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                            distributionType=UNIFORM, fieldName='', localCsys=None)

                        region = a.sets['Set-left']
                        mdb.models[ModelName].DisplacementBC(name='BC-2', createStepName='Step-1', 
                            region=region, u1=0.0, u2=0.0, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                            distributionType=UNIFORM, fieldName='', localCsys=None)
                        
                        for num, up in enumerate(a.sets['Set-topBody'].nodes):
                            if abs(up.coordinates[0]-L)<tol or abs(up.coordinates[0])<tol:
                                continue
                            down = [n for n in a.sets['Set-bottomBody'].nodes if abs(n.coordinates[0]-up.coordinates[0])<tol][0]
                            frac = up.coordinates[0]/L
                            region = a.Set(nodes=mesh.MeshNodeArray([up, down]), name=f'Nset-{num}')
                            mdb.models[ModelName].DisplacementBC(name=f'BC-frac{num}', createStepName='Step-1', 
                                region=region, u1=frac, u2=0.0, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                                distributionType=UNIFORM, fieldName='', localCsys=None)

                    elif caseCmatrix.lower() == "b":
                        a = mdb.models[ModelName].rootAssembly
                        region = a.sets['Set-topBody']
                        mdb.models[ModelName].DisplacementBC(name='BC-1', createStepName='Step-1', 
                            region=region, u1=0.0, u2=1.0, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                            distributionType=UNIFORM, fieldName='', localCsys=None)

                        region = a.sets['Set-bottomBody']
                        mdb.models[ModelName].DisplacementBC(name='BC-2', createStepName='Step-1', 
                            region=region, u1=0.0, u2=0.0, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                            distributionType=UNIFORM, fieldName='', localCsys=None)
                        
                        for num, l in enumerate(a.sets['Set-left'].nodes):
                            if abs(l.coordinates[1]-H)<tol or abs(l.coordinates[1])<tol:
                                continue
                            r = [n for n in a.sets['Set-right'].nodes if abs(n.coordinates[1]-l.coordinates[1])<tol][0]
                            frac = l.coordinates[1]/L
                            region = a.Set(nodes=mesh.MeshNodeArray([l, r]), name=f'Nset-{num}')
                            mdb.models[ModelName].DisplacementBC(name=f'BC-frac{num}', createStepName='Step-1', 
                                region=region, u1=0.0, u2=frac, ur3=UNSET, amplitude='Amp-1', fixed=OFF, 
                                distributionType=UNIFORM, fieldName='', localCsys=None)

                    elif caseCmatrix.lower() == "c":
                        a = mdb.models[ModelName].rootAssembly

                        epsC = 1.0

                        bL = set(n.label for n in a.sets['Set-left'].nodes)
                        bR = set(n.label for n in a.sets['Set-right'].nodes)
                        bT = set(n.label for n in a.sets['Set-topBody'].nodes)
                        bB = set(n.label for n in a.sets['Set-bottomBody'].nodes)
                        all_ids = list(bL | bR | bT | bB)

                        all_nodes = dict((n.label, n) for n in a.instances[a.instances.keys()[0]].nodes)\
                            if hasattr(a, 'instances') else dict((n.label, n) for n in a.nodes)

                        for num, nid in enumerate(all_ids):
                            nd = all_nodes[nid]
                            X, Y = nd.coordinates[0], nd.coordinates[1]
                            ux = epsC * (Y/H)
                            uy = epsC * (X/L)

                            nset_name = 'Nset_C_%d' % num
                            bc_name   = 'BC_C_%d'   % num
                            region = a.Set(nodes=mesh.MeshNodeArray([nd]), name=nset_name)

                            mdb.models[ModelName].DisplacementBC(
                                name=bc_name, createStepName='Step-1',
                                region=region, u1=ux, u2=0.0, ur3=UNSET,
                                amplitude='Amp-1', fixed=OFF,
                                distributionType=UNIFORM, fieldName='', localCsys=None
                            )
                
                elif BCtype.lower() == "periodic":
                    a = mdb.models[ModelName].rootAssembly
                    model = mdb.models[ModelName]

                    eps = 1.0e-4  

                    left  = sorted(a.sets['Set-left'      ].nodes, key=lambda n: n.coordinates[1])
                    right = sorted(a.sets['Set-right'     ].nodes, key=lambda n: n.coordinates[1])
                    bot   = sorted(a.sets['Set-bottomBody'].nodes, key=lambda n: n.coordinates[0])
                    top   = sorted(a.sets['Set-topBody'   ].nodes, key=lambda n: n.coordinates[0])

                    assert len(left)==len(right) and len(bot)==len(top), "Periodic pairing mismatch: check edge sets."
                    LR_pairs = list(zip(left, right))
                    BT_pairs = list(zip(bot,  top))

                    corner = min(left, key=lambda n: (abs(n.coordinates[0])+abs(n.coordinates[1])))
                    a.Set(nodes=mesh.MeshNodeArray([corner]), name='PBC_Anchor')
                    model.DisplacementBC(name='BC_PBC_Anchor', createStepName='Initial',
                                        region=a.sets['PBC_Anchor'], u1=0.0, u2=0.0, ur3=UNSET,
                                        amplitude=UNSET, fixed=OFF, distributionType=UNIFORM)

                    rpx = a.ReferencePoint(point=(0.0,0.0,0.0)); RPX = 'RP_Per_X'
                    a.Set(referencePoints=(a.referencePoints[rpx.id],), name=RPX)

                    rpy = a.ReferencePoint(point=(0.0,0.0,0.0)); RPY = 'RP_Per_Y'
                    a.Set(referencePoints=(a.referencePoints[rpy.id],), name=RPY)

                    rptb = a.ReferencePoint(point=(0.0,0.0,0.0)); RPTB = 'RP_TB'
                    a.Set(referencePoints=(a.referencePoints[rptb.id],), name=RPTB)

                    rplr = a.ReferencePoint(point=(0.0,0.0,0.0)); RPLR = 'RP_LR'
                    a.Set(referencePoints=(a.referencePoints[rplr.id],), name=RPLR)

                    if caseCmatrix.lower() == "a":
                        model.DisplacementBC(name='BC_RPX',  createStepName='Step-1',
                                            region=a.sets[RPX], u1=eps*L,  u2=UNSET, ur3=UNSET, amplitude='Amp-1')
                        model.DisplacementBC(name='BC_RPTB', createStepName='Step-1',
                                            region=a.sets[RPTB], u1=UNSET, u2=0.0, ur3=UNSET, amplitude='Amp-1')

                        for k,(nL,nR) in enumerate(LR_pairs):
                            nmL = 'PBC_L_%d'%k; nmR = 'PBC_R_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nL]), name=nmL)
                            a.Set(nodes=mesh.MeshNodeArray([nR]), name=nmR)
                            model.Equation(name='Eq_LR_u1_%d'%k,
                                terms=(( 1.0, nmR, 1), (-1.0, nmL, 1), (-1.0, RPX, 1)))
                            model.Equation(name='Eq_LR_u2_%d'%k,
                                terms=(( 1.0, nmR, 2), (-1.0, nmL, 2)))

                        for k,(nB,nT) in enumerate(BT_pairs):
                            nmB = 'PBC_B_%d'%k; nmT = 'PBC_T_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nB]), name=nmB)
                            a.Set(nodes=mesh.MeshNodeArray([nT]), name=nmT)
                            model.Equation(name='Eq_BT_u1_%d'%k,
                                terms=(( 1.0, nmT, 1), (-1.0, nmB, 1)))
                            model.Equation(name='Eq_BT_u2_%d'%k,
                                terms=(( 1.0, nmT, 2), (-1.0, nmB, 2), (-1.0, RPTB, 2)))

                        model.HistoryOutputRequest(name='H_RPX_A',  createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPX])
                        model.HistoryOutputRequest(name='H_RPTB_A', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPTB])

                    elif caseCmatrix.lower() == "b":
                        model.DisplacementBC(name='BC_RPY',  createStepName='Step-1',
                                            region=a.sets[RPY], u1=UNSET, u2=eps*H, ur3=UNSET, amplitude='Amp-1')
                        model.DisplacementBC(name='BC_RPLR', createStepName='Step-1',
                                            region=a.sets[RPLR], u1=0.0,   u2=UNSET, ur3=UNSET, amplitude='Amp-1')

                        for k,(nB,nT) in enumerate(BT_pairs):
                            nmB = 'PBC_B_%d'%k; nmT = 'PBC_T_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nB]), name=nmB)
                            a.Set(nodes=mesh.MeshNodeArray([nT]), name=nmT)
                            model.Equation(name='Eq_BT_u2_%d'%k,
                                terms=(( 1.0, nmT, 2), (-1.0, nmB, 2), (-1.0, RPY, 2)))
                            model.Equation(name='Eq_BT_u1_%d'%k,
                                terms=(( 1.0, nmT, 1), (-1.0, nmB, 1)))

                        for k,(nL,nR) in enumerate(LR_pairs):
                            nmL = 'PBC_L_%d'%k; nmR = 'PBC_R_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nL]), name=nmL)
                            a.Set(nodes=mesh.MeshNodeArray([nR]), name=nmR)
                            model.Equation(name='Eq_LR_u1_%d'%k,
                                terms=(( 1.0, nmR, 1), (-1.0, nmL, 1), (-1.0, RPLR, 1)))
                            model.Equation(name='Eq_LR_u2_%d'%k,
                                terms=(( 1.0, nmR, 2), (-1.0, nmL, 2)))

                        model.HistoryOutputRequest(name='H_RPY_B',  createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPY])
                        model.HistoryOutputRequest(name='H_RPLR_B', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPLR])

                    elif caseCmatrix.lower() == "c":
                        gamma = eps
                        model.DisplacementBC(name='BC_RPX', createStepName='Step-1',
                                            region=a.sets[RPX], u1=0.5*gamma*H, u2=UNSET, ur3=UNSET, amplitude='Amp-1')
                        model.DisplacementBC(name='BC_RPY', createStepName='Step-1',
                                            region=a.sets[RPY], u1=UNSET, u2=0.5*gamma*L, ur3=UNSET, amplitude='Amp-1')

                        for k,(nL,nR) in enumerate(LR_pairs):
                            nmL = 'PBC_L_%d'%k; nmR = 'PBC_R_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nL]), name=nmL)
                            a.Set(nodes=mesh.MeshNodeArray([nR]), name=nmR)
                            model.Equation(name='Eq_LR_u1_%d'%k,
                                terms=(( 1.0, nmR, 1), (-1.0, nmL, 1)))
                            model.Equation(name='Eq_LR_u2_%d'%k,
                                terms=(( 1.0, nmR, 2), (-1.0, nmL, 2), (-1.0, RPY, 2)))

                        for k,(nB,nT) in enumerate(BT_pairs):
                            nmB = 'PBC_B_%d'%k; nmT = 'PBC_T_%d'%k
                            a.Set(nodes=mesh.MeshNodeArray([nB]), name=nmB)
                            a.Set(nodes=mesh.MeshNodeArray([nT]), name=nmT)
                            model.Equation(name='Eq_BT_u1_%d'%k,
                                terms=(( 1.0, nmT, 1), (-1.0, nmB, 1), (-1.0, RPX, 1)))
                            model.Equation(name='Eq_BT_u2_%d'%k,
                                terms=(( 1.0, nmT, 2), (-1.0, nmB, 2)))

                        model.HistoryOutputRequest(name='H_RPX_C', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPX])
                        model.HistoryOutputRequest(name='H_RPY_C', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPY])
                        model.HistoryOutputRequest(name='H_RPX_C', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPTB])
                        model.HistoryOutputRequest(name='H_RPY_C', createStepName='Step-1',
                                                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), region=a.sets[RPLR])
            
            else:
                a = mdb.models[ModelName].rootAssembly
                region = a.sets['Set-top']
                mdb.models[ModelName].DisplacementBC(name='BC-1', createStepName='Step-1', 
                    region=region, u1=0.0, u2=1.0, ur3=0.0, amplitude='Amp-1', fixed=OFF, 
                    distributionType=UNIFORM, fieldName='', localCsys=None)

                region = a.sets['Set-bottom']
                mdb.models[ModelName].DisplacementBC(name='BC-2', createStepName='Step-1', 
                    region=region, u1=0.0, u2=0.0, ur3=0.0, amplitude='Amp-1', fixed=OFF, 
                    distributionType=UNIFORM, fieldName='', localCsys=None)
                
            # ############################################################################################
            # ###################################### Job #################################################
            # ############################################################################################

            userSubroutine = ''

            mdb.Job(name=Job, model=ModelName, description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
                nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
                contactPrint=OFF, historyPrint=OFF, userSubroutine=userSubroutine, scratch='', 
                resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=cpus, 
                activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=cpus)
            
            if (finalRun.lower() == 'inp' or finalRun.lower() == 'input'):
                mdb.jobs[Job].writeInput(consistencyChecking=OFF)
                with open(Job+'.inp', 'a') as f:
                    f.write('**\n**FREQUENCIES:\n')
                    for freq in frequencies:
                        f.write("**" + str(freq) + '\n')
                    f.write('**END FREQUENCIES\n')
            
            elif (finalRun.lower() == 'yes'):
                mdb.jobs[Job].writeInput(consistencyChecking=OFF)
                mdb.jobs[Job].submit(consistencyChecking=OFF)
                mdb.jobs[Job].waitForCompletion()
                with open(Job+'.inp', 'a') as f:
                    f.write('**\n**FREQUENCIES:\n')
                    for freq in frequencies:
                        f.write("**" + str(freq) + '\n')
                    f.write('**END FREQUENCIES\n')
                endtime = time.time()
                print(endtime - starttime, "== time for job", Job)
            
            if stiffMatrix:
                mdb.jobs[Job].writeInput(consistencyChecking=OFF)
                mdb.jobs[Job].waitForCompletion()
        
            
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ###################################### Input for CT Specimen #########################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################            
        
    if (MechanicalModel.lower() == 'fracture' or MechanicalModel.lower() == 'both'):
        
        ModelName = f"Fracture-{latticeType}-{int(nnx)}-{int(fac*100)}{imper}-{dist}-{targeted_disorder}-{idNum}"
        if imper == 'per':
            ModelName = f"Fracture-{latticeType}-{int(nnx)}-per-{idNum}"
        # ModelName = f"Fracture-{latticeType}-{int(nnx)}-A-per-{idNum}"
        Job = ModelName
        
        #############################################################################################
        ######################################### Crack #############################################
        #############################################################################################
        
        xCrS = [-0.1*W, H/2-unitCellSize*0.2]                                         # crack starting point bottomLeft
        if fac == 0.0:                                                                # crack end point topRight
            xCrE = [a0 - 0.2*unitCellSize, H/2+unitCellSize*0.2]
        else:
            xCrE = [a0 - 1.2*fac*unitCellSize, H/2+unitCellSize*0.2]
        
        ridge1 = [-10000, -10000, -10000, -10000]#[0.2*W, (H/2)-(0.1*W), 0.35*W, (H/2)+(0.1*W)]
        ridge2 = [-10000, -10000, -10000, -10000]#[-0.1*W, (H/2)-(0.21*W), 0.25*W, (H/2)+(0.21*W)]
        
        if latticeType.lower() == "fcc" or "square" in latticeType.lower() or latticeType.lower() == "tri":
            CrRegSTAT = [xCrE[0]-(1.6*unitCellSize), xCrE[0]+(3.1*unitCellSize), (H/2)-(2.1*unitCellSize), (H/2)+(2.1*unitCellSize)]
            CrRegMESH = [xCrE[0]-(2.1*unitCellSize), xCrE[0]+(5.6*unitCellSize), (H/2)-(4.1*unitCellSize), (H/2)+(4.1*unitCellSize)]
        elif latticeType.lower() == "kagome" or latticeType.lower() == "hex":
            CrRegSTAT = [xCrE[0]-(3.1*unitCellSize), xCrE[0]+(6.1*unitCellSize), (H/2)-(4.1*unitCellSize), (H/2)+(4.1*unitCellSize)]
            CrRegMESH = [xCrE[0]-(4.6*unitCellSize), xCrE[0]+(10.1*unitCellSize), (H/2)-(7.6*unitCellSize), (H/2)+(7.6*unitCellSize)]
        
        if userMaterial.lower() == "ti" or userMaterial.lower() == "al":
            bcDia     = 0.1875*W
            fixityLoc = [L-W, (H/2)-0.375*W, 0.0]
            loadLoc   = [L-W, (H/2)+0.375*W, 0.0]
        elif userMaterial.lower() == "sic":
            bcDia     = 0.25*W
            fixityLoc = [L-W, (H/2)-0.275*W, 0.0]
            loadLoc   = [L-W, (H/2)+0.275*W, 0.0]

        delNodes = []
        for kk in range(0,len(nodes)):
            xCoord = nodes[kk][1]
            yCoord = nodes[kk][2]
            point = (xCoord,yCoord)
            insideTest = insidePoint(xCrS,xCrE,point)
            if insideTest:
                delNodes.append(nodes[kk][0]-1)
                continue
            insideTest1 = insidePoint(ridge1[:2],ridge1[2:],point)
            if insideTest1:
                delNodes.append(nodes[kk][0]-1)
                continue
            insideTest2 = insidePoint(ridge2[:2],ridge2[2:],point)
            if insideTest2:
                delNodes.append(nodes[kk][0]-1)
                continue

        delNodes = np.array(delNodes, dtype=int)
        nodes = np.delete(nodes,delNodes,0)
        nodesR = np.delete(nodesR,delNodes,0)

        for kk in range(0,len(nodes)):
            nodes[kk][0] = kk+1
            nodesR[kk][0] = kk+1
            
        #############################################################################################
        #################################### Strut Elements #########################################
        #############################################################################################
                
        element = connectivity(latticeType, nodes, geom)
        
        delNodes = []
        for ik in range(0,len(element)):
            x1 = nodesR[int(element[ik][1]-1)][1]
            x2 = nodesR[int(element[ik][2]-1)][1]
            y1 = nodesR[int(element[ik][1]-1)][2]
            y2 = nodesR[int(element[ik][2]-1)][2]
            midPointX = (x1+x2)/2
            midPointY = (y1+y2)/2
            point = (midPointX,midPointY)
            insideTest = insidePoint(xCrS,xCrE,point)
            if insideTest:
                delNodes.append(element[ik][0]-1)
            insideTest1 = insidePoint(ridge1[2:],ridge1[2:],point)
            if insideTest1:
                delNodes.append(nodes[ik][0]-1)
                continue
            insideTest2 = insidePoint(ridge2[2:],ridge2[2:],point)
            if insideTest2:
                delNodes.append(nodes[ik][0]-1)
                continue

        delNodes = np.array(delNodes, dtype=int)
        element = np.delete(element,delNodes,0)
        
        #############################################################################################
        ################################ Radius Calculation #########################################
        #############################################################################################
        
        if outofPlaneThick is None:
            outofPlaneThick = B
        
        length = np.zeros(shape=(len(element),1))
        for ik in range(0,len(element)):
            x1 = nodesR[int(element[ik][1]-1)][1]
            x2 = nodesR[int(element[ik][2]-1)][1]
            y1 = nodesR[int(element[ik][1]-1)][2]
            y2 = nodesR[int(element[ik][2]-1)][2]
            length[ik][0] = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        
        if (crossSection.lower() == 'circ'):
            constants = [4 *relDensity, (L + H - np.pi * sum(length)), 4 * relDensity * (L * H)]
            dia_opt = np.roots(constants)
            dia_est = 2 * relDensity * 4 * L * H / (sum(length) * 2 * np.pi)
            diff_sqr = [(dia_opt[0] - dia_est) ** 2, (dia_opt[1] - dia_est) ** 2]
            index = np.argmin(diff_sqr)

            rad = dia_opt[index]/ 2
            Area = 3.142*rad*rad
            
        elif (crossSection.lower() == 'rect'):
            thick_est = relDensity * L * H * outofPlaneThick / (sum(length) * outofPlaneThick)
            
            if thickness is None:
                    thickness = thick_est
            Area = 1.0*thickness

        if (units.lower() == 'millimeter'):
            tol = 1e-3
        elif (units.lower() == 'meter'):
            tol = 1e-6
        else:
            print('Please enter units scale')

        ############################################################################################
        ####################################### Part Making ########################################
        ############################################################################################

        if (elemType == B32):
            dimenObject = THREE_D
        elif (elemType == B21):
            dimenObject = TWO_D_PLANAR
        elif (elemType == B22):
            dimenObject = TWO_D_PLANAR
        elif (elemType == B23):
            dimenObject = TWO_D_PLANAR
            
        mdb.Model(name=ModelName , modelType=STANDARD_EXPLICIT)

        s = mdb.models[ModelName].ConstrainedSketch(name='__profile__', 
            sheetSize=200.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.setPrimaryObject(option=STANDALONE)

        for ik in range(len(element)):
            s.Line(point1=nodesR[int(element[ik][1]-1)][1:3], point2=nodesR[int(element[ik][2]-1)][1:3])
        p = mdb.models[ModelName].Part(name='Part-1', dimensionality=dimenObject, 
            type=DEFORMABLE_BODY)
        p = mdb.models[ModelName].parts['Part-1']
        p.BaseWire(sketch=s)
        s.unsetPrimaryObject()
        p = mdb.models[ModelName].parts['Part-1']
        session.viewports['Viewport: 1'].setValues(displayedObject=p)

        p = mdb.models[ModelName].parts['Part-1']
        e = p.edges
        edges = e.getByBoundingBox(0.0,0.0,0.0,L,H,0.0)
        p.Set(edges=edges, name='AllPartSet')

        if (sizeVar.lower() == 'yes'):
            p = mdb.models[ModelName].parts['Part-1']
            e = p.edges
            for numEdge in range(0,len(element)):
                x1 = nodesR[int(element[numEdge][1]-1)][1]
                x2 = nodesR[int(element[numEdge][2]-1)][1]
                y1 = nodesR[int(element[numEdge][1]-1)][2]
                y2 = nodesR[int(element[numEdge][2]-1)][2]
                xMid = (x1+x2)/2.0
                yMid = (y1+y2)/2.0
                edges = e.findAt(((xMid, yMid, 0.0), ))
                p.Set(edges=edges, name='edgeElem-'+str(numEdge))
            
        ############################################################################################
        ################################## Material Properties #####################################
        ############################################################################################

        if (userMaterial.lower() == 'al'):
            E, nu = 70000.0, 0.3
            E, nu  = pStrainProperties(E, nu)
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((2.7e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=((
                134.0, 0.0), (134.3, 0.001996429), (134.5, 0.002835692), (135.0, 
                0.004015369), (135.5, 0.005678497), (136.0, 0.008020233), (136.5, 
                0.011313326), (137.0, 0.015938495), (137.5, 0.022426529), (138.0, 
                0.031516535), (138.5, 0.044236464), (139.0, 0.062014286)))
            mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                table=((0.027544286, 0.33333, 0.0), ))
            mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                type=DISPLACEMENT, softening=TABULAR, table=(
                (0.0, 0.0),
                (0.0036, FineElemSizeFT*0.0021),
                (0.0045, FineElemSizeFT*0.00422), 
                (0.0104, FineElemSizeFT*0.0063),
                (0.022,  FineElemSizeFT*0.0084),
                (0.0336, FineElemSizeFT*0.0105),
                (0.0466, FineElemSizeFT*0.0126),
                (0.0599, FineElemSizeFT*0.0147),
                (0.0761, FineElemSizeFT*0.0168),
                (0.095,  FineElemSizeFT*0.01891),
                (0.1173, FineElemSizeFT*0.02101),
                (0.1443, FineElemSizeFT*0.02311),
                (0.1761, FineElemSizeFT*0.02521),
                (0.2144, FineElemSizeFT*0.02731),
                (0.2578, FineElemSizeFT*0.02928), 
                (0.3029, FineElemSizeFT*0.03098),
                (0.3457, FineElemSizeFT*0.03236),
                (0.3866, FineElemSizeFT*0.03335),
                (0.4301, FineElemSizeFT*0.034),
                (0.4665, FineElemSizeFT*0.03446),
                (1.0,    FineElemSizeFT*0.03547)))

        elif (userMaterial.lower() == 'sic'):
            E, nu = 410000, 0.14
            E, nu  = pStrainProperties(E, nu)
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((550, 0.0),
                (550.1, 0.00001)))
            mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                table=((0.00001, 0.333333, 0.0), ))
            mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                type=DISPLACEMENT, softening=TABULAR, table=(
                (0.0, 0.0),
                (0.2, FineElemSizeUT*0.00001),
                (0.4, FineElemSizeUT*0.00002), 
                (0.6, FineElemSizeUT*0.00003),
                (0.8, FineElemSizeUT*0.00004),
                (1.0, FineElemSizeUT*0.00005)))

        elif (userMaterial.lower() == 'ti'):
            E, nu = 123000, 0.3
            E, nu  = pStrainProperties(E, nu)
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((4.43e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((E, nu), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((932,         0),
                (947.411802,    0.003453491),
                (957.4512331, 0.006906981),
                (966.1307689, 0.010360472),
                (974.030469, 0.013813962),
                (981.3967087, 0.017267453),
                (988.3639577, 0.020720943),
                (995.0160058, 0.024174434),
                (1001.409616, 0.027627924),
                (1007.585541, 0.031081415),
                (1013.574312, 0.034534905),
                (1019.399572, 0.037988396),
                (1025.08011, 0.041441886),
                (1030.631179, 0.044895377),
                (1036.065381, 0.048348867),
                (1041.393285, 0.051802358),
                (1046.623866, 0.055255848),
                (1051.764831, 0.058709339),
                (1056.822861, 0.062162829),
                (1061.803799, 0.06561632),
                (1066.712789, 0.06906981),
                (1071.554397, 0.072523301),
                (1076.332693, 0.075976791),
                (1081.05133, 0.079430282),
                (1085.713601, 0.082883772),
                (1090.322488, 0.086337263),
                (1094.880702, 0.089790753),
                (1099.39072, 0.093244244),
                (1103.854808, 0.096697734),
                (1108.275052, 0.100151225),
                (1112.653372, 0.103604715)))
            mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                table=((0.102268174, 0.333333, 0.0), ))
            mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                type=DISPLACEMENT, softening=TABULAR, table=(
                (0.0, 0.0),
                (0.2, FineElemSizeFT*0.0001),
                (0.4, FineElemSizeFT*0.0002), 
                (0.6, FineElemSizeFT*0.0003),
                (0.8, FineElemSizeFT*0.0004),
                (1.0, FineElemSizeFT*0.0005)))

        if (sizeVar.lower() == 'yes'):
            relativeDensityUpdated = 10000
            while relativeDensityUpdated < (relDensity - 0.001) or relativeDensityUpdated > (relDensity + 0.001):

                lowerLim = (1.0 - beta) * thickness
                upperLim = (1.0 + beta) * thickness

                thick = np.random.uniform(lowerLim, upperLim, len(element))

                latticeVolume = np.dot(length[:].T, thick*outofPlaneThick)
                
                relativeDensityUpdated = latticeVolume/(L * H * outofPlaneThick)

                if relativeDensityUpdated > relDensity:
                    thickness = thickness - 0.001
                else:
                    thickness = thickness + 0.001
            
            for numEdge in range(0,len(element)):       
                mdb.models[ModelName].RectangularProfile(name='Rect'+str(numEdge), a=outofPlaneThick, b=thick[numEdge])

                mdb.models[ModelName].BeamSection(name='BeamSec'+str(numEdge), integration=DURING_ANALYSIS, 
                    poissonRatio=0.0, profile='Rect'+str(numEdge), material=userMaterial, 
                    temperatureVar=LINEAR, consistentMassMatrix=False)

                p = mdb.models[ModelName].parts['Part-1']
                region = p.sets['edgeElem-'+str(numEdge)]
                p.SectionAssignment(region=region, sectionName='BeamSec'+str(numEdge), offset=0.0, 
                    offsetType=MIDDLE_SURFACE, offsetField='', 
                    thicknessAssignment=FROM_SECTION)

                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))

        elif (sizeVar.lower() == 'no'):
            if (crossSection.lower() == 'circ'):
                mdb.models[ModelName].CircularProfile(name='Circ', r=rad)

                mdb.models[ModelName].BeamSection(name='BeamSec', integration=DURING_ANALYSIS, 
                    poissonRatio=0.0, profile='Circ', material=userMaterial, 
                    temperatureVar=LINEAR, consistentMassMatrix=False)

                p = mdb.models[ModelName].parts['Part-1']
                region = p.sets['AllPartSet']
                p.SectionAssignment(region=region, sectionName='BeamSec', offset=0.0, 
                    offsetType=MIDDLE_SURFACE, offsetField='', 
                    thicknessAssignment=FROM_SECTION)

                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))

            elif (crossSection.lower() == 'rect'):

                mdb.models[ModelName].RectangularProfile(name='Rect', a=outofPlaneThick, b=thickness)

                mdb.models[ModelName].BeamSection(name='BeamSec', integration=DURING_ANALYSIS, 
                    poissonRatio=0.0, profile='Rect', material=userMaterial, 
                    temperatureVar=LINEAR, consistentMassMatrix=False)

                p = mdb.models[ModelName].parts['Part-1']
                region = p.sets['AllPartSet']
                p.SectionAssignment(region=region, sectionName='BeamSec', offset=0.0, 
                    offsetType=MIDDLE_SURFACE, offsetField='', 
                    thicknessAssignment=FROM_SECTION)

                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
            
        ###########################################################################################
        ####################################### Assembly ##########################################
        ###########################################################################################

        a = mdb.models[ModelName].rootAssembly
        a.DatumCsysByDefault(CARTESIAN)
        p = mdb.models[ModelName].parts['Part-1']
        a.Instance(name='Part-1-1', part=p, dependent=OFF)

        RFid = mdb.models[ModelName].rootAssembly.ReferencePoint(point=(loadLoc[0], loadLoc[1], loadLoc[2])).id
        r1 = mdb.models[ModelName].rootAssembly.referencePoints
        refPoints1=(r1[RFid], )
        a.Set(referencePoints=refPoints1, name='load')

        RFid = mdb.models[ModelName].rootAssembly.ReferencePoint(point=(fixityLoc[0], fixityLoc[1], fixityLoc[2])).id
        r1 = mdb.models[ModelName].rootAssembly.referencePoints
        refPoints1=(r1[RFid], )
        a.Set(referencePoints=refPoints1, name='fixity')

        ############################################################################################
        ######################################## Step ##############################################
        ############################################################################################

        mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
            timePeriod=STEP_TIME)
            
        # ############################################################################################
        # ######################################## Mesh ##############################################
        # ############################################################################################

        a = mdb.models[ModelName].rootAssembly
        e = a.instances['Part-1-1'].edges
        edges = e.getByBoundingBox(0.0,0.0,0.0,L,H,0.0)
        elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
        pickedRegions =(edges, )
        a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
        partInstances = (a.instances['Part-1-1'], )
        a.seedPartInstance(regions=partInstances, size=CoarseElemSizeFT, deviationFactor=0.1, 
            minSizeFactor=0.1)
        a = mdb.models[ModelName].rootAssembly
        partInstances =(a.instances['Part-1-1'], )
        a.generateMesh(regions=partInstances)
        
        a = mdb.models[ModelName].rootAssembly
        e = a.instances['Part-1-1'].edges
        edges = e.getByBoundingBox(CrRegMESH[0], CrRegMESH[2], 0.0, CrRegMESH[1], CrRegMESH[3], 0.0)
        elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
        pickedRegions =(edges, )
        a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
        partInstances = (a.instances['Part-1-1'], )
        a.seedEdgeBySize(edges=edges, size=FineElemSizeFT, deviationFactor=0.1, 
            minSizeFactor=0.1, constraint=FINER)
        a = mdb.models[ModelName].rootAssembly
        partInstances =(a.instances['Part-1-1'], )
        a.generateMesh(regions=partInstances)

        all_nodes = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].nodes
        fix_nodes = []
        load_nodes = []
        for n in all_nodes:
            xcoord = n.coordinates[0]
            ycoord = n.coordinates[1]
            if in_circle(fixityLoc[0], fixityLoc[1], 0.5*bcDia, xcoord, ycoord):
                fix_nodes.append(n)
            if in_circle(loadLoc[0], loadLoc[1], 0.5*bcDia, xcoord, ycoord):
                load_nodes.append(n)
            
        ln = mesh.MeshNodeArray(fix_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-fix')

        ln = mesh.MeshNodeArray(load_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-load')
        
        all_nodes = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].nodes
        cracktip_node_labels = []
        for n in all_nodes:
            xcoord = n.coordinates[0]
            ycoord = n.coordinates[1]
            if xcoord > CrRegSTAT[0] and xcoord < CrRegSTAT[1] and ycoord > CrRegSTAT[2] and ycoord < CrRegSTAT[3]:
                cracktip_node_labels.append(n.label)
        
#        print(len(cracktip_node_labels))      
#        ln = mesh.MeshNodeArray(cracktip_node_labels)
#        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='temp')
        
        all_elements = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].elements
        cracktip_elements = []
        for e in all_elements:
            connected_node1 = e.connectivity[0]+1
            connected_node2 = e.connectivity[1]+1
            if connected_node1 in cracktip_node_labels or connected_node2 in cracktip_node_labels:
                cracktip_elements.append(e)
        
        e1 = mesh.MeshElementArray(cracktip_elements)
        mdb.models[ModelName].rootAssembly.Set(elements=e1 , name='Set-cracktip')

        # ############################################################################################
        # ################################## Output Requests #########################################
        # ############################################################################################

        if (userMaterial.lower() == 'material-1'):    
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'PE', 'LE', 'UT', 'RF', 'EVOL', 'SDV'), numIntervals=FieldOut_frames)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)

        else:
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'PE', 'LE', 'UT', 'RF', 'SDEG', 'DMICRT', 'STATUS'), numIntervals=FieldOut_frames)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), numIntervals=HistOut_frames, region=regionDef, 
                sectionPoints=DEFAULT, rebar=EXCLUDE)

            regionDef=mdb.models[ModelName].rootAssembly.sets['load']
            mdb.models[ModelName].HistoryOutputRequest(name='H-Output-2', createStepName='Step-1', 
                variables=('S', 'E', 'PE', 'LE', 'U', 'RF'), numIntervals=HistOut_frames, region=regionDef, 
                sectionPoints=DEFAULT, rebar=EXCLUDE)
            
            regionDef=mdb.models[ModelName].rootAssembly.sets['Set-cracktip']
            mdb.models[ModelName].HistoryOutputRequest(name='H-Output-3', createStepName='Step-1', 
                variables=('STATUS', ), numIntervals=HistOut_frames, region=regionDef, 
                sectionPoints=DEFAULT, rebar=EXCLUDE)
                
        # ############################################################################################
        # #################################### Interactions ##########################################
        # ############################################################################################

        a = mdb.models[ModelName].rootAssembly
        region1=a.sets['fixity']
        region2=a.sets['Set-fix']
        mdb.models[ModelName].Coupling(name='couplingFixity', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, ur3=ON)    

        a = mdb.models[ModelName].rootAssembly
        region1=a.sets['load']
        region2=a.sets['Set-load']
        mdb.models[ModelName].Coupling(name='couplingLoad', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, ur3=ON)

        # if (elemType == B32):
            # a = mdb.models[ModelName].rootAssembly
            # mdb.models[ModelName].ContactProperty('IntProp-1')
            # mdb.models[ModelName].interactionProperties['IntProp-1'].NormalBehavior(
                # pressureOverclosure=HARD, allowSeparation=ON, 
                # constraintEnforcementMethod=DEFAULT)
            # mdb.models[ModelName].interactionProperties['IntProp-1'].TangentialBehavior(
                # formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
                # pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
                # 0.3, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
                # fraction=0.005, elasticSlipStiffness=None)
            # mdb.models[ModelName].ContactExp(name='Int-1', createStepName='Initial')
            # mdb.models[ModelName].interactions['Int-1'].includedPairs.setValuesInStep(
                # stepName='Initial', useAllstar=ON)
            # mdb.models[ModelName].interactions['Int-1'].contactPropertyAssignments.appendInStep(
                # stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp-1'), ))

        # ############################################################################################
        # #################################### Loading ###############################################
        # ############################################################################################

        if sm_amp:
            mdb.models[ModelName].SmoothStepAmplitude(name='Amp-1', 
                timeSpan=STEP, data=((0.0, 0.0), (STEP_TIME, strainAppFT*H)))
        else:
            mdb.models[ModelName].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
            smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (STEP_TIME, strainAppFT*H)))

        a = mdb.models[ModelName].rootAssembly
        region = a.sets['load']
        mdb.models[ModelName].DisplacementBC(name='BC-1', createStepName='Step-1', 
            region=region, u1=0.0, u2=1.0, amplitude='Amp-1', fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)

        region = a.sets['fixity']
        mdb.models[ModelName].DisplacementBC(name='BC-2', createStepName='Step-1', 
            region=region, u1=0.0, u2=0.0, amplitude='Amp-1', fixed=OFF, 
            distributionType=UNIFORM, fieldName='', localCsys=None)
            
        # ############################################################################################
        # ###################################### Job #################################################
        # ############################################################################################

        userSubroutine = ''

        mdb.Job(name=Job, model=ModelName, description='', type=ANALYSIS, 
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
            memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
            nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
            contactPrint=OFF, historyPrint=OFF, userSubroutine=userSubroutine, scratch='', 
            resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=cpus, 
            activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=cpus)
        
        if (finalRun.lower() == 'inp' or finalRun.lower() == 'input'):
            mdb.jobs[Job].writeInput(consistencyChecking=OFF)
            with open(Job+'.inp', 'a') as f:
                f.write('**\n**FREQUENCIES:\n')
                for freq in frequencies:
                    f.write("**" + str(freq) + '\n')
                f.write('**END FREQUENCIES\n')
        
        elif (finalRun.lower() == 'yes'):
            mdb.jobs[Job].writeInput(consistencyChecking=OFF)
            mdb.jobs[Job].submit(consistencyChecking=OFF)
            mdb.jobs[Job].waitForCompletion()
            with open(Job+'.inp', 'a') as f:
                f.write('**\n**FREQUENCIES:\n')
                for freq in frequencies:
                    f.write("**" + str(freq) + '\n')
                f.write('**END FREQUENCIES\n')
            endtime = time.time()
            print(endtime - starttime, "== time for job", Job)
