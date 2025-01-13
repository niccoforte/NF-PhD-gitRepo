from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from numpy import *
import math
import os
import sys
executeOnCaeStartup()

############################################################################################
####################################### INPUT ##############################################
############################################################################################

unitCellSize = 10.0                         # Strut length
latticeType = 'FCC'                         # 'FCC', 'tri', 'hex', 'kagome'
MechanicalModel = 'both'                    # 'fracture', 'ductile', 'both'
userMaterial = 'ti'                         # 'al', 'sic', 'ti'
nnx = 10                                    # number of Unit cells in X direction
relDensity = 0.2                            # relative density
distribution = 'uniform'                    # 'uniform', 'normal', 'exponential'
crossSection = 'rect'

finalRun = 'yes'
numberOfRuns = 1
initialJob = 1
cpus = 12
FieldOut_frames = 10
HistOut_frames = 200

nodeVar = 'no'                               # distortion
fac = 0.2
sizeVar = 'no'
beta = 0.2

stiffMatrix = False
UTval = False

cmdIN = sys.argv[8:]
if len(cmdIN) > 0:
    latticeType = str(cmdIN[0])
    dis = str(cmdIN[1])
    nnx = int(cmdIN[2])
    unitCellSize = float(cmdIN[3])
    MechanicalModel = str(cmdIN[4])
    userMaterial = str(cmdIN[5])
    relDensity = float(cmdIN[6])
    initialJob = int(cmdIN[7])
    numberOfRuns = int(cmdIN[8])
    cpus = int(cmdIN[9])
    FieldOut_frames = int(cmdIN[10])
    HistOut_frames = int(cmdIN[11])
    
    path = str(cmdIN[12])
    
    stiffMatrix = bool(int(cmdIN[13]))
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
    
    if path.lower() == "val":
        pDir = "C:\\Users\\exy053\\Documents\\validation\\"+str(int(unitCellSize))+"\\"+str(relDensity)
    elif path.lower() == "psc":
        pDir = "C:\\Users\\exy053\\Documents\\PerSizeConv3\\"+str(int(unitCellSize))
    elif path.lower() == "dsc":
        pDir = "C:\\Users\\exy053\\Documents\\disConv\\"+latticeType
    elif path.lower() == "sic":
        pDir = "C:\\Users\\exy053\\Documents\\SiC"
    elif path.lower() == "rd":
        pDir = "C:\\Users\\exy053\\Documents\\relD\\"+str(relDensity)
    elif path.lower() == "mc":
        pDir = "C:\\Users\\exy053\\Documents\\ModelChanges"
    else:
        pDir = "C:\\Users\\exy053\\Documents\\" + str(path)

if stiffMatrix:
    MechanicalModel = 'ductile'
    pDir = "C:\\Users\\exy053\\Documents\\stiffMatrix"
    finalRun = 'no'
    UTval = False

if UTval:
    latticeType = "tri"
    nnx = 20
    unitCellSize = 10.0
    MechanicalModel = 'ductile'
    finalRun = 'no'
    userMaterial = 'al'
    nodeVar = 'no'
    sizeVar = 'no'
    pDir = "C:\\Users\\exy053\\Documents\\al\\"
    stiffMatrix = False

os.chdir(pDir)

STEP_TIME = 1E-1
sm_amp = False
if userMaterial.lower() == "ti":                # lower amp = higher Kjic
    if latticeType.lower() == "fcc":            # amplitude (uniax = strainAppUT * H; FT = stainAppFT * H)
        strainAppUT = 0.035                                               # FINAL 30 - 0.035
        strainAppFT = 0.050                                               # FINAL 30 - 0.05
    elif latticeType.lower() == "tri":
        strainAppUT = 0.100  #0.100  #30-0.1                                      # FINAL 30 - 0.100
        strainAppFT = 0.080  #100-0.025 80-0.05 50-0.1 30-0.08            # FINAL 30 - 0.080
    elif latticeType.lower() == "kagome":
        strainAppUT = 0.072  #26-0.065 20-0.072                           # FINAL 20 - 0.072
        strainAppFT = 0.067  #70-0.025 26-0.052 20-0.067                  # FINAL 20 - 0.067
    elif latticeType.lower() == "hex":
        strainAppUT = 0.100  #14-0.15 24-0.1 34-0.045                     # FINAL 30 - 0.05
        strainAppFT = 0.032  #20-0.05 50-0.032                            # FINAL 30 - 0.05
elif userMaterial.lower() == "sic":
    if latticeType.lower() == "fcc":
        strainAppUT = 0.00125
        strainAppFT = 0.00125
    elif latticeType.lower() == "tri":
        strainAppUT = 0.002
        strainAppFT = 0.002
    elif latticeType.lower() == "kagome":
        strainAppUT = 0.002
        strainAppFT = 0.002
    elif latticeType.lower() == "hex":
        strainAppUT = 0.002
        strainAppFT = 0.002
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

if latticeType.lower() == "fcc":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/5.0          # minimum coarse element size
    FineElemSizeUT   = unitCellSize/5.0          # mimimum fine element size
    CoarseElemSizeFT = unitCellSize/5.0
    FineElemSizeFT   = unitCellSize/15.0
elif latticeType.lower() == "tri":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/5.0
    FineElemSizeUT   = unitCellSize/5.0
    CoarseElemSizeFT = unitCellSize/2.0
    FineElemSizeFT   = unitCellSize/15.0
elif latticeType.lower() == "kagome":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/2.0          # minimum coarse element size
    FineElemSizeUT   = unitCellSize/10.0         # mimimum fine element size
    CoarseElemSizeFT = unitCellSize/2.0
    FineElemSizeFT   = unitCellSize/30.0
elif latticeType.lower() == "hex":
    BracketElemSize  = unitCellSize/1.0
    CoarseElemSizeUT = unitCellSize/5.0          # minimum coarse element size
    FineElemSizeUT   = unitCellSize/20.0         # mimimum fine element size
    CoarseElemSizeFT = unitCellSize/2.0
    FineElemSizeFT   = unitCellSize/45.0

############################################################################################
############################################################################################
############################################################################################
	
def node(latticeType, L, H, nnx, nny, totalNodes, totalBracketNodes, fac, distribution):
    if latticeType.lower() == "fcc":
        unitX = L / nnx
        unitY = H / nny
        
        nodes = zeros(shape=(totalNodes,3)) 
        nodesR = zeros(shape=(totalNodes,3))
        bracket_nodes = zeros(shape=(totalBracketNodes, 3))

        count = 0
        x = 0
        y = 0
        for i in range(1, nny+2):
            for j in range(1, nnx+2):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = 0
       
        x = unitX / 2.0
        y = unitY / 2.0
        for i in range(1, nny+1):
            for j in range(1, nnx+1):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = unitX / 2.0
        
        x = -2.0*unitX
        y = H + unitY
        for i in range(1, 4):
            for j in range(1, nnx+6):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1

            y = y + unitY
            x = -2.0*unitX
            
        x = -2.0*unitX
        y = -unitY 
        for i in range(1, 4):
            for j in range(1, nnx+6):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1

            y = y - unitY
            x = -2.0*unitX
        
        x = -1.5*unitX
        y = H + (0.5*unitY)
        for i in range(1, 4):
            for j in range(1, nnx+5):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1

            y = y + unitY
            x = -1.5*unitX
        
        x = -1.5*unitX
        y = -0.5*unitY
        for i in range(1, 4):
            for j in range(1, nnx+5):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1

            y = y - unitY
            x = -1.5*unitX
        
        Dnodes_brackets = []
        Dcoords = []
        Dcoordsr = [-unitX, L+unitX, -2.0*unitX, L+(2.0*unitX), 0, H, -unitY, H+unitY, -1.5*unitX, L+1.5*unitX, -0.5*unitY, H+0.5*unitY]
        for i in Dcoordsr:
            Dcoords.append(round(i,2))
        for node in bracket_nodes:
            if round(node[1],2) in Dcoords[:2]:
                if round(node[2],2) in Dcoords[4:6]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
            if round(node[1],2) in Dcoords[2:4]:
                if round(node[2],2) in Dcoords[4:8]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
            if round(node[1],2) in Dcoords[8:10]:
                if round(node[2],2) in Dcoords[10:]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
        
        #print(nodes)
        #print(bracket_nodes)
        
        bracket_nodes = delete(bracket_nodes, Dnodes_brackets, 0)

        xCoord = nodes[:, 1]
        yCoord = nodes[:, 2]
        bottomNodes = argwhere(yCoord == 0)
        topNodes = argwhere(yCoord == H)
        leftNodes = argwhere(xCoord == 0)
        rightNodes = argwhere(xCoord == L)
        
        boundaryNodes = concatenate((bottomNodes, leftNodes, topNodes, rightNodes))
        #boundaryNodes = concatenate((bottomNodes, topNodes))
        boundaryNodes, inx = unique(boundaryNodes,return_index=True)
        boundaryNodesInx = nodes[boundaryNodes, 0]
        
        nodeIndex = nodes[:, 0]
        nonboundaryNodes = setdiff1d(nodeIndex, boundaryNodesInx)

        nonboundaryCoordX = nodes[nonboundaryNodes.astype(int)-1, 1]
        nonboundaryCoordY = nodes[nonboundaryNodes.astype(int)-1, 2]
        
        delta = 0.5 * sqrt(unitX * unitX + unitY * unitY) * fac
        if (distribution.lower() == 'uniform'):
            randX = random.uniform(-delta, delta, len(nonboundaryNodes))
            randY = random.uniform(-delta, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'normal'):
            randX = random.normal(0.0, delta, len(nonboundaryNodes))
            randY = random.normal(0.0, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'exponential'):
            #delta = 2.5
            randX = random.exponential(1/delta, len(nonboundaryNodes))
            randY = random.exponential(1/delta, len(nonboundaryNodes))
        

        nonboundaryCoordX = nonboundaryCoordX + randX
        nonboundaryCoordY = nonboundaryCoordY + randY
        
        nodesR[:,0] = nodes[:,0]
        nodesR[:,1] = nodes[:,1]
        nodesR[:,2] = nodes[:,2]
        nodesR[nonboundaryNodes.astype(int)-1, 1] = nonboundaryCoordX
        nodesR[nonboundaryNodes.astype(int)-1, 2] = nonboundaryCoordY
    
    elif latticeType.lower() == "tri":
        unitX = 0.5 * sqrt(3) * unitCellSize
        unitY = unitCellSize

        nodes = zeros(shape=(totalNodes, 3))
        nodesR = zeros(shape=(totalNodes, 3))
        bracket_nodes = zeros(shape=(totalBracketNodes, 3))
        
        count = 0
        x = 0
        y = 0
        for i in range(1, nny + 2):
            for j in range(1, int(round(nnx / 2.000001)) + 2):
                nodes[count][0] = count + 1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + 2 * unitX
                count = count + 1

            y = y + unitY
            x = 0
        
        x = unitX
        y = unitY / 2.0
        for i in range(1, nny + 1):
            for j in range(1, int(round(nnx / 2.000001) + 1)):
                nodes[count][0] = count + 1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + 2 * unitX
                count = count + 1

            y = y + unitY
            x = unitX
        
        x = -2.0*unitX
        y = H + unitY
        for i in range(1, 4):
            for j in range(1, int(round(nnx / 2.000001)) + 4):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1

            y = y + unitY
            x = -2.0*unitX
        
        x = -unitX
        y = H + (0.5*unitY)
        for i in range(1, 4):
            for j in range(1, int(round(nnx / 2.000001) + 3)):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1

            y = y + unitY
            x = -unitX
            
        x = -2.0*unitX
        y = -unitY 
        for i in range(1, 4):
            for j in range(1, int(round(nnx / 2.000001)) + 4):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1

            y = y - unitY
            x = -2.0*unitX
        
        x = -unitX
        y = -0.5*unitY
        for i in range(1, 4):
            for j in range(1, int(round(nnx / 2.000001) + 3)):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1

            y = y - unitY
            x = -unitX
        
        x = -unitX
        y = H + (3.0*unitY)
        for j in range(1, int(round(nnx / 2.000001) + 3)):
            bracket_nodes[count - totalNodes][0] = count + 1
            bracket_nodes[count - totalNodes][1] = x
            bracket_nodes[count - totalNodes][2] = y
            x = x + 2.0*unitX
            count = count + 1
        
        x = -unitX
        y = -(3.0*unitY)
        for j in range(1, int(round(nnx / 2.000001) + 3)):
            bracket_nodes[count - totalNodes][0] = count + 1
            bracket_nodes[count - totalNodes][1] = x
            bracket_nodes[count - totalNodes][2] = y
            x = x + 2.0*unitX
            count = count + 1
        
        Dnodes_brackets = []
        Dcoords = [round(-2.0*unitX,2), round(L+(2.0*unitX),2), round(-unitY,2), round(H+unitY,2)]
        for node in bracket_nodes:
            if round(node[1],2) in Dcoords[:2]:
                if round(node[2],2) in Dcoords[2:]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
        
        #print(nodes)
        #print(bracket_nodes)
        
        bracket_nodes = delete(bracket_nodes, Dnodes_brackets, 0)
        
        xCoord = nodes[:, 1]
        yCoord = nodes[:, 2]
        bottomNodes = argwhere(yCoord == 0)
        topNodes = argwhere((yCoord < H + 1e-3) & (yCoord > H - 1e-3))
        leftNodes = argwhere(xCoord == 0)
        rightNodes = argwhere((xCoord < L + 1e-3) & (xCoord > L - 1e-3))

        boundaryNodes = concatenate((bottomNodes, leftNodes, topNodes, rightNodes))
        # boundaryNodes = concatenate((bottomNodes, topNodes))
        boundaryNodes, inx = unique(boundaryNodes, return_index=True)
        boundaryNodesInx = nodes[boundaryNodes, 0]

        nodeIndex = nodes[:, 0]
        nonboundaryNodes = setdiff1d(nodeIndex, boundaryNodesInx)

        nonboundaryCoordX = nodes[nonboundaryNodes.astype(int) - 1, 1]
        nonboundaryCoordY = nodes[nonboundaryNodes.astype(int) - 1, 2]

        delta = unitCellSize * fac
        if (distribution.lower() == 'uniform'):
            randX = random.uniform(-delta, delta, len(nonboundaryNodes))
            randY = random.uniform(-delta, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'normal'):
            randX = random.normal(0.0, delta, len(nonboundaryNodes))
            randY = random.normal(0.0, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'exponential'):
            # delta = 2.5
            randX = random.exponential(1 / delta, len(nonboundaryNodes))
            randY = random.exponential(1 / delta, len(nonboundaryNodes))

        nonboundaryCoordX = nonboundaryCoordX + randX
        nonboundaryCoordY = nonboundaryCoordY + randY

        nodesR[:, 0] = nodes[:, 0]
        nodesR[:, 1] = nodes[:, 1]
        nodesR[:, 2] = nodes[:, 2]
        nodesR[nonboundaryNodes.astype(int) - 1, 1] = nonboundaryCoordX
        nodesR[nonboundaryNodes.astype(int) - 1, 2] = nonboundaryCoordY

    elif latticeType.lower() == "kagome":
        unitX = float(unitCellSize)
        unitY = sqrt(3) * float(unitCellSize)
        
        nodes = zeros(shape=(totalNodes,3))
        nodesR = zeros(shape=(totalNodes,3))
        bracket_nodes = zeros(shape=(totalBracketNodes, 3))

        count = 0
        x = 0
        y = 0
        for i in range(1, nny+2):
            for j in range(1, 2*nnx+1):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = 0
     
        x = 1.5*unitX
        y = unitY / 2.0
        for i in range(1, int(math.ceil(nny/2.0)+1)):
            for j in range(1, nnx):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + 2*unitX
                count = count + 1
            
            y = y + 2.0*unitY
            x = 1.5*unitX

        x = 0.5*unitX
        y = 1.5*unitY
        for i in range(1, int(floor(nny/2)+1)):
            for j in range(1, nnx+1):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + 2*unitX
                count = count + 1
            
            y = y + 2.0*unitY
            x = unitX/2.0
        
        x = -2.0 * unitX
        y = H + unitY
        for i in range(1, 4):
            for j in range(1, 2*nnx+5):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = -2.0 * unitX
     
        x = (-2.0 + 0.5)*unitX
        y = H + 0.5*unitY
        for i in range(1, 3):
            for j in range(1, nnx+3):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1
            
            y = y + 2.0*unitY
            x = (-2.0 + 0.5)*unitX

        x = (-2.0 + 1.5)*unitX
        y = H + 1.5*unitY
        for i in range(1, 2):
            for j in range(1, nnx+2):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1
            
            y = y + 2.0*unitY
            x = (-2.0 + 1.5)*unitX
        
        x = -2.0 * unitX
        y = -unitY
        for i in range(1, 4):
            for j in range(1, 2*nnx+5):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y - unitY
            x = -2.0 * unitX
     
        x = (-2.0 + 0.5)*unitX
        y = -0.5*unitY
        for i in range(1, 3):
            for j in range(1, nnx+3):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1
            
            y = y - 2.0*unitY
            x = (-2.0 + 0.5)*unitX

        x = (-2.0 + 1.5)*unitX
        y = -1.5*unitY
        for i in range(1, 2):
            for j in range(1, nnx+2):
                bracket_nodes[count - totalNodes][0] = count + 1
                bracket_nodes[count - totalNodes][1] = x
                bracket_nodes[count - totalNodes][2] = y
                x = x + 2.0*unitX
                count = count + 1
            
            y = y - 2.0*unitY
            x = (-2.0 + 1.5)*unitX
        
        Dnodes_brackets = []
        DcoordsX = []
        DcoordsY = []
        DcoordsXr = [-unitX, -2.0*unitX, -0.5*unitX, -1.5*unitX, L+unitX, L+(2.0*unitX), L+(0.5*unitX), L+(1.5*unitX)]
        DcoordsYr = [-0.5*unitY, H+0.5*unitY, -unitY, H+unitY, -1.5*unitY, H+1.5*unitY, -2.0*unitY, H+(2.0*unitY), 
                    -2.5*unitY, H+(2.5*unitY), -3.0*unitY, H+(3.0*unitY)]
        for i in DcoordsXr:
            DcoordsX.append(round(i,2))
        for i in DcoordsYr:
            DcoordsY.append(round(i,2))
        for node in bracket_nodes:
            if round(node[2],2) in DcoordsY[:2]:
                if round(node[1],2) in DcoordsX[2:4] or node[1] in DcoordsX[6:]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
            if round(node[2],2) in DcoordsY[2:4]:
                if round(node[1],2) in DcoordsX[:2] or node[1] in DcoordsX[4:6]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
            if round(node[2],2) in DcoordsY[6:8]:
                if round(node[1],2) == DcoordsX[1] or node[1] == DcoordsX[5]:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
        
        bracket_nodes = delete(bracket_nodes, Dnodes_brackets, 0)
        
        xCoord = nodes[:, 1]
        yCoord = nodes[:, 2]
        bottomNodes = argwhere(yCoord == 0)
        topNodes = argwhere((yCoord < H+1e-3) & (yCoord > H-1e-3))
        leftNodes = argwhere(xCoord == 0)
        rightNodes = argwhere((xCoord < L+1e-3) & (xCoord > L-1e-3))
        
        boundaryNodes = concatenate((bottomNodes, leftNodes, topNodes, rightNodes))
        #boundaryNodes = concatenate((bottomNodes, topNodes))
        boundaryNodes, inx = unique(boundaryNodes,return_index=True)
        boundaryNodesInx = nodes[boundaryNodes, 0]
        
        nodeIndex = nodes[:, 0]
        nonboundaryNodes = setdiff1d(nodeIndex, boundaryNodesInx)

        nonboundaryCoordX = nodes[nonboundaryNodes.astype(int)-1, 1]
        nonboundaryCoordY = nodes[nonboundaryNodes.astype(int)-1, 2]
        
        delta = unitCellSize * fac
        if (distribution.lower() == 'uniform'):
            randX = random.uniform(-delta, delta, len(nonboundaryNodes))
            randY = random.uniform(-delta, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'normal'):
            randX = random.normal(0.0, delta, len(nonboundaryNodes))
            randY = random.normal(0.0, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'exponential'):
            #delta = 2.5
            randX = random.exponential(1/delta, len(nonboundaryNodes))
            randY = random.exponential(1/delta, len(nonboundaryNodes))
        

        nonboundaryCoordX = nonboundaryCoordX + randX
        nonboundaryCoordY = nonboundaryCoordY + randY
        
        nodesR[:,0] = nodes[:,0]
        nodesR[:,1] = nodes[:,1]
        nodesR[:,2] = nodes[:,2]
        nodesR[nonboundaryNodes.astype(int)-1, 1] = nonboundaryCoordX
        nodesR[nonboundaryNodes.astype(int)-1, 2] = nonboundaryCoordY
    
    elif latticeType.lower() == "hex":
        unitX = sqrt(3)*unitCellSize
        unitY = 2*unitCellSize
            
        nodes = zeros(shape=(totalNodes,3))
        nodesR = zeros(shape=(totalNodes,3))
        bracket_nodes = zeros(shape=(totalBracketNodes, 3))


        count = 0
        x = 0.5*unitX
        y = 0
        for i in range(1, int(math.ceil(nny/2.0)+1)):
            for j in range(1, nnx+1):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = 0.5*unitX
            for j in range(1,nnx+1):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitCellSize
            x = 0.5*unitX

        x = 0
        y = unitCellSize/2.0
        for i in range(1, int(math.ceil(nny/2.0)+1)):
            for j in range(1, nnx+2):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitCellSize
            x = 0 
            for j in range(1, nnx+2):
                nodes[count][0] = count+1
                nodes[count][1] = x
                nodes[count][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = 0
        
        x = [-0.5*unitX, L+0.5*unitX]
        y = H
        for xx in x:
            bracket_nodes[count-totalNodes][0] = count+1
            bracket_nodes[count-totalNodes][1] = xx
            bracket_nodes[count-totalNodes][2] = y
            count = count + 1
        
        x = -1.5*unitX
        y = H + unitCellSize
        for i in range(1, 3):
            for j in range(1, nnx+5):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = -1.5*unitX
            for j in range(1,nnx+5):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitCellSize
            x = -1.5*unitX

        x = -2.0*unitX
        y = H + unitY - unitCellSize/2.0
        for i in range(1, 3):
            for j in range(1, nnx+6):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitCellSize
            x = -2.0*unitX
            for j in range(1, nnx+6):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y + unitY
            x = -2.0*unitX
        
        x = [-0.5*unitX, L+0.5*unitX]
        y = 0
        for xx in x:
            bracket_nodes[count-totalNodes][0] = count+1
            bracket_nodes[count-totalNodes][1] = xx
            bracket_nodes[count-totalNodes][2] = y
            count = count + 1
        
        x = -1.5*unitX
        y = -unitCellSize
        for i in range(1, 3):
            for j in range(1, nnx+5):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y - unitY
            x = -1.5*unitX
            for j in range(1,nnx+5):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y - unitCellSize
            x = -1.5*unitX

        x = -2.0*unitX
        y = -unitY + unitCellSize/2.0
        for i in range(1, 3):
            for j in range(1, nnx+6):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y - unitCellSize
            x = -2.0*unitX
            for j in range(1, nnx+6):
                bracket_nodes[count-totalNodes][0] = count+1
                bracket_nodes[count-totalNodes][1] = x
                bracket_nodes[count-totalNodes][2] = y
                x = x + unitX
                count = count + 1
            
            y = y - unitY
            x = -2.0*unitX
        
        Dnodes_brackets = []
        DcoordsX = []
        DcoordsY = []
        DcoordsXr = [-1.5*unitX, -2.0*unitX, L+1.5*unitX, L+2.0*unitX]
        DcoordsYr = [H+unitCellSize, H+1.5*unitCellSize, H+unitY+0.5*unitCellSize, 
                     -unitCellSize, -1.5*unitCellSize, -unitY-0.5*unitCellSize]
        for i in DcoordsXr:
            DcoordsX.append(round(i,2))
        for i in DcoordsYr:
            DcoordsY.append(round(i,2))
        for node in bracket_nodes:
            if round(node[2],2) in DcoordsY:
                if round(node[1],2) in DcoordsX:
                    Dnodes_brackets.append(node[0]-totalNodes-1)
        
        bracket_nodes = delete(bracket_nodes, Dnodes_brackets, 0)
        
        xCoord = nodes[:, 1]
        yCoord = nodes[:, 2]
        bottomNodes = argwhere((yCoord>=-1e-3) & (yCoord<=+1e-3))
        topNodes = argwhere((yCoord>=H-1e-3) & (yCoord<=H+1e-3))
        leftNodes = argwhere((xCoord>=-1e-3) & (xCoord<=+1e-3))
        rightNodes = argwhere((xCoord>=L-1e-3) & (xCoord<=L+1e-3))
        
        boundaryNodes = concatenate((bottomNodes, leftNodes, topNodes, rightNodes))
        #boundaryNodes = concatenate((bottomNodes, topNodes))
        boundaryNodes, inx = unique(boundaryNodes,return_index=True)
        boundaryNodesInx = nodes[boundaryNodes, 0]
        
        nodeIndex = nodes[:, 0]
        nonboundaryNodes = setdiff1d(nodeIndex, boundaryNodesInx)

        nonboundaryCoordX = nodes[nonboundaryNodes.astype(int)-1, 1]
        nonboundaryCoordY = nodes[nonboundaryNodes.astype(int)-1, 2]
        
        delta = unitCellSize * fac
        if (distribution.lower() == 'uniform'):
            randX = random.uniform(-delta, delta, len(nonboundaryNodes))
            randY = random.uniform(-delta, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'normal'):
            randX = random.normal(0.0, delta, len(nonboundaryNodes))
            randY = random.normal(0.0, delta, len(nonboundaryNodes))
        elif (distribution.lower() == 'exponential'):
            #delta = 2.5
            randX = random.exponential(1/delta, len(nonboundaryNodes))
            randY = random.exponential(1/delta, len(nonboundaryNodes))
        

        nonboundaryCoordX = nonboundaryCoordX + randX
        nonboundaryCoordY = nonboundaryCoordY + randY
        
        nodesR[:,0] = nodes[:,0]
        nodesR[:,1] = nodes[:,1]
        nodesR[:,2] = nodes[:,2]
        nodesR[nonboundaryNodes.astype(int)-1, 1] = nonboundaryCoordX
        nodesR[nonboundaryNodes.astype(int)-1, 2] = nonboundaryCoordY
    
    return nodes, nodesR, bracket_nodes

def connectivity(latticeType, unitCellSize, nodes):
    radius = unitCellSize + unitCellSize*1e-3
    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        if (latticeType.lower() == "fcc" and nodes[ii][1])%2 == 1.0 and (nodes[ii][2])%2 == 1.0:
            continue
        distance = sqrt(array(nodes[ii, 1] - nodes[:, 1])**2 + array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = argwhere(distance <= radius)
        nearNodes = setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1            
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    break
    indexRemove = []
    for i in range(0,len(dummyElem)):
        if (dummyElem[i][0] == 0 and dummyElem[i][1] == 0 and dummyElem[i][2] == 0):
            indexRemove.append(i)
    realElem = delete(dummyElem, [indexRemove], axis=0)
    realElem = realElem + 1
    for i in range(len(realElem)):
        realElem[i][0] = i+1
    return realElem

def insidePoint(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius

def geometry(LAT, l, nnx, rD=0.2, FTcalc=False, brackets=False, stiffMatrix=False, stiffCalc=False, nodeCount=False, UTval=False, mode=None):
    if stiffMatrix or stiffCalc:
        nnx = 10
    
    if (LAT.lower() == 'fcc'):
        L = float(l * nnx)
        H0 = 0.96 * L
        Hs = [l*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/l
        if round(nny) % 2.0 == 0.0:
            if H/L >= 0.96:
                H = H - l
                nny = H/l
            elif H/L < 0.96:
                H = H + l
                nny = H/l
        W = L/1.25
        a = [L/nnx*i for i in range(nnx+1)]
        a0 = min(a, key=lambda x:abs(x-(0.75*W)))
        if FTcalc:
            a0 = a0 - 0.25*W
        ai = [a0 + ((l/2)*(i)) for i in range(nnx)]
        vol = L*H
        if stiffCalc:
            nny = nnx
            L = float(l * nnx)
            H = float(l * nny)
            if mode.lower() == "unit":
                vol = l**2
            elif mode.lower() == "lattice":
                vol = L*H
        if stiffMatrix:
            nny = nnx
            L = float(l * nnx)
            H = float(l * nny)
        if brackets:
            nnx = nnx + 4
            L = float(l * nnx)
            nny = nny + 6
            H = l * nny
        nny = int(round(nny))
        totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
        totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
        if nodeCount:
            if mode.lower() == "fracture":
                totalNodes = totalNodes - round(nnx/1.66666667)
            elif mode.lower() == "ductile":
                totalNodes = totalNodes + totalBracketNodes - 8
            if stiffMatrix:
                nnx, nny = 10, 10
                totalNodes = (nnx + 1) * (nny + 1) + nnx * nny
        
    elif (LAT.lower() == 'tri'):
        if nnx % 2.0 == 1.0:
            nnx = nnx - 1
        L = 0.5 * (3.0**(0.5)) * l * nnx
        H0 = 0.96 * L
        Hs = [l*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/l
        if round(nny) % 2.0 == 0.0:
            if H/L >= 0.96:
                H = H - l
                nny = H/l
            elif H/L < 0.96:
                H = H + l
                nny = H/l
        if UTval:
            nny = 18
            H = nny*l
        W = L/1.25
        a = [L/(nnx/2)*i for i in range(nnx+1)]
        a0 = min(a, key=lambda x:abs(x-(0.75*W)))
        if FTcalc:
            a0 = a0 - 0.25*W
        ai = [a0 + ((0.5*(3.0**(0.5))*l)*(i)) for i in range(nnx)]
        vol = L*H
        if stiffCalc:
            nny = nnx
            L = (3**0.5) * l * nnx
            H = l*nny
            if mode.lower() == "unit":
                vol = l*(2*l*(3**0.5)/2)
            elif mode.lower() == "lattice":
                vol = L*H
        if stiffMatrix:
            nnx = nnx * 2
            nny = nnx / 2
            L = 0.5 * (3.0**(0.5)) * l * nnx
            H = l * nny
        if brackets:
            nnx = nnx + 4
            L = 0.5 * (3.0**(0.5)) * l * nnx
            nny = nny + 6
            H = l * nny
        nny = int(round(nny))
        totalNodes = int(round(((nnx / 1.99999) + 1) * (nny + 1)) + round((nnx / 1.99999) * nny))
        totalBracketNodes = int(round(((nnx/1.99999) + 3) * 3 * 2) + round(((nnx/1.99999) + 2) * 3 * 2) + 2*(nnx/2.0 + 2))
        if nodeCount:
            if mode.lower() == "fracture":
                totalNodes = totalNodes - round(nnx/3.33333333)
            elif mode.lower() == "ductile":
                totalNodes = totalNodes + totalBracketNodes - 4
            if stiffMatrix:
                nnx, nny = 10, 10
                totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)

    elif (LAT.lower() == 'kagome'):
        L = l*(2.0*nnx - 1)
        H0 = 0.96 * L
        Hs = [(3.0**0.5)*l*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/((3.0**0.5)*l)
        if round(nny) % 2.0 == 0.0:
            if H/L >= 0.96:
                H = H - ((3.0**0.5)*l)
                nny = H/((3.0**0.5)*l)
            elif H/L < 0.96:
                H = H + ((3.0**0.5)*l)
                nny = H/((3.0**0.5)*l)
        W = L/1.25
        if round(nny) % 4 == 3:
            a = [2*L*i/(2*nnx-1) + 0.5*l for i in range(nnx+1)]
        elif round(nny) % 4 == 1:
            a = [2*L*i/(2*nnx-1) + 1.5*l for i in range(nnx+1)]
        a0 = min(a, key=lambda x:abs(x-(0.75*W)))
        if FTcalc:
            a0 = a0 - 0.25*W
        ai = [a0 + ((2*l)*(i)) for i in range(nnx)]
        vol = L*H
        if stiffCalc:
            nny = nnx
            L = l*(2.0*nnx - 1)
            H = (3**0.5)*l*nny
            if mode.lower() == "unit":
                Tvol = (3*l)*(4*l*((3**0.5)/2))
            elif mode.lower() == "lattice":
                vol = L*H
        if stiffMatrix:
            nny = nnx
            L = l*(2.0*nnx - 1)
            H = (3.0**0.5)*l*nny
        if brackets:
            nnx = nnx + 2
            L = l*(2.0*nnx - 1)
            nny = nny + 6
            H = l * nny
        nny = int(round(nny))
        totalNodes = int(round((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2)))
        totalBracketNodes = int(round(((2*nnx+4)*3 + (nnx+2)*2 + (nnx+1)) * 2))
        if nodeCount:
            if mode.lower() == "fracture":
                totalNodes = totalNodes - round(nnx/1.75)
            elif mode.lower() == "ductile":
                totalNodes = totalNodes + totalBracketNodes - 16
            if stiffMatrix:
                nnx, nny = 10, 10
                totalNodes = int((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2))
        
    elif (LAT.lower() == 'hex'):
        L = (3.0**(0.5))*l*nnx
        H0 = 0.96 * L
        Hs = [(0.5*l)+(1.5*l*i) for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = (H-(0.5*l))/(1.5*l)
        if round(nny) % 2.0 == 0.0:
            if H/L >= 0.96:
                H = H - 1.5*l
                nny = (H-(0.5*l))/(1.5*l)
            elif H/L < 0.96:
                H = H + 1.5*l
                nny = (H-(0.5*l))/(1.5*l)
        nny = int(round(nny))
        W = L/1.25
        if round(nny) % 4 == 3:
            a = [((3.0**0.5)/2)*l + (L-((3.0**0.5)*l))/(nnx-1)*i for i in range(nnx+1)]
        elif round(nny) % 4 == 1:
            a = [L/nnx*i for i in range(nnx+1)]
        a0 = min(a, key=lambda x:abs(x-(0.75*W)))
        if FTcalc:
            a0 = a0 - 0.25*W
        ai = [a0 + (((3.0**0.5)*(l/2))*(i)) for i in range(nnx)]
        vol = L*H
        if stiffCalc:
            nny = nnx
            L = (3**0.5)*l*nnx
            H = 3*l*nny
            if mode.lower() == "unit":
                Tvol = (3*l)*(2*l*((3**0.5)/2))
            elif mode.lower() == "lattice":
                vol = L*H
        if stiffMatrix:
            nny = nnx * 2 + 1
            L = (3.0**(0.5))*l*nnx
            H = 3*l*nny
        if brackets:
            nnx = nnx + 4
            L = (3.0**(0.5))*l*nnx
            nny = nny + 8
            H = l * nny
        nny = int(round(nny))
        totalNodes = int(round(2*(nnx)*math.ceil(nny/2.0) + 2*(nnx+1)*math.ceil(nny/2.0)))
        totalBracketNodes = int(round(((nnx+5)*4 + (nnx+4)*4)*2 + 4))
        if nodeCount:
            if mode.lower() == "fracture":
                totalNodes = totalNodes
            elif mode.lower() == "ductile":
                totalNodes = totalNodes + totalBracketNodes - 12
            if stiffMatrix:
                nnx, nny = 10, 10
                totalNodes = ((2*nny) * (nnx+1)) + (((2*nny)+1) * nnx) + 50
    B = 0.5*W

    return [nnx, nny, L, H, W, B, a0, ai, totalNodes, totalBracketNodes, vol, l, LAT]

############################################################################################
############################################################################################
############################################################################################

if (nodeVar == 'no' and sizeVar == 'no'):
    imper = 'per'
elif (nodeVar == 'yes' and sizeVar == 'no'):
    imper = 'disNodes'
elif (nodeVar == 'no' and sizeVar == 'yes'):
    imper = 'disStruts'
else:
    imper = 'disNodesStruts'

if (finalRun.lower() == 'yes'):
    initial = initialJob
    numOfJobs = initial + numberOfRuns
elif (finalRun.lower() == 'no'):
    initial = initialJob
    numOfJobs = initial + 1

if nodeVar == "no":
    fac = 0.0

for idNum in range(initial,numOfJobs):
    
    #data 	     = sys.argv[-1]
    PBC          = 'no'
    elemType     = B21
    units        = 'millimeter'    # mass = tonn, length = millimeter, stress = MPa

    ############################################################################################
    ####################### Probability Distribution & Dimentions ##############################
    ############################################################################################

    if (distribution.lower() == 'uniform'):
        fac = fac
    elif (distribution.lower() == 'normal'):
        fac = (2*fac)/sqrt(2*pi*exp(1))
    elif (distribution.lower() == 'exponential'):
        fac = exp(1)/(2*fac)

    geom = geometry(latticeType, unitCellSize, nnx, stiffMatrix=stiffMatrix, UTval = UTval)
    nnx = geom[0]
    nny = geom[1]
    L = geom[2]
    H = geom[3]
    W = geom[4]
    B = geom[5]
    a0 = geom[6]
    ai = geom[7]
    totalNodes = geom[8]
    totalBracketNodes = geom[9]
    
    nodes, nodesR, bracket_nodes = node(latticeType, L, H, nnx, nny, totalNodes, totalBracketNodes, fac, distribution)

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
    
        ModelName    = 'Ductile' + '-' + latticeType + '-' + str(int(nnx)) + '-' + imper + '-' + str(idNum)
        Job          = 'Ductile' + '-' + latticeType + '-' + str(int(nnx)) + '-' + imper + '-' + str(idNum)
        if stiffMatrix and latticeType.lower() == "tri":
            ModelName    = 'Ductile' + '-' + latticeType + '-' + str(int(nnx/2)) + '-' + imper + '-' + str(idNum)
            Job          = 'Ductile' + '-' + latticeType + '-' + str(int(nnx/2)) + '-' + imper + '-' + str(idNum)

        #############################################################################################
        #################################### Brackets ###############################################
        #############################################################################################
        
        if stiffMatrix:
            nodes_duct = nodes
            nodesR_duct = nodesR
        else:
            nodes_duct = append(nodes, bracket_nodes, axis=0)
            nodesR_duct = append(nodesR, bracket_nodes, axis=0)
        
        #############################################################################################
        #################################### Strut Elements #########################################
        #############################################################################################
        
        element = connectivity(latticeType, unitCellSize, nodes)
        element_duct = connectivity(latticeType, unitCellSize, nodes_duct)
        
        #############################################################################################
        ################################ Radius Calculation #########################################
        #############################################################################################
        
        outofPlaneThick = 1.0
        if UTval:
            outofPlaneThick = 2.0
        
        length = zeros(shape=(len(element),1))
        for ik in range(0,len(element)-1):
            x1 = nodesR[int(element[ik][1]-1)][1]
            x2 = nodesR[int(element[ik][2]-1)][1]
            y1 = nodesR[int(element[ik][1]-1)][2]
            y2 = nodesR[int(element[ik][2]-1)][2]
            length[ik][0] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        if (crossSection.lower() == 'circ'):
            constants = [4 *relDensity, (L + H - pi * sum(length)), 4 * relDensity * (L * H)]
            dia_opt = roots(constants)
            dia_est = 2 * relDensity * 4 * L * H / (sum(length) * 2 * pi)
            diff_sqr = [(dia_opt[0] - dia_est) ** 2, (dia_opt[1] - dia_est) ** 2]
            index = argmin(diff_sqr)

            rad = dia_opt[index]/ 2
            Area = 3.14159*rad*rad
            
        elif (crossSection.lower() == 'rect'):
            thick_est = relDensity * L * H * outofPlaneThick / (sum(length) * outofPlaneThick)
            
            thickness = thick_est
            if UTval:
                thickness = 1.0
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
        elemBrackets = delete(element_duct, bodyElem_idxs, 0)
        for numEdge in range(len(element), len(element_duct)):
            x1 = nodesR_duct[int(elemBrackets[numEdge-len(element)][1]-1)][1]
            x2 = nodesR_duct[int(elemBrackets[numEdge-len(element)][2]-1)][1]
            y1 = nodesR_duct[int(elemBrackets[numEdge-len(element)][1]-1)][2]
            y2 = nodesR_duct[int(elemBrackets[numEdge-len(element)][2]-1)][2]
            xMid = (x1+x2)/2.0
            yMid = (y1+y2)/2.0
            edges = e.findAt(((xMid, yMid, 0.0), ))
            brackElems.append(edges)
        if stiffMatrix:
            brackElems = e.findAt(((0.0, 0.0, 0.0), ))
        p.Set(edges=brackElems, name='BracketSet')
        
        ############################################################################################
        ################################## Material Properties #####################################
        ############################################################################################
        
        if (userMaterial.lower() == 'al'):
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((2.7e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((70000.0, 0.3), ))
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
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((410000, 0.14), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((550, 0.0),
                 (550.01,	0.00001)))
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
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((4.43e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((123000, 0.3), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((932,	        0),
                (947.411802,    0.003453491),
                (957.4512331,	0.006906981),
                (966.1307689,	0.010360472),
                (974.030469,	0.013813962),
                (981.3967087,	0.017267453),
                (988.3639577,	0.020720943),
                (995.0160058,	0.024174434),
                (1001.409616,	0.027627924),
                (1007.585541,	0.031081415),
                (1013.574312,	0.034534905),
                (1019.399572,	0.037988396),
                (1025.08011,	0.041441886),
                (1030.631179,	0.044895377),
                (1036.065381,	0.048348867),
                (1041.393285,	0.051802358),
                (1046.623866,	0.055255848),
                (1051.764831,	0.058709339),
                (1056.822861,	0.062162829),
                (1061.803799,	0.06561632),
                (1066.712789,	0.06906981),
                (1071.554397,	0.072523301),
                (1076.332693,	0.075976791),
                (1081.05133,	0.079430282),
                (1085.713601,	0.082883772),
                (1090.322488,	0.086337263),
                (1094.880702,	0.089790753),
                (1099.39072,	0.093244244),
                (1103.854808,	0.096697734),
                (1108.275052,	0.100151225),
                (1112.653372,	0.103604715)))
            mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                table=((0.102268174, 0.333333, 0.0), ))
            mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                type=DISPLACEMENT, softening=TABULAR, table=(
                (0.0, 0.0),
                (0.2, FineElemSizeUT*0.00001),
                (0.4, FineElemSizeUT*0.00002), 
                (0.6, FineElemSizeUT*0.00003),
                (0.8, FineElemSizeUT*0.00004),
                (1.0, FineElemSizeUT*0.00005)))

        if (sizeVar.lower() == 'yes'):
            relativeDensityUpdated = 10000
            while relativeDensityUpdated < (relDensity - 0.001) or relativeDensityUpdated > (relDensity + 0.001):

                lowerLim = (1.0 - beta) * thickness
                upperLim = (1.0 + beta) * thickness

                thick = random.uniform(lowerLim, upperLim, len(element))
                latticeVolume = dot(length[:].T, thick*outofPlaneThick)
                
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
#            if (crossSection.lower() == 'rect'):
            mdb.models[ModelName].RectangularProfile(name='RectBody', a=outofPlaneThick, b=thickness)
            mdb.models[ModelName].RectangularProfile(name='RectBracket', a=outofPlaneThick, b=2*thickness)
            
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
#            
#            elif (crossSection.lower() == 'circ'):
#            
#                mdb.models[ModelName].CircularProfile(name='Circ', r=rad)
#
#                mdb.models[ModelName].BeamSection(name='BeamSec', integration=DURING_ANALYSIS, 
#                    poissonRatio=0.0, profile='Circ', material=userMaterial, 
#                    temperatureVar=LINEAR, consistentMassMatrix=False)
#
#                p = mdb.models[ModelName].parts['Part-1']
#                region = p.sets['AllPartSet']
#                p.SectionAssignment(region=region, sectionName='BeamSec', offset=0.0, 
#                    offsetType=MIDDLE_SURFACE, offsetField='', 
#                    thicknessAssignment=FROM_SECTION)
#
#                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
 
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

        mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
            timePeriod=STEP_TIME)
            
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
                if ycoord > (-3*sqrt(3)*unitCellSize)-tol and ycoord < (-3*sqrt(3)*unitCellSize)+tol:
                    bottom_nodes.append(n)
                if xcoord > L-tol and xcoord < L+tol and ycoord > -tol and ycoord < (H+tol):
                    right_nodes.append(n)
                if ycoord > (H+3*sqrt(3)*unitCellSize)-tol and ycoord < (H+3*sqrt(3)*unitCellSize)+tol:
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

        ln = mesh.MeshNodeArray(left_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-left')

        ln = mesh.MeshNodeArray(bottom_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-bottom')

        ln = mesh.MeshNodeArray(right_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-right')

        ln = mesh.MeshNodeArray(top_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-top')
        
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
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'PE', 'LE', 'UT', 'RF', 'SDEG', 'DMICRT', 'STATUS'), numIntervals=FieldOut_frames)
                
            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), numIntervals=HistOut_frames, region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)
                
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
        
        if (finalRun.lower() == 'yes'):
            mdb.jobs[Job].writeInput(consistencyChecking=OFF)
            mdb.jobs[Job].submit(consistencyChecking=OFF)
            mdb.jobs[Job].waitForCompletion()
        
        if stiffMatrix:
            mdb.jobs[Job].writeInput(consistencyChecking=OFF)
            
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################
# ###################################### Input for CT Specimen #########################################
# ######################################################################################################
# ######################################################################################################
# ######################################################################################################            
        
    if (MechanicalModel.lower() == 'fracture' or MechanicalModel.lower() == 'both'):
        
        ModelName    = 'Fracture' + '-' + latticeType + '-' + str(int(nnx)) + '-' + imper + '-' + str(idNum)
        Job          = 'Fracture' + '-' + latticeType + '-' + str(int(nnx)) + '-' + imper + '-' + str(idNum)
        
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
        
        if latticeType.lower() == "fcc" or latticeType.lower() == "tri":
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

        nodes = delete(nodes,delNodes,0)
        nodesR = delete(nodesR,delNodes,0)

        for kk in range(0,len(nodes)):
            nodes[kk][0] = kk+1
            nodesR[kk][0] = kk+1
            
        #############################################################################################
        #################################### Strut Elements #########################################
        #############################################################################################
                
        element = connectivity(latticeType, unitCellSize, nodes)
        
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
                delNodes.append(nodes[kk][0]-1)
                continue
            insideTest2 = insidePoint(ridge2[2:],ridge2[2:],point)
            if insideTest2:
                delNodes.append(nodes[kk][0]-1)
                continue

        element = delete(element,delNodes,0)
        
        #############################################################################################
        ################################ Radius Calculation #########################################
        #############################################################################################
        
        outofPlaneThick = B
        
        length = zeros(shape=(len(element),1))
        for ik in range(0,len(element)):
            x1 = nodesR[int(element[ik][1]-1)][1]
            x2 = nodesR[int(element[ik][2]-1)][1]
            y1 = nodesR[int(element[ik][1]-1)][2]
            y2 = nodesR[int(element[ik][2]-1)][2]
            length[ik][0] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        if (crossSection.lower() == 'circ'):
            constants = [4 *relDensity, (L + H - pi * sum(length)), 4 * relDensity * (L * H)]
            dia_opt = roots(constants)
            dia_est = 2 * relDensity * 4 * L * H / (sum(length) * 2 * pi)
            diff_sqr = [(dia_opt[0] - dia_est) ** 2, (dia_opt[1] - dia_est) ** 2]
            index = argmin(diff_sqr)

            rad = dia_opt[index]/ 2
            Area = 3.142*rad*rad
            
        elif (crossSection.lower() == 'rect'):
            thick_est = relDensity * L * H * outofPlaneThick / (sum(length) * outofPlaneThick)
            
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
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((2.7e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((70000.0, 0.3), ))
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
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((418196.654, 0.163), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((550, 0.0),
                (550.1,	0.00001)))
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
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((4.43e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((135164.8352, 0.428571429), ))
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((932,	        0),
                (947.411802,    0.003453491),
                (957.4512331,	0.006906981),
                (966.1307689,	0.010360472),
                (974.030469,	0.013813962),
                (981.3967087,	0.017267453),
                (988.3639577,	0.020720943),
                (995.0160058,	0.024174434),
                (1001.409616,	0.027627924),
                (1007.585541,	0.031081415),
                (1013.574312,	0.034534905),
                (1019.399572,	0.037988396),
                (1025.08011,	0.041441886),
                (1030.631179,	0.044895377),
                (1036.065381,	0.048348867),
                (1041.393285,	0.051802358),
                (1046.623866,	0.055255848),
                (1051.764831,	0.058709339),
                (1056.822861,	0.062162829),
                (1061.803799,	0.06561632),
                (1066.712789,	0.06906981),
                (1071.554397,	0.072523301),
                (1076.332693,	0.075976791),
                (1081.05133,	0.079430282),
                (1085.713601,	0.082883772),
                (1090.322488,	0.086337263),
                (1094.880702,	0.089790753),
                (1099.39072,	0.093244244),
                (1103.854808,	0.096697734),
                (1108.275052,	0.100151225),
                (1112.653372,	0.103604715)))
            mdb.models[ModelName].materials[userMaterial].DuctileDamageInitiation(
                table=((0.102268174, 0.333333, 0.0), ))
            mdb.models[ModelName].materials[userMaterial].ductileDamageInitiation.DamageEvolution(
                type=DISPLACEMENT, softening=TABULAR, table=(
                (0.0, 0.0),
                (0.2, FineElemSizeFT*0.001),
                (0.4, FineElemSizeFT*0.002), 
                (0.6, FineElemSizeFT*0.003),
                (0.8, FineElemSizeFT*0.004),
                (1.0, FineElemSizeFT*0.005)))

        if (sizeVar.lower() == 'yes'):
            relativeDensityUpdated = 10000
            while relativeDensityUpdated < (relDensity - 0.001) or relativeDensityUpdated > (relDensity + 0.001):

                lowerLim = (1.0 - beta) * thickness
                upperLim = (1.0 + beta) * thickness

                thick = random.uniform(lowerLim, upperLim, len(element))

                latticeVolume = dot(length[:].T, thick*outofPlaneThick)
                
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
#            if (crossSection.lower() == 'circ'):
#                mdb.models[ModelName].CircularProfile(name='Circ', r=rad)
#
#                mdb.models[ModelName].BeamSection(name='BeamSec', integration=DURING_ANALYSIS, 
#                    poissonRatio=0.0, profile='Circ', material=userMaterial, 
#                    temperatureVar=LINEAR, consistentMassMatrix=False)
#
#                p = mdb.models[ModelName].parts['Part-1']
#                region = p.sets['AllPartSet']
#                p.SectionAssignment(region=region, sectionName='BeamSec', offset=0.0, 
#                    offsetType=MIDDLE_SURFACE, offsetField='', 
#                    thicknessAssignment=FROM_SECTION)
#
#                p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, -1.0))
#
#            elif (crossSection.lower() == 'rect'):

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
            connected_node1 = e.connectivity[0]
            connected_node2 = e.connectivity[1]
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
        
        if (finalRun.lower() == 'yes'):
            mdb.jobs[Job].writeInput(consistencyChecking=OFF)
            mdb.jobs[Job].submit(consistencyChecking=OFF)
            mdb.jobs[Job].waitForCompletion()