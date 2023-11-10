from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from numpy import *
import math
import os
executeOnCaeStartup()

#os.chdir(r"D:\Research\Paper\Paper4_fracturetoughness\ABAQUS\SizeStudy\brittle")
	
def node_new(L, H, nnx, nny, fac, distribution):

    unitX = L / nnx
    unitY = H / nny

    totalNodes = (nnx + 1) * (nny + 1) + nnx * nny
    nodes = zeros(shape=(totalNodes,3))
    nodesR = zeros(shape=(totalNodes,3))

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
   
    x = unitX / 2
    y = unitY / 2

    for i in range(1, nny+1):
        for j in range(1, nnx+1):
            nodes[count][0] = count+1
            nodes[count][1] = x
            nodes[count][2] = y
            x = x + unitX
            count = count + 1
        
        y = y + unitY
        x = unitX / 2

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
    
    return nodes, nodesR

def node_new_tri(L, H, nnx, nny, unitCellSize, fac, distribution):
    unitX = 0.5 * sqrt(3) * unitCellSize
    unitY = unitCellSize

    #totalNodes = (nnx - 1) * (nny + 1) + int(round(nnx / 2)) * nny
    totalNodes = int(round(nnx / 1.99999)) * (nny + 1) + int(round(nnx / 1.99999)) * nny
    #print(totalNodes)
    nodes = zeros(shape=(totalNodes, 3))
    nodesR = zeros(shape=(totalNodes, 3))

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
    y = unitY / 2

    for i in range(1, nny + 1):
        for j in range(1, int(round(nnx / 2.000001) + 2)):
            nodes[count][0] = count + 1
            nodes[count][1] = x
            nodes[count][2] = y
            x = x + 2 * unitX
            count = count + 1

        y = y + unitY
        x = unitX

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

    return nodes, nodesR

def node_Kagome(L, H, nnx, nny, unitCellSize, fac, distribution):

    unitX = unitCellSize
    unitY = sqrt(3) * unitCellSize

    totalNodes = (2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*floor(nny/2)
    nodes = zeros(shape=(totalNodes,3))
    nodesR = zeros(shape=(totalNodes,3))

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
    y = unitY / 2

    for i in range(1, int(math.ceil(nny/2.0)+1)):
        for j in range(1, nnx):
            nodes[count][0] = count+1
            nodes[count][1] = x
            nodes[count][2] = y
            x = x + 2*unitX
            count = count + 1
        
        y = y + 2*unitY
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
        
        y = y + 2*unitY
        x = unitX/2
        
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
    
    return nodes, nodesR

def node_Hex(L, H, nnx, nny, unitCellSize, fac, distribution):
    unitX = sqrt(3) *unitCellSize
    unitY = 2 * unitCellSize
    totalNodes = 2*(nnx)*math.ceil(nny/2.0) + 2*(nnx+1)*math.ceil(nny/2.0)
    nodes = zeros(shape=(totalNodes,3))
    nodesR = zeros(shape=(totalNodes,3))

    count = 0
    x = 0.5*sqrt(3)*unitCellSize
    y = 0
    for i in range(1, int(math.ceil(nny/2.0)+1)):
        for j in range(1, nnx+1):
            nodes[count][0] = count+1
            nodes[count][1] = x
            nodes[count][2] = y
            x = x + unitX
            count = count + 1
        
        y = y + unitY
        x = 0.5 * sqrt(3) * unitCellSize
        
        for j in range(1,nnx+1):
            nodes[count][0] = count+1
            nodes[count][1] = x
            nodes[count][2] = y
            x = x + unitX
            count = count + 1
        
        y = y + unitCellSize
        x = 0.5 * sqrt(3) * unitCellSize

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
    
    return nodes, nodesR

def connectivity(L, H, nnx, nny, nodes, latticeType):

    radius = min(L, H) / min(nnx, nny)
    numElem = (nnx + 1) * (nny + 1)


    dummyElem = []
    count = 0
    for ii in range(numElem):
        distance = sqrt(array(nodes[ii, 1] - nodes[:, 1])**2 + array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = argwhere(distance <= radius)
        nearNodes = setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            #dummyElem[count, 0] = count
            #dummyElem[count, 1] = ii
            #dummyElem[count, 2] = nearNodes[jj]
            #print count,ii, nearNodes[jj]
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1
        
    
    count = 0
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    count = count + 1
                    break
    
    if (latticeType.lower() == 'fcc'):
        realElem1 = dummyElem
        indexRemove = []
        for i in range(1,len(dummyElem)):
            if (dummyElem[i][0] == 0):
                indexRemove.append(i)
            
        
        realElem2 = delete(realElem1,[indexRemove],axis=0)
        realElem2 = realElem2 + 1
        for i in range(len(realElem2)):
            realElem2[i][0] = i+1
    
    elif (latticeType.lower() == 'tri'):
        
        for k in range(len(dummyElem)):
            y2 = nodes[dummyElem[k][2],2]
            y1 = nodes[dummyElem[k][1],2]
            x2 = nodes[dummyElem[k][2],1]
            x1 = nodes[dummyElem[k][1],1]
            dy = y2-y1
            dx = x2-x1
            #print y2, y1, x2, x1
            if dx != 0:
                tan = dy/dx
            else:
                tan = float('Inf')
            #tan = (nodes[realElem2[k][2],2]-nodes[realElem2[k][1],2])/float(nodes[realElem2[k][2],1]-nodes[realElem2[k][1],1])
            if (tan==0):
                dummyElem[k][:] = [0, 0, 0]
                #print tan, dummyElem[k][:]
        
        realElem1 = dummyElem
        indexRemove = []
        for i in range(1,len(dummyElem)):
            if (dummyElem[i][0] == 0):
                indexRemove.append(i)
        
        realElem2 = delete(realElem1,[indexRemove],axis=0)
        realElem2 = realElem2 + 1
        for i in range(len(realElem2)):
            realElem2[i][0] = i+1 
            
    #print realElem2
    return realElem2

def connectivity_tri(L, H, nnx, nny, unitCellSize, nodes, latticeType):

    radius = unitCellSize + 1e-3
    numElem = (nnx + 1) * (nny + 1)


    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        distance = sqrt(array(nodes[ii, 1] - nodes[:, 1])**2 + array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = argwhere(distance <= radius)
        nearNodes = setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            #dummyElem[count, 0] = count
            #dummyElem[count, 1] = ii
            #dummyElem[count, 2] = nearNodes[jj]
            #print count,ii, nearNodes[jj]
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1
        
    
    count = 0
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    count = count + 1
                    break
    

    realElem1 = dummyElem
    indexRemove = []
    for i in range(1,len(dummyElem)):
        if (dummyElem[i][0] == 0):
            indexRemove.append(i)
        
    
    realElem2 = delete(realElem1,[indexRemove],axis=0)
    realElem2 = realElem2 + 1
    for i in range(len(realElem2)):
        realElem2[i][0] = i+1
            
    #print realElem2
    return realElem2
    
def connectivity_kagome(L, H, nnx, nny, unitCellSize, nodes, latticeType):

    radius = unitCellSize + 1e-3

    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        distance = sqrt(array(nodes[ii, 1] - nodes[:, 1])**2 + array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = argwhere(distance <= radius)
        nearNodes = setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            #dummyElem[count, 0] = count
            #dummyElem[count, 1] = ii
            #dummyElem[count, 2] = nearNodes[jj]
            #print count,ii, nearNodes[jj]
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1
        
    # xCoord = nodes[:, 1]
    # yCoord = nodes[:, 2]
    # bottomNodes = argwhere(yCoord == 0)
    # topNodes = argwhere(yCoord == H)
    # leftNodes = argwhere(xCoord == 0)
    # rightNodes = argwhere(xCoord == L)
    
    
    count = 0
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    count = count + 1
                    break
    

    realElem1 = dummyElem
    indexRemove = []
    for i in range(1,len(dummyElem)):
        if (dummyElem[i][0] == 0):
            indexRemove.append(i)
        
    
    realElem2 = delete(realElem1,[indexRemove],axis=0)
    realElem2 = realElem2 + 1
    for i in range(len(realElem2)):
        realElem2[i][0] = i+1
            
    print(len(realElem2))
    return realElem2

def connectivity_hex(L, H, nnx, nny, unitCellSize, nodes, latticeType):

    radius = unitCellSize + 1e-3
    #print(nodes)
    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        distance = sqrt(array(nodes[ii, 1] - nodes[:, 1])**2 + array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = argwhere(distance <= radius)
        nearNodes = setdiff1d(inside.astype(int), [ii])
        for jj in range(len(nearNodes)):
            #dummyElem[count, 0] = count
            #dummyElem[count, 1] = ii
            #dummyElem[count, 2] = nearNodes[jj]
            #print count,ii, nearNodes[jj]
            dummyElem.append([count,ii,nearNodes[jj]])
            count = count + 1
        
    # xCoord = nodes[:, 1]
    # yCoord = nodes[:, 2]
    # bottomNodes = argwhere(yCoord == 0)
    # topNodes = argwhere(yCoord == H)
    # leftNodes = argwhere(xCoord == 0)
    # rightNodes = argwhere(xCoord == L)
    
    
    count = 0
    for i in range(len(dummyElem)):
        for j in range(len(dummyElem)):
            if (dummyElem[i][1] == dummyElem[j][2]):
                if (dummyElem[i][2] == dummyElem[j][1]):
                    dummyElem[j][:] = [0, 0, 0]
                    count = count + 1
                    break
    

    realElem1 = dummyElem
    indexRemove = []
    for i in range(1,len(dummyElem)):
        if (dummyElem[i][0] == 0):
            indexRemove.append(i)
        
    
    realElem2 = delete(realElem1,[indexRemove],axis=0)
    realElem2 = realElem2 + 1
    for i in range(len(realElem2)):
        realElem2[i][0] = i+1
            
    #print('number of elements:', len(realElem2))
    return realElem2

def insidePoint(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius
    
############################################################################################
####################################### Input ##############################################
############################################################################################

unitCellSize = 10               # unit cell size in horizontal and vertical direction
latticeType = 'tri'
MechanicalModel = 'both'    # 'fracture', 'ductile', 'both'
finalRun = 'yes'
numberOfRuns = 500
userMaterial = 'aluminium'# 'aluminium', 'silicon carbide', 'titanium alloy'
nnx = 21                        # number of Unit cells in X direction
nny = 21                        # number of Unit cells in Y direction
fac = 0.2                       # distortion
elemSize = 1.0                  # minimum element size
relDensity = 0.2                # relative density
strainApp = 0.03
distribution = 'uniform'        # 'uniform', 'normal', 'exponential'
crossSection = 'rect'
cpus = 16

sizeVar = 'no'
beta = 0.80

if (fac == 0.0):
    imper = 'per'
else:
    imper = 'dis'


if (finalRun.lower() == 'yes'):
    initial = 1
    numOfJobs = initial + numberOfRuns
elif (finalRun.lower() == 'no'):
    initial = 1
    numOfJobs = initial + 1

for idNum in range(initial,numOfJobs):

    
    #data 	     = sys.argv[-1]
    PBC          = 'no'
    elemType     = B21
    units        = 'millimeter'    # mass = tonn, length = millimeter, stress = MPa

    #SpringStiffness = 1e-6*71300*L

    ############################################################################################
    ############################## Probability Distribution ####################################
    ############################################################################################

    if (distribution.lower() == 'uniform'):
        fac = fac
    elif (distribution.lower() == 'normal'):
        fac = (2*fac)/sqrt(2*pi*exp(1))
    elif (distribution.lower() == 'exponential'):
        fac = exp(1)/(2*fac)

    if (latticeType.lower() == 'fcc'):
        L = unitCellSize * nnx
        H = unitCellSize * nny
        nodes, nodesR = node_new(L, H, nnx, nny, fac, distribution)
        
    elif (latticeType.lower() == 'tri'):
        L = 0.5 * sqrt(3) * unitCellSize * nnx
        H = unitCellSize * nny
        nodes, nodesR = node_new_tri(L, H, nnx, nny, unitCellSize, fac, distribution)

    elif (latticeType.lower() == 'kagome'):
        L = unitCellSize*(2*nnx - 1)
        H = sqrt(3)*unitCellSize*nny
        nodes, nodesR = node_Kagome(L, H, nnx, nny, unitCellSize, fac, distribution)
        
    elif (latticeType.lower() == 'hex'):
        L = sqrt(3)*unitCellSize*nnx
        H = 2*unitCellSize*math.ceil(nny/2.0)+unitCellSize*floor(nny/2)
        nodes, nodesR = node_Hex(L, H, nnx, nny, unitCellSize, fac, distribution)  
        
    #############################################################################################
    ############################## INPUT for CT Specimen ########################################
    #############################################################################################

    if (MechanicalModel.lower() == 'fracture' or MechanicalModel.lower() == 'both'):
        
        ModelName    = 'Fracture' + '_' + latticeType + '_' + str(nnx) + 'X' + str(nny) + '_' + imper + '-' + str(idNum)
        Job          = 'Fracture' + '_' + latticeType + '_' + str(nnx) + 'X' + str(nny) + '_' + imper + '-' + str(idNum)

        xCrS = [-0.1*L, H/2-unitCellSize*0.2]        # crack starting point bottomLeft
        xCrE = [0.5*(L/1.25)+0.25*(L/1.25), H/2+unitCellSize*0.2]     # crack end point topRight

        bcDia     = 0.5*0.25*(L/1.25)
        fixityLoc = [L-(L/1.25), (H/2)-0.355*(L/1.25)]
        loadLoc   = [L-(L/1.25), (H/2)+0.355*(L/1.25)]

        delNodes = []
        for kk in range(0,len(nodes)):
            xCoord = nodes[kk][1]
            yCoord = nodes[kk][2]
            point = (xCoord,yCoord)
            insideTest = insidePoint(xCrS,xCrE,point)
            if insideTest:
                delNodes.append(nodes[kk][0]-1)

        nodes = delete(nodes,delNodes,0)
        nodesR = delete(nodesR,delNodes,0)

        for kk in range(0,len(nodes)):
            nodes[kk][0] = kk+1
            nodesR[kk][0] = kk+1

        if (latticeType.lower() == 'fcc'):
            element = connectivity(L,H,nnx,nny,nodes,latticeType)
            
        elif (latticeType.lower() == 'tri'):
            element = connectivity_tri(L,H,nnx,nny, unitCellSize,nodes,latticeType)
            
        elif (latticeType.lower() == 'kagome'):
            element = connectivity_kagome(L,H,nnx,nny,unitCellSize,nodes,latticeType)
        
        elif (latticeType.lower() == 'hex'):
            element = connectivity_hex(L,H,nnx,nny,unitCellSize,nodes,latticeType)
            
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

        element = delete(element,delNodes,0)
        
        #############################################################################################
        ################################ Radius Calculation #########################################
        #############################################################################################
        
        outofPlaneThick = nnx*unitCellSize/1.25/2
        
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

        p = mdb.models[ModelName].parts['Part-1']
        e = p.edges
        if (sizeVar.lower() == 'yes'):
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


        if (userMaterial.lower() == 'aluminium'):
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
                (0.0036, elemSize*0.0021),
                (0.0045, elemSize*0.00422), 
                (0.0104, elemSize*0.0063),
                (0.022,  elemSize*0.0084),
                (0.0336, elemSize*0.0105),
                (0.0466, elemSize*0.0126),
                (0.0599, elemSize*0.0147),
                (0.0761, elemSize*0.0168),
                (0.095,  elemSize*0.01891),
                (0.1173, elemSize*0.02101),
                (0.1443, elemSize*0.02311),
                (0.1761, elemSize*0.02521),
                (0.2144, elemSize*0.02731),
                (0.2578, elemSize*0.02928), 
                (0.3029, elemSize*0.03098),
                (0.3457, elemSize*0.03236),
                (0.3866, elemSize*0.03335),
                (0.4301, elemSize*0.034),
                (0.4665, elemSize*0.03446),
                (1.0,    elemSize*0.03547)))

        elif (userMaterial.lower() == 'silicon carbide'):
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((418196.0, 0.163), ))
            mdb.models[ModelName].materials[userMaterial].UserDefinedField()

        elif (userMaterial.lower() == 'titanium alloy'):
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((4.42e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((135165.0, 0.428), ))
            mdb.models[ModelName].materials[userMaterial].UserDefinedField()
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((930.0,	0),
                (932.0,	0.002),
                (949.9,	0.00305977),
                (967.8,	0.004644084),
                (985.7,	0.006995041),
                (1003.6,	0.01045873),
                (1021.5,	0.015526706),
                (1039.4,	0.022892786),
                (1057.3,	0.033530402),
                (1075.2,	0.048797358),
                (1093.1,	0.070577003),
                (1111.0,	0.10146748)))

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

        if (sizeVar.lower() == 'no'):
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

        RFid = mdb.models[ModelName].rootAssembly.ReferencePoint(point=(L-(L/1.25), (H/2)+0.355*(L/1.25), 0.0)).id
        r1 = mdb.models[ModelName].rootAssembly.referencePoints
        refPoints1=(r1[RFid], )
        a.Set(referencePoints=refPoints1, name='load')

        RFid = mdb.models[ModelName].rootAssembly.ReferencePoint(point=(L-(L/1.25), (H/2)-0.355*(L/1.25), 0.0)).id
        r1 = mdb.models[ModelName].rootAssembly.referencePoints
        refPoints1=(r1[RFid], )
        a.Set(referencePoints=refPoints1, name='fixity')

        ############################################################################################
        ######################################## Step ##############################################
        ############################################################################################

        mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
            timePeriod=0.1)
            
        if (userMaterial.lower() == 'material-1'):    
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'PE', 'LE', 'UT', 'RF', 'EVOL', 'SDV'), numIntervals=200)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)

        else:
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'LE', 'UT', 'RF', 'SDEG', 'DMICRT', 'STATUS'), numIntervals=500)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)
            
        # ############################################################################################
        # ######################################## Mesh ##############################################
        # ############################################################################################


        a = mdb.models[ModelName].rootAssembly
        e = a.instances['Part-1-1'].edges
        edges = e.getByBoundingBox(0.0,0.0,0.0,L,H,0.0)
        elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
        pickedRegions =(edges, )
        a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
        partInstances =(a.instances['Part-1-1'], )
        a.seedPartInstance(regions=partInstances, size=elemSize, deviationFactor=0.1, 
            minSizeFactor=0.1)
        a = mdb.models[ModelName].rootAssembly
        partInstances =(a.instances['Part-1-1'], )
        a.generateMesh(regions=partInstances)


        all_nodes = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].nodes
        fix_nodes = []
        load_nodes = []
        for n in all_nodes:
            xcoord = n.coordinates[0]
            ycoord = n.coordinates[1]
            if in_circle(L-(L/1.25), (H/2)-0.355*(L/1.25), bcDia, xcoord, ycoord):
                fix_nodes.append(n)
            if in_circle(L-(L/1.25), (H/2)+0.355*(L/1.25), bcDia, xcoord, ycoord):
                load_nodes.append(n)
            

        ln = mesh.MeshNodeArray(fix_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-fix')

        ln = mesh.MeshNodeArray(load_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-load')


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

        mdb.models[ModelName].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
            smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (0.1, strainApp*H)))

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

        if (userMaterial.lower() == 'aluminium'):
            userSubroutine = ''
        elif (userMaterial.lower() == 'silicon carbide'):
            userSubroutine = 'VUSDFLD_britt.for'
        elif (userMaterial.lower() == 'titanium alloy'):
            userSubroutine = 'VUSDFLD_duct.for'

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
    
        ModelName    = 'Ductile' + '_' + latticeType + '_' + str(nnx) + 'X' + str(nny) + '_' + imper + '-' + str(idNum)
        Job          = 'Ductile' + '_' + latticeType + '_' + str(nnx) + 'X' + str(nny) + '_' + imper + '-' + str(idNum)

        if (latticeType.lower() == 'fcc'):
            element = connectivity(L,H,nnx,nny,nodes,latticeType)
            
        elif (latticeType.lower() == 'tri'):
            element = connectivity_tri(L,H,nnx,nny, unitCellSize,nodes,latticeType)
            
        elif (latticeType.lower() == 'kagome'):
            element = connectivity_kagome(L,H,nnx,nny,unitCellSize,nodes,latticeType)
        
        elif (latticeType.lower() == 'hex'):
            element = connectivity_hex(L,H,nnx,nny,unitCellSize,nodes,latticeType)
            
        #############################################################################################
        ################################ Radius Calculation #########################################
        #############################################################################################
        
        outofPlaneThick = 1
        
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

        p = mdb.models[ModelName].parts['Part-1']
        e = p.edges
        if (sizeVar.lower() == 'yes'):
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

        if (userMaterial.lower() == 'aluminium'):
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
                (0.0036, elemSize*0.0021),
                (0.0045, elemSize*0.00422), 
                (0.0104, elemSize*0.0063),
                (0.022,  elemSize*0.0084),
                (0.0336, elemSize*0.0105),
                (0.0466, elemSize*0.0126),
                (0.0599, elemSize*0.0147),
                (0.0761, elemSize*0.0168),
                (0.095,  elemSize*0.01891),
                (0.1173, elemSize*0.02101),
                (0.1443, elemSize*0.02311),
                (0.1761, elemSize*0.02521),
                (0.2144, elemSize*0.02731),
                (0.2578, elemSize*0.02928), 
                (0.3029, elemSize*0.03098),
                (0.3457, elemSize*0.03236),
                (0.3866, elemSize*0.03335),
                (0.4301, elemSize*0.034),
                (0.4665, elemSize*0.03446),
                (1.0,    elemSize*0.03547)))

        elif (userMaterial.lower() == 'silicon carbide'):
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((3.21e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((418196.0, 0.163), ))
            mdb.models[ModelName].materials[userMaterial].UserDefinedField()

        elif (userMaterial.lower() == 'titanium alloy'):
            mdb.models[ModelName].Material(name=userMaterial)
            mdb.models[ModelName].materials[userMaterial].Density(table=((4.42e-09, ), ))
            mdb.models[ModelName].materials[userMaterial].Elastic(table=((135165.0, 0.428), ))
            mdb.models[ModelName].materials[userMaterial].UserDefinedField()
            mdb.models[ModelName].materials[userMaterial].Plastic(table=
                ((930.0,	0),
                (932.0,	0.002),
                (949.9,	0.00305977),
                (967.8,	0.004644084),
                (985.7,	0.006995041),
                (1003.6,	0.01045873),
                (1021.5,	0.015526706),
                (1039.4,	0.022892786),
                (1057.3,	0.033530402),
                (1075.2,	0.048797358),
                (1093.1,	0.070577003),
                (1111.0,	0.10146748)))

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

        if (sizeVar.lower() == 'no'):
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


        ############################################################################################
        ######################################## Step ##############################################
        ############################################################################################

        mdb.models[ModelName].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
            timePeriod=0.1)
            
        if (userMaterial.lower() == 'material-1'):    
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'PE', 'LE', 'UT', 'RF', 'EVOL', 'SDV'), numIntervals=200)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)

        else:
            mdb.models[ModelName].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'S', 'LE', 'UT', 'RF', 'SDEG', 'DMICRT', 'STATUS'), numIntervals=100)

            regionDef=mdb.models[ModelName].rootAssembly.allInstances['Part-1-1'].sets['AllPartSet']
            mdb.models[ModelName].historyOutputRequests['H-Output-1'].setValues(variables=(
                'ALLIE', 'ALLKE', 'ALLSE'), region=regionDef, sectionPoints=DEFAULT, 
                rebar=EXCLUDE)
            
        # ############################################################################################
        # ######################################## Mesh ##############################################
        # ############################################################################################


        a = mdb.models[ModelName].rootAssembly
        e = a.instances['Part-1-1'].edges
        edges = e.getByBoundingBox(0.0,0.0,0.0,L,H,0.0)
        elemType1 = mesh.ElemType(elemCode=elemType, elemLibrary=STANDARD)
        pickedRegions =(edges, )
        a.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
        partInstances =(a.instances['Part-1-1'], )
        a.seedPartInstance(regions=partInstances, size=elemSize, deviationFactor=0.1, 
            minSizeFactor=0.1)
        a = mdb.models[ModelName].rootAssembly
        partInstances =(a.instances['Part-1-1'], )
        a.generateMesh(regions=partInstances)
        
        all_nodes = mdb.models[ModelName].rootAssembly.instances['Part-1-1'].nodes
        
        if latticeType.lower() == 'hex':
            left_nodes = []
            bottom_nodes = []
            right_nodes = []
            top_nodes = []
            for n in all_nodes:
                xcoord = n.coordinates[0]
                ycoord = n.coordinates[1]
                if xcoord > -tol and xcoord < +tol:
                    left_nodes.append(n)
                if ycoord > -tol and ycoord < +tol:
                    bottom_nodes.append(n)
                if xcoord > L-tol and xcoord < L+tol:
                    right_nodes.append(n)
                if ycoord > H-tol and ycoord < H+tol:
                    top_nodes.append(n)
        else:
            rfNodes = []
            val = 0
            for kk in range(0,nnx+1):
                rfNodes.append(val)
                val = val + unitCellSize
            
            left_nodes = []
            bottom_nodes = []
            right_nodes = []
            top_nodes = []
            for n in all_nodes:
                xcoord = n.coordinates[0]
                ycoord = n.coordinates[1]
                if xcoord > -tol and xcoord < +tol:
                    left_nodes.append(n)
                if ycoord > -tol and ycoord < +tol:
                    bottom_nodes.append(n)
                if xcoord > L-tol and xcoord < L+tol:
                    right_nodes.append(n)
                if ycoord > H-tol and ycoord < H+tol:
                    exist = xcoord in rfNodes
                    if exist:
                        top_nodes.append(n)
        


        ln = mesh.MeshNodeArray(left_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-left')

        ln = mesh.MeshNodeArray(bottom_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-bottom')

        ln = mesh.MeshNodeArray(right_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-right')

        ln = mesh.MeshNodeArray(top_nodes)
        mdb.models[ModelName].rootAssembly.Set(nodes=ln , name='Set-top')


        # ############################################################################################
        # #################################### Interactions ##########################################
        # ############################################################################################


        # #  no interaction for ductile model # #
        

        # ############################################################################################
        # #################################### Loading ###############################################
        # ############################################################################################

#elif (PBC.lower() == 'no'):
        mdb.models[ModelName].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
            smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (0.1, strainApp*H)))

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