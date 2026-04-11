from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import *
import numpy as np
import math
from fractions import Fraction
from resources.lattices import Geometry

randX_all = None
randY_all = None

def node(
    latticeType,
    L,
    H,
    nnx,
    nny,
    totalNodes,
    totalBracketNodes,
    delta,
    distribution,
    unitCellSize=None,
    targeted_disorder="all",
    idNum=None,
    initialJob=None,
    numberOfRuns=1,
    frequencies=None,
    opt_dis_x=None,
    opt_dis_y=None,
):
    if unitCellSize is None:
        unitCellSize = L / nnx
    if frequencies is None:
        frequencies = []

    if latticeType.lower() == "fcc":
        unitX = L / nnx
        unitY = H / nny
        
        nodes = np.zeros(shape=(totalNodes,3)) 
        nodesR = np.zeros(shape=(totalNodes,3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))

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
    
    if latticeType.lower() == "square":
        unitX = L / nnx
        unitY = H / nny
        
        nodes = np.zeros(shape=(totalNodes,3)) 
        nodesR = np.zeros(shape=(totalNodes,3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))

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
    
    if latticeType.lower() == "45square":
        unitX = L / nnx
        unitY = H / nny
        
        nodes = np.zeros(shape=(totalNodes,3)) 
        nodesR = np.zeros(shape=(totalNodes,3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))

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
    
    elif latticeType.lower() == "tri":
        unitX = 0.5 * np.sqrt(3) * unitCellSize
        unitY = unitCellSize

        nodes = np.zeros(shape=(totalNodes, 3))
        nodesR = np.zeros(shape=(totalNodes, 3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))
        
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

    elif latticeType.lower() == "kagome":
        unitX = float(unitCellSize)
        unitY = np.sqrt(3) * float(unitCellSize)
        
        nodes = np.zeros(shape=(totalNodes,3))
        nodesR = np.zeros(shape=(totalNodes,3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))

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
        for i in range(1, int(np.floor(nny/2)+1)):
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
    
    elif latticeType.lower() == "hex":
        unitX = np.sqrt(3)*unitCellSize
        unitY = 2*unitCellSize
            
        nodes = np.zeros(shape=(totalNodes,3))
        nodesR = np.zeros(shape=(totalNodes,3))
        bracket_nodes = np.zeros(shape=(totalBracketNodes, 3))


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
    
    Dnodes_brackets = np.array(Dnodes_brackets, dtype=int)
    bracket_nodes = np.delete(bracket_nodes, Dnodes_brackets, 0)

    xCoord = nodes[:, 1]
    yCoord = nodes[:, 2]
    bottomNodes = np.argwhere((yCoord >= -1e-3) & (yCoord <= 1e-3))
    topNodes = np.argwhere((yCoord <= H + 1e-3) & (yCoord >= H - 1e-3))
    leftNodes = np.argwhere((xCoord >= -1e-3) & (xCoord <= 1e-3))
    rightNodes = np.argwhere((xCoord <= L + 1e-3) & (xCoord >= L - 1e-3))
    
    boundaryNodes = np.concatenate((bottomNodes, leftNodes, topNodes, rightNodes))
    boundaryNodes, inx = np.unique(boundaryNodes,return_index=True)
    boundaryNodesInx = nodes[boundaryNodes, 0]
    
    nodeIndex = nodes[:, 0]
    nonboundaryNodes = np.setdiff1d(nodeIndex, boundaryNodesInx)

    xCoord = nodes[nonboundaryNodes.astype(int)-1][:, 1]
    yCoord = nodes[nonboundaryNodes.astype(int)-1][:, 2]
    if targeted_disorder == "X":
        disNodes_pos = np.argwhere(((yCoord/H >= (xCoord-1.0*unitCellSize)/L) & (yCoord/H <= (xCoord+1.0*unitCellSize)/L)))
        disNodes_neg = np.argwhere(((yCoord/H >= (L-xCoord-1.0*unitCellSize)/L) & (yCoord/H <= (L-xCoord+1.0*unitCellSize)/L)))
        disNodes = np.concatenate((disNodes_pos, disNodes_neg))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "nX":
        disNodes12 = np.argwhere(((yCoord/H >= (L-xCoord+1.0*unitCellSize)/L) & (yCoord/H >= (xCoord+1.0*unitCellSize)/L)))
        disNodes23 = np.argwhere(((yCoord/H <= (L-xCoord-1.0*unitCellSize)/L) & (yCoord/H >= (xCoord+1.0*unitCellSize)/L)))
        disNodes34 = np.argwhere(((yCoord/H <= (L-xCoord-1.0*unitCellSize)/L) & (yCoord/H <= (xCoord-1.0*unitCellSize)/L)))
        disNodes41 = np.argwhere(((yCoord/H >= (L-xCoord+1.0*unitCellSize)/L) & (yCoord/H <= (xCoord-1.0*unitCellSize)/L)))
        disNodes = np.concatenate((disNodes12, disNodes23, disNodes34, disNodes41))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "v":
        disNodes1 = np.argwhere(((xCoord >= 1*L/4-0.75*unitCellSize) & (xCoord <= 1*L/4+0.75*unitCellSize)))
        disNodes2 = np.argwhere(((xCoord >= 2*L/4-0.75*unitCellSize) & (xCoord <= 2*L/4+0.75*unitCellSize)))
        disNodes3 = np.argwhere(((xCoord >= 3*L/4-0.75*unitCellSize) & (xCoord <= 3*L/4+0.75*unitCellSize)))
        disNodes = np.concatenate((disNodes1, disNodes2, disNodes3))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "h":
        disNodes1 = np.argwhere(((yCoord >= 1*H/4-0.75*unitCellSize) & (yCoord <= 1*H/4+0.75*unitCellSize)))
        disNodes2 = np.argwhere(((yCoord >= 2*H/4-0.75*unitCellSize) & (yCoord <= 2*H/4+0.75*unitCellSize)))
        disNodes3 = np.argwhere(((yCoord >= 3*H/4-0.75*unitCellSize) & (yCoord <= 3*H/4+0.75*unitCellSize)))
        disNodes = np.concatenate((disNodes1, disNodes2, disNodes3))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "o":
        disNodes1 = np.argwhere((((yCoord - H/2)**2 + (xCoord - L/2)**2 >= (3*unitCellSize)**2) & 
                              ((yCoord - H/2)**2 + (xCoord - L/2)**2 <= (6*unitCellSize)**2)))
        disNodes = np.concatenate((disNodes1))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "oo":
        disNodes1 = np.argwhere((((yCoord - H/2)**2 + (xCoord - L/2)**2 >= (2*unitCellSize)**2) & 
                              ((yCoord - H/2)**2 + (xCoord - L/2)**2 <= (4*unitCellSize)**2)))
        disNodes2 = np.argwhere((((yCoord - H/2)**2 + (xCoord - L/2)**2 >= (6*unitCellSize)**2) & 
                              ((yCoord - H/2)**2 + (xCoord - L/2)**2 <= (8*unitCellSize)**2)))
        disNodes = np.concatenate((disNodes1, disNodes2))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "D":
        disNodes1 = np.argwhere(((yCoord/(H/2) >= ((3*L/2)-xCoord-1.0*unitCellSize)/(L/2)) & (yCoord/(H/2) <= ((3*L/2)-xCoord+1.0*unitCellSize)/(L/2))))
        disNodes2 = np.argwhere(((yCoord/(H/2) >= ((L/2)+xCoord-1.0*unitCellSize)/(L/2)) & (yCoord/(H/2) <= ((L/2)+xCoord+1.0*unitCellSize)/(L/2))))
        disNodes3 = np.argwhere(((yCoord/(H/2) >= ((L/2)-xCoord-1.0*unitCellSize)/(L/2)) & (yCoord/(H/2) <= ((L/2)-xCoord+1.0*unitCellSize)/(L/2))))
        disNodes4 = np.argwhere(((yCoord/(H/2) >= ((-L/2)+xCoord-1.0*unitCellSize)/(L/2)) & (yCoord/(H/2) <= ((-L/2)+xCoord+1.0*unitCellSize)/(L/2))))
        disNodes = np.concatenate((disNodes1, disNodes2, disNodes3, disNodes4))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "DD":
        disNodes11 = np.argwhere(((yCoord >= H/2) & (xCoord >= L/2) & (yCoord/(H/3) >= ((4*L/3)-xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((4*L/3)-xCoord+1.0*unitCellSize)/(L/3))))
        disNodes12 = np.argwhere(((yCoord >= H/2) & (xCoord >= L/2) & (yCoord/(H/3) >= ((5*L/3)-xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((5*L/3)-xCoord+1.0*unitCellSize)/(L/3))))
        disNodes21 = np.argwhere(((yCoord >= H/2) & (xCoord <= L/2) & (yCoord/(H/3) >= ((L/3)+xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((L/3)+xCoord+1.0*unitCellSize)/(L/3))))
        disNodes22 = np.argwhere(((yCoord >= H/2) & (xCoord <= L/2) & (yCoord/(H/3) >= ((2*L/3)+xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((2*L/3)+xCoord+1.0*unitCellSize)/(L/3))))
        disNodes31 = np.argwhere(((yCoord <= H/2) & (xCoord <= L/2) & (yCoord/(H/3) >= ((L/3)-xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((L/3)-xCoord+1.0*unitCellSize)/(L/3))))
        disNodes32 = np.argwhere(((yCoord <= H/2) & (xCoord <= L/2) & (yCoord/(H/3) >= ((2*L/3)-xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((2*L/3)-xCoord+1.0*unitCellSize)/(L/3))))
        disNodes41 = np.argwhere(((yCoord <= H/2) & (xCoord >= L/2) & (yCoord/(H/3) >= ((-2*L/3)+xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((-2*L/3)+xCoord+1.0*unitCellSize)/(L/3))))
        disNodes42 = np.argwhere(((yCoord <= H/2) & (xCoord >= L/2) & (yCoord/(H/3) >= ((-L/3)+xCoord-1.0*unitCellSize)/(L/3)) & (yCoord/(H/3) <= ((-L/3)+xCoord+1.0*unitCellSize)/(L/3))))
        disNodes = np.concatenate((disNodes11, disNodes12, disNodes21, disNodes22, disNodes31, disNodes32, disNodes41, disNodes42))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "DDD":
        disNodes11 = np.argwhere(((yCoord >= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((5*L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((5*L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes12 = np.argwhere(((yCoord >= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((6*L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((6*L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes13 = np.argwhere(((yCoord >= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((7*L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((7*L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes21 = np.argwhere(((yCoord >= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes22 = np.argwhere(((yCoord >= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((2*L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((2*L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes23 = np.argwhere(((yCoord >= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((3*L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((3*L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes31 = np.argwhere(((yCoord <= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes32 = np.argwhere(((yCoord <= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((2*L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((2*L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes33 = np.argwhere(((yCoord <= H/2) & (xCoord <= L/2) & (yCoord/(H/4) >= ((3*L/4)-xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((3*L/4)-xCoord+1.0*unitCellSize)/(L/4))))
        disNodes41 = np.argwhere(((yCoord <= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((-3*L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((-3*L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes42 = np.argwhere(((yCoord <= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((-2*L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((-2*L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes43 = np.argwhere(((yCoord <= H/2) & (xCoord >= L/2) & (yCoord/(H/4) >= ((-L/4)+xCoord-1.0*unitCellSize)/(L/4)) & (yCoord/(H/4) <= ((-L/4)+xCoord+1.0*unitCellSize)/(L/4))))
        disNodes = np.concatenate((disNodes11, disNodes12, disNodes13, disNodes21, disNodes22, disNodes23, disNodes31, disNodes32, 
                                  disNodes33, disNodes41, disNodes42, disNodes43))
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    elif targeted_disorder == "xs":
        Hs = [0, H/3, 2*H/3, H]
        Ls = [0, L/3, 2*L/3, L]
        disNodes = []
        for i in range(len(Hs)-1):
            for j in range(len(Ls)-1):
                disNodes_pos = np.argwhere((((yCoord-Hs[i])/(Hs[i+1]-Hs[i]) >= ((xCoord-Ls[j])-0.5*unitCellSize)/(Ls[j+1]-Ls[j])) & ((yCoord-Hs[i])/(Hs[i+1]-Hs[i]) <= ((xCoord-Ls[j])+0.5*unitCellSize)/(Ls[j+1]-Ls[j])) & (yCoord>Hs[i]+0.5*unitCellSize) & (yCoord<Hs[i+1]-0.5*unitCellSize) & (xCoord>Ls[j]+0.5*unitCellSize) & (xCoord<Ls[j+1]-0.5*unitCellSize)))
                disNodes_neg = np.argwhere((((yCoord-Hs[i])/(Hs[i+1]-Hs[i]) >= ((Ls[j+1]-Ls[j])-(xCoord-Ls[j])-0.5*unitCellSize)/(Ls[j+1]-Ls[j])) & ((yCoord-Hs[i])/(Hs[i+1]-Hs[i]) <= ((Ls[j+1]-Ls[j])-(xCoord-Ls[j])+0.5*unitCellSize)/(Ls[j+1]-Ls[j])) & (yCoord>Hs[i]+0.5*unitCellSize) & (yCoord<Hs[i+1]-0.5*unitCellSize) & (xCoord>Ls[j]+0.5*unitCellSize) & (xCoord<Ls[j+1]-0.5*unitCellSize)))
                disNodes.append(np.concatenate((disNodes_pos, disNodes_neg)))
        disNodes = np.array(np.concatenate((np.array(disNodes, dtype=object))), dtype=int)
        disNodes, inx = np.unique(disNodes,return_index=True)
        disNodes = nonboundaryNodes[disNodes].flatten()
        disorderNodes = disNodes
    else:
        disorderNodes = nonboundaryNodes
    
    if (distribution.lower() == 'uniform'):
        randX = np.random.uniform(-delta, delta, len(disorderNodes))
        randY = np.random.uniform(-delta, delta, len(disorderNodes))
    elif (distribution.lower() == 'lhs_uniform'):
        if idNum is None or initialJob is None:
            raise ValueError("node(..., distribution='lhs_uniform') requires idNum and initialJob.")
        global randX_all, randY_all
        first_run = int(idNum - initialJob) == 0
        needs_reset = (
            randX_all is None
            or randY_all is None
            or len(randX_all) != int(numberOfRuns)
            or (len(randX_all) > 0 and len(randX_all[0]) != len(disorderNodes))
        )
        if first_run or needs_reset:
            randX_all = LHS_uniform(var=len(disorderNodes), strats=numberOfRuns, lim=delta)
            randY_all = LHS_uniform(var=len(disorderNodes), strats=numberOfRuns, lim=delta)
        randX = randX_all[idNum-initialJob]
        randY = randY_all[idNum-initialJob]
    elif (distribution.lower() == 'normal'):
        randX = np.random.normal(0.0, delta, len(disorderNodes))
        randY = np.random.normal(0.0, delta, len(disorderNodes))
    elif (distribution.lower() == 'exponential'):
        randX = np.random.exponential(1/delta, len(disorderNodes))
        randY = np.random.exponential(1/delta, len(disorderNodes))
    elif (distribution.lower() == 'frequency'):
        rows = set(nodes[disorderNodes.astype(int)-1,2])
        while len(frequencies) < (2*len(rows)):
            f = random_low_alias_freq(dx=unitCellSize)
            if f not in frequencies:
                frequencies.append(f)
        rand = nodes[disorderNodes.astype(int)-1]
        rand[:,1] = np.zeros(len(disorderNodes))
        rand[:,2] = np.zeros(len(disorderNodes))
        for i, y in enumerate(rows):
            dN = nodes[disorderNodes.astype(int)-1]
            dN_xs = dN[np.argwhere(dN[:,2] == y)][:,:,:2]
            r = np.array([[j[0,0], triangle_wave(j[0,1]+(2*unitCellSize), frequencies[2*i], delta), triangle_wave(j[0,1], frequencies[2*i+1], delta)] 
                          for j in dN_xs])
            idxs = np.isin(rand[:,0], r[:,0])
            sorter = np.argsort(r[:,0])
            match_idxs = sorter[np.searchsorted(r[:,0], rand[idxs,0], sorter=sorter)]
            rand[idxs,1] = r[match_idxs,1]
            rand[idxs,2] = r[match_idxs,2]
        randX = rand[:,1]
        randY = rand[:,2]
    elif (distribution.lower() == 'opt'):
        if opt_dis_x is None or opt_dis_y is None:
            raise ValueError("node(..., distribution='opt') requires opt_dis_x and opt_dis_y.")
        randX = opt_dis_x
        randY = opt_dis_y
    elif (distribution.lower() == 'opt-f'):
        rows = set(nodes[disorderNodes.astype(int)-1,2])
        rand = nodes[disorderNodes.astype(int)-1]
        rand[:,1] = np.zeros(len(disorderNodes))
        rand[:,2] = np.zeros(len(disorderNodes))
        for i, y in enumerate(rows):
            dN = nodes[disorderNodes.astype(int)-1]
            dN_xs = dN[np.argwhere(dN[:,2] == y)][:,:,:2]
            r = np.array([[j[0,0], triangle_wave(j[0,1]+(2*unitCellSize), frequencies[2*i], delta), triangle_wave(j[0,1], frequencies[2*i+1], delta)] 
                          for j in dN_xs])
            idxs = np.isin(rand[:,0], r[:,0])
            sorter = np.argsort(r[:,0])
            match_idxs = sorter[np.searchsorted(r[:,0], rand[idxs,0], sorter=sorter)]
            rand[idxs,1] = r[match_idxs,1]
            rand[idxs,2] = r[match_idxs,2]
        randX = rand[:,1]
        randY = rand[:,2]

    disorderCoordX = nodes[disorderNodes.astype(int)-1,1] + randX
    disorderCoordY = nodes[disorderNodes.astype(int)-1,2] + randY
    
    nodesR[:,0] = nodes[:,0]
    nodesR[:,1] = nodes[:,1]
    nodesR[:,2] = nodes[:,2]
    nodesR[disorderNodes.astype(int)-1,1] = disorderCoordX
    nodesR[disorderNodes.astype(int)-1,2] = disorderCoordY
    
    return nodes, nodesR, bracket_nodes

def insidePoint(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius


def pStrainProperties(E, v):
    return E/(1-v**2), v/(1-v)

def LHS_uniform(var, strats, lim, mean=0, plot=False):
    lower_limits = np.linspace(mean-lim, mean+lim, strats, endpoint=False)
    upper_limits = lower_limits + ((lower_limits[-1] - lower_limits[0])/(len(lower_limits)-1))
    
    points = np.zeros((strats, var))
    for i in range(var):
        points[:, i] = np.random.uniform(lower_limits, upper_limits, size=strats)
        np.random.shuffle(points[:, i])
    return points

def triangle_wave(x, f, A):
    frac = np.mod(f * x, 1.0)
    tri = np.where(frac < 0.5, 2 * frac, 2 * (1 - frac))
    return A * (2 * tri - 1)

def sine_wave(x, f, A):
    return A * np.sin(2 * np.pi * f * x)

def is_well_approximable(alpha, q_max=20, tol=1e-6):
    frac = Fraction(alpha).limit_denominator(q_max)
    return abs(alpha - frac.numerator/frac.denominator) < tol

def random_low_alias_freq(dx=10.0, q_max=20, tol=1e-6):
    f_max = 1.0 / (2.0 * dx)
    while True:
        f = np.random.uniform(0, f_max)
        alpha = f * dx
        if not is_well_approximable(alpha, q_max=q_max, tol=tol):
            return f

def export_frequencies(inpFile, expFile):
    with open(inpFile, 'r') as f:
        lines = f.readlines()

    freq_start = int([lines.index(line) for line in lines if "**FREQUENCIES:" in line][0]) + 1
    freq_end = int([lines.index(line) for line in lines if "**END FREQUENCIES" in line][0])
    frequencies = [line.strip().strip("**") for line in lines[freq_start:freq_end]]
    
    with open(expFile, 'w') as f:
        for freq in frequencies:
            f.write(freq + '\n')

def export_nodes(inpFile, expFile=None, latticeType=None, unitCellSize=10.0, stiffMatrix=False, totalNodes=None):
    with open(inpFile, 'r') as f:
        lines = f.readlines()

    nodes_start = int([lines.index(line) for line in lines if "*Node" in line][0]) + 1
    all_nodes_end = int([lines.index(line) for line in lines if "*Element" in line][0])

    if totalNodes is None:
        LAT = inpFile.split('-')[1]
        nnx = int(inpFile.split('-')[2])
        if "Fracture" in inpFile:
            mode = 'fracture'
        elif "Ductile" in inpFile:
            mode = 'ductile'
        else:
            mode = 'ductile'

        if latticeType is None:
            latticeType = LAT

        geom = Geometry(latticeType, unitCellSize, nnx)
        if stiffMatrix:
            geom.stiffnessMatrix()
        geom.nodeCount(mode=mode, stiffMatrix=stiffMatrix)
        totalNodes = geom.totalNodes

    nodes_end = nodes_start + int(totalNodes)
    node_lines = lines[nodes_start:nodes_end]
    nodes = [[float(i.strip().strip('\n')) for i in line.split(",")] for line in node_lines]

    if expFile is not None:
        with open(expFile, 'w') as f:
            for line in node_lines:
                f.write(line)
    else:
        with open(inpFile.split('.inp')[0] + "\\" + "NodesElems.csv", 'w+') as f:
            f.write("*Nodes\n")
            for node in nodes:
                f.write("{}, {}, {}\n".format(int(node[0]), node[1], node[2]))

    return np.array(nodes)

def export_struts(inpFile, expFile):
    with open(inpFile, 'r') as f:
        lines = f.readlines()

    strut_lines = [lines[lines.index(line)+1].split(' ') for line in lines if "*Beam Section, elset=" in line]
    thicks = [float(line[-1].strip('\n')) for line in strut_lines]
    thicks_check = list(set(thicks))
    thicks_check.sort(reverse = True)
    if len(thicks_check) == 2:
        if round(thicks_check[0],3) == round(2*thicks_check[1],3):
            thicks = [t for t in thicks if t != thicks_check[0]]
    elif len(thicks_check) > 2:
        if round(thicks_check[0],1) == 2*round(np.mean(thicks_check[1:]),1):
            thicks = [t for t in thicks if t != thicks_check[0]]
        
    with open(expFile, 'w') as f:
        for thick in thicks:
            f.write(str(thick) + '\n')

def nodes_in_set(ra, name, prefix='Node '):
    pairs = set()
    if name in ra.nodeSets:
        for n in ra.nodeSets[name].nodes[0]:
            pairs.add((prefix + ".".join([n.instanceName, str(n.label)])))
    for iname, inst in ra.instances.items():
        if name in inst.nodeSets:
            for n in inst.nodeSets[name].nodes:
                pairs.add((prefix + ".".join([iname, str(n.label)])))
    return pairs

def get_DuctData(Job, H, L, B, Cmatrix=False, case_Cmatrix=None, BCtype="periodic", expected_steps=201):
    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = [["U2", "RF2"]]
    nodeSets = ["Set-top"]
    case = case_Cmatrix.lower() if case_Cmatrix is not None else ""
    if Cmatrix and case == 'a':
        variables = [["U1", "RF1"], ["U1", "RF2"]]
        if BCtype.lower() == "kubc": nodeSets = ["Set-right", "Set-topBody"]
        elif BCtype.lower() == "periodic": nodeSets = ["RP_Per_X", "RP_TB"]
    if Cmatrix and case == 'b':
        variables = [["U2", "RF2"], ["U2", "RF1"]]
        if BCtype.lower() == "kubc": nodeSets = ["Set-topBody", "Set-right"]
        elif BCtype.lower() == "periodic": nodeSets = ["RP_Per_Y", "RP_LR"]
    if Cmatrix and case == 'c':
        variables = [["U1", "RF1"], ["U2", "RF2"], ["U1", "RF1"], ["U2", "RF2"]]
        if BCtype.lower() == "kubc": nodeSets = ["Set-topBody", "Set-right", "Set-right", "Set-topBody"]
        elif BCtype.lower() == "periodic": nodeSets = ["RP_Per_X", "RP_Per_Y", "RP_LR", "RP_TB"]

    reg_load = 'Node '
    if Cmatrix and BCtype.lower() == "periodic": reg_load = 'Node ASSEMBLY'
    
    all_Us = []
    all_RFs = []
    for nodeSet, variable in zip(nodeSets, variables):
        Us_nodes, RFs_nodes = [], []
        target_nodes = nodes_in_set(odb.rootAssembly, nodeSet.upper(), prefix=reg_load)
        for reg in odb.steps[step].historyRegions.keys():
            if reg_load in reg and reg in target_nodes:
                try:
                    Us = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variable[0]].data]
                    RFs = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variable[1]].data]
                    Us = list(np.nan_to_num(Us))
                    RFs = list(np.nan_to_num(RFs))
                except:
                    Us = list(np.zeros(expected_steps))
                    RFs = list(np.zeros(expected_steps))
                if len(Us) != expected_steps:
                    Us, RFs = Us[:-1], RFs[:-1]
                    intrv = (Us[20] - Us[10])/10
                    Us = Us + [(i+1)*intrv for i in range(len(Us), expected_steps)]
                    RFs = RFs + list(np.zeros(expected_steps-len(RFs)))
                Us_nodes.append(Us)
                RFs_nodes.append(RFs)
        all_Us.append(Us_nodes)
        all_RFs.append(RFs_nodes)
    odb.close()

    all_SE = []
    for i, (Us_nodes, RFs_nodes) in enumerate(zip(all_Us, all_RFs)):
        Us_steps = np.transpose(Us_nodes)
        RFs_steps = np.transpose(RFs_nodes)
        
        numNodes = len(Us_steps[0])
        strain = []
        stress = []
        for j, (Us_step, RFs_step) in enumerate(zip(Us_steps, RFs_steps)):
            Usum = 0.0
            RFsum = 0.0
            # if Cmatrix and case_Cmatrix.lower() == 'c':
            #     RFs_step = np.sort(RFs_step)[:-1]
            for U, RF in zip(Us_step, RFs_step):
                Usum += U
                RFsum += RF
            
            e = Usum/numNodes/H
            s = RFsum/(L*B)

            if Cmatrix:
                if case == 'a':
                    if i == 0:
                        e = Usum/numNodes/L
                        s = RFsum/(H*B)
                    elif i == 1:
                        e = all_SE[0][j][0]
                        s = RFsum/(L*B)
                if case == 'b':
                    if i == 0:
                        e = Usum/numNodes/H
                        s = RFsum/(L*B)
                    elif i == 1:
                        e = all_SE[0][j][0]
                        s = RFsum/(H*B)
                if case == 'c':
                    if i == 0:
                        e = Usum/numNodes/H
                        s = RFsum/(L*B)
                    elif i == 1:
                        e = Usum/numNodes/L
                        s = RFsum/(H*B)
                    elif i == 2:
                        e = all_SE[0][j][0]
                        s = RFsum/(H*B)
                    elif i == 3:
                        e = all_SE[1][j][0]
                        s = RFsum/(L*B)

            
            strain.append(e)
            stress.append(s)

        STEPS_OUT = np.transpose([strain, stress])
        all_SE.append(STEPS_OUT)
    all_SE = np.array(all_SE)
    
    if Cmatrix and case == 'c':
        e12 = all_SE[0][:,0]
        s12 = all_SE[0][:,1]
        e21 = all_SE[1][:,0]
        s21 = all_SE[1][:,1]
        e13 = all_SE[2][:,0]
        s13 = all_SE[2][:,1]
        e23 = all_SE[3][:,0]
        s23 = all_SE[3][:,1]
        all_SE = np.array([np.transpose([e12, s12]), 
                           np.transpose([e21, s21]),
                           np.transpose([e13, s13]), 
                           np.transpose([e23, s23])])
    
    return all_SE

def get_FracData(Job, expected_steps=201):
    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = ["U2", "RF2", "STATUS"]

    reg_load = 'Node ASSEMBLY.1'
    reg_cracktip = 'Element '
    
    try:
        U2 = [i[1] for i in odb.steps[step].historyRegions[reg_load].historyOutputs[variables[0]].data]
        RF2 = [i[1] for i in odb.steps[step].historyRegions[reg_load].historyOutputs[variables[1]].data]
        U2 = list(np.nan_to_num(U2))
        RF2 = list(np.nan_to_num(RF2))
    except:
        U2 = list(np.zeros(expected_steps))
        RF2 = list(np.zeros(expected_steps))
    if len(U2) != expected_steps:
        U2, RF2 = U2[:-1], RF2[:-1]
        intrv = (U2[20] - U2[10])/10
        U2 = U2 + [(i+1)*intrv for i in range(len(U2), expected_steps)]
        RF2 = RF2 + list(np.zeros(expected_steps-len(RF2)))
        
    ALL_STATUS = []
    for reg in odb.steps[step].historyRegions.keys():
        if reg_cracktip in reg:
            try:
                STATUS = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[2]].data]
                STATUS = list(np.nan_to_num(STATUS))
            except:
                STATUS = list(np.zeros(expected_steps))
            if len(STATUS) != expected_steps:
                STATUS = STATUS[:-1]
                STATUS = STATUS + list(np.zeros(expected_steps-len(STATUS)))
            ALL_STATUS.append(STATUS)
    
    ALL_STATUS = np.transpose(ALL_STATUS)
    
    STEPS_OUT = []
    for U, RF, STAT in zip(U2, RF2, ALL_STATUS):
        OUT = [U, RF]
        for el_STAT in STAT:
            OUT.append(el_STAT)
        STEPS_OUT.append(OUT)
    odb.close()
    return STEPS_OUT

def export_Udata(job, totalNodes, mode="ductile"):
    odb = openOdb(path=job)
    step = odb.steps['Step-1']

    count = 0
    for frame in step.frames:
        utField = frame.fieldOutputs['UT']
        nodes_Us = []
        for val in utField.values:
            if int(val.nodeLabel) > int(totalNodes):
                continue
            nodeU = [int(val.nodeLabel), val.data[0], val.data[1]]
            nodes_Us.append(nodeU)
        nodes_Us.sort()
        if mode.lower() == "fracture":
            nodes_Us = np.delete(np.array(nodes_Us), [0, 2], axis=0)
        with open(job.split('.odb')[0]+"\\"+"frame"+str(count)+".csv", "w+") as f:
            f.write("Node Label, U1, U2\n")
            for nodeU in nodes_Us:
                f.write("{}, {}, {}\n".format(nodeU[0], nodeU[1], nodeU[2]))
        count = count + 1
    
    odb.close()

