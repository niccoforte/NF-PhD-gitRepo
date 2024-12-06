import os
import numpy as np
import math

os.chdir("C:\\Users\\exy053\\Documents\\validation\\10\\FCC")

def node_count(file):
    latticeType = file.split('-')[1]
    nnx = int(file.split('-')[2])
    unitCellSize = 10.0
    
    if (latticeType.lower() == 'fcc'):
        L = float(unitCellSize * nnx)
        H0 = 0.96 * L
        Hs = [unitCellSize*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/unitCellSize
        if round(nny) % 2.0 == 0:
            if H/L >= 0.96:
                H = H - unitCellSize
                nny = H/unitCellSize
            elif H/L < 0.96:
                H = H + unitCellSize
                nny = H/unitCellSize
        nny = int(round(nny)) 
        totalNodes = (nnx + 1) * (nny + 1) + nnx * nny
        if "Fracture" in file:
            totalNodes = totalNodes - round(nnx/1.66666667)
        elif "Ductile" in file:
            totalBracketNodes = (nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2 - 8
            totalNodes = totalNodes + totalBracketNodes
    elif (latticeType.lower() == 'tri'):
        L = 0.5 * np.sqrt(3) * unitCellSize * nnx
        H0 = 0.96 * L
        Hs = [unitCellSize*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/unitCellSize
        if round(nny) % 2.0 == 0:
            if H/L >= 0.96:
                H = H - unitCellSize
                nny = H/unitCellSize
            elif H/L < 0.96:
                H = H + unitCellSize
                nny = H/unitCellSize
        nny = int(round(nny))
        totalNodes = (int(round(nnx / 1.99999) + 1) * (nny + 1)) + (int(round(nnx / 1.99999)) * nny)
        if "Fracture" in file:
            totalNodes = totalNodes - round(nnx/3.33333333)
        elif "Ductile" in file:
            totalBracketNodes = (int(round(nnx/1.99999) + 3) * 3 * 2) + (int(round(nnx/1.99999) + 2) * 3 * 2)
            totalNodes = totalNodes + totalBracketNodes - 4
    elif (latticeType.lower() == 'kagome'):
        L = unitCellSize*(2.0*nnx - 1)
        H0 = 0.96 * L
        Hs = [(3**0.5)*unitCellSize*i for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = H/((3**0.5)*unitCellSize)
        if round(nny) % 2.0 == 0:
            if H/L >= 0.96:
                H = H - ((3**0.5)*unitCellSize)
                nny = H/((3**0.5)*unitCellSize)
            elif H/L < 0.96:
                H = H + ((3**0.5)*unitCellSize)
                nny = H/((3**0.5)*unitCellSize)
        nny = int(round(nny))
        totalNodes = int((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2))
        if "Fracture" in file:
            totalNodes = totalNodes - round(nnx/1.75)
        elif "Ductile" in file:
            totalBracketNodes = (int(2*nnx+4)*3 + (nnx+2)*2 + (nnx+1)) * 2  - 16
            totalNodes = totalNodes + totalBracketNodes
    elif (latticeType.lower() == 'hex'):
        L = np.sqrt(3)*unitCellSize*nnx
        H0 = 0.96 * L
        Hs = [(0.5*unitCellSize)+(1.5*unitCellSize*i) for i in range(100)]
        H = min(Hs, key=lambda x:abs(x-H0))
        nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
        if round(nny) % 2.0 == 0:
            if H/L >= 0.96:
                H = H - 1.5*unitCellSize
                nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
            elif H/L < 0.96:
                H = H + 1.5*unitCellSize
                nny = (H-(0.5*unitCellSize))/(1.5*unitCellSize)
        nny = int(round(nny))
        totalNodes = int(2*(nnx)*math.ceil(nny/2.0) + 2*(nnx+1)*math.ceil(nny/2.0))
        if "Fracture" in file:
            totalNodes = totalNodes
        elif "Ductile" in file:
            totalBracketNodes = int((nnx+5)*4 + (nnx+4)*4)*2 + 4 - 12
            totalNodes = totalNodes + totalBracketNodes
    
    return totalNodes

def export_nodes(inpFile, expFile):
    totalNodes = node_count(inpFile)
    
    with open(inpFile, 'r') as f:
        lines = f.readlines()
    
    nodes_start = int([lines.index(line) for line in lines if "*Node" in line][0]) + 1
    all_nodes_end = int([lines.index(line) for line in lines if "*Element" in line][0])
    nodes = [[float(i.strip().strip('\n')) for i in line.split(",")] for line in lines[nodes_start:all_nodes_end]]
    
    # for i in range(len(nodes)):
        # dx = abs(nodes[i][1] - nodes[i-1][1])
        # dy = abs(nodes[i][2] - nodes[i-1][2])
        
        # if (dx > 1.0 and dx < 3.0) or (dy > 0.0 and dy < 5.0):
            # nodes_end = nodes_start + (int(nodes[i][0]) - 2)
            # break
    nodes_end = nodes_start + totalNodes
    
    node_lines = lines[nodes_start:nodes_end]
    
    with open(expFile, 'w') as f:
        for line in node_lines:
            f.write(line)


if not os.path.exists("transfer"):
    os.makedirs("transfer")

for file in os.scandir():
    if 'per' in file.name or 'disNodes' in file.name:
        if file.name.endswith('.inp') and 'Ductile' in file.name:
            expFile = "transfer/IN-n" + file.name[:-4].replace('_','-') + ".csv"
            node_count(file.name)
            export_nodes(file.name, expFile)
        elif file.name.endswith('.inp') and 'Fracture' in file.name:
            expFile = "transfer/IN-n" + file.name[:-4].replace('_','-') + ".csv"
            export_nodes(file.name, expFile)
        else:
            pass
    else:
        pass