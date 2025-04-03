import os
import numpy as np
import math
import sys

cmdIN = sys.argv[2:]
if len(cmdIN) > 0:
    latticeType = str(cmdIN[0])
    dis = str(cmdIN[1])
    fac = float(cmdIN[2])
    beta = float(cmdIN[2])
    nnx = int(cmdIN[3])
    unitCellSize = float(cmdIN[4])
    relDensity = float(cmdIN[5])
    initialJob = int(cmdIN[6])
    numberOfRuns = int(cmdIN[7])
    cpus = int(cmdIN[8])
    finalRun = 'yes'
    MechanicalModel = 'both'
    stiffMatrix = False
    UTval = False
        
    if dis == 'per':
        nodeVar = 'no'
        sizeVar = 'no'
    elif dis == 'disNodes':
        nodeVar = 'yes'
        sizeVar = 'no'
    elif dis == 'disStruts':
        nodeVar = 'no'
        sizeVar = 'yes'
    else:
        raise Exception("Invalid disorder input.")

def rDthickness(LAT, l, t=None, rD=None):
    if LAT.lower() == "fcc":
        A = 2*(1+np.sqrt(2))
    elif LAT.lower() == "tri":
        A = 2*np.sqrt(3)
    elif LAT.lower() == "kagome":
        A = np.sqrt(3)
    elif LAT.lower() == "hex":
        A = 2/np.sqrt(3)
        
    if t:
        rD = A*(t/l)
        return rD
    elif rD:
        t = (l*rD)/A
        return t

def geometry(LAT, l, nnx, rD=0.2, FTcalc=False, brackets=False, stiffMatrix=False, stiffCalc=False, nodeCount=False, UTval=False, mode=None):
    if stiffMatrix or stiffCalc:
        nnx = 10
    t = rDthickness(LAT, l, rD=rD)
    
    if (LAT.lower() == 'fcc' or LAT.lower() == 'fcc2'):
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
        deltaNM = 0.5 * np.sqrt(l*l + l*l)
        
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
        deltaNM = l

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
        deltaNM = l
        
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
        deltaNM = l
    B = 0.5*W
                
    return totalNodes


def export_nodes(inpFile, expFile):
    LAT = inpFile.split('-')[1]
    nnx = int(inpFile.split('-')[2])
    l = 10.0
    if "Fracture" in inpFile:
        mode = 'fracture'
    elif "Ductile" in inpFile:
        mode = 'ductile'
    totalNodes = geometry(LAT, l, nnx, stiffMatrix=stiffMatrix, nodeCount=True, mode=mode)
    
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


if not os.path.exists("transfer"):
    os.makedirs("transfer")

for file in os.scandir():
    if 'per' in file.name or 'disNodes' in file.name:
        if file.name.endswith('.inp'):
            expFile = "transfer/IN-n" + file.name[:-4].replace('_','-') + ".csv"
            export_nodes(file.name, expFile)

    if 'per' in file.name or 'disStruts' in file.name:
        if file.name.endswith('.inp'):
            expFile = "transfer/IN-s" + file.name[:-4].replace('_','-') + ".csv"
            export_struts(file.name, expFile)