from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys
import math
import os

mode = "any"
unitCellSize = 10.0

LAT = "tri"
DIS = "per"
nnx = 30

initial = 1
numOfJobs = 1

pDir = r"C:\\Users\\exy053\\Documents\\continuum"
#pDir = r"C:\\Users\\exy053\\Documents\\validation\\3\\0.13"
#pDir = r"C:\\Users\\exy053\\Documents\\PerSizeConv4\\10"

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
    
    stiffMatrix = False
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
      
os.chdir(pDir)

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

    return [nnx, nny, L, H, W, B, a0, ai, totalNodes, totalBracketNodes, deltaNM, vol, l, t, LAT]  # len=14


def export_Udata(job, totalNodes):
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
        with open(job.split('\\')[-1].split('.')[0]+"\\"+"frame"+str(count)+".csv", "w") as f:
            f.write("Node Label, U1, U2\n")
            for nodeU in nodes_Us:
                f.write("{}, {}, {}\n".format(nodeU[0], nodeU[1], nodeU[2]))
        count = count + 1
    
    odb.close()

def export_nodes(job, totalNodes):
    with open(job, 'r') as f:
        lines = f.readlines()
    
    nodes_start = int([lines.index(line) for line in lines if "*Node" in line][0]) + 1
    nodes_end = int(nodes_start + totalNodes)
    nodes = [[float(i.strip().strip('\n')) for i in line.split(",")] for line in lines[nodes_start:nodes_end]]
    
    with open(job.split('\\')[-1].split('.')[0]+"\\"+"NodesElems.csv", 'w') as f:
        f.write("*Nodes\n")
        for node in nodes:
            f.write("{}, {}, {}\n".format(int(node[0]), node[1], node[2]))
    return np.array(nodes)

def connectivity(job, LAT, nodes, geom, stiff=False, mode=None):
    radius = geom[12] + geom[12]*1e-3
    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        if (LAT.lower() == "fcc" and nodes[ii][1])%2 == 1.0 and (nodes[ii][2])%2 == 1.0:
            continue
        distance = np.sqrt(np.array(nodes[ii, 1] - nodes[:, 1])**2 + np.array(nodes[ii, 2] - nodes[:, 2])**2)
        inside = np.argwhere(distance <= radius)
        nearNodes = np.setdiff1d(inside.astype(int), [ii])
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
        if stiff:
            if LAT.lower() == "tri" and mode.lower() == "unit":
                if (dummyElem[i][1] == 3 and dummyElem[i][2] == 4) or (dummyElem[i][1] == 4 and dummyElem[i][2] == 3):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 0 and dummyElem[i][2] == 3) or (dummyElem[i][1] == 3 and dummyElem[i][2] == 0):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 3 and dummyElem[i][2] == 5) or (dummyElem[i][1] == 5 and dummyElem[i][2] == 3):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 1 and dummyElem[i][2] == 4) or (dummyElem[i][1] == 4 and dummyElem[i][2] == 1):
                    indexRemove.append(i)
                elif (dummyElem[i][1] == 4 and dummyElem[i][2] == 6) or (dummyElem[i][1] == 6 and dummyElem[i][2] == 4):
                    indexRemove.append(i)
            elif LAT.lower() == "tri" and mode.lower() == "lattice":
                if ((nodes[dummyElem[i][1]][1] == 0 and nodes[dummyElem[i][2]][1] == 0) or 
                    (nodes[dummyElem[i][1]][1] == geom[3] and nodes[dummyElem[i][2]][1] == geom[3])):
                    indexRemove.append(i)
    realElem = np.delete(dummyElem, [indexRemove], axis=0)
    if stiff:
        realElem = realElem
    else:
        realElem = realElem + 1
    for i in range(len(realElem)):
        realElem[i][0] = i+1
    
    with open(job.split('\\')[-1].split('.')[0]+"\\"+"NodesElems.csv", 'a') as f:
        f.write("*Elems\n")
        for elem in realElem:
            f.write("{}, {}, {}\n".format(int(elem[0]), int(elem[1]), int(elem[2])))
    return realElem


if mode.lower() == 'any':
    for curDirectory, folders, files in os.walk(pDir):
        odbs = [f for f in files if f.endswith('.odb')]
        inps = [f for f in files if f.endswith('.inp')]
        for odb in odbs:
            odbPath = os.path.join(curDirectory, odb)
            if not os.path.exists(odb.split('.')[0]):
                os.makedirs(odb.split('.')[0])
            
            sim = odb.split('.')[0]
            MechMode = sim.split('-')[0]
            LAT = sim.split('-')[1]
            nnx = int(sim.split('-')[2])
            geom = geometry(LAT, unitCellSize, nnx, nodeCount=True, mode=MechMode)
            
            export_Udata(odbPath, geom[8])
            
        for inp in inps:
            inpPath = os.path.join(curDirectory, inp)
            if not os.path.exists(inp.split('.')[0]):
                os.makedirs(inp.split('.')[0])
                
            sim = inp.split('.')[0]
            MechMode = sim.split('-')[0]
            LAT = sim.split('-')[1]
            nnx = int(sim.split('-')[2])
            geom = geometry(LAT, unitCellSize, nnx, nodeCount=True, mode=MechMode)
            
            nodes = export_nodes(inpPath, geom[8])
            elems = connectivity(inpPath, LAT, nodes, geom)

if (mode.lower() == 'ductile' or mode.lower() == 'both'):
    MechMode = 'Ductile'
    for kk in range(initial, initial+numOfJobs):
        sim = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk)
        if not os.path.exists(sim):
            os.makedirs(sim)
        
        odbPath = sim + ".odb"
        inpPath = sim + ".inp"
        geom = geometry(LAT, unitCellSize, nnx, nodeCount=True, mode=MechMode)
        
        export_Udata(odbPath, geom[8])
        nodes = export_nodes(inpPath, geom[8])
        elems = connectivity(inpPath, LAT, nodes, geom)
        
if (mode.lower() == 'fracture' or mode.lower() == 'both'):
    MechMode = 'Fracture'
    for kk in range(initial, initial+numOfJobs):    
        sim = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk)    
        if not os.path.exists(sim):
            os.makedirs(sim)
        
        odbPath = sim + ".odb"
        inpPath = sim + ".inp"
        geom = geometry(LAT, unitCellSize, nnx, nodeCount=True, mode=MechMode)
        
        export_Udata(odbPath, geom[8])
        nodes = export_nodes(inpPath, geom[8])
        elems = connectivity(inpPath, LAT, nodes, geom)
