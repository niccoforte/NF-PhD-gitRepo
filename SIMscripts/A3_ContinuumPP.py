from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys
import math
import os

mode = "any"
unitCellSize = 10.0

LAT = "FCC"
DIS = "per"
nnx = 20

initial = 1
numOfJobs = 1

pDir = r"Z:\\p1\\data\\Ti\\disNodes\\"
os.chdir(pDir)

def insidePoint(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False
   
class Geometry:
    def __init__(self, LAT, l, nnx, rD=0.2):
        self.LAT = LAT
        self.l = l
        self.nnx = nnx
        self.rD = rD

        self.t = self.rDthickness(rD=rD)

        if LAT.lower() == 'fcc':
            L = float(l * nnx)
            Lmin = L
            H0 = 0.96 * L
            Hs = [l * i for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = H / l

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((l / 2) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
            deltaNM = 0.5 * np.sqrt(l * l + l * l)

        elif LAT.lower() == 'square':
            L = float(l * nnx)
            Lmin = L
            H0 = 0.96 * L
            Hs = [l * i for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = H / l

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((l / 2) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1)))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2))
            deltaNM = l

        elif LAT.lower() == '45square':
            L = float(2**(1/2) * l * nnx)
            Lmin = L
            H0 = 0.96 * L
            Hs = [2**(1/2) * l * i for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = H / ((2**(1/2))*l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - ((2**(1/2))*l)
                    nny = H / ((2**(1/2))*l)
                elif H / L < 0.96:
                    H = H + ((2**2)*l)
                    nny = H / ((2**(1/2))*l)

            W = L / 1.25
            a = [L / nnx * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + (((2**(1/2))*l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
            deltaNM = l

        elif LAT.lower() == 'tri':
            if nnx % 2.0 == 1.0:
                nnx = nnx - 1
            L = 0.5 * (3.0 ** 0.5) * l * nnx
            Lmin = L
            H0 = 0.96 * L
            Hs = [l * i for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = H / l

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - l
                    nny = H / l
                elif H / L < 0.96:
                    H = H + l
                    nny = H / l

            W = L / 1.25
            a = [L / (nnx / 2) * i for i in range(nnx + 1)]
            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((0.5 * (3.0 ** 0.5) * l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round(((nnx / 1.99999) + 1) * (nny + 1)) +
                             round((nnx / 1.99999) * nny))
            totalBracketNodes = int(round(((nnx / 1.99999) + 3) * 3 * 2) +
                                    round(((nnx / 1.99999) + 2) * 3 * 2) +
                                    2 * (nnx / 2.0 + 2))
            deltaNM = l

        elif LAT.lower() == 'kagome':
            L = l * (2.0 * nnx - 1)
            Lmin = L - 3*l
            H0 = 0.96 * L
            Hs = [(3.0 ** 0.5) * l * i for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = H / ((3.0 ** 0.5) * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - ((3.0 ** 0.5) * l)
                    nny = H / ((3.0 ** 0.5) * l)
                elif H / L < 0.96:
                    H = H + ((3.0 ** 0.5) * l)
                    nny = H / ((3.0 ** 0.5) * l)

            W = L / 1.25
            if round(nny) % 4 == 3:
                a = [2 * L * i / (2 * nnx - 1) + 0.5 * l for i in range(nnx + 1)]
            elif round(nny) % 4 == 1:
                a = [2 * L * i / (2 * nnx - 1) + 1.5 * l for i in range(nnx + 1)]
            else:
                a = [2 * L * i / (2 * nnx - 1) + 0.5 * l for i in range(nnx + 1)]

            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + ((2 * l) * (i)) for i in range(nnx)]
            vol = L * H

            nny = int(round(nny))
            totalNodes = int(round((2 * nnx * (nny + 1)) +
                                   (nnx - 1) * math.ceil(nny / 2.0) +
                                   (nnx) * math.floor(nny / 2)))
            totalBracketNodes = int(round(((2 * nnx + 4) * 3 + (nnx + 2) * 2 + (nnx + 1)) * 2))
            deltaNM = l

        elif LAT.lower() == 'hex':
            L = (3.0 ** 0.5) * l * nnx
            Lmin = L - ((3.0 ** 0.5) * l)
            H0 = 0.96 * L
            Hs = [(0.5 * l) + (1.5 * l * i) for i in range(100)]
            H = min(Hs, key=lambda x: abs(x - H0))
            nny = (H - (0.5 * l)) / (1.5 * l)

            if round(nny) % 2.0 == 0.0:
                if H / L >= 0.96:
                    H = H - 1.5 * l
                    nny = (H - (0.5 * l)) / (1.5 * l)
                elif H / L < 0.96:
                    H = H + 1.5 * l
                    nny = (H - (0.5 * l)) / (1.5 * l)

            nny = int(round(nny))
            W = L / 1.25

            if round(nny) % 4 == 3:
                a = [((3.0 ** 0.5) / 2) * l + (L - ((3.0 ** 0.5) * l)) / (nnx - 1) * i
                     for i in range(nnx + 1)]
            elif round(nny) % 4 == 1:
                a = [L / nnx * i for i in range(nnx + 1)]
            else:
                a = [L / nnx * i for i in range(nnx + 1)]

            a0 = min(a, key=lambda x: abs(x - (0.75 * W)))
            ai = [a0 + (((3.0 ** 0.5) * (l / 2)) * (i)) for i in range(nnx)]
            vol = L * H

            totalNodes = int(round(2 * (nnx) * math.ceil(nny / 2.0) +
                                   2 * (nnx + 1) * math.ceil(nny / 2.0)))
            totalBracketNodes = int(round(((nnx + 5) * 4 + (nnx + 4) * 4) * 2 + 4))
            deltaNM = l

        self.nnx = nnx
        self.nny = nny
        self.L = L
        self.Lmin = Lmin
        self.H = H
        self.W = W
        self.a0 = a0
        self.ai = ai
        self.vol = vol
        self.totalNodes = totalNodes
        self.totalBracketNodes = totalBracketNodes
        self.deltaNM = deltaNM
        self.B = 0.5 * self.W

    def rDthickness(self, t=None, rD=None):
        if self.LAT.lower() == "fcc":
            A = 2*(1+np.sqrt(2))
        elif "square" in self.LAT.lower():
            A = 2
        elif self.LAT.lower() == "tri":
            A = 2*np.sqrt(3)
        elif self.LAT.lower() == "kagome":
            A = np.sqrt(3)
        elif self.LAT.lower() == "hex":
            A = 2/np.sqrt(3)
            
        if t:
            self.rD = A*(t/self.l)
        elif rD:
            self.t = (self.l*rD)/A

    def stiffnessMatrix(self, stiffCalc=False):
        nnx = 10
        if self.LAT.lower() == "fcc":
            nny = nnx
            L = float(self.l * nnx)
            H = float(self.l * nny)
            vol = L * H
            if stiffCalc:
                if stiffCalc.lower() == "unit":
                    vol = self.l ** 2
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
        elif self.LAT.lower() == "tri":
            if stiffCalc:
                nny = nnx
                L = (3 ** 0.5) * self.l * nnx
                H = self.l * nny
                vol = L * H
                if stiffCalc.lower() == "unit":
                    vol = self.l * (2 * self.l * (3 ** 0.5) / 2)
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
            else:
                nnx = nnx * 2
                nny = nnx / 2
                L = 0.5 * (3.0 ** 0.5) * self.l * nnx
                H = self.l * nny
                vol = L * H
        elif self.LAT.lower() == "kagome":
            nny = nnx
            L = self.l * (2.0 * nnx - 1)
            H = (3.0 ** 0.5) * self.l * nny
            vol = L * H
            if stiffCalc:
                if stiffCalc.lower() == "unit":
                    vol = (3 * self.l) * (4 * self.l * ((3 ** 0.5) / 2))
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
        elif self.LAT.lower() == "hex":
            L = (3 ** 0.5) * self.l * nnx
            if stiffCalc:
                nny = nnx
                H = 3 * self.l * nny
                vol = L * H
                if stiffCalc.lower() == "unit":
                    vol = (3 * self.l) * (2 * self.l * ((3 ** 0.5) / 2))
                elif stiffCalc.lower() == "lattice":
                    vol = L * H
            else:
                nny = nnx * 2 + 1
                H = 3 * self.l * nny
                vol = L * H
        self.nnx = nnx
        self.nny = int(round(nny))
        self.L = L
        self.H = H
        self.vol = vol

    def FTcalc(self):
        self.a0 = self.a0 - 0.25 * self.W
        if self.LAT.lower() == "fcc":
            self.ai = [self.a0 + ((self.l / 2) * (i)) for i in range(self.nnx)]
        elif "square" in self.LAT.lower():
            self.ai = [self.a0 + ((self.l * 2**(1/2)) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "tri":
            self.ai = [self.a0 + ((0.5 * (3.0 ** 0.5) * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "kagome":
            self.ai = [self.a0 + ((2 * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "hex":
            self.ai = [self.a0 + (((3.0 ** 0.5) * (self.l / 2)) * (i)) for i in range(self.nnx)]

    def nodeCount(self, mode=False, stiffMatrix=False):
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "45square":
            totalNodes = int(round((self.nnx + 1) * (self.nny + 1) + self.nnx * self.nny))
            totalBracketNodes = int(round((self.nnx + 5) * 3 * 2 + (self.nnx + 4) * 3 * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 1.66666667)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 8
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)
        elif self.LAT.lower() == "square":
            totalNodes = int(round((nnx + 1) * (nny + 1)))
            totalBracketNodes = int(round((nnx + 5) * 3 * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 4
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1))
        elif self.LAT.lower() == "tri":
            totalNodes = int(round(((self.nnx / 1.99999) + 1) * (self.nny + 1)) +
                             round((self.nnx / 1.99999) * self.nny))
            totalBracketNodes = int(round(((self.nnx / 1.99999) + 3) * 3 * 2) +
                                    round(((self.nnx / 1.99999) + 2) * 3 * 2) +
                                    2 * (self.nnx / 2.0 + 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 3.33333333)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 4
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)
        elif self.LAT.lower() == "kagome":
            totalNodes = int(round((2 * self.nnx * (self.nny + 1)) +
                                   (self.nnx - 1) * math.ceil(self.nny / 2.0) +
                                   (self.nnx) * math.floor(self.nny / 2)))
            totalBracketNodes = int(round(((2 * self.nnx + 4) * 3 + (self.nnx + 2) * 2 + (self.nnx + 1)) * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 1.75)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 16
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2))
        elif self.LAT.lower() == "hex":
            totalNodes = int(round(2 * (self.nnx) * math.ceil(self.nny / 2.0) +
                                   2 * (self.nnx + 1) * math.ceil(self.nny / 2.0)))
            totalBracketNodes = int(round(((self.nnx + 5) * 4 + (self.nnx + 4) * 4) * 2 + 4))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 12
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = ((2*nny) * (nnx+1)) + (((2*nny)+1) * nnx) + 50

    def UTval(self):
        self.nnx = 20
        self.l = 10.0
        self.nny = 18
        self.H = self.nny * self.l
        self.vol = self.L * self.H
        self.totalNodes = int(round(((self.nnx / 1.99999) + 1) * (self.nny + 1)) +
                              round((self.nnx / 1.99999) * self.nny))
        self.totalBracketNodes = int(round(((self.nnx / 1.99999) + 3) * 3 * 2) +
                                     round(((self.nnx / 1.99999) + 2) * 3 * 2) +
                                     2 * (self.nnx / 2.0 + 2))


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

def export_nodes(job, totalNodes):
    with open(job, 'r') as f:
        lines = f.readlines()
    
    nodes_start = int([lines.index(line) for line in lines if "*Node" in line][0]) + 1
    nodes_end = int(nodes_start + totalNodes)
    nodes = [[float(i.strip().strip('\n')) for i in line.split(",")] for line in lines[nodes_start:nodes_end]]
    
    with open(job.split('.inp')[0]+"\\"+"NodesElems.csv", 'w+') as f:
        f.write("*Nodes\n")
        for node in nodes:
            f.write("{}, {}, {}\n".format(int(node[0]), node[1], node[2]))
    return np.array(nodes)

def connectivity(LAT, nodes, geom, job=None, stiff=False, mode=None):
    radius = geom.l + geom.l*1e-3
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
                    (nodes[dummyElem[i][1]][1] == geom.H and nodes[dummyElem[i][2]][1] == geom.H)):
                    indexRemove.append(i)
    realElem = np.delete(dummyElem, [indexRemove], axis=0)
    if stiff:
        realElem = realElem
    else:
        realElem = realElem + 1

    if mode and mode.lower() == "fracture":
        xCrS = [-0.1*geom.W, geom.H/2-geom.l*0.2]                                         # crack starting point bottomLeft
        xCrE = [geom.a0 - 0.2*geom.l, geom.H/2+geom.l*0.2]
    
        delElems = []
        for ik in range(0,len(realElem)):
            x1 = nodes[int(realElem[ik][1]-1)][1]
            x2 = nodes[int(realElem[ik][2]-1)][1]
            y1 = nodes[int(realElem[ik][1]-1)][2]
            y2 = nodes[int(realElem[ik][2]-1)][2]
            midPointX = (x1+x2)/2
            midPointY = (y1+y2)/2
            point = (midPointX,midPointY)
            insideTest = insidePoint(xCrS,xCrE,point)
            if insideTest:
                delElems.append(realElem[ik][0])

        delElems = np.array(delElems, dtype=int)
        realElem = np.delete(realElem, delElems, 0)
    
    for i in range(len(realElem)):
        realElem[i][0] = i+1

    if job is not None:
        with open(job.split('.inp')[0]+"\\"+"NodesElems.csv", 'a') as f:
            f.write("*Elems\n")
            for elem in realElem:
                f.write("{}, {}, {}\n".format(int(elem[0]), int(elem[1]), int(elem[2])))
    return realElem


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
            
            nodes = export_nodes(inpPath, geom.totalNodes)
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
        nodes = export_nodes(inpPath, geom.totalNodes)
        elems = connectivity(inpPath, LAT, nodes, geom)
        
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
        nodes = export_nodes(inpPath, geom.totalNodes)
        elems = connectivity(LAT, nodes, geom, job=inpPath)
