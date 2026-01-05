from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys
import math
import os

mode = "any"                             # "ductile", "fracture", "both", "any"
unitCellSize = 1.459 #10.0
Cmatrix = False

LAT = "FCC"
DIS = "per"
nnx = 20
simN = 1
odbName = f"{LAT}-{nnx}-{DIS}-{simN}"

initial = 1
numberOfRuns = 1
expected_steps = 201

pDir = "Z:\p1\sims\SiC\Validation" #C:\\temp"

cmdIN = sys.argv[8:]
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
    Cmatrix = False
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

if Cmatrix:
    BCtype = "periodic"
    pDir = f"Z:\\p1\\sims\\Ti\\stiffMatrix\\Cmatrix-{BCtype}"

os.chdir(pDir)


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

def get_DuctData(Job, H, L, B):
    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = [["U2", "RF2"]]
    nodeSets = ["Set-top"]
    if Cmatrix and case_Cmatrix.lower() == 'a':
        variables = [["U1", "RF1"], ["U1", "RF2"]]
        if BCtype.lower() == "kubc": nodeSets = ["Set-right", "Set-topBody"]
        elif BCtype.lower() == "periodic": nodeSets = ["RP_Per_X", "RP_TB"]
    if Cmatrix and case_Cmatrix.lower() == 'b':
        variables = [["U2", "RF2"], ["U2", "RF1"]]
        if BCtype.lower() == "kubc": nodeSets = ["Set-topBody", "Set-right"]
        elif BCtype.lower() == "periodic": nodeSets = ["RP_Per_Y", "RP_LR"]
    if Cmatrix and case_Cmatrix.lower() == 'c':
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
                if case_Cmatrix.lower() == 'a':
                    if i == 0:
                        e = Usum/numNodes/L
                        s = RFsum/(H*B)
                    elif i == 1:
                        e = all_SE[0][j][0]
                        s = RFsum/(L*B)
                if case_Cmatrix.lower() == 'b':
                    if i == 0:
                        e = Usum/numNodes/H
                        s = RFsum/(L*B)
                    elif i == 1:
                        e = all_SE[0][j][0]
                        s = RFsum/(H*B)
                if case_Cmatrix.lower() == 'c':
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
    
    if Cmatrix and case_Cmatrix.lower() == 'c':
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

def get_FracData(Job):
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


if mode.lower() == 'any':
    for curDirectory, folders, files in os.walk(pDir):
        odbs = [f for f in files if f.endswith('.odb')]
        for odb in odbs:
            odbPath = os.path.join(curDirectory, odb)
            if not os.path.exists(curDirectory + "/transfer"):
                os.makedirs(curDirectory + "/transfer")
            
            LAT = odb.split('-')[1]
            unitCellSize = unitCellSize
            nnx = int(odb.split('-')[2])
            if Cmatrix:
                case_Cmatrix = list(odb.split('-')[3])[-1].lower()
            geom = Geometry(LAT, unitCellSize, nnx) 
            H, L, B = geom.H, geom.L, geom.B
            if 'Ductile' in odb:
                Job = odbPath
                data = curDirectory + "/transfer/OUT-" + odb.split('.')[0]
                OUT = get_DuctData(Job, H, L, B)
                for i, out in enumerate(OUT):
                    if Cmatrix:
                        np.savetxt(data+f'-{i}.csv', out, delimiter=",")
                    else:
                        np.savetxt(data+'.csv', out, delimiter=",")
            elif 'Fracture' in odb:
                Job = odbPath
                data = curDirectory + "/transfer/OUT-" + odb.split('.')[0]
                OUT = get_FracData(Job)
                np.savetxt(data+'.csv', OUT, delimiter=",")
            

if (mode.lower() == 'ductile' or mode.lower() == 'both'):
    if not os.path.exists("transfer"):
        os.makedirs("transfer")
    
    geom = Geometry(LAT, unitCellSize, nnx)
    H, L, B = geom.H, geom.L, geom.B
    MechMode = 'Ductile'
    for kk in range(initial, initial+numberOfRuns):
        Job = f"{MechMode}-{odbName}.odb"
        data = f"transfer/OUT-{MechMode}-{odbName}.csv"
        OUT = get_DuctData(Job, H, L, B)
        np.savetxt(data, OUT, delimiter=",")


if (mode.lower() == 'fracture' or mode.lower() == 'both'):
    if not os.path.exists("transfer"):
        os.makedirs("transfer")
    
    MechMode = 'Fracture'
    for kk in range(initial, initial+numberOfRuns):
        Job = f"{MechMode}-{odbName}.odb"
        data = f"transfer/OUT-{MechMode}-{odbName}.csv"
        OUT = get_FracData(Job)
        np.savetxt(data, OUT, delimiter=",")