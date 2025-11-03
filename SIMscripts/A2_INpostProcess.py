import os
import numpy as np
import math
import sys

stiffMatrix = False
distribution = "lhs_uniform"

pDir = "C:\\temp"

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
    pDir = "Z:\\p1\\sims\\Ti\\StiffMatrix"

os.chdir(pDir)

class Geometry:
    def __init__(self, LAT, l, nnx, rD=0.2):
        self.LAT = LAT
        self.l = l
        self.nnx = nnx
        self.rD = rD

        self.t = self.rDthickness(rD=rD)

        if LAT.lower() == 'fcc' or LAT.lower() == 'fcc2':
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
        if self.LAT.lower() == "fcc" or self.LAT.lower() == 'fcc2':
            A = 2*(1+np.sqrt(2))
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
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "fcc2":
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
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "fcc2":
            self.a0 = self.a0 - 0.25 * self.W
            self.ai = [self.a0 + ((self.l / 2) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "tri":
            self.a0 = self.a0 - 0.25 * self.W
            self.ai = [self.a0 + ((0.5 * (3.0 ** 0.5) * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "kagome":
            self.a0 = self.a0 - 0.25 * self.W
            self.ai = [self.a0 + ((2 * self.l) * (i)) for i in range(self.nnx)]
        elif self.LAT.lower() == "hex":
            self.a0 = self.a0 - 0.25 * self.W
            self.ai = [self.a0 + (((3.0 ** 0.5) * (self.l / 2)) * (i)) for i in range(self.nnx)]

    def nodeCount(self, mode=False, stiffMatrix=False):
        if self.LAT.lower() == "fcc" or self.LAT.lower() == "fcc2":
            totalNodes = int(round((self.nnx + 1) * (self.nny + 1) + self.nnx * self.nny))
            totalBracketNodes = int(round((self.nnx + 5) * 3 * 2 + (self.nnx + 4) * 3 * 2))
            if mode and mode.lower() == "fracture":
                self.totalNodes = totalNodes - round(self.nnx / 1.66666667)
            elif mode and mode.lower() == "ductile":
                self.totalNodes = totalNodes + totalBracketNodes - 8
            if stiffMatrix:
                nnx, nny = 10, 10
                self.totalNodes = int((nnx + 1) * (nny + 1)) + int(nnx * nny)
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


def export_frequencies(inpFile, expFile):
    with open(inpFile, 'r') as f:
        lines = f.readlines()

    freq_start = int([lines.index(line) for line in lines if "**FREQUENCIES:" in line][0]) + 1
    freq_end = int([lines.index(line) for line in lines if "**END FREQUENCIES" in line][0])
    frequencies = [line.strip().strip("**") for line in lines[freq_start:freq_end]]
    
    with open(expFile, 'w') as f:
        for freq in frequencies:
            f.write(freq + '\n')

def export_nodes(inpFile, expFile):
    LAT = inpFile.split('-')[1]
    nnx = int(inpFile.split('-')[2])
    l = 10.0
    if "Fracture" in inpFile:
        mode = 'fracture'
    elif "Ductile" in inpFile:
        mode = 'ductile'
    
    geom = Geometry(latticeType, unitCellSize, nnx)
    if stiffMatrix:
        geom.stiffnessMatrix()
    geom.nodeCount(mode=mode, stiffMatrix=stiffMatrix)
    totalNodes = geom.totalNodes
    
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

freq = False
if distribution.lower() == "frequency" or distribution.lower() == "opt-f":
    freq = True

if not os.path.exists("transfer"):
    os.makedirs("transfer")

for file in os.scandir():
    if 'per' in file.name or 'disNodes' in file.name:
        if file.name.endswith('.inp'):
            expFile_n = "transfer/IN-n" + file.name[:-4].replace('_','-') + ".csv"
            expFile_f = "transfer/IN-f" + file.name[:-4].replace('_','-') + ".csv"
            export_nodes(file.name, expFile_n)
            if freq:
                export_frequencies(file.name, expFile_f)

    if 'per' in file.name or 'disStruts' in file.name:
        if file.name.endswith('.inp'):
            expFile = "transfer/IN-s" + file.name[:-4].replace('_','-') + ".csv"
            export_struts(file.name, expFile)