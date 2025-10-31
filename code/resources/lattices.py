import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    # def brackets(self): # add vol if don't delete
    #     nnx = self.nnx + 4
    #     nny = self.nny + 6
    #     H = self.l * nny
    #     if self.LAT.lower() == "fcc" or self.LAT.lower() == "fcc2":
    #         L = float(self.l * nnx)
    #         totalNodes = int(round((nnx + 1) * (nny + 1) + nnx * nny))
    #         totalBracketNodes = int(round((nnx + 5) * 3 * 2 + (nnx + 4) * 3 * 2))
    #     elif self.LAT.lower() == "tri":
    #         L = 0.5 * (3.0 ** 0.5) * self.l * nnx         
    #         totalNodes = int(round(((nnx / 1.99999) + 1) * (nny + 1)) + round((nnx / 1.99999) * nny))
    #         totalBracketNodes = int(round(((nnx / 1.99999) + 3) * 3 * 2) +
    #                                 round(((nnx / 1.99999) + 2) * 3 * 2) +
    #                                 2 * (nnx / 2.0 + 2))
    #     elif self.LAT.lower() == "kagome":
    #         nnx = self.nnx + 2
    #         L = self.l * (2.0 * nnx - 1)
    #         H = (3.0 ** 0.5) * self.l * nny
    #         totalNodes = int(round((2*nnx*(nny+1)) + (nnx-1)*math.ceil(nny/2.0) + (nnx)*math.floor(nny/2)))
    #         totalBracketNodes = int(round(((2 * nnx + 4) * 3 + (nnx + 2) * 2 + (nnx + 1)) * 2))
    #     elif self.LAT.lower() == "hex":
    #         nny = self.nny + 8
    #         L = (3.0 ** 0.5) * self.l * nnx
    #         H = (0.5 * self.l) + (1.5 * self.l * nny)
    #         totalNodes = int(round(2*(nnx)*math.ceil(nny/2.0) + 2*(nnx+1)*math.ceil(nny/2.0)))
    #         totalBracketNodes = int(round(((nnx + 5) * 4 + (nnx + 4) * 4) * 2 + 4))
    #     self.nnx = nnx
    #     self.nny = int(round(nny))
    #     self.L = L
    #     self.H = H
    #     self.totalNodes = totalNodes
    #     self.totalBracketNodes = totalBracketNodes
    #     self.B = 0.5 * self.W

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


def pStrainProperties(E, v):
    return E/(1-v**2), v/(1-v)

def effProperties(LAT, E_s, rD):
    if LAT.lower() == "fcc":
        E = 1
        v = 0.3
    elif LAT.lower() == "tri":
        E = E_s*(1/3)*(rD**(1))
        v = 0.25
    elif LAT.lower() == "kagome":
        E = E_s*(1/3)*(rD**(1))
        v = 0.25
    elif LAT.lower() == "hex":
        E = E_s*(3/2)*(rD**(3))
        v = 0.25
    return E, v


def find_nodes(LAT, geom, dis, mode='lattice', stiff=False, path="Z:/p1/sims/Ti", sim=1):
    l = geom.l
    if mode.lower() == "unit":
        if LAT.lower() == "fcc":
            nodes = np.array([[0,0,0],
                              [0,l,0],
                              [l,l,0],
                              [l,0,0],
                              [l/2,l/2,0]])
        elif LAT.lower() == "tri":
            nodes = np.array([[0,0,0],
                              [l,0,0],
                              [l/2,l*(np.sqrt(3)/2),0],
                              [0,l*(np.sqrt(3)/2),0],
                              [l,l*(np.sqrt(3)/2),0],
                              [0,2*l*(np.sqrt(3)/2),0],
                              [l,2*l*(np.sqrt(3)/2),0]])
        elif LAT.lower() == "kagome":
            nodes = np.array([[l*(np.sqrt(3)/2),0,0],
                              [l*(np.sqrt(3)/2),l,0],
                              [0,1.5*l,0],
                              [l*(np.sqrt(3)/2),2*l,0],
                              [l*(np.sqrt(3)/2),3*l,0],
                              [2*l*(np.sqrt(3)/2),2.5*l,0],
                              [3*l*(np.sqrt(3)/2),3*l,0],
                              [3*l*(np.sqrt(3)/2),2*l,0],
                              [4*l*(np.sqrt(3)/2),1.5*l,0],
                              [3*l*(np.sqrt(3)/2),l,0],
                              [3*l*(np.sqrt(3)/2),0,0],
                              [2*l*(np.sqrt(3)/2),0.5*l,0]])
        elif LAT.lower() == "hex":
            nodes = np.array([[0,l*(np.sqrt(3)/2),0],
                              [0.5*l,l*(np.sqrt(3)/2),0],
                              [l,2*l*(np.sqrt(3)/2),0],
                              [2*l,2*l*(np.sqrt(3)/2),0],
                              [2.5*l,l*(np.sqrt(3)/2),0],
                              [3*l,l*(np.sqrt(3)/2),0],
                              [2*l,0,0],
                              [l,0,0]])
        Dnodes = nodes

    elif mode.lower() == "lattice":
        pDir = path
        if stiff:
            pDir = "Z:/p1/sims/Ti/stiffMatrix"
        nodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom.nnx) + "-per-1.csv"
        
        nodes_df = pd.read_csv(nodeFile, header=None, usecols=[1, 2])
        nodes = nodes_df.to_numpy() / 1000
    
        if stiff:
            if LAT.lower() == "tri":
                ys = [0, geom.H]
                xs = [l*(np.sqrt(3)/2) + l*np.sqrt(3)*i for i in range(geom.nnx)]
                add_nodes = []
                for y in ys:
                    for x in xs:
                        add_nodes.append([x, y])
                nodes = np.append(nodes, add_nodes, axis=0)

            del_nodes1 = np.argwhere(nodes[:, 1] > geom.H+1e-6)
            del_nodes2 = np.argwhere(nodes[:, 1] < 0)
            del_nodes3 = np.argwhere(nodes[:, 0] > geom.L+1e-6)
            del_nodes4 = np.argwhere(nodes[:, 0] < 0)
            del_nodes = np.concatenate([del_nodes1, del_nodes2, del_nodes3, del_nodes4])
            nodes = np.delete(nodes, del_nodes, axis=0)
        Dnodes = nodes
      
        if dis.lower() == "dn":
            DnodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom.nnx) + "-20disNodes-lhs-all-" + str(sim) + ".csv"
            Dnodes_df = pd.read_csv(DnodeFile, header=None, usecols=[1, 2])
            Dnodes = Dnodes_df.to_numpy() / 1000
            if stiff:
                if LAT.lower() == "tri":
                    Dnodes = np.append(Dnodes, add_nodes, axis=0)
                elif LAT.lower() == "hex":
                    topNodes_idxs = np.argwhere(nodes[:, 1] == geom.H).flatten()
                    DtopNodes_idxs1 = np.argwhere(Dnodes[:, 1] > geom.H-0.25*geom.l)
                    DtopNodes_idxs2 = np.argwhere(Dnodes[:, 1] < geom.H+0.25*geom.l)
                    DtopNodes_idxs = np.intersect1d(DtopNodes_idxs1, DtopNodes_idxs2)
                    Dnodes[DtopNodes_idxs] = nodes[topNodes_idxs]
                Dnodes = np.delete(Dnodes, del_nodes, axis=0)
    return nodes, Dnodes

def connectivity(LAT, nodes, geom, stiff=False, mode=None):
    radius = geom.l*(1+1e-3)
    dummyElem = []
    count = 0
    for ii in range(len(nodes)):
        if (LAT.lower() == "fcc" and nodes[ii][0]*1000)%2 == 1.0 and (nodes[ii][1]*1000)%2 == 1.0:
            continue
        distance = np.sqrt(np.array(nodes[ii, 0] - nodes[:, 0])**2 + np.array(nodes[ii, 1] - nodes[:, 1])**2)
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
    for i in range(len(realElem)):
        realElem[i][0] = i+1
    return realElem


def plot_lattice(elems, nodes, geom):
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    for elem, tt in zip(elems, geom.t):
        node1 = nodes[elem[1]]
        node2 = nodes[elem[2]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], linewidth=tt*2000, c='black')
    plt.show()


### Stiffness matrix calculation functions
def edgeElems(nodes, elems, geom):
    t_old = geom.t
    t_new = np.array([t_old] * len(elems))
    min_x, max_x = min(nodes[:,0]), max(nodes[:,0])
    min_y, max_y = min(nodes[:,1]), max(nodes[:,1])
    for i, elem in enumerate(elems):
        node1 = nodes[elem[1]]
        node2 = nodes[elem[2]]
        if (node1[0] == max_x and node2[0] == max_x) or (node1[0] == min_x and node2[0] == min_x):
            t_new[i] = t_old/2
        elif (node1[1] == max_y and node2[1] == max_y) or (node1[1] == min_y and node2[1] == min_y):
            t_new[i] = t_old/2
    geom.t = t_new
    return geom

def get_n0s(nodes, realElem):
    n0s = []
    for elem in realElem:
        n0 = nodes[elem[2]] - nodes[elem[1]]
        n0s.append(n0)
    n0s = np.array(n0s)
    return n0s

def get_ns(n0s):
    ns = []
    for n0 in n0s:
        ns.append(n_values(n0))
    return np.array(ns)
    
def n_values(n0):
    return np.array([np.cos(np.arctan([n0[1]/n0[0]]))[0], np.sin(np.arctan([n0[1]/n0[0]]))[0]])

def get_Nmatrix(ns):
    N = []
    for i in range(len(ns)):
        N.append([ns[i][0]*ns[i][0], ns[i][1]*ns[i][1], ns[i][0]*ns[i][1]])
    return np.array(N)

def calc_c(n0, geom, i):
    return (geom.t[i]*(np.sqrt(n0[0]**2 + n0[1]**2))) / (geom.vol)

def calc_Kmatrix(LAT, l, nnx, mode, dis='per', count=0, plot=False):
    rD = 0.2
    E_s = 123e9
    v_s = 0.3

    geom = Geometry(LAT, l, nnx, stiffCalc=True, mode=mode)
    
    nodes, Dnodes = find_nodes(LAT, geom, dis, mode=mode, stiff=True)
    elems = connectivity(LAT, nodes, stiff=True, geom=geom, mode=mode)
    geomK = edgeElems(nodes, elems, geom)

    if plot:
        plot_lattice(elems, nodes, geomK)

    n0s = get_n0s(nodes, elems)
    ns = get_ns(n0s)
    Nmatrix = get_Nmatrix(ns)

    c0matrix = np.zeros((len(ns),len(ns)))
    for i in range(len(ns)):
        c0matrix[i][i] = calc_c(n0s[i], geomK, i)
    #print(np.sum(c0matrix))
    Kmatrix = E_s*np.matmul(np.matmul(Nmatrix.T, c0matrix), Nmatrix).round(10)
    
    Ks = []
    Ks.append(Kmatrix)
    
    for jj in range(1, count+1):
        nodes, Dnodes = find_nodes(LAT, geom, dis, mode=mode, stiff=True, sim=jj)
        if dis.lower() == 'dn':                                   # import struts thicks from struts.csv file
            nodes = Dnodes
        
        if plot:
            plot_lattice(elems, nodes, geomK)
            
        n0s = get_n0s(nodes, elems)
        ns = get_ns(n0s)
        Nmatrix = get_Nmatrix(ns)

        c0matrix = np.zeros((len(ns),len(ns)))
        for i in range(len(ns)):
            c0matrix[i][i] = calc_c(n0s[i], geomK, i)
        #print(np.sum(c0matrix))
        Kmatrix = E_s*np.matmul(np.matmul(Nmatrix.T, c0matrix), Nmatrix).round(10)
        Ks.append(Kmatrix)

    return np.array(Ks)

def check_isotropy(K):
    if round(K[0][0]/K[0][1], 3) == 3.0 and round(K[0][0]/K[1][1], 3) == 1.0 and round(K[0][1]/K[2][2], 3) == 1.0:
        return True
    else:
        return False

def calc_effectiveProperties(K):
    iso = check_isotropy(K)
    v = 1/(K[0][0]/K[0][1]+1)
    E = (K[0][0]*((1+v)*(1-2*v)))/(1-v)
    return E, v, iso


def plot_IsotropyVariation(Ks, stiff=True, zener=True, properties=True):
    if stiff:
        K11, K33, K13, K23 = [], [], [], []
        for K in Ks[1:]:
            K11.append((Ks[0][0][0]-K[0][0])/Ks[0][0][0])
            K33.append((Ks[0][2][2]-K[2][2])/Ks[0][2][2])
            K13.append(K[0][2]/Ks[0][0][0])
            K23.append(K[1][2]/Ks[0][0][0])


        fig1, (ax1, ax2) = plt.subplots(1, 2)
        fig1.set_figheight(6)
        fig1.set_figwidth(18)

        ax1.set_ylabel('$(K^{p}-K^{d})/K^{p}$ [%]', fontsize=14, fontname="Times New Roman")
        ax1.set_xlabel('Randomly Generated Model No.', fontsize=14, fontname="Times New Roman")

        ax1.set_ylim([-0.1, 0.1])
        #ax1.set_xticks([i+1 for i in range(len(K11))])

        ax1.plot([i+1 for i in range(len(K11))], K11, 'bo-', label='$K_{11}$')
        ax1.plot([i+1 for i in range(len(K33))], K33, 'r^-', label='$K_{33}$')

        # ax1.grid()
        ax1.legend()

        ax2.set_ylabel('$K^{d}/K^{p}_{11}$', fontsize=14, fontname="Times New Roman")
        ax2.set_xlabel('Randomly Generated Model No.', fontsize=14, fontname="Times New Roman")

        ax2.set_ylim([-0.05, 0.05])
        #ax2.set_xticks([i+1 for i in range(len(K11))])

        ax2.plot([i+1 for i in range(len(K13))], K13, 'b^-', label='$K^{d}_{13}$')
        ax2.plot([i+1 for i in range(len(K23))], K23, 'ro-', label='$K^{d}_{23}$')

        # ax2.grid()
        ax2.legend()

    if zener:
        Zp  = 2*Ks[0][2][2]/(Ks[0][0][0]-Ks[0][0][1])
        Zs = []
        for Z in Ks[1:]:
            Zs.append(2*Z[2][2]/(Z[0][0]-Z[0][1]))

        fig2, ax3 = plt.subplots(1, 1)
        fig2.set_figheight(5)
        fig2.set_figwidth(9)

        ax3.set_ylabel('Zener Ratio, $a$', fontsize=15, fontname="Times New Roman")
        ax3.set_xlabel('Randomly Generated Model No.', fontsize=15, fontname="Times New Roman")

        # ax3.set_ylim([Zp-0.15, Zp+0.15])
        # ax3.set_xticks([i+1 for i in range(len(K11))])
        ax3.axhline(y=Zp, color='k', linestyle='--', label='Perfect Lattice Zener Ratio, $a^{p}$ = %.2f' % Zp)
        ax3.plot([i+1 for i in range(len(Zs))], Zs, 'bo-', label='Random Lattice Zener Ratio, $a^{d}$')

        # ax3.grid()
        ax3.legend()

    if properties:
        vp = 1/(Ks[0][0][0]/Ks[0][0][1] + 1)
        Ep = (Ks[0][0][0]*((1+vp)*(1-2*vp)))/(1-vp)
        vs, Es = [], []
        for K in Ks[1:]:
            vp = 1/(K[0][0]/K[0][1] + 1)
            Ep = (K[0][0]*((1+vp)*(1-2*vp)))/(1-vp)
            vs.append(vp)
            Es.append(Ep)
        
        fig3, ax4 = plt.subplots(1, 1)
        fig3.set_figheight(5)
        fig3.set_figwidth(9)
        ax5 = ax4.twinx()

        ax4.set_ylabel("Poisson's Ratio, $\\nu$", fontsize=15, fontname="Times New Roman")
        ax4.set_xlabel('Randomly Generated Model No.', fontsize=15, fontname="Times New Roman")
        ax5.set_ylabel("Young's Modulus, $E$ [GPa]", fontsize=15, fontname="Times New Roman")

        # ax4.set_ylim([vp-0.05, vp+0.05])
        # ax5.set_ylim([min(Es)/1e9 - 5, max(Es)/1e9 + 5])
        # ax4.set_xticks([i+1 for i in range(len(K11))])
        ax4.axhline(y=vp, color='b', linestyle='--', label="Perfect Lattice Poisson's Ratio, $\\nu^{p}$ = %.2f" % vp)
        ax4.plot([i+1 for i in range(len(vs))], vs, 'bo-', label="Random Lattice Poisson's Ratio, $\\nu^{d}$")
        ax5.axhline(y=Ep/1e9, color='r', linestyle='--', label="Perfect Lattice Young's Modulus, $E^{p}$ = %.2f GPa" % (Ep/1e9))
        ax5.plot([i+1 for i in range(len(Es))], np.array(Es)/1e9, 'r^-', label="Random Lattice Young's Modulus, $E^{d}$")

        # ax4.grid()
        ax4.legend()
        ax5.legend()
    
    plt.show()

