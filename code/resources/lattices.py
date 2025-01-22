import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    return [nnx, nny, L, H, W, B, a0, ai, totalNodes, totalBracketNodes, vol, l, t, LAT]  # len=14


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


def get_nodes(LAT, geom, dis, mode='lattice', stiff=False, path="C:/Users/exy053/Documents/", sim=1):
    if mode.lower() == "unit":
        if LAT.lower() == "fcc":
            nodes = np.array([[0,0,0],
                              [0,geom[11],0],
                              [geom[11],geom[11],0],
                              [geom[11],0,0],
                              [geom[11]/2,geom[11]/2,0]])
        elif LAT.lower() == "tri":
            nodes = np.array([[0,0,0],
                              [geom[11],0,0],
                              [geom[11]/2,geom[11]*(np.sqrt(3)/2),0],
                              [0,geom[11]*(np.sqrt(3)/2),0],
                              [geom[11],geom[11]*(np.sqrt(3)/2),0],
                              [0,2*geom[11]*(np.sqrt(3)/2),0],
                              [geom[11],2*geom[11]*(np.sqrt(3)/2),0]])
        elif LAT.lower() == "kagome":
            nodes = np.array([[geom[11]*(np.sqrt(3)/2),0,0],
                              [geom[11]*(np.sqrt(3)/2),geom[11],0],
                              [0,1.5*geom[11],0],
                              [geom[11]*(np.sqrt(3)/2),2*geom[11],0],
                              [geom[11]*(np.sqrt(3)/2),3*geom[11],0],
                              [2*geom[11]*(np.sqrt(3)/2),2.5*geom[11],0],
                              [3*geom[11]*(np.sqrt(3)/2),3*geom[11],0],
                              [3*geom[11]*(np.sqrt(3)/2),2*geom[11],0],
                              [4*geom[11]*(np.sqrt(3)/2),1.5*geom[11],0],
                              [3*geom[11]*(np.sqrt(3)/2),geom[11],0],
                              [3*geom[11]*(np.sqrt(3)/2),0,0],
                              [2*geom[11]*(np.sqrt(3)/2),0.5*geom[11],0]])
        elif LAT.lower() == "hex":
            nodes = np.array([[0,geom[11]*(np.sqrt(3)/2),0],
                              [0.5*geom[11],geom[11]*(np.sqrt(3)/2),0],
                              [geom[11],2*geom[11]*(np.sqrt(3)/2),0],
                              [2*geom[11],2*geom[11]*(np.sqrt(3)/2),0],
                              [2.5*geom[11],geom[11]*(np.sqrt(3)/2),0],
                              [3*geom[11],geom[11]*(np.sqrt(3)/2),0],
                              [2*geom[11],0,0],
                              [geom[11],0,0]])
        Dnodes = nodes

    elif mode.lower() == "lattice":
        pDir = path
        if stiff:
             pDir = "C:/Users/exy053/Documents/stiffMatrix"
        nodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom[0]) + "-per-1.csv"
        
        nodes_df = pd.read_csv(nodeFile, header=None, usecols=[1, 2])
        nodes = nodes_df.to_numpy() / 1000
    
        if stiff:
            if LAT.lower() == "tri":
                ys = [0, geom[3]]
                xs = [geom[11]*(np.sqrt(3)/2) + geom[11]*np.sqrt(3)*i for i in range(geom[0])]
                add_nodes = []
                for y in ys:
                    for x in xs:
                        add_nodes.append([x, y])
                nodes = np.append(nodes, add_nodes, axis=0)

            del_nodes1 = np.argwhere(nodes[:, 1] > geom[3]+1e-6)
            del_nodes2 = np.argwhere(nodes[:, 1] < 0)
            del_nodes3 = np.argwhere(nodes[:, 0] > geom[2]+1e-6)
            del_nodes4 = np.argwhere(nodes[:, 0] < 0)
            del_nodes = np.concatenate([del_nodes1, del_nodes2, del_nodes3, del_nodes4])
            nodes = np.delete(nodes, del_nodes, axis=0)
        Dnodes = nodes
      
        if dis.lower() == "dn":
            DnodeFile = pDir + "/transfer/IN-nDuctile-" + LAT + "-" + str(geom[0]) + "-disNodes-" + str(sim) + ".csv"
            Dnodes_df = pd.read_csv(DnodeFile, header=None, usecols=[1, 2])
            Dnodes = Dnodes_df.to_numpy() / 1000
            if stiff:
                if LAT.lower() == "tri":
                    Dnodes = np.append(Dnodes, add_nodes, axis=0)
                elif LAT.lower() == "hex":
                    topNodes_idxs = np.argwhere(nodes[:, 1] == geom[3]).flatten()
                    DtopNodes_idxs1 = np.argwhere(Dnodes[:, 1] > geom[3]-0.25*geom[11])
                    DtopNodes_idxs2 = np.argwhere(Dnodes[:, 1] < geom[3]+0.25*geom[11])
                    DtopNodes_idxs = np.intersect1d(DtopNodes_idxs1, DtopNodes_idxs2)
                    Dnodes[DtopNodes_idxs] = nodes[topNodes_idxs]
                Dnodes = np.delete(Dnodes, del_nodes, axis=0)
    return nodes, Dnodes

def connectivity(LAT, nodes, geom, stiff=False, mode=None):
    radius = geom[11] + geom[11]*1e-3
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
                    (nodes[dummyElem[i][1]][1] == geom[3] and nodes[dummyElem[i][2]][1] == geom[3])):
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
    for elem, tt in zip(elems, geom[12]):
        node1 = nodes[elem[1]]
        node2 = nodes[elem[2]]
        plt.plot([node1[0], node2[0]], [node1[1], node2[1]], linewidth=tt*2500, c='black')
    plt.show()