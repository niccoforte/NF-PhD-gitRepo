from abaqus import *
from abaqusConstants import *
from odbAccess import *
import numpy as np
import sys
import math
import os

mode = "any"                             # "ductile", "fracture", "both", "any"
unitCellSize = 10.0

LAT = "FCC"
DIS = "disNodes"
nnx = 16

initial = 1
numberOfRuns = 1
expected_steps = 201

pDir = r"C:\\temp" #Users\\exy053\\Documents\\TargetedDisorder\\xs" #\\MeshConv\\1-5-1"

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

    return H, L


def get_DuctData(Job, H, L):
    odb = openOdb(path=Job) 
    step = "Step-1"
    variables = ["U2", "RF2"]

    reg_load = 'Node '
    
    U2s = []
    RF2s = []
    for reg in odb.steps[step].historyRegions.keys():
        if reg_load in reg:
            try:
                U2 = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[0]].data]
                RF2 = [float(i[1]) for i in odb.steps[step].historyRegions[reg].historyOutputs[variables[1]].data]
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
            U2s.append(U2)
            RF2s.append(RF2)
            
    
    U2s = np.transpose(U2s)
    RF2s = np.transpose(RF2s)
    
    numNodes = len(U2s[0])
    strain = []
    stress = []
    for Us_step, RFs_step in zip(U2s, RF2s):
        Usum = 0.0
        RFsum = 0.0
        for U, RF in zip(Us_step, RFs_step):
            Usum += U
            RFsum += RF
        e = Usum/numNodes/H
        s = RFsum/(L*1)
        strain.append(e)
        stress.append(s)

    STEPS_OUT = np.transpose([strain, stress])
    odb.close()
    return STEPS_OUT

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
            H, L = geometry(LAT, unitCellSize, nnx, brackets=True)
            if 'Ductile' in odb:
                Job = odbPath
                data = curDirectory + "/transfer/OUT-" + odb.split('.')[0] + '.csv'
                OUT = get_DuctData(Job, H, L)
                np.savetxt(data, OUT, delimiter=",")
            elif 'Fracture' in odb:
                Job = odbPath
                data = curDirectory + "/transfer/OUT-" + odb.split('.')[0] + '.csv'
                OUT = get_FracData(Job)
                np.savetxt(data, OUT, delimiter=",")
            

if (mode.lower() == 'ductile' or mode.lower() == 'both'):
    if not os.path.exists("transfer"):
        os.makedirs("transfer")
    
    H, L = geometry(LAT, unitCellSize, nnx, brackets=True)
    MechMode = 'Ductile'
    for kk in range(initial, initial+numberOfRuns):
        Job = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".odb"
        data = "transfer/OUT-" + MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".csv"
        OUT = get_DuctData(Job, H, L)
        np.savetxt(data, OUT, delimiter=",")


if (mode.lower() == 'fracture' or mode.lower() == 'both'):
    if not os.path.exists("transfer"):
        os.makedirs("transfer")
    
    MechMode = 'Fracture'
    for kk in range(initial, initial+numberOfRuns):
        Job = MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".odb"
        data = "transfer/OUT-" + MechMode + "-" + LAT + "-" + str(nnx) + "-" + DIS + "-" + str(kk) + ".csv"
        OUT = get_FracData(Job)
        np.savetxt(data, OUT, delimiter=",")