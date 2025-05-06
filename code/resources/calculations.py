import numpy as np
import pandas as pd


def smooth(y_old):
    y_new = []
    for indx, yy in enumerate(y_old):
        if indx == 0 or indx == len(y_old)-1:
            y_new.append(yy)
        elif indx == 1 or indx == len(y_old)-2:
            new = (y_old[indx-1] + yy + y_old[indx+1])/3
            y_new.append(new)
        else:
            new = (y_old[indx-2] + y_old[indx-1] + yy + y_old[indx+1] + y_old[indx+2])/5
            y_new.append(new)
    return y_new


def get_nodes(nodesCSV, lineStart=None, lineEnd=None):
    with open(nodesCSV, 'r') as f:
        lines = f.readlines()
    
    if lineStart:   
        lines = lines[lineStart:lineEnd]   
    node_lines = [line.split(',') for line in lines]
    
    nodes = [[float(elem.strip('\n').strip()) for elem in line] for line in node_lines]
    nodesCoords = [node[1:] for node in nodes]
    nodes_df = pd.DataFrame(nodesCoords, columns=['x','y'])
    
    return nodes, nodesCoords, nodes_df

def get_struts(thicksCSV):
    with open(thicksCSV, 'r') as f:
        lines = f.readlines()
    
    thicks = [float(line.strip('\n')) for line in lines]
    
    return thicks


def get_ductileData(CSVout, crit=0.25, delimiter=','):
    output_df = pd.read_csv(CSVout, names=['x', 'y'], usecols=['x', 'y'], delimiter=delimiter)
    e = [0] + output_df.x.tolist()[1:]
    s = [0] + output_df.y.tolist()[1:]
    s_sm = smooth(smooth(s))
    output_df["x"], output_df["y"], output_df["y_sm"] = e, s, s_sm
    
    s_max_indx = s_sm.index(max(s_sm))
    frac = 0.0
    for indx, row in output_df[s_max_indx:].iterrows():
        if row[2] <= crit*max(s_sm):
            frac = indx
            break
    output_df.loc[-1] = [frac, frac, frac]
    output_df.index = output_df.index + 1
    output_df.sort_index(inplace=True)
    return output_df

def calcUT(df):
    frac = int(df.x.tolist()[0])
    df = df[1:].reset_index(drop=True)
    e = df.x.tolist()
    s_sm = df.y_sm.tolist()
    
    ductility = e[frac]
    strength = max(s_sm)
    stiff_rng = range(int(0.1*s_sm.index(strength)))
    stiffness = np.average([s_sm[i+1]-s_sm[i] for i in stiff_rng])/np.average([e[i+1]-e[i] for i in stiff_rng])
    
    return ductility, strength, stiffness


def get_fractureData(outputCSV):
    output_df = pd.read_csv(outputCSV, names=['x', 'y'], usecols=['x', 'y'])
    d = [0] + [dd/1000 for dd in output_df.x.tolist()[1:]]
    F = [0] + output_df.y.tolist()[1:]
    F_sm = smooth(smooth(F))
    output_df["x"], output_df["y"], output_df["y_sm"] = d, F, F_sm
    
    status_df = pd.read_csv(outputCSV, header=None).drop(columns=[0, 1])
    frac = 0.0
    for indx, row in status_df.iterrows():
        if 0.0 in list(row):
            frac = F_sm.index(max(F_sm[:indx]))
            break
    output_df.loc[-1] = [frac, frac, frac]
    output_df.index = output_df.index + 1
    output_df.sort_index(inplace=True)
    return output_df

def calc_FaW(a, W):
    return ((2+(a/W))*(0.886+(4.64*(a/W))-(13.32*((a/W)**2))+(14.72*((a/W)**3))-(5.6*((a/W)**4))))/((1-(a/W))**(3/2))

def calc_Apl(d, F_sm, dd, P, frac, n):
    if n == 0:
        slope_idx = int(0.15*d.index(dd))
        slope = np.average([F_sm[i+1]-F_sm[i] for i in range(slope_idx)])/np.average([d[i+1]-d[i] for i in range(slope_idx)])

        upper_curve = F_sm[:frac]
        interval = dd / len(upper_curve)
        F_integral = sum([Fi*interval for Fi in upper_curve])

        d_slope = dd - (P/slope)
        slope_integral = 0.5 * (dd - d_slope) * P

        A_pl = F_integral - slope_integral
    else:
        pass
    return A_pl

def calcFT(df, geom, E_eff, n_Ks=1, validation=False, E=None):  
    frac = int(df.x.tolist()[0])
    df = df[1:].reset_index(drop=True)
    d = df.x.tolist()
    F_sm = df.y_sm.tolist()
    
    W = geom[4]
    B = geom[5]
    ai = geom[7]
    
    if validation == True:
        E_eff = E          # CHECK VAL
    
    P = max(F_sm[:frac])
    frac = F_sm.index(P)
    dd = d[frac]
    
    Ks = []
    Kjs = []
    for n in range(n_Ks):
        f_a_W = calc_FaW(ai[n], W)
        K = (P/(B*(W**(1/2)))) * f_a_W
        
        J_el = ((K**2)*(1-(dd**2))) / E_eff
        A_pl = calc_Apl(d, F_sm, dd, P, frac, n)
        if n == 0:
            J_pl = ((2+(0.522*(W-ai[n])/W))*A_pl) / (B*(W-ai[n]))
        else:
            pass

        J = J_el + J_pl
        Kj = (E_eff*J)**0.5
                
        Ks.append(K)
        Kjs.append(Kj)
    
    return P, dd, Ks, Kjs
