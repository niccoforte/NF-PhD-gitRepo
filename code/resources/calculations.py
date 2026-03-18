import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import copy


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

def get_frequencies(freqCSV):
    with open(freqCSV, 'r') as f:
        lines = f.readlines()
    
    freqs = [float(line.strip('\n')) for line in lines]
    
    return freqs

def get_struts(thicksCSV):
    with open(thicksCSV, 'r') as f:
        lines = f.readlines()
    
    thicks = [float(line.strip('\n')) for line in lines]
    
    return thicks


def get_ductileData(CSVout, crit=0.25, delimiter=',', typ='n'):
    if typ == 'n':
        output_df = pd.read_csv(CSVout, names=['x', 'y'], usecols=['x', 'y'], delimiter=delimiter)
    elif typ == 'a':
        output_df = pd.read_csv(CSVout, names=['i', 'x', 'y'], usecols=['i', 'x', 'y'], delimiter=delimiter)
        output_df = output_df.drop(columns=['i'])
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
    final = None
    for indx, row in output_df[s_max_indx:].iterrows():
        if row[2] <= 0.01*max(s_sm):
            final = indx-1
            break
    if final is not None:
        output_df.y_sm = s_sm[:final+1] + [0]*(len(output_df)-final-1)
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
    
    idx_Smax = s_sm.index(strength)
    idx_WoF = None
    for si in s_sm[idx_Smax:]:
        if si <= 0.01*strength:
            idx_WoF = s_sm[1:].index(si)+1
            break
    if idx_WoF is not None:
        work_of_frac = np.trapz(s_sm[:idx_WoF], e[:idx_WoF])
    else:
        work_of_frac = np.trapz(s_sm, e)

    return ductility, strength, stiffness, work_of_frac


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
            frac = F_sm.index(np.max(F_sm[:indx], initial=0))
            break    
    F_max_indx = F_sm.index(max(F_sm))
    final = None
    for indx, row in output_df[F_max_indx:].iterrows():
        if row[2] <= 0.01*max(F_sm):
            final = indx-1
            break
    if final is not None:
        output_df.y_sm = F_sm[:final+1] + [0]*(len(output_df)-final-1)
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

def calc_p_poly(lam_bar, rho_bar, a_W):
    v_lr = np.array([
        1.0,
        lam_bar,
        rho_bar,
        lam_bar**2,
        lam_bar * rho_bar,
        rho_bar**2,
        lam_bar**3,
        lam_bar**2 * rho_bar,
        lam_bar * rho_bar**2,
    ])

    C = np.array([
        [ 0.30817 ,  2.1609  , -5.72   ,  6.2711 , -2.4512  ],
        [ 0.61118 , -4.5341  , 12.698  , -14.428 ,  5.5896  ],
        [ 0.098042,  2.3871  , -8.6206 , 10.742  , -4.3791  ],
        [ 0.1443  ,  0.098426,  0.47793, -2.5788 ,  1.9266  ],
        [-1.2451  ,  8.9848  , -22.996 , 26.09   , -11.037  ],
        [ 0.85309 , -3.104   ,  7.3777 , -9.0334 ,  4.0197  ],
        [-0.14716 ,  1.752   , -6.0346 ,  7.7217 , -3.3098  ],
        [-0.16912 ,  0.82943 , -2.7216 ,  4.5667 , -2.6053  ],
        [ 0.12556 , -0.35037 , -0.53346,  0.90753, -0.042327],
    ])

    v_a = np.array([1.0, a_W, a_W**2, a_W**3, a_W**4])

    p = v_lr @ C @ v_a
    return float(p)

def calc_FaW_aniso(a, W, C):
    from resources.lattices import calc_anisoParams
    lambda_aniso, rho_aniso = calc_anisoParams(C)
    if lambda_aniso < 0.03162 or lambda_aniso > 10.0 or rho_aniso < 0.1 or rho_aniso > 10.0:
        print(f"WARNING: lambda: {lambda_aniso} or rho: {rho_aniso} out of bounds for f(a/W) calculation.")

    a_bar = np.log10(a/W)
    lambda_bar = np.log10(lambda_aniso)
    rho_bar = np.log10(rho_aniso+1)
    D = -0.066112 + 0.75681*a_bar - 0.015*rho_bar + 0.58136*(a_bar**2) - 0.08451*a_bar*rho_bar

    f_a_W = calc_p_poly(lambda_bar, rho_bar, a/W)*(lambda_aniso**D)*((1+0.006689*rho_aniso)**0.47151)*((2*(2+(a/W)))/((1-(a/W))**(3/2)))*(((2*(lambda_aniso**(3/2)))/(1+rho_aniso))**(1/4))
    return f_a_W

def calcFT(df, geom, E_eff_pe, n_Ks=1, validation=False, E=123e9, C=None):  
    frac = int(df.x.tolist()[0])
    df = df[1:].reset_index(drop=True)
    d = df.x.tolist()
    F_sm = df.y_sm.tolist()
    
    W = geom.W
    B = geom.B
    ai = geom.ai
    if validation == True:
        E_eff_pe = E          # TODO: CHECK FT VAL
    
    P = F_sm[frac]
    dd = d[frac]
    
    Ks = []
    Kjs = []
    for n in range(n_Ks):
        if geom.iso == True:
            f_a_W = calc_FaW(ai[n], W)
        elif geom.iso == False:
            if C is None:
                from resources.lattices import calcC_mohr
                C = calcC_mohr(copy.deepcopy(geom), "unit", E_s=E)[0]
            f_a_W = calc_FaW_aniso(ai[n], W, C)
        K = (P/(B*(W**(1/2)))) * f_a_W

        J_el = (K**2) / E_eff_pe
        A_pl = calc_Apl(d, F_sm, dd, P, frac, n)
        if n == 0:
            J_pl = ((2+(0.522*(W-ai[n])/W))*A_pl) / (B*(W-ai[n]))
        else:
            pass

        J = J_el + J_pl
        Kj = (E_eff_pe*J)**0.5
                
        Ks.append(K)
        Kjs.append(Kj)
    
    return P, dd, Ks, Kjs


def FEA_run(x_new, iter, path, argv):
    new_sample_file = f"{path}/Opt/BO_sample{iter}.txt"
    np.savetxt(new_sample_file, x_new.flatten()[np.newaxis], fmt='%.6f')

    try:
        print("-> Running Abaqus simulation and post-processing...")
        abq_argv = f"{argv} {iter}"
        
        abq_run_file = r"C:\\Users\\exy053\\Documents\\p1git-Lattices\\SIMscripts\\A1_FractureToughness-Ductility.py"
        abq_run_cmd = f"abaqus cae noGUI={abq_run_file} -- {abq_argv}"
        subprocess.run(abq_run_cmd, shell=True, check=True)

        abq_inPP_file = r"C:\\Users\\exy053\\Documents\\p1git-Lattices\\SIMscripts\\A2_INpostProcess.py"
        abq_inPP_cmd = f"abaqus cae noGUI={abq_inPP_file} -- {abq_argv}"
        subprocess.run(abq_inPP_cmd, shell=True, check=True)

        abq_outPP_file = r"C:\\Users\\exy053\\Documents\\p1git-Lattices\\SIMscripts\\A2_OUTpostProcess.py"
        abq_outPP_cmd = f"abaqus cae noGUI={abq_outPP_file} -- {abq_argv}"
        subprocess.run(abq_outPP_cmd, shell=True, check=True)

        CSVout = f"{path}/Opt/transfer/OUT-Ductile-{argv[0]}-{int(argv[1])}-{int(argv[7]*100)}{argv[6]}-opt-{argv[9]}-{iter}.csv"
        UTdf = get_ductileData(CSVout, crit=0.25)
        ductility, strength, stiffness = calcUT(UTdf)
        score = ductility
        return score
    
    except subprocess.CalledProcessError as e:
        print(f"!!! An Abaqus process failed: {e}")
        return 0

    except Exception as e:
        print(f"!!! An unexpected error occurred in the FEA workflow: {e}")
        return 0


def plot_curve(df_list, typ, label=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(15)
    
    for l, df in enumerate(df_list):
        frac = int(df.x.tolist()[0])
        x = df.x.tolist()[1:]
        y = df.y.tolist()[1:]
        y_sm = df.y_sm.tolist()[1:]
        
        if label:
            l = label
        
        ax1.plot(x, y, label=l)
        ax2.plot(x, y_sm, label=l)
        
        if len(df_list) <= 1:
            ax1.axhline(y=y[frac], c='r')
            ax1.axvline(x=x[frac], ymax=0.6, c='r')
            ax1.axhline(y=0.25*max(y_sm), c='g')
            ax2.axhline(y=0.25*max(y_sm), c='g')
            ax2.axhline(y=y_sm[frac], c='r')
            ax2.axvline(x=x[frac], ymax=0.6, c='r')

            if typ.lower() == "ft":
                slope_loc = 0.15*x.index(x[frac])
                slope_idx = int(slope_loc)

                ax1.axvline(x=0.15*x[frac], ymax=0.6, c='g')
                ax2.axvline(x=0.15*x[frac], ymax=0.6, c='g')

                ax2.plot([x[0], x[slope_idx]], [y_sm[0], y_sm[slope_idx]], c='r')
                ax2.plot([x[0], x[slope_idx]], [y[0], y[slope_idx]], c='g')

                slope = np.average([y_sm[i+1]-y_sm[i] for i in range(slope_idx)])/np.average([x[i+1]-x[i] for i in range(slope_idx)])
                ax1.plot([i for i in x[:slope_idx]], [i*slope for i in x[:slope_idx]], c='b')
                ax2.plot([i for i in x[:slope_idx]], [i*slope for i in x[:slope_idx]], c='b')

    # ax1.grid()
    # ax2.grid()
    ax1.legend()
    ax2.legend()
    plt.show()