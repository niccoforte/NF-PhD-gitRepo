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


def filter_force(time, force, cutoff, order=2, zero_phase=False):
    """Uniform-grid Butterworth; restore genuine times, never interpolate STATUS.

    DC startup and causal second-order filtering follow Abaqus defaults. Small
    output-time jitter is interpolated explicitly; gaps >1.5 nominal intervals
    are rejected. A noncausal sensitivity is separately labelled.
    """
    from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

    time, force = np.asarray(time, float), np.asarray(force, float)
    if len(time) < 4 or len(time) != len(force):
        raise ValueError("Need at least four paired samples")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(force)):
        raise ValueError("Missing/nonfinite continuous histories cannot be filtered")
    dt = np.diff(time)
    if np.any(dt <= 0) or np.max(dt) > 1.5 * np.median(dt):
        raise ValueError("Time grid is nonmonotone or contains a missing interval")
    grid = np.linspace(time[0], time[-1], len(time))
    fs = (len(time) - 1) / (time[-1] - time[0])
    if not 0 < cutoff < fs / 2:
        raise ValueError("Cutoff must be below native-history Nyquist frequency")
    uniform_force = np.interp(grid, time, force)
    sos = butter(order, cutoff, fs=fs, output="sos")
    if zero_phase:
        filtered = sosfiltfilt(sos, uniform_force)
    else:
        filtered, _ = sosfilt(sos, uniform_force,
                              zi=sosfilt_zi(sos) * uniform_force[0])
    return np.interp(time, grid, filtered), {
        "fs_Hz": fs, "nyquist_Hz": fs / 2, "cutoff_Hz": cutoff,
        "order": order, "zero_phase": zero_phase,
        "max_time_remap_s": float(np.max(np.abs(grid - time))),
    }


def initial_work_stiffness(x, y, end_fraction=.4, min_samples=6, baseline_response=0.):
    """Initial work-equivalent slope over one declared low-load interval.

    Match initial work W=E*x_end^2/2 through the first end_fraction crossing.
    Under linear elasticity this equals the ordinary modulus/load stiffness.
    The physical unloaded origin is used, not one noisy initial reaction.
    A known preload can be supplied explicitly as baseline_response.
    The estimate is not asserted to be a homogenised material tensor.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned finite observations required")
    if not 0 < end_fraction < 1:
        raise ValueError("Initial loading fraction must lie inside zero and one")
    if not np.isfinite(baseline_response):
        raise ValueError("Known finite baseline required")
    u, p = x-x[0], y-float(baseline_response)
    p=p.copy();p[0]=0.
    peak = int(np.argmax(p))
    if peak == 0 or p[peak] <= 0:
        return {"status": "unresolved_positive_initial_loading", "estimate": None}
    crossed = np.flatnonzero(p[:peak+1] >= end_fraction*p[peak])
    if not len(crossed):
        return {"status": "unobserved_initial_fraction", "estimate": None}
    end = int(crossed[0])
    a, b = u[:end+1], p[:end+1]
    if len(a) < 2 or np.any(np.diff(a) <= 0) or a[-1] <= 0:
        return {"status": "unresolved_monotone_initial_coordinate", "estimate": None, "end_index": end}
    work = np.r_[0., np.cumsum(.5*(b[1:]+b[:-1])*np.diff(a))]
    q = a*a
    estimate = float(2*work[-1]/q[-1])
    span = float(np.ptp(b))
    residual = float(np.sqrt(np.mean((b-estimate*a)**2))/max(span, 1e-30))
    energy_residual = float(np.sqrt(np.mean((2*work-estimate*q)**2))/max(abs(2*work[-1]), 1e-30))
    return dict(status="resolved_estimate" if len(a) >= min_samples and estimate > 0 else "low_resolution_estimate",
                estimate=estimate, end_index=end, samples=len(a), peak_index=peak,
                end_fraction=end_fraction, end_coordinate=float(a[-1]), initial_work=float(work[-1]),
                secant_energy_estimate=float(2*work[-1]/a[-1]**2),
                linearity_residual_over_stress_span=residual,
                energy_fit_residual=energy_residual, baseline_x=float(x[0]), baseline_y=float(baseline_response),
                observed_initial_response=float(y[0]),
                definition="E=2*integral(response dcoordinate)/coordinate_end^2 through declared filtered-peak fraction; physical unloaded origin")


def stress_to_mpa(stress):
    stress = float(stress)
    return stress / 1e6 if abs(stress) > 1e6 else stress


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
        if row["y_sm"] <= crit*max(s_sm):
            frac = indx
            break
    final = None
    for indx, row in output_df[s_max_indx:].iterrows():
        if row["y_sm"] <= 0.01*max(s_sm):
            final = indx-1
            break
    if final is not None:
        output_df.y_sm = s_sm[:final+1] + [0]*(len(output_df)-final-1)
    output_df.loc[-1] = [frac, frac, frac]
    output_df.index = output_df.index + 1
    output_df.sort_index(inplace=True)
    return output_df

def calcUT(df):
    frac = int(df.y_sm.tolist()[0])
    df = df[1:].reset_index(drop=True)
    e = np.asarray(df.x.tolist(), dtype=float)
    s_sm = np.asarray(df.y_sm.tolist(), dtype=float)
    
    ductility = e[frac]
    strength = np.max(s_sm)
    idx_Smax = int(np.argmax(s_sm))
    
    stiff_end = int(0.1 * idx_Smax)
    fit_e, fit_s = e[:stiff_end + 1], s_sm[:stiff_end + 1]
    # A line needs two distinct finite samples, not three. Keep the same window.
    if (stiff_end < 1 or not np.isfinite(fit_e).all() or not np.isfinite(fit_s).all()
            or len(np.unique(fit_e)) < 2):
        stiffness = np.nan
    else:
        stiffness, _ = np.polyfit(fit_e, fit_s, 1)

    idx_WoF = None
    for idx in range(idx_Smax, len(s_sm)):
        if s_sm[idx] <= 0.01 * strength:
            idx_WoF = idx
            break
    if idx_WoF is None:
        idx_WoF = len(s_sm) - 1

    work_of_frac = np.trapz(s_sm[:idx_WoF + 1], e[:idx_WoF + 1])

    return ductility, strength, stiffness, work_of_frac


def get_fractureData(outputCSV):
    output_df = pd.read_csv(outputCSV, names=['x', 'y'], usecols=['x', 'y'])
    d = [0] + output_df.x.tolist()[1:]
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
        if row["y_sm"] <= 0.01*max(F_sm):
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
        frac_idx = int(frac)
        d = np.asarray(d, dtype=float)
        F_sm = np.asarray(F_sm, dtype=float)

        if frac_idx <= 0 or frac_idx >= len(d):
            return np.nan

        slope_idx = max(1, int(0.15 * frac_idx))
        slope_d = d[:slope_idx + 1]
        slope_F = F_sm[:slope_idx + 1]
        if len(np.unique(slope_d)) < 2:
            return np.nan
        slope, _ = np.polyfit(slope_d, slope_F, 1)
        if not np.isfinite(slope) or slope == 0:
            return np.nan

        F_integral = np.trapz(F_sm[:frac_idx + 1], d[:frac_idx + 1])
        slope_integral = 0.5 * (P / slope) * P

        A_pl = F_integral - slope_integral
    else:
        pass
    return A_pl

def calc_p_poly(lam_bar, chi_bar, a_W):
    v_lr = np.array([
        1.0,
        lam_bar,
        chi_bar,
        lam_bar**2,
        lam_bar * chi_bar,
        chi_bar**2,
        lam_bar**3,
        lam_bar**2 * chi_bar,
        lam_bar * chi_bar**2,
    ])

    C = np.array([
        [ 0.30817 ,  2.1609  ,  -5.72   ,   6.2711 ,  -2.4512  ],
        [ 0.61118 , -4.5341  ,  12.698  , -14.428  ,   5.5896  ],
        [ 0.098042,  2.3871  ,  -8.6206 ,  10.742  ,  -4.3791  ],
        [ 0.1443  ,  0.098426,   0.47793,  -2.5788 ,   1.9266  ],
        [-1.2451  ,  8.9848  , -22.996  ,  26.09   , -11.037  ],
        [ 0.85309 , -3.104   ,   7.3777 ,  -9.0334 ,   4.0197  ],
        [-0.14716 ,  1.752   ,  -6.0346 ,   7.7217 ,  -3.3098  ],
        [-0.16912 ,  0.82943 ,  -2.7216 ,   4.5667 ,  -2.6053  ],
        [ 0.12556 , -0.35037 ,  -0.53346,   0.90753,  -0.042327],
    ])

    v_a = np.array([1.0, a_W, a_W**2, a_W**3, a_W**4])

    p = v_lr @ C @ v_a
    return float(p)

def calc_FaW_aniso(a, W, C=None, S=None):
    """Ortega's fitted C(T) factor for explicitly supplied source-axis C or S.

    C/S must already describe the intended constraint, with source axis1
    along loading and axis2 along the crack, in order (1,2,6). The returned
    factor uses Ortega's source-axis stress-intensity convention. Pair it
    with the conversion modulus in that same order, or convert both to the
    crack-axis1 convention. This pure fit does not infer plane strain or
    reorder axes. chi is Ortega's rho, not density.
    """
    from resources.lattices import calc_anisoParams
    lambda_aniso, chi = calc_anisoParams(C=C, S=S)
    if not 0 < a < W:
        raise ValueError("Ortega's C(T) fit requires 0 < a < W.")
    if lambda_aniso < 0.03162 or lambda_aniso > 10.0 or chi < 0.1 or chi > 10.0:
        print(f"WARNING: lambda: {lambda_aniso} or chi: {chi} out of bounds for f(a/W) calculation.")

    a_bar = np.log10(a/W)
    lambda_bar = np.log10(lambda_aniso)
    chi_bar = np.log10(chi+1)
    D = -0.066112 + 0.75681*a_bar - 0.015*chi_bar + 0.58136*(a_bar**2) - 0.08451*a_bar*chi_bar

    f_a_W = calc_p_poly(lambda_bar, chi_bar, a/W)*(lambda_aniso**D)*((1+0.006689*chi)**0.47151)*((2*(2+(a/W)))/((1-(a/W))**(3/2)))*(((2*(lambda_aniso**(3/2)))/(1+chi))**(1/4))
    return f_a_W

def ortega_crack_axis_factor(a, W, elastic, source_axis1=1):
    """Convert the loading-axis Ortega factor to the supplied crack-axis M."""
    from resources.lattices import calc_anisoParams

    if source_axis1 not in (0, 1):
        raise ValueError("Ortega source axis must be 0 (x) or 1 (y)")
    order = [source_axis1, 1-source_axis1, 2]
    S_source = elastic["S_xy_pe"][np.ix_(order, order)]
    lam, chi = calc_anisoParams(S=S_source)
    source_modulus = np.sqrt(2 * lam**1.5 / (1 + chi)) / S_source[0, 0]
    return calc_FaW_aniso(a, W, S=S_source) * np.sqrt(elastic["M_I_pe"] / source_modulus)


def compact_tension_initiation(displacement, force, index, a, B, W, modulus,
                               stiffness, geometry_factor):
    """C(T) J at a verified native endpoint, with N, mm and MPa inputs.

    The supplied modulus and geometry factor must use the same crack-axis
    convention. Stiffness is the initial cracked-specimen load slope in N/mm.
    Signed area and an elastic-only result remain available to diagnose a
    failed work partition. Reported K values are in MPa sqrt(m).
    """
    from scipy.integrate import trapezoid

    d, f = np.asarray(displacement, float), np.asarray(force, float)
    if (d.ndim != 1 or d.shape != f.shape or not isinstance(index, (int, np.integer))
            or not 0 < index < len(d)):
        raise ValueError("An observed noninitial sample on paired histories is required")
    d, f = d[:index + 1], f[:index + 1]
    if not (np.isfinite(d).all() and np.isfinite(f).all()) or np.any(np.diff(d) < 0):
        raise ValueError("Endpoint must bound a finite monotone loading history")
    if (not np.all(np.isfinite([a, B, W, modulus, stiffness, geometry_factor]))
            or not 0 < a < W or min(B, modulus, stiffness, geometry_factor, f[-1]) <= 0):
        raise ValueError("Positive geometry, modulus, stiffness and endpoint load required")

    load = float(f[-1])
    work = float(trapezoid(f, d))
    elastic_work = load**2 / (2 * stiffness)
    area = work - elastic_work
    # Round-off at an exactly elastic endpoint must not masquerade as
    # resolved negative plastic work. Keep the signed area for diagnostics.
    numerical_zero = 32 * np.finfo(float).eps * max(abs(work), elastic_work)
    valid = bool(area >= -numerical_zero)
    eta = 2 + .522 * (W - a) / W
    ki = load * geometry_factor / (B * np.sqrt(W))
    jel = ki**2 / modulus
    jpl_signed = eta * area / (B * (W - a))
    signed_total = jel + jpl_signed
    jpl = eta * max(area, 0.) / (B * (W - a)) if valid else None
    total = jel + jpl if valid else None
    return {
        "sample_index": int(index), "displacement_mm": float(d[-1]), "force_N": load,
        "work_N_mm": work, "elastic_work_N_mm": elastic_work, "Apl_N_mm": area,
        "stiffness_N_mm": float(stiffness), "modulus_MPa": float(modulus), "eta": float(eta),
        "Jel_N_per_mm": float(jel), "Jpl_N_per_mm": float(jpl) if valid else None,
        "Jpl_signed_diagnostic_N_per_mm": float(jpl_signed),
        "J_N_per_mm": float(total) if valid else None,
        "KI_MPa_sqrt_m": float(ki / np.sqrt(1000)),
        "KJ_signed_area_diagnostic_MPa_sqrt_m": float(np.sqrt(modulus * signed_total / 1000))
        if signed_total >= 0 else None,
        "KJ_MPa_sqrt_m": float(np.sqrt(modulus * total / 1000)) if valid else None,
        "valid_plastic_area": valid,
    }


def calcFT(df, geom, E_eff_pe, n_Ks=1, validation=False, E=123e9, C=None,
           C_state="plane_stress", v_s=0.3, ortega_axis1=None):
    """Calculate legacy initiation measures with a consistent FCC elastic chain.

    E_eff_pe is the mode-I conversion modulus M, not a directional Young's
    modulus. FCC physical C is reduced using the prismatic mixture model;
    declare C_state='plane_strain' for already-reduced beam homogenisation.
    Unequal-axial-modulus FCC requires an explicit Ortega source-axis map
    (0=x or 1=y). Ortega's factor is converted to the crack-axis1 convention
    of the supplied M by preserving K**2/M. Equal axial moduli make that
    conversion immaterial.
    The plastic-area calibration and historical validation override are
    retained, not independently established by this elastic correction.
    """
    if not getattr(geom, "ftcalc_done", False):
        geom.FTcalc()

    frac = int(df.y_sm.tolist()[0])
    df = df[1:].reset_index(drop=True)
    d = df.x.tolist()
    F_sm = df.y_sm.tolist()
    
    if validation == True:
        E_eff_pe = E          # TODO: CHECK FT VAL
    E_eff_pe = stress_to_mpa(E_eff_pe)

    fcc_elastic = None
    if geom.LAT.lower() == "fcc":
        from resources.lattices import calcC_mohr, prismatic_plane_strain
        if C is None:
            if C_state != "plane_stress":
                raise ValueError("Supply C explicitly when it is already plane-strain reduced.")
            C = calcC_mohr(copy.deepcopy(geom), "unit", E_s=E)[0]
        fcc_elastic = prismatic_plane_strain(C, geom.rD, E, v_s, C_state=C_state)
        M_I_pe = stress_to_mpa(fcc_elastic["M_I_pe"])
        if not np.isclose(E_eff_pe, M_I_pe, rtol=1e-6, atol=0):
            raise ValueError("FCC E_eff_pe must equal M_I_pe from the same reduced compliance used for its geometry factor.")
        if ortega_axis1 not in (None, 0, 1):
            raise ValueError("ortega_axis1 must be 0 (x), 1 (y), or None.")
        if (ortega_axis1 is None
                and not np.isclose(fcc_elastic["lambda_pe"], 1., rtol=1e-6, atol=0)):
            raise ValueError("Unequal FCC axial moduli require an explicitly verified Ortega source-axis mapping.")

    W = geom.W
    B = geom.B
    ai = np.asarray(geom.ai, dtype=float)
    
    P = F_sm[frac]
    dd = d[frac]
    
    Ks = []
    Kjs = []
    for n in range(n_Ks):
        if geom.iso == True:
            f_a_W = calc_FaW(ai[n], W)
        elif geom.iso == False:
            if fcc_elastic is not None:
                source_axis1 = 1 if ortega_axis1 is None else ortega_axis1
                f_a_W = ortega_crack_axis_factor(ai[n], W, fcc_elastic, source_axis1)
            elif C is None:
                from resources.lattices import calcC_mohr
                C = calcC_mohr(copy.deepcopy(geom), "unit", E_s=E)[0]
                f_a_W = calc_FaW_aniso(ai[n], W, C)
            else:
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
    if typ.lower() == "ut":
        ax1.set_xlabel("Strain")
        ax1.set_ylabel("Stress [MPa]")
        ax2.set_xlabel("Strain")
        ax2.set_ylabel("Smoothed Stress [MPa]")
    elif typ.lower() == "ft":
        ax1.set_xlabel("Displacement [mm]")
        ax1.set_ylabel("Force [N]")
        ax2.set_xlabel("Displacement [mm]")
        ax2.set_ylabel("Smoothed Force [N]")
    ax1.legend()
    ax2.legend()
    plt.show()
