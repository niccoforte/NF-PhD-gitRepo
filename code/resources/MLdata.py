from resources.imports import *
from resources.calculations import calcUT, calcFT
from resources.lattices import Geometry, effProperties

from matplotlib.gridspec import GridSpec

from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import clone


def load_data(inputs, outputs, f_inputs=None, no_outliers=False):
    IN_df   = pd.read_csv(inputs, index_col=0)
    OUTr_df = pd.read_csv(outputs, index_col=0)
    INf_df  = None
    if f_inputs is not None:
        INf_df = pd.read_csv(f_inputs, index_col=0)
    if no_outliers:
        IN_df_noOutliers  = pd.read_csv(no_outliers[0], index_col=0)
        OUT_df_noOutliers = pd.read_csv(no_outliers[1], index_col=0)
        INf_df_noOutliers = None
        if no_outliers[2] is not None:
            INf_df_noOutliers = pd.read_csv(no_outliers[2], index_col=0)

    dIN_df  = IN_df - IN_df.iloc[0].values
    INfixed_cols = dIN_df.loc[:, (dIN_df == 0.0).all()].columns
    dIN_df  = dIN_df.drop(columns=INfixed_cols) 
    dOUT_df = OUTr_df - OUTr_df.iloc[1].values
    dOUT_df = dOUT_df.drop(columns='0')
    dOUT_df = dOUT_df.iloc[1:].sort_index()
    OUT_df  = OUTr_df.iloc[1:].sort_index()

    perINr_df = IN_df.loc[:0].T
    perINr_df = perINr_df.rename(columns={0: "in"})
    perINr_df = perINr_df
    perIN_df  = IN_df.loc[:0].drop(columns=INfixed_cols).T
    perIN_df  = perIN_df.rename(columns={0: "in"})
    perIN_df  = perIN_df
    perOUT_df = OUTr_df.iloc[:2].T.iloc[1:]
    perOUT_df.columns = ["x", "y"]

    if no_outliers:
        IN_df   = IN_df_noOutliers
        OUT_df  = OUT_df_noOutliers
        INf_df  = INf_df_noOutliers
        dIN_df  = IN_df - IN_df.iloc[0].values
        dIN_df  = dIN_df.drop(columns=INfixed_cols) 
        dOUT_df = OUT_df - OUT_df.iloc[1].values
        dOUT_df = dOUT_df.drop(columns='0')
    
    return IN_df, OUT_df, INf_df, perINr_df, perIN_df, perOUT_df, dIN_df, dOUT_df

def prep_UTdata(dIN_df, dOUT_df, perOUT_df, OUT_df, INf_df=None):
    dIN = dIN_df.to_numpy()
    dOUT = dOUT_df.to_numpy()
    xOUT = np.linspace(0, max(perOUT_df.x.tolist()), len(dOUT[0]))
    INf = None
    if INf_df is not None:
        INf = INf_df.to_numpy()
    
    ducts, strens, stiffs, WoFs = [], [], [], []
    for _, row in OUT_df.iterrows():
        UT_df = pd.DataFrame({'x':np.insert(xOUT,0,row[0]), 'y_sm':row})
        ductility, strength, stiffness, WoF = calcUT(UT_df)
        
        ducts.append(ductility)
        strens.append(strength)
        stiffs.append(stiffness)
        WoFs.append(WoF)
        
    props = np.array([ducts, strens, stiffs, WoFs])
    props_df = pd.DataFrame(props.T, columns=['Ductility', 'Strength', 'Stiffness', 'WoF'], index=OUT_df.index)
    return dIN, dOUT, INf, xOUT, props, props_df

def prep_FTdata(dIN_df, dOUT_df, perOUT_df, OUT_df, geom, E_eff_pe, INf_df=None):
    dIN = dIN_df.to_numpy()
    dOUT = dOUT_df.to_numpy()
    xOUT = np.linspace(0, max(perOUT_df.x.tolist()), len(dOUT[0]))
    INf = None
    if INf_df is not None:
        INf = INf_df.to_numpy()
    
    Kjs, Ks, Ps, ds = [], [], [], []
    for indx, row in OUT_df.iterrows():
        FT_df = pd.DataFrame({'x':np.insert(xOUT,0,row[0]), 'y_sm':row})
        P, dd, K, Kj = calcFT(FT_df, geom, E_eff_pe, n_Ks=1, iso="auto")
        
        Kjs.append(Kj[0])
        Ks.append(K[0])
        Ps.append(P)
        ds.append(dd)
    
    props = [Kjs, Ks, Ps, ds]
    props_df = pd.DataFrame(np.array(props).T, columns=['K_JIC', 'K_IC', 'Force', 'Displacement'], index=OUT_df.index)
    return dIN, dOUT, INf, xOUT, props, props_df

def prep_MULTIdata(IN_dfs, OUT_dfs, dIN_dfs, dOUT_dfs, props_dfs, INf_dfs, E_eff_pe):
    UT_props_df, FT_props_df = props_dfs
    UT_IN_df, FT_IN_df = IN_dfs
    UT_OUT_df, FT_OUT_df = OUT_dfs
    UT_dIN_df, FT_dIN_df = dIN_dfs
    UT_dOUT_df, FT_dOUT_df = dOUT_dfs
    if INf_dfs[0] is not None and INf_dfs[1] is not None:
        UT_INf_df, FT_INf_df = INf_dfs

    common_idxs = UT_props_df.index.intersection(FT_props_df.index)
    common_props_df = pd.concat([UT_props_df.loc[common_idxs], FT_props_df.loc[common_idxs]], axis=1)
    UT_IN_df, FT_IN_df = UT_IN_df.loc[common_idxs], FT_IN_df.loc[common_idxs]
    UT_OUT_df, FT_OUT_df = UT_OUT_df.loc[common_idxs], FT_OUT_df.loc[common_idxs]
    UT_dIN_df, FT_dIN_df = UT_dIN_df.loc[common_idxs], FT_dIN_df.loc[common_idxs]
    UT_dOUT_df, FT_dOUT_df = UT_dOUT_df.loc[common_idxs], FT_dOUT_df.loc[common_idxs]
    if INf_dfs[0] is not None and INf_dfs[1] is not None:
        UT_INf_df, FT_INf_df = UT_INf_df.loc[common_idxs], FT_INf_df.loc[common_idxs]

    norm_df = (common_props_df/common_props_df.iloc[0])
    common_props_df["Multi"] = norm_df["Ductility"]**2 + norm_df["K_JIC"]**2 + norm_df["WoF"] + norm_df["Displacement"] + norm_df["Strength"]
    common_props_df["FCL"] = (common_props_df["K_JIC"]**2 / E_eff_pe) / (common_props_df["WoF"] * 1e6)
    common_props_df = common_props_df.replace([np.inf, -np.inf], np.nan).dropna()
    return common_props_df, [UT_IN_df, FT_IN_df], [UT_OUT_df, FT_OUT_df], [UT_dIN_df, FT_dIN_df], [UT_dOUT_df, FT_dOUT_df], [UT_INf_df, FT_INf_df] if INf_dfs[0] is not None and INf_dfs[1] is not None else INf_dfs


def find_outliers(data):
    mean = np.mean(data)
    stdev = np.std(data)
    if type(data) is not list:
        data = data.tolist()    
    outlier_idxs = [data.index(x) for x in data if (x < mean - 3*stdev) or (x > mean + 3*stdev) if data.index(x) != 0]
    return np.array(outlier_idxs, dtype="int")

def remove_outliers(dIN_r, dOUT_r, props_r, IN_df, OUT_df, dIN_df, dOUT_df, props_df,INf_r=None, INf_df=None, manual=None):
    all_outlier_idxs = []
    for prop_r in props_r:
        idxs = find_outliers(data=prop_r)
        if len(idxs) == 0:
            continue
        for idx in idxs:
            all_outlier_idxs.append(idx)
    outlier_idxs = np.array(list(set(all_outlier_idxs)), dtype="int")
    
    if len(outlier_idxs) > 0:
        dIN = np.delete(dIN_r, outlier_idxs, axis=0)
        dOUT = np.delete(dOUT_r, outlier_idxs, axis=0)
        INf = None
        if INf_r is not None:
            INf = np.delete(INf_r, outlier_idxs, axis=0)

        IN_df = IN_df.drop(IN_df.iloc[outlier_idxs].index)
        OUT_df = OUT_df.drop(OUT_df.iloc[outlier_idxs].index)
        dIN_df = dIN_df.drop(dIN_df.iloc[outlier_idxs].index)
        dOUT_df = dOUT_df.drop(dOUT_df.iloc[outlier_idxs].index)
        props_df = props_df.drop(props_df.iloc[outlier_idxs].index)
        if INf_df is not None:
            INf_df = INf_df.drop(INf_df.iloc[outlier_idxs].index)
        props1 = []
        for prop_r in props_r:
            prop = np.delete(prop_r, outlier_idxs, axis=0)
            props1.append(prop)
        props1 = np.array(props1)
        
        if manual is not None:
            manual = np.array(manual, dtype="int")
            dIN = np.delete(dIN, manual, axis=0)
            dOUT = np.delete(dOUT, manual, axis=0)
            if INf is not None:
                INf = np.delete(INf, manual, axis=0)

            IN_df = IN_df.drop(IN_df.iloc[manual].index)
            OUT_df = OUT_df.drop(OUT_df.iloc[manual].index)
            dIN_df = dIN_df.drop(dIN_df.iloc[manual].index)
            dOUT_df = dOUT_df.drop(dOUT_df.iloc[manual].index)
            props_df = props_df.drop(props_df.iloc[manual].index)
            if INf_df is not None:
                INf_df = INf_df.drop(INf_df.iloc[manual].index)
            props = []
            for prop1 in props1:
                prop = np.delete(prop1, manual, axis=0)
                props.append(prop)
            props = np.array(props)
        else:
            props = props1

    else:
        dIN, dOUT, INf, props = dIN_r, dOUT_r, INf_r, props_r
    
    return dIN, dOUT, INf, props, IN_df, OUT_df, dIN_df, dOUT_df, props_df, INf_df

def split_data(dIN, dOUT, props, INf, split=0.85, split_idxs=None):
    if split_idxs is None:
        idxs = list(range(len(dOUT)))
        random.shuffle(idxs)
        train_idxs = idxs[:int(split*len(dOUT))]
        test_idxs = [i for i in idxs if i not in train_idxs]
        train_idxs, val_idxs = train_idxs[:int(split*len(train_idxs))], train_idxs[int(split*len(train_idxs)):]
    else:
        train_idxs, val_idxs, test_idxs = split_idxs
    
    train_in = dIN[train_idxs]
    val_in = dIN[val_idxs]
    test_in = dIN[test_idxs]

    train_out = dOUT[train_idxs]
    val_out = dOUT[val_idxs]
    test_out = dOUT[test_idxs]

    train_props = props[:, train_idxs]
    val_props = props[:, val_idxs]
    test_props = props[:, test_idxs]

    train_in_f, val_in_f, test_in_f = None, None, None
    if INf is not None:
        train_in_f = INf[train_idxs]
        val_in_f = INf[val_idxs]
        test_in_f = INf[test_idxs]
    
    train = [train_in, train_out, train_props, train_in_f]
    val = [val_in, val_out, val_props, val_in_f]
    test = [test_in, test_out, test_props, test_in_f]
    split_idxs = [train_idxs, val_idxs, test_idxs]
    
    return train, val, test, split_idxs

def save_MLdata(perIN_df, perOUT_df, train, val, test, IN_df, OUT_df, dIN_df, dOUT_df, INf_df, props_df, PATH, mode, dis):
    if mode == "UT":
        model = "Ductile"
    elif mode == "FT":
        model = "Fracture"
    os.makedirs(PATH+f"/MLdata/{mode}", exist_ok=True)
    
    perIN_df.to_csv(PATH + f"MLdata/{mode}/{mode}-perIN.csv")
    perOUT_df.to_csv(PATH + f"MLdata/{mode}/{mode}-perOUT.csv")

    IN_df.to_csv(PATH + f"{model}-{dis}-IN-noOutliers.csv")
    OUT_df.to_csv(PATH + f"{model}-{dis}-OUT-noOutliers.csv")

    dIN_df.to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-allIN.csv")
    pd.DataFrame(train[0]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-trainIN.csv")
    pd.DataFrame(val[0]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-valIN.csv")
    pd.DataFrame(test[0]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-testIN.csv")
    
    dOUT_df.to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-allOUT.csv")
    pd.DataFrame(train[1]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-trainOUT.csv")
    pd.DataFrame(val[1]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-valOUT.csv")
    pd.DataFrame(test[1]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-testOUT.csv")

    props_df.to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-allProps.csv")
    pd.DataFrame(train[2]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-trainProps.csv")
    pd.DataFrame(val[2]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-valProps.csv")
    pd.DataFrame(test[2]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-testProps.csv")

    if train[-1] is not None:
        INf_df.to_csv(PATH + f"{model}-{dis}-INf-noOutliers.csv")
        INf_df.to_csv(PATH + f"MLdata/{mode}-{dis}-allINf.csv")
        pd.DataFrame(train[3]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-trainINf.csv")
        pd.DataFrame(val[3]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-valINf.csv")
        pd.DataFrame(test[3]).to_csv(PATH + f"MLdata/{mode}/{mode}-{dis}-testINf.csv")

def save_MULTIdata(IN_dfs, OUT_dfs, dIN_dfs, dOUT_dfs, common_props_df, INf_dfs, PATH, dis):
    os.makedirs(PATH+"/MLdata/multi", exist_ok=True)

    UT_dIN = dIN_dfs[0].to_numpy()
    UT_dOUT = dOUT_dfs[0].to_numpy()
    
    FT_dIN = dIN_dfs[1].to_numpy()
    FT_dOUT = dOUT_dfs[1].to_numpy()

    common_props = common_props_df.to_numpy().T

    UT_INf = None
    FT_INf = None
    if INf_dfs[0] is not None and INf_dfs[1] is not None:
        UT_INf = INf_dfs[0].to_numpy()
        FT_INf = INf_dfs[1].to_numpy()

    UT_train, UT_val, UT_test, UT_split_idxs = split_data(UT_dIN, UT_dOUT, common_props, UT_INf, split_idxs=None)
    FT_train, FT_val, FT_test, FT_split_idxs = split_data(FT_dIN, FT_dOUT, common_props, FT_INf, split_idxs=UT_split_idxs)

    IN_dfs[0].to_csv(PATH + f"MULTI-Ductile-{dis}-IN-noOutliers.csv")
    IN_dfs[1].to_csv(PATH + f"MULTI-Fracture-{dis}-IN-noOutliers.csv")
    OUT_dfs[0].to_csv(PATH + f"MULTI-Ductile-{dis}-OUT-noOutliers.csv")
    OUT_dfs[1].to_csv(PATH + f"MULTI-Fracture-{dis}-OUT-noOutliers.csv")

    dIN_dfs[0].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-allIN.csv")
    dIN_dfs[1].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-allIN.csv")
    pd.DataFrame(UT_train[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-trainIN.csv")
    pd.DataFrame(UT_val[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-valIN.csv")
    pd.DataFrame(UT_test[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-testIN.csv")
    pd.DataFrame(FT_train[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-trainIN.csv")
    pd.DataFrame(FT_val[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-valIN.csv")
    pd.DataFrame(FT_test[0]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-testIN.csv")

    dOUT_dfs[0].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-allOUT.csv")
    dOUT_dfs[1].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-allOUT.csv")
    pd.DataFrame(UT_train[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-trainOUT.csv")
    pd.DataFrame(UT_val[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-valOUT.csv")
    pd.DataFrame(UT_test[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-testOUT.csv")
    pd.DataFrame(FT_train[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-trainOUT.csv")
    pd.DataFrame(FT_val[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-valOUT.csv")
    pd.DataFrame(FT_test[1]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-testOUT.csv")

    common_props_df.to_csv(PATH + f"MLdata/multi/MULTI-{dis}-allProps.csv")
    pd.DataFrame(UT_train[2]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-trainProps.csv")
    pd.DataFrame(UT_val[2]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-valProps.csv")
    pd.DataFrame(UT_test[2]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-testProps.csv")

    if UT_INf is not None and FT_INf is not None:
        INf_dfs[0].to_csv(PATH + f"MULTI-Ductile-{dis}-allINf.csv")
        INf_dfs[0].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-allINf.csv")
        INf_dfs[1].to_csv(PATH + f"MULTI-Fracture-{dis}-allINf.csv")
        INf_dfs[1].to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-allINf.csv")
        pd.DataFrame(UT_train[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-trainINf.csv")
        pd.DataFrame(UT_val[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-valINf.csv")
        pd.DataFrame(UT_test[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-UT-testINf.csv")
        pd.DataFrame(FT_train[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-trainINf.csv")
        pd.DataFrame(FT_val[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-valINf.csv")
        pd.DataFrame(FT_test[3]).to_csv(PATH + f"MLdata/multi/MULTI-{dis}-FT-testINf.csv")
    

def plot_sampling(df, LAT, l, indx=None, num=5, by="lattice"):
    if by.lower() == "node":
        df = df.T
    if indx is None:
        indx = random.sample(df.index.tolist(), num)
    elif type(indx) is int:
        indx = [indx]
    for i in indx:
        plt.figure(figsize=(10, 6))
        plt.hist(df.loc[i].to_numpy()/(l*1000), bins=25, alpha=0.7, color='blue')
        plt.title(f'Distribution of Disorder for {LAT.upper()} {by.capitalize()} {int(i)}', 
                  fontsize=18, fontname="Times New Roman")
        plt.xlabel('Normalized Disorder', fontsize=15, fontname="Times New Roman")
        plt.ylabel('Frequency', fontsize=15, fontname="Times New Roman")
        # plt.grid(True)
        plt.show()

def locSims(props_df):
    return pd.DataFrame((props_df.iloc[1:].idxmax(), props_df.iloc[1:].idxmin()), index=['Max', 'Min'])

def get_stats(props_df):
    return pd.DataFrame((props_df.iloc[1:].mean(), props_df.iloc[1:].std()), index=['Mean', 'Std'])

def plot_frequency(raw_data, data, test, bins=50):
    raw_data = np.array(data)
    data = np.array(data)
    
    if test == "UT":
        x_label = 'Normalized Ductility'
    elif test == "FT":
        x_label = 'Normalized Fracture Toughness ($K_{JIC}$)'
    elif test == "FCL":
        x_label = 'Normalized Fracto-Cohesive Length ($L_{f}$)'
    
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.set_figheight(5)
    fig1.set_figwidth(15)
    
    ax1.set_title('Raw Data', fontsize=25, fontname="Times New Roman")
    ax1.axvline(x=raw_data[0]/raw_data[0], color='orangered', label="Perfect")
    ax1.hist(raw_data[1:]/raw_data[0], bins=bins, label='Disordered')
    ax1.set_ylabel('Frequency', fontsize=20, fontname="Times New Roman")
    ax1.set_xlabel(x_label, fontsize=20, fontname="Times New Roman")
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=10)
    ax1.legend(prop={'size':11})
    
    ax2.set_title('Without Outliers', fontsize=25, fontname="Times New Roman")
    ax2.axvline(x=data[0]/data[0], color='orangered', label="Perfect")
    ax2.hist(data[1:]/data[0], bins=bins, label='Disordered')
    ax2.set_ylabel('Frequency', fontsize=20, fontname="Times New Roman")
    ax2.set_xlabel(x_label, fontsize=20, fontname="Times New Roman")
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=10)
    ax2.legend(prop={'size':11})
    
    plt.show()

def plot_properties(x_data, y_data, test, include_freq=False, compare_ax=None, highlight=None):
    x_data_np = np.array(x_data)
    y_data_np = np.array(y_data)

    if compare_ax is not None:
        include_freq = False
    
    if test == "UT":
        title = "Uniaxial Tension"
        x_label = 'Normalized Ductility'
        y_label = 'Normalized Strength'
    elif test == "FT":
        title = "Compact Tension"
        x_label = 'Normalized $K_{JIC}$'
        y_label = 'Normalized Displacement'
    elif test == "MULTI":
        title = "Multiple Properties"
        x_label = 'Normalized Ductility'
        y_label = 'Normalized $K_{JIC}$'
    elif test == "FCL":
        title = "Fracto-Cohesive Length"
        x_label = 'Normalized $L_{f}$'
        y_label = 'Normalized WoF'
    
    x_norm = x_data_np / x_data_np[0]
    y_norm = y_data_np / y_data_np[0]

    if include_freq:
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(4, 4, figure=fig)
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
        ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
        
        ax_scatter.scatter(x_norm[1:], y_norm[1:], c="orangered", alpha=0.7, marker="x", label="Disordered")
        ax_scatter.scatter(x_norm[0], y_norm[0], c="k", marker="*", label="Perfect")
        ax_scatter.axvline(1, linestyle='--', color="k")
        ax_scatter.axhline(1, linestyle='--', color="k")
        ax_scatter.set_xlabel(x_label, fontsize=20, fontname="Times New Roman")
        ax_scatter.set_ylabel(y_label, fontsize=20, fontname="Times New Roman")
        ax_scatter.legend()
        # ax_scatter.set_title(title, fontsize=25, fontname="Times New Roman")

        ax_histx.hist(x_norm[1:], bins=30, color='blue', alpha=0.3)
        ax_histx.axvline(x=1, linestyle='--', color="k")
        ax_histy.hist(y_norm[1:], bins=30, color='green', alpha=0.3, orientation='horizontal')
        ax_histy.axhline(y=1, linestyle='--', color="k")

        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

        ax_histx.set_ylabel('Frequency', fontsize=20, fontname="Times New Roman")
        ax_histy.set_xlabel('Frequency', fontsize=20, fontname="Times New Roman")
        
        fig.tight_layout()

    else:
        if compare_ax is not None:
            fig, ax = compare_ax
            d_label = "Disordered - Mech."
            col = "blue"
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            d_label = "Disordered"
            col = "orangered"
        ax.scatter(x_norm[1:], y_norm[1:], label=d_label, c=col, alpha=0.7, marker="x")
        if highlight is not None:
            for idx in highlight:
                ax.plot(x_data.loc[idx]/x_data[0], y_data.loc[idx]/y_data[0], c="lime", marker="o", markersize=18, markerfacecolor='none', markeredgewidth=3)
        if compare_ax is None:
            ax.scatter(x_norm[0], y_norm[0], label='Perfect', c="k", marker="*")
        ax.scatter(x_norm[0], y_norm[0], c="k", marker="*")
        ax.axvline(x=1, linestyle='--', color="k")
        ax.axhline(y=1, linestyle='--', color="k")
        # ax.set_title(title, fontsize=25, fontname="Times New Roman")
        ax.set_xlabel(x_label, fontsize=32, fontname="Times New Roman")
        ax.set_ylabel(y_label, fontsize=32, fontname="Times New Roman")
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.tick_params(axis='both', which='minor', labelsize=25)
        # ax.legend(prop={'size':20})
        fig.tight_layout()

    return fig, ax

def plot_curve(OUT_df, xOUT, mode, pi=0, idx=None, q=15, compare_ax=None):
    if compare_ax is not None:
        fig2, ax1 = compare_ax
        d_label = "Disordered - Mech."
        col = "blue"
    else:
        fig2, (ax1) = plt.subplots(1, 1)
        d_label = "Disordered"
        col = "orangered"
    fig2.set_figheight(5)
    fig2.set_figwidth(9)
    
    p = OUT_df.loc[pi].tolist()[1:]
    indx = int(OUT_df.loc[pi].tolist()[0])

    if compare_ax is None:
        ax1.plot(xOUT/xOUT[indx], [i/max(p) for i in p], label="Periodic", c='k')
        # plt.legend()
    
    if idx:
        if type(idx) is not list:
            idx = [int(idx)]
        for i in idx:
            d = OUT_df.loc[i].values.tolist()[1:]
            indx2 = int(OUT_df.loc[i].tolist()[0])
            ax1.plot(xOUT/xOUT[indx], [j/max(p) for j in d], label=f'{d_label} {i}', c=col)
            ax1.axvline(x=xOUT[indx2]/xOUT[indx], ymax=0.2, c=col, linestyle='--')
            ax1.axhline(y=max(d)/max(p), xmax=0.2, c=col, linestyle='--')
        
    else:
        idxs = OUT_df.index.tolist()[1:]
        if q == 'all':
            idxs = idxs
        else:
            idxs = random.sample(idxs, q)
            print(idxs)
        
        for idxx in idxs:
            d = OUT_df.loc[idxx].tolist()[1:]
            ax1.plot(xOUT/xOUT[indx], [i/max(p) for i in d], label=f'Disordered{idxx}')
    
    if mode.lower() == "ut":
        ax1.set_ylabel('Normalized Stress', fontsize=32, fontname="Times New Roman")
        ax1.set_xlabel('Normalized Strain', fontsize=32, fontname="Times New Roman")
    if mode.lower() == "ft":
        ax1.set_ylabel('Normalized Force', fontsize=32, fontname="Times New Roman")
        ax1.set_xlabel('Normalized Displacement', fontsize=32, fontname="Times New Roman")
    
    ax1.axvline(x=1, ymax=0.2, c='k', linestyle='--')
    ax1.axhline(y=1, xmax=0.2, c='k', linestyle='--')
    ax1.set_ylim(bottom=-0.1, top=1.1)
    
    if compare_ax is None:
        ax1.plot(xOUT/xOUT[indx], [i/max(p) for i in p], c='k')

    # if idx or q != 'all' and q <= 10:
    #     ax1.legend(prop={'size':11})
        
    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.tick_params(axis='both', which='minor', labelsize=25)
    # ax1.grid()

    return fig2, ax1


def load_TrainTestData(CSV_all_in, CSV_all_out, CSV_all_props, CSV_train_in, CSV_train_out, CSV_trainProps, CSV_val_in, 
                       CSV_val_out, CSV_valProps, CSV_test_in, CSV_test_out, CSV_testProps):
    all_in     = pd.read_csv(CSV_all_in, index_col=0, header=0).to_numpy()
    all_out    = pd.read_csv(CSV_all_out, index_col=0, header=0).to_numpy()
    allProps   = pd.read_csv(CSV_all_props, index_col=0, header=0).to_numpy()
    train_in   = pd.read_csv(CSV_train_in, index_col=0, header=0).to_numpy()
    train_out  = pd.read_csv(CSV_train_out, index_col=0, header=0).to_numpy()
    trainProps = pd.read_csv(CSV_trainProps, index_col=0, header=0).to_numpy()
    val_in     = pd.read_csv(CSV_val_in, index_col=0, header=0).to_numpy()
    val_out    = pd.read_csv(CSV_val_out, index_col=0, header=0).to_numpy()
    valProps   = pd.read_csv(CSV_valProps, index_col=0, header=0).to_numpy()
    test_in    = pd.read_csv(CSV_test_in, index_col=0, header=0).to_numpy()
    test_out   = pd.read_csv(CSV_test_out, index_col=0, header=0).to_numpy()
    testProps  = pd.read_csv(CSV_testProps, index_col=0, header=0).to_numpy()

    all, train, val, test = [all_in, all_out, allProps], [train_in, train_out, trainProps], [val_in, val_out, valProps], [test_in, test_out, testProps]
    return all, train, val, test

def load_freqInputData(CSV_all_in_f, CSV_train_in_f, CSV_val_in_f, CSV_test_in_f):
    all_in_f   = pd.read_csv(CSV_all_in_f, index_col=0, header=0).to_numpy()
    train_in_f = pd.read_csv(CSV_train_in_f, index_col=0, header=0).to_numpy()
    val_in_f   = pd.read_csv(CSV_val_in_f, index_col=0, header=0).to_numpy()
    test_in_f  = pd.read_csv(CSV_test_in_f, index_col=0, header=0).to_numpy()
    return all_in_f, train_in_f, val_in_f, test_in_f


def standardize(x, minx, maxx, mode=0):
    if mode == 0:
        return (x - minx)/(maxx - minx)
    if mode == 1:
        return (x*(maxx - minx)) + minx

def normalize(x, mean, std, mode=0):
    if mode == 0:
        return (x - mean)/std
    if mode == 1:
        return (x*std) + mean

class Dataset_(Dataset):
    def __init__(self, x, y, edge_index=None):
        self.x = x
        self.y = y
        self.edge_index=edge_index
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]
    
class PCA_:
    def __init__(self):
        self.data = None
        self.n_components = None

        self.pca = None
        self.final_pca = None
        self.reduced_data = None
        
    
    def fit(self, data, verbose=False, plot=False):
        self.data = data
        self.pca = PCA()
        self.pca.fit(self.data)

        n_components_50 = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.5)[0][0] + 1
        n_components_80 = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.8)[0][0] + 1
        n_components_90 = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.9)[0][0] + 1
        n_components_95 = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.95)[0][0] + 1
        n_components_100 = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= 0.999999)[0][0] + 1

        if verbose:
            print(f"Number of components to capture 50% variance: {n_components_50}")
            print(f"Number of components to capture 80% variance: {n_components_80}")
            print(f"Number of components to capture 90% variance: {n_components_90}")
            print(f"Number of components to capture 95% variance: {n_components_95}")
            print(f"Number of components to capture 100% variance: {n_components_100}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(np.cumsum(self.pca.explained_variance_ratio_), 'b-')
            plt.xlabel('Number of Components', fontsize=15, fontname="Times New Roman")
            plt.ylabel('Cumulative Explained Variance', fontsize=15, fontname="Times New Roman")
            plt.title('Explained Variance by PCA Components', fontsize=18, fontname="Times New Roman")
            plt.hlines(y=1, xmin=-10, xmax=n_components_100, color='green', linestyle='--', alpha=0.5, label='100% Threshold')
            plt.vlines(x=n_components_100, ymin=-1, ymax=1, color='green', linestyle='--', alpha=0.5)
            plt.hlines(y=0.95, xmin=-10, xmax=n_components_95, color='teal', linestyle='--', alpha=0.5, label='95% Threshold')
            plt.vlines(x=n_components_95, ymin=-1, ymax=0.95, color='teal', linestyle='--', alpha=0.5)
            plt.hlines(y=0.90, xmin=-10, xmax=n_components_90, color='orange', linestyle='--', alpha=0.5, label='90% Threshold')
            plt.vlines(x=n_components_90, ymin=-1, ymax=0.90, color='orange', linestyle='--', alpha=0.5)
            plt.hlines(y=0.80, xmin=-10, xmax=n_components_80, color='orangered', linestyle='--', alpha=0.5, label='80% Threshold')
            plt.vlines(x=n_components_80, ymin=-1, ymax=0.80, color='orangered', linestyle='--', alpha=0.5)
            plt.hlines(y=0.50, xmin=-10, xmax=n_components_50, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
            plt.vlines(x=n_components_50, ymin=-1, ymax=0.50, color='red', linestyle='--', alpha=0.5)
            plt.ylim(min(np.cumsum(self.pca.explained_variance_ratio_))-0.02, 1.02)
            plt.xlim(-5, len(self.pca.explained_variance_ratio_)+10)
            plt.legend()
            plt.show()

    def reduce(self, data=None, scale=False, accuracy=0.999999, n_components=None, verbose=False):
        if data is None:
            data = self.data
        self.n_components = np.where(np.cumsum(self.pca.explained_variance_ratio_) >= accuracy)[0][0] + 1
        if n_components is not None:
            self.n_components = n_components
        self.final_pca = PCA(n_components=self.n_components)
        self.reduced_data = self.final_pca.fit_transform(data)

        if verbose:
            print(f"Original data shape: {self.data.shape}")
            print(f"Reduced data shape: {self.reduced_data.shape}")
        
        return self.reduced_data
    
    def reconstruct(self, data, scale=False):
        reconstructed_data = self.final_pca.inverse_transform(data)
        if scale:
            reconstructed_data = self.scaler.inverse_transform(reconstructed_data)
        return reconstructed_data

class DATA:
    def __init__(
        self, 
        path=1, 
        path_add='', 
        load=False, 
        LAT="FCC", 
        nnx=None,
        dis="disNodes", 
        dN=20, 
        mechMode="both",
        multi=False,
        nsims=None,
        model="MLP", 
        freq=False,
        scale=False,
        reduce_dim=False,
        format=0
    ):
        self.path = path
        self.path_add = path_add
        self.load = load
        self.LAT = LAT
        self.nnx = nnx
        self.dis = dis
        self.dN = dN
        self.mechMode = mechMode
        self.multi = multi
        self.nsims = nsims
        self.model = model
        self.freq = freq
        
        self.scale = scale
        if scale:
            if scale[0].lower() == "minmax" or scale[0].lower() == "maxmin":
                self.scaler = MinMaxScaler()
            elif scale[0].lower() == "standardscaler" or scale[0].lower() == "standard":
                self.scaler = StandardScaler()
            elif scale[0].lower() == "standardize":
                self.scaler = standardize
            elif scale[0].lower() == "normalize":
                self.scaler = normalize

        self.reduce_dim = reduce_dim
        if reduce_dim:
            if reduce_dim[0].lower() == "pca":
                self.reducer = PCA_()
            elif reduce_dim[0].lower() == "autoencoder":
                self.reducer = None

        if path_add.lower() == "frequency":
            self.freq = True

        if mechMode.lower() == "ut":
            self.UTmechTest = True
            self.FTmechTest = False
            self.multi = False
        elif mechMode.lower() == "ft":
            self.UTmechTest = False
            self.FTmechTest = True
            self.multi = False
        elif mechMode.lower() == "both":
            self.UTmechTest = True
            self.FTmechTest = True
            self.multi = multi
        
        if nnx is None:
            if LAT.lower() in ["fcc", "kagome", "hex"]:
                self.nnx = 20
            elif LAT.lower() == "tri":
                self.nnx = 30
        self.geom = Geometry(LAT=self.LAT, l=0.010, nnx=self.nnx)
        
        self.E_s = 123e9  ## Pa
        self.v_s = 0.3
        self.E_eff, self.v_eff, self.E_eff_pe, self.v_eff_pe = effProperties(self.LAT, self.geom, self.E_s, self.v_s, self.geom.rD, mode="stiff", C=None, ortho=not self.geom.iso)

        self.get_DataPath()

        if load:
            self.get_DataFiles()
            self.load_data()

            if format == 1 and model.lower() == "mlp" or model.lower() == "gpr":
                self.load_DisDist_v1()
            elif format == 2 and model.lower() == "mlp":
                self.load_DisDist_v2()

    def get_DataPath(self):
        pData = 'Z:/p1/data/'

        pAl          = pData + 'Al/'
        pAK          = pAl + 'AK/'
        pUTdisNodes  = pAK + 'Ductile-disNodes-FCC-12X16/'
        pUTdisNodes2 = pAK + '20_RD02_10mm/'
        pUTdisStruts = pAK + 'Ductile-disStruts-FCC-12X16/'
        pFTdisNodes  = pAK + 'Fracture-disNodes/'

        pTi    = pData + 'Ti/'
        pTiLAT = pTi + f'{self.dis}/{self.path_add}/{self.dN}/{self.LAT}/'

        if self.path == 0:
            self.PATH = pUTdisNodes2
        elif self.path == 1:
            self.PATH = pTiLAT
        else:
            self.PATH = str(self.path)+"/"

    def get_DataFiles(self):
        if self.UTmechTest and not self.multi:
            self.UT_INcsv             = self.PATH + f'Ductile-disNodes-IN.csv'
            self.UT_INcsv_noOutliers  = self.PATH + f'Ductile-disNodes-IN-noOutliers.csv'
            self.UT_OUTcsv            = self.PATH + f'Ductile-disNodes-OUT.csv'
            self.UT_OUTcsv_noOutliers = self.PATH + f'Ductile-disNodes-OUT-noOutliers.csv'

            self.UT_CSV_all_in    = self.PATH + f'MLdata/UT/UT-{self.dis}-allIN.csv'
            self.UT_CSV_all_out   = self.PATH + f'MLdata/UT/UT-{self.dis}-allOUT.csv'
            self.UT_CSV_train_in  = self.PATH + f'MLdata/UT/UT-{self.dis}-trainIN.csv'
            self.UT_CSV_train_out = self.PATH + f'MLdata/UT/UT-{self.dis}-trainOUT.csv'
            self.UT_CSV_val_in    = self.PATH + f'MLdata/UT/UT-{self.dis}-valIN.csv'
            self.UT_CSV_val_out   = self.PATH + f'MLdata/UT/UT-{self.dis}-valOUT.csv'
            self.UT_CSV_test_in   = self.PATH + f'MLdata/UT/UT-{self.dis}-testIN.csv'
            self.UT_CSV_test_out  = self.PATH + f'MLdata/UT/UT-{self.dis}-testOUT.csv'

            self.UT_CSV_allProps   = self.PATH + f'MLdata/UT/UT-{self.dis}-allProps.csv'
            self.UT_CSV_trainProps = self.PATH + f'MLdata/UT/UT-{self.dis}-trainProps.csv'
            self.UT_CSV_valProps   = self.PATH + f'MLdata/UT/UT-{self.dis}-valProps.csv'
            self.UT_CSV_testProps  = self.PATH + f'MLdata/UT/UT-{self.dis}-testProps.csv'
            
            self.UT_INcsv_f            = None
            self.UT_INcsv_f_noOutliers = None

            self.UT_CSV_all_in_f       = None
            self.UT_CSV_train_in_f     = None
            self.UT_CSV_val_in_f       = None
            self.UT_CSV_test_in_f      = None

            if self.freq:
                self.UT_INcsv_f            = self.PATH + f'Ductile-disNodes-INf.csv'
                self.UT_INcsv_f_noOutliers = self.PATH + f'Ductile-disNodes-INf-noOutliers.csv'

                self.UT_CSV_all_in_f       = self.PATH + f'MLdata/UT/UT-{self.dis}-allINf.csv'
                self.UT_CSV_train_in_f     = self.PATH + f'MLdata/UT/UT-{self.dis}-trainINf.csv'
                self.UT_CSV_val_in_f       = self.PATH + f'MLdata/UT/UT-{self.dis}-valINf.csv'
                self.UT_CSV_test_in_f      = self.PATH + f'MLdata/UT/UT-{self.dis}-testINf.csv'

        if self.FTmechTest and not self.multi:
            self.FT_INcsv             = self.PATH + f'Fracture-disNodes-IN.csv'
            self.FT_INcsv_noOutliers  = self.PATH + f'Fracture-disNodes-IN-noOutliers.csv'
            self.FT_OUTcsv            = self.PATH + f'Fracture-disNodes-OUT.csv'
            self.FT_OUTcsv_noOutliers = self.PATH + f'Fracture-disNodes-OUT-noOutliers.csv'

            self.FT_CSV_all_in    = self.PATH + f'MLdata/FT/FT-{self.dis}-allIN.csv'
            self.FT_CSV_all_out   = self.PATH + f'MLdata/FT/FT-{self.dis}-allOUT.csv'
            self.FT_CSV_train_in  = self.PATH + f'MLdata/FT/FT-{self.dis}-trainIN.csv'
            self.FT_CSV_train_out = self.PATH + f'MLdata/FT/FT-{self.dis}-trainOUT.csv'
            self.FT_CSV_val_in    = self.PATH + f'MLdata/FT/FT-{self.dis}-valIN.csv'
            self.FT_CSV_val_out   = self.PATH + f'MLdata/FT/FT-{self.dis}-valOUT.csv'
            self.FT_CSV_test_in   = self.PATH + f'MLdata/FT/FT-{self.dis}-testIN.csv'
            self.FT_CSV_test_out  = self.PATH + f'MLdata/FT/FT-{self.dis}-testOUT.csv'

            self.FT_CSV_allProps   = self.PATH + f'MLdata/FT/FT-{self.dis}-allProps.csv'
            self.FT_CSV_trainProps = self.PATH + f'MLdata/FT/FT-{self.dis}-trainProps.csv'
            self.FT_CSV_valProps   = self.PATH + f'MLdata/FT/FT-{self.dis}-valProps.csv'
            self.FT_CSV_testProps  = self.PATH + f'MLdata/FT/FT-{self.dis}-testProps.csv'
            
            self.FT_INcsv_f            = None
            self.FT_INcsv_f_noOutliers = None

            self.FT_CSV_all_in_f       = None
            self.FT_CSV_train_in_f     = None
            self.FT_CSV_val_in_f       = None
            self.FT_CSV_test_in_f      = None

            if self.freq:
                self.FT_INcsv_f            = self.PATH + f'Fracture-disNodes-INf.csv'
                self.FT_INcsv_f_noOutliers = self.PATH + f'Fracture-disNodes-INf-noOutliers.csv'

                self.FT_CSV_all_in_f       = self.PATH + f'MLdata/FT/FT-{self.dis}-allINf.csv'
                self.FT_CSV_train_in_f     = self.PATH + f'MLdata/FT/FT-{self.dis}-trainINf.csv'
                self.FT_CSV_val_in_f       = self.PATH + f'MLdata/FT/FT-{self.dis}-valINf.csv'
                self.FT_CSV_test_in_f      = self.PATH + f'MLdata/FT/FT-{self.dis}-testINf.csv'

        if self.UTmechTest and self.FTmechTest and self.multi:
            self.UT_CSV_allProps   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-allProps.csv'
            self.UT_CSV_trainProps = self.PATH + f'MLdata/multi/MULTI-{self.dis}-trainProps.csv'
            self.UT_CSV_valProps   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-valProps.csv'
            self.UT_CSV_testProps  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-testProps.csv'
            self.FT_CSV_allProps   = self.UT_CSV_allProps
            self.FT_CSV_trainProps = self.UT_CSV_trainProps
            self.FT_CSV_valProps   = self.UT_CSV_valProps
            self.FT_CSV_testProps  = self.UT_CSV_testProps

            self.UT_INcsv             = self.PATH + f'Ductile-disNodes-IN.csv'
            self.UT_INcsv_noOutliers  = self.PATH + f'MULTI-Ductile-disNodes-IN-noOutliers.csv'
            self.UT_OUTcsv            = self.PATH + f'Ductile-disNodes-OUT.csv'
            self.UT_OUTcsv_noOutliers = self.PATH + f'MULTI-Ductile-disNodes-OUT-noOutliers.csv'

            self.UT_CSV_all_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allIN.csv'
            self.UT_CSV_all_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allOUT.csv'
            self.UT_CSV_train_in  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainIN.csv'
            self.UT_CSV_train_out = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainOUT.csv'
            self.UT_CSV_val_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valIN.csv'
            self.UT_CSV_val_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valOUT.csv'
            self.UT_CSV_test_in   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testIN.csv'
            self.UT_CSV_test_out  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testOUT.csv'
            
            self.UT_INcsv_f            = None
            self.UT_INcsv_f_noOutliers = None

            self.UT_CSV_all_in_f       = None
            self.UT_CSV_train_in_f     = None
            self.UT_CSV_val_in_f       = None
            self.UT_CSV_test_in_f      = None

            if self.freq:
                self.UT_INcsv_f            = self.PATH + f'Ductile-disNodes-INf.csv'
                self.UT_INcsv_f_noOutliers = self.PATH + f'MULTI-Ductile-disNodes-INf-noOutliers.csv'

                self.UT_CSV_all_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allINf.csv'
                self.UT_CSV_train_in_f     = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainINf.csv'
                self.UT_CSV_val_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valINf.csv'
                self.UT_CSV_test_in_f      = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testINf.csv'

            self.FT_INcsv             = self.PATH + f'Fracture-disNodes-IN.csv'
            self.FT_INcsv_noOutliers  = self.PATH + f'MULTI-Fracture-disNodes-IN-noOutliers.csv'
            self.FT_OUTcsv            = self.PATH + f'Fracture-disNodes-OUT.csv'
            self.FT_OUTcsv_noOutliers = self.PATH + f'MULTI-Fracture-disNodes-OUT-noOutliers.csv'

            self.FT_CSV_all_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allIN.csv'
            self.FT_CSV_all_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allOUT.csv'
            self.FT_CSV_train_in  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainIN.csv'
            self.FT_CSV_train_out = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainOUT.csv'
            self.FT_CSV_val_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valIN.csv'
            self.FT_CSV_val_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valOUT.csv'
            self.FT_CSV_test_in   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testIN.csv'
            self.FT_CSV_test_out  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testOUT.csv'
            
            self.FT_INcsv_f            = None
            self.FT_INcsv_f_noOutliers = None

            self.FT_CSV_all_in_f       = None
            self.FT_CSV_train_in_f     = None
            self.FT_CSV_val_in_f       = None
            self.FT_CSV_test_in_f      = None

            if self.freq:
                self.FT_INcsv_f            = self.PATH + f'Fracture-disNodes-INf.csv'
                self.FT_INcsv_f_noOutliers = self.PATH + f'MULTI-Fracture-disNodes-INf-noOutliers.csv'

                self.FT_CSV_all_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allINf.csv'
                self.FT_CSV_train_in_f     = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainINf.csv'
                self.FT_CSV_val_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valINf.csv'
                self.FT_CSV_test_in_f      = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testINf.csv'

    def load_data(self):
        if self.UTmechTest:
            self.UT_IN_df, \
            self.UT_OUT_df, \
            self.UT_INf_df, \
            self.UT_perINr_df, \
            self.UT_perIN_df, \
            self.UT_perOUT_df, \
            self.UT_dIN_df, \
            self.UT_dOUT_df = load_data(self.UT_INcsv, 
                                        self.UT_OUTcsv, 
                                        self.UT_INcsv_f if self.freq else None,
                                        no_outliers=[self.UT_INcsv_noOutliers, 
                                                    self.UT_OUTcsv_noOutliers, 
                                                    self.UT_INcsv_f_noOutliers if self.freq else None])
            self.UT_xOUT = np.linspace(0, max(self.UT_perOUT_df.x.tolist()), len(self.UT_dOUT_df.to_numpy()[0]))
            UT_all, UT_train, UT_val, UT_test = load_TrainTestData(self.UT_CSV_all_in,
                                                                    self.UT_CSV_all_out,
                                                                    self.UT_CSV_allProps,
                                                                    self.UT_CSV_train_in, 
                                                                    self.UT_CSV_train_out,
                                                                    self.UT_CSV_trainProps, 
                                                                    self.UT_CSV_val_in, 
                                                                    self.UT_CSV_val_out, 
                                                                    self.UT_CSV_valProps,
                                                                    self.UT_CSV_test_in, 
                                                                    self.UT_CSV_test_out,
                                                                    self.UT_CSV_testProps)
            UT_all_in, UT_all_out, UT_allProps       = UT_all
            UT_train_in, UT_train_out, UT_trainProps = UT_train
            UT_val_in, UT_val_out, UT_valProps       = UT_val
            UT_test_in, UT_test_out, UT_testProps    = UT_test

            if self.freq:
                UT_all_in, UT_train_in, UT_val_in, UT_test_in = load_freqInputData(self.UT_CSV_all_in_f,
                                                                                    self.UT_CSV_train_in_f, 
                                                                                    self.UT_CSV_val_in_f, 
                                                                                    self.UT_CSV_test_in_f)

            if self.model.lower() == "mlp" or self.model.lower() == "gpr":
                self.UT_all_in   = UT_all_in
                self.UT_train_in = UT_train_in
                self.UT_val_in   = UT_val_in
                self.UT_test_in  = UT_test_in
            elif self.model.lower() == "gnn":
                self.UT_all_in   = UT_all_in.reshape(*UT_all_in.shape[:-1], UT_all_in.shape[-1]//2, 2)
                self.UT_train_in = UT_train_in.reshape(*UT_train_in.shape[:-1], UT_train_in.shape[-1]//2, 2)
                self.UT_val_in   = UT_val_in.reshape(*UT_val_in.shape[:-1], UT_val_in.shape[-1]//2, 2)
                self.UT_test_in  = UT_test_in.reshape(*UT_test_in.shape[:-1], UT_test_in.shape[-1]//2, 2)
            
            cols = ['Ductility', 'Strength', 'Stiffness', 'WoF']
            if self.multi:
                cols = ['Ductility', 'Strength', 'Stiffness', 'WoF', 'K_JIC', 'K_IC', 'Force', 'Displacement', 'Multi', 'FCL']
            self.UT_all_out, self.UT_allProps, self.UT_allProps_df       = UT_all_out, UT_allProps, pd.DataFrame(UT_allProps, columns=cols, index=self.UT_OUT_df.index)
            self.UT_train_out, self.UT_trainProps, self.UT_trainProps_df = UT_train_out, UT_trainProps, pd.DataFrame(UT_trainProps.T, columns=cols)
            self.UT_val_out, self.UT_valProps, self.UT_valProps_df       = UT_val_out, UT_valProps, pd.DataFrame(UT_valProps.T, columns=cols)
            self.UT_test_out, self.UT_testProps, self.UT_testProps_df    = UT_test_out, UT_testProps, pd.DataFrame(UT_testProps.T, columns=cols)

            if self.nsims is not None:
                self.UT_all_in, self.UT_train_in, self.UT_val_in, self.UT_test_in = self.UT_all_in[:self.nsims], self.UT_train_in[:self.nsims], self.UT_val_in[:self.nsims], self.UT_test_in[:self.nsims]
                self.UT_all_out, self.UT_train_out, self.UT_val_out, self.UT_test_out = self.UT_all_out[:self.nsims], self.UT_train_out[:self.nsims], self.UT_val_out[:self.nsims], self.UT_test_out[:self.nsims]
                self.UT_allProps, self.UT_trainProps, self.UT_valProps, self.UT_testProps = self.UT_allProps[:self.nsims], self.UT_trainProps[:self.nsims], self.UT_valProps[:self.nsims], self.UT_testProps[:self.nsims]
                self.UT_allProps_df, self.UT_trainProps_df, self.UT_valProps_df, self.UT_testProps_df = self.UT_allProps_df[:self.nsims], self.UT_trainProps_df[:self.nsims], self.UT_valProps_df[:self.nsims], self.UT_testProps_df[:self.nsims]
                self.UT_OUT_df = self.UT_OUT_df[:self.nsims]
                self.UT_IN_df = self.UT_IN_df[:self.nsims]
                self.UT_dIN_df = self.UT_dIN_df[:self.nsims]
                self.UT_dOUT_df = self.UT_dOUT_df[:self.nsims]
                if self.freq:
                    self.UT_INf_df = self.UT_INf_df[:self.nsims]

            if self.scale:
                if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.UT_INscaler = clone(self.scaler)
                    self.UT_train_in = self.UT_INscaler.fit_transform(self.UT_train_in)
                    self.UT_all_in   = self.UT_INscaler.transform(self.UT_all_in)
                    self.UT_val_in   = self.UT_INscaler.transform(self.UT_val_in)
                    self.UT_test_in  = self.UT_INscaler.transform(self.UT_test_in)
                if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.UT_OUTscaler = clone(self.scaler)
                    self.UT_train_out = self.UT_OUTscaler.fit_transform(self.UT_train_out)
                    self.UT_all_out   = self.UT_OUTscaler.transform(self.UT_all_out)
                    self.UT_val_out   = self.UT_OUTscaler.transform(self.UT_val_out)
                    self.UT_test_out  = self.UT_OUTscaler.transform(self.UT_test_out)
                if "props" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.UT_PROPscaler = clone(self.scaler)
                    self.UT_trainProps = self.UT_PROPscaler.fit_transform(self.UT_trainProps.T).T
                    self.UT_allProps   = self.UT_PROPscaler.transform(self.UT_allProps.T).T
                    self.UT_valProps   = self.UT_PROPscaler.transform(self.UT_valProps.T).T
                    self.UT_testProps  = self.UT_PROPscaler.transform(self.UT_testProps.T).T
            
            if self.reduce_dim:
                if "in" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                    self.UT_INreducer = copy.deepcopy(self.reducer)
                    self.UT_INreducer.fit(self.UT_train_in)
                    self.UT_all_in   = self.UT_INreducer.reduce(self.UT_all_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_train_in = self.UT_INreducer.reduce(self.UT_train_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_val_in   = self.UT_INreducer.reduce(self.UT_val_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_test_in  = self.UT_INreducer.reduce(self.UT_test_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                if "out" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                    self.UT_OUTreducer = copy.deepcopy(self.reducer)
                    self.UT_OUTreducer.fit(self.UT_train_out)
                    self.UT_all_out   = self.UT_OUTreducer.reduce(self.UT_all_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_train_out = self.UT_OUTreducer.reduce(self.UT_train_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_val_out   = self.UT_OUTreducer.reduce(self.UT_val_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.UT_test_out  = self.UT_OUTreducer.reduce(self.UT_test_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])

        if self.FTmechTest:
            self.FT_IN_df, \
            self.FT_OUT_df, \
            self.FT_INf_df, \
            self.FT_perINr_df, \
            self.FT_perIN_df, \
            self.FT_perOUT_df, \
            self.FT_dIN_df, \
            self.FT_dOUT_df = load_data(self.FT_INcsv, 
                                        self.FT_OUTcsv, 
                                        self.FT_INcsv_f if self.freq else None,
                                        no_outliers=[self.FT_INcsv_noOutliers, 
                                                    self.FT_OUTcsv_noOutliers, 
                                                    self.FT_INcsv_f_noOutliers if self.freq else None])
            self.FT_xOUT = np.linspace(0, max(self.FT_perOUT_df.x.tolist()), len(self.FT_dOUT_df.to_numpy()[0]))
            FT_all, FT_train, FT_val, FT_test = load_TrainTestData(self.FT_CSV_all_in,
                                                    self.FT_CSV_all_out,
                                                    self.FT_CSV_allProps,
                                                    self.FT_CSV_train_in, 
                                                    self.FT_CSV_train_out,
                                                    self.FT_CSV_trainProps, 
                                                    self.FT_CSV_val_in, 
                                                    self.FT_CSV_val_out, 
                                                    self.FT_CSV_valProps,
                                                    self.FT_CSV_test_in, 
                                                    self.FT_CSV_test_out,
                                                    self.FT_CSV_testProps)
            FT_all_in, FT_all_out, FT_allProps       = FT_all
            FT_train_in, FT_train_out, FT_trainProps = FT_train
            FT_val_in, FT_val_out, FT_valProps       = FT_val
            FT_test_in, FT_test_out, FT_testProps    = FT_test

            if self.freq:
                FT_all_in, FT_train_in, FT_val_in, FT_test_in = load_freqInputData(self.FT_CSV_all_in_f,
                                                                    self.FT_CSV_train_in_f, 
                                                                    self.FT_CSV_val_in_f, 
                                                                    self.FT_CSV_test_in_f)

            if self.model.lower() == "mlp" or self.model.lower() == "gpr":
                self.FT_all_in   = FT_all_in
                self.FT_train_in = FT_train_in
                self.FT_val_in   = FT_val_in
                self.FT_test_in  = FT_test_in
            elif self.model.lower() == "gnn":
                self.FT_all_in   = FT_all_in.reshape(*FT_all_in.shape[:-1], FT_all_in.shape[-1]//2, 2)
                self.FT_train_in = FT_train_in.reshape(*FT_train_in.shape[:-1], FT_train_in.shape[-1]//2, 2)
                self.FT_val_in   = FT_val_in.reshape(*FT_val_in.shape[:-1], FT_val_in.shape[-1]//2, 2)
                self.FT_test_in  = FT_test_in.reshape(*FT_test_in.shape[:-1], FT_test_in.shape[-1]//2, 2)
            
            cols = ['K_JIC', 'K_IC', 'Force', 'Displacement']
            if self.multi:
                cols = ['Ductility', 'Strength', 'Stiffness', 'WoF', 'K_JIC', 'K_IC', 'Force', 'Displacement', 'Multi', 'FCL']
            self.FT_all_out, self.FT_allProps, self.FT_allProps_df       = FT_all_out, FT_allProps, pd.DataFrame(FT_allProps, columns=cols, index=self.FT_OUT_df.index)
            self.FT_train_out, self.FT_trainProps, self.FT_trainProps_df = FT_train_out, FT_trainProps, pd.DataFrame(FT_trainProps.T, columns=cols)
            self.FT_val_out, self.FT_valProps, self.FT_valProps_df       = FT_val_out, FT_valProps, pd.DataFrame(FT_valProps.T, columns=cols)
            self.FT_test_out, self.FT_testProps, self.FT_testProps_df    = FT_test_out, FT_testProps, pd.DataFrame(FT_testProps.T, columns=cols)

            if self.nsims is not None:
                self.FT_all_in, self.FT_train_in, self.FT_val_in, self.FT_test_in = self.FT_all_in[:self.nsims], self.FT_train_in[:self.nsims], self.FT_val_in[:self.nsims], self.FT_test_in[:self.nsims]
                self.FT_all_out, self.FT_train_out, self.FT_val_out, self.FT_test_out = self.FT_all_out[:self.nsims], self.FT_train_out[:self.nsims], self.FT_val_out[:self.nsims], self.FT_test_out[:self.nsims]
                self.FT_allProps, self.FT_trainProps, self.FT_valProps, self.FT_testProps = self.FT_allProps[:self.nsims], self.FT_trainProps[:self.nsims], self.FT_valProps[:self.nsims], self.FT_testProps[:self.nsims]
                self.FT_allProps_df, self.FT_trainProps_df, self.FT_valProps_df, self.FT_testProps_df = self.FT_allProps_df[:self.nsims], self.FT_trainProps_df[:self.nsims], self.FT_valProps_df[:self.nsims], self.FT_testProps_df[:self.nsims]
                self.FT_OUT_df = self.FT_OUT_df[:self.nsims]
                self.FT_IN_df = self.FT_IN_df[:self.nsims]
                self.FT_dIN_df = self.FT_dIN_df[:self.nsims]
                self.FT_dOUT_df = self.FT_dOUT_df[:self.nsims]
                if self.freq:
                    self.FT_INf_df = self.FT_INf_df[:self.nsims]

            if self.scale:
                if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.FT_INscaler = clone(self.scaler)
                    self.FT_train_in = self.FT_INscaler.fit_transform(self.FT_train_in)
                    self.FT_all_in   = self.FT_INscaler.transform(self.FT_all_in)
                    self.FT_val_in   = self.FT_INscaler.transform(self.FT_val_in)
                    self.FT_test_in  = self.FT_INscaler.transform(self.FT_test_in)
                if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.FT_OUTscaler = clone(self.scaler)
                    self.FT_train_out = self.FT_OUTscaler.fit_transform(self.FT_train_out)
                    self.FT_all_out   = self.FT_OUTscaler.transform(self.FT_all_out)
                    self.FT_val_out   = self.FT_OUTscaler.transform(self.FT_val_out)
                    self.FT_test_out  = self.FT_OUTscaler.transform(self.FT_test_out)
                if "props" in self.scale[1].lower() or "all" in self.scale[1].lower():
                    self.FT_PROPscaler = clone(self.scaler)
                    self.FT_trainProps = self.FT_PROPscaler.fit_transform(self.FT_trainProps.T).T
                    self.FT_allProps   = self.FT_PROPscaler.transform(self.FT_allProps.T).T
                    self.FT_valProps   = self.FT_PROPscaler.transform(self.FT_valProps.T).T
                    self.FT_testProps  = self.FT_PROPscaler.transform(self.FT_testProps.T).T
            
            if self.reduce_dim:
                if "in" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                    self.FT_INreducer = copy.deepcopy(self.reducer)
                    self.FT_INreducer.fit(self.FT_train_in)
                    self.FT_all_in   = self.FT_INreducer.reduce(self.FT_all_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_train_in = self.FT_INreducer.reduce(self.FT_train_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_val_in   = self.FT_INreducer.reduce(self.FT_val_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_test_in  = self.FT_INreducer.reduce(self.FT_test_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                if "out" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                    self.FT_OUTreducer = copy.deepcopy(self.reducer)
                    self.FT_OUTreducer.fit(self.FT_train_out)
                    self.FT_all_out   = self.FT_OUTreducer.reduce(self.FT_all_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_train_out = self.FT_OUTreducer.reduce(self.FT_train_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_val_out   = self.FT_OUTreducer.reduce(self.FT_val_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    self.FT_test_out  = self.FT_OUTreducer.reduce(self.FT_test_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
    
        if self.UTmechTest and self.FTmechTest and self.multi:
            self.common_allProps, self.common_allProps_df = self.UT_allProps, self.UT_allProps_df
            self.common_trainProps, self.common_trainProps_df = self.UT_trainProps, self.UT_trainProps_df
            self.common_valProps, self.common_valProps_df = self.UT_valProps, self.UT_valProps_df
            self.common_testProps, self.common_testProps_df = self.UT_testProps, self.UT_testProps_df

            if self.nsims is not None:
                self.common_allProps, self.common_allProps_df = self.common_allProps[:self.nsims], self.common_allProps_df[:self.nsims]
                self.common_trainProps, self.common_trainProps_df = self.common_trainProps[:self.nsims], self.common_trainProps_df[:self.nsims]
                self.common_valProps, self.common_valProps_df = self.common_valProps[:self.nsims], self.common_valProps_df[:self.nsims]
                self.common_testProps, self.common_testProps_df = self.common_testProps[:self.nsims], self.common_testProps_df[:self.nsims]

    def load_DisDist_v1(self):
        self.train_in1 = self.perIN_df.to_numpy().reshape(len(self.perIN_df)//2, 2)

        train_out1 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
        self.dx_out1 = train_out1[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2,1)
        self.dy_out1 = train_out1[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2,1)
    
    def load_DisDist_v2(self):
        self.train_in1 = self.perIN_df.to_numpy().reshape(len(self.perIN_df)//2, 2)
        self.train_in2 = np.array([self.train_in1.flatten()]*2)

        train_out2 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
        dx_out2 = train_out2[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2)
        self.dx_out2 = np.stack((dx_out2, dx_out2), axis=1)
        dy_out2 = train_out2[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2)
        self.dy_out2 = np.stack((dy_out2, dy_out2), axis=1)



#### Normalisation and Standardization Data Class Variables
        # self.inParams = dataParams(self.all_in)
        # self.outParams = dataParams(self.all_out)

        # self.train_inST = standardize(self.train_in, self.inParams[0], self.inParams[1])
        # self.train_outST = standardize(self.train_out, self.outParams[0], self.outParams[1])
        # self.val_inST = standardize(self.val_in, self.inParams[0], self.inParams[1])
        # self.val_outST = standardize(self.val_out, self.outParams[0], self.outParams[1])
        # self.test_inST = standardize(self.test_in, self.inParams[0], self.inParams[1])
        # self.test_outST = standardize(self.test_out, self.outParams[0], self.outParams[1])
        # self.all_inST = standardize(self.all_in, self.inParams[0], self.inParams[1])
        # self.all_outST = standardize(self.all_out, self.outParams[0], self.outParams[1])
        
        # self.train_inNM = normalize(self.train_in, self.inParams[2], self.inParams[3])
        # self.train_outNM = normalize(self.train_out, self.outParams[2], self.outParams[3])
        # self.val_inNM = normalize(self.val_in, self.inParams[2], self.inParams[3])
        # self.val_outNM = normalize(self.val_out, self.outParams[2], self.outParams[3])
        # self.test_inNM = normalize(self.test_in, self.inParams[2], self.inParams[3])
        # self.test_outNM = normalize(self.test_out, self.outParams[2], self.outParams[3])
        # self.all_inNM = normalize(self.all_in, self.inParams[2], self.inParams[3])
        # self.all_outNM = normalize(self.all_out, self.outParams[2], self.outParams[3])


        # self.inParams1 = dataParams(self.train_in1)
        # self.train_in1ST = standardize(self.train_in1, self.inParams1[0], self.inParams1[1])
        # self.train_in1NM = normalize(self.train_in1, self.inParams1[2], self.inParams1[3])

        # self.outParams1dx = dataParams(self.dx_out1)
        # self.outParams1dy = dataParams(self.dy_out1)
        # self.dx_out1ST = standardize(self.dx_out1, self.outParams1dx[0], self.outParams1dx[1])
        # self.dy_out1ST = standardize(self.dy_out1, self.outParams1dy[0], self.outParams1dy[1])
        # self.dx_out1NM = normalize(self.dx_out1, self.outParams1dx[2], self.outParams1dx[3])
        # self.dy_out1NM = normalize(self.dy_out1, self.outParams1dy[2], self.outParams1dy[3])
        

        # self.inParams2 = dataParams(self.train_in2)
        # self.train_in2ST = standardize(self.train_in2, self.inParams2[0], self.inParams2[1])
        # self.train_in2NM = normalize(self.train_in2, self.inParams2[2], self.inParams2[3])

        # self.outParams2dx = dataParams(self.dx_out2)
        # self.outParams2dy = dataParams(self.dx_out2)
        # self.dx_out2ST = standardize(self.dx_out2, self.outParams2dx[0], self.outParams2dx[1])
        # self.dx_out2ST = standardize(self.dx_out2, self.outParams2dy[0], self.outParams2dy[1])
        # self.dx_out2NM = normalize(self.dx_out2, self.outParams2dx[2], self.outParams2dx[3])
        # self.dx_out2NM = normalize(self.dx_out2, self.outParams2dy[2], self.outParams2dy[3])