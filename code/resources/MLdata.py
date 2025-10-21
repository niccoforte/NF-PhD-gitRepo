from resources.imports import *
from resources.calculations import calcUT, calcFT
from resources.lattices import geometry

from matplotlib.gridspec import GridSpec

from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


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
    
    ducts, strens, stiffs = [], [], []
    for _, row in OUT_df.iterrows():
        UT_df = pd.DataFrame({'x':np.insert(xOUT,0,row[0]), 'y_sm':row})
        ductility, strength, stiffness = calcUT(UT_df)
        
        ducts.append(ductility)
        strens.append(strength)
        stiffs.append(stiffness)
        
    props = np.array([ducts, strens, stiffs])
    return dIN, dOUT, INf, xOUT, props

def prep_FTdata(dIN_df, dOUT_df, perOUT_df, OUT_df, geom, E_eff, INf_df=None):
    dIN = dIN_df.to_numpy()
    dOUT = dOUT_df.to_numpy()
    xOUT = np.linspace(0, max(perOUT_df.x.tolist()), len(dOUT[0]))
    INf = None
    if INf_df is not None:
        INf = INf_df.to_numpy()
    
    Kjs, Ks, Ps, ds = [], [], [], []
    for indx, row in OUT_df.iterrows():
        FT_df = pd.DataFrame({'x':np.insert(xOUT,0,row[0]), 'y_sm':row})
        P, dd, K, Kj = calcFT(FT_df, geom, E_eff, n_Ks=1)
        
        Kjs.append(Kj[0])
        Ks.append(K[0])
        Ps.append(P)
        ds.append(dd)
    
    props = [Kjs, Ks, Ps, ds]
    return dIN, dOUT, INf, xOUT, props

def find_outliers(data):
    mean = np.mean(data)
    stdev = np.std(data)
    if type(data) is not list:
        data = data.tolist()    
    outlier_idxs = [data.index(x) for x in data if (x < mean - 3*stdev) or (x > mean + 3*stdev) if data.index(x) != 0]
    return np.array(outlier_idxs, dtype="int")

def remove_outliers(dIN_r, dOUT_r, props_r, IN_df, OUT_df, dIN_df, dOUT_df, INf_r=None, INf_df=None, manual=None):
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
    
    return dIN, dOUT, INf, props, IN_df, OUT_df, dIN_df, dOUT_df, INf_df

def split_data(dIN, dOUT, props, INf, split=0.85):
    idxs = list(range(len(dOUT)))
    random.shuffle(idxs)
    train_idxs = idxs[:int(split*len(dOUT))]
    test_idxs = [i for i in idxs if i not in train_idxs]
    train_idxs, val_idxs = train_idxs[:int(split*len(train_idxs))], train_idxs[int(split*len(train_idxs)):]
    
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
    
    return train, val, test

def save_MLdata(
    perIN_df, 
    perOUT_df, 
    train, 
    val, 
    test, 
    IN_df, 
    OUT_df, 
    dIN_df, 
    dOUT_df, 
    INf_df, 
    props, 
    PATH, 
    mode, 
    dis
):
    if mode == "UT":
        model = "Ductile"
    elif mode == "FT":
        model = "Fracture"
    os.makedirs(PATH+"/MLdata", exist_ok=True)
    
    perIN_df.to_csv(PATH + f"MLdata/{mode}-perIN.csv")
    perOUT_df.to_csv(PATH + f"MLdata/{mode}-perOUT.csv")

    IN_df.to_csv(PATH + f"{model}-{dis}-IN-noOutliers.csv")
    OUT_df.to_csv(PATH + f"{model}-{dis}-OUT-noOutliers.csv")

    dIN_df.to_csv(PATH + f"MLdata/{mode}-{dis}-allIN.csv")
    pd.DataFrame(train[0]).to_csv(PATH + f"MLdata/{mode}-{dis}-trainIN.csv")
    pd.DataFrame(val[0]).to_csv(PATH + f"MLdata/{mode}-{dis}-valIN.csv")
    pd.DataFrame(test[0]).to_csv(PATH + f"MLdata/{mode}-{dis}-testIN.csv")
    
    dOUT_df.to_csv(PATH + f"MLdata/{mode}-{dis}-allOUT.csv")
    pd.DataFrame(train[1]).to_csv(PATH + f"MLdata/{mode}-{dis}-trainOUT.csv")
    pd.DataFrame(val[1]).to_csv(PATH + f"MLdata/{mode}-{dis}-valOUT.csv")
    pd.DataFrame(test[1]).to_csv(PATH + f"MLdata/{mode}-{dis}-testOUT.csv")

    pd.DataFrame(props).to_csv(PATH + f"MLdata/{mode}-{dis}-allProps.csv")
    pd.DataFrame(train[2]).to_csv(PATH + f"MLdata/{mode}-{dis}-trainProps.csv")
    pd.DataFrame(val[2]).to_csv(PATH + f"MLdata/{mode}-{dis}-valProps.csv")
    pd.DataFrame(test[2]).to_csv(PATH + f"MLdata/{mode}-{dis}-testProps.csv")

    if train[-1] is not None:
        INf_df.to_csv(PATH + f"{model}-{dis}-INf-noOutliers.csv")
        INf_df.to_csv(PATH + f"MLdata/{mode}-{dis}-allINf.csv")
        pd.DataFrame(train[3]).to_csv(PATH + f"MLdata/{mode}-{dis}-trainINf.csv")
        pd.DataFrame(val[3]).to_csv(PATH + f"MLdata/{mode}-{dis}-valINf.csv")
        pd.DataFrame(test[3]).to_csv(PATH + f"MLdata/{mode}-{dis}-testINf.csv")

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
        plt.grid(True)
        plt.show()

def locSims(prop, OUT_df):
    max_idx, min_idx = prop.tolist().index(max(prop[1:])), prop.tolist().index(min(prop[1:]))
    nSim_max, nSim_min = OUT_df.iloc[max_idx].name, OUT_df.iloc[min_idx].name
    return nSim_max, nSim_min

def get_stats(props):
    stats = []
    for prop in props:
        mean = np.mean(prop[1:])
        st_dev = np.std(prop[1:])
        stats.append([mean, st_dev])
    return stats

def plot_frequency(raw_data, data, test, bins=50):
    raw_data = np.array(data)
    data = np.array(data)
    
    if test == "UT":
        x_label = 'Normalized Ductility'
    elif test == "FT":
        x_label = 'Normalized Fracture Toughness ($K_{IC}$)'
    
    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.set_figheight(5)
    fig1.set_figwidth(15)
    
    ax1.set_title('Raw Data', fontsize=18, fontname="Times New Roman")
    ax1.axvline(x=raw_data[0]/raw_data[0], color='orangered', label="Perfect")
    ax1.hist(raw_data[1:]/raw_data[0], bins=bins, label='Disordered')
    ax1.set_ylabel('Frequency', fontsize=15, fontname="Times New Roman")
    ax1.set_xlabel(x_label, fontsize=15, fontname="Times New Roman")
    ax1.legend()
    
    ax2.set_title('Without Outliers', fontsize=18, fontname="Times New Roman")
    ax2.axvline(x=data[0]/data[0], color='orangered', label="Perfect")
    ax2.hist(data[1:]/data[0], bins=bins, label='Disordered')
    ax2.set_ylabel('Frequency', fontsize=15, fontname="Times New Roman")
    ax2.set_xlabel(x_label, fontsize=15, fontname="Times New Roman")
    ax2.legend()
    
    plt.show()

def plot_properties(x_data, y_data, test, include_freq=False, compare_ax=None):
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if compare_ax is not None:
        include_freq = False
    
    if test == "UT":
        title = "Uniaxial Tension"
        x_label = 'Normalized Ductility'
        y_label = 'Normalized Strength'
    elif test == "FT":
        title = "Compact Tension"
        x_label = 'Normalized Fracture Toughness ($K_{IC}$)'
        y_label = 'Normalized Displacement'
    
    x_norm = x_data / x_data[0]
    y_norm = y_data / y_data[0]

    if include_freq:
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(4, 4, figure=fig)
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
        ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
        
        ax_scatter.scatter(x_norm[0], y_norm[0], c="k", marker="*", label="Perfect")
        ax_scatter.scatter(x_norm[1:], y_norm[1:], c="orangered", alpha=0.7, marker="x", label="Disordered")
        ax_scatter.axvline(1, linestyle='--', color="k")
        ax_scatter.axhline(1, linestyle='--', color="k")
        ax_scatter.set_xlabel(x_label, fontsize=15, fontname="Times New Roman")
        ax_scatter.set_ylabel(y_label, fontsize=15, fontname="Times New Roman")
        ax_scatter.legend()
        ax_scatter.set_title(title, fontsize=18, fontname="Times New Roman")

        ax_histx.hist(x_norm[1:], bins=30, color='blue', alpha=0.3)
        ax_histx.axvline(x=1, linestyle='--', color="k")
        ax_histy.hist(y_norm[1:], bins=30, color='green', alpha=0.3, orientation='horizontal')
        ax_histy.axhline(y=1, linestyle='--', color="k")

        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

        ax_histx.set_ylabel('Frequency', fontsize=15, fontname="Times New Roman")
        ax_histy.set_xlabel('Frequency', fontsize=15, fontname="Times New Roman")
        
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
            ax.scatter(x_norm[0], y_norm[0], label='Perfect', c="k", marker="*")
        ax.scatter(x_norm[1:], y_norm[1:], label=d_label, c=col, alpha=0.7, marker="x")
        ax.axvline(x=1, linestyle='--', color="k")
        ax.axhline(y=1, linestyle='--', color="k")
        ax.set_title(title, fontsize=18, fontname="Times New Roman")
        ax.set_xlabel(x_label, fontsize=15, fontname="Times New Roman")
        ax.set_ylabel(y_label, fontsize=15, fontname="Times New Roman")
        ax.legend()
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
        ax1.plot(xOUT/xOUT[indx], [i/max(p) for i in p], label="Perfect", c='k')
    
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
        ax1.set_ylabel('Normalized Stress', fontsize=15, fontname="Times New Roman")
        ax1.set_xlabel('Normalized Strain', fontsize=15, fontname="Times New Roman")
    if mode.lower() == "ft":
        ax1.set_ylabel('Normalized Force', fontsize=15, fontname="Times New Roman")
        ax1.set_xlabel('Normalized Displacement', fontsize=15, fontname="Times New Roman")
    
    ax1.axvline(x=1, ymax=0.2, c='k', linestyle='--')
    ax1.axhline(y=1, xmax=0.2, c='k', linestyle='--')
    ax1.set_ylim(bottom=-0.1, top=1.1)

    if idx or q != 'all' and q <= 10:
        ax1.legend()
    ax1.grid()

    return fig2, ax1


def load_TrainTestData(CSV_all_in, CSV_all_out, CSV_all_props, CSV_train_in, CSV_train_out, CSV_trainProps, CSV_val_in, CSV_val_out, CSV_valProps, CSV_test_in, CSV_test_out, CSV_testProps):
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


def dataParams(x):
    return [np.min(x), np.max(x), np.mean(x), np.std(x)]

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
        dis="disNodes", 
        dN=20, 
        mechMode="UT",
        model="MLP", 
        freq=False,
        scale=False, 
        format=0
    ):
        self.path = path
        self.path_add = path_add
        self.LAT = LAT
        self.dis = dis
        self.dN = dN
        self.mechMode = mechMode
        self.model = model
        self.freq = freq
        
        self.scale = scale
        if scale:
            if scale[0].lower() == "minmax":
                self.scaler = MinMaxScaler()
            elif scale[0].lower() == "standardscaler" or scale.lower() == "standard":
                self.scaler = StandardScaler()
            elif scale[0].lower() == "standardize":
                self.scaler = standardize
            elif scale[0].lower() == "normalize":
                self.scaler = normalize

        if path_add.lower() == "frequency":
            self.freq = True

        if mechMode.lower() == "ut":
            self.mechTest = "Ductile"
        elif mechMode.lower() == "ft":
            self.mechTest = "Fracture"
        elif mechMode.lower() == "both":
            self.mechTest = "both"
        
        self.geom = geometry(LAT=self.LAT, l=10)
        nnx, nny, L, H, W, B, a0, ai, totalNodes, totalBracketNodes, deltaNM, vol, l, t, LAT = self.geom
        self.nnx = nnx
        self.nny = nny

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
            self.PATH  = pUTdisNodes2
        elif self.path == 1:
            self.PATH  = pTiLAT
        else:
            self.PATH  = str(self.path)+"/"

    def get_DataFiles(self):
        self.INcsv             = self.PATH + f'{self.mechTest}-disNodes-IN.csv'
        self.INcsv_noOutliers  = self.PATH + f'{self.mechTest}-disNodes-IN-noOutliers.csv'
        self.OUTcsv            = self.PATH + f'{self.mechTest}-disNodes-OUT.csv'
        self.OUTcsv_noOutliers = self.PATH + f'{self.mechTest}-disNodes-OUT-noOutliers.csv'

        self.CSV_all_in    = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-allIN.csv'
        self.CSV_all_out   = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-allOUT.csv'
        self.CSV_train_in  = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-trainIN.csv'
        self.CSV_train_out = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-trainOUT.csv'
        self.CSV_val_in    = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-valIN.csv'
        self.CSV_val_out   = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-valOUT.csv'
        self.CSV_test_in   = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-testIN.csv'
        self.CSV_test_out  = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-testOUT.csv'

        self.CSV_allProps   = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-allProps.csv'
        self.CSV_trainProps = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-trainProps.csv'
        self.CSV_valProps   = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-valProps.csv'
        self.CSV_testProps  = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-testProps.csv'

        if self.freq:
            self.INcsv_f            = self.PATH + f'{self.mechTest}-disNodes-INf.csv'
            self.INcsv_f_noOutliers = self.PATH + f'{self.mechTest}-disNodes-INf-noOutliers.csv'

            self.CSV_all_in_f       = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-allINf.csv'
            self.CSV_train_in_f     = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-trainINf.csv'
            self.CSV_val_in_f       = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-valINf.csv'
            self.CSV_test_in_f      = self.PATH + f'MLdata/{self.mechMode}-{self.dis}-testINf.csv'

    def load_data(self):
        self.IN_df, self.OUT_df, self.INf_df, self.perINr_df, self.perIN_df, self.perOUT_df, self.dIN_df, self.dOUT_df = load_data(self.INcsv, 
                                                                                                                                   self.OUTcsv, 
                                                                                                                                   self.INcsv_f if self.freq else None,
                                                                                                                                   no_outliers=[self.INcsv_noOutliers, 
                                                                                                                                                self.OUTcsv_noOutliers, 
                                                                                                                                                self.INcsv_f_noOutliers if self.freq else None])
        all, train, val, test = load_TrainTestData(self.CSV_all_in,
                                                   self.CSV_all_out,
                                                   self.CSV_allProps,
                                                   self.CSV_train_in, 
                                                   self.CSV_train_out,
                                                   self.CSV_trainProps, 
                                                   self.CSV_val_in, 
                                                   self.CSV_val_out, 
                                                   self.CSV_valProps,
                                                   self.CSV_test_in, 
                                                   self.CSV_test_out,
                                                   self.CSV_testProps)
        all_in, all_out, allProps       = all
        train_in, train_out, trainProps = train
        val_in, val_out, valProps       = val
        test_in, test_out, testProps    = test

        if self.freq:
            all_in, train_in, val_in, test_in = load_freqInputData(self.CSV_all_in_f,
                                                                   self.CSV_train_in_f, 
                                                                   self.CSV_val_in_f, 
                                                                   self.CSV_test_in_f)

        if self.model.lower() == "mlp" or self.model.lower() == "gpr":
            self.all_in   = all_in
            self.train_in = train_in
            self.val_in   = val_in
            self.test_in  = test_in
        elif self.model.lower() == "gnn":
            self.all_in   = all_in.reshape(*all_in.shape[:-1], all_in.shape[-1]//2, 2)
            self.train_in = train_in.reshape(*train_in.shape[:-1], train_in.shape[-1]//2, 2)
            self.val_in   = val_in.reshape(*val_in.shape[:-1], val_in.shape[-1]//2, 2)
            self.test_in  = test_in.reshape(*test_in.shape[:-1], test_in.shape[-1]//2, 2)
        self.all_out, self.allProps     = all_out, allProps
        self.train_out, self.trainProps = train_out, trainProps
        self.val_out, self.valProps     = val_out, valProps
        self.test_out, self.testProps   = test_out, testProps

        if self.scale:
            if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                self.INscaler = self.scaler
                self.all_in   = self.INscaler.fit_transform(self.all_in)
                self.train_in = self.INscaler.transform(self.train_in)
                self.val_in   = self.INscaler.transform(self.val_in)
                self.test_in  = self.INscaler.transform(self.test_in)
            if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                self.OUTscaler = self.scaler
                self.all_out   = self.OUTscaler.fit_transform(self.all_out)
                self.train_out = self.OUTscaler.transform(self.train_out)
                self.val_out   = self.OUTscaler.transform(self.val_out)
                self.test_out  = self.OUTscaler.transform(self.test_out)
            if "props" in self.scale[1].lower() or "all" in self.scale[1].lower():
                self.PROPscaler = self.scaler
                self.allProps   = self.PROPscaler.fit_transform(self.allProps.T).T
                self.trainProps = self.PROPscaler.transform(self.trainProps.T).T
                self.valProps   = self.PROPscaler.transform(self.valProps.T).T
                self.testProps  = self.PROPscaler.transform(self.testProps.T).T

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