from resources.imports import *
from resources.calculations import calcUT, calcFT
from resources.lattices import Geometry, effProperties

from matplotlib.gridspec import GridSpec

from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import clone


def load_data(inputs, outputs, f_inputs=None, props=None):
    IN_df   = pd.read_csv(inputs, index_col=0, header=0)
    OUTr_df = pd.read_csv(outputs, index_col=0, header=0)
    INf_df  = None
    if f_inputs is not None:
        INf_df = pd.read_csv(f_inputs, index_col=0, header=0)
    props_df = None
    if props is not None:
        props_df = pd.read_csv(props, index_col=0, header=0)

    dIN_df  = IN_df - IN_df.iloc[0].values
    INfixed_cols = dIN_df.loc[:, (dIN_df == 0.0).all()].columns
    dIN_df  = dIN_df.drop(columns=INfixed_cols) 
    dOUT_df = OUTr_df - OUTr_df.iloc[1].values
    dOUT_df = dOUT_df.drop(columns='0')
    dOUT_df = dOUT_df.iloc[1:].sort_index()
    OUT_df  = OUTr_df.iloc[0].to_frame().T.append(OUTr_df.iloc[1:].sort_index())
    
    return IN_df, OUT_df, INf_df, dIN_df, dOUT_df, props_df


def UTprops(OUT_df):    
    ducts, strens, stiffs, WoFs = [], [], [], []
    for _, row in OUT_df.iloc[1:].iterrows():
        UT_df = pd.DataFrame({'x':OUT_df.iloc[0].to_list(), 'y_sm':row})
        ductility, strength, stiffness, WoF = calcUT(UT_df)
        
        ducts.append(ductility)
        strens.append(strength)
        stiffs.append(stiffness)
        WoFs.append(WoF)
        
    props = np.array([ducts, strens, stiffs, WoFs])
    props_df = pd.DataFrame(props.T, columns=['Ductility', 'Strength', 'Stiffness', 'WoF'], index=OUT_df.iloc[1:].index)
    return props_df

def FTprops(OUT_df, geom, E_eff_pe):
    Kjs, Ks, Ps, ds = [], [], [], []
    for _, row in OUT_df.iloc[1:].iterrows():
        FT_df = pd.DataFrame({'x':OUT_df.iloc[0].to_list(), 'y_sm':row})
        P, dd, K, Kj = calcFT(FT_df, geom, E_eff_pe, n_Ks=1)
        
        Kjs.append(Kj[0])
        Ks.append(K[0])
        Ps.append(P)
        ds.append(dd)
    
    props = [Kjs, Ks, Ps, ds]
    props_df = pd.DataFrame(np.array(props).T, columns=['K_JIC', 'K_IC', 'Force', 'Displacement'], index=OUT_df.iloc[1:].index)
    return props_df

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
    norm_df = (common_props_df/common_props_df.loc[0])
    common_props_df["Multi"] = norm_df["Ductility"]**2 + norm_df["K_JIC"]**2 + norm_df["WoF"] + norm_df["Displacement"] + norm_df["Strength"]
    common_props_df["FCL"] = (common_props_df["K_JIC"]**2 / E_eff_pe) / (common_props_df["WoF"] * 1e6)
    common_props_df = common_props_df.replace([np.inf, -np.inf], np.nan).dropna()
    common_idxs = common_props_df.index

    UT_IN_df, FT_IN_df = UT_IN_df.loc[common_idxs], FT_IN_df.loc[common_idxs]
    UT_OUT_df, FT_OUT_df = UT_OUT_df.loc[common_idxs], FT_OUT_df.loc[common_idxs]
    UT_dIN_df, FT_dIN_df = UT_dIN_df.loc[common_idxs], FT_dIN_df.loc[common_idxs]
    UT_dOUT_df, FT_dOUT_df = UT_dOUT_df.loc[common_idxs], FT_dOUT_df.loc[common_idxs]
    if INf_dfs[0] is not None and INf_dfs[1] is not None:
        UT_INf_df, FT_INf_df = UT_INf_df.loc[common_idxs], FT_INf_df.loc[common_idxs]

    return common_props_df, [UT_IN_df, FT_IN_df], [UT_OUT_df, FT_OUT_df], [UT_dIN_df, FT_dIN_df], [UT_dOUT_df, FT_dOUT_df], [UT_INf_df, FT_INf_df] if INf_dfs[0] is not None and INf_dfs[1] is not None else INf_dfs

def remove_outliers(IN_df, OUT_df, dIN_df, dOUT_df, props_df, INf_df=None, manual=None):
    z_scores = (props_df - props_df.mean()) / props_df.std()
    outlier_idxs = props_df.iloc[1:].index[(z_scores.iloc[1:].abs() > 3).any(axis=1)]

    if len(outlier_idxs) > 0:
        IN_df = IN_df.drop(IN_df.loc[outlier_idxs].index)
        OUT_df = OUT_df.drop(OUT_df.loc[outlier_idxs].index)
        dIN_df = dIN_df.drop(dIN_df.loc[outlier_idxs].index)
        dOUT_df = dOUT_df.drop(dOUT_df.loc[outlier_idxs].index)
        props_df = props_df.drop(props_df.loc[outlier_idxs].index)
        if INf_df is not None:
            INf_df = INf_df.drop(INf_df.loc[outlier_idxs].index)
        
        if manual is not None:
            manual = np.array(manual, dtype="int")

            IN_df = IN_df.drop(IN_df.loc[manual].index)
            OUT_df = OUT_df.drop(OUT_df.loc[manual].index)
            dIN_df = dIN_df.drop(dIN_df.loc[manual].index)
            dOUT_df = dOUT_df.drop(dOUT_df.loc[manual].index)
            props_df = props_df.drop(props_df.loc[manual].index)
            if INf_df is not None:
                INf_df = INf_df.drop(INf_df.loc[manual].index)
    
    return IN_df, OUT_df, dIN_df, dOUT_df, props_df, INf_df


def save_MLdata(IN_df, OUT_df, props_df, PATH, mode, dis, INf_df=None):
    os.makedirs(PATH+f"/MLdata", exist_ok=True)

    IN_df.to_csv(PATH + f"MLdata/{mode}-{mode}-{dis}-allIN.csv")
    OUT_df.to_csv(PATH + f"MLdata/{mode}-{mode}-{dis}-allOUT.csv")
    props_df.to_csv(PATH + f"MLdata/{mode}-{dis}-allProps.csv")
    if INf_df is not None:
        INf_df.to_csv(PATH + f"MLdata/{mode}-{mode}-{dis}-allINf.csv")

def save_MULTIdata(IN_dfs, OUT_dfs, common_props_df, PATH, dis, INf_dfs=None):
    os.makedirs(PATH+"/MLdata", exist_ok=True)

    IN_dfs[0].to_csv(PATH + f"MLdata/MULTI-UT-{dis}-allIN.csv")
    IN_dfs[1].to_csv(PATH + f"MLdata/MULTI-FT-{dis}-allIN.csv")
    OUT_dfs[0].to_csv(PATH + f"MLdata/MULTI-UT-{dis}-allOUT.csv")
    OUT_dfs[1].to_csv(PATH + f"MLdata/MULTI-FT-{dis}-allOUT.csv")
    common_props_df.to_csv(PATH + f"MLdata/MULTI-{dis}-allProps.csv")
    if INf_dfs is not None and INf_dfs[0] is not None and INf_dfs[1] is not None:
        INf_dfs[0].to_csv(PATH + f"MLdata/MULTI-UT-{dis}-allINf.csv")
        INf_dfs[1].to_csv(PATH + f"MLdata/MULTI-FT-{dis}-allINf.csv")
    

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
                
def get_stats(props_df):
    stats = pd.DataFrame((props_df.iloc[1:].mean(), props_df.iloc[1:].std()), index=['Mean', 'Std'])
    stats_Pdiff = (stats - props_df.iloc[0]) / props_df.iloc[0]
    stats_Pdiff.index = ['\\%d Mean', '\\%d Std']
    return pd.concat([stats, stats_Pdiff]), pd.DataFrame((props_df.iloc[1:].idxmax(), props_df.iloc[1:].idxmin()), index=['Max', 'Min'])

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

def plot_curve(OUT_df, mode, idx=None, q=15, compare_ax=None):
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
    
    xOUT = OUT_df.iloc[0].to_numpy()[1:]
    p = OUT_df.iloc[1].tolist()[1:]
    indx = int(OUT_df.iloc[1].tolist()[0])

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


def load_splitData(PATH, mechMode, dis, split_name="Final"):
    train_in   = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-trainIN-{split_name}.csv", index_col=0, header=0).to_numpy()
    train_out  = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-trainOUT-{split_name}.csv", index_col=0, header=0).to_numpy()
    trainProps = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-trainProps-{split_name}.csv", index_col=0, header=0).to_numpy()
    val_in     = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-valIN-{split_name}.csv", index_col=0, header=0).to_numpy()
    val_out    = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-valOUT-{split_name}.csv", index_col=0, header=0).to_numpy()
    valProps   = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-valProps-{split_name}.csv", index_col=0, header=0).to_numpy()
    test_in    = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-testIN-{split_name}.csv", index_col=0, header=0).to_numpy()
    test_out   = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-testOUT-{split_name}.csv", index_col=0, header=0).to_numpy()
    testProps  = pd.read_csv(f"{PATH}splits/{mechMode}-{dis}-testProps-{split_name}.csv", index_col=0, header=0).to_numpy()

    train, val, test = [train_in, train_out, trainProps], [val_in, val_out, valProps], [test_in, test_out, testProps]
    return train, val, test

def load_akData(CSV_train_in, CSV_train_out, CSV_val_in, CSV_val_out, CSV_test_in, CSV_test_out):
    train_in   = pd.read_csv(CSV_train_in, index_col=0, header=0).to_numpy()
    train_out  = pd.read_csv(CSV_train_out, index_col=0, header=0).to_numpy()
    val_in     = pd.read_csv(CSV_val_in, index_col=0, header=0).to_numpy()
    val_out    = pd.read_csv(CSV_val_out, index_col=0, header=0).to_numpy()
    test_in    = pd.read_csv(CSV_test_in, index_col=0, header=0).to_numpy()
    test_out   = pd.read_csv(CSV_test_out, index_col=0, header=0).to_numpy()

    train, val, test = [train_in, train_out], [val_in, val_out], [test_in, test_out]
    return train, val, test

def split_data(dIN, dOUT, props, split=0.8):
    train_in, test_in, train_out, test_out, train_props, test_props = train_test_split(dIN, dOUT, props, train_size=split,
                                                                                       random_state=None, shuffle=True, stratify=None)
    train_in, val_in, train_out, val_out, train_props, val_props = train_test_split(train_in, train_out, train_props, train_size=split,
                                                                                    random_state=None, shuffle=True, stratify=None)
    
    train = [train_in, train_out, train_props]
    val = [val_in, val_out, val_props]
    test = [test_in, test_out, test_props]
    
    return train, val, test

def save_splitData(): # TODO: finish
    pass

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

class Dataset_(Dataset): # TODO: check if this is needed or if we can just use the standard PyTorch Dataset class
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
        load_split=False,
        split_frac=0.8,
        save_split=False,
        LAT="FCC", 
        nnx=None,
        dis="disNodes", 
        dN=20, 
        mechMode="MULTI",
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
        self.load_split = load_split
        self.split_frac = split_frac
        self.LAT = LAT
        self.nnx = nnx
        self.dis = dis
        self.dN = dN
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

        self.mechMode = mechMode.upper()
        if mechMode.lower() == "ut":
            self.mechTest = "Ductile"
            self.UTmechTest = True
            self.FTmechTest = False
        elif mechMode.lower() == "ft":
            self.mechTest = "Fracture"
            self.UTmechTest = False
            self.FTmechTest = True
        elif mechMode.lower() == "multi":
            self.UTmechTest = True
            self.FTmechTest = True
        
        if nnx is None:
            if LAT.lower() in ["fcc", "kagome", "hex"]:
                self.nnx = 20
            elif LAT.lower() == "tri":
                self.nnx = 30
        
        self.calc_geom()
        self.get_DataPath()

        if load:
            # self.get_DataFiles()
            if path == 0:
                self.load_data(typ="a")
            else:
                self.load_data()

            if format == 1 and model.lower() == "mlp" or model.lower() == "gpr":
                self.load_DisDist_v1()
            elif format == 2 and model.lower() == "mlp":
                self.load_DisDist_v2()

    def calc_geom(self):
        self.geom = Geometry(LAT=self.LAT, l=10, nnx=self.nnx)
        self.E_s = 123e9  ## Pa
        self.v_s = 0.3
        self.E_eff, self.v_eff, self.E_eff_pe, self.v_eff_pe = effProperties(self.LAT, self.geom)

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
    
    def load_data(self, typ="n"):
        if typ.lower() == "a":
            self.UT_IN_df, \
            self.UT_OUT_df, \
            self.UT_INf_df, \
            self.UT_dIN_df, \
            self.UT_dOUT_df, \
            self.UT_props_df = load_data(self.PATH+f"{self.mechTest}-{self.dis}-IN.csv", 
                                        self.PATH+f"{self.mechTest}-{self.dis}-OUT.csv")
            UT_train, UT_val, UT_test = load_akData(self.PATH+f"NN-{self.mechMode}-{self.dis}-trainIN.csv", 
                                                    self.PATH+f"NN-{self.mechMode}-{self.dis}-trainOUT.csv",
                                                    self.PATH+f"NN-{self.mechMode}-{self.dis}-valIN.csv", 
                                                    self.PATH+f"NN-{self.mechMode}-{self.dis}-valOUT.csv", 
                                                    self.PATH+f"NN-{self.mechMode}-{self.dis}-testIN.csv", 
                                                    self.PATH+f"NN-{self.mechMode}-{self.dis}-testOUT.csv")

            self.UT_train_in, self.UT_train_out = UT_train
            self.UT_val_in, self.UT_val_out = UT_val
            self.UT_test_in, self.UT_test_out = UT_test

            if self.scale:
                    if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.UT_INscaler = clone(self.scaler)
                        self.UT_train_in = self.UT_INscaler.fit_transform(self.UT_train_in)
                        self.UT_val_in   = self.UT_INscaler.transform(self.UT_val_in)
                        self.UT_test_in  = self.UT_INscaler.transform(self.UT_test_in)
                    if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.UT_OUTscaler = clone(self.scaler)
                        self.UT_train_out = self.UT_OUTscaler.fit_transform(self.UT_train_out)
                        self.UT_val_out   = self.UT_OUTscaler.transform(self.UT_val_out)
                        self.UT_test_out  = self.UT_OUTscaler.transform(self.UT_test_out)

        else:
            if self.UTmechTest:
                self.UT_IN_df, \
                self.UT_OUT_df, \
                self.UT_dIN_df, \
                self.UT_INf_df, \
                self.UT_dOUT_df, \
                self.UT_props_df = load_data(self.PATH+f"MLdata/{self.mechMode}-UT-{self.dis}-allIN.csv", 
                                             self.PATH+f"MLdata/{self.mechMode}-UT-{self.dis}-allOUT.csv", 
                                             self.PATH+f"MLdata/{self.mechMode}-UT-{self.dis}-allINf.csv" if self.freq else None,
                                             self.PATH+f"MLdata/{self.mechMode}-{self.dis}-allProps.csv")
                cols = ['Ductility', 'Strength', 'Stiffness', 'WoF']
                if self.mechMode.lower() == "multi":
                    cols = ['Ductility', 'Strength', 'Stiffness', 'WoF', 'K_JIC', 'K_IC', 'Force', 'Displacement', 'Multi', 'FCL']
                self.UT_props_df.columns = cols

                if self.nsims is not None:
                    self.UT_props_df, self.UT_OUT_df, self.UT_dOUT_df, self.UT_IN_df, self.UT_dIN_df = self.UT_props_df[:self.nsims], self.UT_OUT_df[:self.nsims], self.UT_dOUT_df[:self.nsims], self.UT_IN_df[:self.nsims], self.UT_dIN_df[:self.nsims]
                    if self.freq:
                        self.UT_INf_df = self.UT_INf_df[:self.nsims]
                
                if self.load_split:
                    UT_train, UT_val, UT_test = load_splitData(self.PATH, self.mechMode, self.dis, self.load_split, self.freq)
                else:
                    UT_train, UT_val, UT_test = split_data(self.UT_dIN_df, self.UT_dOUT_df, self.UT_props_df, split=self.split_frac)
                self.UT_train_in, self.UT_train_out, self.UT_trainProps = UT_train
                self.UT_val_in, self.UT_val_out, self.UT_valProps       = UT_val
                self.UT_test_in, self.UT_test_out, self.UT_testProps    = UT_test

                # self.UT_trainProps_df = pd.DataFrame(self.UT_trainProps.T, columns=cols)
                # self.UT_valProps_df   = pd.DataFrame(self.UT_valProps.T, columns=cols)
                # self.UT_testProps_df  = pd.DataFrame(self.UT_testProps.T, columns=cols)

                if self.scale:
                    if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.UT_INscaler = clone(self.scaler)
                        self.UT_train_in = self.UT_INscaler.fit_transform(self.UT_train_in)
                        self.UT_val_in   = self.UT_INscaler.transform(self.UT_val_in)
                        self.UT_test_in  = self.UT_INscaler.transform(self.UT_test_in)
                    if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.UT_OUTscaler = clone(self.scaler)
                        self.UT_train_out = self.UT_OUTscaler.fit_transform(self.UT_train_out)
                        self.UT_val_out   = self.UT_OUTscaler.transform(self.UT_val_out)
                        self.UT_test_out  = self.UT_OUTscaler.transform(self.UT_test_out)
                    if "props" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.UT_PROPscaler = clone(self.scaler)
                        self.UT_trainProps = self.UT_PROPscaler.fit_transform(self.UT_trainProps.T).T
                        self.UT_valProps   = self.UT_PROPscaler.transform(self.UT_valProps.T).T
                        self.UT_testProps  = self.UT_PROPscaler.transform(self.UT_testProps.T).T
                
                if self.reduce_dim:
                    if "in" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                        self.UT_INreducer = copy.deepcopy(self.reducer)
                        self.UT_INreducer.fit(self.UT_train_in)
                        self.UT_train_in = self.UT_INreducer.reduce(self.UT_train_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.UT_val_in   = self.UT_INreducer.reduce(self.UT_val_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.UT_test_in  = self.UT_INreducer.reduce(self.UT_test_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    if "out" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                        self.UT_OUTreducer = copy.deepcopy(self.reducer)
                        self.UT_OUTreducer.fit(self.UT_train_out)
                        self.UT_train_out = self.UT_OUTreducer.reduce(self.UT_train_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.UT_val_out   = self.UT_OUTreducer.reduce(self.UT_val_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.UT_test_out  = self.UT_OUTreducer.reduce(self.UT_test_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])

                if self.model.lower() == "gnn":
                    self.UT_train_in = self.UT_train_in.reshape(*self.UT_train_in.shape[:-1], self.UT_train_in.shape[-1]//2, 2)
                    self.UT_val_in   = self.UT_val_in.reshape(*self.UT_val_in.shape[:-1], self.UT_val_in.shape[-1]//2, 2)
                    self.UT_test_in  = self.UT_test_in.reshape(*self.UT_test_in.shape[:-1], self.UT_test_in.shape[-1]//2, 2)


            if self.FTmechTest:
                self.FT_IN_df, \
                self.FT_OUT_df, \
                self.FT_INf_df, \
                self.FT_dIN_df, \
                self.FT_dOUT_df, \
                self.FT_props_df = load_data(self.PATH+f"MLdata/{self.mechMode}-FT-{self.dis}-allIN.csv", 
                                             self.PATH+f"MLdata/{self.mechMode}-FT-{self.dis}-allOUT.csv", 
                                             self.PATH+f"MLdata/{self.mechMode}-FT-{self.dis}-allINf.csv" if self.freq else None,
                                             self.PATH+f"MLdata/{self.mechMode}-{self.dis}-allProps.csv")
                cols = ['K_JIC', 'K_IC', 'Force', 'Displacement']
                if self.mechMode.lower() == "multi":
                    cols = ['Ductility', 'Strength', 'Stiffness', 'WoF', 'K_JIC', 'K_IC', 'Force', 'Displacement', 'Multi', 'FCL']
                self.FT_props_df.columns = cols

                if self.nsims is not None:
                    self.FT_props_df, self.FT_OUT_df, self.FT_dOUT_df, self.FT_IN_df, self.FT_dIN_df = self.FT_props_df[:self.nsims], self.FT_OUT_df[:self.nsims], self.FT_dOUT_df[:self.nsims], self.FT_IN_df[:self.nsims], self.FT_dIN_df[:self.nsims]
                    if self.freq:
                        self.FT_INf_df = self.FT_INf_df[:self.nsims]

                if self.load_split:
                    FT_train, FT_val, FT_test = load_splitData(self.PATH, self.mechMode, self.dis, self.load_split, self.freq)
                else:
                    FT_train, FT_val, FT_test = split_data(self.FT_dIN_df, self.FT_dOUT_df, self.FT_props_df, split=self.split_frac)
                self.FT_train_in, self.FT_train_out, self.FT_trainProps = FT_train
                self.FT_val_in, self.FT_val_out, self.FT_valProps       = FT_val
                self.FT_test_in, self.FT_test_out, self.FT_testProps    = FT_test                
                
                # self.FT_trainProps_df = pd.DataFrame(self.FT_trainProps.T, columns=cols)
                # self.FT_valProps_df   = pd.DataFrame(self.FT_valProps.T, columns=cols)
                # self.FT_testProps_df  = pd.DataFrame(self.FT_testProps.T, columns=cols)

                if self.scale:
                    if "in" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.FT_INscaler = clone(self.scaler)
                        self.FT_train_in = self.FT_INscaler.fit_transform(self.FT_train_in)
                        self.FT_val_in   = self.FT_INscaler.transform(self.FT_val_in)
                        self.FT_test_in  = self.FT_INscaler.transform(self.FT_test_in)
                    if "out" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.FT_OUTscaler = clone(self.scaler)
                        self.FT_train_out = self.FT_OUTscaler.fit_transform(self.FT_train_out)
                        self.FT_val_out   = self.FT_OUTscaler.transform(self.FT_val_out)
                        self.FT_test_out  = self.FT_OUTscaler.transform(self.FT_test_out)
                    if "props" in self.scale[1].lower() or "all" in self.scale[1].lower():
                        self.FT_PROPscaler = clone(self.scaler)
                        self.FT_trainProps = self.FT_PROPscaler.fit_transform(self.FT_trainProps.T).T
                        self.FT_valProps   = self.FT_PROPscaler.transform(self.FT_valProps.T).T
                        self.FT_testProps  = self.FT_PROPscaler.transform(self.FT_testProps.T).T
                
                if self.reduce_dim:
                    if "in" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                        self.FT_INreducer = copy.deepcopy(self.reducer)
                        self.FT_INreducer.fit(self.FT_train_in)
                        self.FT_train_in = self.FT_INreducer.reduce(self.FT_train_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.FT_val_in   = self.FT_INreducer.reduce(self.FT_val_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.FT_test_in  = self.FT_INreducer.reduce(self.FT_test_in, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                    if "out" in self.reduce_dim[1].lower() or "all" in self.reduce_dim[1].lower():
                        self.FT_OUTreducer = copy.deepcopy(self.reducer)
                        self.FT_OUTreducer.fit(self.FT_train_out)
                        self.FT_train_out = self.FT_OUTreducer.reduce(self.FT_train_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.FT_val_out   = self.FT_OUTreducer.reduce(self.FT_val_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        self.FT_test_out  = self.FT_OUTreducer.reduce(self.FT_test_out, accuracy=self.reduce_dim[2], n_components=self.reduce_dim[3])
                        
                if self.model.lower() == "gnn":
                    self.FT_train_in = self.FT_train_in.reshape(*self.FT_train_in.shape[:-1], self.FT_train_in.shape[-1]//2, 2)
                    self.FT_val_in   = self.FT_val_in.reshape(*self.FT_val_in.shape[:-1], self.FT_val_in.shape[-1]//2, 2)
                    self.FT_test_in  = self.FT_test_in.reshape(*self.FT_test_in.shape[:-1], self.FT_test_in.shape[-1]//2, 2)
                

            if self.UTmechTest and self.FTmechTest:  # TODO: remove?
                self.common_props_df = self.UT_props_df





    # def get_DataFiles(self):
    #     if self.UTmechTest and not self.multi:
    #         if self.path == 1:
    #             MLpath = self.PATH + "MLdata/UT/"
    #         elif self.path == 0:
    #             MLpath = self.PATH + "NN-"
    #         self.UT_INcsv             = self.PATH + f'Ductile-disNodes-IN.csv'
    #         self.UT_INcsv_noOutliers  = self.PATH + f'Ductile-disNodes-IN-noOutliers.csv'
    #         self.UT_OUTcsv            = self.PATH + f'Ductile-disNodes-OUT.csv'
    #         self.UT_OUTcsv_noOutliers = self.PATH + f'Ductile-disNodes-OUT-noOutliers.csv'

    #         self.UT_CSV_all_in    = f'{MLpath}UT-{self.dis}-allIN.csv'
    #         self.UT_CSV_all_out   = f'{MLpath}UT-{self.dis}-allOUT.csv'
    #         self.UT_CSV_train_in  = f'{MLpath}UT-{self.dis}-trainIN.csv'
    #         self.UT_CSV_train_out = f'{MLpath}UT-{self.dis}-trainOUT.csv'
    #         self.UT_CSV_val_in    = f'{MLpath}UT-{self.dis}-valIN.csv'
    #         self.UT_CSV_val_out   = f'{MLpath}UT-{self.dis}-valOUT.csv'
    #         self.UT_CSV_test_in   = f'{MLpath}UT-{self.dis}-testIN.csv'
    #         self.UT_CSV_test_out  = f'{MLpath}UT-{self.dis}-testOUT.csv'

    #         self.UT_CSV_allProps   = f'{MLpath}UT-{self.dis}-allProps.csv'
    #         self.UT_CSV_trainProps = f'{MLpath}UT-{self.dis}-trainProps.csv'
    #         self.UT_CSV_valProps   = f'{MLpath}UT-{self.dis}-valProps.csv'
    #         self.UT_CSV_testProps  = f'{MLpath}UT-{self.dis}-testProps.csv'
            
    #         self.UT_INcsv_f            = None
    #         self.UT_INcsv_f_noOutliers = None

    #         self.UT_CSV_all_in_f       = None
    #         self.UT_CSV_train_in_f     = None
    #         self.UT_CSV_val_in_f       = None
    #         self.UT_CSV_test_in_f      = None

    #         if self.freq:
    #             self.UT_INcsv_f            = self.PATH + f'Ductile-disNodes-INf.csv'
    #             self.UT_INcsv_f_noOutliers = self.PATH + f'Ductile-disNodes-INf-noOutliers.csv'

    #             self.UT_CSV_all_in_f       = f'{MLpath}UT-{self.dis}-allINf.csv'
    #             self.UT_CSV_train_in_f     = f'{MLpath}UT-{self.dis}-trainINf.csv'
    #             self.UT_CSV_val_in_f       = f'{MLpath}UT-{self.dis}-valINf.csv'
    #             self.UT_CSV_test_in_f      = f'{MLpath}UT-{self.dis}-testINf.csv'

    #     if self.FTmechTest and not self.multi:
    #         self.FT_INcsv             = self.PATH + f'Fracture-disNodes-IN.csv'
    #         self.FT_INcsv_noOutliers  = self.PATH + f'Fracture-disNodes-IN-noOutliers.csv'
    #         self.FT_OUTcsv            = self.PATH + f'Fracture-disNodes-OUT.csv'
    #         self.FT_OUTcsv_noOutliers = self.PATH + f'Fracture-disNodes-OUT-noOutliers.csv'

    #         self.FT_CSV_all_in    = self.PATH + f'MLdata/FT/FT-{self.dis}-allIN.csv'
    #         self.FT_CSV_all_out   = self.PATH + f'MLdata/FT/FT-{self.dis}-allOUT.csv'
    #         self.FT_CSV_train_in  = self.PATH + f'MLdata/FT/FT-{self.dis}-trainIN.csv'
    #         self.FT_CSV_train_out = self.PATH + f'MLdata/FT/FT-{self.dis}-trainOUT.csv'
    #         self.FT_CSV_val_in    = self.PATH + f'MLdata/FT/FT-{self.dis}-valIN.csv'
    #         self.FT_CSV_val_out   = self.PATH + f'MLdata/FT/FT-{self.dis}-valOUT.csv'
    #         self.FT_CSV_test_in   = self.PATH + f'MLdata/FT/FT-{self.dis}-testIN.csv'
    #         self.FT_CSV_test_out  = self.PATH + f'MLdata/FT/FT-{self.dis}-testOUT.csv'

    #         self.FT_CSV_allProps   = self.PATH + f'MLdata/FT/FT-{self.dis}-allProps.csv'
    #         self.FT_CSV_trainProps = self.PATH + f'MLdata/FT/FT-{self.dis}-trainProps.csv'
    #         self.FT_CSV_valProps   = self.PATH + f'MLdata/FT/FT-{self.dis}-valProps.csv'
    #         self.FT_CSV_testProps  = self.PATH + f'MLdata/FT/FT-{self.dis}-testProps.csv'
            
    #         self.FT_INcsv_f            = None
    #         self.FT_INcsv_f_noOutliers = None

    #         self.FT_CSV_all_in_f       = None
    #         self.FT_CSV_train_in_f     = None
    #         self.FT_CSV_val_in_f       = None
    #         self.FT_CSV_test_in_f      = None

    #         if self.freq:
    #             self.FT_INcsv_f            = self.PATH + f'Fracture-disNodes-INf.csv'
    #             self.FT_INcsv_f_noOutliers = self.PATH + f'Fracture-disNodes-INf-noOutliers.csv'

    #             self.FT_CSV_all_in_f       = self.PATH + f'MLdata/FT/FT-{self.dis}-allINf.csv'
    #             self.FT_CSV_train_in_f     = self.PATH + f'MLdata/FT/FT-{self.dis}-trainINf.csv'
    #             self.FT_CSV_val_in_f       = self.PATH + f'MLdata/FT/FT-{self.dis}-valINf.csv'
    #             self.FT_CSV_test_in_f      = self.PATH + f'MLdata/FT/FT-{self.dis}-testINf.csv'

    #     if self.UTmechTest and self.FTmechTest and self.multi:
    #         self.UT_CSV_allProps   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-allProps.csv'
    #         self.UT_CSV_trainProps = self.PATH + f'MLdata/multi/MULTI-{self.dis}-trainProps.csv'
    #         self.UT_CSV_valProps   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-valProps.csv'
    #         self.UT_CSV_testProps  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-testProps.csv'
    #         self.FT_CSV_allProps   = self.UT_CSV_allProps
    #         self.FT_CSV_trainProps = self.UT_CSV_trainProps
    #         self.FT_CSV_valProps   = self.UT_CSV_valProps
    #         self.FT_CSV_testProps  = self.UT_CSV_testProps

    #         self.UT_INcsv             = self.PATH + f'Ductile-disNodes-IN.csv'
    #         self.UT_INcsv_noOutliers  = self.PATH + f'MULTI-Ductile-disNodes-IN-noOutliers.csv'
    #         self.UT_OUTcsv            = self.PATH + f'Ductile-disNodes-OUT.csv'
    #         self.UT_OUTcsv_noOutliers = self.PATH + f'MULTI-Ductile-disNodes-OUT-noOutliers.csv'

    #         self.UT_CSV_all_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allIN.csv'
    #         self.UT_CSV_all_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allOUT.csv'
    #         self.UT_CSV_train_in  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainIN.csv'
    #         self.UT_CSV_train_out = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainOUT.csv'
    #         self.UT_CSV_val_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valIN.csv'
    #         self.UT_CSV_val_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valOUT.csv'
    #         self.UT_CSV_test_in   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testIN.csv'
    #         self.UT_CSV_test_out  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testOUT.csv'
            
    #         self.UT_INcsv_f            = None
    #         self.UT_INcsv_f_noOutliers = None

    #         self.UT_CSV_all_in_f       = None
    #         self.UT_CSV_train_in_f     = None
    #         self.UT_CSV_val_in_f       = None
    #         self.UT_CSV_test_in_f      = None

    #         if self.freq:
    #             self.UT_INcsv_f            = self.PATH + f'Ductile-disNodes-INf.csv'
    #             self.UT_INcsv_f_noOutliers = self.PATH + f'MULTI-Ductile-disNodes-INf-noOutliers.csv'

    #             self.UT_CSV_all_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-allINf.csv'
    #             self.UT_CSV_train_in_f     = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-trainINf.csv'
    #             self.UT_CSV_val_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-valINf.csv'
    #             self.UT_CSV_test_in_f      = self.PATH + f'MLdata/multi/MULTI-{self.dis}-UT-testINf.csv'

    #         self.FT_INcsv             = self.PATH + f'Fracture-disNodes-IN.csv'
    #         self.FT_INcsv_noOutliers  = self.PATH + f'MULTI-Fracture-disNodes-IN-noOutliers.csv'
    #         self.FT_OUTcsv            = self.PATH + f'Fracture-disNodes-OUT.csv'
    #         self.FT_OUTcsv_noOutliers = self.PATH + f'MULTI-Fracture-disNodes-OUT-noOutliers.csv'

    #         self.FT_CSV_all_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allIN.csv'
    #         self.FT_CSV_all_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allOUT.csv'
    #         self.FT_CSV_train_in  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainIN.csv'
    #         self.FT_CSV_train_out = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainOUT.csv'
    #         self.FT_CSV_val_in    = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valIN.csv'
    #         self.FT_CSV_val_out   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valOUT.csv'
    #         self.FT_CSV_test_in   = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testIN.csv'
    #         self.FT_CSV_test_out  = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testOUT.csv'
            
    #         self.FT_INcsv_f            = None
    #         self.FT_INcsv_f_noOutliers = None

    #         self.FT_CSV_all_in_f       = None
    #         self.FT_CSV_train_in_f     = None
    #         self.FT_CSV_val_in_f       = None
    #         self.FT_CSV_test_in_f      = None

    #         if self.freq:
    #             self.FT_INcsv_f            = self.PATH + f'Fracture-disNodes-INf.csv'
    #             self.FT_INcsv_f_noOutliers = self.PATH + f'MULTI-Fracture-disNodes-INf-noOutliers.csv'

    #             self.FT_CSV_all_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-allINf.csv'
    #             self.FT_CSV_train_in_f     = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-trainINf.csv'
    #             self.FT_CSV_val_in_f       = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-valINf.csv'
    #             self.FT_CSV_test_in_f      = self.PATH + f'MLdata/multi/MULTI-{self.dis}-FT-testINf.csv

    # def load_DisDist_v1(self):
    #     self.train_in1 = self.perIN_df.to_numpy().reshape(len(self.perIN_df)//2, 2)

    #     train_out1 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
    #     self.dx_out1 = train_out1[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2,1)
    #     self.dy_out1 = train_out1[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2,1)
    
    # def load_DisDist_v2(self):
    #     self.train_in1 = self.perIN_df.to_numpy().reshape(len(self.perIN_df)//2, 2)
    #     self.train_in2 = np.array([self.train_in1.flatten()]*2)

    #     train_out2 = self.train_in.reshape(len(self.train_in),len(self.train_in[0])//2,2)
    #     dx_out2 = train_out2[:,:,0].reshape(len(self.train_in),len(self.train_in[0])//2)
    #     self.dx_out2 = np.stack((dx_out2, dx_out2), axis=1)
    #     dy_out2 = train_out2[:,:,1].reshape(len(self.train_in),len(self.train_in[0])//2)
    #     self.dy_out2 = np.stack((dy_out2, dy_out2), axis=1)



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
