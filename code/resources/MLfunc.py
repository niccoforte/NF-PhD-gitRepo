from resources.imports import *

import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx
import networkx as nx
import optuna
import json
import copy

from sklearn.model_selection import GridSearchCV


def absErr(A, B, typ=None, axis=None):
    if typ == "mean":
        return np.mean(np.abs(A - B), axis=axis)
    elif typ == "sum":
        return np.sum(np.abs(A - B), axis=axis)
    else:
        return np.abs(A - B)

def mse(A, B, typ=False, axis=None):
    if typ == "mean":
        return np.mean((A - B)**2, axis=axis)
    elif typ == "sum":
        return np.sum((A - B)**2, axis=axis)
    else:
        return (A - B)**2

def rmse(A, B, typ=False, axis=None):
    if typ == "mean":
        return np.sqrt(np.mean((A - B)**2, axis=axis))
    elif typ == "sum":
        return np.sqrt(np.sum((A - B)**2, axis=axis))
    else:
        return np.sqrt((A - B)**2)

### GAUSSIAN PROCESS FUNCTIONS

def plot_Kmatrix(gpr, x=None):
    if x is not None:
        K_pre = gpr.K(x)
        K_post = gpr.GPR.kernel_(x)
    else:
        K_pre = gpr.K(gpr.x)
        K_post = gpr.GPR.kernel_(gpr.x)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(K_pre, cmap='viridis', aspect='auto')
    ax1.set_title("Covariance Matrix Before Training")
    ax1.set_xlabel("Training Points")
    ax1.set_ylabel("Training Points")

    im2 = ax2.imshow(K_post, cmap='viridis', aspect='auto')
    ax2.set_title("Covariance Matrix After Training")
    ax2.set_xlabel("Training Points")
    ax2.set_ylabel("Training Points")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)

    plt.show()

def get_MeanStdCov(gpr, x, y):
    X_input = np.array([[x, y]])
    mean, std = gpr.GPR.predict(X_input, return_std=True)
    mean, cov = gpr.GPR.predict(X_input, return_cov=True)
    return mean[0][0], std[0], cov[0][0]

def Fsurface(gpr, density=50):
    x_values = np.linspace(-10, 130, density)
    y_values = np.linspace(-10, 170, density)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    means2, stds2, covs2 = [], [], []
    for X_row, Y_row in zip(X_grid, Y_grid):
        means1, stds1, covs1 = [], [], []
        for x, y in zip(X_row, Y_row):
            mean, std, cov = get_MeanStdCov(gpr, x, y)
            means1.append(mean)
            stds1.append(std)
            covs1.append(cov)
        means2.append(means1)
        stds2.append(stds1)
        covs2.append(covs1)
    means2, stds2, covs2 = np.array(means2), np.array(stds2), np.array(covs2)
    return x_values, y_values, means2, stds2, covs2

def plot_Fsurface(x_values, y_values, val, typ="3d"):
    X_grid, Y_grid = np.meshgrid(x_values, y_values)
    if val.shape != X_grid.shape:
        val.reshape(X_grid.shape)
    
    if typ.lower() == "3d" or typ.lower() == "both":
        fig1 = plt.figure(figsize=(10, 6))
        ax = fig1.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot_surface(X_grid, Y_grid, val, cmap="viridis")
        
    if typ.lower() == "cont" or typ.lower() == "both":
        fig2 = plt.figure(figsize=(10, 6))
        ax = fig2.add_subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        c = ax.contourf(X_grid, Y_grid, val, 50, cmap="viridis", vmin=np.min(val), vmax=np.max(val))
        fig2.colorbar(c)

    plt.show()


### NEURAL NETWORK FUNCTIONS

def train_model(
    typ, 
    model, 
    lossf, 
    n_epochs, 
    opt, 
    train_dataloader, 
    val_dataloader=None, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    scheduler=None, 
    earlyStop=None, 
    verbose=10, 
    optTrial=None, 
    RMSEtarget=False
):
    train_lossLog, val_lossLog = [], []
    best_loss, best_mse, best_rmse, best_epoch, best_model_state = 1000, 1000, (False, 1000), 0, None
    for epoch in range(1, n_epochs+1):
        train_lossSum, train_metric_sse, train_n = 0, 0, 0
        model.train()
        for batch in train_dataloader:
            opt.zero_grad(set_to_none=True)
            if typ.lower() == "gnn":
                data_batch = batch.to(device)
                x, y = data_batch.x.float(), data_batch.y.float()
                edge_index = data_batch.edge_index
                batch_vec = data_batch.batch
                y_predict = model(x, edge_index, batch_vec)
                y = data_batch.y.view(data_batch.num_graphs, -1)
            else:
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                y_predict = model(x)
            opt_loss = lossf(y_predict, y)
            opt_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_lossSum += opt_loss.item()
            metric_mse_batch = torch.mean((y_predict - y) ** 2)
            train_metric_sse += metric_mse_batch.item()*y.numel()
            train_n += y.numel()
            if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,
                          torch.optim.lr_scheduler.CosineAnnealingLR)):
                scheduler.step()

        train_lossAvg = train_lossSum/len(train_dataloader)
        train_metric_mse = train_metric_sse/train_n
        train_metric_rmse = math.sqrt(train_metric_mse)
        train_lossLog.append(train_lossAvg)
        
        if val_dataloader:
            model.eval()
            val_lossSum, val_metric_sse, val_n = 0, 0, 0
            with torch.no_grad():
                for batch in val_dataloader:
                    if typ.lower() == "gnn":
                        data_batch = batch.to(device)
                        x = data_batch.x.float()
                        y = data_batch.y.float()
                        edge_index = data_batch.edge_index
                        batch_vec = data_batch.batch
                        y_predict = model(x, edge_index, batch_vec)
                        y = data_batch.y.view(data_batch.num_graphs, -1)

                    else:
                        x, y = batch[0].float().to(device), batch[1].float().to(device)
                        y_predict = model(x)
                    
                    val_opt_loss = lossf(y_predict, y)
                    val_lossSum += val_opt_loss.item()
                    val_metric_mse_batch = torch.mean((y_predict - y) ** 2)
                    val_metric_sse += val_metric_mse_batch.item()*y.numel()
                    val_n += y.numel()

                val_lossAvg = val_lossSum/len(val_dataloader)
                val_metric_mse = val_metric_sse/val_n
                val_metric_rmse = math.sqrt(val_metric_mse)
                val_lossLog.append(val_lossAvg)

        lossAvg = val_lossAvg if val_dataloader else train_lossAvg
        mse = val_metric_mse if val_dataloader else train_metric_mse
        rmse = val_metric_rmse if val_dataloader else train_metric_rmse

        if lossAvg < best_loss:
            best_loss = lossAvg
            best_mse = mse
            best_rmse = (False, rmse)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
        
        if optTrial:
            optTrial.report(lossAvg, epoch)
            if optTrial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if RMSEtarget:
            if RMSEtarget is True:
                target = 0.1*np.sqrt(1/12)
            else:
                target = float(RMSEtarget)
            if rmse <= target:
                best_rmse = (True, rmse)
                print(f"RMSE check passed at epoch {epoch} with RMSE: {rmse:.4f}")
                break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(lossAvg)
        _lr = opt.param_groups[0]['lr']

        if epoch % verbose == 0 or epoch == 1 or epoch == n_epochs:
            if val_dataloader:
                print(f" -> Epoch {epoch}/{n_epochs} || LOSS - train: {train_lossAvg:.6f}, val: {val_lossAvg:.6f} | MSE - train: {train_metric_mse:.6f}, val: {val_metric_mse:.6f} | RMSE - train: {train_metric_rmse:.6f}, val: {val_metric_rmse:.6f} | LR: {_lr:.2e}")
            else:
                print(f" -> Epoch {epoch}/{n_epochs} || LOSS - train: {train_lossAvg:.6f} | MSE - train: {train_metric_mse:.6f} | RMSE - train: {train_metric_rmse:.6f} | LR: {_lr:.2e}")
        
        if earlyStop:
            earlyStop(lossAvg)
            if earlyStop.early_stop:
                break
        
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"================ Training Complete ================\n Best Epoch: {best_epoch}, with LOSS: {best_loss:.6f}, MSE: {best_mse:.6f} and RMSE: {best_rmse[1]:.6f} ==================")
    return model, epoch, train_lossLog, val_lossLog, best_loss, best_mse, best_rmse, best_epoch

def predict_model(typ, model, test_dataloader):
    test_outputs = []
    truth = []
    with torch.no_grad():
        for batch in test_dataloader:
            if typ.lower() == "gnn":
                device = next(model.parameters()).device
                data_batch = batch.to(device)
                y_predict = model(data_batch.x.float(), data_batch.edge_index, data_batch.batch)
                y = data_batch.y.view(data_batch.num_graphs, -1)
            else:
                x, y = batch[:][0].float(), batch[:][1].float()
                y_predict = model(x)
            test_outputs.append(y_predict.detach().cpu().numpy())
            truth.append(y.detach().cpu().numpy())
    test_outputs = np.concatenate(test_outputs)
    truth = np.concatenate(truth)
    return test_outputs, truth

def weights_init(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

def _activation(act):
    if act == "relu":
        act_f = nn.ReLU()
    elif act == "sigmoid":
        act_f = nn.Sigmoid()
    elif act == "tanh":
        act_f = nn.Tanh()
    elif act == "leakyrelu":
        act_f = nn.LeakyReLU()
    elif act == "softplus":
        act_f = nn.Softplus()
    elif act == "elu":
        act_f = nn.ELU()
    elif act == "gelu":
        act_f = nn.GELU()
    elif act == "selu":
        act_f = nn.SELU()
    elif act == "swish":
        act_f = nn.SiLU()
    elif act == "hardtanh":
        act_f = nn.Hardtanh()
    elif act == "prelu":
        act_f = nn.PReLU()
    elif act == "rrelu":
        act_f = nn.RReLU()
    elif act == "softsign":
        act_f = nn.Softsign()
    elif act == "softshrink":
        act_f = nn.Softshrink()
    elif act == "hardswish":
        act_f = nn.Hardswish()
    elif act == "mish":
        act_f = nn.Mish()
    elif act == "tanhshrink":
        act_f = nn.Tanhshrink()
    elif act == "logsigmoid":
        act_f = nn.LogSigmoid()
    else:
        print("[WARNING] Activation function not recognized. Using ReLU as default.")
        act_f = nn.ReLU()
    return act_f

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.verbose:
                print(f" !!! Early stopping triggered after {self.patience} epochs without improvement !!!")
            self.early_stop = True  # Set flag for stopping


### HYPERPARAMETER OPTIMIZATION

def GPR_HPopt(data, gpr, params, cv=5, verb=0):
    grid_search = GridSearchCV(gpr, params, cv=cv, scoring=None, verbose=verb)
    return grid_search.fit(data[0], data[1])

def objective(trial, model):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_dim = trial.suggest_int('hidden_dim', 8, 64, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 8, 16])
    h_size = [hidden_dim] * n_layers

    model_instance = MODEL( # type: ignore
        typ="MLP",
        model="INSERT MODEL HERE",
        lossf=nn.MSELoss(),
        opt=("adam", 0),
        batch=batch_size,
        lr=lr,
        data=model.data,
        train_dataloader=model.train_dataloader,
        val_dataloader=None,
        test_dataloader=None,
        scheduler=("min", 0.7, 10, 1e-4, True),
        earlyStop=EarlyStopping(patience=10, verbose=False),
        w_init=weights_init,
        optTrial=trial
    )

    model_instance.train(n_epochs=50, verbose=50)
    return model_instance.best_loss

def hOpt(objective, n_trials=50, prnt=False, save=False, path="models/etc", name="sample"):
    pruner = optuna.pruners.MedianPruner()

    if save:
        os.makedirs(f"{path}/{name}/HPO", exist_ok=True)
        storage_name = f"sqlite:///{path}/{name}/HPO/full_study.db"
        study_name = name
        study = optuna.create_study(storage=storage_name,
                                    study_name=study_name,
                                    direction="minimize",
                                    pruner=pruner,
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", pruner=pruner)
    
    study.optimize(objective, n_trials=n_trials)

    if prnt:
        print("\n" + "="*50)
        print(" Optimization Finished. Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        print(f"\nBest trial \n Loss: {best_trial.value}")
        print("\n Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
    
    if save:
        best_params = best_trial.params
        with open(f"{path}/{name}/HPO/best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
    
    return study

def load_bestParams(path="models/etc", name="sample"):
    with open(f"{path}/{name}/HPO/best_params.json", "r") as f:
        best_params = json.load(f)

    return best_params


### Custom Loss Functions
# TODO: GPU integration
# TODO: Custom loss functions (e.g., quantile loss, physics-informed loss)

class CustomQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.5, 0.5, 0.5], zone_boundaries=(50,130), err_type="L2"):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.zone_boundaries = zone_boundaries
        self.err_type = err_type

    def _zone_slices(self, n_points):
        if self.zone_boundaries is None:
            b1 = n_points // 3
            b2 = (2 * n_points) // 3
        else:
            if len(self.zone_boundaries) != 2:
                raise ValueError("zone_boundaries must be a tuple/list of length 2.")
            b1, b2 = int(self.zone_boundaries[0]), int(self.zone_boundaries[1])
        if not (0 < b1 < b2 < n_points):
            raise ValueError(f"Invalid zone boundaries ({b1}, {b2}) for n_points={n_points}.")
        return (
            (slice(None), slice(0, b1)),
            (slice(None), slice(b1, b2)),
            (slice(None), slice(b2, n_points)),
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the forward pass of the squared quantile (expectile) loss function.
        Corrects a mathematical typo found in Eq C.5 of the reference paper.
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        zones = self._zone_slices(y_true.shape[-1])
        total_loss = y_true.new_tensor(0.0)

        for i, zone in enumerate(zones):
            # 1. Use Squared Error as intended by Eq C.5
            err = y_true[zone] - y_pred[zone]
            err_sq = (err) ** 2
            lmbda = self.quantiles[i].to(y_true.device, dtype=y_true.dtype)

            # 2. Fix the typo: Use (1 - lambda) for overestimations
            if self.err_type.lower() == "l1":
                zone_loss = torch.where(err < 0, (1 - lmbda) * torch.abs(err), lmbda * torch.abs(err))
            elif self.err_type.lower() == "l2":
                zone_loss = torch.where(err < 0, (1 - lmbda) * err_sq, lmbda * err_sq)
            total_loss = total_loss + zone_loss.sum()

        # Average out the loss over all points in the batch 
        # (Mathematically maps to the 1/(3n) normalization from the paper)
        return total_loss / y_true.numel()

class CustomQuantileLossMATLAB(nn.Module):
    """
    MATLAB-compatible forward loss matching qlmseRegressionLayerCustom.forwardLoss.
    Uses the same class style/signature as CustomQuantileLoss for pipeline compatibility.
    """
    def __init__(self, quantiles=[0.5, 0.5, 0.5], zone_boundaries=None, err_type="L2"):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.zone_boundaries = zone_boundaries
        self.err_type = err_type

    def _zone_slices(self, n_points):
        # MATLAB behavior: seg = R./3; vec1=1:round(seg); vec2=round(seg)+1:2*round(seg); vec3=...
        if self.zone_boundaries is None:
            b1 = int(round(n_points / 3.0))
            b2 = int(2 * round(n_points / 3.0))
        else:
            if len(self.zone_boundaries) != 2:
                raise ValueError("zone_boundaries must be a tuple/list of length 2.")
            b1, b2 = int(self.zone_boundaries[0]), int(self.zone_boundaries[1])
        if not (0 < b1 < b2 <= n_points):
            raise ValueError(f"Invalid zone boundaries ({b1}, {b2}) for n_points={n_points}.")
        return (
            (slice(None), slice(0, b1)),
            (slice(None), slice(b1, b2)),
            (slice(None), slice(b2, n_points)),
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        zones = self._zone_slices(y_true.shape[-1])
        total_loss = y_true.new_tensor(0.0)

        for i, zone in enumerate(zones):
            err_sq = (y_pred[zone] - y_true[zone]) ** 2
            lmbda = self.quantiles[i].to(y_true.device, dtype=y_true.dtype)
            # MATLAB: mean(sum(max(lambda*e, (lambda-1)*e)))
            zone_term = torch.maximum(lmbda * err_sq, (lmbda - 1.0) * err_sq)
            zone_loss = zone_term.sum(dim=1).mean()
            total_loss = total_loss + zone_loss

        # MATLAB: loss = (loss1 + loss2 + loss3) / 3
        return total_loss / 3.0

def physics_loss(target, output):
    return torch.abs(torch.sum(target) - torch.sum(output))


### Plotting Functions

def plot_loss(epoch, train, val=None):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(1, epoch, len(train)), train, label="Training", c="darkblue")
    if val:
        plt.plot(np.linspace(1, epoch, len(val)), val, label="Validation", c="orangered")
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training and Validation Loss Vs Epoch")
    plt.legend()
    # plt.grid()
    plt.show()

def plot_predictions(perOUT, test_outputs, truth=None, mode="ut", indx=0):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(perOUT[0], test_outputs[indx]+perOUT[1], s=5, label=f"Prediction-{indx}", c="orangered")
    if truth is not None:
        plt.scatter(perOUT[0], truth[indx]+perOUT[1], s=5, label=f"Truth-{indx}", c="darkgreen")
        plt.bar(perOUT[0], absErr(truth[indx], test_outputs[indx]), width=(max(perOUT[0])-min(perOUT[0]))/(len(perOUT[0])), alpha=0.25, label="Error", color="gray")
    if mode.lower() == "ut":
        plt.ylabel("Macroscopic Stress ($\sigma$) [MPa]")
        plt.xlabel("Macroscopic Strain ($\epsilon$)")
    elif mode.lower() == "ft":
        plt.ylabel("Load-line Force ($F$) [N]")
        plt.xlabel("Load-line Dispalcement ($d$)")
    plt.legend()
    # plt.grid()
    plt.show()

def plot_Distribution(train_in1, test_outputs, truth=None, typ="contour"):
    x_, y_ = train_in1[:,0], train_in1[:,1]
    if truth is not None:
        val_ = truth - test_outputs
        title = "Error Distribution"
    else:
        val_ = test_outputs
        title = "Output Distribution"

    if typ.lower() == "contour" or typ.lower() == "cont" or typ.lower() == "both":
        fig1 = plt.figure(figsize=(10, 6))
        ax = fig1.add_subplot()
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ctr = ax.tricontourf(x_, y_, val_, cmap="viridis")
        plt.colorbar(ctr)
    if typ.lower() == "3d" or typ.lower() == "both":
        fig2 = plt.figure(figsize=(10, 6))
        ax = fig2.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ctr = ax.plot_trisurf(x_, y_, val_, cmap="viridis")
        plt.colorbar(ctr) 
    plt.show()


### GNN Functions
# TODO: Fix

def visualize_graphNetwork(loader, pos=None, colors=None, layout="kk"):
    for batch in loader:
        dat = batch.get_example(0)
        break
    G = to_networkx(dat, to_undirected=True)
    
    plt.figure(figsize=(7,7))
    if pos is None:
        if layout.lower() == "kk":
            pos = nx.kamada_kawai_layout(G)
        elif layout.lower() == "spec":
            pos = nx.spectral_layout(G)
        elif layout.lower() == "spring":
            pos = nx.spring_layout(G, seed=1)
        elif layout.lower() == "planar":
            pos = nx.planar_layout(G)
        elif layout.lower() == "rand":
            pos = nx.random_layout(G)
    
    if colors is None:
        colors = dat.x[:, 0].detach().numpy()
    
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=colors, cmap='viridis')
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5)
    
    plt.show()

