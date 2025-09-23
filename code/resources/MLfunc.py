from resources.imports import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torch_geometric.utils import to_networkx
import networkx as nx
import optuna
import json

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

def train_model(typ, model, lossf, n_epochs, opt, train_dataloader, val_dataloader=None, scheduler=None, earlyStop=None, verbose=10, optTrial=None, RMSEtarget=False):
    train_lossLog = []
    val_lossLog = []
    best_loss, best_rmse, best_epoch, best_model_state = 1000, (False, 1000), 0, None
    for epoch in range(1, n_epochs+1):
        train_lossSum = 0
        train_rmse = 0
        model.train()
        for batch in train_dataloader:
            if typ.lower() == "gnn":
                x, y = batch.x.float(), batch.y.float()
                y_predict = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(batch.num_graphs, -1)
            else:
                x, y = batch[:][0].float(), batch[:][1].float()
                y_predict = model(x)
            loss = lossf(y_predict, y)
            train_lossLog.append(loss.item())
            train_lossSum += loss.item()
            train_rmse += np.sqrt(loss.item())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        train_lossAvg = train_lossSum/len(train_dataloader)
        train_rmseAvg = train_rmse/len(train_dataloader)
        
        if val_dataloader:
            model.eval()
            with torch.no_grad():
                val_lossSum = 0
                val_rmse = 0
                for batch in val_dataloader:
                    if typ.lower() == "gnn":
                        x, y = batch.x.float(), batch.y.float()
                        y_predict = model(batch.x, batch.edge_index, batch.batch)
                        y = batch.y.view(batch.num_graphs, -1)
                    else:
                        x, y = batch[:][0].float(), batch[:][1].float()
                        y_predict = model(x)
                    loss = lossf(y_predict, y)

                    val_lossLog.append(loss.item())
                    val_lossSum += loss.item()
                    val_rmse += np.sqrt(loss.item())

                val_lossAvg = val_lossSum/len(val_dataloader)
                val_rmseAvg = val_rmse/len(val_dataloader)

        lossAvg = val_lossAvg if val_dataloader else train_lossAvg
        rmse_lossAvg = val_rmseAvg if val_dataloader else train_rmseAvg

        if lossAvg < best_loss:
            best_loss = lossAvg
            best_rmse = (False, rmse_lossAvg)
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if optTrial:
            optTrial.report(lossAvg, epoch)
            if optTrial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if verbose:
            if epoch == 1 or epoch % int(verbose) == 0:
                print("Epoch:", epoch, "- Loss:", lossAvg, "- RMSE Loss:", rmse_lossAvg)
        
        if RMSEtarget:

            if RMSEtarget is True:
                target = 0.1*np.sqrt(1/12)
            else:
                target = float(RMSEtarget)
            if rmse_lossAvg <= target:
                best_rmse = (True, best_rmse)
                print(f"RMSE check passed at epoch {epoch} with RMSE: {rmse_lossAvg:.4f}")
                break

        if scheduler:
            scheduler.step(lossAvg)
        if earlyStop:
            earlyStop(lossAvg)
            if earlyStop.early_stop:
                break
        
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"Best Epoch: {best_epoch}, with loss {best_loss}")
    return model, epoch, train_lossLog, val_lossLog, best_loss, best_rmse, best_epoch

def predict_model(typ, model, test_dataloader):
    test_outputs = []
    truth = []
    with torch.no_grad():
        for batch in test_dataloader:
            if typ.lower() == "gnn":
                x, y = batch.x.float(), batch.y.float()
                y_predict = model(x, batch.edge_index, batch.batch)  # CHECK
                y = batch.y.view(batch.num_graphs, -1)
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
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

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
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
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

def QuantileLoss():
    pass

def custom_loss(target, output):
    return torch.mean((output - target)**2)

def physics_loss(target, output):
    return torch.abs(torch.sum(target) - torch.sum(output))


### Plotting Functions

def plot_loss(epoch, train, val=None):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(1, epoch, len(train)), train, label="Training")
    if val:
        plt.plot(np.linspace(1, epoch, len(val)), val, label="Validation")
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.yscale("log")
    plt.title("Training and Validation Loss Vs Epoch")
    plt.legend()
    plt.grid()
    plt.show()

def plot_StressStrainOUT(perOUT, test_outputs, truth=None, indx=0):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(perOUT[0], test_outputs[indx]+perOUT[1], s=5, label=f"Prediction-{indx}")
    if truth is not None:
        plt.scatter(perOUT[0], truth[indx]+perOUT[1], s=5, label=f"Truth-{indx}")
        plt.bar(perOUT[0], absErr(truth[indx], test_outputs[indx]), width=(max(perOUT[0])-min(perOUT[0]))/(len(perOUT[0])), alpha=0.25, label="Error")
    plt.ylabel("Stress ($\sigma$) [MPa]")
    plt.xlabel("Strain ($\epsilon$)")
    plt.legend()
    plt.grid()
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

