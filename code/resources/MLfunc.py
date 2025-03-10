import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.utils import to_networkx
import networkx as nx


### Machine Learning Functions

def train_model(typ, model, lossf, n_epochs, opt, train_dataloader, val_dataloader=None, scheduler=None, earlyStop=None, verbose=10):
    train_lossLog = []
    val_lossLog = []
    best_loss, best_epoch = 1000, 0
    for epoch in range(1, n_epochs+1):
        train_lossSum = 0
        for batch in train_dataloader:
            if typ.lower() == "gcn":
                x, y = batch.x.float(), batch.y.float()
                y_predict = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(batch.num_graphs, -1)
            else:
                x, y = batch[:][0].float(), batch[:][1].float()
                y_predict = model(x)
            loss = lossf(y_predict, y)
            train_lossSum += loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_lossLog.append(loss.item())
        train_lossAvg = train_lossSum/len(train_dataloader)
        
        if val_dataloader:
            with torch.no_grad():
                val_lossSum = 0
                for batch in val_dataloader:
                    if typ.lower() == "gcn":
                        x, y = batch.x.float(), batch.y.float()
                        y_predict = model(batch.x, batch.edge_index, batch.batch)
                        y = batch.y.view(batch.num_graphs, -1)
                    else:
                        x, y = batch[:][0].float(), batch[:][1].float()
                        y_predict = model(x)
                    loss = lossf(y_predict, y)

                    val_lossLog.append(loss.item())
                    val_lossSum += loss

                val_lossAvg = val_lossSum/len(val_dataloader)
            if scheduler:
                scheduler.step(val_lossAvg)
            if earlyStop:
                earlyStop(val_lossAvg)
                if earlyStop.early_stop:
                    break
        else:
            if scheduler:
                scheduler.step(train_lossAvg)
            if earlyStop:
                earlyStop(train_lossAvg)
                if earlyStop.early_stop:
                    break
        
        if torch.abs(loss) < best_loss:
            best_loss = torch.abs(loss).item()
            best_epoch = epoch
        
        if verbose:
            if epoch == 1 or epoch % int(verbose) == 0:
                print("Epoch:", epoch, "- Loss:", loss.item())
            
    print(f"Best Epoch: {best_epoch}, with loss {best_loss}")
    # torch.save(model.state_dict(), "new_best_deep_ritz1.mdl")
    return model, epoch, train_lossLog, val_lossLog

def predict_model(typ, model, test_dataloader):
    test_outputs = []
    with torch.no_grad():
        for batch in test_dataloader:
            if typ.lower() == "gcn":
                x, y = batch.x.float(), batch.y.float()
                y_predict = model(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(batch.num_graphs, -1)
            else:
                x, y = batch[:][0].float(), batch[:][1].float()
                y_predict = model(x)
            test_outputs.append(y_predict.detach().cpu().numpy())
    test_outputs = np.concatenate(test_outputs)
    return test_outputs

def weights_init(m, dist="xavier"):
    if isinstance(m, nn.Linear):
        if dist.lower() == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif dist.lower() == "kaiming":
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.0)

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


### Custom Loss Functions

def QuantileLoss():
    pass

def custom_loss(target, output):   ### TODO: Quantile loss
    return torch.mean((output - target)**2)

def physics_loss(target, output):
    return torch.abs(torch.sum(target) - torch.sum(output))


### Plotting Functions

def plot_loss(epoch, train, val):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(1, epoch, len(train)), train)
    plt.plot(np.linspace(1, epoch, len(val)), val)
    plt.xlabel("Training Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.yscale("log")
    plt.title("Training and Validation Loss Vs Epoch")
    plt.legend(["Training", "Validation"])
    plt.grid()
    plt.show()

def plot_StressStrainOUT(perOUT, train_out, test_outputs):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(perOUT[0], train_out[10]+perOUT[1], s=5, label="Truth")
    plt.scatter(perOUT[0], test_outputs[10]+perOUT[1], s=5, label="Prediction")
    plt.ylabel("Stress ($\sigma$) [MPa]")
    plt.xlabel("Strain ($\epsilon$)")
    plt.legend()
    plt.grid()
    plt.show()

# TODO: ERROR PLOTTING 

def plot_Distribution(train_in1, test_outputs, dx_out1=None, typ="contour"):
    x_, y_ = train_in1[:,0], train_in1[:,1]
    if dx_out1 is not None:
        val_ = dx_out1 - test_outputs
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

def visualize_graphNetwork(loader, colors=None, layout="kk"):
    for batch in loader:
        dat = batch.get_example(0)
        break
    G = to_networkx(dat, to_undirected=True)
    
    plt.figure(figsize=(7,7))
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

