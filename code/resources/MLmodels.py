import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


### Gaussian Process Regression model
class GPRmodel(GPR):
    def __init__(self, K, restarts, alpha, data=None):
        super(GPRmodel, self).__init__()
        self.K = self.set_K(K)
        self.res = restarts
        self.alpha = alpha
        
        self.GPR = self.build()
        if data:
            self.x = data[0]
            self.y = data[1]
            self.train()
    
    def set_K(self, K):
        return K
    
    def build(self):
        return GPR(kernel=self.K, n_restarts_optimizer=self.res, alpha=self.alpha)
    
    def train(self):
        self.GPR.fit(self.x, self.y)


### Multi-Layer Perceptron model
class MLPhlayer(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(MLPhlayer, self).__init__()
        self.norm = norm
        self.fc = nn.Linear(in_size, out_size)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
        elif norm == 'instance':
            self.normL = nn.InstanceNorm1d(out_size)
    
    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, norm=None):
        super(MLP, self).__init__()

        if len(h_size) > 0:
            self.fcIN = nn.Linear(in_size, h_size[0])
            self.hlayers = nn.ModuleList([
                MLPhlayer(i, j, norm) for i, j in zip(h_size[:-1], h_size[1:])])
        else:
            self.hlayers = None
            self.fcIN = None
            h_size = [in_size]
        
        self.fcOUT = nn.Linear(h_size[-1], out_size)
        self.act = nn.ReLU()

        self.norm = norm
        if norm == 'layer':
            self.normL = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normL = nn.BatchNorm1d(h_size[0])
        if norm == 'instance':
            self.normL = nn.InstanceNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        

    def forward(self, x):
        if self.fcIN:
            x = self.fcIN(x)
        if self.norm:
            x = self.normL(x)
        if self.fcIN:
            x = self.act(x)
        if self.hlayers:
            for layer in self.hlayers:
                x = layer(x)
        x = self.fcOUT(x)
        return x


### Graph Convolutional Network model
class GCNhlayer(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(GCNhlayer, self).__init__()
        self.norm = norm
        
        self.Gconv = GCNConv(in_size, out_size)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        if norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x, edge_index):
        x = self.Gconv(x, edge_index)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_size, h_size, out_size, norm=None, pool="mean"):
        super(GCN, self).__init__()
        self.norm = norm
        self.pool = pool
        
        self.GconvIN = GCNConv(in_size, h_size[0])
        # self.GconvOUT = GCNConv(h_size[-1], out_size)
        self.fcOUT = nn.Linear(h_size[-1], out_size)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        self.hlayers = nn.ModuleList([GCNhlayer(i, j, norm) for i, j in zip(h_size[:-1], h_size[1:])])

    def forward(self, x, edge_index, batch):
        x = self.GconvIN(x, edge_index)
        if self.norm: 
            x = self.normIN(x)
        x = self.act(x)
        for layer in self.hlayers:
            x = layer(x, edge_index)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        x = self.fcOUT(x)
        return x


### Graph Attetion Network model
class GAThlayer(nn.Module):
    def __init__(self, in_size, out_size, heads=1, norm=None):
        super(GAThlayer, self).__init__()
        self.norm = norm
        self.heads = heads
        
        self.GATconv = GATConv(in_size, out_size // heads, heads=heads)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x, edge_index):
        x = self.GATconv(x, edge_index)
        if self.normL:
            x = self.normL(x)
        x = self.act(x)
        return x

class GAT(nn.Module):
    def __init__(self, in_size, h_size, out_size, heads=1, norm=None, pool="mean"):
        super(GAT, self).__init__()
        self.norm = norm
        self.heads = heads
        self.pool = pool.lower()

        self.GATconvIN = GATConv(in_size, h_size[0] // heads, heads=heads)
        # self.GATconvOUT = GATConv(h_size[-1], out_size // heads, heads=heads)
        self.act = nn.ReLU()
        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0])
        elif norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        self.hlayers = nn.ModuleList([GAThlayer(i, j, heads=heads, norm=norm) for i, j in zip(h_size[:-1], h_size[1:])])
        self.fcOUT = nn.Linear(h_size[-1], out_size)

    def forward(self, x, edge_index, batch):
        x = self.GATconvIN(x, edge_index)
        if self.norm:
            x = self.normIN(x)
        x = self.act(x)
        for layer in self.hlayers:
            x = layer(x, edge_index)
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        x = self.fcOUT(x)
        return x

# TODO: Add Deep Ritz / ResNet models once finished.