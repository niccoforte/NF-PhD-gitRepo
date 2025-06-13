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
class resBlock(nn.Module):
    def __init__(self, size, act, norm=None, bias=True):
        super(resBlock, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(size, size*2, bias=bias)
        self.fc2 = nn.Linear(size*2, size, bias=bias)
        self.act = act
        if norm == 'layer':
            self.normL1 = nn.LayerNorm(size*2)
            self.normL2 = nn.LayerNorm(size)
        elif norm == 'batch':
            self.normL1 = nn.BatchNorm1d(size*2)
            self.normL2 = nn.BatchNorm1d(size)

    def forward(self, x):
        skip = x
        x = self.fc1(x)
        if self.norm:
            x = self.normL1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.norm:
            x = self.normL2(x)
        x = x + skip
        x = self.act(x)
        return x #+ skip

class mlpBlock(nn.Module):
    def __init__(self, in_size, out_size, act, norm=None, bias=True):
        super(mlpBlock, self).__init__()
        self.norm = norm
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.act = act
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size, block="mlp", norm=None, bias=True):
        super(MLP, self).__init__()
        
        self.act = nn.Sigmoid()

        if len(h_size) > 0:
            self.fcIN = nn.Linear(in_size, h_size[0], bias=bias)
            if block == "mlp":
                self.hlayers = nn.ModuleList([
                    mlpBlock(i, j, self.act, norm, bias) for i, j in zip(h_size[:-1], h_size[1:])])
            elif block == "res":
                self.hlayers = nn.ModuleList([
                    resBlock(i, self.act, norm, bias) for i in h_size])
        else:
            self.hlayers = None
            self.fcIN = None
            h_size = [in_size]
        
        self.fcOUT = nn.Linear(h_size[-1], out_size, bias=bias)

        self.norm = norm
        if norm == 'layer':
            self.normL = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normL = nn.BatchNorm1d(h_size[0])
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
    def __init__(self, in_size, out_size, act, norm=None):
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
        self.act = nn.Sigmoid()
        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0])
        if norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        self.hlayers = nn.ModuleList([GCNhlayer(i, j, self.act, norm) for i, j in zip(h_size[:-1], h_size[1:])])

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
    def __init__(self, in_size, out_size, act, heads=1, norm=None):
        super(GAThlayer, self).__init__()
        self.norm = norm
        self.heads = heads
        
        self.GATconv = GATConv(in_size, out_size // heads, heads=heads)
        self.act = act
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
        self.act = nn.Sigmoid()
        if norm == 'layer':
            self.normIN = nn.LayerNorm(h_size[0])
        elif norm == 'batch':
            self.normIN = nn.BatchNorm1d(h_size[0])
        self.dropout = nn.Dropout(0.25)
        self.hlayers = nn.ModuleList([GAThlayer(i, j, self.act, heads=heads, norm=norm) for i, j in zip(h_size[:-1], h_size[1:])])
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

class Autoencoder(nn.Module):
    def __init__(self, in_size, latent_size, h_size=None):
        super(Autoencoder, self).__init__()

        if h_size is None:
            h_size = in_size // 2

        self.encoder = MLP(in_size, h_size, latent_size, block="mlp")
        self.decoder = MLP(latent_size, h_size, in_size, block="mlp")
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

# TODO: Add macro model class with all hyperparameters and model selection.