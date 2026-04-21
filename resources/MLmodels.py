from resources.imports import *

from resources.lattices import connectivity
from resources.MLfunc import train_model, predict_model, plot_loss, plot_predictions, absErr, _activation, visualize_graphNetwork
# from resources.MLdata import Dataset_

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as gDataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torchinfo import summary
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from torch.serialization import safe_globals
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.preprocessing import MinMaxScaler


class MODEL:
    def __init__(
        self, 
        typ, 
        model, 
        lossf, 
        opt, 
        batch, 
        lr, 
        data,
        mechMode="both",
        UT_dataloaders=None, 
        FT_dataloaders=None,
        scheduler=None, 
        earlyStop=None, 
        w_init=False, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optTrial=None
    ):
        self.typ = typ
        self.model = model
        if isinstance(lossf, (list, tuple)):
            if len(lossf) == 0:
                raise ValueError("lossf list/tuple must contain at least one loss function.")
            self.losses = list(lossf)
            if len(self.losses) == 1:
                self.lossf = self.losses[0]
            else:
                def _combined_loss(y_pred, y_true):
                    total = self.losses[0](y_pred, y_true)
                    for lf in self.losses[1:]:
                        total = total + lf(y_pred, y_true)
                    return total
                self.lossf = _combined_loss
        else:
            self.losses = [lossf]
            self.lossf = lossf
        
        self.batch = batch
        self.lr = lr
        self.data = data
        self.UT_dataloaders = UT_dataloaders
        self.FT_dataloaders = FT_dataloaders
        self.earlyStop = earlyStop
        if w_init:
            self.model.apply(w_init)
        self.device = device
        self.model.to(device)
        self.optTrial = optTrial

        if data.UTmechTest:
            self.UT_model = copy.deepcopy(model)
            self.UT_opt = torch.optim.Adam(self.UT_model.parameters(), lr=lr, weight_decay=opt[1])
            
            if typ.lower() == "gnn":
                self.UT_nodes = data.UT_IN_df[data.UT_IN_df.columns.intersection(data.UT_dIN_df.columns)].iloc[0].to_numpy().reshape(int(len(data.UT_dIN_df.iloc[0])/2), 2)
                self.UT_edges = connectivity(data.LAT, self.UT_nodes, data.geom)[:, 1:] - 1
                self.UT_edge_index = torch.tensor(self.UT_edges, dtype=torch.long).t().contiguous()

                self.UT_trainDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.UT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.UT_train_in, data.UT_train_out)]
                self.UT_valDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.UT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.UT_val_in, data.UT_val_out)]
                self.UT_testDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.UT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.UT_test_in, data.UT_test_out)]
            
                if UT_dataloaders is None:
                    self.UT_train_dataloader = gDataLoader(dataset=self.UT_trainDS, batch_size=self.batch, shuffle=True)
                    self.UT_val_dataloader = gDataLoader(dataset=self.UT_valDS, batch_size=self.batch, shuffle=False)
                    self.UT_test_dataloader = gDataLoader(dataset=self.UT_testDS, batch_size=self.batch, shuffle=False)
                else:
                    self.UT_train_dataloader = UT_dataloaders[0]
                    self.UT_val_dataloader = UT_dataloaders[1]
                    self.UT_test_dataloader = UT_dataloaders[2]
            else:
                self.UT_trainDS = self._make_tensor_dataset(data.UT_train_in, data.UT_train_out)
                self.UT_valDS = self._make_tensor_dataset(data.UT_val_in, data.UT_val_out)
                self.UT_testDS = self._make_tensor_dataset(data.UT_test_in, data.UT_test_out)
            
                if UT_dataloaders is None:
                    self.UT_train_dataloader = DataLoader(dataset=self.UT_trainDS, batch_size=self.batch, shuffle=True)
                    self.UT_val_dataloader = DataLoader(dataset=self.UT_valDS, batch_size=self.batch, shuffle=False)
                    self.UT_test_dataloader = DataLoader(dataset=self.UT_testDS, batch_size=self.batch, shuffle=False)
                else:
                    self.UT_train_dataloader = UT_dataloaders[0]
                    self.UT_val_dataloader = UT_dataloaders[1]
                    self.UT_test_dataloader = UT_dataloaders[2]

        if data.FTmechTest:
            self.FT_model = copy.deepcopy(model)
            self.FT_opt = torch.optim.Adam(self.FT_model.parameters(), lr=lr, weight_decay=opt[1])
            
            if typ.lower() == "gnn":
                self.FT_nodes = data.FT_IN_df[data.FT_IN_df.columns.intersection(data.FT_dIN_df.columns)].iloc[0].to_numpy().reshape(int(len(data.FT_dIN_df.iloc[0])/2), 2)
                self.FT_edges = connectivity(data.LAT, self.FT_nodes, data.geom)[:, 1:] - 1
                self.FT_edge_index = torch.tensor(self.FT_edges, dtype=torch.long).t().contiguous()

                self.FT_trainDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.FT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.FT_train_in, data.FT_train_out)]
                self.FT_valDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.FT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.FT_val_in, data.FT_val_out)]
                self.FT_testDS = [Data(x=torch.tensor(i, dtype=torch.float), edge_index=self.FT_edge_index, y=torch.tensor(j, dtype=torch.float)) for i,j in zip(data.FT_test_in, data.FT_test_out)]
            
                if FT_dataloaders is None:
                    self.FT_train_dataloader = gDataLoader(dataset=self.FT_trainDS, batch_size=self.batch, shuffle=True)
                    self.FT_val_dataloader = gDataLoader(dataset=self.FT_valDS, batch_size=self.batch, shuffle=False)
                    self.FT_test_dataloader = gDataLoader(dataset=self.FT_testDS, batch_size=self.batch, shuffle=False)
                else:
                    self.FT_train_dataloader = FT_dataloaders[0]
                    self.FT_val_dataloader = FT_dataloaders[1]
                    self.FT_test_dataloader = FT_dataloaders[2]
            else:
                if data.UTmechTest:
                    self.FT_trainDS = self._make_tensor_dataset(data.UT_train_in, data.FT_train_out)
                    self.FT_valDS = self._make_tensor_dataset(data.UT_val_in, data.FT_val_out)
                    self.FT_testDS = self._make_tensor_dataset(data.UT_test_in, data.FT_test_out)
                else:
                    self.FT_trainDS = self._make_tensor_dataset(data.FT_train_in, data.FT_train_out)
                    self.FT_valDS = self._make_tensor_dataset(data.FT_val_in, data.FT_val_out)
                    self.FT_testDS = self._make_tensor_dataset(data.FT_test_in, data.FT_test_out)

                if FT_dataloaders is None:
                    self.FT_train_dataloader = DataLoader(dataset=self.FT_trainDS, batch_size=self.batch, shuffle=True)
                    self.FT_val_dataloader = DataLoader(dataset=self.FT_valDS, batch_size=self.batch, shuffle=False)
                    self.FT_test_dataloader = DataLoader(dataset=self.FT_testDS, batch_size=self.batch, shuffle=False)
                else:
                    self.FT_train_dataloader = FT_dataloaders[0]
                    self.FT_val_dataloader = FT_dataloaders[1]
                    self.FT_test_dataloader = FT_dataloaders[2]
        
        if scheduler:
            if data.UTmechTest:
                self.UTscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.UT_opt, 
                                                                            mode=scheduler[0], 
                                                                            factor=scheduler[1],
                                                                            patience=scheduler[2], 
                                                                            threshold=scheduler[3],
                                                                            threshold_mode="abs")
            if data.FTmechTest:
                self.FTscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.FT_opt, 
                                                                            mode=scheduler[0], 
                                                                            factor=scheduler[1],
                                                                            patience=scheduler[2], 
                                                                            threshold=scheduler[3],
                                                                            threshold_mode="abs")
        else:
            self.UTscheduler = None
            self.FTscheduler = None
    
    def train(self, n_epochs, verbose=10, plot=False, RMSEtarget=False):
        if self.data.UTmechTest:
            self.UT_model, \
                self.UT_epoch, \
                self.UT_train_lossLog, \
                self.UT_val_lossLog, \
                self.UT_best_loss, \
                self.UT_best_mse, \
                self.UT_best_rmse, \
                self.UT_best_epoch = train_model(self.typ, 
                                                 self.UT_model, 
                                                 self.lossf, 
                                                 n_epochs, 
                                                 self.UT_opt, 
                                                 self.UT_train_dataloader, 
                                                 val_dataloader=self.UT_val_dataloader, 
                                                 device=self.device,
                                                 scheduler=self.UTscheduler, 
                                                 earlyStop=copy.deepcopy(self.earlyStop), 
                                                 verbose=verbose,
                                                 optTrial=self.optTrial,
                                                 RMSEtarget=RMSEtarget)
        
            if plot:
                plot_loss(self.UT_epoch, self.UT_train_lossLog, self.UT_val_lossLog)

        if self.data.FTmechTest:
            self.FT_model, \
                self.FT_epoch, \
                self.FT_train_lossLog, \
                self.FT_val_lossLog, \
                self.FT_best_loss, \
                self.FT_best_mse, \
                self.FT_best_rmse, \
                self.FT_best_epoch = train_model(self.typ, 
                                                 self.FT_model, 
                                                 self.lossf, 
                                                 n_epochs, 
                                                 self.FT_opt, 
                                                 self.FT_train_dataloader, 
                                                 val_dataloader=self.FT_val_dataloader, 
                                                 device=self.device,
                                                 scheduler=self.FTscheduler, 
                                                 earlyStop=copy.deepcopy(self.earlyStop), 
                                                 verbose=verbose,
                                                 optTrial=self.optTrial,
                                                 RMSEtarget=RMSEtarget)
        
            if plot:
                plot_loss(self.FT_epoch, self.FT_train_lossLog, self.FT_val_lossLog)
    
    def predict(self, test_dataloader=None, plot=False):
        if (
            self.data.UTmechTest
            and self.data.FTmechTest
            and test_dataloader is not None
            and not isinstance(test_dataloader, dict)
            and not (isinstance(test_dataloader, (list, tuple)) and len(test_dataloader) == 2)
        ):
            raise ValueError(
                "For UT+FT prediction, pass test_dataloader as {'ut': ..., 'ft': ...} or (ut_loader, ft_loader)."
            )

        if test_dataloader is None:
            UT_loader = self.UT_test_dataloader if self.data.UTmechTest else None
            FT_loader = self.FT_test_dataloader if self.data.FTmechTest else None
        elif isinstance(test_dataloader, dict):
            UT_loader = test_dataloader.get("ut", self.UT_test_dataloader if self.data.UTmechTest else None)
            FT_loader = test_dataloader.get("ft", self.FT_test_dataloader if self.data.FTmechTest else None)
        elif isinstance(test_dataloader, (list, tuple)) and len(test_dataloader) == 2:
            UT_loader, FT_loader = test_dataloader
        else:
            UT_loader = test_dataloader
            FT_loader = test_dataloader

        if self.data.UTmechTest:
            self.UT_test_outputs, self.UT_truth = predict_model(self.typ,
                                                                self.UT_model,
                                                                UT_loader)
            UT_reconstructor = getattr(self.data, "UT_OUTreconstructor", None)
            if callable(UT_reconstructor):
                self.UT_test_outputs = UT_reconstructor(self.UT_test_outputs)
                self.UT_truth = UT_reconstructor(self.UT_truth)

            self.UT_err = absErr(self.UT_test_outputs, self.UT_truth, typ="sum", axis=1)
            self.UT_best, self.UT_worst = self.UT_err.tolist().index(min(self.UT_err)), self.UT_err.tolist().index(max(self.UT_err))
            print(f"Best prediction: {self.UT_best}, Worst prediction: {self.UT_worst}")

            if plot:
                plot_predictions(self.data.UT_OUT_df, self.UT_test_outputs, truth=self.UT_truth, mode="ut", indx=self.UT_best)
                plot_predictions(self.data.UT_OUT_df, self.UT_test_outputs, truth=self.UT_truth, mode="ut", indx=self.UT_worst)
        
        if self.data.FTmechTest:
            self.FT_test_outputs, self.FT_truth = predict_model(self.typ,
                                                                self.FT_model,
                                                                FT_loader)
            FT_reconstructor = getattr(self.data, "FT_OUTreconstructor", None)
            if callable(FT_reconstructor):
                self.FT_test_outputs = FT_reconstructor(self.FT_test_outputs)
                self.FT_truth = FT_reconstructor(self.FT_truth)

            self.FT_err = absErr(self.FT_test_outputs, self.FT_truth, typ="sum", axis=1)
            self.FT_best, self.FT_worst = self.FT_err.tolist().index(min(self.FT_err)), self.FT_err.tolist().index(max(self.FT_err))
            print(f"Best prediction: {self.FT_best}, Worst prediction: {self.FT_worst}")

            if plot:
                plot_predictions(self.data.FT_OUT_df, self.FT_test_outputs, truth=self.FT_truth, mode="ft", indx=self.FT_best)
                plot_predictions(self.data.FT_OUT_df, self.FT_test_outputs, truth=self.FT_truth, mode="ft", indx=self.FT_worst)
        
    def summary(self):
        if self.typ.lower() == "gnn":
            if self.data.UTmechTest or (self.data.UTmechTest and self.data.FTmechTest):
                sample_batch = next(iter(self.UT_train_dataloader)).to(self.device)
                visualize_graphNetwork(self.UT_train_dataloader, pos=self.UT_nodes, colors=None, layout="kk")
                return summary(self.UT_model, input_data=(sample_batch.x, self.UT_edge_index, sample_batch.batch))
            elif self.data.FTmechTest:
                sample_batch = next(iter(self.FT_train_dataloader)).to(self.device)
                visualize_graphNetwork(self.FT_train_dataloader, pos=self.FT_nodes, colors=None, layout="kk")
                return summary(self.FT_model, input_data=(sample_batch.x, self.FT_edge_index, sample_batch.batch))
        else:
            if self.data.UTmechTest and hasattr(self, "UT_model"):
                model_for_summary = self.UT_model
            elif self.data.FTmechTest and hasattr(self, "FT_model"):
                model_for_summary = self.FT_model
            else:
                model_for_summary = self.model
            return summary(model_for_summary, input_size=(self.batch, model_for_summary.in_size))
    
    def save(self, path, name):
        payload = {"version": 2}
        if hasattr(self, "UT_model"):
            payload["UT_model_state_dict"] = self.UT_model.state_dict()
        if hasattr(self, "FT_model"):
            payload["FT_model_state_dict"] = self.FT_model.state_dict()
        if hasattr(self, "UT_model"):
            payload["model_state_dict"] = self.UT_model.state_dict()
        elif hasattr(self, "FT_model"):
            payload["model_state_dict"] = self.FT_model.state_dict()
        else:
            payload["model_state_dict"] = self.model.state_dict()
        torch.save(payload, f"{path}/{name}.mdl")

    def load(self, path, name):
        state = torch.load(f"{path}/{name}.mdl", map_location=self.device)

        if isinstance(state, dict) and (
            "UT_model_state_dict" in state
            or "FT_model_state_dict" in state
            or "model_state_dict" in state
        ):
            if "UT_model_state_dict" in state and hasattr(self, "UT_model"):
                self.UT_model.load_state_dict(state["UT_model_state_dict"])
            if "FT_model_state_dict" in state and hasattr(self, "FT_model"):
                self.FT_model.load_state_dict(state["FT_model_state_dict"])

            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            elif "UT_model_state_dict" in state:
                self.model.load_state_dict(state["UT_model_state_dict"])
            elif "FT_model_state_dict" in state:
                self.model.load_state_dict(state["FT_model_state_dict"])
        else:
            self.model.load_state_dict(state)
            if hasattr(self, "UT_model"):
                self.UT_model.load_state_dict(state)
            if hasattr(self, "FT_model"):
                self.FT_model.load_state_dict(state)

    @staticmethod
    def _make_tensor_dataset(x, y):
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        return TensorDataset(x_tensor, y_tensor)

    
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
    def __init__(
        self, 
        size, 
        act, 
        norm=None, 
        bias=True
    ):
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
    def __init__(
        self, 
        in_size, 
        out_size, 
        act, 
        norm=None,
        bias=True
    ):
        super(mlpBlock, self).__init__()
        self.norm = norm
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self._act = _activation(act)
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size)
    
    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.normL(x)
        x = self._act(x)
        return x

class MLP(nn.Module):
    def __init__(
        self, 
        in_size, 
        h_size, 
        out_size, 
        act="relu", 
        block="mlp", 
        norm=None, 
        dropout=0.0,
        bias=True
    ):
        super().__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        
        self._act = _activation(act)

        if len(h_size) > 0:
            self.fcIN = nn.Linear(in_size, h_size[0], bias=bias)
            if block.lower() == "mlp":
                self.hlayers = nn.ModuleList([
                     mlpBlock(i, j, act, norm, bias) for i, j in zip(h_size[:-1], h_size[1:])
                ])
            elif block.lower() == "res":
                self.hlayers = nn.ModuleList([
                    resBlock(i, act, norm, bias) for i in h_size])
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
        
        self.dropout = dropout
        if dropout > 0.0:
            self.dropoutL = nn.Dropout(dropout)

    def forward(self, x):
        if self.fcIN:
            x = self.fcIN(x)
            if self.norm:
                x = self.normL(x)
            x = self._act(x)
            if self.dropout > 0.0:
                x = self.dropoutL(x)
        if self.hlayers:
            for layer in self.hlayers[:-1]:
                x = layer(x)
                x = self.dropoutL(x) if self.dropout > 0.0 else x
            x = self.hlayers[-1](x)
        x = self.fcOUT(x)
        return x


### Transformer model
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Transformer(nn.Module):
    def __init__(
        self,
        in_size,
        h_size,
        out_size,
        d_model=128,
        n_heads=8,
        n_layers=3,
        ff_mult=4,
        token_size=2,
        act="gelu",
        block="mlp",
        norm="layer",
        dropout=0.1,
        bias=True,
        pool="mean",
        use_cls_token=True,
        pos_encoding="learned"
    ):
        super().__init__()

        if token_size < 1:
            raise ValueError("token_size must be >= 1.")
        if d_model < 4:
            raise ValueError("d_model must be >= 4.")

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.token_size = token_size
        self.pool = pool.lower()
        self.use_cls_token = use_cls_token
        self.pos_encoding = pos_encoding.lower()
        self.norm = norm
        self.dropout = dropout

        self.pad_size = (token_size - (in_size % token_size)) % token_size
        self.seq_len = (in_size + self.pad_size) // token_size
        self.total_len = self.seq_len + (1 if use_cls_token else 0)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.input_proj = nn.Linear(token_size, d_model, bias=bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        if self.pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.total_len, d_model))
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
            self.pos_encoder = None
        elif self.pos_encoding == "sinusoidal":
            self.pos_embed = None
            self.pos_encoder = SinusoidalPositionEncoding(self.total_len, d_model)
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'.")

        tr_act = act.lower() if act.lower() in ["relu", "gelu"] else "gelu"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation=tr_act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if norm == "layer":
            self.head_norm = nn.LayerNorm(d_model)
        elif norm == "batch":
            self.head_norm = nn.BatchNorm1d(d_model)
        else:
            self.head_norm = None

        self._act = _activation(act)
        self.dropoutL = nn.Dropout(dropout) if dropout > 0.0 else None

        if len(h_size) > 0:
            self.fcIN = nn.Linear(d_model, h_size[0], bias=bias)
            if block.lower() == "mlp":
                self.hlayers = nn.ModuleList([
                    mlpBlock(i, j, act, norm, bias=bias) for i, j in zip(h_size[:-1], h_size[1:])
                ])
            elif block.lower() == "res":
                if len(set(h_size)) > 1:
                    raise ValueError("For block='res', all entries in h_size must be the same.")
                self.hlayers = nn.ModuleList([
                    resBlock(i, act, norm, bias=bias) for i in h_size
                ])
            else:
                raise ValueError("block must be either 'mlp' or 'res'.")
            self.fcOUT = nn.Linear(h_size[-1], out_size, bias=bias)
        else:
            self.fcIN = None
            self.hlayers = None
            self.fcOUT = nn.Linear(d_model, out_size, bias=bias)

    def _pool_tokens(self, x):
        if self.pool == "cls":
            if not self.use_cls_token:
                raise ValueError("pool='cls' requires use_cls_token=True.")
            return x[:, 0, :]

        tokens = x[:, 1:, :] if self.use_cls_token else x
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "max":
            return tokens.max(dim=1).values
        raise ValueError("pool must be one of ['mean', 'max', 'cls'].")

    def _apply_head_norm(self, x):
        if self.head_norm is None:
            return x
        if isinstance(self.head_norm, nn.BatchNorm1d):
            return self.head_norm(x)
        return self.head_norm(x)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError("Transformer expects input shape [batch, features].")
        if x.size(-1) != self.in_size:
            raise ValueError(f"Expected input feature size {self.in_size}, got {x.size(-1)}.")

        if self.pad_size > 0:
            x = F.pad(x, (0, self.pad_size), mode="constant", value=0.0)
        x = x.view(x.size(0), self.seq_len, self.token_size)
        x = self.input_proj(x)

        if self.use_cls_token:
            cls_tok = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tok, x], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :x.size(1), :]
        else:
            x = self.pos_encoder(x)

        x = self.encoder(x)
        x = self._pool_tokens(x)
        x = self._apply_head_norm(x)

        if self.fcIN is not None:
            x = self.fcIN(x)
            x = self._act(x)
            if self.dropoutL is not None:
                x = self.dropoutL(x)
            if self.hlayers:
                for layer in self.hlayers[:-1]:
                    x = layer(x)
                    if self.dropoutL is not None:
                        x = self.dropoutL(x)
                x = self.hlayers[-1](x)

        x = self.fcOUT(x)
        return x


### Graph Neural Network models
class gcnBlock(nn.Module):
    def __init__(self, in_size, out_size, act, norm=None, bias=True):
        super(gcnBlock, self).__init__()
        self.norm = norm
        
        self.Gconv = GCNConv(in_size, out_size, bias=bias)
        self.act = _activation(act)
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

class gatBlock(nn.Module):
    def __init__(self, in_size, out_size, act, heads=1, norm=None, bias=True):
        super(gatBlock, self).__init__()
        self.norm = norm
        self.heads = heads
        self.act = _activation(act)
        
        self.GATconv = GATConv(in_size, out_size, heads=heads, concat=True, bias=bias)
        if norm == 'layer':
            self.normL = nn.LayerNorm(out_size * heads)
        elif norm == 'batch':
            self.normL = nn.BatchNorm1d(out_size * heads)
    
    def forward(self, x, edge_index):
        x = self.GATconv(x, edge_index)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class GNN(nn.Module):
    def __init__(
        self,
        in_size,
        h_size,
        out_size,
        act="relu",
        block="gcn",
        norm=None,
        dropout=0.0,
        bias=True,
        heads=1,
        pool="mean"
    ):
        super(GNN, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.block = block.lower()
        self.norm = norm
        self.dropout = dropout
        self.pool = pool.lower()
        self.heads = heads
        self.act = _activation(act)

        if len(h_size) == 0:
            raise ValueError("h_size must contain at least one hidden dimension for GNN.")
        if self.block not in ["gcn", "gat"]:
            raise ValueError("block must be either 'gcn' or 'gat'.")

        if self.block.lower() == "gcn":
            self.GconvIN = GCNConv(in_size, h_size[0], bias=bias)
            hidden_out_dim = h_size[-1]
            self.hlayers = nn.ModuleList([
                gcnBlock(i, j, act, norm, bias=bias) for i, j in zip(h_size[:-1], h_size[1:])
            ])
            norm_in_dim = h_size[0]
        elif self.block.lower():
            self.GconvIN = GATConv(in_size, h_size[0], heads=heads, concat=True, bias=bias)
            hidden_out_dim = h_size[-1] * heads
            self.hlayers = nn.ModuleList([
                gatBlock(i * heads, j, act, heads=heads, norm=norm, bias=bias)
                for i, j in zip(h_size[:-1], h_size[1:])
            ])
            norm_in_dim = h_size[0] * heads

        if norm == 'layer':
            self.normIN = nn.LayerNorm(norm_in_dim)
        elif norm == 'batch':
            self.normIN = nn.BatchNorm1d(norm_in_dim)

        self.dropoutL = nn.Dropout(dropout) if dropout > 0.0 else None
        self.fcOUT = nn.Linear(hidden_out_dim, out_size, bias=bias)

    def forward(self, x, edge_index, batch):
        x = self.GconvIN(x, edge_index)
        if self.norm:
            x = self.normIN(x)
        x = self.act(x)
        if self.dropoutL is not None:
            x = self.dropoutL(x)

        for layer in self.hlayers:
            x = layer(x, edge_index)
            if self.dropoutL is not None:
                x = self.dropoutL(x)

        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError("pool must be either 'mean' or 'add'.")

        x = self.fcOUT(x)
        return x


### Other Models
class Autoencoder(nn.Module):
    def __init__(self, in_size, latent_size, h_size=None, block="mlp"):
        super(Autoencoder, self).__init__()

        self.encoder = MLP(in_size, h_size, latent_size, block=block)
        self.decoder = MLP(latent_size, h_size, in_size, block=block)
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


### Optimization Algorithms
class BoTorchOptimizer:
    def __init__(
        self,
        fitness_fn,
        input_shape,
        DATApath,
        bounds=(-0.2, 0.2),
        X_initial=None,
        y_initial=None,
        resume=False,
        save_log=True,
        save_state=True,
        argv=None
    ):
        self.fitness_fn = fitness_fn
        self.input_shape = input_shape
        self.DATApath = DATApath
        self.dim = np.prod(self.input_shape)
        self.save_log = save_log
        self.save_state = save_state
        self.argv = argv

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        
        log_file = f"{DATApath}/Opt/BO_log.txt"
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.state_file = os.path.join(log_dir, 'BO_optimizer_state.pt')
            
        self.train_X = None
        self.train_Y = None
        self.scaler = None

        self.starting_iteration = 1

        if resume and os.path.exists(self.state_file):
            print(f"--- Resuming optimization from saved state: {self.state_file} ---")
            state = torch.load(self.state_file, weights_only=False)
            self.train_X = state['train_X'].to(self.device, dtype=self.dtype)
            self.train_Y = state['train_Y'].to(self.device, dtype=self.dtype)
            self.scaler = state['scaler']
            self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=self.device, dtype=self.dtype)
            
            for filename in os.listdir(log_dir):
                if filename.startswith("BO_sample") and filename.endswith(".txt"):
                    sample_num = int(filename.split(".")[0][9:])
                    if sample_num >= self.starting_iteration:
                        self.starting_iteration = sample_num + 1
            print(f"Found highest sample number {self.starting_iteration-1} in directory.")
            print(f"Next iteration will be numbered starting from {self.starting_iteration}.")
            
            if self.save_log:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base, ext = os.path.splitext(log_file)
                self.log_file = f"{base}_resume_{timestamp}{ext}"

            print(f"Starting new log file for this run at: {self.log_file}")
        elif X_initial is not None and y_initial is not None:
            print(f"--- Warm-starting optimizer with {len(y_initial)} initial points. ---")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            X_reshaped = X_initial.reshape(len(y_initial), -1)
            self.scaler.fit(X_reshaped)
            X_scaled = self.scaler.transform(X_reshaped)
            
            self.train_X = torch.tensor(X_scaled, device=self.device, dtype=self.dtype)
            self.train_Y = torch.tensor(y_initial, device=self.device, dtype=self.dtype).unsqueeze(-1)
            self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], device=self.device, dtype=self.dtype)

            if self.save_log:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                base, ext = os.path.splitext(log_file)
                self.log_file = f"{base}_initial_{timestamp}{ext}"
                print(f"Starting new log file for this run at: {self.log_file}")
        else:
            self.bounds = torch.tensor([[bounds[0]] * self.dim, [bounds[1]] * self.dim], device=self.device, dtype=self.dtype)

    def _save_state(self):
        if not self.save_state:
            return
        
        state = {
            'train_X': self.train_X,
            'train_Y': self.train_Y,
            'scaler': self.scaler
        }
        torch.save(state, self.state_file)
        print(f"-> Optimizer state saved to {self.state_file}")
    
    def _get_fitted_model(self):
        model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def _report_progress(self, iteration, history):
        if self.save_log:
            with open(self.log_file, 'a') as f:
                f.write(f"Iteration: {iteration:4} | Best Fitness Found: {max(history):10.4f}\n")
        
        plt.clf()
        plt.plot(history, marker='o', linestyle='-', markersize=4)
        plt.title(f'BoTorch Convergence (Iter: {iteration})')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Score Found')
        plt.grid(True)
        plt.tight_layout()
        plt.pause(0.01)

    def run(self, n_iterations=50, n_initial_points=10, beta=4.0):
        if self.train_X is None:
            print(f"--- Starting fresh run. Generating {n_initial_points} points using LHS. ---")
            sampler = scipy.stats.qmc.LatinHypercube(d=self.dim)
            X_init_np = sampler.random(n=n_initial_points)
            if self.scaler is None:
                low, high = self.bounds[0,0].item(), self.bounds[1,0].item()
                X_init_np = low + (high - low) * X_init_np
            
            y_init_np = np.array([self.fitness_fn(x.reshape(self.input_shape), 0) for x in X_init_np])
            self.train_X = torch.tensor(X_init_np, device=self.device, dtype=self.dtype)
            self.train_Y = torch.tensor(y_init_np, device=self.device, dtype=self.dtype).unsqueeze(-1)
        
        history = [self.train_Y.max().item()]
        self._report_progress(self.starting_iteration - 1, history)

        for i in range(self.starting_iteration, self.starting_iteration + n_iterations):
            print(f"\n--- Iteration {i}/{self.starting_iteration + n_iterations - 1} ---")

            print("1. Fitting the GPR model...")
            model = self._get_fitted_model()
            
            print("2. Redefining and optimizing the acquisition function...")
            acq_function = UpperConfidenceBound(model, beta=beta)
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=self.bounds,
                q=1,
                num_restarts=20,
                raw_samples=512
            )
            
            new_x_scaled = candidate.detach()
            
            print("3. Evaluating new candidate with fitness function...")
            physical_x_flat = new_x_scaled.cpu().numpy().reshape(1, -1)
            if self.scaler:
                physical_x_flat = self.scaler.inverse_transform(physical_x_flat)
            physical_x = physical_x_flat.reshape(self.input_shape)
            new_y_val = self.fitness_fn(physical_x, i, self.DATApath, self.argv)
            print(f"   -> Score found: {new_y_val:.4f}")
            
            new_y = torch.tensor([[new_y_val]], device=self.device, dtype=self.dtype)
            
            self.train_X = torch.cat([self.train_X, new_x_scaled])
            self.train_Y = torch.cat([self.train_Y, new_y])
            
            history.append(self.train_Y.max().item())
            self._report_progress(i, history)

            self._save_state()

        plt.ioff()
        print("\nOptimization finished.")
        plt.show()
        
        best_idx = self.train_Y.argmax()
        best_solution_scaled = self.train_X[best_idx]
        best_score = self.train_Y[best_idx].item()

        best_solution_physical = best_solution_scaled.reshape(1, -1)
        if self.scaler:
            best_solution_physical = self.scaler.inverse_transform(best_solution_physical)
        return best_solution_physical.reshape(self.input_shape), best_score, self.train_X, self.train_Y

