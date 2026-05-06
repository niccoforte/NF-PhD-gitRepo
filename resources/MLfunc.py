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
        val = val.reshape(X_grid.shape)
    
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
# Training function & helpers
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
    best_loss, best_mse, best_rmse, best_epoch, best_model_state = float("inf"), float("inf"), (False, float("inf")), 0, None
    for epoch in range(1, n_epochs+1):
        train_stats = _new_epoch_stats()
        model.train()
        for batch in train_dataloader:
            opt.zero_grad(set_to_none=True)
            y_predict, y = _forward_batch(typ, model, batch, device, context="training")
            opt_loss = lossf(y_predict, y)
            opt_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            _update_epoch_stats(train_stats, opt_loss, y_predict, y)
            if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR,
                          torch.optim.lr_scheduler.CosineAnnealingLR)):
                scheduler.step()

        train_lossAvg, train_metric_mse, train_metric_rmse, train_target_range = _finalize_epoch_stats(train_stats)
        train_lossLog.append(train_lossAvg)
        
        if val_dataloader:
            val_lossAvg, val_metric_mse, val_metric_rmse, val_target_range = _evaluate_model(
                typ, model, lossf, val_dataloader, device, context="validation"
            )
            val_lossLog.append(val_lossAvg)

        lossAvg = val_lossAvg if val_dataloader else train_lossAvg
        mse = val_metric_mse if val_dataloader else train_metric_mse
        rmse = val_metric_rmse if val_dataloader else train_metric_rmse
        target_range = val_target_range if val_dataloader else train_target_range

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
            target = _rmse_target_value(RMSEtarget, target_range)
            if rmse <= target:
                best_rmse = (True, rmse)
                print(f"RMSE check passed at epoch {epoch} with RMSE: {rmse:.4f}")
                break

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(lossAvg)
        _lr = opt.param_groups[0]['lr']

        if verbose:
            _verbose_every = max(1, int(verbose))
            _should_log = (epoch % _verbose_every == 0) or (epoch == 1) or (epoch == n_epochs)
        else:
            _should_log = (epoch == 1) or (epoch == n_epochs)

        if _should_log:
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

def _validate_gnn_batch(data_batch, y_predict=None, y=None, context="GNN batch"):
    if not hasattr(data_batch, "x") or data_batch.x is None:
        raise ValueError(f"{context}: missing node feature tensor 'x'.")
    if data_batch.x.ndim != 2:
        raise ValueError(f"{context}: expected x shape [num_nodes, node_features], got {tuple(data_batch.x.shape)}.")

    if not hasattr(data_batch, "edge_index") or data_batch.edge_index is None:
        raise ValueError(f"{context}: missing 'edge_index'.")
    if data_batch.edge_index.ndim != 2 or data_batch.edge_index.shape[0] != 2:
        raise ValueError(f"{context}: expected edge_index shape [2, num_edges], got {tuple(data_batch.edge_index.shape)}.")
    if data_batch.edge_index.dtype != torch.long:
        raise TypeError(f"{context}: edge_index must have dtype torch.long, got {data_batch.edge_index.dtype}.")

    num_nodes = data_batch.x.shape[0]
    if data_batch.edge_index.numel() > 0:
        edge_min = int(data_batch.edge_index.min().item())
        edge_max = int(data_batch.edge_index.max().item())
        if edge_min < 0 or edge_max >= num_nodes:
            raise ValueError(
                f"{context}: edge_index contains node ids outside [0, {num_nodes - 1}] "
                f"(min={edge_min}, max={edge_max})."
            )

    if not hasattr(data_batch, "batch") or data_batch.batch is None:
        raise ValueError(f"{context}: missing PyG batch vector.")
    if data_batch.batch.ndim != 1:
        raise ValueError(f"{context}: expected batch vector shape [num_nodes], got {tuple(data_batch.batch.shape)}.")
    if data_batch.batch.numel() != num_nodes:
        raise ValueError(f"{context}: batch vector length {data_batch.batch.numel()} does not match num_nodes {num_nodes}.")

    num_graphs = int(data_batch.num_graphs)
    if num_graphs <= 0:
        raise ValueError(f"{context}: num_graphs must be positive, got {num_graphs}.")
    if data_batch.batch.numel() > 0:
        graph_max = int(data_batch.batch.max().item())
        if graph_max >= num_graphs:
            raise ValueError(f"{context}: batch vector references graph {graph_max}, but num_graphs={num_graphs}.")

    y_tensor = y if y is not None else getattr(data_batch, "y", None)
    if y_tensor is None:
        raise ValueError(f"{context}: missing target tensor 'y'.")
    if y_tensor.numel() % num_graphs != 0:
        raise ValueError(
            f"{context}: target values ({y_tensor.numel()}) cannot be reshaped across "
            f"{num_graphs} graphs."
        )

    if y_predict is not None:
        expected_y = y_tensor.view(num_graphs, -1)
        if tuple(y_predict.shape) != tuple(expected_y.shape):
            raise ValueError(
                f"{context}: prediction shape {tuple(y_predict.shape)} does not match "
                f"target shape {tuple(expected_y.shape)}."
            )

def _forward_batch(typ, model, batch, device, context):
    if str(typ).lower() in ["gnn", "gcn", "gat"]:
        data_batch = batch.to(device)
        _validate_gnn_batch(data_batch, context=context)
        y_predict = model(data_batch.x.float(), data_batch.edge_index, data_batch.batch)
        y = data_batch.y.float().view(data_batch.num_graphs, -1)
        _validate_gnn_batch(data_batch, y_predict=y_predict, y=y, context=context)
        return y_predict, y

    x, y = batch[0].float().to(device), batch[1].float().to(device)
    return model(x), y

def _new_epoch_stats():
    return {
        "loss_sum": 0.0,
        "loss_weight": 0,
        "metric_sse": 0.0,
        "n": 0,
        "target_min": float("inf"),
        "target_max": -float("inf"),
    }

def _update_epoch_stats(stats, opt_loss, y_predict, y):
    batch_weight = y.shape[0] if y.ndim > 0 else 1
    stats["loss_sum"] += opt_loss.item()*batch_weight
    stats["loss_weight"] += batch_weight
    stats["target_min"] = min(stats["target_min"], float(y.min().item()))
    stats["target_max"] = max(stats["target_max"], float(y.max().item()))

    metric_mse_batch = torch.mean((y_predict.detach() - y) ** 2)
    stats["metric_sse"] += metric_mse_batch.item()*y.numel()
    stats["n"] += y.numel()

def _finalize_epoch_stats(stats):
    if stats["loss_weight"] == 0 or stats["n"] == 0:
        raise ValueError("Cannot finalize epoch statistics for an empty dataloader.")
    loss_avg = stats["loss_sum"]/stats["loss_weight"]
    mse = stats["metric_sse"]/stats["n"]
    rmse = math.sqrt(mse)
    target_range = stats["target_max"] - stats["target_min"]
    return loss_avg, mse, rmse, target_range

def _evaluate_model(typ, model, lossf, dataloader, device, context="validation"):
    stats = _new_epoch_stats()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            y_predict, y = _forward_batch(typ, model, batch, device, context=context)
            opt_loss = lossf(y_predict, y)
            _update_epoch_stats(stats, opt_loss, y_predict, y)
    return _finalize_epoch_stats(stats)

def _rmse_target_value(RMSEtarget, target_range):
    if RMSEtarget is True:
        return 0.1*target_range*np.sqrt(1/12)
    return float(RMSEtarget)

# Predict function
def predict_model(typ, model, test_dataloader):
    test_outputs = []
    truth = []
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            y_predict, y = _forward_batch(typ, model, batch, device, context="prediction")
            test_outputs.append(y_predict.detach().cpu().numpy())
            truth.append(y.detach().cpu().numpy())
    if was_training:
        model.train()
    test_outputs = np.concatenate(test_outputs)
    truth = np.concatenate(truth)
    return test_outputs, truth

# Weights initialization
def make_weights_init(act="relu", bias_value=0.0, distribution="normal"):
    act_name = _activation(act, return_name=True)
    distribution = distribution.lower()

    if distribution not in ["normal", "uniform"]:
        raise ValueError("distribution must be either 'normal' or 'uniform'.")

    def _init(m):
        if not isinstance(m, nn.Linear):
            return

        if act_name == "relu":
            init_fn = nn.init.kaiming_normal_ if distribution == "normal" else nn.init.kaiming_uniform_
            init_fn(m.weight, nonlinearity="relu")
        elif act_name in ["leakyrelu", "prelu", "rrelu"]:
            negative_slope = getattr(act, "negative_slope", 0.01)
            init_fn = nn.init.kaiming_normal_ if distribution == "normal" else nn.init.kaiming_uniform_
            init_fn(m.weight, a=float(negative_slope), nonlinearity="leaky_relu")
        else:
            gain_name = act_name if act_name in ["linear", "sigmoid", "tanh", "selu"] else "linear"
            gain = nn.init.calculate_gain(gain_name)
            init_fn = nn.init.xavier_normal_ if distribution == "normal" else nn.init.xavier_uniform_
            init_fn(m.weight, gain=gain)

        if m.bias is not None:
            nn.init.constant_(m.bias, bias_value)

    _init.__name__ = f"weights_init_{act_name}_{distribution}"
    _init.init_config = {
        "act": act_name,
        "bias_value": bias_value,
        "distribution": distribution,
    }
    return _init

def _infer_weight_init_activation(model):
    for attr in ["_act", "act"]:
        activation = getattr(model, attr, None)
        if isinstance(activation, nn.Module):
            return activation
        if isinstance(activation, str):
            return activation

    for module in model.modules():
        if isinstance(module, tuple(_activation(return_types=True).values())):
            return module
    return "linear"

def resolve_weight_init(w_init, model):
    if w_init is None:
        return None
    if isinstance(w_init, str):
        if w_init.lower() == "auto":
            return make_weights_init(act=_infer_weight_init_activation(model), bias_value=0.0)
        raise ValueError("w_init must be None, 'auto', or a callable such as make_weights_init(...).")
    if callable(w_init):
        return w_init
    raise TypeError("w_init must be None, 'auto', or a callable such as make_weights_init(...).")

# EasyStopping algorithm
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

# Other helper functions
def _activation(act="relu", return_name=False, return_types=False):
    activation_types = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
        "softplus": nn.Softplus,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "selu": nn.SELU,
        "swish": nn.SiLU,
        "hardtanh": nn.Hardtanh,
        "prelu": nn.PReLU,
        "rrelu": nn.RReLU,
        "softsign": nn.Softsign,
        "softshrink": nn.Softshrink,
        "hardswish": nn.Hardswish,
        "mish": nn.Mish,
        "tanhshrink": nn.Tanhshrink,
        "logsigmoid": nn.LogSigmoid,
    }

    if return_types:
        return activation_types

    if act is None:
        act_name = "linear"
    elif isinstance(act, str):
        act_name = act.strip().lower()
    else:
        act_name = None
        for name, activation_type in activation_types.items():
            if isinstance(act, activation_type):
                act_name = name
                break
        if act_name is None:
            act_name = act.__class__.__name__.lower()

    if return_name:
        return act_name

    if act_name in activation_types:
        act_f = activation_types[act_name]()
    else:
        print("[WARNING] Activation function not recognized. Using ReLU as default.")
        act_f = nn.ReLU()
    return act_f


### Custom Loss Functions
class CombinedCurveLoss(nn.Module):
    """
    Convenience wrapper for normalized curve-level losses. Weights with value 0 disable a term.
    """
    def __init__(
        self,
        mse_weight=0.1,
        weighted_mse_weight=1.0,
        derivative_weight=0.5,
        peak_weight=0.2,
        energy_weight=0.2,
        peak_location_weight=0.05,
        zone_boundaries=(67, 134),
        zone_weights=(1.0, 5.0, 2.0),
        x_values=None,
        reduction="mean",
        derivative_order=1,
        normalization_eps=1e-8,
        SoftPeak_beta=20.0,
    ):
        super().__init__()
        reduction = reduction.lower()
        if reduction not in ["mean", "sum"]:
            raise ValueError("CombinedCurveLoss reduction must be 'mean' or 'sum'.")

        self.mse_weight = float(mse_weight)
        self.weighted_mse_weight = float(weighted_mse_weight)
        self.derivative_weight = float(derivative_weight)
        self.peak_weight = float(peak_weight)
        self.energy_weight = float(energy_weight)
        self.peak_location_weight = float(peak_location_weight)
        self.reduction = reduction
        self.derivative_order = int(derivative_order)
        self.normalization_eps = float(normalization_eps)
        self.SoftPeak_beta = float(SoftPeak_beta)

        self.weighted_mse = WeightedCurveMSELoss(
            zone_boundaries=zone_boundaries,
            zone_weights=zone_weights,
            reduction=reduction,
            normalize_by_target_range=True,
            eps=normalization_eps,
        )
        self.derivative = CurveDerivativeLoss(
            x_values=x_values,
            order=derivative_order,
            reduction=reduction,
            eps=normalization_eps,
        )
        self.peak = PeakStressLoss(reduction=reduction, normalize=True, eps=normalization_eps)
        self.energy = StrainEnergyLoss(
            x_values=x_values,
            normalize=True,
            reduction=reduction,
            eps=normalization_eps,
        )
        self.peak_location = SoftPeakLocationLoss(
            x_values=x_values,
            beta=SoftPeak_beta,
            reduction=reduction,
            normalize=True,
            eps=normalization_eps,
        )

    def _curve_value_scale(self, y_true):
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(0)
        sample_scale = y_true.max(dim=1).values - y_true.min(dim=1).values
        fallback = torch.maximum(sample_scale.max(), y_true.abs().max())
        fallback = torch.where(
            fallback > self.normalization_eps,
            fallback,
            torch.ones_like(fallback),
        )
        sample_scale = torch.where(sample_scale > self.normalization_eps, sample_scale, fallback)
        return sample_scale.unsqueeze(1)

    def _mse_component(self, y_pred, y_true):
        err_sq = (y_pred - y_true) ** 2 / (self._curve_value_scale(y_true) ** 2)
        if self.reduction == "sum":
            return err_sq.sum()
        return err_sq.mean()

    def component_losses(self, y_pred: torch.Tensor, y_true: torch.Tensor, weighted=False):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
        components = {}
        if self.mse_weight:
            components["mse"] = self._mse_component(y_pred, y_true)
        if self.weighted_mse_weight:
            components["weighted_mse"] = self.weighted_mse(y_pred, y_true)
        if self.derivative_weight:
            components["derivative"] = self.derivative(y_pred, y_true)
        if self.peak_weight:
            components["peak"] = self.peak(y_pred, y_true)
        if self.energy_weight:
            components["energy"] = self.energy(y_pred, y_true)
        if self.peak_location_weight:
            components["peak_location"] = self.peak_location(y_pred, y_true)

        if not weighted:
            return components
        weights = {
            "mse": self.mse_weight,
            "weighted_mse": self.weighted_mse_weight,
            "derivative": self.derivative_weight,
            "peak": self.peak_weight,
            "energy": self.energy_weight,
            "peak_location": self.peak_location_weight,
        }
        return {name: weights[name] * value for name, value in components.items()}

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        total = y_true.new_tensor(0.0)
        for component in self.component_losses(y_pred, y_true, weighted=True).values():
            total = total + component
        return total


class QuantileLoss(nn.Module):
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

class QuantileLossMATLAB(nn.Module):
    """
    MATLAB-compatible forward loss matching qlmseRegressionLayerCustom.forwardLoss.
    Uses the same class style/signature as CustomQuantileLoss for pipeline compatibility.
    """
    def __init__(self, quantiles=[0.5, 0.5, 0.5], zone_boundaries=None, err_type="L2"):
        super().__init__()
        # Backward compatibility: some legacy notebook cells passed a base loss
        # module as first positional arg (e.g., nn.MSELoss(...)).
        if isinstance(quantiles, nn.Module):
            print("[WARNING] CustomQuantileLossMATLAB received a loss module as first argument; defaulting quantiles to [0.5, 0.5, 0.5].")
            quantiles = [0.5, 0.5, 0.5]

        if isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.detach().cpu().tolist()
        elif not isinstance(quantiles, (list, tuple)):
            if hasattr(quantiles, "tolist"):
                quantiles = quantiles.tolist()
            else:
                raise TypeError("quantiles must be a list/tuple/tensor-like object of numeric values.")

        if len(quantiles) < 3:
            raise ValueError("quantiles must contain at least 3 values (one for each output zone).")

        self.register_buffer("quantiles", torch.tensor(list(quantiles)[:3], dtype=torch.float32))
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

class PhysicalRegularizationLoss(nn.Module):
    """
    Penalizes high point-to-point oscillations in predicted output curves.
    - Oscillation penalty starts once |y_j - y_{j-1}| exceeds delta_tolerance.
    - Penalty weight decays after epsilon_tolerance as epsilon increases.
    """
    def __init__(
        self,
        delta_tolerance=0.0,
        epsilon_tolerance=1.0,
        epsilon_decay=2.0,
        oscillation_power=2.0,
        reduction="mean",
    ):
        super().__init__()
        self.delta_tolerance = float(delta_tolerance)
        self.epsilon_tolerance = float(epsilon_tolerance)
        self.epsilon_decay = float(epsilon_decay)
        self.oscillation_power = float(oscillation_power)
        self.reduction = reduction.lower()

    def _epsilon_weights(self, n_diffs, y_pred):
        # Differences are defined between consecutive points; use mid-point epsilon.
        eps = torch.linspace(0.0, 1.0, steps=n_diffs + 1, device=y_pred.device, dtype=y_pred.dtype)
        eps_mid = 0.5 * (eps[1:] + eps[:-1])
        weights = torch.where(
            eps_mid <= self.epsilon_tolerance,
            torch.ones_like(eps_mid),
            torch.exp(-self.epsilon_decay * (eps_mid - self.epsilon_tolerance)),
        )
        return weights

    def forward(self, y_pred, y_true):
        self.delta_tolerance = (y_true.max() - y_true.min()) / 100

        diffs = torch.abs(y_pred[:, 1:] - y_pred[:, :-1])
        excess = torch.relu(diffs - self.delta_tolerance)
        if self.oscillation_power != 1.0:
            excess = excess ** self.oscillation_power
        weights = self._epsilon_weights(diffs.shape[1], y_pred).unsqueeze(0)
        penalty = excess * weights

        if self.reduction == "sum":
            return penalty.sum()
        if self.reduction == "none":
            return penalty
        return penalty.mean()

class WeightedCurveMSELoss(nn.Module):
    """
    MSE with pointwise or zone-based weights along the output curve.
    """
    def __init__(
        self,
        weights=None,
        zone_boundaries=None,
        zone_weights=None,
        normalize_weights=True,
        normalize_by_target_range=False,
        reduction="mean",
        eps=1e-8,
    ):
        super().__init__()
        self.zone_boundaries = zone_boundaries
        self.zone_weights = zone_weights
        self.normalize_weights = bool(normalize_weights)
        self.normalize_by_target_range = bool(normalize_by_target_range)
        self.reduction = reduction.lower()
        self.eps = float(eps)
        if weights is None:
            self.register_buffer("weights", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32), persistent=False)

    def _weights_for(self, n_points, y_true):
        if self.weights.numel() > 0:
            if self.weights.numel() != n_points:
                raise ValueError(f"weights length ({self.weights.numel()}) does not match output length ({n_points}).")
            w = self.weights.to(y_true.device, y_true.dtype)
        else:
            if self.zone_boundaries is None:
                b1 = n_points // 3
                b2 = (2 * n_points) // 3
            else:
                if len(self.zone_boundaries) != 2:
                    raise ValueError("zone_boundaries must be a tuple/list of length 2.")
                b1, b2 = int(self.zone_boundaries[0]), int(self.zone_boundaries[1])
            if not (0 < b1 < b2 < n_points):
                raise ValueError(f"Invalid zone boundaries ({b1}, {b2}) for n_points={n_points}.")
            zone_weights = [1.0, 5.0, 2.0] if self.zone_weights is None else list(self.zone_weights)
            if len(zone_weights) != 3:
                raise ValueError("zone_weights must contain exactly 3 values.")
            w = y_true.new_ones(n_points)
            w[:b1] = float(zone_weights[0])
            w[b1:b2] = float(zone_weights[1])
            w[b2:] = float(zone_weights[2])

        if self.normalize_weights:
            w = w / w.mean().clamp_min(torch.finfo(w.dtype).eps)
        return w

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        w = self._weights_for(y_true.shape[-1], y_true).unsqueeze(0)
        loss = w * (y_pred - y_true) ** 2
        if self.normalize_by_target_range:
            y_scale = y_true.max(dim=1).values - y_true.min(dim=1).values
            fallback = torch.maximum(y_scale.max(), y_true.abs().max())
            fallback = torch.where(fallback > self.eps, fallback, torch.ones_like(fallback))
            y_scale = torch.where(y_scale > self.eps, y_scale, fallback).unsqueeze(1)
            loss = loss / (y_scale ** 2)
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

class CurveDerivativeLoss(nn.Module):
    """
    Penalizes mismatch in normalized point-to-point curve derivatives.
    """
    def __init__(self, x_values=None, order=1, reduction="mean", eps=1e-8):
        super().__init__()
        self.order = int(order)
        self.reduction = reduction.lower()
        self.eps = float(eps)
        if self.order < 1:
            raise ValueError("order must be >= 1.")
        x_values = _coerce_curve_x_values(x_values)
        if x_values is None:
            self.register_buffer("x_values", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("x_values", torch.tensor(x_values, dtype=torch.float32), persistent=False)

    def _x(self, n_points, y):
        if self.x_values.numel() > 0:
            if self.x_values.numel() != n_points:
                raise ValueError(f"x_values length ({self.x_values.numel()}) does not match output length ({n_points}).")
            x = self.x_values.to(y.device, y.dtype)
        else:
            x = torch.linspace(0.0, 1.0, steps=n_points, device=y.device, dtype=y.dtype)
        x_span = (x.max() - x.min()).clamp_min(self.eps)
        return (x - x.min()) / x_span

    def _curve_value_scale(self, y):
        sample_scale = y.max(dim=1).values - y.min(dim=1).values
        fallback = torch.maximum(sample_scale.max(), y.abs().max())
        fallback = torch.where(fallback > self.eps, fallback, torch.ones_like(fallback))
        return torch.where(sample_scale > self.eps, sample_scale, fallback).unsqueeze(1)

    def _normalized_derivative(self, y, y_scale):
        x = self._x(y.shape[-1], y)
        out = y / y_scale
        for _ in range(self.order):
            dx = (x[1:] - x[:-1]).abs().clamp_min(self.eps)
            out = (out[:, 1:] - out[:, :-1]) / dx.unsqueeze(0)
            x = 0.5 * (x[1:] + x[:-1])
        return out

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        y_scale = self._curve_value_scale(y_true)
        diff_err = self._normalized_derivative(y_pred, y_scale) - self._normalized_derivative(y_true, y_scale)

        loss = diff_err ** 2
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

class PeakStressLoss(nn.Module):
    """
    Penalizes mismatch in maximum curve value.
    """
    def __init__(self, reduction="mean", normalize=False, eps=1e-8):
        super().__init__()
        self.reduction = reduction.lower()
        self.normalize = bool(normalize)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        err = y_pred.max(dim=1).values - y_true.max(dim=1).values
        if self.normalize:
            y_scale = y_true.max(dim=1).values - y_true.min(dim=1).values
            fallback = torch.maximum(y_scale.max(), y_true.abs().max())
            fallback = torch.where(fallback > self.eps, fallback, torch.ones_like(fallback))
            y_scale = torch.where(y_scale > self.eps, y_scale, fallback)
            err = err / y_scale
        loss = err ** 2
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

class StrainEnergyLoss(nn.Module):
    """
    Squared error between integrated predicted and target curves up to fracture index.
    fracture_indices can be:
    - None: integrate full curve
    - 1D tensor/list/ndarray of per-sample indices
    - callable(batch_size, n_points, device) -> indices
    """
    def __init__(self, x_values=None, fracture_indices=None, reduction="mean", normalize=False, eps=1e-8):
        super().__init__()
        self.fracture_indices = fracture_indices
        self.reduction = reduction.lower()
        self.normalize = bool(normalize)
        self.eps = float(eps)
        x_values = _coerce_curve_x_values(x_values)
        if x_values is None:
            self.register_buffer("x_values", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("x_values", torch.tensor(x_values, dtype=torch.float32), persistent=False)

    def _get_fracture_indices(self, batch_size, n_points, device):
        if self.fracture_indices is None:
            return torch.full((batch_size,), n_points - 1, dtype=torch.long, device=device)
        if callable(self.fracture_indices):
            idx = self.fracture_indices(batch_size=batch_size, n_points=n_points, device=device)
            return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)
        idx = torch.as_tensor(self.fracture_indices, dtype=torch.long, device=device).view(-1)
        if idx.numel() == 1:
            idx = idx.repeat(batch_size)
        if idx.numel() != batch_size:
            raise ValueError(
                f"fracture_indices length ({idx.numel()}) must match batch size ({batch_size}) "
                "or be a single scalar."
            )
        return idx.clamp(min=0, max=n_points - 1)

    def _integral_up_to(self, y_row, idx):
        end = int(idx.item()) + 1
        if end <= 1:
            return y_row.new_tensor(0.0)
        if self.x_values.numel() > 0:
            if self.x_values.numel() != y_row.numel():
                raise ValueError(
                    f"x_values length ({self.x_values.numel()}) does not match output length ({y_row.numel()})."
                )
            return torch.trapz(y_row[:end], self.x_values[:end].to(y_row.device, y_row.dtype))
        return torch.trapz(y_row[:end])

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        batch_size, n_points = y_pred.shape
        fracture_idx = self._get_fracture_indices(batch_size, n_points, y_pred.device)

        err_sq_terms = []
        for i in range(batch_size):
            end = int(fracture_idx[i].item()) + 1
            pred_int = self._integral_up_to(y_pred[i], fracture_idx[i])
            true_int = self._integral_up_to(y_true[i], fracture_idx[i])
            err = pred_int - true_int
            if self.normalize:
                if self.x_values.numel() > 0:
                    x = self.x_values[:end].to(y_true.device, y_true.dtype)
                    norm = torch.trapz(torch.abs(y_true[i, :end]), x)
                else:
                    norm = torch.trapz(torch.abs(y_true[i, :end]))
                if bool((norm <= self.eps).item()):
                    norm = y_true.new_tensor(1.0)
                err = err / norm
            err_sq_terms.append(err ** 2)
        err_sq = torch.stack(err_sq_terms)

        if self.reduction == "sum":
            return err_sq.sum()
        if self.reduction == "none":
            return err_sq
        return err_sq.mean()

class SoftPeakLocationLoss(nn.Module):
    """
    Differentiable strain-at-peak loss using soft-argmax for predictions.
    """
    def __init__(self, x_values=None, beta=20.0, reduction="mean", normalize=False, eps=1e-8):
        super().__init__()
        self.beta = float(beta)
        self.reduction = reduction.lower()
        self.normalize = bool(normalize)
        self.eps = float(eps)
        x_values = _coerce_curve_x_values(x_values)
        if x_values is None:
            self.register_buffer("x_values", torch.tensor([], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("x_values", torch.tensor(x_values, dtype=torch.float32), persistent=False)

    def _x(self, n_points, y):
        if self.x_values.numel() > 0:
            if self.x_values.numel() != n_points:
                raise ValueError(f"x_values length ({self.x_values.numel()}) does not match output length ({n_points}).")
            return self.x_values.to(y.device, y.dtype)
        return torch.linspace(0.0, 1.0, steps=n_points, device=y.device, dtype=y.dtype)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")
        if y_pred.ndim == 1:
            y_pred = y_pred.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
        if y_pred.ndim != 2:
            raise ValueError(f"Expected 2D tensors [batch, n_points], got ndim={y_pred.ndim}")

        x = self._x(y_true.shape[-1], y_true)
        y_scale = y_true.max(dim=1).values - y_true.min(dim=1).values
        fallback = torch.maximum(y_scale.max(), y_true.abs().max())
        fallback = torch.where(fallback > self.eps, fallback, torch.ones_like(fallback))
        y_scale = torch.where(y_scale > self.eps, y_scale, fallback).unsqueeze(1)
        pred_weights = torch.softmax(self.beta * y_pred / y_scale, dim=1)
        pred_peak_x = (pred_weights * x.unsqueeze(0)).sum(dim=1)
        true_peak_x = x[y_true.argmax(dim=1)]
        err = pred_peak_x - true_peak_x
        if self.normalize:
            x_scale = (x.max() - x.min()).clamp_min(self.eps)
            err = err / x_scale
        loss = err ** 2
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _coerce_curve_x_values(x_values):
    if x_values is None:
        return None
    if hasattr(x_values, "iloc") and getattr(x_values, "ndim", None) == 2:
        row = x_values.iloc[0]
        if hasattr(row, "iloc"):
            row = row.iloc[1:]
        return row.to_numpy(dtype=float) if hasattr(row, "to_numpy") else np.asarray(row, dtype=float)
    if hasattr(x_values, "to_numpy"):
        return x_values.to_numpy(dtype=float)
    return x_values


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

def plot_predictions(OUT_df, test_outputs, truth=None, mode="ut", indx=0, d_out=True):
    fig = plt.figure(figsize=(10, 5))
    x = OUT_df.iloc[0][1:].to_numpy()
    if d_out:
        y = OUT_df.iloc[1][1:].to_numpy()
    else:
        y = np.zeros_like(x)

    plt.plot(x, test_outputs[indx]+y, "-", label=f"Prediction-{indx}", c="orangered")
    if truth is not None:
        plt.plot(x, truth[indx]+y, "-", label=f"Truth-{indx}", c="darkgreen")
        plt.bar(x, absErr(truth[indx], test_outputs[indx]), width=(max(x)-min(x))/(len(x)), alpha=0.25, label="Error", color="gray")
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


### HYPERPARAMETER OPTIMIZATION
def GPR_HPopt(data, gpr, params, cv=5, verb=0):
    grid_search = GridSearchCV(gpr, params, cv=cv, scoring=None, verbose=verb)
    return grid_search.fit(data[0], data[1])

def _hopt_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if isinstance(cfg, dict) else default

def _hopt_by_typ(value, typ):
    if not isinstance(value, dict):
        return value
    typ = str(typ).lower()
    typ_keys = {"mlp", "tr", "gnn", "gcn", "gat", "default"}
    if not any(key in value for key in typ_keys):
        return value
    return value.get(typ, value.get("default"))

def _hopt_sample(trial, name, spec, default=None):
    if callable(spec):
        return spec(trial)
    if spec is None:
        return default
    if isinstance(spec, dict):
        kind = spec.get("type", "categorical")
        if kind == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step", None))
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
        if kind == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        if kind == "fixed":
            return spec.get("value", default)
        raise ValueError(f"Unsupported HPO spec type '{kind}' for '{name}'.")
    if isinstance(spec, (list, tuple)):
        return trial.suggest_categorical(name, list(spec))
    return spec

def _hopt_sample_weight(trial, name, cfg, default):
    spec = _hopt_get(cfg, name, None)
    if spec is not None:
        return float(_hopt_sample(trial, name, spec, default))
    use_term = trial.suggest_categorical(f"use_{name}", [True, False])
    if not use_term:
        return 0.0
    if default == 0.0:
        return trial.suggest_float(name, 1e-3, 1.0, log=True)
    low = max(default*0.1, 1e-3)
    high = max(default*10.0, low)
    return trial.suggest_float(name, low, high, log=True)

def _hopt_mode_data(data, mode):
    mode = mode.upper()
    if mode == "UT" and getattr(data, "UTmechTest", False):
        return getattr(data, "UT_train_in"), getattr(data, "UT_train_out")
    if mode == "FT" and getattr(data, "FTmechTest", False):
        in_arr = getattr(data, "UT_train_in", None) if getattr(data, "UTmechTest", False) else getattr(data, "FT_train_in")
        return in_arr, getattr(data, "FT_train_out")
    raise ValueError(f"DATA object does not contain {mode} training data.")

def _hopt_primary_mode(data):
    return "UT" if getattr(data, "UTmechTest", False) else "FT"

def _hopt_io_sizes(data, typ):
    typ = str(typ).lower()
    x, y = _hopt_mode_data(data, _hopt_primary_mode(data))
    x_shape = np.asarray(x).shape
    y_shape = np.asarray(y).shape
    if len(y_shape) < 2:
        raise ValueError(f"Expected output array with shape [samples, outputs], got {y_shape}.")
    out_size = int(y_shape[-1])

    if typ == "mlp":
        if len(x_shape) != 2:
            raise ValueError(
                "MLP HPO expects flat DATA inputs with shape [samples, features]. "
                f"Got {x_shape}; create/use an MLP-compatible DATA object."
            )
        return {"in_size": int(x_shape[-1]), "out_size": out_size}

    if typ == "tr":
        if len(x_shape) != 3:
            raise ValueError(
                "Transformer HPO expects token DATA inputs with shape [samples, nodes, features]. "
                f"Got {x_shape}; create/use a Transformer-compatible DATA object."
            )
        return {"in_size": int(x_shape[-1]), "seq_len": int(x_shape[-2]), "out_size": out_size}

    if typ in ["gnn", "gcn", "gat"]:
        if len(x_shape) != 3:
            raise ValueError(
                "GNN/GAT/GCN HPO expects node DATA inputs with shape [samples, nodes, features]. "
                f"Got {x_shape}; create/use a graph-compatible DATA object."
            )
        return {"in_size": int(x_shape[-1]), "out_size": out_size}

    raise ValueError("typ must be one of ['mlp', 'tr', 'gnn', 'gcn', 'gat'].")

def hOpt_suggest_model_params(trial, typ, data, search_space=None):
    typ = str(typ).lower()
    cfg = _hopt_get(search_space, "model", search_space)
    io = _hopt_io_sizes(data, typ)

    act = _hopt_sample(trial, f"{typ}_act", _hopt_get(cfg, "act", ["relu", "gelu", "elu", "mish"]), "relu")
    norm = _hopt_sample(trial, f"{typ}_norm", _hopt_get(cfg, "norm", [None, "layer"]), None)
    dropout = float(_hopt_sample(trial, f"{typ}_dropout", _hopt_get(cfg, "dropout", {"type": "float", "low": 0.0, "high": 0.35}), 0.0))
    head_dropout = float(_hopt_sample(trial, f"{typ}_head_dropout", _hopt_get(cfg, "head_dropout", {"type": "float", "low": 0.0, "high": 0.35}), dropout))

    if typ == "mlp":
        depth = int(_hopt_sample(trial, "mlp_depth", _hopt_get(cfg, "depth", [2, 3, 4, 5]), 3))
        width = int(_hopt_sample(trial, "mlp_width", _hopt_get(cfg, "width", [64, 128, 256, 512]), 256))
        block = _hopt_sample(trial, "mlp_block", _hopt_get(cfg, "block", ["mlp", "res"]), "mlp")
        params = {
            **io,
            "h_size": [width]*depth,
            "act": act,
            "block": block,
            "norm": norm,
            "dropout": dropout,
            "head_norm": _hopt_sample(trial, "mlp_head_norm", _hopt_get(cfg, "head_norm", [None, "layer"]), None),
            "head_dropout": head_dropout,
        }
        return params

    if typ == "tr":
        d_model = int(_hopt_sample(trial, "tr_d_model", _hopt_get(cfg, "d_model", [32, 64, 128, 256]), 128))
        head_options = [h for h in _hopt_get(cfg, "n_heads", [1, 2, 4, 8]) if d_model % int(h) == 0]
        if len(head_options) == 0:
            raise ValueError(f"No valid Transformer n_heads choices divide d_model={d_model}.")
        n_heads = int(trial.suggest_categorical("tr_n_heads", head_options))
        pool = _hopt_sample(trial, "tr_pool", _hopt_get(cfg, "pool", ["mean", "cls", "add", "max"]), "mean")
        use_cls_token = True if pool == "cls" else bool(_hopt_sample(trial, "tr_use_cls_token", _hopt_get(cfg, "use_cls_token", [True, False]), True))
        head_depth = int(_hopt_sample(trial, "tr_head_depth", _hopt_get(cfg, "head_depth", [0, 1, 2]), 1))
        head_width = int(_hopt_sample(trial, "tr_head_width", _hopt_get(cfg, "head_width", [64, 128, 256, 512]), 128))
        params = {
            **io,
            "h_size": [head_width]*head_depth,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": int(_hopt_sample(trial, "tr_n_layers", _hopt_get(cfg, "n_layers", [1, 2, 3, 4]), 2)),
            "ff_mult": int(_hopt_sample(trial, "tr_ff_mult", _hopt_get(cfg, "ff_mult", [2, 4, 6, 8]), 4)),
            "act": act,
            "encoder_act": _hopt_sample(trial, "tr_encoder_act", _hopt_get(cfg, "encoder_act", ["gelu", "relu"]), "gelu"),
            "block": _hopt_sample(trial, "tr_head_block", _hopt_get(cfg, "block", ["mlp", "res"]), "mlp"),
            "norm": norm,
            "dropout": dropout,
            "att_dropout": float(_hopt_sample(trial, "tr_att_dropout", _hopt_get(cfg, "att_dropout", {"type": "float", "low": 0.0, "high": 0.35}), dropout)),
            "head_norm": _hopt_sample(trial, "tr_head_norm", _hopt_get(cfg, "head_norm", ["same", None, "layer"]), "same"),
            "head_dropout": head_dropout,
            "pool": pool,
            "use_cls_token": use_cls_token,
            "pos_encoding": _hopt_sample(trial, "tr_pos_encoding", _hopt_get(cfg, "pos_encoding", ["learned", "sinusoidal"]), "learned"),
        }
        return params

    if typ in ["gnn", "gcn", "gat"]:
        block = typ if typ in ["gcn", "gat"] else _hopt_sample(trial, "gnn_block", _hopt_get(cfg, "block", ["gcn", "gat"]), "gcn")
        depth = int(_hopt_sample(trial, f"{typ}_depth", _hopt_get(cfg, "depth", [2, 3, 4]), 3))
        width = int(_hopt_sample(trial, f"{typ}_width", _hopt_get(cfg, "width", [32, 64, 128, 256]), 128))
        params = {
            **io,
            "h_size": [width]*depth,
            "act": act,
            "block": block,
            "norm": norm,
            "dropout": dropout,
            "att_dropout": float(_hopt_sample(trial, f"{typ}_att_dropout", _hopt_get(cfg, "att_dropout", {"type": "float", "low": 0.0, "high": 0.35}), dropout)),
            "head_norm": _hopt_sample(trial, f"{typ}_head_norm", _hopt_get(cfg, "head_norm", [None, "layer"]), None),
            "head_dropout": head_dropout,
            "heads": int(_hopt_sample(trial, f"{typ}_heads", _hopt_get(cfg, "heads", [1, 2, 4, 8]), 1)) if block == "gat" else 1,
            "pool": _hopt_sample(trial, f"{typ}_pool", _hopt_get(cfg, "pool", ["mean", "add"]), "mean"),
        }
        return params

    raise ValueError("typ must be one of ['mlp', 'tr', 'gnn', 'gcn', 'gat'].")

def hOpt_build_model(typ, params):
    from resources.MLmodels import MLP, Transformer, GNN

    typ = str(typ).lower()
    if typ == "mlp":
        return MLP(**params)
    if typ == "tr":
        return Transformer(**params)
    if typ in ["gnn", "gcn", "gat"]:
        params = dict(params)
        if typ in ["gcn", "gat"]:
            params["block"] = typ
        return GNN(**params)
    raise ValueError("typ must be one of ['mlp', 'tr', 'gnn', 'gcn', 'gat'].")

def _hopt_curve_x_values(data, mode):
    out_df = getattr(data, f"{mode}_OUT_df", None)
    if out_df is None or not hasattr(out_df, "iloc"):
        return None
    x_values = out_df.iloc[0]
    if hasattr(x_values, "drop") and "0" in list(x_values.index):
        x_values = x_values.drop(labels=["0"])
    return np.asarray(x_values, dtype=float)

def _hopt_loss_task_cfg(loss_cfg, mode):
    if not isinstance(loss_cfg, dict):
        return {}
    task_cfg = dict(loss_cfg.get(mode, {}))
    for key, value in loss_cfg.items():
        if key not in ["UT", "FT", "task", "tasks"]:
            task_cfg.setdefault(key, value)
    return task_cfg

def hOpt_suggest_loss_params(trial, loss_cfg=None):
    family = _hopt_sample(trial, "loss_family", _hopt_get(loss_cfg, "family", ["combined"]), "combined")
    if family.lower() in ["mse", "nn.mse", "mse_loss"]:
        return {"family": "mse"}
    if family.lower() != "combined":
        raise ValueError("Supported HPO loss families are 'combined' and 'mse'.")

    params = {
        "family": "combined",
        "mse_weight": _hopt_sample_weight(trial, "mse_weight", loss_cfg, 0.1),
        "weighted_mse_weight": _hopt_sample_weight(trial, "weighted_mse_weight", loss_cfg, 1.0),
        "derivative_weight": _hopt_sample_weight(trial, "derivative_weight", loss_cfg, 0.25),
        "peak_weight": _hopt_sample_weight(trial, "peak_weight", loss_cfg, 0.2),
        "energy_weight": _hopt_sample_weight(trial, "energy_weight", loss_cfg, 0.2),
        "peak_location_weight": _hopt_sample_weight(trial, "peak_location_weight", loss_cfg, 0.05),
        "reduction": _hopt_sample(trial, "loss_reduction", _hopt_get(loss_cfg, "reduction", ["mean"]), "mean"),
        "derivative_order": int(_hopt_sample(trial, "loss_derivative_order", _hopt_get(loss_cfg, "derivative_order", [1, 2]), 1)),
        "SoftPeak_beta": float(_hopt_sample(trial, "loss_SoftPeak_beta", _hopt_get(loss_cfg, "SoftPeak_beta", [5.0, 10.0, 20.0, 40.0]), 20.0)),
        "normalization_eps": float(_hopt_sample(trial, "loss_normalization_eps", _hopt_get(loss_cfg, "normalization_eps", {"type": "fixed", "value": 1e-8}), 1e-8)),
    }
    return params

def hOpt_build_loss(loss_params, data, loss_cfg=None):
    if loss_params.get("family", "combined") == "mse":
        return nn.MSELoss()

    task_losses = {}
    for mode in ["UT", "FT"]:
        if not getattr(data, f"{mode}mechTest", False):
            continue
        task_cfg = _hopt_loss_task_cfg(loss_cfg, mode)
        params = {k: v for k, v in loss_params.items() if k != "family"}
        params["zone_boundaries"] = _hopt_get(task_cfg, "zone_boundaries", (67, 134))
        params["zone_weights"] = _hopt_get(task_cfg, "zone_weights", (1.0, 5.0, 2.0))
        params["x_values"] = _hopt_get(task_cfg, "x_values", _hopt_curve_x_values(data, mode))
        task_losses[mode] = CombinedCurveLoss(**params)

    if len(task_losses) == 1:
        return next(iter(task_losses.values()))
    return task_losses

def hOpt_suggest_training_params(trial, typ=None, train_cfg=None):
    typ = str(typ).lower() if typ is not None else None
    default_batches = [4, 8, 16, 32] if typ in ["gnn", "gcn", "gat"] else [8, 16, 32, 64]
    opt_name = _hopt_sample(trial, "optimizer", _hopt_get(train_cfg, "optimizer", ["adamw", "adam"]), "adamw")
    params = {
        "lr": float(_hopt_sample(trial, "lr", _hopt_get(train_cfg, "lr", {"type": "float", "low": 1e-5, "high": 5e-3, "log": True}), 1e-3)),
        "weight_decay": float(_hopt_sample(trial, "weight_decay", _hopt_get(train_cfg, "weight_decay", {"type": "float", "low": 1e-8, "high": 1e-2, "log": True}), 0.0)),
        "opt_name": opt_name,
        "batch": int(_hopt_sample(trial, "batch", _hopt_get(train_cfg, "batch", default_batches), default_batches[0])),
        "n_epochs": int(_hopt_sample(trial, "n_epochs", _hopt_get(train_cfg, "n_epochs", {"type": "fixed", "value": 75}), 75)),
        "verbose": int(_hopt_sample(trial, "verbose", _hopt_get(train_cfg, "verbose", {"type": "fixed", "value": 0}), 0)),
        "metric": _hopt_sample(trial, "objective_metric", _hopt_get(train_cfg, "metric", ["loss"]), "loss"),
        "w_init": _hopt_sample(trial, "w_init", _hopt_get(train_cfg, "w_init", ["auto"]), "auto"),
    }

    scheduler = _hopt_sample(trial, "scheduler", _hopt_get(train_cfg, "scheduler", ["none", "plateau"]), "none")
    if scheduler in [None, False, "none"]:
        params["scheduler"] = None
    elif scheduler == "plateau":
        factor = float(_hopt_sample(trial, "scheduler_factor", _hopt_get(train_cfg, "scheduler_factor", [0.3, 0.5, 0.7]), 0.5))
        patience = int(_hopt_sample(trial, "scheduler_patience", _hopt_get(train_cfg, "scheduler_patience", [5, 10, 15]), 10))
        threshold = float(_hopt_sample(trial, "scheduler_threshold", _hopt_get(train_cfg, "scheduler_threshold", {"type": "fixed", "value": 1e-4}), 1e-4))
        params["scheduler"] = ("plateau", "min", factor, patience, threshold)
    else:
        raise ValueError("Supported HPO schedulers are 'none' and 'plateau'.")

    early_stop = _hopt_sample(trial, "early_stop", _hopt_get(train_cfg, "early_stop", [True]), True)
    if early_stop:
        patience = int(_hopt_sample(trial, "early_stop_patience", _hopt_get(train_cfg, "early_stop_patience", [10, 20, 30]), 20))
        min_delta = float(_hopt_sample(trial, "early_stop_min_delta", _hopt_get(train_cfg, "early_stop_min_delta", {"type": "fixed", "value": 1e-4}), 1e-4))
        params["earlyStop"] = EarlyStopping(patience=patience, min_delta=min_delta, verbose=False)
    else:
        params["earlyStop"] = None

    return params

def _hopt_task_metric(model_instance, mode, metric):
    metric = str(metric).lower()
    if metric == "loss":
        return float(getattr(model_instance, f"{mode}_best_loss"))
    if metric == "mse":
        return float(getattr(model_instance, f"{mode}_best_mse"))
    if metric == "rmse":
        rmse_value = getattr(model_instance, f"{mode}_best_rmse")
        return float(rmse_value[1] if isinstance(rmse_value, (list, tuple)) else rmse_value)
    raise ValueError("objective metric must be one of ['loss', 'mse', 'rmse'].")

def make_hOpt_objective(
    typ,
    data,
    model_space=None,
    loss_space=None,
    train_space=None,
    mechMode=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    RMSEtarget=False,
    return_model=False,
):
    def _objective(trial):
        from resources.MLmodels import MODEL

        model_params = hOpt_suggest_model_params(trial, typ, data, search_space=model_space)
        loss_params = hOpt_suggest_loss_params(trial, loss_cfg=loss_space)
        train_params = hOpt_suggest_training_params(trial, typ=typ, train_cfg=train_space)

        model = hOpt_build_model(typ, model_params)
        lossf = hOpt_build_loss(loss_params, data, loss_cfg=loss_space)

        model_instance = MODEL(
            typ=typ,
            model=model,
            lossf=lossf,
            opt=(train_params["opt_name"], train_params["weight_decay"]),
            batch=train_params["batch"],
            lr=train_params["lr"],
            data=data,
            mechMode=mechMode if mechMode is not None else getattr(data, "mechMode", "both"),
            scheduler=train_params["scheduler"],
            earlyStop=train_params["earlyStop"],
            w_init=train_params["w_init"],
            device=device,
            optTrial=trial,
            scan_matches_on_init=False,
        )

        model_instance.train(
            n_epochs=train_params["n_epochs"],
            verbose=train_params["verbose"],
            plot=False,
            RMSEtarget=RMSEtarget,
        )

        values = []
        if getattr(data, "UTmechTest", False) and hasattr(model_instance, "UT_best_loss"):
            values.append(_hopt_task_metric(model_instance, "UT", train_params["metric"]))
        if getattr(data, "FTmechTest", False) and hasattr(model_instance, "FT_best_loss"):
            values.append(_hopt_task_metric(model_instance, "FT", train_params["metric"]))
        if not values:
            raise ValueError("HPO objective could not find UT/FT validation metrics after training.")

        score = float(np.mean(values))
        trial.set_user_attr("typ", typ)
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("loss_params", loss_params)
        trial.set_user_attr("train_params", {k: v for k, v in train_params.items() if k != "earlyStop"})
        trial.set_user_attr("task_scores", values)
        if return_model:
            trial.set_user_attr("model_instance", model_instance)
        return score

    return _objective

def objective(trial, model):
    """Legacy objective for a preconfigured MODEL template."""
    if not hasattr(model, "data") or not hasattr(model, "model"):
        raise ValueError("objective expects a configured MODEL instance template.")

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)

    lossf = getattr(model, "lossf_cfg", None)
    if lossf is None:
        lossf = model.losses if hasattr(model, "losses") else model.lossf
    template_model = copy.deepcopy(model.model)
    model_instance = model.__class__(
        typ=model.typ,
        model=template_model,
        lossf=lossf,
        opt=("adam", weight_decay),
        batch=batch_size,
        lr=lr,
        data=model.data,
        mechMode=getattr(model.data, "mechMode", "both"),
        scheduler=None,
        earlyStop=copy.deepcopy(model.earlyStop),
        w_init="auto",
        device=model.device,
        optTrial=trial,
        scan_matches_on_init=False
    )

    model_instance.train(n_epochs=50, verbose=50)
    losses = []
    if getattr(model_instance.data, "UTmechTest", False) and hasattr(model_instance, "UT_best_loss"):
        losses.append(model_instance.UT_best_loss)
    if getattr(model_instance.data, "FTmechTest", False) and hasattr(model_instance, "FT_best_loss"):
        losses.append(model_instance.FT_best_loss)

    if not losses:
        raise ValueError("objective could not find UT/FT best loss values after training.")
    return float(np.mean(losses))

def hOpt(
    objective,
    n_trials=50,
    prnt=False,
    save=False,
    path="models/etc",
    name="sample",
    sampler=None,
    pruner=None,
    seed=None,
    timeout=None,
    n_jobs=1,
    show_progress_bar=False,
):
    if sampler is None:
        sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed)
    if pruner is None:
        base_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        pruner = optuna.pruners.PatientPruner(base_pruner, patience=5) if hasattr(optuna.pruners, "PatientPruner") else base_pruner

    if save:
        os.makedirs(f"{path}/{name}/HPO", exist_ok=True)
        storage_name = f"sqlite:///{path}/{name}/HPO/full_study.db"
        study_name = name
        study = optuna.create_study(storage=storage_name,
                                    study_name=study_name,
                                    direction="minimize",
                                    sampler=sampler,
                                    pruner=pruner,
                                    load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress_bar)

    best_trial = study.best_trial

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
        with open(f"{path}/{name}/HPO/best_trial_user_attrs.json", "w") as f:
            json.dump(_hopt_json_safe(best_trial.user_attrs), f, indent=4)
    
    return study

def hOpt_model(
    typ,
    data,
    n_trials=50,
    model_space=None,
    loss_space=None,
    train_space=None,
    mechMode=None,
    prnt=True,
    save=False,
    path="models/etc",
    name=None,
    seed=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    timeout=None,
    n_jobs=1,
    RMSEtarget=False,
    show_progress_bar=False,
):
    typ = str(typ).lower()
    objective_fn = make_hOpt_objective(
        typ=typ,
        data=data,
        model_space=model_space,
        loss_space=loss_space,
        train_space=train_space,
        mechMode=mechMode,
        device=device,
        RMSEtarget=RMSEtarget,
    )
    study_name = name if name is not None else f"{typ}_hOpt"
    return hOpt(
        objective_fn,
        n_trials=n_trials,
        prnt=prnt,
        save=save,
        path=path,
        name=study_name,
        seed=seed,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

def hOpt_compare(
    typs,
    data,
    n_trials_per_typ=50,
    model_space=None,
    loss_space=None,
    train_space=None,
    mechMode=None,
    prnt=True,
    save=False,
    path="models/etc",
    name="model_compare",
    seed=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    timeout=None,
    n_jobs=1,
    RMSEtarget=False,
    show_progress_bar=False,
):
    studies = {}
    for typ in typs:
        typ_key = str(typ).lower()
        typ_data = _hopt_by_typ(data, typ_key)
        if typ_data is None:
            raise ValueError(f"No DATA object supplied for typ='{typ}'.")
        study = hOpt_model(
            typ=typ_key,
            data=typ_data,
            n_trials=n_trials_per_typ,
            model_space=_hopt_by_typ(model_space, typ_key),
            loss_space=_hopt_by_typ(loss_space, typ_key),
            train_space=_hopt_by_typ(train_space, typ_key),
            mechMode=mechMode,
            prnt=prnt,
            save=save,
            path=path,
            name=f"{name}_{typ_key}",
            seed=seed,
            device=device,
            timeout=timeout,
            n_jobs=n_jobs,
            RMSEtarget=RMSEtarget,
            show_progress_bar=show_progress_bar,
        )
        studies[typ_key] = study

    return studies

def hOpt_best_summary(studies):
    if isinstance(studies, optuna.study.Study):
        return {
            "best_value": studies.best_value,
            "best_params": studies.best_params,
            "best_user_attrs": studies.best_trial.user_attrs,
        }
    return {
        typ: {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
        }
        for typ, study in studies.items()
    }

def _hopt_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _hopt_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_hopt_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def load_bestParams(path="models/etc", name="sample"):
    with open(f"{path}/{name}/HPO/best_params.json", "r") as f:
        best_params = json.load(f)

    return best_params
