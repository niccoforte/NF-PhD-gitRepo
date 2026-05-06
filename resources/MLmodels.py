from resources.imports import *

from resources.lattices import connectivity
from resources.MLfunc import (
    train_model,
    predict_model,
    plot_loss,
    plot_predictions,
    curve_performance_diagnostics,
    print_curve_diagnostics,
    plot_curve_diagnostics,
    plot_prediction_error_curves,
    plot_curve_correlation_matrix,
    collect_layer_activations,
    summarize_activations,
    plot_activation_summary,
    absErr,
    _activation,
    visualize_graphNetwork,
    resolve_weight_init,
)

import json
import hashlib
import re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
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
        w_init=None, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        optTrial=None,
        scan_matches_on_init=True
    ):
        self.typ = typ
        self.model_template = model
        self.model = self.model_template
        self.mechMode = mechMode
        self.opt = opt
        self.scheduler_cfg = scheduler
        self.w_init = w_init
        self.data = data
        _model_configure_sequence_input(typ, self.model, data=data)
        self.w_init_fn = resolve_weight_init(w_init, self.model)
        self.w_init_name = getattr(self.w_init_fn, "__name__", str(w_init)) if self.w_init_fn else None
        self.lossf_cfg = lossf
        self.losses, self.lossf, self.task_losses, self.task_lossf = _model_resolve_task_losses(lossf, data)
        self.UT_losses, self.FT_losses = self.task_losses.get("UT", self.losses), self.task_losses.get("FT", self.losses)
        self.UT_lossf, self.FT_lossf = self.task_lossf.get("UT", self.lossf), self.task_lossf.get("FT", self.lossf)
        
        self.batch = batch
        self.lr = lr
        self.UT_dataloaders = UT_dataloaders
        self.FT_dataloaders = FT_dataloaders
        self.earlyStop = earlyStop
        if self.w_init_fn is not None:
            self.model.apply(self.w_init_fn)
        self.device = device
        self.model.to(device)
        self.optTrial = optTrial
        self.descriptor = {}
        self.match_scan = {"strict": [], "coarse": []}

        if data.UTmechTest:
            self._init_task("UT", UT_dataloaders)

        if data.FTmechTest:
            self._init_task("FT", FT_dataloaders)

        if not data.UTmechTest:
            self.UTscheduler = None
        if not data.FTmechTest:
            self.FTscheduler = None

        _model_refresh_descriptor(self)
        if scan_matches_on_init:
            self.match_scan = _model_scan_matching_checkpoints(
                self,
                search_root="models",
                recursive=True,
                print_results=True,
            )

    def _make_task_model(self):
        return copy.deepcopy(self.model_template).to(self.device)

    def _init_task(self, mode, dataloaders=None):
        components = _model_make_task_components(
            mode=mode,
            typ=self.typ,
            model=self._make_task_model(),
            opt=self.opt,
            lr=self.lr,
            data=self.data,
            batch=self.batch,
            dataloaders=dataloaders,
        )
        _model_assign_task_components(self, mode, components)
        setattr(
            self,
            f"{mode}scheduler",
            _model_scheduler(getattr(self, f"{mode}_opt"), self.scheduler_cfg),
        )
    
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
                                                 self.UT_lossf, 
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
                                                 self.FT_lossf, 
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
    
    def predict(
        self,
        test_dataloader=None,
        plot=False,
        diagnostics=True,
        diag_plot=False,
        diag_samples=64,
        zone_boundaries=None,
    ):
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
            self._predict_mode(
                "UT",
                UT_loader,
                split="test",
                plot=plot,
                diagnostics=diagnostics,
                diag_plot=diag_plot,
                diag_samples=diag_samples,
                zone_boundaries=zone_boundaries,
            )
        
        if self.data.FTmechTest:
            self._predict_mode(
                "FT",
                FT_loader,
                split="test",
                plot=plot,
                diagnostics=diagnostics,
                diag_plot=diag_plot,
                diag_samples=diag_samples,
                zone_boundaries=zone_boundaries,
            )

    def _predict_mode(
        self,
        mode,
        loader,
        split="test",
        plot=False,
        diagnostics=True,
        diag_plot=False,
        diag_samples=64,
        zone_boundaries=None,
    ):
        mode = mode.upper()
        mode_lower = mode.lower()
        task_model = getattr(self, f"{mode}_model")
        outputs, truth = predict_model(self.typ, task_model, loader)
        outputs = _model_reconstruct_output(self.data, mode, outputs)
        truth = _model_reconstruct_output(self.data, mode, truth)

        setattr(self, f"{mode}_{split}_outputs", outputs)
        if split == "test":
            setattr(self, f"{mode}_test_outputs", outputs)
            setattr(self, f"{mode}_truth", truth)
        else:
            setattr(self, f"{mode}_{split}_truth", truth)

        err = absErr(outputs, truth, typ="sum", axis=1)
        best = err.tolist().index(min(err))
        worst = err.tolist().index(max(err))
        mae = float(np.mean(np.abs(outputs - truth)))
        mse = float(np.mean((outputs - truth) ** 2))
        rmse = float(np.sqrt(mse))
        mean_sum_abs_err = float(np.mean(err))
        for name, value in [
            ("err", err),
            ("best", best),
            ("worst", worst),
            ("mae", mae),
            ("mse", mse),
            ("rmse", rmse),
            ("mean_sum_abs_err", mean_sum_abs_err),
        ]:
            setattr(self, f"{mode}_{name}" if split == "test" else f"{mode}_{split}_{name}", value)

        print(
            f"{mode} {split} metrics | MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, "
            f"mean summed abs err: {mean_sum_abs_err:.6f}"
        )
        print(f"Best prediction: {best}, Worst prediction: {worst}")

        if diagnostics:
            diag = _model_curve_diagnostics(
                self,
                mode,
                outputs,
                truth,
                split=split,
                zone_boundaries=zone_boundaries,
            )
            setattr(self, f"{mode}_{split}_diagnostics", diag)
            if split == "test":
                setattr(self, f"{mode}_diagnostics", diag)
                setattr(self, f"{mode}_prediction_summary", diag["summary"])
            print_curve_diagnostics(diag, label=f"{mode} {split}")
            if diag_plot:
                plot_curve_diagnostics(
                    getattr(self.data, f"{mode}_OUT_df"),
                    outputs,
                    truth,
                    diagnostics=diag,
                    mode=mode_lower,
                    max_samples=diag_samples,
                    sort_by="rmse",
                )

        if plot:
            plot_predictions(getattr(self.data, f"{mode}_OUT_df"), outputs, truth=truth, mode=mode_lower, indx=best,
                             d_out=False)
            plot_predictions(getattr(self.data, f"{mode}_OUT_df"), outputs, truth=truth, mode=mode_lower, indx=worst,
                             d_out=False)

        return outputs, truth

    def evaluate_split(
        self,
        split="test",
        mode=None,
        plot=False,
        diagnostics=True,
        diag_plot=False,
        diag_samples=64,
        zone_boundaries=None,
    ):
        split = str(split).lower()
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test'].")
        modes = [mode.upper()] if mode is not None else []
        if not modes:
            if self.data.UTmechTest:
                modes.append("UT")
            if self.data.FTmechTest:
                modes.append("FT")
        results = {}
        for task_mode in modes:
            loader = getattr(self, f"{task_mode}_{split}_dataloader")
            outputs, truth = self._predict_mode(
                task_mode,
                loader,
                split=split,
                plot=plot,
                diagnostics=diagnostics,
                diag_plot=diag_plot,
                diag_samples=diag_samples,
                zone_boundaries=zone_boundaries,
            )
            results[task_mode] = (outputs, truth, getattr(self, f"{task_mode}_{split}_diagnostics", None))
        return results if len(results) != 1 else next(iter(results.values()))

    def plot_diagnostics(self, mode="UT", split="test", max_samples=64, sort_by="rmse"):
        mode = mode.upper()
        split = str(split).lower()
        diagnostics = getattr(self, f"{mode}_{split}_diagnostics", None)
        if diagnostics is None and split == "test":
            diagnostics = getattr(self, f"{mode}_diagnostics", None)
        outputs = getattr(self, f"{mode}_{split}_outputs", None)
        truth = getattr(self, f"{mode}_{split}_truth", None)
        if outputs is None and split == "test":
            outputs = getattr(self, f"{mode}_test_outputs", None)
            truth = getattr(self, f"{mode}_truth", None)
        if outputs is None or truth is None:
            raise ValueError(f"No stored {mode} {split} predictions. Run predict() or evaluate_split('{split}') first.")
        if diagnostics is None:
            diagnostics = _model_curve_diagnostics(self, mode, outputs, truth, split=split)
        return plot_curve_diagnostics(
            getattr(self.data, f"{mode}_OUT_df"),
            outputs,
            truth,
            diagnostics=diagnostics,
            mode=mode.lower(),
            max_samples=max_samples,
            sort_by=sort_by,
        )

    def plot_error_curves(self, mode="UT", split="test", max_samples=50, sort_by="rmse"):
        mode = mode.upper()
        split = str(split).lower()
        diagnostics = getattr(self, f"{mode}_{split}_diagnostics", None)
        if diagnostics is None and split == "test":
            diagnostics = getattr(self, f"{mode}_diagnostics", None)
        outputs = getattr(self, f"{mode}_{split}_outputs", None)
        truth = getattr(self, f"{mode}_{split}_truth", None)
        if outputs is None and split == "test":
            outputs = getattr(self, f"{mode}_test_outputs", None)
            truth = getattr(self, f"{mode}_truth", None)
        if outputs is None or truth is None:
            raise ValueError(f"No stored {mode} {split} predictions. Run predict() or evaluate_split('{split}') first.")
        if diagnostics is None:
            diagnostics = _model_curve_diagnostics(self, mode, outputs, truth, split=split)
        return plot_prediction_error_curves(
            getattr(self.data, f"{mode}_OUT_df"),
            outputs,
            truth,
            diagnostics=diagnostics,
            mode=mode.lower(),
            max_samples=max_samples,
            sort_by=sort_by,
        )

    def plot_correlation_matrix(self, mode="UT", split="test", columns=None, method="pearson"):
        mode = mode.upper()
        split = str(split).lower()
        diagnostics = getattr(self, f"{mode}_{split}_diagnostics", None)
        if diagnostics is None and split == "test":
            diagnostics = getattr(self, f"{mode}_diagnostics", None)
        if diagnostics is None:
            outputs = getattr(self, f"{mode}_{split}_outputs", None)
            truth = getattr(self, f"{mode}_{split}_truth", None)
            if outputs is None and split == "test":
                outputs = getattr(self, f"{mode}_test_outputs", None)
                truth = getattr(self, f"{mode}_truth", None)
            if outputs is None or truth is None:
                raise ValueError(f"No stored {mode} {split} predictions. Run predict() or evaluate_split('{split}') first.")
            diagnostics = _model_curve_diagnostics(self, mode, outputs, truth, split=split)
        return plot_curve_correlation_matrix(diagnostics, columns=columns, method=method)

    def activation_diagnostics(
        self,
        mode="UT",
        split="test",
        layer_names=None,
        max_batches=1,
        plot=True,
    ):
        mode = mode.upper()
        split = str(split).lower()
        task_model = getattr(self, f"{mode}_model")
        loader = getattr(self, f"{mode}_{split}_dataloader")
        activations = collect_layer_activations(
            self.typ,
            task_model,
            loader,
            layer_names=layer_names,
            max_batches=max_batches,
            device=self.device,
        )
        summary = summarize_activations(activations)
        setattr(self, f"{mode}_{split}_activations", activations)
        setattr(self, f"{mode}_{split}_activation_summary", summary)
        if plot and not summary.empty:
            plot_activation_summary(summary)
        return summary
        
    def summary(self):
        if _model_is_gnn_type(self.typ):
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
            if self.typ.lower() == "tr":
                if self.data.UTmechTest:
                    sample_x = next(iter(self.UT_train_dataloader))[0].float().to(self.device)
                elif self.data.FTmechTest:
                    sample_x = next(iter(self.FT_train_dataloader))[0].float().to(self.device)
                else:
                    sample_x = None
                if sample_x is not None:
                    return summary(model_for_summary, input_data=sample_x)
            return summary(model_for_summary, input_size=(self.batch, model_for_summary.in_size))

    def find_matching_checkpoint(self, path=None, strict=True):
        return _model_find_matching_checkpoint(self, path=path, strict=strict)

    def scan_matching_checkpoints(self, search_root="models", recursive=True, print_results=True):
        self.match_scan = _model_scan_matching_checkpoints(
            self,
            search_root=search_root,
            recursive=recursive,
            print_results=print_results,
        )
        return self.match_scan

    def save(self, path=None, name=None):
        return _model_save_checkpoint(self, path=path, name=name)

    def load(self, model_path=None, match_current=False, strict_match=True, fallback_to_latest=False, require_descriptor_json=True):
        return _model_load_checkpoint(
            self,
            model_path=model_path,
            match_current=match_current,
            strict_match=strict_match,
            fallback_to_latest=fallback_to_latest,
            require_descriptor_json=require_descriptor_json,
        )


def _model_reconstruct_output(data, mode, values):
    reconstructor = getattr(data, f"{mode}_OUTreconstructor", None)
    values = np.asarray(values, dtype=float)
    return reconstructor(values) if callable(reconstructor) else values

def _model_split_truth(data, mode, split):
    attr = f"{mode}_{split}_out"
    if not hasattr(data, attr):
        return None
    return _model_reconstruct_output(data, mode, getattr(data, attr))

def _model_task_zone_boundaries(model_obj, mode):
    losses = getattr(model_obj, f"{mode}_losses", None)
    if losses is None:
        losses = getattr(model_obj, "losses", [])
    for loss in losses:
        zone_boundaries = getattr(loss, "zone_boundaries", None)
        if zone_boundaries is not None:
            return tuple(zone_boundaries)
        weighted_mse = getattr(loss, "weighted_mse", None)
        zone_boundaries = getattr(weighted_mse, "zone_boundaries", None)
        if zone_boundaries is not None:
            return tuple(zone_boundaries)
    return None

def _model_zone_boundaries_for_mode(model_obj, mode, zone_boundaries):
    if isinstance(zone_boundaries, dict):
        return zone_boundaries.get(mode, zone_boundaries.get(mode.lower()))
    if zone_boundaries is not None:
        return zone_boundaries
    return _model_task_zone_boundaries(model_obj, mode)

def _model_curve_diagnostics(model_obj, mode, outputs, truth, split="test", zone_boundaries=None):
    mode = mode.upper()
    zone_cfg = _model_zone_boundaries_for_mode(model_obj, mode, zone_boundaries)
    train_truth = _model_split_truth(model_obj.data, mode, "train")
    try:
        return curve_performance_diagnostics(
            outputs,
            truth,
            x_values=getattr(model_obj.data, f"{mode}_OUT_df"),
            train_truth=train_truth,
            zone_boundaries=zone_cfg,
        )
    except ValueError as exc:
        if zone_cfg is None:
            raise
        print(f"{mode} {split} diagnostics warning: {exc} Falling back to equal thirds.")
        return curve_performance_diagnostics(
            outputs,
            truth,
            x_values=getattr(model_obj.data, f"{mode}_OUT_df"),
            train_truth=train_truth,
            zone_boundaries=None,
        )


### Gaussian Process Regression model
class GPRmodel(GPR):
    def __init__(self, K, restarts, alpha, data=None):
        super().__init__()
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
        super().__init__()
        self.norm = _resolve_norm(norm)
        self.fc1 = nn.Linear(size, size*2, bias=bias)
        self.fc2 = nn.Linear(size*2, size, bias=bias)
        self.act = _activation(act)
        self.normL1 = _make_norm(self.norm, size*2)
        self.normL2 = _make_norm(self.norm, size)

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
        super().__init__()
        self.norm = _resolve_norm(norm)
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self._act = _activation(act)
        self.normL = _make_norm(self.norm, out_size)
    
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
        h_size=None,
        out_size=None, 
        act="relu", 
        block="mlp", 
        norm=None, 
        dropout=0.0,
        bias=True,
        head_norm=None,
        head_dropout=None
    ):
        super().__init__()

        if out_size is None:
            raise ValueError("out_size must be provided.")
        h_size = [] if h_size is None else list(h_size)
        self.block = block.lower()
        if self.block == "res" and len(h_size) > 0 and len(set(h_size)) > 1:
            h_size = [h_size[0]] * len(h_size)

        self.in_size = int(in_size)
        self.h_size = h_size
        self.out_size = out_size
        self.norm = _resolve_norm(norm)
        self.dropout = _resolve_dropout(dropout, 0.0)
        self.head_norm = _resolve_norm(head_norm)
        self.head_dropout = _resolve_dropout(head_dropout, None)
        self.bias = bias
        
        self._act = _activation(act)

        if len(h_size) > 0:
            self.fcIN = nn.Linear(in_size, h_size[0], bias=bias)
            if self.block == "mlp":
                self.hlayers = nn.ModuleList([
                     mlpBlock(i, j, act, self.norm, bias) for i, j in zip(h_size[:-1], h_size[1:])
                ])
            elif self.block == "res":
                self.hlayers = nn.ModuleList([
                    resBlock(i, act, self.norm, bias) for i in h_size])
            else:
                raise ValueError("block must be either 'mlp' or 'res'.")
        else:
            self.hlayers = None
            self.fcIN = None
            h_size = [in_size]
        
        self.fcOUT = nn.Linear(h_size[-1], out_size, bias=bias)

        self.normL = _make_norm(self.norm, h_size[0])
        self.head_normL = _make_norm(self.head_norm, h_size[-1]) if self.fcIN else None
        
        if self.dropout > 0.0:
            self.dropoutL = nn.Dropout(self.dropout)
        self.head_dropoutL = nn.Dropout(self.head_dropout) if self.head_dropout is not None and self.head_dropout > 0.0 else None

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
        if self.fcIN and self.head_norm:
            x = self.head_normL(x)
        if self.head_dropoutL is not None:
            x = self.head_dropoutL(x)
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
        seq_len=None,
        h_size=None,
        out_size=None,
        d_model=128,
        n_heads=8,
        n_layers=3,
        ff_mult=4,
        token_size=2,
        act="gelu",
        encoder_act=None,
        block="mlp",
        norm="layer",
        dropout=0.1,
        att_dropout=None,
        head_norm="same",
        head_dropout=None,
        bias=True,
        pool="mean",
        use_cls_token=True,
        pos_encoding="learned"
    ):
        super().__init__()

        if isinstance(in_size, (list, tuple)):
            if len(in_size) != 2:
                raise ValueError("in_size as a tuple/list must be (seq_len, node_features).")
            if seq_len is None:
                seq_len = int(in_size[0])
            in_size = int(in_size[1])
        if out_size is None:
            raise ValueError("out_size must be provided.")
        if token_size < 1:
            raise ValueError("token_size must be >= 1.")
        if d_model < 4:
            raise ValueError("d_model must be >= 4.")

        self.in_size = int(in_size)
        h_size = [] if h_size is None else list(h_size)

        self.h_size = h_size
        self.out_size = out_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.token_size = token_size
        self.encoder_act = encoder_act
        self.block = block.lower()
        self.pool = pool.lower()
        self.use_cls_token = use_cls_token
        self.pos_encoding = pos_encoding.lower()
        self.norm = _resolve_norm(norm)
        self.dropout = _resolve_dropout(dropout, 0.0)
        self.att_dropout = _resolve_dropout(att_dropout, self.dropout)
        self.head_norm = self.norm if isinstance(head_norm, str) and head_norm.lower() == "same" else _resolve_norm(head_norm)
        self.head_dropout = _resolve_dropout(head_dropout, self.dropout)
        self.bias = bias
        if self.pool not in ["mean", "add", "max", "cls"]:
            raise ValueError("pool must be one of ['mean', 'add', 'max', 'cls'].")
        if self.pool == "cls" and not self.use_cls_token:
            raise ValueError("pool='cls' requires use_cls_token=True.")

        self.pad_size = 0
        self.seq_len = None if seq_len is None else int(seq_len)
        self.total_len = None if self.seq_len is None else self.seq_len + (1 if use_cls_token else 0)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.input_proj = nn.Linear(self.in_size, d_model, bias=bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_cls_token else None

        self.pos_embed = None
        self.pos_encoder = None
        if self.pos_encoding not in ["learned", "sinusoidal"]:
            raise ValueError("pos_encoding must be 'learned' or 'sinusoidal'.")
        if self.seq_len is not None:
            self._init_positional_encoding(self.seq_len)

        act_name = _activation(act if encoder_act is None else encoder_act, return_name=True)
        tr_act = act_name if act_name in ["relu", "gelu"] else "gelu"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=self.att_dropout,
            activation=tr_act,
            batch_first=True,
            bias=bias
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_normL = _make_norm(self.head_norm, d_model)

        self._act = _activation(act)
        self.dropoutL = nn.Dropout(self.head_dropout) if self.head_dropout > 0.0 else None

        if len(h_size) > 0:
            self.fcIN = nn.Linear(d_model, h_size[0], bias=bias)
            if self.block == "mlp":
                self.hlayers = nn.ModuleList([
                    mlpBlock(i, j, act, self.head_norm, bias=bias) for i, j in zip(h_size[:-1], h_size[1:])
                ])
            elif self.block == "res":
                if len(set(h_size)) > 1:
                    raise ValueError("For block='res', all entries in h_size must be the same.")
                self.hlayers = nn.ModuleList([
                    resBlock(i, act, self.head_norm, bias=bias) for i in h_size
                ])
            else:
                raise ValueError("block must be either 'mlp' or 'res'.")
            self.fcOUT = nn.Linear(h_size[-1], out_size, bias=bias)
        else:
            self.fcIN = None
            self.hlayers = None
            self.fcOUT = nn.Linear(d_model, out_size, bias=bias)

    def _init_positional_encoding(self, seq_len):
        seq_len = int(seq_len)
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")

        if self.seq_len is not None and self.seq_len != seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {seq_len}.")

        self.seq_len = seq_len
        self.total_len = self.seq_len + (1 if self.use_cls_token else 0)
        device = self.input_proj.weight.device
        dtype = self.input_proj.weight.dtype

        if self.pos_encoding == "learned":
            self.pos_encoder = None
            if self.pos_embed is None or self.pos_embed.size(1) != self.total_len:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.total_len, self.d_model, device=device, dtype=dtype))
                nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        elif self.pos_encoding == "sinusoidal":
            self.pos_embed = None
            if self.pos_encoder is None or self.pos_encoder.pe.size(1) != self.total_len:
                self.pos_encoder = SinusoidalPositionEncoding(self.total_len, self.d_model).to(device=device, dtype=dtype)

    def configure_input_shape(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(f"Transformer expects token input shape [nodes, features], got {tuple(input_shape)}.")
        seq_len, feature_size = int(input_shape[0]), int(input_shape[1])
        if feature_size != self.in_size:
            raise ValueError(f"Expected token feature size {self.in_size}, got {feature_size}.")
        self._init_positional_encoding(seq_len)

    def _pool_tokens(self, x):
        if self.pool == "cls":
            if not self.use_cls_token:
                raise ValueError("pool='cls' requires use_cls_token=True.")
            return x[:, 0, :]

        tokens = x[:, 1:, :] if self.use_cls_token else x
        if self.pool == "mean":
            return tokens.mean(dim=1)
        if self.pool == "add":
            return tokens.sum(dim=1)
        if self.pool == "max":
            return tokens.max(dim=1).values
        raise ValueError("pool must be one of ['mean', 'add', 'max', 'cls'].")

    def _apply_head_norm(self, x):
        if self.head_normL is None:
            return x
        if isinstance(self.head_normL, nn.BatchNorm1d):
            return self.head_normL(x)
        return self.head_normL(x)

    def forward(self, x):
        if x.dim() == 2:
            if x.size(-1) != self.in_size:
                raise ValueError(
                    f"2D Transformer input is interpreted as [nodes, features]; "
                    f"expected feature size {self.in_size}, got {x.size(-1)}."
                )
            if self.seq_len is not None and x.size(0) != self.seq_len:
                raise ValueError(
                    f"2D Transformer input is interpreted as one sample with {self.seq_len} nodes; "
                    f"got {x.size(0)} nodes."
                )
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Transformer expects input shape [batch, nodes, features].")
        if x.size(-1) != self.in_size:
            raise ValueError(f"Expected token feature size {self.in_size}, got {x.size(-1)}.")
        self.configure_input_shape(x.shape[1:])

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
        super().__init__()
        self.norm = _resolve_norm(norm)
        
        self.Gconv = GCNConv(in_size, out_size, bias=bias)
        self.act = _activation(act)
        self.normL = _make_norm(self.norm, out_size)
    
    def forward(self, x, edge_index):
        x = self.Gconv(x, edge_index)
        if self.norm:
            x = self.normL(x)
        x = self.act(x)
        return x

class gatBlock(nn.Module):
    def __init__(self, in_size, out_size, act, heads=1, norm=None, dropout=0.0, bias=True):
        super().__init__()
        self.norm = _resolve_norm(norm)
        self.heads = heads
        self.att_dropout = _resolve_dropout(dropout, 0.0)
        self.act = _activation(act)
        
        self.GATconv = GATConv(in_size, out_size, heads=heads, concat=True, dropout=self.att_dropout, bias=bias)
        self.normL = _make_norm(self.norm, out_size * heads)
    
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
        h_size=None,
        out_size=None,
        act="relu",
        block="gcn",
        norm=None,
        dropout=0.0,
        att_dropout=None,
        head_norm=None,
        head_dropout=None,
        bias=True,
        heads=1,
        pool="add"
    ):
        super().__init__()

        if out_size is None:
            raise ValueError("out_size must be provided.")
        h_size = [] if h_size is None else list(h_size)

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size
        self.block = block.lower()
        self.norm = _resolve_norm(norm)
        self.dropout = _resolve_dropout(dropout, 0.0)
        self.att_dropout = _resolve_dropout(att_dropout, self.dropout) if self.block == "gat" and len(h_size) > 0 else None
        self.head_norm = _resolve_norm(head_norm)
        self.head_dropout = _resolve_dropout(head_dropout, self.dropout)
        self.pool = pool.lower()
        self.heads = heads
        self.bias = bias
        self.act = _activation(act)

        if self.block not in ["gcn", "gat"]:
            raise ValueError("block must be either 'gcn' or 'gat'.")

        if len(h_size) == 0:
            self.GconvIN = None
            self.hlayers = nn.ModuleList()
            hidden_out_dim = in_size
            norm_in_dim = None
        elif self.block.lower() == "gcn":
            self.GconvIN = GCNConv(in_size, h_size[0], bias=bias)
            hidden_out_dim = h_size[-1]
            self.hlayers = nn.ModuleList([
                gcnBlock(i, j, act, self.norm, bias=bias) for i, j in zip(h_size[:-1], h_size[1:])
            ])
            norm_in_dim = h_size[0]
        elif self.block == "gat":
            self.GconvIN = GATConv(in_size, h_size[0], heads=heads, concat=True, dropout=self.att_dropout, bias=bias)
            hidden_out_dim = h_size[-1] * heads
            self.hlayers = nn.ModuleList([
                gatBlock(i * heads, j, act, heads=heads, norm=self.norm, dropout=self.att_dropout, bias=bias)
                for i, j in zip(h_size[:-1], h_size[1:])
            ])
            norm_in_dim = h_size[0] * heads

        self.normIN = _make_norm(self.norm, norm_in_dim) if norm_in_dim is not None else None
        self.head_normL = _make_norm(self.head_norm, hidden_out_dim)

        self.dropoutL = nn.Dropout(self.dropout) if self.dropout > 0.0 else None
        self.head_dropoutL = nn.Dropout(self.head_dropout) if self.head_dropout > 0.0 else None
        self.fcOUT = nn.Linear(hidden_out_dim, out_size, bias=bias)

    def forward(self, x, edge_index, batch):
        if self.GconvIN is not None:
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

        if self.head_norm:
            x = self.head_normL(x)
        if self.head_dropoutL is not None:
            x = self.head_dropoutL(x)

        x = self.fcOUT(x)
        return x


### Other Models
class Autoencoder(nn.Module):
    def __init__(
        self,
        in_size,
        latent_size,
        h_size=None,
        decoder_h_size=None,
        block="mlp",
        act="relu",
        norm=None,
        dropout=0.0,
        head_norm=None,
        head_dropout=None,
        bias=True
    ):
        super().__init__()

        h_size = [] if h_size is None else list(h_size)
        decoder_h_size = list(reversed(h_size)) if decoder_h_size is None else list(decoder_h_size)

        self.in_size = in_size
        self.latent_size = latent_size
        self.h_size = h_size
        self.decoder_h_size = decoder_h_size
        self.block = block.lower()
        self.act = act
        self.norm = _resolve_norm(norm)
        self.dropout = _resolve_dropout(dropout, 0.0)
        self.head_norm = _resolve_norm(head_norm)
        self.head_dropout = _resolve_dropout(head_dropout, None)
        self.bias = bias

        self.encoder = MLP(
            in_size,
            h_size,
            latent_size,
            act=act,
            block=block,
            norm=self.norm,
            dropout=self.dropout,
            bias=bias,
            head_norm=self.head_norm,
            head_dropout=self.head_dropout,
        )
        self.decoder = MLP(
            latent_size,
            decoder_h_size,
            in_size,
            act=act,
            block=block,
            norm=self.norm,
            dropout=self.dropout,
            bias=bias,
            head_norm=self.head_norm,
            head_dropout=self.head_dropout,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent):
        return self.decoder(latent)
        
    def forward(self, x, return_latent=False):
        latent = self.encode(x)
        recon = self.decode(latent)
        if return_latent:
            return recon, latent
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


### HELPER FUNCTIONS
# MODEL helpers
def _model_resolve_losses(lossf):
    if isinstance(lossf, (list, tuple)):
        if len(lossf) == 0:
            raise ValueError("lossf list/tuple must contain at least one loss function.")
        losses = list(lossf)
        if len(losses) == 1:
            return losses, losses[0]

        def _combined_loss(y_pred, y_true):
            total = losses[0](y_pred, y_true)
            for lf in losses[1:]:
                total = total + lf(y_pred, y_true)
            return total

        return losses, _combined_loss
    return [lossf], lossf

def _model_resolve_task_losses(lossf, data=None):
    if not isinstance(lossf, dict):
        losses, resolved_loss = _model_resolve_losses(lossf)
        task_losses = {"UT": losses, "FT": losses}
        task_lossf = {"UT": resolved_loss, "FT": resolved_loss}
        return losses, resolved_loss, task_losses, task_lossf

    loss_key_aliases = {
        "ut": "UT",
        "uat": "UT",
        "uniaxial": "UT",
        "tension": "UT",
        "ft": "FT",
        "ct": "FT",
        "c(t)": "FT",
        "fracture": "FT",
        "default": "DEFAULT",
        "both": "DEFAULT",
        "all": "DEFAULT",
    }
    normalized = {
        loss_key_aliases.get(str(key).strip().lower(), str(key).strip().upper()): value
        for key, value in lossf.items()
    }
    default_loss = normalized.get("DEFAULT")

    enabled_modes = []
    if data is None or getattr(data, "UTmechTest", False):
        enabled_modes.append("UT")
    if data is None or getattr(data, "FTmechTest", False):
        enabled_modes.append("FT")
    if not enabled_modes:
        enabled_modes = ["UT", "FT"]

    task_losses = {}
    task_lossf = {}
    for mode in ("UT", "FT"):
        task_loss = normalized.get(mode, default_loss)
        if task_loss is None:
            if mode in enabled_modes:
                raise ValueError(
                    f"lossf dict must include '{mode}' or a 'default' loss when {mode} data is enabled."
                )
            continue

        losses, resolved_loss = _model_resolve_losses(task_loss)
        task_losses[mode] = losses
        task_lossf[mode] = resolved_loss

    all_losses = []
    seen = set()
    for mode in ("UT", "FT"):
        for lf in task_losses.get(mode, []):
            lf_id = id(lf)
            if lf_id not in seen:
                all_losses.append(lf)
                seen.add(lf_id)

    if not all_losses:
        raise ValueError("lossf dict did not resolve to any loss functions.")

    resolved_loss = task_lossf.get("UT", task_lossf.get("FT"))
    return all_losses, resolved_loss, task_losses, task_lossf

def _model_configure_sequence_input(typ, model, data=None, x=None):
    if str(typ).lower() != "tr" or not hasattr(model, "configure_input_shape"):
        return

    if x is None and data is not None:
        if getattr(data, "UTmechTest", False) and hasattr(data, "UT_train_in"):
            x = data.UT_train_in
        elif getattr(data, "FTmechTest", False) and hasattr(data, "FT_train_in"):
            x = data.FT_train_in

    if x is None:
        return

    shape = np.shape(x)
    if len(shape) < 3:
        raise ValueError(f"TR input arrays must have shape [samples, nodes, features]; got {shape}.")
    model.configure_input_shape(shape[1:])

def _model_optimizer(params, opt=("adam", 0.0), lr=1e-3):
    if opt is None:
        opt = ("adam", 0.0)

    extra_kwargs = {}
    if isinstance(opt, str):
        opt_name, weight_decay = opt, 0.0
    elif isinstance(opt, (list, tuple)):
        if len(opt) == 0:
            raise ValueError("opt list/tuple must contain at least an optimizer name.")
        opt_name = opt[0]
        weight_decay = opt[1] if len(opt) > 1 else 0.0
        if len(opt) > 2 and isinstance(opt[2], dict):
            extra_kwargs = dict(opt[2])
    else:
        raise TypeError("opt must be a string or tuple/list such as ('adam', weight_decay).")

    opt_name = str(opt_name).lower()
    optimizers = {
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamax": torch.optim.Adamax,
        "adamw": torch.optim.AdamW,
        "asgd": torch.optim.ASGD,
        "nadam": torch.optim.NAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
        "sparseadam": torch.optim.SparseAdam,
    }
    if opt_name not in optimizers:
        raise ValueError(f"Unknown optimizer '{opt_name}'. Available options: {sorted(optimizers)}.")

    kwargs = {"lr": lr, **extra_kwargs}
    if opt_name not in ["rprop", "sparseadam"]:
        kwargs["weight_decay"] = weight_decay
    return optimizers[opt_name](params, **kwargs)

def _model_scheduler(optimizer, scheduler):
    if scheduler is None:
        return None
    if not isinstance(scheduler, (list, tuple)):
        raise TypeError("scheduler must be None or a tuple/list configuration.")
    if len(scheduler) == 0:
        raise ValueError("scheduler tuple/list cannot be empty.")

    name = str(scheduler[0]).lower()
    if name in ["plateau", "reduce", "reduce_on_plateau", "reducelronplateau"]:
        if len(scheduler) < 5:
            raise ValueError("plateau scheduler must be ('plateau', mode, factor, patience, threshold).")
        _, mode, factor, patience, threshold = scheduler[:5]
    elif name in ["min", "max"]:
        if len(scheduler) < 4:
            raise ValueError("legacy plateau scheduler must be (mode, factor, patience, threshold).")
        mode, factor, patience, threshold = scheduler[:4]
    else:
        raise ValueError(
            "Unsupported scheduler configuration. Use legacy (mode, factor, patience, threshold) "
            "or explicit ('plateau', mode, factor, patience, threshold)."
        )

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="abs",
    )

def _model_is_gnn_type(typ):
    return str(typ).lower() in ["gnn", "gcn", "gat"]

def _model_task_graph(mode, data):
    in_df = getattr(data, f"{mode}_IN_df")
    train_in_df = getattr(data, f"{mode}_train_in_df", None)
    if hasattr(train_in_df, "columns"):
        node_columns = list(train_in_df.columns)
    elif hasattr(data, f"{mode}_body_columns"):
        node_columns = list(getattr(data, f"{mode}_body_columns"))
    else:
        din_df = getattr(data, f"{mode}_dIN_df")
        node_columns = list(in_df.columns.intersection(din_df.columns))

    if len(node_columns) % 2 != 0:
        raise ValueError(f"{mode}: graph node columns must be paired x/y columns; got {len(node_columns)} columns.")
    if not set(node_columns).issubset(set(in_df.columns)):
        raise ValueError(f"{mode}: graph node columns are not all present in {mode}_IN_df.")

    nodes = in_df.loc[:, node_columns].iloc[0].to_numpy(dtype=float)
    nodes = nodes.reshape(len(node_columns)//2, 2)
    edges = connectivity(data.LAT, nodes, data.geom)[:, 1:] - 1
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return nodes, edges, edge_index

def _model_make_loaders(trainDS, valDS, testDS, batch, dataloaders, gnn=False):
    if dataloaders is not None:
        return dataloaders[0], dataloaders[1], dataloaders[2]

    loader = gDataLoader if gnn else DataLoader
    return (
        loader(dataset=trainDS, batch_size=batch, shuffle=True),
        loader(dataset=valDS, batch_size=batch, shuffle=False),
        loader(dataset=testDS, batch_size=batch, shuffle=False),
    )

def _model_make_task_components(mode, typ, model, opt, lr, data, batch, dataloaders=None):
    components = {
        "model": model,
    }

    if _model_is_gnn_type(typ):
        nodes, edges, edge_index = _model_task_graph(mode, data)
        train_x, val_x, test_x = (
            getattr(data, f"{mode}_train_in"),
            getattr(data, f"{mode}_val_in"),
            getattr(data, f"{mode}_test_in"),
        )
        train_y, val_y, test_y = (
            getattr(data, f"{mode}_train_out"),
            getattr(data, f"{mode}_val_out"),
            getattr(data, f"{mode}_test_out"),
        )
        components["opt"] = _model_optimizer(model.parameters(), opt=opt, lr=lr)
        trainDS = [
            Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))
            for x, y in zip(train_x, train_y)
        ]
        valDS = [
            Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))
            for x, y in zip(val_x, val_y)
        ]
        testDS = [
            Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, y=torch.tensor(y, dtype=torch.float))
            for x, y in zip(test_x, test_y)
        ]
        train_loader, val_loader, test_loader = _model_make_loaders(
            trainDS, valDS, testDS, batch, dataloaders, gnn=True
        )
        components.update({
            "nodes": nodes,
            "edges": edges,
            "edge_index": edge_index,
            "trainDS": trainDS,
            "valDS": valDS,
            "testDS": testDS,
            "train_dataloader": train_loader,
            "val_dataloader": val_loader,
            "test_dataloader": test_loader,
        })
        return components

    if mode == "UT":
        train_x, val_x, test_x = data.UT_train_in, data.UT_val_in, data.UT_test_in
        train_y, val_y, test_y = data.UT_train_out, data.UT_val_out, data.UT_test_out
    elif getattr(data, "UTmechTest", False):
        train_x, val_x, test_x = data.UT_train_in, data.UT_val_in, data.UT_test_in
        train_y, val_y, test_y = data.FT_train_out, data.FT_val_out, data.FT_test_out
    else:
        train_x, val_x, test_x = data.FT_train_in, data.FT_val_in, data.FT_test_in
        train_y, val_y, test_y = data.FT_train_out, data.FT_val_out, data.FT_test_out

    _model_configure_sequence_input(typ, model, x=train_x)
    components["opt"] = _model_optimizer(model.parameters(), opt=opt, lr=lr)
    trainDS = _model_make_tensor_dataset(train_x, train_y)
    valDS = _model_make_tensor_dataset(val_x, val_y)
    testDS = _model_make_tensor_dataset(test_x, test_y)
    train_loader, val_loader, test_loader = _model_make_loaders(
        trainDS, valDS, testDS, batch, dataloaders, gnn=False
    )
    components.update({
        "trainDS": trainDS,
        "valDS": valDS,
        "testDS": testDS,
        "train_dataloader": train_loader,
        "val_dataloader": val_loader,
        "test_dataloader": test_loader,
    })
    return components

def _model_assign_task_components(model_obj, mode, components):
    for name, value in components.items():
        setattr(model_obj, f"{mode}_{name}", value)

# MODEL saving and loading helpers
def _mp_slugify(value, default="na", max_len=64, preserve_case=False):
    if value is None:
        return default
    text = str(value).strip()
    if not preserve_case:
        text = text.lower()
    if not text:
        return default
    text = text.replace("\\", "-").replace("/", "-")
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("._-")
    if not text:
        return default
    return text[:max_len]

def _mp_format_num(value):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isclose(val, round(val)):
        return str(int(round(val)))
    if val == 0:
        return "0"
    if abs(val) >= 1e3 or abs(val) < 1e-2:
        return f"{val:.1e}"
    return f"{val:.4g}"

def _mp_format_filename_num(value):
    try:
        from decimal import Decimal, InvalidOperation
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return str(value)
    except Exception:
        return str(value)

    if dec == dec.to_integral_value():
        return str(int(dec))

    sign = "m" if dec < 0 else ""
    text = format(abs(dec).normalize(), "f")
    if "." in text:
        whole, frac = text.split(".", 1)
        text = f"{whole}{frac.rstrip('0')}"
    return f"{sign}{text}"

def _mp_bool_token(value):
    return "True" if bool(value) else "False"

def _mp_range_split_token(data):
    range_split = getattr(data, "range_split", None)
    if isinstance(range_split, dict):
        in_range = bool(range_split.get("input", False))
        out_range = bool(range_split.get("output", False))
    else:
        in_range = bool(getattr(data, "input_range_split", False))
        out_range = bool(getattr(data, "output_range_split", False))
    return f"in{_mp_bool_token(in_range)}-out{_mp_bool_token(out_range)}"

def _mp_node_feature_token(data):
    model_name = str(getattr(data, "model", "")).lower()
    if model_name not in ["tr", "gnn", "gcn", "gat"]:
        return None
    tr_params = getattr(data, "tr_params", None)
    if not isinstance(tr_params, dict):
        return None
    geom = _mp_bool_token(tr_params.get("geom_feats", False))
    coord = _mp_bool_token(tr_params.get("coord_norm", False))
    return f"nodeFeat-geom{geom}-coordNorm{coord}"

def _mp_data_descriptor(data, model_obj=None):
    if data is None:
        return "data-default"

    parts = []
    for key in ["LAT", "dis"]:
        if hasattr(data, key):
            val = getattr(data, key)
            if val is not None and str(val).strip():
                parts.append(_mp_slugify(val, max_len=24, preserve_case=True))

    if hasattr(data, "dN") and getattr(data, "dN") is not None:
        parts.append(f"dN-{_mp_slugify(_mp_format_num(getattr(data, 'dN')), max_len=24, preserve_case=True)}")

    path_add = getattr(data, "path_add", None)
    if path_add is not None and str(path_add).strip():
        parts.append(f"pathAdd-{_mp_slugify(path_add, max_len=24, preserve_case=True)}")

    if hasattr(data, "d_data") and getattr(data, "d_data") is not None:
        parts.append(f"data-{_mp_slugify(getattr(data, 'd_data'), max_len=24, preserve_case=True)}")

    nsims = getattr(data, "nsims", None)
    parts.append(f"nSims-{_mp_slugify('all' if nsims is None else nsims, max_len=24, preserve_case=True)}")

    load_split = getattr(data, "load_split", False)
    if load_split:
        split_desc = f"split-load-{_mp_slugify(load_split, max_len=24, preserve_case=True)}" if isinstance(load_split, str) else "split-load"
    else:
        split_frac = getattr(data, "split_frac", None)
        split_frac = "na" if split_frac is None else _mp_format_num(split_frac)
        split_seed = getattr(data, "split_seed", None)
        split_seed = "None" if split_seed is None else split_seed
        split_desc = f"split-frac-{_mp_slugify(split_frac, max_len=16)}-seed-{_mp_slugify(split_seed, max_len=16, preserve_case=True)}-range-{_mp_range_split_token(data)}"
    parts.append(split_desc)

    round_decimals = getattr(data, "round_decimals", None)
    parts.append(f"round-{_mp_slugify('None' if round_decimals is None else round_decimals, max_len=16, preserve_case=True)}")

    node_feature_token = _mp_node_feature_token(data)
    if node_feature_token is not None:
        parts.append(node_feature_token)

    descriptor_obj = model_obj
    if descriptor_obj is None:
        class _DescriptorObj:
            def __init__(self, data):
                self.data = data
        descriptor_obj = _DescriptorObj(data)

    if getattr(data, "scale", None):
        scale_sig = _mp_scaler_signature(descriptor_obj)
        parts.append(f"scale-{_mp_slugify(getattr(data, 'scale', None), max_len=16, preserve_case=True)}-cfgHash{_mp_json_sha1(scale_sig)[:6]}")
    else:
        parts.append("scale-None")

    if getattr(data, "reduce_dim", None):
        reducer_sig = _mp_reducer_signature(descriptor_obj)
        parts.append(f"rDim-{_mp_slugify(getattr(data, 'reduce_dim', None), max_len=16, preserve_case=True)}-cfgHash{_mp_json_sha1(reducer_sig)[:6]}")
    else:
        parts.append("rDim-None")

    return "_".join(parts) if parts else "data-default"

def _mp_to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_mp_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _mp_to_serializable(v) for k, v in value.items()}
    return str(value)

def _mp_component_signature(component):
    if component is None:
        return None
    sign = {"class": component.__class__.__name__}
    if hasattr(component, "get_params"):
        try:
            sign["params"] = _mp_to_serializable(component.get_params(deep=False))
        except Exception:
            pass
    return sign

def _mp_json_sha1(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def _mp_weight_decay(model_obj):
    if hasattr(model_obj, "UT_opt") and model_obj.UT_opt.param_groups:
        return model_obj.UT_opt.param_groups[0].get("weight_decay", None)
    if hasattr(model_obj, "FT_opt") and model_obj.FT_opt.param_groups:
        return model_obj.FT_opt.param_groups[0].get("weight_decay", None)
    if isinstance(getattr(model_obj, "opt", None), (list, tuple)) and len(model_obj.opt) > 1:
        return model_obj.opt[1]
    return None

def _mp_scaler_signature(model_obj):
    data = getattr(model_obj, "data", None)
    if data is None:
        return None
    return {
        "scale": _mp_to_serializable(getattr(data, "scale", None)),
        "scale_reduced": _mp_to_serializable(getattr(data, "scale_reduced", None)),
        "scaler": _mp_component_signature(getattr(data, "scaler", None)),
    }

def _mp_reducer_signature(model_obj):
    data = getattr(model_obj, "data", None)
    if data is None:
        return None
    return {
        "reduce_dim": _mp_to_serializable(getattr(data, "reduce_dim", None)),
        "reducer": _mp_component_signature(getattr(data, "reducer", None)),
    }

def _mp_collect_data_signature(model_obj):
    data = getattr(model_obj, "data", None)
    signature = {}
    if data is None:
        return signature

    for key in (
        "mechMode", "LAT", "dis", "dN", "d_data", "nsims", "split_frac",
        "split_seed", "range_split", "input_range_split", "output_range_split",
        "round_decimals", "tr_params", "model",
        "scale", "scale_reduced", "reduce_dim", "path", "path_add", "load_split",
        "UTmechTest", "FTmechTest",
    ):
        if hasattr(data, key):
            signature[key] = _mp_to_serializable(getattr(data, key))

    for key in ["UT_train_in", "UT_train_out", "FT_train_in", "FT_train_out"]:
        if hasattr(data, key):
            signature[f"{key}_shape"] = list(np.shape(getattr(data, key)))

    signature["scaler_signature"] = _mp_scaler_signature(model_obj)
    signature["reducer_signature"] = _mp_reducer_signature(model_obj)
    return signature

def _mp_collect_model_signature(model_obj):
    def _is_simple(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return True
        if isinstance(value, np.generic):
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_simple(v) for v in value)
        return False

    losses = []
    if hasattr(model_obj, "losses"):
        for lf in model_obj.losses:
            losses.append(getattr(lf, "__class__", type(lf)).__name__)

    weight_decay = _mp_weight_decay(model_obj)
    model_attrs = {}
    for key in (
        "in_size", "h_size", "out_size", "act", "block", "norm", "dropout",
        "att_dropout", "head_norm", "head_dropout", "bias",
        "latent_size", "decoder_h_size",
        "heads", "n_heads", "n_layers", "d_model", "ff_mult", "token_size", "encoder_act",
        "pool", "use_cls_token", "pos_encoding", "seq_len", "pad_size", "total_len",
    ):
        if hasattr(model_obj.model, key):
            model_attrs[key] = _mp_to_serializable(getattr(model_obj.model, key))

    for key, value in vars(model_obj.model).items():
        if key.startswith("_") or key in model_attrs:
            continue
        if isinstance(value, (nn.Module, nn.Parameter, nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            continue
        if _is_simple(value):
            model_attrs[key] = _mp_to_serializable(value)

    def _opt_sig(opt_obj):
        if opt_obj is None:
            return None
        out = {"class": opt_obj.__class__.__name__}
        if getattr(opt_obj, "param_groups", None):
            pg = dict(opt_obj.param_groups[0])
            pg.pop("params", None)
            out["param_group_0"] = _mp_to_serializable(pg)
        return out

    optimizer_signature = {
        "opt_cfg": _mp_to_serializable(getattr(model_obj, "opt", None)),
        "UT_opt": _opt_sig(getattr(model_obj, "UT_opt", None)),
        "FT_opt": _opt_sig(getattr(model_obj, "FT_opt", None)),
    }

    scheduler_signature = {
        "scheduler_cfg": _mp_to_serializable(getattr(model_obj, "scheduler_cfg", None)),
        "UT_scheduler_class": getattr(getattr(model_obj, "UTscheduler", None), "__class__", type(None)).__name__
        if hasattr(model_obj, "UTscheduler") else None,
        "FT_scheduler_class": getattr(getattr(model_obj, "FTscheduler", None), "__class__", type(None)).__name__
        if hasattr(model_obj, "FTscheduler") else None,
    }

    es = getattr(model_obj, "earlyStop", None)
    earlystop_signature = None
    if es is not None:
        earlystop_signature = {"class": es.__class__.__name__}
        try:
            vals = {}
            for k, v in vars(es).items():
                if k.startswith("_"):
                    continue
                if _is_simple(v):
                    vals[k] = _mp_to_serializable(v)
            if vals:
                earlystop_signature["attrs"] = vals
        except Exception:
            pass

    return {
        "typ": model_obj.typ,
        "model_class": model_obj.model.__class__.__name__,
        "model_attrs": model_attrs,
        "mechMode": getattr(model_obj, "mechMode", None),
        "batch": model_obj.batch,
        "lr": model_obj.lr,
        "weight_decay": _mp_to_serializable(weight_decay),
        "losses": losses,
        "optimizer": optimizer_signature,
        "scheduler": scheduler_signature,
        "earlyStop": earlystop_signature,
        "w_init": {
            "enabled": bool(getattr(model_obj, "w_init", False)),
            "name": _mp_to_serializable(getattr(model_obj, "w_init_name", None)),
        },
        "device": str(model_obj.device),
    }

def _mp_build_setup_signature(model_obj, include_data_values=True):
    model_structure = {
        key: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for key, tensor in model_obj.model.state_dict().items()
    }
    signature = {
        "model_signature": _mp_collect_model_signature(model_obj),
        "data_signature": _mp_collect_data_signature(model_obj),
        "model_structure": model_structure,
    }
    if include_data_values:
        data = getattr(model_obj, "data", None)
        fp = {}
        if data is not None:
            for key in (
                "UT_train_in", "UT_train_out", "UT_val_in", "UT_val_out", "UT_test_in", "UT_test_out",
                "FT_train_in", "FT_train_out", "FT_val_in", "FT_val_out", "FT_test_in", "FT_test_out",
                "UT_trainProps", "UT_valProps", "UT_testProps",
                "FT_trainProps", "FT_valProps", "FT_testProps",
            ):
                if hasattr(data, key):
                    value = getattr(data, key)
                    if value is not None:
                        try:
                            arr = np.ascontiguousarray(np.asarray(value))
                            h = hashlib.sha1()
                            h.update(str(arr.shape).encode("utf-8"))
                            h.update(str(arr.dtype).encode("utf-8"))
                            h.update(arr.tobytes())
                            fp[key] = h.hexdigest()
                        except Exception:
                            fp[key] = f"unhashable:{type(value).__name__}"
        signature["data_value_fingerprints"] = fp
    return signature

def _mp_resolve_model_dir(model_obj, path):
    if path is None:
        data = getattr(model_obj, "data", None)
        typ = str(getattr(model_obj, "typ", "")).lower()
        block = str(getattr(model_obj.model, "block", "")).lower()
        model_type = block if typ in ["gnn", "gcn", "gat"] and block in ["gcn", "gat"] else typ
        model_type = _mp_slugify(model_type, default="model")
        if data is None:
            data_desc = "data-default"
        else:
            data_desc = _mp_data_descriptor(data, model_obj=model_obj)

        if data is not None and getattr(data, "path", None) == 0:
            return os.path.join("models", "Akash", data_desc, model_type)
        ut = bool(getattr(data, "UTmechTest", False))
        ft = bool(getattr(data, "FTmechTest", False))
        task = "multi" if (ut and ft) else ("ut" if ut else ("ft" if ft else "other"))
        return os.path.join("models", task, data_desc, model_type)
    path = str(path)
    if os.path.isabs(path):
        return path
    if path.startswith("models") or path.startswith("models/") or path.startswith("models\\"):
        return path
    return os.path.join("models", path)

def _model_refresh_descriptor(model_obj, path=None, name=None):
    metrics = {}
    for key in (
        "UT_best_loss", "UT_best_mse", "UT_best_rmse", "UT_best_epoch",
        "FT_best_loss", "FT_best_mse", "FT_best_rmse", "FT_best_epoch",
        "UT_prediction_summary", "FT_prediction_summary",
    ):
        if hasattr(model_obj, key):
            metrics[key] = _mp_to_serializable(getattr(model_obj, key))

    model_signature = _mp_collect_model_signature(model_obj)
    data_signature = _mp_collect_data_signature(model_obj)
    coarse_signature = {"model_signature": model_signature, "data_signature": data_signature}
    exact_signature = _mp_build_setup_signature(model_obj, include_data_values=True)
    model_obj.descriptor = {
        "version": 3,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "path": path,
        "name": name,
        "model_signature": model_signature,
        "data_signature": data_signature,
        "coarse_setup_signature": coarse_signature,
        "coarse_setup_hash": _mp_json_sha1(coarse_signature),
        "exact_setup_signature": exact_signature,
        "exact_setup_hash": _mp_json_sha1(exact_signature),
        "metrics": metrics,
    }
    model_obj.descriptor["descriptor_created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    return model_obj.descriptor

def _model_scan_matching_checkpoints(model_obj, search_root="models", recursive=True, print_results=True):
    root_dir = Path(search_root)
    matches = {"strict": [], "coarse": []}
    if not root_dir.exists():
        if print_results:
            print(f"[MODEL scan] Search root does not exist: {root_dir}")
        return matches

    strict_hash = _mp_json_sha1(_mp_build_setup_signature(model_obj, include_data_values=True))
    coarse_signature = {
        "model_signature": _mp_collect_model_signature(model_obj),
        "data_signature": _mp_collect_data_signature(model_obj),
    }
    coarse_hash = _mp_json_sha1(coarse_signature)

    json_iter = root_dir.rglob("*.json") if recursive else root_dir.glob("*.json")
    for meta_file in json_iter:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue

        strict_saved_hash = metadata.get("exact_setup_hash", None)
        if strict_saved_hash is None and metadata.get("exact_setup_signature", None) is not None:
            strict_saved_hash = _mp_json_sha1(metadata["exact_setup_signature"])

        coarse_saved_hash = metadata.get("coarse_setup_hash", None)
        if coarse_saved_hash is None and metadata.get("coarse_setup_signature", None) is not None:
            coarse_saved_hash = _mp_json_sha1(metadata["coarse_setup_signature"])

        strict_hit = strict_saved_hash == strict_hash
        coarse_hit = coarse_saved_hash == coarse_hash
        if not strict_hit and not coarse_hit:
            continue

        model_file = meta_file.with_suffix(".mdl")
        if not model_file.exists():
            continue

        rec = str(model_file)
        if strict_hit:
            matches["strict"].append(rec)
        elif coarse_hit:
            matches["coarse"].append(rec)

    if print_results:
        print(f"[MODEL scan] strict matches: {len(matches['strict'])}")
        for p in matches["strict"]:
            print(f"  - {p}")
        print(f"[MODEL scan] coarse-only matches: {len(matches['coarse'])}")
        for p in matches["coarse"]:
            print(f"  - {p}")

    return matches

def _model_find_matching_checkpoint(model_obj, path=None, strict=True, recursive=True):
    search_roots = [Path(_mp_resolve_model_dir(model_obj, path))]
    if path is None:
        search_roots.append(Path("models"))

    matches = {"strict": [], "coarse": []}
    seen_roots = set()
    for model_dir in search_roots:
        root_key = str(model_dir.resolve()) if model_dir.exists() else str(model_dir)
        if root_key in seen_roots or not model_dir.exists():
            continue
        seen_roots.add(root_key)
        root_matches = _model_scan_matching_checkpoints(
            model_obj,
            search_root=str(model_dir),
            recursive=recursive,
            print_results=False,
        )
        matches["strict"].extend(root_matches["strict"])
        matches["coarse"].extend(root_matches["coarse"])

    candidates = matches["strict"] if strict else (matches["strict"] + matches["coarse"])
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return candidates[0]

def _mp_loss_filename_token(model_obj):
    losses = []
    if hasattr(model_obj, "losses"):
        for lf in model_obj.losses:
            losses.append(getattr(lf, "__class__", type(lf)).__name__)
    losses = list(dict.fromkeys(losses))
    return f"loss-{_mp_slugify('-'.join(losses) if losses else 'None', max_len=48, preserve_case=True)}"

def _mp_early_stop_filename_token(model_obj):
    es = getattr(model_obj, "earlyStop", None)
    if es is None:
        return "earlyStop-None"
    patience = getattr(es, "patience", None)
    if patience is not None:
        return f"earlyStop-patience{_mp_slugify(patience, max_len=16, preserve_case=True)}"
    return f"earlyStop-{_mp_slugify(es.__class__.__name__, max_len=24, preserve_case=True)}"

def _mp_w_init_filename_token(model_obj):
    w_init = getattr(model_obj, "w_init", None)
    if not w_init:
        return "wInit-None"
    raw = "Auto" if isinstance(w_init, str) and w_init.lower() == "auto" else w_init
    return f"wInit-{_mp_slugify(raw, max_len=24, preserve_case=True)}"

def _mp_scheduler_filename_token(model_obj):
    sch_cfg = getattr(model_obj, "scheduler_cfg", None)
    if sch_cfg is None or sch_cfg is False:
        return "sch-None"
    if isinstance(sch_cfg, (list, tuple)) and len(sch_cfg) >= 3:
        sch_name = str(sch_cfg[0]).strip().lower()
        if sch_name in ["plateau", "reduce", "reduce_on_plateau", "reducelronplateau"] and len(sch_cfg) >= 5:
            mode_raw, factor_raw, patience_raw, threshold_raw = sch_cfg[1], sch_cfg[2], sch_cfg[3], sch_cfg[4]
        else:
            mode_raw, factor_raw, patience_raw = sch_cfg[0], sch_cfg[1], sch_cfg[2]
            threshold_raw = sch_cfg[3] if len(sch_cfg) >= 4 else None
        parts = [
            "sch",
            _mp_slugify(str(mode_raw).strip().lower(), max_len=16),
            _mp_slugify(_mp_format_filename_num(factor_raw), max_len=16),
            _mp_slugify(patience_raw, max_len=16, preserve_case=True),
        ]
        if threshold_raw is not None:
            parts.append(_mp_slugify(_mp_format_filename_num(threshold_raw), max_len=16))
        return "-".join(parts)
    return "sch-Custom"

def _model_save_checkpoint(model_obj, path=None, name=None):
    path = _mp_resolve_model_dir(model_obj, path)
    os.makedirs(path, exist_ok=True)
    if name is None:
        lr_tok = f"lr-{_mp_slugify(_mp_format_filename_num(model_obj.lr), max_len=16)}"
        b_tok = f"batch-{_mp_slugify(model_obj.batch, max_len=16)}"
        weight_decay = _mp_weight_decay(model_obj)
        wd_tok = f"wd-{_mp_slugify(_mp_format_filename_num(weight_decay), default='na', max_len=16)}"
        opt_name = "opt-None"
        ut_opt = getattr(model_obj, "UT_opt", None)
        ft_opt = getattr(model_obj, "FT_opt", None)
        primary_opt_name = ut_opt.__class__.__name__ if ut_opt is not None else (ft_opt.__class__.__name__ if ft_opt is not None else None)
        if primary_opt_name is not None:
            opt_name = f"opt-{_mp_slugify(primary_opt_name, max_len=12, preserve_case=True)}"
        sch_tok = _mp_scheduler_filename_token(model_obj)
        es_tok = _mp_early_stop_filename_token(model_obj)
        wi_tok = _mp_w_init_filename_token(model_obj)
        loss_tok = _mp_loss_filename_token(model_obj)
        cfg = {"model": _mp_collect_model_signature(model_obj), "data": _mp_collect_data_signature(model_obj)}
        cfg_hash = _mp_json_sha1(cfg)[:8]
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [f"date-{ts}", opt_name, lr_tok, b_tok, wd_tok, loss_tok, es_tok, wi_tok, sch_tok, f"setupHash-{cfg_hash}"]
        name = "_".join(parts)
    name = str(name)
    if name.endswith(".mdl"):
        name = name[:-4]

    metadata = _model_refresh_descriptor(model_obj, path=path, name=name)
    payload = {"version": 2}
    if hasattr(model_obj, "UT_model"):
        payload["UT_model_state_dict"] = model_obj.UT_model.state_dict()
    if hasattr(model_obj, "FT_model"):
        payload["FT_model_state_dict"] = model_obj.FT_model.state_dict()
    if hasattr(model_obj, "UT_model"):
        payload["model_state_dict"] = model_obj.UT_model.state_dict()
    elif hasattr(model_obj, "FT_model"):
        payload["model_state_dict"] = model_obj.FT_model.state_dict()
    else:
        payload["model_state_dict"] = model_obj.model.state_dict()
    payload["metadata"] = metadata
    model_file = os.path.join(path, f"{name}.mdl")
    meta_file = os.path.join(path, f"{name}.json")
    torch.save(payload, model_file)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return model_file

def _model_load_checkpoint(
    model_obj,
    model_path=None,
    match_current=False,
    strict_match=True,
    fallback_to_latest=False,
    require_descriptor_json=True,
):
    # Exact-file mode: when model_path points to a .mdl file, load that file only.
    if model_path is not None and str(model_path).lower().endswith(".mdl"):
        model_file = Path(model_path)
    else:
        model_dir = Path(_mp_resolve_model_dir(model_obj, model_path))
        if match_current:
            # With no explicit directory, match search also scans models/ for older naming layouts.
            matched = _model_find_matching_checkpoint(
                model_obj,
                path=model_path,
                strict=strict_match,
                recursive=True,
            )
            if matched is not None:
                model_file = Path(matched)
            elif fallback_to_latest:
                # Fallback is constrained to the provided/default lowest-level descriptor directory only.
                candidates = sorted(model_dir.glob("*.mdl"), key=lambda p: p.stat().st_mtime, reverse=True)
                if len(candidates) == 0:
                    raise FileNotFoundError(
                        f"No model checkpoints found in '{model_dir}'. "
                        f"No strict match found for current setup either."
                    )
                model_file = candidates[0]
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for the current setup in '{model_dir}'. "
                    f"Use strict_match=False for coarse matching or fallback_to_latest=True."
                )
        else:
            candidates = sorted(model_dir.glob("*.mdl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(candidates) == 0:
                raise FileNotFoundError(
                    f"No model checkpoints found in '{model_dir}'. Pass a full .mdl file path via model_path "
                    "or verify the directory."
                )
            model_file = candidates[0]

    if not model_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: '{model_file}'")

    meta_file = model_file.with_suffix(".json")
    loaded_descriptor = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            loaded_descriptor = json.load(f)
    elif require_descriptor_json:
        raise FileNotFoundError(
            f"Descriptor JSON not found for checkpoint '{model_file}'. "
            f"Expected: '{meta_file}'."
        )

    state = torch.load(str(model_file), map_location=model_obj.device)
    if isinstance(state, dict) and (
        "UT_model_state_dict" in state
        or "FT_model_state_dict" in state
        or "model_state_dict" in state
    ):
        if "UT_model_state_dict" in state and hasattr(model_obj, "UT_model"):
            model_obj.UT_model.load_state_dict(state["UT_model_state_dict"])
        if "FT_model_state_dict" in state and hasattr(model_obj, "FT_model"):
            model_obj.FT_model.load_state_dict(state["FT_model_state_dict"])

        if "model_state_dict" in state:
            model_obj.model.load_state_dict(state["model_state_dict"])
        elif "UT_model_state_dict" in state:
            model_obj.model.load_state_dict(state["UT_model_state_dict"])
        elif "FT_model_state_dict" in state:
            model_obj.model.load_state_dict(state["FT_model_state_dict"])
    else:
        model_obj.model.load_state_dict(state)
        if hasattr(model_obj, "UT_model"):
            model_obj.UT_model.load_state_dict(state)
        if hasattr(model_obj, "FT_model"):
            model_obj.FT_model.load_state_dict(state)

    if isinstance(loaded_descriptor, dict):
        model_obj.descriptor = loaded_descriptor
    else:
        _model_refresh_descriptor(model_obj, path=str(model_file.parent), name=model_file.stem)
    return str(model_file)

# ML model classes helpers
def _model_make_tensor_dataset(x, y):
    x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)

def _resolve_dropout(dropout, fallback=0.0):
    return dropout if dropout is not None else fallback

def _resolve_norm(norm):
    if norm is None or norm is False:
        return None
    if isinstance(norm, str):
        norm = norm.lower()
        if norm in ["none", "off", "false"]:
            return None
    if norm not in ["layer", "batch"]:
        raise ValueError("norm must be one of None, 'layer', or 'batch'.")
    return norm

def _make_norm(norm, size):
    norm = _resolve_norm(norm)
    if norm == "layer":
        return nn.LayerNorm(size)
    if norm == "batch":
        return nn.BatchNorm1d(size)
    return None
