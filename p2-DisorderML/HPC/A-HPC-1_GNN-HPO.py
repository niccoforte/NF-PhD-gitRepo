#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_nsims(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ["", "all", "none", "null"]:
        return None
    return int(value)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_metadata(path, payload):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="HPC GPU HPO comparison run for UT GCN and GAT stress-strain models.")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split", default="", help="Saved split name without the 'split-' prefix.")
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials-per-typ", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--nsims", default="all", help="Number of simulations, or 'all'.")
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT", "MULTI"])
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--name", default="GNN_full_hOpt")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA for local debugging.")
    parser.add_argument("--no-progress", action="store_true", help="Disable Optuna progress bar.")
    return parser.parse_args()


def split_size_summary(data):
    split_sizes = {}
    for mode in ("UT", "FT"):
        if getattr(data, f"{mode}mechTest", False):
            split_sizes[mode] = {
                "train": int(len(getattr(data, f"{mode}_train_in"))),
                "val": int(len(getattr(data, f"{mode}_val_in"))),
                "test": int(len(getattr(data, f"{mode}_test_in"))),
            }
    return split_sizes


def main():
    args = parse_args()
    nsims = parse_nsims(args.nsims)

    print("Importing torch...")
    import torch

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import hOpt_best_summary, hOpt_compare
    print("Imports completed.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Use --allow-cpu only for local/debug runs.")

    run_config = vars(args).copy()
    run_config["nsims_resolved"] = nsims
    write_metadata(os.environ.get("ML_RUN_METADATA"), {"script": Path(__file__).name, "run_config": run_config})

    dat_gnn = DATA(
        path=args.data_path,
        path_add="",
        load=True,
        load_split=args.split or False,
        split_frac=args.split_frac,
        split_seed=args.seed,
        range_split=(True, False),
        save_split=False,
        LAT=args.lat,
        dis=args.dis,
        dN=args.dN,
        d_data=args.d_data,
        mechMode=args.task,
        nsims=nsims,
        model="GNN",
        scale=("symm", "inout"),
        reduce_dim=False,
        round_decimals=args.round_decimals,
        geom_feats=(True, True),
    )
    print(f"Split sizes: {split_size_summary(dat_gnn)}")

    gnn_model_space = {
        "gcn": {
            "depth": [2, 3, 4],
            "width": [64, 128, 256],
            "act": ["relu", "gelu", "mish"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.20},
            "pool": ["mean"],
        },
        "gat": {
            "depth": [1, 2, 3],
            "width": [32, 64, 128],
            "heads": [1, 2, 4],
            "act": ["relu", "gelu"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "att_dropout": {"type": "float", "low": 0.0, "high": 0.20},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.20},
            "pool": ["mean"],
        },
    }

    gnn_loss_space = {
        "family": ["mse", "combined"],
        "mse_weight": [0.25, 0.5, 1.0],
        "weighted_mse_weight": [0.0, 0.5, 1.0, 2.0],
        "derivative_weight": [0.0, 0.02, 0.05, 0.10],
        "peak_weight": [0.0, 0.10, 0.25, 0.50],
        "energy_weight": [0.0, 0.05, 0.10, 0.25],
        "peak_location_weight": [0.0, 0.02, 0.05],
        "reduction": ["mean"],
        "derivative_order": [1],
        "SoftPeak_beta": [10.0, 20.0, 40.0],
        "UT": {
            "zone_boundaries": (67, 134),
            "zone_weights": (1.0, 5.0, 2.0),
        },
    }

    gnn_train_space = {
        "optimizer": ["adamw"],
        "lr": {"type": "float", "low": 3e-5, "high": 1e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-8, "high": 1e-3, "log": True},
        "batch": [4, 8, 16],
        "n_epochs": {"type": "fixed", "value": 150},
        "metric": ["rmse"],
        "scheduler": ["plateau"],
        "scheduler_factor": [0.3, 0.5],
        "scheduler_patience": [8, 15],
        "scheduler_threshold": {"type": "fixed", "value": 1e-4},
        "early_stop": [True],
        "early_stop_patience": [25, 40],
        "early_stop_min_delta": {"type": "fixed", "value": 1e-5},
        "verbose": {"type": "fixed", "value": 0},
    }

    studies = hOpt_compare(
        typs=["gcn", "gat"],
        data=dat_gnn,
        n_trials_per_typ=args.n_trials_per_typ,
        model_space=gnn_model_space,
        loss_space=gnn_loss_space,
        train_space=gnn_train_space,
        seed=args.seed,
        device=device,
        save=True,
        save_best_model=True,
        name=args.name,
        n_jobs=args.n_jobs,
        show_progress_bar=not args.no_progress,
    )

    hOpt_best_summary(studies)


if __name__ == "__main__":
    main()
