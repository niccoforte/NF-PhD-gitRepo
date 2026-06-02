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


def parse_components(value):
    if value is None:
        return ("U1", "U2")
    parts = str(value).replace(",", " ").split()
    parts = tuple(part.strip() for part in parts if part.strip())
    if len(parts) == 0:
        raise ValueError("--components cannot be empty.")
    return parts


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


def write_json(path, payload):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def normalize_typ(value):
    value = str(value).strip().lower()
    aliases = {
        "gcn": "gcn",
        "gat": "gat",
        "gnn": "gcn",
        "tr": "tr",
        "transformer": "tr",
    }
    if value not in aliases:
        raise ValueError("Field HPO models must be GCN, GAT, GNN, TR, or Transformer. MLP is not field-compatible.")
    return aliases[value]


def parse_models(value):
    parts = str(value).replace(",", " ").split()
    models = []
    for part in parts:
        typ = normalize_typ(part)
        if typ not in models:
            models.append(typ)
    if not models:
        raise ValueError("--models cannot be empty.")
    return models


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full cross-model GPU HPO for nodal displacement field outputs."
    )
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials-per-typ", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=0.0, help="Per-model Optuna timeout. 0 means no timeout.")
    parser.add_argument("--nsims", default="all", help="Number of simulations, or 'all'.")
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT", "MULTI"])
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--components", default="U1,U2", help="Field components, e.g. 'U1,U2', 'U2', or 'Umag'.")
    parser.add_argument("--keep-frame0", action="store_true", help="Keep the unloaded frame 0 in the target.")
    parser.add_argument("--models", default="GCN,GAT,TR")
    parser.add_argument("--study-name", default="Field-CrossModelHPO")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA for local/debug runs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable Optuna progress bars.")
    return parser.parse_args()


def build_field_data(DATA, args, typ, nsims, field_config):
    model_token = "TR" if typ == "tr" else typ.upper()
    return DATA(
        path=args.data_path,
        path_add="",
        load=True,
        load_split=False,
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
        model=model_token,
        output_kind="field",
        field_config=field_config,
        freq=False,
        scale=("symm", "inout"),
        reduce_dim=False,
        round_decimals=args.round_decimals,
        geom_feats=(True, True),
    )


def field_model_space():
    return {
        "gcn": {
            "depth": [2, 3, 4, 5, 6],
            "width": [32, 64, 128, 256, 384, 512],
            "act": ["relu", "gelu", "mish"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.35},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.30},
        },
        "gat": {
            "depth": [1, 2, 3, 4, 5],
            "width": [16, 32, 64, 128, 192, 256],
            "heads": [1, 2, 4, 8],
            "act": ["relu", "gelu", "mish"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.35},
            "att_dropout": {"type": "float", "low": 0.0, "high": 0.35},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.30},
        },
        "tr": {
            "d_model": [64, 128, 192, 256, 384],
            "n_heads": [1, 2, 4, 8],
            "n_layers": [2, 3, 4, 5, 6],
            "ff_mult": [2, 4, 6, 8],
            "head_depth": [0, 1, 2, 3],
            "head_width": [64, 128, 256, 512],
            "use_cls_token": [False, True],
            "act": ["relu", "gelu", "mish"],
            "encoder_act": ["gelu", "relu"],
            "block": ["mlp", "res"],
            "norm": ["layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.35},
            "att_dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "head_norm": ["same", None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "pos_encoding": ["learned", "sinusoidal"],
        },
    }


def field_loss_space():
    return {
        "family": ["field_mse"],
        "reduction": ["mean"],
        "eps": {"type": "fixed", "value": 1e-12},
    }


def field_train_space():
    common = {
        "optimizer": ["adamw"],
        "scheduler": ["plateau"],
        "scheduler_factor": [0.2, 0.3, 0.5, 0.7],
        "scheduler_patience": [12, 20, 35],
        "scheduler_threshold": {"type": "fixed", "value": 1e-4},
        "early_stop": [True],
        "early_stop_patience": [50, 75, 100],
        "early_stop_min_delta": {"type": "fixed", "value": 1e-5},
        "metric": ["rmse"],
        "n_epochs": {"type": "fixed", "value": 450},
        "verbose": {"type": "fixed", "value": 0},
    }
    return {
        "gcn": {
            **common,
            "lr": {"type": "float", "low": 3e-6, "high": 2e-3, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [1, 2, 4, 8],
        },
        "gat": {
            **common,
            "lr": {"type": "float", "low": 3e-6, "high": 2e-3, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [1, 2, 4, 8],
        },
        "tr": {
            **common,
            "lr": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [1, 2, 4],
        },
    }


def main():
    args = parse_args()
    models = parse_models(args.models)
    nsims = parse_nsims(args.nsims)
    timeout = None if args.timeout_hours <= 0 else int(args.timeout_hours * 3600)
    components = parse_components(args.components)
    field_config = {
        "components": components,
        "drop_frame0": not args.keep_frame0,
        "layout": "auto",
    }

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
    run_config.update({
        "models_resolved": models,
        "nsims_resolved": nsims,
        "timeout_seconds_per_model": timeout,
        "output_kind": "field",
        "components_resolved": components,
        "field_config": field_config,
    })
    write_json(os.environ.get("ML_RUN_METADATA"), {"script": Path(__file__).name, "run_config": run_config})

    data_by_typ = {}
    for typ in models:
        data_by_typ[typ] = build_field_data(DATA, args, typ, nsims, field_config)
        print(f"{typ.upper()} split sizes: {split_size_summary(data_by_typ[typ])}")
        for mode in ("UT", "FT"):
            if getattr(data_by_typ[typ], f"{mode}mechTest", False):
                print(f"{typ.upper()} {mode} field shape: {getattr(data_by_typ[typ], f'{mode}_field_shape')}")
                print(f"{typ.upper()} {mode} field components: {getattr(data_by_typ[typ], f'{mode}_field_components')}")

    studies = hOpt_compare(
        typs=models,
        data=data_by_typ,
        n_trials_per_typ=args.n_trials_per_typ,
        model_space=field_model_space(),
        loss_space=field_loss_space(),
        train_space=field_train_space(),
        seed=args.seed,
        device=device,
        save=True,
        save_best_model=True,
        name=args.study_name,
        n_jobs=args.n_jobs,
        timeout=timeout,
        show_progress_bar=not args.no_progress,
    )

    summary = hOpt_best_summary(studies)
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
