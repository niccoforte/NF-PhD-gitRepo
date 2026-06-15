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


def default_study_name(value):
    value = str(value or "").strip()
    if value:
        return value
    for env_name in ["ML_JOB_NAME", "SLURM_JOB_NAME"]:
        env_value = str(os.environ.get(env_name, "")).strip()
        if env_value:
            return env_value
    return "FieldToCurve-CrossModelHPO"


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
        raise ValueError("Field-to-curve HPO models must be GCN, GAT, GNN, TR, or Transformer. MLP is not used for this input contract.")
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
        description="Cross-model GPU HPO for field-input to curve-output models."
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
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT"])
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--components", default="U1,U2", help="Field input components, e.g. 'U1,U2', 'U2', or 'Umag'.")
    parser.add_argument("--keep-frame0", action="store_true", help="Keep the unloaded field frame 0 in the input.")
    parser.add_argument("--models", default="GCN,GAT,TR")
    parser.add_argument("--output-reduction", default="pca", choices=["pca", "none"], help="Use PCA curve latents or full ordered curve outputs.")
    parser.add_argument("--pca-components", type=int, default=None, help="Fixed PCA latent dimension. Defaults to 16 when --pca-accuracy is not set.")
    parser.add_argument("--pca-accuracy", type=float, default=None, help="Variance threshold for PCA component selection. Ignored when --pca-components is set.")
    parser.add_argument("--no-scale-reduced", action="store_true", help="Do not scale PCA latent curve outputs.")
    parser.add_argument("--study-name", default="", help="HPO study/folder name. Empty uses the Slurm job name from -J.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA for local/debug runs.")
    parser.add_argument("--no-progress", action="store_true", help="Disable Optuna progress bars.")
    return parser.parse_args()


def output_reduce_dim(args):
    output_reduction = str(args.output_reduction or "pca").strip().lower()
    if output_reduction in ["none", "false", "off", "full", "curve"]:
        return False
    if output_reduction != "pca":
        raise ValueError("--output-reduction must be 'pca' or 'none'.")
    if args.pca_components is not None and args.pca_components < 1:
        raise ValueError("--pca-components must be >= 1 when set.")
    if args.pca_accuracy is not None and not 0 < args.pca_accuracy <= 1:
        raise ValueError("--pca-accuracy must be in the range (0, 1].")
    n_components = 16 if args.pca_components is None and args.pca_accuracy is None else args.pca_components
    accuracy = args.pca_accuracy if n_components is None else None
    return ("PCA", "out", accuracy, n_components, not args.no_scale_reduced)


def pca_summary(data):
    rows = {}
    for mode in ("UT", "FT"):
        if not getattr(data, f"{mode}mechTest", False):
            continue
        reducer = getattr(data, f"{mode}_OUTreducer", None)
        if reducer is None:
            continue
        explained = getattr(reducer, "explained_variance_ratio_", None)
        rows[mode] = {
            "latent_dim": int(getattr(data, f"{mode}_train_out").shape[-1]),
            "explained_variance": float(np.sum(explained)) if explained is not None else None,
        }
    return rows


def build_field_to_curve_data(DATA, args, typ, nsims, field_input_config, reduce_dim):
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
        input_kind="field",
        output_kind="curve",
        field_input_config=field_input_config,
        freq=False,
        scale=("symm", "inout"),
        reduce_dim=reduce_dim,
        round_decimals=args.round_decimals,
        geom_feats=(True, True),
    )


def field_to_curve_model_space():
    return {
        "gcn": {
            "depth": [2, 3, 4, 5],
            "width": [64, 128, 192, 256, 384],
            "act": ["relu", "gelu", "mish"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "pool": ["mean", "add"],
        },
        "gat": {
            "depth": [1, 2, 3, 4],
            "width": [32, 64, 96, 128, 192],
            "heads": [1, 2, 4],
            "act": ["relu", "gelu", "mish"],
            "norm": [None, "layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "att_dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "head_norm": [None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "pool": ["mean", "add"],
        },
        "tr": {
            "d_model": [96, 128, 160, 192, 256],
            "n_heads": [2, 4, 8],
            "n_layers": [2, 3, 4, 5],
            "ff_mult": [2, 4, 6],
            "head_depth": [1, 2],
            "head_width": [96, 128, 192, 256],
            "pool": ["mean", "cls", "max"],
            "use_cls_token": [False, True],
            "act": ["relu", "gelu", "mish"],
            "encoder_act": ["gelu", "relu"],
            "block": ["mlp", "res"],
            "norm": ["layer"],
            "dropout": {"type": "float", "low": 0.0, "high": 0.30},
            "att_dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "head_norm": ["same", None, "layer"],
            "head_dropout": {"type": "float", "low": 0.0, "high": 0.25},
            "pos_encoding": ["learned", "sinusoidal"],
        },
    }


def field_to_curve_loss_space():
    return {
        "family": ["mse"],
    }


def field_to_curve_train_space():
    common = {
        "optimizer": ["adamw"],
        "scheduler": ["plateau"],
        "scheduler_factor": [0.3, 0.5, 0.7],
        "scheduler_patience": [8, 12, 20],
        "scheduler_threshold": {"type": "fixed", "value": 1e-4},
        "early_stop": [True],
        "early_stop_patience": [25, 35, 50],
        "early_stop_min_delta": {"type": "fixed", "value": 1e-5},
        "metric": ["rmse"],
        "n_epochs": {"type": "fixed", "value": 300},
        "verbose": {"type": "fixed", "value": 0},
    }
    return {
        "gcn": {
            **common,
            "lr": {"type": "float", "low": 5e-6, "high": 1e-3, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [4, 8, 16],
        },
        "gat": {
            **common,
            "lr": {"type": "float", "low": 5e-6, "high": 1e-3, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [2, 4, 8],
        },
        "tr": {
            **common,
            "lr": {"type": "float", "low": 3e-6, "high": 8e-4, "log": True},
            "weight_decay": {"type": "float", "low": 1e-9, "high": 3e-3, "log": True},
            "batch": [2, 4, 8],
        },
    }


def run_exact_named_hpo(hOpt_compare, **kwargs):
    old_context = os.environ.get("ML_RUN_CONTEXT")
    if str(old_context or "").strip().lower() == "hpc":
        os.environ["ML_RUN_CONTEXT"] = ""
    try:
        return hOpt_compare(**kwargs)
    finally:
        if old_context is None:
            os.environ.pop("ML_RUN_CONTEXT", None)
        else:
            os.environ["ML_RUN_CONTEXT"] = old_context


def main():
    args = parse_args()
    args.study_name = default_study_name(args.study_name)
    models = parse_models(args.models)
    nsims = parse_nsims(args.nsims)
    timeout = None if args.timeout_hours <= 0 else int(args.timeout_hours * 3600)
    components = parse_components(args.components)
    field_input_config = {
        "components": components,
        "drop_frame0": not args.keep_frame0,
        "layout": "auto",
    }
    reduce_dim = output_reduce_dim(args)

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
        "input_kind": "field",
        "output_kind": "curve",
        "output_layout": "FieldToCurve",
        "components_resolved": components,
        "field_input_config": field_input_config,
        "reduce_dim": reduce_dim,
        "scale": ["symm", "inout"],
        "hpo_study_name_is_exact": True,
    })
    write_json(os.environ.get("ML_RUN_METADATA"), {"script": Path(__file__).name, "run_config": run_config})

    data_by_typ = {}
    for typ in models:
        data_by_typ[typ] = build_field_to_curve_data(DATA, args, typ, nsims, field_input_config, reduce_dim)
        print(f"{typ.upper()} split sizes: {split_size_summary(data_by_typ[typ])}")
        for mode in ("UT", "FT"):
            if getattr(data_by_typ[typ], f"{mode}mechTest", False):
                train_in = getattr(data_by_typ[typ], f"{mode}_train_in")
                train_out = getattr(data_by_typ[typ], f"{mode}_train_out")
                print(f"{typ.upper()} {mode} field input shape: {getattr(data_by_typ[typ], f'{mode}_field_input_shape', None)}")
                print(f"{typ.upper()} {mode} token shape per sample: {train_in.shape[1:]}")
                if reduce_dim:
                    print(f"{typ.upper()} {mode} latent output size: {train_out.shape[-1]}")
                else:
                    print(f"{typ.upper()} {mode} output curve size: {train_out.shape[-1]}")
        if reduce_dim:
            print(f"{typ.upper()} PCA summary: {pca_summary(data_by_typ[typ])}")

    studies = run_exact_named_hpo(
        hOpt_compare,
        typs=models,
        data=data_by_typ,
        n_trials_per_typ=args.n_trials_per_typ,
        model_space=field_to_curve_model_space(),
        loss_space=field_to_curve_loss_space(),
        train_space=field_to_curve_train_space(),
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
