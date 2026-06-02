#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np


def json_safe(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
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


def parse_nsims(value):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ["", "all", "none", "null"]:
        return None
    return int(value)


def parse_optional_norm(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ["", "none", "false", "0", "no"]:
        return None
    return value


def add_optional_bool_pair(parser, name, default=None, help_on=None, help_off=None):
    group = parser.add_mutually_exclusive_group()
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_on)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_off)
    parser.set_defaults(**{dest: default})


def canonical_model_type(value):
    value = str(value).strip().lower()
    aliases = {
        "mlp": "MLP",
        "gcn": "GCN",
        "gat": "GAT",
        "gnn": "GNN",
        "tr": "TR",
        "transformer": "TR",
    }
    if value not in aliases:
        raise ValueError("model type must be one of MLP, GCN, GAT, GNN, TR, Transformer.")
    return aliases[value]


def context_label(label):
    label = str(label or "").strip()
    if not label:
        return None
    if os.environ.get("ML_RUN_CONTEXT", "").strip().lower() == "hpc" and not label.lower().startswith("hpc-"):
        return f"HPC-{label}"
    return label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic single-GPU curve smoke test for MLP, GCN, GAT, and Transformer models."
    )
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT", "MULTI"])
    parser.add_argument("--model-type", default="GAT", help="MLP, GCN, GAT, GNN, TR, or Transformer.")
    parser.add_argument("--run-label", default="", help="Optional run descriptor. Empty uses the framework timestamp.")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split", default="", help="Saved curve split name without the 'split-' prefix.")
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=0, help="0 chooses a model-appropriate default.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nsims", default="128", help="Number of simulations, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--round-decimals", type=int, default=5)

    add_optional_bool_pair(
        parser,
        "range-split",
        default=True,
        help_on="Force range-covering samples into training.",
        help_off="Use a purely random split.",
    )
    add_optional_bool_pair(
        parser,
        "geom-feats",
        default=None,
        help_on="Include x0/y0 and boundary flags in node features.",
        help_off="Use displacement-only node features.",
    )
    add_optional_bool_pair(
        parser,
        "coord-norm",
        default=None,
        help_on="Normalize geometric coordinates when geometry features are enabled.",
        help_off="Use physical geometric coordinates when geometry features are enabled.",
    )

    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--act", default="gelu")
    parser.add_argument("--norm", default="layer")
    parser.add_argument("--head-norm", default="layer")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--att-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--pool", default="mean", choices=["mean", "add", "max", "cls"])

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--encoder-act", default="gelu")
    parser.add_argument("--pos-encoding", default="learned", choices=["learned", "sinusoidal"])
    add_optional_bool_pair(
        parser,
        "use-cls-token",
        default=None,
        help_on="Use a CLS token for Transformer runs.",
        help_off="Disable the Transformer CLS token.",
    )

    parser.add_argument("--loss", default="combined", choices=["mse", "combined"])
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--weighted-mse-weight", type=float, default=0.5)
    parser.add_argument("--derivative-weight", type=float, default=0.05)
    parser.add_argument("--peak-weight", type=float, default=0.25)
    parser.add_argument("--energy-weight", type=float, default=0.10)
    parser.add_argument("--peak-location-weight", type=float, default=0.02)
    parser.add_argument("--soft-peak-beta", type=float, default=20.0)
    parser.add_argument("--derivative-order", type=int, default=1)

    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA.")
    return parser.parse_args()


def active_modes(data):
    modes = []
    if getattr(data, "UTmechTest", False):
        modes.append("UT")
    if getattr(data, "FTmechTest", False):
        modes.append("FT")
    return modes


def primary_mode(data):
    return "UT" if getattr(data, "UTmechTest", False) else "FT"


def curve_loss_for_mode(data, mode, args, CombinedCurveLoss):
    return CombinedCurveLoss(
        mse_weight=args.mse_weight,
        weighted_mse_weight=args.weighted_mse_weight,
        derivative_weight=args.derivative_weight,
        peak_weight=args.peak_weight,
        energy_weight=args.energy_weight,
        peak_location_weight=args.peak_location_weight,
        x_values=getattr(data, f"{mode}_OUT_df"),
        reduction="mean",
        derivative_order=args.derivative_order,
        SoftPeak_beta=args.soft_peak_beta,
    )


def build_loss(data, args, nn, CombinedCurveLoss):
    if args.loss == "mse":
        return nn.MSELoss(reduction="mean")
    losses = {mode: curve_loss_for_mode(data, mode, args, CombinedCurveLoss) for mode in active_modes(data)}
    return next(iter(losses.values())) if len(losses) == 1 else losses


def build_inner_model(args, model_type, in_shape, out_size, device, MLP, GNN, Transformer):
    norm = parse_optional_norm(args.norm)
    head_norm = parse_optional_norm(args.head_norm)

    if model_type == "MLP":
        return MLP(
            in_size=int(in_shape[-1]),
            h_size=[args.hidden_size for _ in range(args.layers)],
            out_size=out_size,
            act=args.act,
            block="res",
            norm=norm,
            dropout=args.dropout,
            head_norm=head_norm,
            head_dropout=args.head_dropout,
        ).to(device)

    if model_type in ["GCN", "GAT", "GNN"]:
        block = "gat" if model_type == "GAT" else "gcn"
        return GNN(
            in_size=int(in_shape[-1]),
            h_size=[args.hidden_size for _ in range(args.layers)],
            out_size=out_size,
            act=args.act,
            block=block,
            norm=norm,
            dropout=args.dropout,
            att_dropout=args.att_dropout,
            head_norm=head_norm,
            head_dropout=args.head_dropout,
            bias=True,
            heads=args.heads if block == "gat" else 1,
            pool=args.pool if args.pool in ["mean", "add"] else "mean",
        ).to(device)

    if model_type == "TR":
        if args.d_model % args.n_heads != 0:
            raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
        pool = args.pool
        use_cls_token = args.use_cls_token
        if use_cls_token is None:
            use_cls_token = pool == "cls"
        if pool == "cls":
            use_cls_token = True
        if pool not in ["mean", "add", "max", "cls"]:
            pool = "mean"
        return Transformer(
            in_size=int(in_shape[-1]),
            seq_len=int(in_shape[-2]),
            h_size=[args.head_width for _ in range(args.head_layers)],
            out_size=out_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_mult=args.ff_mult,
            act=args.act,
            encoder_act=args.encoder_act,
            block="mlp",
            norm=norm,
            dropout=args.dropout,
            att_dropout=args.att_dropout,
            head_norm=head_norm,
            head_dropout=args.head_dropout,
            pool=pool,
            use_cls_token=bool(use_cls_token),
            pos_encoding=args.pos_encoding,
        ).to(device)

    raise ValueError(f"Unsupported model type: {model_type}")


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
    model_type = canonical_model_type(args.model_type)
    nsims = parse_nsims(args.nsims)
    node_model = model_type in ["GCN", "GAT", "GNN", "TR"]

    geom_enabled = args.geom_feats
    if geom_enabled is None:
        geom_enabled = node_model
    coord_norm = args.coord_norm
    if coord_norm is None:
        coord_norm = bool(geom_enabled)
    if not geom_enabled:
        coord_norm = False

    print("Importing torch...")
    import torch
    import torch.nn as nn

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import CombinedCurveLoss, EarlyStopping
    from resources.MLmodels import GNN, MODEL, MLP, Transformer
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

    batch = args.batch
    if batch <= 0:
        batch = 8 if model_type in ["GCN", "GAT", "GNN", "TR"] else 32

    run_config = vars(args).copy()
    run_config.update({
        "task": args.task,
        "model_type": model_type,
        "output_kind": "curve",
        "nsims_resolved": nsims,
        "batch_resolved": batch,
        "geom_feats_resolved": [bool(geom_enabled), bool(coord_norm)],
    })
    metadata = {
        "script": Path(__file__).name,
        "data_path": args.data_path,
        "run_config": run_config,
    }
    write_json(os.environ.get("ML_RUN_METADATA"), metadata)

    data = DATA(
        path=args.data_path,
        load=True,
        load_split=args.split or False,
        split_frac=args.split_frac,
        split_seed=args.seed,
        range_split=(args.range_split, False),
        save_split=False,
        LAT=args.lat,
        dis=args.dis,
        dN=args.dN,
        d_data=args.d_data,
        mechMode=args.task,
        nsims=nsims,
        model=model_type,
        output_kind="curve",
        freq=False,
        scale=("symm", "inout"),
        reduce_dim=False,
        round_decimals=args.round_decimals,
        geom_feats=(bool(geom_enabled), bool(coord_norm)),
    )

    split_sizes = split_size_summary(data)
    print(f"Split sizes: {split_sizes}")

    mode = primary_mode(data)
    in_shape = getattr(data, f"{mode}_train_in").shape[1:]
    out_size = int(getattr(data, f"{mode}_train_out").shape[-1])
    print(f"Input shape per sample: {in_shape}")
    print(f"Output curve size: {out_size}")

    inner_model = build_inner_model(args, model_type, in_shape, out_size, device, MLP, GNN, Transformer)
    print(inner_model)
    lossf = build_loss(data, args, nn, CombinedCurveLoss)

    model = MODEL(
        typ=data.model,
        model=inner_model,
        lossf=lossf,
        opt=("adamw", args.weight_decay),
        batch=batch,
        lr=args.lr,
        data=data,
        mechMode=data.mechMode,
        scheduler=("plateau", "min", args.scheduler_factor, args.scheduler_patience, args.scheduler_threshold),
        earlyStop=EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_delta, verbose=True),
        w_init="auto",
        device=device,
        optTrial=None,
        scan_matches_on_init=False,
    )

    model.train(n_epochs=args.epochs, verbose=args.verbose, plot=False)

    eval_split = "test"
    if any(sizes["test"] == 0 for sizes in split_sizes.values()):
        eval_split = "val"
        print("Test split is empty; evaluating the validation split instead.")
    model.evaluate_split(eval_split, diagnostics=True, diag_plot=False)

    checkpoint = model.save(path=None, name=context_label(args.run_label))
    results_dir = model.save_results(run_config=run_config, eval_split=eval_split, metadata=metadata)

    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved results in: {results_dir}")


if __name__ == "__main__":
    main()
