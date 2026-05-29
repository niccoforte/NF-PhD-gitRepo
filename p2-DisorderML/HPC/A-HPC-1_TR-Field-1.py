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


def parse_components(value):
    if value is None:
        return ("U1", "U2")
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).replace(",", " ").split()
    parts = tuple(p.strip() for p in parts if str(p).strip())
    if len(parts) == 0:
        raise ValueError("--components cannot be empty.")
    return parts


def add_optional_bool_pair(parser, name, default=None, help_on=None, help_off=None):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_on)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false", help=help_off)
    parser.set_defaults(**{name.replace("-", "_"): default})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU Transformer run for UT displacement field outputs."
    )
    parser.add_argument("--run-label", default=os.environ.get("RUN_LABEL", "TR-Field-UT-1"))
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsims", default="all", help="Number of simulations, or 'all'.")
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--components", default="U1,U2", help="Field components, e.g. 'U1,U2' or 'U2'.")
    parser.add_argument("--keep-frame0", action="store_true", help="Keep the unloaded frame 0 in the target.")

    add_optional_bool_pair(
        parser,
        "range-split",
        default=True,
        help_on="Force range-covering input samples into training.",
        help_off="Use a purely random split.",
    )
    add_optional_bool_pair(
        parser,
        "geom-feats",
        default=True,
        help_on="Include x0/y0 and boundary flags in node features.",
        help_off="Use displacement-only node features.",
    )
    add_optional_bool_pair(
        parser,
        "coord-norm",
        default=True,
        help_on="Normalize geometric coordinates when geometry features are enabled.",
        help_off="Use physical geometric coordinates when geometry features are enabled.",
    )
    add_optional_bool_pair(
        parser,
        "scale-targets",
        default=True,
        help_on="Symmetrically scale both inputs and field outputs.",
        help_off="Scale inputs only.",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--act", default="gelu")
    parser.add_argument("--encoder-act", default="gelu")
    parser.add_argument("--block", default="mlp", choices=["mlp", "res"])
    parser.add_argument("--norm", default="layer")
    parser.add_argument("--head-norm", default="layer")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--att-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--pos-encoding", default="learned", choices=["learned", "sinusoidal"])
    parser.add_argument("--use-cls-token", action="store_true", help="Add a CLS token. Node pooling ignores it.")

    parser.add_argument("--loss", default="masked", choices=["auto", "masked", "mse"])
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=60)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--diag-samples", type=int, default=64)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA for local/debug runs.")
    return parser.parse_args()


def split_size_summary(data):
    return {
        "UT": {
            "train": int(len(data.UT_train_in)),
            "val": int(len(data.UT_val_in)),
            "test": int(len(data.UT_test_in)),
        }
    }


def has_invalid_targets(data):
    arrays = [data.UT_train_out, data.UT_val_out, data.UT_test_out]
    return any(not np.isfinite(arr).all() for arr in arrays)


def build_loss(args, data, nn, MaskedFieldMSELoss):
    invalid_targets = has_invalid_targets(data)
    if args.loss in ["auto", "masked"] and invalid_targets:
        print("Using MaskedFieldMSELoss because field targets contain invalid/masked NaN entries.")
        return MaskedFieldMSELoss(reduction="mean")
    if args.loss == "masked":
        print("Using MaskedFieldMSELoss.")
        return MaskedFieldMSELoss(reduction="mean")
    if args.loss == "auto":
        print("Using standard MSELoss because all field targets are finite.")
        return nn.MSELoss(reduction="mean")
    if invalid_targets:
        raise ValueError("Standard MSELoss cannot be used because field targets contain NaNs. Use --loss masked.")
    return nn.MSELoss(reduction="mean")


def main():
    args = parse_args()
    nsims = parse_nsims(args.nsims)
    components = parse_components(args.components)

    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
    if args.head_layers < 0:
        raise ValueError("--head-layers must be >= 0.")
    if args.block == "res" and args.head_layers > 0:
        print("For block='res', all head widths are kept identical.")

    geom_enabled = bool(args.geom_feats)
    coord_norm = bool(args.coord_norm) if geom_enabled else False
    scale_target = "inout" if args.scale_targets else "in"

    print("Importing torch...")
    import torch
    import torch.nn as nn

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import EarlyStopping, MaskedFieldMSELoss
    from resources.MLmodels import MODEL, Transformer
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

    field_config = {
        "components": components,
        "drop_frame0": not args.keep_frame0,
        "layout": "auto",
    }
    run_config = vars(args).copy()
    run_config.update({
        "task": "UT",
        "model": "TR",
        "output_kind": "field",
        "nsims_resolved": nsims,
        "components_resolved": components,
        "field_config": field_config,
        "geom_feats_resolved": [geom_enabled, coord_norm],
        "scale": ["symm", scale_target],
    })
    metadata = {
        "script": Path(__file__).name,
        "run_label": args.run_label,
        "data_path": args.data_path,
        "run_config": run_config,
    }
    write_json(os.environ.get("ML_RUN_METADATA"), metadata)

    print("Loading UT field DATA...")
    data = DATA(
        path=args.data_path,
        path_add="",
        load=True,
        load_split=False,
        split_frac=args.split_frac,
        split_seed=args.seed,
        range_split=(args.range_split, False),
        save_split=False,
        LAT=args.lat,
        dis=args.dis,
        dN=args.dN,
        d_data=args.d_data,
        mechMode="UT",
        nsims=nsims,
        model="TR",
        output_kind="field",
        field_config=field_config,
        freq=False,
        scale=("symm", scale_target),
        reduce_dim=False,
        round_decimals=args.round_decimals,
        geom_feats=(geom_enabled, coord_norm),
    )

    split_sizes = split_size_summary(data)
    print(f"Split sizes: {split_sizes}")
    print(f"UT train input shape: {data.UT_train_in.shape}")
    print(f"UT train output shape: {data.UT_train_out.shape}")
    print(f"UT field shape: {data.UT_field_shape}")
    print(f"UT field components: {data.UT_field_components}")

    n_frames, n_nodes, n_components = data.UT_field_shape
    in_nodes = int(data.UT_train_in.shape[-2])
    out_size = int(data.UT_train_out.shape[-1])
    expected_out_size = int(n_frames * n_components)
    if in_nodes != int(n_nodes):
        raise ValueError(f"UT input node count ({in_nodes}) does not match field node count ({n_nodes}).")
    if out_size != expected_out_size:
        raise ValueError(f"UT output width ({out_size}) does not match frames*components ({expected_out_size}).")

    in_size = int(data.UT_train_in.shape[-1])
    seq_len = int(data.UT_train_in.shape[-2])
    h_size = [args.head_width for _ in range(args.head_layers)]

    print(f"Transformer config: seq_len={seq_len}, in_size={in_size}, out_size={out_size}")
    inner_model = Transformer(
        in_size=in_size,
        seq_len=seq_len,
        h_size=h_size,
        out_size=out_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        act=args.act,
        encoder_act=args.encoder_act,
        block=args.block,
        norm=parse_optional_norm(args.norm),
        dropout=args.dropout,
        att_dropout=args.att_dropout,
        head_norm=parse_optional_norm(args.head_norm),
        head_dropout=args.head_dropout,
        pool="node",
        use_cls_token=bool(args.use_cls_token),
        pos_encoding=args.pos_encoding,
    ).to(device)
    print(inner_model)

    lossf = build_loss(args, data, nn, MaskedFieldMSELoss)

    model = MODEL(
        typ=data.model,
        model=inner_model,
        lossf=lossf,
        opt=("adamw", args.weight_decay),
        batch=args.batch,
        lr=args.lr,
        data=data,
        mechMode=data.mechMode,
        scheduler=("plateau", "min", args.scheduler_factor, args.scheduler_patience, args.scheduler_threshold),
        earlyStop=EarlyStopping(
            patience=args.early_stop_patience,
            min_delta=args.early_stop_delta,
            verbose=True,
        ),
        w_init="auto",
        device=device,
        optTrial=None,
        scan_matches_on_init=False,
    )

    print("Starting training...")
    model.train(n_epochs=args.epochs, verbose=args.verbose, plot=False)

    eval_split = "test" if split_sizes["UT"]["test"] > 0 else "val"
    if eval_split != "test":
        print("Test split is empty; evaluating the validation split instead.")
    model.evaluate_split(eval_split, diagnostics=True, diag_plot=False, diag_samples=args.diag_samples)

    checkpoint = model.save(path=None, name=args.run_label)
    results_dir = model.save_results(
        run_config=run_config,
        eval_split=eval_split,
        metadata=metadata,
    )

    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved trial results in: {results_dir}")


if __name__ == "__main__":
    main()
