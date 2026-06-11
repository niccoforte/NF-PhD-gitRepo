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


def parse_components(value):
    if value is None:
        return ("U1", "U2")
    parts = str(value).replace(",", " ").split()
    parts = tuple(part.strip() for part in parts if part.strip())
    if len(parts) == 0:
        raise ValueError("--components cannot be empty.")
    return parts


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


def context_label(label):
    label = str(label or "").strip()
    return label if label else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU field-to-curve Transformer smoke test."
    )
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT"])
    parser.add_argument("--model-type", default="TR", choices=["TR", "Transformer", "tr", "transformer"])
    parser.add_argument("--run-label", default="", help="Optional run descriptor. Empty uses the framework timestamp.")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=0, help="0 chooses a Transformer-friendly default.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nsims", default="64", help="Number of simulations, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--components", default="U1,U2", help="Field input components, e.g. 'U1,U2' or 'Umag'.")
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--keep-frame0", action="store_true", help="Keep the unloaded field frame 0 in the input.")

    add_optional_bool_pair(
        parser,
        "range-split",
        default=True,
        help_on="Force range-covering input/property samples into training.",
        help_off="Use a purely random split.",
    )
    add_optional_bool_pair(
        parser,
        "geom-feats",
        default=True,
        help_on="Append x0/y0 and boundary flags to field node tokens.",
        help_off="Use displacement-history token features only.",
    )
    add_optional_bool_pair(
        parser,
        "coord-norm",
        default=True,
        help_on="Normalize geometric token coordinates.",
        help_off="Use physical geometric token coordinates.",
    )

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--head-width", type=int, default=128)
    parser.add_argument("--act", default="gelu")
    parser.add_argument("--encoder-act", default="gelu")
    parser.add_argument("--norm", default="layer")
    parser.add_argument("--head-norm", default="layer")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--att-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--pos-encoding", default="learned", choices=["learned", "sinusoidal"])
    parser.add_argument("--use-cls-token", action="store_true")

    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-delta", type=float, default=1e-5)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA.")
    return parser.parse_args()


def split_size_summary(data):
    return {
        mode: {
            "train": int(len(getattr(data, f"{mode}_train_in"))),
            "val": int(len(getattr(data, f"{mode}_val_in"))),
            "test": int(len(getattr(data, f"{mode}_test_in"))),
        }
        for mode in ("UT", "FT")
        if getattr(data, f"{mode}mechTest", False)
    }


def main():
    args = parse_args()
    active_mode = args.task.upper()
    nsims = parse_nsims(args.nsims)
    components = parse_components(args.components)
    if args.pca_components < 1:
        raise ValueError("--pca-components must be >= 1.")
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")

    print("Importing torch...")
    import torch
    import torch.nn as nn

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import EarlyStopping
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

    batch = args.batch if args.batch > 0 else 4
    field_input_config = {
        "components": components,
        "drop_frame0": not args.keep_frame0,
        "layout": "auto",
    }
    reduce_dim = ("PCA", "out", None, int(args.pca_components), True)

    run_config = vars(args).copy()
    run_config.update({
        "task": active_mode,
        "model_type": "TR",
        "input_kind": "field",
        "output_kind": "curve",
        "output_layout": "FieldToCurve",
        "nsims_resolved": nsims,
        "batch_resolved": batch,
        "components_resolved": components,
        "field_input_config": field_input_config,
        "reduce_dim": reduce_dim,
        "scale": ["symm", "inout"],
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
        load_split=False,
        split_frac=args.split_frac,
        split_seed=args.seed,
        range_split=(args.range_split, False),
        save_split=False,
        LAT=args.lat,
        dis=args.dis,
        dN=args.dN,
        d_data=args.d_data,
        mechMode=active_mode,
        nsims=nsims,
        model="TR",
        input_kind="field",
        output_kind="curve",
        field_input_config=field_input_config,
        freq=False,
        scale=("symm", "inout"),
        reduce_dim=reduce_dim,
        round_decimals=args.round_decimals,
        geom_feats=(bool(args.geom_feats), bool(args.coord_norm) if args.geom_feats else False),
    )

    split_sizes = split_size_summary(data)
    print(f"Split sizes: {split_sizes}")
    train_in = getattr(data, f"{active_mode}_train_in")
    train_out = getattr(data, f"{active_mode}_train_out")
    print(f"{active_mode} field input shape: {getattr(data, f'{active_mode}_field_input_shape', None)}")
    print(f"{active_mode} field token shape per sample: {train_in.shape[1:]}")
    print(f"{active_mode} latent output size: {train_out.shape[-1]}")
    pca = getattr(data, f"{active_mode}_OUTreducer", None)
    if pca is not None and hasattr(pca, "explained_variance_ratio_"):
        print(f"{active_mode} PCA explained variance ({args.pca_components} components): {float(np.sum(pca.explained_variance_ratio_)):.6f}")

    in_shape = train_in.shape[1:]
    out_size = int(train_out.shape[-1])
    transformer = Transformer(
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
        norm=parse_optional_norm(args.norm),
        dropout=args.dropout,
        att_dropout=args.att_dropout,
        head_norm=parse_optional_norm(args.head_norm),
        head_dropout=args.head_dropout,
        pool="mean",
        use_cls_token=bool(args.use_cls_token),
        pos_encoding=args.pos_encoding,
    ).to(device)
    print(transformer)

    model = MODEL(
        typ=data.model,
        model=transformer,
        lossf=nn.MSELoss(reduction="mean"),
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

    checkpoint = model.save(path=None, name=context_label(args.run_label), save_results=False)
    results_dir = model.save_results(run_config=run_config, eval_split=eval_split, metadata=metadata)

    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved results in: {results_dir}")


if __name__ == "__main__":
    main()
