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


def resolve_loss_family(args, reduce_dim):
    requested = str(args.loss or "auto").strip().lower()
    if reduce_dim:
        if requested == "combined":
            raise ValueError("--loss combined requires --output-reduction none; curve terms are not meaningful on PCA latents.")
        return "mse"
    return "combined" if requested == "auto" else requested


def curve_x_values(data, mode):
    out_df = getattr(data, f"{mode}_OUT_df", None)
    if out_df is None or not hasattr(out_df, "iloc"):
        return None
    x_values = out_df.iloc[0]
    if hasattr(x_values, "drop") and "0" in list(x_values.index):
        x_values = x_values.drop(labels=["0"])
    return np.asarray(x_values, dtype=float)


def combined_curve_loss_params(data, mode, args, curve_default_zone_boundaries, curve_default_zone_weights):
    return {
        "mse_weight": args.mse_weight,
        "weighted_mse_weight": args.weighted_mse_weight,
        "derivative_weight": args.derivative_weight,
        "peak_weight": args.peak_weight,
        "energy_weight": args.energy_weight,
        "peak_location_weight": args.peak_location_weight,
        "zone_boundaries": curve_default_zone_boundaries(mode),
        "zone_weights": curve_default_zone_weights(),
        "x_values": curve_x_values(data, mode),
        "reduction": "mean",
        "derivative_order": args.derivative_order,
        "SoftPeak_beta": args.soft_peak_beta,
        "normalization_eps": args.loss_eps,
    }


def build_loss(data, mode, args, reduce_dim, nn, CombinedCurveLoss, curve_default_zone_boundaries, curve_default_zone_weights):
    family = resolve_loss_family(args, reduce_dim)
    if family == "mse":
        return nn.MSELoss(reduction="mean")
    if family != "combined":
        raise ValueError(f"Unsupported loss family: {family}")
    return CombinedCurveLoss(
        **combined_curve_loss_params(
            data,
            mode,
            args,
            curve_default_zone_boundaries,
            curve_default_zone_weights,
        )
    )


def loss_summary(args, mode, reduce_dim, curve_default_zone_boundaries, curve_default_zone_weights):
    family = resolve_loss_family(args, reduce_dim)
    summary = {"requested": args.loss, "family": family}
    if family == "combined":
        summary.update({
            "mse_weight": args.mse_weight,
            "weighted_mse_weight": args.weighted_mse_weight,
            "derivative_weight": args.derivative_weight,
            "peak_weight": args.peak_weight,
            "energy_weight": args.energy_weight,
            "peak_location_weight": args.peak_location_weight,
            "zone_boundaries": curve_default_zone_boundaries(mode),
            "zone_weights": curve_default_zone_weights(),
            "derivative_order": args.derivative_order,
            "SoftPeak_beta": args.soft_peak_beta,
            "normalization_eps": args.loss_eps,
        })
    return summary


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
        description="Single-GPU field-to-curve Transformer training run."
    )
    parser.add_argument("--task", type=str.upper, default="UT", choices=["UT", "FT"])
    parser.add_argument("--model-type", default="TR", choices=["TR", "Transformer", "tr", "transformer"])
    parser.add_argument("--run-label", default="", help="Optional run descriptor. Empty uses the framework timestamp.")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch", type=int, default=0, help="0 chooses a Transformer-friendly default.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--nsims", default="all", help="Number of simulations, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    parser.add_argument("--round-decimals", type=int, default=5)
    parser.add_argument("--components", default="U1,U2", help="Field input components, e.g. 'U1,U2' or 'Umag'.")
    parser.add_argument("--keep-frame0", action="store_true", help="Keep the unloaded field frame 0 in the input.")
    parser.add_argument("--output-reduction", default="pca", choices=["pca", "none"], help="Use PCA curve latents or full ordered curve outputs.")
    parser.add_argument("--pca-components", type=int, default=None, help="Fixed PCA latent dimension. Defaults to 16 when --pca-accuracy is not set.")
    parser.add_argument("--pca-accuracy", type=float, default=None, help="Variance threshold for PCA component selection. Ignored when --pca-components is set.")
    parser.add_argument("--no-scale-reduced", action="store_true", help="Do not scale PCA latent curve outputs.")

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

    parser.add_argument("--loss", default="auto", choices=["auto", "mse", "combined"], help="'auto' uses MSE for PCA latents and combined loss for full curves.")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--weighted-mse-weight", type=float, default=0.5)
    parser.add_argument("--derivative-weight", type=float, default=0.05)
    parser.add_argument("--peak-weight", type=float, default=0.25)
    parser.add_argument("--energy-weight", type=float, default=0.10)
    parser.add_argument("--peak-location-weight", type=float, default=0.02)
    parser.add_argument("--soft-peak-beta", type=float, default=20.0)
    parser.add_argument("--derivative-order", type=int, default=1)
    parser.add_argument("--loss-eps", type=float, default=1e-8)

    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-threshold", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=50)
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
    reduce_dim = output_reduce_dim(args)
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads}).")
    if args.derivative_order < 1:
        raise ValueError("--derivative-order must be >= 1.")
    if args.loss_eps <= 0:
        raise ValueError("--loss-eps must be positive.")
    resolve_loss_family(args, reduce_dim)

    print("Importing torch...")
    import torch
    import torch.nn as nn

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import (
        CombinedCurveLoss,
        EarlyStopping,
        curve_default_zone_boundaries,
        curve_default_zone_weights,
    )
    from resources.MLmodels import MODEL, Transformer
    print("Imports completed.")
    loss_config = loss_summary(
        args,
        active_mode,
        reduce_dim,
        curve_default_zone_boundaries,
        curve_default_zone_weights,
    )

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
        "loss_config": loss_config,
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
    if reduce_dim:
        print(f"{active_mode} latent output size: {train_out.shape[-1]}")
        pca = getattr(data, f"{active_mode}_OUTreducer", None)
        if pca is not None and hasattr(pca, "explained_variance_ratio_"):
            print(f"{active_mode} PCA explained variance ({train_out.shape[-1]} components): {float(np.sum(pca.explained_variance_ratio_)):.6f}")
    else:
        print(f"{active_mode} output curve size: {train_out.shape[-1]}")
    print(f"Loss: {loss_config}")

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
        lossf=build_loss(
            data,
            active_mode,
            args,
            reduce_dim,
            nn,
            CombinedCurveLoss,
            curve_default_zone_boundaries,
            curve_default_zone_weights,
        ),
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
