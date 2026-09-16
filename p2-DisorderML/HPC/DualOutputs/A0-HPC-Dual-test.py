#!/usr/bin/env python3
"""Single-GPU joint UT/FT field-and-curve training, staged by B1_ML-new.sh."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="MULTI", choices=["MULTI"])
    parser.add_argument("--model-type", default="TR", choices=["TR", "Transformer"])
    parser.add_argument("--run-label", default="")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--run-root", default=os.environ.get("ML_RUN_ROOT"), help="Override the output root; B1 sets ML_RUN_ROOT.")
    parser.add_argument("--nsims", default="all", help="Full paired data by default; an integer limits debug runs.")
    parser.add_argument("--split-frac", type=float, default=0.9)
    parser.add_argument("--range-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--components", default="U1,U2")
    parser.add_argument("--keep-frame0", action="store_true", help="Include the unloaded frame; default matches the other field runners.")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val", help="Use test only for an explicitly locked final evaluation.")
    for stage in ("field", "curve"):
        parser.add_argument(f"--{stage}-d-model", type=int, default=128)
        parser.add_argument(f"--{stage}-n-heads", type=int, default=4)
        parser.add_argument(f"--{stage}-n-layers", type=int, default=3)
        parser.add_argument(f"--{stage}-ff-mult", type=int, default=4)
        parser.add_argument(f"--{stage}-dropout", type=float, default=0.1)
        for mode in ("ut", "ft"):
            parser.add_argument(f"--{stage}-{mode}-weight", type=float, default=1.0)
    parser.add_argument("--encoder-act", choices=["gelu", "relu"], default="gelu")
    parser.add_argument("--pos-encoding", choices=["none", "learned", "sinusoidal"], default="none")
    parser.add_argument("--curve-pool", choices=["cls", "mean"], default="cls")
    parser.add_argument("--curve-cls-token", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--loss", choices=["mse", "combined"], default="combined", help="Curve loss; field loss defaults to masked MSE.")
    parser.add_argument("--field-loss-variant", choices=["baseline", "spatial", "temporal", "both", "weighted"])
    parser.add_argument("--spatial-weight", type=float, default=0.1)
    parser.add_argument("--temporal-weight", type=float, default=0.1)
    parser.add_argument("--localization-gain", type=float, default=0.)
    parser.add_argument("--fixed-selection-score", action="store_true", help="Use the unchanged balanced validation score across loss variants.")
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
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for local/debug runs.")
    args = parser.parse_args(argv)
    try:
        args.nsims = None if str(args.nsims).lower() == "all" else int(args.nsims)
    except ValueError:
        parser.error("--nsims must be 'all' or a positive integer.")
    for name in ("epochs", "batch", "early_stop_patience", "derivative_order"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.nsims is not None and args.nsims < 1:
        parser.error("--nsims must be positive.")
    if not 0 < args.split_frac < 1 or args.lr <= 0 or args.loss_eps <= 0:
        parser.error("Require 0 < split-frac < 1, lr > 0, and loss-eps > 0.")
    if not 0 < args.scheduler_factor < 1 or args.num_workers < 0 or args.weight_decay < 0:
        parser.error("Invalid scheduler factor, worker count, or weight decay.")
    for stage in ("field", "curve"):
        for suffix in ("d_model", "n_heads", "n_layers", "ff_mult"):
            if getattr(args, f"{stage}_{suffix}") < 1:
                parser.error(f"{stage} Transformer dimensions must be positive.")
        if getattr(args, f"{stage}_d_model") % getattr(args, f"{stage}_n_heads"):
            parser.error(f"{stage} d-model must be divisible by n-heads.")
        if not 0 <= getattr(args, f"{stage}_dropout") < 1:
            parser.error(f"{stage} dropout must be in [0, 1).")
    if any(value < 0 for key, value in vars(args).items() if key.endswith("_weight")):
        parser.error("Loss weights must be non-negative.")
    if any(getattr(args, f"{stage}_{mode}_weight") <= 0 for stage in ("field", "curve") for mode in ("ut", "ft")):
        parser.error("This jointly supervised runner requires all four task weights > 0.")
    if not args.components.replace(",", " ").split():
        parser.error("--components cannot be empty.")
    return args


def main(argv=None, preset=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from torch import nn
    import resources.MLdual as dual_module
    from resources.MLdual import DUAL_DATA, DUAL_MODEL, DualLoss, DualStageTransformer
    from resources.MLfunc import CombinedCurveLoss, curve_default_zone_boundaries, curve_default_zone_weights
    from resources.MLmodels import _mp_run_root, _mp_slugify

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. --allow-cpu is for explicit local/debug runs.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}; torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    label = args.run_label or os.environ.get("ML_JOB_NAME") or f"multi-dual-{datetime.datetime.now():%y%m%d-%H%M%S}"
    label = _mp_slugify(label, max_len=96, preserve_case=True)
    root = Path(args.run_root).expanduser() if args.run_root else _mp_run_root()
    relative = Path("MULTI") / "Dual" / "Transformer" / label
    run_dir = root / relative
    archive_root = os.environ.get("ML_ARCHIVE_ROOT")
    archive_dir = Path(archive_root) / relative if archive_root else None
    for destination in (run_dir, archive_dir):
        if destination is not None and destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Run already exists: {destination}. Choose a new --run-label.")
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "script": Path(__file__).name, "started_at": datetime.datetime.now().isoformat(),
        "run_layout": {"task": "MULTI", "output_kind": "Dual", "model": "Transformer", "run_name": label},
        "run_dir": str(run_dir), "archive_run_dir": str(archive_dir) if archive_dir else None,
        "run_config": vars(args), "device": device, "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_revision": os.environ.get("ML_SOURCE_REVISION"),
        "preset": preset,
        "source_hashes": {
            str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(dual_module.__file__))
        },
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if os.environ.get("ML_RUN_METADATA"):
        external = Path(os.environ["ML_RUN_METADATA"])
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Loading paired data: {args.data_path}; nsims={args.nsims or 'all'}", flush=True)
    data = DUAL_DATA.from_files(
        path=args.data_path, nsims=args.nsims, split_frac=args.split_frac, split_seed=args.seed,
        range_split=(args.range_split, False), load_split=False, save_split=False,
        LAT="FCC", nnx=20, dis="disNodes", dN=0.2, d_data="in", freq=False,
        field_input_config={
            "components": tuple(args.components.replace(",", " ").split()),
            "drop_frame0": not args.keep_frame0, "layout": "auto",
        },
    )
    split_sizes = {split: len(ds) for split, ds in data.datasets.items()}
    if any(size == 0 for size in split_sizes.values()):
        raise ValueError(f"All paired splits must be nonempty: {split_sizes}. Increase --nsims or use --no-range-split for a debug subset.")
    data.to_json(run_dir / "model_data.json")
    from resources.MLmetrics import write_dual_input_audit
    write_dual_input_audit(data, run_dir / "results" / "input_audit")
    print(f"Paired splits: {split_sizes}; canonical nodes: {data.n_nodes}", flush=True)
    print(f"Native nodes: { {mode: int(mask.sum()) for mode, mask in data.node_masks.items()} }")
    print(f"Field widths: {data.field_feature_sizes}; curve widths: {data.curve_sizes}")

    stage_config = {
        stage: {
            **{key: getattr(args, f"{stage}_{key}") for key in ("d_model", "n_heads", "n_layers", "ff_mult", "dropout")},
            "activation": args.encoder_act, "pos_encoding": args.pos_encoding,
        }
        for stage in ("field", "curve")
    }
    stage_config["curve"].update(pool=args.curve_pool, use_cls_token=args.curve_cls_token)
    network = DualStageTransformer.from_data(data, field_kwargs=stage_config["field"], curve_kwargs=stage_config["curve"])
    weights = {kind: {mode: getattr(args, f"{kind}_{mode.lower()}_weight") for mode in ("UT", "FT")} for kind in ("field", "curve")}
    curve_losses = nn.MSELoss()
    if args.loss == "combined":
        curve_losses = {
            mode: CombinedCurveLoss(
                mse_weight=args.mse_weight, weighted_mse_weight=args.weighted_mse_weight,
                derivative_weight=args.derivative_weight, peak_weight=args.peak_weight,
                energy_weight=args.energy_weight, peak_location_weight=args.peak_location_weight,
                zone_boundaries=curve_default_zone_boundaries(mode), zone_weights=curve_default_zone_weights(),
                x_values=data.metadata["curve_x_values"][mode], derivative_order=args.derivative_order,
                SoftPeak_beta=args.soft_peak_beta, normalization_eps=args.loss_eps,
            ) for mode in ("UT", "FT")
        }
    field_losses = None
    if args.field_loss_variant:
        from resources.MLfield import field_loss_from_data, field_loss_weights
        field_losses = {mode:field_loss_from_data(data, mode, **field_loss_weights(
            args.field_loss_variant, args.spatial_weight, args.temporal_weight, args.localization_gain))
            for mode in ("UT", "FT")}
    objective = DualLoss(
        field_loss=field_losses,
        curve_loss=curve_losses, weights=weights,
        curve_normalizers=data.normalizers["curve"] if args.loss == "combined" else None,
    )
    selection_score = None
    if args.fixed_selection_score:
        from resources.MLdualHPO import DualValidationScore
        selection_score = DualValidationScore(data)
    model = DUAL_MODEL(
        network, objective, data=data, opt=("adamw", args.weight_decay), batch=args.batch,
        lr=args.lr, device=device, num_workers=args.num_workers,
        selection_metric=selection_score,
        scheduler=("plateau", "min", args.scheduler_factor, args.scheduler_patience, args.scheduler_threshold),
    )
    print(network)
    print(f"Parameters: {sum(p.numel() for p in network.parameters()):,}; joint weights: {weights}", flush=True)
    started = time.monotonic()
    model.train(
        args.epochs, verbose=args.verbose, early_stop_patience=args.early_stop_patience,
        early_stop_delta=args.early_stop_delta, checkpoint_path=run_dir, metadata=metadata,
    )
    metadata["training_seconds"] = time.monotonic() - started
    checkpoint = model.save(run_dir, metadata=metadata)
    results = model.save_results(eval_split=args.eval_split, run_config=vars(args), metadata=metadata)
    if args.field_loss_variant:
        from resources.MLmetrics import save_field_motion_diagnostics
        for mode in ("UT", "FT"):
            save_field_motion_diagnostics(results, data, mode, args.eval_split)
        model.save_true_field_curves(results, args.eval_split)
    metadata.update({"checkpoint": checkpoint, "results_dir": results, "status": "complete"})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}\nSaved {args.eval_split} results: {results}", flush=True)
    return model


if __name__ == "__main__":
    main()
