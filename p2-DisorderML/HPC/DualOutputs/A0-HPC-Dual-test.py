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
    parser.add_argument("--split-seed", type=int, default=None, help="Experiments default to fixed split 42 independently of training seed.")
    parser.add_argument("--experiment", choices=["baseline", "crack_face", "local_graph", "partial", "private", "true_field", "detach", "residual", "localization"],
                        help="One opt-in change relative to baseline; never mutates an HPO study.")
    parser.add_argument("--base-model-json", help="Reuse saved dual architecture/loss/training settings, not weights; CLI overrides still apply.")
    parser.add_argument("--private-layers", type=int, default=1)
    parser.add_argument("--true-curve-weight", type=float, default=0.5)
    parser.add_argument("--residual-scale-floor", type=float, default=0.1)
    parser.add_argument("--minimum-ut-strength", type=float, default=None, help="Optional physical screening threshold; no arbitrary threshold is assumed.")
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
        parser.add_argument(f"--{stage}-head-dropout", type=float, default=None)
        parser.add_argument(f"--{stage}-attention-dropout", type=float, default=None)
        parser.add_argument(f"--{stage}-head-hidden-mult", type=int, default=0)
        parser.add_argument(f"--{stage}-activation", choices=["relu", "gelu"], default=None)
        parser.add_argument(f"--{stage}-position", choices=["none", "learned", "sinusoidal"], default=None)
        parser.add_argument(f"--{stage}-norm-first", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--curve-lr-factor", type=float, default=1.)
    parser.add_argument("--grad-clip", type=float, default=None)
    initial, _ = parser.parse_known_args(argv)
    if initial.base_model_json:
        descriptor = json.loads(Path(initial.base_model_json).read_text())
        if descriptor.get("kind") != "dual-stage-transformer":
            parser.error("--base-model-json requires a dual checkpoint descriptor.")
        defaults = {}
        for stage in ("field", "curve"):
            cfg = descriptor["model_config"][f"{stage}_model"]
            if cfg.get("private_layers", 0) or cfg.get("local_graph"):
                parser.error("The experiment anchor must be a fully shared, non-graph baseline.")
            for key in ("d_model", "n_heads", "n_layers", "ff_mult", "dropout", "head_dropout", "attention_dropout", "head_hidden_mult", "activation", "norm_first"):
                if key in cfg: defaults[f"{stage}_{key}"] = cfg[key]
            defaults[f"{stage}_position"] = cfg.get("pos_encoding", "none")
        cfg = descriptor["model_config"]["curve_model"]
        defaults.update(curve_pool=cfg["pool"], curve_cls_token=cfg["use_cls_token"])
        training = descriptor["training"]
        defaults.update({k: training[k] for k in ("batch", "lr", "curve_lr_factor", "grad_clip") if k in training})
        defaults.update(optimizer=training["opt"][0], weight_decay=training["opt"][1])
        recorded = descriptor.get("metadata", {}).get("run_config", {})
        if "early_stop_patience" in recorded:
            defaults["early_stop_patience"] = recorded["early_stop_patience"]
        if recorded.get("curve_loss") in {"mse", "combined"}:
            defaults["loss"] = recorded["curve_loss"]
        if training.get("scheduler"):
            s = training["scheduler"]
            defaults.update(scheduler_factor=s[2], scheduler_patience=s[3], scheduler_threshold=s[4])
        for kind, modes in descriptor["loss_config"]["weights"].items():
            defaults.update({f"{kind}_{mode.lower()}_weight": weight for mode, weight in modes.items()})
        # Loss modules themselves are restored below; their full definitions are not guessed from CLI names.
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.base_model_json and not args.experiment:
        parser.error("--base-model-json is for fresh --experiment runs, not resume.")
    if args.experiment:
        args.fixed_selection_score = True
        if args.eval_split != "val": parser.error("Development experiments must use validation, not locked test.")
        if args.split_seed is None: args.split_seed = 42
        if args.field_loss_variant: parser.error("Do not combine experiment and independent loss-ablation switches.")
        if args.experiment == "partial" and not 0 < args.private_layers < args.field_n_layers:
            parser.error("Partial sharing requires 0 < private-layers < field-n-layers.")
        if args.experiment == "localization" and args.localization_gain < 0:
            parser.error("Localization gain cannot be negative.")
        if args.experiment == "residual" and not 0 < args.residual_scale_floor <= 1:
            parser.error("Residual scale floor must lie in (0,1].")
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
        path=args.data_path, nsims=args.nsims, split_frac=args.split_frac, split_seed=args.split_seed if args.split_seed is not None else args.seed,
        crack_face=args.experiment == "crack_face",
        residual_fields=args.experiment == "residual", residual_scale_floor=args.residual_scale_floor,
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
            **{key: getattr(args, f"{stage}_{key}") for key in ("d_model", "n_heads", "n_layers", "ff_mult", "dropout", "head_dropout", "attention_dropout", "head_hidden_mult", "norm_first")},
            "activation": getattr(args, f"{stage}_activation") or args.encoder_act,
            "pos_encoding": getattr(args, f"{stage}_position") or args.pos_encoding,
        }
        for stage in ("field", "curve")
    }
    stage_config["curve"].update(pool=args.curve_pool, use_cls_token=args.curve_cls_token)
    stage_config["field"].update(private_layers=args.field_n_layers if args.experiment == "private" else
                                  args.private_layers if args.experiment == "partial" else 0,
                                  local_graph=args.experiment == "local_graph")
    network = DualStageTransformer.from_data(data, field_kwargs=stage_config["field"], curve_kwargs=stage_config["curve"],
                                             detach_fields=args.experiment == "detach")
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
    if args.base_model_json:
        from resources.MLmodels import _model_build_loss_from_config
        saved_loss = json.loads(Path(args.base_model_json).read_text())["loss_config"]
        objective = DualLoss(
            field_loss={m: _model_build_loss_from_config(v) for m,v in saved_loss["field_losses"].items()},
            curve_loss={m: _model_build_loss_from_config(v) for m,v in saved_loss["curve_losses"].items()},
            weights=weights, scales=saved_loss["scales"],
            curve_normalizers=data.normalizers["curve"] if saved_loss.get("curve_normalizers") is not None else None)
        if any(getattr(v, "structured_field", False) for v in objective.field_losses.values()):
            raise ValueError("Architecture comparisons require an MSE field-loss anchor, not a fitted structured-loss checkpoint.")
        metadata["anchor_sha256"] = hashlib.sha256(Path(args.base_model_json).read_bytes()).hexdigest()
    if args.experiment == "localization":
        from resources.MLfield import field_loss_from_data
        gain = args.localization_gain or 1.0
        # Isolate target-dependent displacement weighting; do not silently add spatial/temporal penalties.
        objective.field_losses = nn.ModuleDict({m: field_loss_from_data(data, m, spatial_weight=0., temporal_weight=0.,
                                                                       localization_gain=gain) for m in ("UT", "FT")})
    metadata["parameter_counts"] = {s: sum(p.numel() for p in getattr(network, f"{s}_model").parameters()) for s in ("field", "curve")}
    metadata["split_hash"] = hashlib.sha256(json.dumps(data.sample_ids, sort_keys=True, default=str).encode()).hexdigest()
    selection_score = None
    if args.fixed_selection_score:
        from resources.MLdualHPO import DualValidationScore
        selection_score = DualValidationScore(data)
    model = DUAL_MODEL(
        network, objective, data=data, opt=(args.optimizer, args.weight_decay), batch=args.batch,
        lr=args.lr, device=device, num_workers=args.num_workers,
        curve_lr_factor=args.curve_lr_factor, grad_clip=args.grad_clip,
        true_curve_weight=args.true_curve_weight if args.experiment == "true_field" else 0.,
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
    if args.field_loss_variant or args.experiment:
        from resources.MLmetrics import save_field_motion_diagnostics
        for mode in ("UT", "FT"):
            save_field_motion_diagnostics(results, data, mode, args.eval_split)
        model.save_true_field_curves(results, args.eval_split)
    if args.experiment:
        from resources.MLmetrics import dual_design_diagnostics
        with np.load(Path(results)/"predictions.npz", allow_pickle=False) as saved:
            screening = dual_design_diagnostics(saved, data.metadata["curve_x_values"], minimum_ut_strength=args.minimum_ut_strength)
        (Path(results)/"design_diagnostics.json").write_text(json.dumps(screening, indent=2))
        lines = ["# Validation design screening", "", "No optimisation, candidate generation, or Pareto search was performed.", ""]
        for mode, values in screening["tasks"].items():
            lines += [f"## {mode}", "", f"Objective: {values['objective']}.",
                      f"Top {values['top_count']} recovery: {values['top_recovery']:.1%}.",
                      f"Rank correlation: {values['rank_spearman']}.",
                      f"Mean true-objective shortfall of the selected set: {values['selection_regret']:.5g} (physical integral units).", ""]
        lines += ["These metrics evaluate an experiment; they do not establish an improvement without a matched baseline.",
                  "UT cutoff uses each curve independently; a missing crossing is reported, not invented. FT area is not a toughness estimate."]
        (Path(results)/"design_diagnostics.md").write_text("\n".join(lines)+"\n")
    metadata.update({"checkpoint": checkpoint, "results_dir": results, "status": "complete"})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}\nSaved {args.eval_split} results: {results}", flush=True)
    return model


if __name__ == "__main__":
    main()
