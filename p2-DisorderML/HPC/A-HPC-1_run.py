#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np


def json_safe(value):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Trial UT MLP run on one HPC GPU.")
    parser.add_argument("--task", default="UT", choices=["UT", "FT", "MULTI"])
    parser.add_argument("--model-type", default="MLP")
    parser.add_argument("--run-label", default="trial-ut-mlp")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--split", default="", help="Saved split name without the 'split-' prefix.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--nsims", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lat", default="FCC")
    parser.add_argument("--dis", default="disNodes")
    parser.add_argument("--dN", type=float, default=0.2)
    parser.add_argument("--d-data", default="in")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Importing torch...")
    import torch
    import torch.nn as nn

    print("Importing project ML framework...")
    from resources.MLdata import DATA
    from resources.MLfunc import EarlyStopping
    from resources.MLmodels import MODEL, MLP
    print("Imports completed.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(os.environ.get("ML_RESULTS_DIR", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": args.task,
        "model_type": args.model_type,
        "run_label": args.run_label,
        "data_path": args.data_path,
        "split": args.split or None,
        "nsims": args.nsims,
        "seed": args.seed,
    }
    metadata_path = os.environ.get("ML_RUN_METADATA")
    if metadata_path:
        write_json(metadata_path, metadata)
    write_json(results_dir / "run_metadata.json", metadata)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    data = DATA(
        path=args.data_path,
        load=True,
        load_split=args.split or False,
        split_frac=0.8,
        split_seed=args.seed,
        range_split=(True, False),
        save_split=False,
        LAT=args.lat,
        dis=args.dis,
        dN=args.dN,
        d_data=args.d_data,
        mechMode=args.task,
        nsims=args.nsims,
        model=args.model_type,
        freq=False,
        scale=("symm", "inout"),
        reduce_dim=False,
        round_decimals=5,
    )

    in_size = data.UT_train_in.shape[-1] if data.UTmechTest else data.FT_train_in.shape[-1]
    out_size = data.UT_train_out.shape[-1] if data.UTmechTest else data.FT_train_out.shape[-1]

    model = MODEL(
        typ=data.model,
        model=MLP(
            in_size=in_size,
            h_size=[128, 64],
            out_size=out_size,
            act="relu",
            block="mlp",
            norm=None,
            dropout=0.0,
        ).to(device),
        lossf=nn.MSELoss(reduction="mean"),
        opt=("adam", args.weight_decay),
        batch=args.batch,
        lr=args.lr,
        data=data,
        mechMode=data.mechMode,
        scheduler=None,
        earlyStop=EarlyStopping(patience=10, min_delta=1e-5, verbose=True),
        w_init="auto",
        device=device,
        optTrial=None,
        scan_matches_on_init=False,
    )

    model.train(n_epochs=args.epochs, verbose=1, plot=False)
    model.evaluate_split("test", diagnostics=True, diag_plot=False)

    checkpoint = model.save(path=None, name=args.run_label)

    metrics = {
        "checkpoint": checkpoint,
        "device": str(device),
        "UT_best_loss": getattr(model, "UT_best_loss", None),
        "UT_best_mse": getattr(model, "UT_best_mse", None),
        "UT_best_rmse": getattr(model, "UT_best_rmse", None),
        "FT_best_loss": getattr(model, "FT_best_loss", None),
        "FT_best_mse": getattr(model, "FT_best_mse", None),
        "FT_best_rmse": getattr(model, "FT_best_rmse", None),
        "UT_prediction_summary": getattr(model, "UT_prediction_summary", None),
        "FT_prediction_summary": getattr(model, "FT_prediction_summary", None),
    }
    write_json(results_dir / "metrics.json", metrics)

    predictions = {}
    for mode in ("UT", "FT"):
        outputs = getattr(model, f"{mode}_test_outputs", None)
        truth = getattr(model, f"{mode}_truth", None)
        if outputs is not None:
            predictions[f"{mode}_outputs"] = outputs
        if truth is not None:
            predictions[f"{mode}_truth"] = truth
    if predictions:
        np.savez(results_dir / "predictions.npz", **predictions)

    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved trial results in: {results_dir}")


if __name__ == "__main__":
    main()
