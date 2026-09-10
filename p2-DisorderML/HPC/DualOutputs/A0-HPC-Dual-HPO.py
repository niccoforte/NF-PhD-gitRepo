#!/usr/bin/env python3
"""Full-data single-GPU joint dual HPO, with explicit study-level resume."""

import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="dual-joint-hpo1")
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--run-root", default=os.environ.get("ML_RUN_ROOT", "data"))
    parser.add_argument("--target-trials", type=int, default=200)
    parser.add_argument("--timeout-hours", type=float, default=230)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsims", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    if min(args.target_trials, args.timeout_hours, args.epochs) <= 0:
        parser.error("Trial count, time budget and epochs must be positive.")
    if not args.study_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.study_name):
        parser.error("Use letters, digits, hyphens or underscores in study-name.")
    import torch
    from resources.MLdual import DUAL_DATA
    from resources.MLdualHPO import run_dual_hpo
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; --allow-cpu is only for explicit debug runs.")
    data = DUAL_DATA.from_files(path=args.data_path, nsims=None if args.nsims == "all" else int(args.nsims),
        split_frac=.9, split_seed=args.seed, range_split=(True, False), load_split=False, save_split=False,
        LAT="FCC", nnx=20, dis="disNodes", dN=.2, d_data="in", freq=False,
        field_input_config={"components": ("U1", "U2"), "drop_frame0": True, "layout": "auto"})
    relative = Path("MULTI/Dual/Transformer/HPO") / args.study_name
    archive = os.environ.get("ML_ARCHIVE_ROOT")
    return run_dual_hpo(data, Path(args.run_root) / relative, study_name=args.study_name,
        target_trials=args.target_trials, timeout_hours=args.timeout_hours, epochs=args.epochs,
        seed=args.seed, device="cuda" if torch.cuda.is_available() else "cpu",
        archive_dir=Path(archive) / relative if archive else None, resume=args.resume)


if __name__ == "__main__":
    main()
