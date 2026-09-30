#!/usr/bin/env python3
"""Explicit small GPU gate for the matched dual suite; launch through B1."""
import argparse
import gc
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-json", required=True)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()
    runner = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Dual-test.py")))["main"]
    variants = ("baseline", "partial", "private", "crack_face", "local_graph", "residual",
                "late_frame", "ft_region", "true_field", "detach",
                "winner_probe", "curve_predicted", "curve_true")
    source = None
    for variant in variants:
        print(f"\nGPU preflight: {variant}", flush=True)
        argv = ["--experiment", variant, "--base-model-json", args.base_model_json,
                "--run-label", f"{args.run_label}-{variant}", "--nsims", "64",
                "--no-range-split", "--epochs", "1", "--batch", "2", "--seed", "42", "--split-seed", "42"]
        if variant in ("winner_probe", "curve_predicted", "curve_true"):
            argv += ["--source-model-json", source]
        model = runner(argv)
        if variant == "baseline":
            source = str(Path(model.model_file).with_suffix(".json"))
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    print("All thirteen GPU/data/save/diagnostic preflights passed; B1 must still archive successfully.", flush=True)


if __name__ == "__main__":
    main()
