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
    parser.add_argument("--curve-loss-suite", action="store_true",
                        help="Gate only baseline plus the four frozen-source corrected-loss fits.")
    parser.add_argument("--sudden-suite", action="store_true", help="Gate baseline and specimen-specific sudden weighting only.")
    parser.add_argument("--independence-suite", action="store_true", help="Gate two OOF field banks and all eight sharing cases.")
    parser.add_argument("--followup-suite", choices=["inputs", "losses"], help="Case1(i) source/corrector or custom-loss gate.")
    args = parser.parse_args()
    if sum((args.sudden_suite, args.curve_loss_suite, args.independence_suite, bool(args.followup_suite))) > 1:
        parser.error("Choose one isolated suite.")
    runner = runpy.run_path(str(Path(__file__).with_name("A0-HPC-Dual-test.py")))["main"]
    if args.followup_suite:
        common = ["--experiment", "baseline", "--base-model-json", args.base_model_json,
                  "--independence-case", "1i", "--followup-study", args.followup_suite,
                  "--nsims", "64", "--no-range-split", "--epochs", "1", "--batch", "2",
                  "--seed", "42", "--split-seed", "42"]
        def fit(label, options):
            model = runner(common + ["--run-label", args.run_label + "-" + label] + options)
            path = Path(model.model_file).parent
            from resources.MLmetrics import postprocess_load_dual_run
            postprocess_load_dual_run(path, load_model=True, data=model.data, device="cpu")
            del model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            return str(path)
        bank = fit("bank", ["--prepare-field-bank"])
        baseline = fit("predicted", ["--field-bank", bank])
        if args.followup_suite == "inputs":
            correction = fit("correction", ["--field-bank", bank, "--prepare-correction-bank",
                                             "--reference-model-json", baseline + "/model.json"])
            for source in ("true", "mixed", "corrected", "corrected_mixed"):
                fit(source, ["--field-bank", bank, "--curve-training-source", source,
                             "--correction-bank", correction])
        else:
            for variant in ("spatial", "temporal", "both", "sudden"):
                custom = ["--field-loss-variant", variant]
                loss_bank = fit("bank-" + variant, ["--prepare-field-bank"] + custom)
                fit(variant, ["--field-bank", loss_bank, "--reference-model-json", baseline + "/model.json"] + custom)
            for loss in ("combined_no_location", "combined_soft"):
                fit(loss, ["--field-bank", bank, "--curve-loss-ablation", loss])
        print("Case1(i) follow-up GPU/data/save/reload gate passed.", flush=True)
        return
    if args.independence_suite:
        from resources.MLindependence import CASES
        # The subset gate cannot exercise full-population split validation.
        model = runner(["--experiment", "baseline", "--base-model-json", args.base_model_json,
                        "--independence-case", "3ii", "--epochs", "1",
                        "--seed", "42", "--split-seed", "42",
                        "--run-label", f"{args.run_label}-full-split"])
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        banks = {}
        common = ["--experiment", "baseline", "--base-model-json", args.base_model_json,
                  "--nsims", "64", "--no-range-split", "--epochs", "1", "--seed", "42", "--split-seed", "42"]
        for case in ("1i", "3i"):
            model = runner(common + ["--independence-case", case, "--prepare-field-bank",
                                    "--run-label", f"{args.run_label}-bank-{case}"])
            banks[case] = str(Path(model.model_file).parent)
            del model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        for case, (independent, _, joint) in CASES.items():
            options = [] if joint else ["--field-bank", banks["1i" if independent else "3i"]]
            model = runner(common + ["--independence-case", case, "--run-label", f"{args.run_label}-{case}p"] + options)
            # The saved descriptor must reload through the maintained review path.
            from resources.MLmetrics import postprocess_load_dual_run
            loaded = postprocess_load_dual_run(Path(model.model_file).parent, load_model=True,
                                               data=model.data, device="cpu")
            del loaded, model
            gc.collect(); torch.cuda.empty_cache()
        print("Eight-case GPU gate passed; successful B1 archiving is still required.", flush=True)
        return
    variants = ("baseline", "partial", "private", "crack_face", "local_graph", "residual",
                "late_frame", "ft_region", "true_field", "detach",
                "winner_probe", "curve_predicted", "curve_true")
    source = None
    cases = [(v, None) for v in variants]
    if args.curve_loss_suite:
        cases = [("baseline", "mse")] + [(v, loss) for v in ("curve_true", "curve_predicted")
                 for loss in ("combined_no_location", "combined_soft")]
    if args.sudden_suite:
        cases = [("baseline", None), ("sudden", None)]
    for variant, ablation in cases:
        print(f"\nGPU preflight: {variant}", flush=True)
        argv = ["--experiment", variant, "--base-model-json", args.base_model_json,
                "--run-label", f"{args.run_label}-{variant}" + (f"-{ablation}" if ablation else ""), "--nsims", "64",
                "--no-range-split", "--epochs", "1", "--batch", "2", "--seed", "42", "--split-seed", "42"]
        if variant in ("winner_probe", "curve_predicted", "curve_true"):
            argv += ["--source-model-json", source]
        if ablation:
            argv += ["--curve-loss-ablation", ablation]
        if variant == "sudden":
            argv += ["--localization-gain", "1"]
        model = runner(argv)
        if variant == "baseline":
            source = str(Path(model.model_file).with_suffix(".json"))
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    print(f"All {len(cases)} GPU/data/save/diagnostic preflights passed; B1 must still archive successfully.", flush=True)


if __name__ == "__main__":
    main()
