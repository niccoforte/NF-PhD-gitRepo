"""Independent field-only locality screen, using the saved Case1(i) control."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time


VARIANTS = ("control", "degree", "graph", "gated_graph", "adjacency", "geometry", "cnn1", "cnn2", "cnn3")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model-json", required=True)
    parser.add_argument("--mode", choices=("UT", "FT"), default="UT")
    parser.add_argument("--variant", choices=VARIANTS, default="control")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "HPC"))
    parser.add_argument("--run-root", default=os.environ.get("ML_RUN_ROOT"))
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsims", default="all")
    parser.add_argument("--batch", type=int, help="Explicit override; default is the saved control batch")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Full-data identity check, then all sixteen tiny train/reload cases")
    parser.add_argument("--preflight-variants", nargs="+", choices=VARIANTS, default=VARIANTS,
                        help="Explicit cases for a disposable execution gate")
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.batch is not None and args.batch < 1):
        parser.error("Epochs and batch must be positive")
    args.nsims = None if args.nsims == "all" else int(args.nsims)
    if args.nsims is not None and args.nsims < 1:
        parser.error("nsims must be positive or all")
    return args


def load_data(args):
    from resources.MLdual import DUAL_DATA
    return DUAL_DATA.from_files(path=args.data_path, nsims=args.nsims, split_frac=.9,
        # Extrema forcing can consume every tiny debug holdout. Production
        # retains the exact reference split; bounded cases use ordinary splits.
        split_seed=42, range_split=(args.nsims is None, False), load_split=False, save_split=False,
        LAT="FCC", nnx=20, dis="disNodes", dN=.2, d_data="in", freq=False,
        field_input_config={"components": ("U1", "U2"), "drop_frame0": True, "layout": "auto"})


def check_reference(data, descriptor, full):
    from resources.MLdual import _json_safe
    saved = descriptor["data"]
    if descriptor.get("metadata", {}).get("curve_training_source") != "mixed":
        raise ValueError("Use the accepted Mixed input-study control descriptor")
    field = descriptor["model_config"]["field_model"]
    if not field.get("independent_tasks") or field.get("local_graph") or field.get("locality"):
        raise ValueError("Reference must have independent plain field Transformers")
    if data.task_feature_names != saved["task_feature_names"]:
        raise ValueError("Control feature schema changed")
    if full:
        for key in ("sample_ids", "normalizers", "node_masks", "reference_task_features"):
            current = data.to_metadata()[key]
            if key == "sample_ids":
                current = {s: list(map(str, values)) for s, values in current.items()}
                expected = {s: list(map(str, values)) for s, values in saved[key].items()}
            else:
                current, expected = _json_safe(current), saved[key]
            if current != expected:
                raise ValueError("Full locality control mismatch: "+key)
        for key in ("canonical_coords", "field_frame_values", "field_components"):
            if _json_safe(data.metadata[key]) != saved["metadata"][key]:
                raise ValueError("Reference geometry/frame contract changed: "+key)


def plot_results(model, results, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from resources.MLfield import fcc_initial_crack_region
    results = Path(results)
    history = pd.DataFrame(model.history)
    fig, ax = plt.subplots(figsize=(7, 3))
    for column in ("train_loss", "val_loss"):
        ax.plot(history["epoch"], np.maximum(history[column], 1e-12), label=column)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Masked field MSE", title=mode+" field-only loss")
    ax.legend(); fig.tight_layout(); fig.savefig(results/"loss-history.png", dpi=160); plt.close(fig)
    with np.load(results/"predictions.npz", allow_pickle=False) as arrays:
        truth, pred = arrays[mode+"_val_field_truth"], arrays[mode+"_val_field_outputs"]
        xy, present = arrays["canonical_coords"], arrays[mode+"_node_mask"]
        ids = arrays["sample_ids"]
    errors = np.sqrt(np.nanmean((pred-truth)**2, axis=(1, 2)))
    picks = [int(np.nanargmin(errors)), int(np.nanargmax(errors))]
    frame = truth.shape[-1]//4  # Middle retained frame, U2; same rule for all cases.
    channel = 2*frame+1
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    lo, hi = np.nanmin(truth[picks, :, channel]), np.nanmax(truth[picks, :, channel])
    error_limit = max(float(np.nanmax(np.abs(pred[picks, :, channel]-truth[picks, :, channel]))), 1e-12)
    for row, sample in enumerate(picks):
        geometry = model.data.normalizers["geometry"]
        initial_xy = xy+model.data.splits["val"]["geometry"][sample]*np.asarray(geometry["scale"]).reshape(2)+np.asarray(geometry["mean"]).reshape(2)
        for column, (label, values) in enumerate((('Truth', truth[sample, :, channel]),
                ('Prediction', pred[sample, :, channel]), ('Signed error', pred[sample, :, channel]-truth[sample, :, channel]))):
            ax = axes[row, column]
            image = ax.scatter(initial_xy[present, 0], initial_xy[present, 1], c=values[present], s=18,
                cmap="coolwarm" if column == 2 else "viridis",
                vmin=-error_limit if column == 2 else lo, vmax=error_limit if column == 2 else hi)
            ax.set(title=f"{label}: {'best' if row == 0 else 'worst'} ID {ids[sample]}", aspect="equal", xlabel="Initial x", ylabel="Initial y")
            if mode == "FT":
                _, box = fcc_initial_crack_region(xy, present)
                ax.set_xlim(box[0, 0], box[1, 0]); ax.set_ylim(box[0, 1], box[1, 1])
            fig.colorbar(image, ax=ax, label="Displacement (dataset length units)")
    fig.suptitle(f"{mode} U2, retained frame index {frame}; rank by whole-field RMSE")
    fig.tight_layout(); fig.savefig(results/"field-examples.png", dpi=160); plt.close(fig)


def run(args, *, configure=None, study="locality", source_paths=()):
    """Shared final-generator execution; an intervention returns an optional loss.

    Check the unchanged data contract before attaching an isolated intervention.
    Named study entry points own their presets; this routine owns training/export.
    """
    import numpy as np
    import torch
    from resources import MLdual, MLlocality
    from resources.MLdual import DUAL_MODEL, DualLoss, DualStageTransformer, _json_safe, _move_to_device
    from resources.MLdualHPO import DualValidationScore
    from resources.MLmetrics import postprocess_load_dual_run, save_field_motion_diagnostics
    from resources.MLmodels import _mp_run_root, _mp_slugify
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA required; CPU is explicit local/debug only")
    reference = Path(args.reference_model_json)
    descriptor = json.loads(reference.read_text())
    data = load_data(args)
    check_reference(data, descriptor, args.nsims is None)
    label = _mp_slugify(args.run_label, max_len=96, preserve_case=True)
    relative = Path("MULTI/Dual/Transformer")/label
    root = Path(args.run_root) if args.run_root else _mp_run_root()
    directory = root/relative
    archive = Path(os.environ["ML_ARCHIVE_ROOT"])/relative if os.environ.get("ML_ARCHIVE_ROOT") else None
    for destination in (directory, archive):
        if destination is not None and destination.exists() and any(destination.iterdir()):
            raise FileExistsError("Refuse to overwrite existing run: "+str(destination))
    directory.mkdir(parents=True, exist_ok=True)
    network = DualStageTransformer.from_config(descriptor["model_config"])
    # Attach the new intervention after plain initialization, so all common
    # model weights start identically in every variant (degree column starts0).
    with torch.random.fork_rng():
        torch.manual_seed(args.seed+100)
        objective = (configure(network, data, args) if configure is not None else
                     MLlocality.configure_field_locality(network, data, args.variant))
    data.to_json(directory/"model_data.json")
    training = descriptor["training"]
    batch = args.batch or training["batch"]
    score = DualValidationScore(data)
    def selection(pred, true, masks):
        return {"selection_score": score(pred, true, masks)["selection_field_"+args.mode]}
    model = DUAL_MODEL(network, objective if objective is not None else DualLoss(), data=data, opt=training["opt"],
        batch=batch, lr=training["lr"], scheduler=training["scheduler"], grad_clip=training["grad_clip"],
        curve_lr_factor=training["curve_lr_factor"], device="cuda" if torch.cuda.is_available() else "cpu",
        selection_metric=selection, field_only_modes=(args.mode,))
    metadata = dict(study=study, variant=args.variant, mode=args.mode, run_config=vars(args),
        device=str(model.device), torch=str(torch.__version__), numpy=np.__version__,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        source_revision=os.environ.get("ML_SOURCE_REVISION"), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        run_layout=dict(task="MULTI", output_kind="Dual", model="Transformer", run_name=label),
        scope="One independent field generator; no OOF or curve training/evaluation",
        source_hashes={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for p in (__file__, MLdual.__file__, MLlocality.__file__, *source_paths)})
    torch.manual_seed(args.seed)
    started = time.monotonic()
    model.train(args.epochs, verbose=1,
        early_stop_patience=descriptor["metadata"]["run_config"]["early_stop_patience"],
        early_stop_delta=descriptor["metadata"]["run_config"]["early_stop_delta"],
        checkpoint_path=directory, metadata=metadata)
    metadata.update(training_seconds=time.monotonic()-started,
        active_parameters=sum(p.numel() for p in model.model.parameters() if p.requires_grad),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    model.save(directory, metadata=metadata)
    results = model.save_results(eval_split="val", run_config=vars(args), metadata=metadata)
    save_field_motion_diagnostics(results, data, args.mode, "val")
    plot_results(model, results, args.mode)
    _, _, _, reloaded = postprocess_load_dual_run(directory, load_model=True, device=model.device)
    batch_values = _move_to_device(next(iter(model.dataloaders["val"])), model.device)
    model.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model._predict_batch(batch_values)["field"][args.mode],
                                   reloaded._predict_batch(batch_values)["field"][args.mode], rtol=1e-5, atol=1e-6)
    metadata.update(status="complete", checkpoint_reload="passed")
    (directory/"run_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2))
    print(study.upper()+"_RUN_OK", args.variant, args.mode, directory, flush=True)
    return model


def main(argv=None):
    args = parse_args(argv)
    if not args.preflight:
        return run(args)
    import torch
    descriptor = json.loads(Path(args.reference_model_json).read_text())
    # Exercise the production population guard before any tiny cases.
    full = argparse.Namespace(**vars(args)); full.nsims = None
    data = load_data(full); check_reference(data, descriptor, True); del data
    gc.collect()
    for variant in args.preflight_variants:
        for mode in ("UT", "FT"):
            case = argparse.Namespace(**vars(args))
            case.variant, case.mode, case.epochs, case.nsims = variant, mode, 1, 64
            case.run_label = args.run_label+"-"+variant+"-"+mode
            run(case); gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"LOCALITY_PREFLIGHT_OK: full-data identities and {2*len(args.preflight_variants)} train/diagnostic/reload cases", flush=True)


if __name__ == "__main__":
    main()
