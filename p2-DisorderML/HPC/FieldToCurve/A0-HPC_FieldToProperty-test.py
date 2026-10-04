"""True-field oracle diagnostic; reuses the dual encoder and legacy trainer.

This predicts archived properties, not curves. It is not a disorder-only
surrogate and its checkpoint must not be opened with a curve-results loader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from resources.MLdual import DUAL_DATA, DualTaskTransformer
from resources.MLfunc import EarlyStopping, train_model

PROPERTIES = ["Strength", "Ductility", "WoF", "K_JIC"]


class FieldPropertyTransformer(nn.Module):
    """Tensor adapter for the existing generic training loop; no new encoder."""
    def __init__(self, config, node_masks):
        super().__init__()
        self.encoder = DualTaskTransformer(**config)
        for mode in ("UT", "FT"):
            self.register_buffer(mode + "_mask", torch.as_tensor(node_masks[mode], dtype=torch.bool))

    def forward(self, fields):
        output = self.encoder({"UT": fields[:, 0], "FT": fields[:, 1]},
                              node_masks={m: getattr(self, m + "_mask") for m in ("UT", "FT")})
        return torch.cat([output["UT"], output["FT"]], dim=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=os.environ.get("ML_DATA_ROOT", "/data/SEMS-TaoLab/Niccolo-Forte/p2"))
    parser.add_argument("--run-root", default=os.environ.get("ML_RUN_ROOT", "."))
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--split-reference", required=True, help="Frozen dual model_data.json; full runs require exact ID equality.")
    parser.add_argument("--base-model-json", required=True, help="Architecture/training settings only, never winner weights.")
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--nsims", type=int, default=None, help="Explicit debug subset; disables full-split identity comparison.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or (args.nsims is not None and args.nsims < 16):
        parser.error("Positive epochs/batch and nsims >=16 required.")
    if Path(args.run_label).name != args.run_label or args.run_label in (".", ".."):
        parser.error("run-label must be a single directory name.")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA required; --allow-cpu is an explicit debug override.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    relative = Path("MULTI/FieldToProperty/Transformer") / args.run_label
    folder = Path(args.run_root) / relative
    for parent in (Path(args.run_root), Path(os.environ.get("ML_ARCHIVE_ROOT", args.run_root))):
        if (parent / relative).exists():
            raise FileExistsError(parent / relative)
    folder.mkdir(parents=True)
    data = DUAL_DATA.from_files(path=args.data_path, nsims=args.nsims, split_frac=.9, split_seed=42,
        range_split=(args.nsims is None, False), load_split=False, save_split=False,
        LAT="FCC", nnx=20, dis="disNodes", dN=.2, d_data="in", freq=False,
        field_input_config={"components": ("U1", "U2"), "drop_frame0": True, "layout": "auto"})
    reference = json.loads(Path(args.split_reference).read_text())
    if args.nsims is None and any(list(map(str, data.sample_ids[s])) != list(map(str, reference["sample_ids"][s]))
                                   for s in ("train", "val", "test")):
        raise ValueError("Specimen IDs differ from the frozen split. No fit performed.")
    prop_path = Path(args.data_path) / "MLdata/MULTI-disNodes-allProps.csv"
    props = pd.read_csv(prop_path, index_col=0)
    if not props.index.is_unique:
        raise ValueError("Duplicate property specimen IDs.")
    # No test targets are selected or scored. Missing/invalid labels fail, never silently drop rows.
    targets = {s: props.loc[list(map(int, data.sample_ids[s])), PROPERTIES].to_numpy(dtype=np.float32)
               for s in ("train", "val")}
    if any(not np.isfinite(y).all() for y in targets.values()):
        raise ValueError("Nonfinite property labels; explicit filtering decision required.")
    mean, scale = targets["train"].mean(0), targets["train"].std(0)
    if np.any(scale <= 0):
        raise ValueError("Constant property in training population.")
    loaders = {}
    for split in ("train", "val"):
        fields = np.stack([data.splits[split]["field"][m] for m in ("UT", "FT")], axis=1)
        loaders[split] = DataLoader(TensorDataset(torch.as_tensor(fields),
            torch.as_tensor((targets[split]-mean)/scale)), batch_size=args.batch, shuffle=split == "train")
    anchor_path = Path(args.base_model_json)
    anchor = json.loads(anchor_path.read_text())
    config = dict(anchor["model_config"]["curve_model"])
    config.update(in_size=data.field_feature_sizes, out_size={"UT": 3, "FT": 1},
                  seq_len=data.n_nodes, context_size=0)
    model = FieldPropertyTransformer(config, data.node_masks).to(device)
    training = anchor["training"]
    lr = training["lr"] * training.get("curve_lr_factor", 1.)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[training["opt"][0]]
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=training["opt"][1])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.6273419968416533, patience=14)
    metadata = {"kind": "field-to-property-oracle", "version": 1, "properties": PROPERTIES,
        "field_source": "true", "context_size": 0, "selection": "equal-property train-standardized MSE",
        "config": config, "args": vars(args), "sample_ids": data.sample_ids,
        "target_mean": mean.tolist(), "target_scale": scale.tolist(),
        "source_revision": os.environ.get("ML_SOURCE_REVISION"), "job_id": os.environ.get("SLURM_JOB_ID"),
        "property_sha256": hashlib.sha256(prop_path.read_bytes()).hexdigest(),
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "status": "training", "device": str(device)}
    metadata_path = folder / "property_model.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if os.environ.get("ML_RUN_METADATA"):
        Path(os.environ["ML_RUN_METADATA"]).write_text(json.dumps({"run_dir": str(folder), **metadata}, indent=2))
    history = []
    def record(epoch, logs, current):
        history.append({"epoch": epoch, **logs})
        pd.DataFrame(history).to_csv(folder / "loss_history.csv", index=False)
    result = train_model("TR", model, nn.MSELoss(), args.epochs, optimizer,
        loaders["train"], loaders["val"], device=device, scheduler=scheduler,
        earlyStop=EarlyStopping(patience=52, min_delta=1e-5), verbose=1, epoch_callback=record)
    model.eval()
    with torch.no_grad():
        prediction = np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in loaders["val"]]) * scale + mean
    truth = targets["val"]
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr
    metrics = {}
    for i, name in enumerate(PROPERTIES):
        rmse = float(np.mean((prediction[:, i]-truth[:, i])**2)**.5)
        baseline = float(np.mean((mean[i]-truth[:, i])**2)**.5)
        metrics[name] = {"rmse": rmse, "train_mean_rmse": baseline,
            "skill_vs_train_mean_rmse": 1-rmse/baseline if baseline else None,
            "r2": float(r2_score(truth[:, i], prediction[:, i])),
            "spearman": float(spearmanr(truth[:, i], prediction[:, i]).statistic)}
    np.savez_compressed(folder / "predictions.npz", prediction=prediction, truth=truth,
                        sample_ids=np.asarray(data.sample_ids["val"]), properties=PROPERTIES)
    torch.save({"state_dict": model.state_dict(), "config": config,
                "node_masks": data.node_masks, "field_normalizers": data.normalizers["field"],
                "target_mean": mean, "target_scale": scale}, folder / "model.mdl")
    metadata.update(status="complete", best_epoch=result[-1], best_validation_loss=result[4])
    metadata_path.write_text(json.dumps(metadata, indent=2))
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for i, (name, ax) in enumerate(zip(PROPERTIES, axes.flat)):
        ax.scatter(truth[:, i], prediction[:, i], s=8, alpha=.4)
        limits = [float(min(truth[:, i].min(), prediction[:, i].min())),
                  float(max(truth[:, i].max(), prediction[:, i].max()))]
        ax.plot(limits, limits, color="black", linestyle="--")
        ax.set(title=name, xlabel="Archived true property", ylabel="Predicted property (same units)")
    fig.savefig(folder / "property_agreement.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots()
    ax.semilogy([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train")
    ax.semilogy([r["epoch"] for r in history], [r["val_loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Standardized property MSE"); ax.legend()
    fig.savefig(folder / "loss_history.png", dpi=150); plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)
    return folder


if __name__ == "__main__":
    main()
