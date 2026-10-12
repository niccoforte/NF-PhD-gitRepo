"""Matched sharing comparison; reuse DUAL_DATA/DUAL_MODEL, not a second trainer.

Stage-separated fits consume cross-fitted TRAIN predictions and frozen validation
predictions. Outer test targets are never evaluated. Banks are reusable artifacts.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from resources.MLdual import DUAL_DATA, DUAL_MODEL, DualLoss, DualStageTransformer, _json_safe
from resources.MLdualHPO import DualValidationScore

# Separate field tasks, separate curve tasks, end-to-end stage training.
CASES = {
    "1i": (True, True, False), "1ii": (True, True, True),
    "2ai": (True, False, False), "2aii": (True, False, True),
    "2bi": (False, True, False), "2bii": (False, True, True),
    "3i": (False, False, False), "3ii": (False, False, True),
}


def training_groups(case, stage=None):
    """Each group gets a fresh optimizer, scheduler and own best checkpoint."""
    fields, curves, joint = CASES[case]
    if stage:
        independent = fields if stage == "field" else curves
        return [[(stage, m)] for m in ("UT", "FT")] if independent else [[(stage, "UT"), (stage, "FT")]]
    if not joint:
        return training_groups(case, "field") + training_groups(case, "curve")
    if fields and curves:
        return [[("field", m), ("curve", m)] for m in ("UT", "FT")]
    return [[(k, m) for k in ("field", "curve") for m in ("UT", "FT")]]


def fingerprint(data):
    digest = hashlib.sha256(json.dumps(_json_safe(data.to_metadata()), sort_keys=True).encode())
    for split in ("train", "val"):
        for kind in ("geometry", "field", "field_mask", "curve"):
            values = data.splits[split][kind]
            for array in values.values() if isinstance(values, dict) else (values,):
                digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


class _CurveInputs(Dataset):
    def __init__(self, base, inputs):
        self.base, self.inputs = base, inputs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return dict(self.base[i], curve_input={m: torch.as_tensor(a[i]) for m, a in self.inputs.items()})


def _fit_groups(network, data, groups, args, path, metadata, cached=None):
    reports, histories = [], []
    score = DualValidationScore(data)
    for group in groups:
        label = "-".join(k + "_" + m for k, m in group)
        # Same shuffle/dropout seed for each independent fit, unaffected by how
        # many other groups ran earlier. No shared optimizer state or stopping.
        torch.manual_seed(args.seed)
        active = []
        for kind, mode in group:
            stage = getattr(network, kind + "_model")
            active.append(stage.tasks[mode] if hasattr(stage, "tasks") else stage)
        network.requires_grad_(False)
        for module in active:
            module.requires_grad_(True)
        weights = {k: {m: float((k, m) in group) for m in ("UT", "FT")} for k in ("field", "curve")}
        def selection(pred, true, masks):
            scores = score(pred, true, masks)
            scores["selection_score"] = float(np.mean([scores[f"selection_{k}_{m}"] for k, m in group]))
            return scores
        curve_only = all(k == "curve" for k, _ in group)
        loaders = None
        if cached is not None:
            loaders = {s: DataLoader(_CurveInputs(data.datasets[s], cached[s]), batch_size=args.batch,
                                    shuffle=s == "train", num_workers=args.num_workers)
                       for s in ("train", "val")}
        trainer = DUAL_MODEL(network, DualLoss(weights=weights), data=data,
            opt=(args.optimizer, args.weight_decay), batch=args.batch, lr=args.lr,
            curve_lr_factor=args.curve_lr_factor, grad_clip=args.grad_clip,
            device="cuda" if torch.cuda.is_available() else "cpu", num_workers=args.num_workers,
            dataloaders=loaders, selection_metric=selection,
            curve_only_source="predicted" if curve_only else None,
            scheduler=("plateau", "min", args.scheduler_factor, args.scheduler_patience, args.scheduler_threshold))
        trainer.frozen_modules = [module for module in network.modules()
                                  if list(module.parameters()) and not any(p.requires_grad for p in module.parameters())]
        trainer.train(args.epochs, verbose=args.verbose, early_stop_patience=args.early_stop_patience,
                      early_stop_delta=args.early_stop_delta, checkpoint_path=Path(path)/"groups"/label,
                      metadata=dict(metadata, training_group=group))
        histories.extend(dict(row, training_group=label) for row in trainer.history)
        reports.append(dict(group=group, best_epoch=trainer.best_epoch, best_score=trainer.best_loss))
    network.requires_grad_(True)
    return reports, histories


def fold_data(data, fit, validation, holdout):
    """Refit ALL scalers inside each fold; OOF IDs cannot influence stopping."""
    splits, ids = {}, {}
    source = data.splits["train"]
    for name, index in zip(("train", "val", "test"), (fit, validation, holdout)):
        g = data.normalizers["geometry"]
        splits[name] = {"geometry": source["geometry"][index] * g["scale"] + g["mean"],
            "field": {m: data.inverse_field(m, source["field"][m][index]) for m in ("UT", "FT")},
            "field_mask": {m: source["field_mask"][m][index] for m in ("UT", "FT")},
            "curve": {m: data.inverse_curve(m, source["curve"][m][index]) for m in ("UT", "FT")}}
        ids[name] = [data.sample_ids["train"][i] for i in index]
    return DUAL_DATA(splits, data.node_masks, data.task_features, sample_ids=ids,
                     metadata=data.metadata, task_feature_names=data.task_feature_names,
                     geometry_feature_names=data.geometry_feature_names)


def _field_predictions(network, data, split, args):
    device = next(network.parameters()).device
    network.eval()
    chunks = {m: [] for m in ("UT", "FT")}
    from resources.MLdual import _move_to_device
    with torch.no_grad():
        for batch in DataLoader(data.datasets[split], batch_size=args.batch, shuffle=False):
            batch = _move_to_device(batch, device)
            fields = network.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])
            for m in chunks:
                chunks[m].append(fields[m].cpu().numpy())
    return {m: np.concatenate(v) for m, v in chunks.items()}


def prepare_bank(model, case, args, path, metadata):
    """Cross-fitting is a training-only preparation, not locked-test evaluation."""
    data, network = model.data, model.model
    initial = copy.deepcopy(network.state_dict())
    n = len(data.train_dataset)
    if n < 3 * args.oof_folds:
        raise ValueError("Too few training specimens for disjoint OOF/inner-validation folds.")
    folds = np.array_split(np.random.default_rng(args.seed).permutation(n), args.oof_folds)
    train_inputs = {m: np.empty_like(data.splits["train"]["field"][m]) for m in ("UT", "FT")}
    manifests = []
    for number, holdout in enumerate(folds):
        remaining = np.setdiff1d(np.arange(n), holdout)
        remaining = np.random.default_rng(args.seed + number).permutation(remaining)
        nval = max(1, round(.1 * len(remaining)))
        validation, fit = remaining[:nval], remaining[nval:]
        folded = fold_data(data, fit, validation, holdout)
        net = DualStageTransformer.from_config(network.get_config())
        net.load_state_dict(initial)
        provenance = {"fold": number, "fit_ids": folded.sample_ids["train"],
                      "selection_ids": folded.sample_ids["val"], "heldout_train_ids": folded.sample_ids["test"]}
        _fit_groups(net, folded, training_groups(case, "field"), args, Path(path)/"folds"/str(number),
                    dict(metadata, crossfit=provenance))
        prediction = _field_predictions(net, folded, "test", args)
        for m in train_inputs:
            physical = folded.inverse_field(m, prediction[m])
            stats = data.normalizers["field"][m]
            train_inputs[m][holdout] = (physical-stats["mean"])/stats["scale"]
            train_inputs[m][holdout[:, None], np.flatnonzero(~data.node_masks[m])] = 0.
        manifests.append(provenance)
        del net, folded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    network.load_state_dict(initial)
    reports, histories = _fit_groups(network, data, training_groups(case, "field"), args, path, metadata)
    val_inputs = _field_predictions(network, data, "val", args)
    config = {k: getattr(args, k) for k in ("seed", "epochs", "batch", "lr", "weight_decay", "optimizer",
              "scheduler_factor", "scheduler_patience", "scheduler_threshold", "early_stop_patience", "early_stop_delta", "oof_folds")}
    manifest = dict(data_fingerprint=fingerprint(data), sample_ids=data.sample_ids, folds=manifests,
                    field_config=network.field_model.get_config(), training=config, groups=reports,
                    note="OOF test-labelled fold is OUTER TRAIN only. No locked test targets used.")
    Path(path, "field_bank.json").write_text(json.dumps(_json_safe(manifest), indent=2))
    np.savez(Path(path, "field_bank.npz"), **{f"{s}_{m}": a for s, values in
                (("train", train_inputs), ("val", val_inputs)) for m, a in values.items()})
    model.history = histories
    model.best_epoch, model.best_loss = None, None
    metadata["group_selection"] = reports
    return reports


def train_case(model, case, args, path, metadata):
    data, network = model.data, model.model
    _, _, joint = CASES[case]
    cached = None
    if not joint:
        bank = Path(args.field_bank)
        manifest = json.loads((bank/"field_bank.json").read_text())
        if manifest["data_fingerprint"] != fingerprint(data):
            raise ValueError("Field bank data/splits/normalizers differ from this run.")
        if manifest["field_config"] != _json_safe(network.field_model.get_config()):
            raise ValueError("Field-bank architecture does not match this case.")
        for key, value in manifest["training"].items():
            if getattr(args, key) != value:
                raise ValueError(f"Field bank training configuration differs: {key}.")
        saved = torch.load(bank/"model.mdl", map_location="cpu", weights_only=False)
        prefix = "field_model."
        network.field_model.load_state_dict({k[len(prefix):]: v for k, v in saved["model_state_dict"].items() if k.startswith(prefix)})
        with np.load(bank/"field_bank.npz", allow_pickle=False) as arrays:
            cached = {s: {m: arrays[f"{s}_{m}"] for m in ("UT", "FT")} for s in ("train", "val")}
        for s in cached:
            for m, a in cached[s].items():
                if a.shape != data.splits[s]["field"][m].shape or not np.isfinite(a).all():
                    raise ValueError("Invalid cached field inputs.")
        metadata["field_bank"] = str(bank)
        metadata["field_bank_sha256"] = hashlib.sha256((bank/"field_bank.npz").read_bytes()).hexdigest()
    groups = training_groups(case) if joint else training_groups(case, "curve")
    reports, histories = _fit_groups(network, data, groups, args, path, metadata, cached)
    model.history = histories
    # A multi-group fit has no unique best epoch/loss; each is recorded separately.
    model.best_epoch, model.best_loss = None, None
    model.curve_only_source = None
    metadata["group_selection"] = reports
    metadata["case"] = case + "p"
    metadata["source_policy"] = "joint predicted" if joint else "cross-fitted train; final frozen validation"
    return reports
