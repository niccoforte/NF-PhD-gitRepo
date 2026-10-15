"""Case 1(i) source/correction and loss ablations using the existing trainer."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from resources.MLdual import (DUAL_MODEL, DualLoss, DualStageTransformer,
    FieldCorrectionStage, IndependentTaskTransformers, _task_stage, _json_safe)
from resources.MLindependence import (load_bank, fingerprint, fold_data,
    prepare_bank, training_groups, _fit_groups, _field_predictions)


def field_objective(data, args):
    from resources.MLfield import field_loss_from_data, field_loss_weights
    options = (dict(spatial_weight=0., temporal_weight=0., localization_gain=1., localization_mode="sudden")
               if args.field_loss_variant == "sudden" else
               field_loss_weights(args.field_loss_variant, args.spatial_weight, args.temporal_weight, 0.))
    losses = {m: field_loss_from_data(data, m, **options) for m in ("UT", "FT")}
    return DualLoss(field_loss=losses)


def new_corrector(network, data, args):
    refiner = IndependentTaskTransformers(in_size=data.field_feature_sizes,
        out_size=data.field_feature_sizes, seq_len=data.n_nodes,
        context_size=data.context_feature_size, pool="node", d_model=args.corrector_d_model,
        n_heads=args.corrector_n_heads, n_layers=args.corrector_n_layers,
        ff_mult=4, dropout=.05, activation="gelu", pos_encoding="sinusoidal")
    # Begin with identity, allowing learned signed residuals rather than
    # forcing the Transformer to reconstruct an already useful whole field.
    for stage in refiner.tasks.values():
        for head in stage.heads.values():
            last = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last.weight)
            if last.bias is not None:
                torch.nn.init.zeros_(last.bias)
    return DualStageTransformer(FieldCorrectionStage(refiner), copy.deepcopy(network.curve_model))


def _fit_corrector(network, data, inputs, args, path, metadata):
    torch.manual_seed(args.seed + 2000)
    net = new_corrector(network, data, args)
    groups, history = _fit_groups(net, data, training_groups("1i", "field"), args,
        path, metadata, cached=inputs, cached_key="geometry")
    return net, groups, history


def corrected_predictions(net, data, inputs, split, args):
    """Reuse batched prediction with cached field inputs in the geometry slot."""
    from torch.utils.data import DataLoader
    from resources.MLindependence import _CurveInputs
    from resources.MLdual import _move_to_device
    net.eval()
    chunks = {m: [] for m in ("UT", "FT")}
    device = next(net.parameters()).device
    with torch.no_grad():
        for batch in DataLoader(_CurveInputs(data.datasets[split], inputs, key="geometry"), batch_size=args.batch):
            batch = _move_to_device(batch, device)
            p = net.field_model(batch["geometry"], batch["task_features"], batch["node_mask"])
            for m in chunks:
                chunks[m].append(p[m].cpu().numpy())
    return {m: np.concatenate(a) for m, a in chunks.items()}


def save_curve_probe(model, path, prefix):
    from resources.MLmetrics import curve_performance_diagnostics
    result = model.predict("val")
    arrays = {"sample_ids": result["sample_id"]}
    summaries = {}
    for m in ("UT", "FT"):
        prediction = model.data.inverse_curve(m, result["prediction"]["curve"][m])
        truth = model.data.inverse_curve(m, result["truth"]["curve"][m])
        diag = curve_performance_diagnostics(prediction, truth,
            x_values=model.data.metadata["curve_x_values"][m],
            train_truth=model.data.inverse_curve(m, model.data.splits["train"]["curve"][m]))
        summaries[m] = diag["summary"]
        table = diag["sample_metrics"].copy()
        table["sample_id"] = result["sample_id"]
        table.to_csv(Path(path)/f"{prefix}_{m}_sample_metrics.csv", index=False)
        arrays.update({f"{m}_outputs": prediction, f"{m}_truth": truth})
    np.savez(Path(path)/f"{prefix}.npz", **arrays)
    (Path(path)/f"{prefix}_metrics.json").write_text(json.dumps(_json_safe(summaries), indent=2))
    model.predictions = {}


def reference_model(model, args, bank_hash=None):
    from resources.MLmetrics import postprocess_load_dual_run
    descriptor = json.loads(Path(args.reference_model_json).read_text())
    meta = descriptor["metadata"]
    if (meta.get("case") != "1ip" or meta["run_config"]["seed"] != args.seed
            or (bank_hash is not None and meta.get("field_bank_sha256") != bank_hash)):
        raise ValueError("Reference must be the matching seeded Case1(i) baseline.")
    return postprocess_load_dual_run(Path(args.reference_model_json).parent,
        load_model=True, data=model.data, device=model.device)[3]


def prepare_correction_bank(model, args, path, metadata):
    """Nested cross-fitting: no correction/field fit sees its heldout curve row.

    Ordinary single-level OOF correctors could indirectly see heldout targets
    through first-stage models fitted on them. Inner field banks remove that path.
    Inner folds' test-labelled rows always belong to OUTER TRAIN, never locked test.
    """
    data, net = model.data, model.model
    cached, manifest = load_bank(net, data, args, args.field_bank)
    # Fold generators start from the same seeded initialization as Case 1(i).
    with torch.random.fork_rng():
        torch.manual_seed(args.seed)
        initial = DualStageTransformer.from_config(net.get_config())
    train = {m: np.empty_like(cached["train"][m]) for m in ("UT", "FT")}
    id_index = {str(v): i for i, v in enumerate(data.sample_ids["train"])}
    seen = []
    for fold in manifest["folds"]:
        groups = [list(map(str,fold[k])) for k in ("fit_ids", "selection_ids", "heldout_train_ids")]
        sets = list(map(set,groups))
        if (any(len(v) != len(s) for v,s in zip(groups,sets)) or any(
                sets[i] & sets[j] for i in range(3) for j in range(i))
                or set.union(*sets) != set(id_index)):
            raise ValueError("Invalid nested fold membership or overlap.")
        seen.extend(groups[2])
    if len(seen) != len(id_index) or set(seen) != set(id_index):
        raise ValueError("Nested heldouts must cover each training specimen exactly once.")
    nested = []
    for fold in manifest["folds"]:
        indices = {k: np.array([id_index[str(v)] for v in fold[k]])
                   for k in ("fit_ids", "selection_ids", "heldout_train_ids")}
        folded = fold_data(data, indices["fit_ids"], indices["selection_ids"], indices["heldout_train_ids"])
        inner_dir = Path(path)/"nested"/str(fold["fold"])
        inner_dir.mkdir(parents=True, exist_ok=True)
        inner = DUAL_MODEL(copy.deepcopy(initial), DualLoss(), data=folded, device=model.device)
        prepare_bank(inner, "1i", args, inner_dir, dict(metadata, nested_outer_fold=fold))
        with np.load(inner_dir/"field_bank.npz", allow_pickle=False) as arrays:
            inputs = {s: {m: arrays[f"{s}_{m}"] for m in ("UT", "FT")} for s in ("train", "val")}
        correction, _, _ = _fit_corrector(inner.model, folded, inputs, args,
                                         inner_dir/"corrector", dict(metadata, nested_outer_fold=fold))
        raw_holdout = _field_predictions(inner.model, folded, "test", args)
        corrected = corrected_predictions(correction, folded, raw_holdout, "test", args)
        for m in train:
            physical = folded.inverse_field(m, corrected[m])
            stats = data.normalizers["field"][m]
            values = (physical-stats["mean"])/stats["scale"]
            values[:, ~data.node_masks[m]] = 0.
            train[m][indices["heldout_train_ids"]] = values
        nested.append(fold)
        del correction, inner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    correction, reports, histories = _fit_corrector(net, data, cached, args, Path(path)/"corrector", metadata)
    val = corrected_predictions(correction, data, cached["val"], "val", args)
    raw_hash = hashlib.sha256((Path(args.field_bank)/"field_bank.npz").read_bytes()).hexdigest()
    reference = reference_model(model, args, raw_hash)
    net.curve_model.load_state_dict(reference.model.curve_model.state_dict())
    net.field_model = FieldCorrectionStage(correction.field_model.correction, net.field_model).to(model.device)
    model.history, model.best_epoch, model.best_loss = histories, None, None
    np.savez(Path(path)/"correction_bank.npz", **{f"{s}_{m}": a for s, values in
             (("train", train), ("val", val)) for m, a in values.items()})
    provenance = dict(kind="nested-field-correction-bank", version=1,
        data_fingerprint=fingerprint(data), sample_ids=data.sample_ids,
        raw_bank_sha256=raw_hash,
        seed=args.seed, folds=nested, refiner=dict(d_model=args.corrector_d_model,
            n_heads=args.corrector_n_heads, n_layers=args.corrector_n_layers),
        note="Nested OOF outer TRAIN only; frozen original curve readout in standard results.")
    (Path(path)/"correction_bank.json").write_text(json.dumps(_json_safe(provenance), indent=2))
    metadata.update(group_selection=reports, correction=provenance,
                    source_policy="corrected predicted fields; frozen Case1(i) curve readout")


def train_followup(model, args, path, metadata):
    data, net = model.data, model.model
    cached, _ = load_bank(net, data, args, args.field_bank)
    metadata.update(field_bank=str(args.field_bank),
        field_bank_sha256=hashlib.sha256((Path(args.field_bank)/"field_bank.npz").read_bytes()).hexdigest())
    raw_field_stage = net.field_model
    corrected = None
    if args.correction_bank:
        bank = Path(args.correction_bank)
        provenance = json.loads((bank/"correction_bank.json").read_text())
        if (provenance.get("version") != 1 or provenance.get("kind") != "nested-field-correction-bank"
                or provenance["data_fingerprint"] != fingerprint(data) or provenance["seed"] != args.seed):
            raise ValueError("Correction bank IDs/scales/seed mismatch.")
        if provenance["raw_bank_sha256"] != hashlib.sha256((Path(args.field_bank)/"field_bank.npz").read_bytes()).hexdigest():
            raise ValueError("Correction bank belongs to another field generator.")
        with np.load(bank/"correction_bank.npz", allow_pickle=False) as a:
            corrected = {s: {m: a[f"{s}_{m}"] for m in ("UT", "FT")} for s in ("train", "val")}
        for s, values in corrected.items():
            for m, a in values.items():
                if a.shape != data.splits[s]["field"][m].shape or not np.isfinite(a).all() or np.any(a[:, ~data.node_masks[m]]):
                    raise ValueError("Invalid corrected bank shape, finite values or absent-node mask.")
        cfg = json.loads((bank/"model.json").read_text())["model_config"]["field_model"]
        stage = _task_stage(cfg)
        saved = torch.load(bank/"model.mdl", map_location="cpu", weights_only=False)["model_state_dict"]
        stage.load_state_dict({k[len("field_model."):]: v for k,v in saved.items() if k.startswith("field_model.")})
    source = args.curve_training_source
    true = {m: data.splits["train"]["field"][m] for m in ("UT", "FT")}
    inputs = corrected if source.startswith("corrected") else cached
    training = true if source == "true" else inputs["train"]
    secondary = true if source in ("mixed", "corrected_mixed") else None
    if source.startswith("corrected"):
        net.field_model = stage.to(model.device)
    if args.reference_model_json:
        # A fixed original curve readout separates field improvement from the
        # benefit of retraining a downstream curve model on a new field source.
        reference = reference_model(model, args)
        original_curve = copy.deepcopy(net.curve_model.state_dict())
        net.curve_model.load_state_dict(reference.model.curve_model.state_dict())
        del reference
        save_curve_probe(model, path, "fixed_readout_curves")
        net.curve_model.load_state_dict(original_curve)
    reports, histories = _fit_groups(net, data, training_groups("1i", "curve"), args, path,
        metadata, cached={"train": training, "val": inputs["val"]}, objective=model.lossf, secondary=secondary)
    model.history, model.best_epoch, model.best_loss = histories, None, None
    model.curve_only_source = None
    metadata.update(group_selection=reports, case="1ip", curve_training_source=source,
                    source_policy="predicted-field deployment; true-field substitution is diagnostic",
                    mixture_probability=.5 if secondary is not None else None)
    # Optional correction substitution is evaluated on the SAME trained curve
    # weights. This differs from a curve fit trained on corrected OOF fields.
    if corrected is not None and not source.startswith("corrected"):
        raw_stage = net.field_model
        net.field_model = stage.to(model.device)
        save_curve_probe(model, path, "corrected_source_curves")
        net.field_model = raw_stage
    elif source.startswith("corrected"):
        corrected_stage = net.field_model
        net.field_model = raw_field_stage
        save_curve_probe(model, path, "uncorrected_source_curves")
        net.field_model = corrected_stage
