"""Opt-in isolated field representations, static context and positive MSE weights."""
import numpy as np
import torch
from torch import nn


def gaussian_frame_weights(n_frames, amplitude, width):
    """Frame-index prior, centre0.5; no spatial coordinate or target dependence."""
    if n_frames < 3 or not np.isfinite([amplitude, width]).all() or amplitude <= 0 or width <= 0:
        raise ValueError("Require at least three frames and positive amplitude/width")
    q = np.linspace(0., 1., n_frames)
    return (1+amplitude*np.exp(-.5*((q-.5)/width)**2))[None, :, None].astype(np.float32)


def residual_field_coordinates(data, mode, floor_fraction=.1):
    """Re-express one task in train-only node/frame/component residual coordinates.

    Update the existing NumPy storage in place, retaining dataset tensor views.
    Fit on train only; masks/physical targets/other task/geometry remain unchanged.
    """
    if not 0 < floor_fraction <= 1:
        raise ValueError("Residual scale floor must lie in (0,1]")
    old = data.normalizers["field"][mode]
    train = data.splits["train"]["field"][mode]
    valid = data.splits["train"]["field_mask"][mode]
    count = valid.sum(0, keepdims=True)
    mean = np.zeros((1, *train.shape[1:]), dtype=np.float64)
    for start in range(0, len(train), 32):
        physical = train[start:start+32].astype(float)*old["scale"]+old["mean"]
        mean += np.where(valid[start:start+32], physical, 0).sum(0, keepdims=True)
    mean /= np.maximum(count, 1)
    variance = np.zeros_like(mean)
    for start in range(0, len(train), 32):
        physical = train[start:start+32].astype(float)*old["scale"]+old["mean"]
        variance += np.where(valid[start:start+32], (physical-mean)**2, 0).sum(0, keepdims=True)
    scale = np.maximum(np.sqrt(variance/np.maximum(count, 1)),
                       np.maximum(floor_fraction*np.asarray(old["scale"]), 1e-8))
    new = dict(mean=mean.astype(np.float32), scale=np.where(count > 0, scale, 1).astype(np.float32))
    for split in data.splits.values():
        values, mask = split["field"][mode], split["field_mask"][mode]
        for start in range(0, len(values), 32):
            physical = values[start:start+32].astype(float)*old["scale"]+old["mean"]
            values[start:start+32] = np.where(mask[start:start+32],
                (physical-new["mean"])/new["scale"], 0).astype(np.float32)
    data.normalizers["field"][mode] = new
    data.metadata["field_representation"] = dict(kind="train_mean_residual", modes=[mode], scale_floor=floor_fraction)


def add_initial_crack_face(network, data):
    from resources.MLdual import dual_node_context
    features, names, spec = dual_node_context(data.metadata["canonical_coords"],
        data.task_features["UT"][:, data.task_feature_names.index("designable")],
        data.node_masks, profile="fcc_ti", crack_face=True)
    if names[:-1] != data.task_feature_names:
        raise ValueError("Static crack-face ablation requires the unchanged eleven-channel schema")
    for mode in ("UT", "FT"):
        data.task_features[mode] = np.column_stack([data.task_features[mode], features[mode][:, -1]])
        for dataset in data.datasets.values():
            dataset.task_features[mode] = torch.from_numpy(data.task_features[mode])
        stage = network.field_model.tasks[mode]
        old = stage.context_projection
        projection = nn.Linear(stage.context_size+1, stage.d_model, bias=False)
        with torch.no_grad():
            projection.weight[:, :-1].copy_(old.weight)
            projection.weight[:, -1].zero_()
        stage.context_projection, stage.context_size = projection, stage.context_size+1
    data.context_feature_size += 1
    data.task_feature_names = names
    data.metadata["context_spec"] = spec
    for dataset in data.datasets.values():
        dataset.context_spec = spec


def configure_field_variation(network, data, args):
    from resources.MLdual import DualLoss
    from resources.MLfield import field_loss_from_data, fcc_initial_crack_region
    variant, mode = args.variant, args.mode
    if variant == "crack_face":
        if mode != "FT":
            raise ValueError("Only FT has an initial crack-face ablation")
        add_initial_crack_face(network, data)
    elif variant == "residual":
        residual_field_coordinates(data, mode)
    elif variant.startswith("box") or variant.startswith("gaussian"):
        frames = data.field_feature_sizes[mode]//2
        if variant.startswith("box"):
            if mode != "FT":
                raise ValueError("Crack-tip box is an FT-only prior")
            region, _ = fcc_initial_crack_region(data.metadata["canonical_coords"], data.node_masks[mode])
            weights = np.where(region[:, None, None], float(variant[-1]), 1.).astype(np.float32)
        else:
            _, amplitude, width = variant.split("_")
            weights = gaussian_frame_weights(frames, float(amplitude[1:]), float(width[1:]))
        # Reuse existing valid-entry mean-one weighting. There are no spatial,
        # temporal or dynamic penalties in these isolated MSE interventions.
        loss = field_loss_from_data(data, mode, spatial_weight=0., temporal_weight=0., fixed_weights=weights)
        losses = dict(DualLoss().field_losses.items()); losses[mode] = loss
        return DualLoss(field_loss=losses)
    else:
        raise ValueError("Unknown field variation: "+variant)
    return DualLoss()
