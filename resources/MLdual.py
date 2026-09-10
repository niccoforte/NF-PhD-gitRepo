"""Joint UT/FT serial Transformer data, model, loss, and training helpers.

The legacy :mod:`resources.MLdata`, :mod:`resources.MLmodels`, and
:mod:`resources.MLfunc` APIs remain unchanged.  This module composes their
existing data contract into the explicitly joint workflow:

    one canonical geometry -> {UT field, FT field}
    {UT field, FT field} -> {UT curve, FT curve}

Both task streams are stacked along the batch dimension inside each stage, so
each stage executes one shared Transformer encoder and two small task heads.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from resources.MLdata import DATA, _data_field_flatten
from resources.MLfunc import MaskedFieldMSELoss
from resources.MLmodels import _model_loss_to_config, _model_optimizer, _model_scheduler


DUAL_MODES = ("UT", "FT")
DUAL_SPLITS = ("train", "val", "test")
__all__ = ["DUAL_DATA", "DualTaskTransformer", "DualStageTransformer", "DualLoss", "DUAL_MODEL"]


def _mode_value(values, mode, name):
    if not isinstance(values, Mapping):
        return values
    aliases = ("CT", "ct", "C(T)", "c(t)") if mode == "FT" else ()
    for key in (mode, mode.lower(), *aliases):
        if key in values:
            return values[key]
    raise KeyError(f"{name} must provide a value for {mode}.")


def _mode_sizes(value, name):
    if isinstance(value, Mapping):
        return {mode: int(_mode_value(value, mode, name)) for mode in DUAL_MODES}
    return {mode: int(value) for mode in DUAL_MODES}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _reference_coordinates(data, mode):
    in_df = getattr(data, f"{mode}_IN_df")
    delta_df = getattr(data, f"{mode}_dINr_df")
    if in_df.shape[1] != delta_df.shape[1] or in_df.shape[1] % 2:
        raise ValueError(f"{mode}: geometry inputs must contain aligned x/y column pairs.")

    common_rows = in_df.index.intersection(delta_df.index)
    if len(common_rows) == 0:
        raise ValueError(f"{mode}: cannot recover reference coordinates from empty aligned inputs.")
    row = common_rows[0]
    reference = in_df.loc[row].to_numpy(dtype=float) - delta_df.loc[row].to_numpy(dtype=float)
    return reference.reshape(-1, 2)


def _match_nodes(canonical_coords, task_coords, tolerance, name):
    canonical = np.asarray(canonical_coords, dtype=float)
    task = np.asarray(task_coords, dtype=float)
    if canonical.ndim != 2 or canonical.shape[1] != 2:
        raise ValueError("canonical_coords must have shape [nodes, 2].")
    if task.ndim != 2 or task.shape[1] != 2:
        raise ValueError(f"{name} coordinates must have shape [nodes, 2].")
    if len(task) > len(canonical):
        raise ValueError(f"{name} has more nodes ({len(task)}) than the canonical geometry ({len(canonical)}).")

    distances = np.linalg.norm(canonical[:, None, :] - task[None, :, :], axis=-1)
    canonical_indices = np.argmin(distances, axis=0)
    matched_distances = distances[canonical_indices, np.arange(len(task))]
    if np.any(matched_distances > tolerance):
        bad = int(np.argmax(matched_distances))
        raise ValueError(
            f"{name} node {bad} has no canonical match within tolerance {tolerance:.3g}; "
            f"nearest distance is {matched_distances[bad]:.3g}."
        )
    if len(np.unique(canonical_indices)) != len(canonical_indices):
        raise ValueError(f"{name} coordinate mapping is not one-to-one.")
    return canonical_indices.astype(np.int64)


def _normalization_flags(value):
    defaults = {"geometry": True, "field": True, "curve": True}
    if isinstance(value, bool):
        return {key: value for key in defaults}
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise TypeError("normalize must be bool, None, or a geometry/field/curve mapping.")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown normalization keys: {sorted(unknown)}.")
    defaults.update({key: bool(item) for key, item in value.items()})
    return defaults


def _fit_standardizer(values, axes, mask=None, eps=1e-8):
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)

    count = valid.sum(axis=axes, keepdims=True)
    if np.any(count == 0):
        raise ValueError("Cannot fit a normalizer because at least one feature has no valid training values.")
    total = np.where(valid, array, 0.0).sum(axis=axes, keepdims=True)
    mean = total / count
    variance = np.where(valid, (array - mean) ** 2, 0.0).sum(axis=axes, keepdims=True) / count
    scale = np.sqrt(variance)
    scale = np.where(scale > eps, scale, 1.0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def _identity_standardizer(values, axes):
    shape = list(np.asarray(values).shape)
    for axis in axes:
        shape[axis] = 1
    return {
        "mean": np.zeros(shape, dtype=np.float32),
        "scale": np.ones(shape, dtype=np.float32),
    }


def _apply_standardizer(values, stats, mask=None):
    array = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(array)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)
    output = (array - stats["mean"]) / stats["scale"]
    return np.where(valid, output, 0.0).astype(np.float32)


def dual_node_context(coords, designable, node_masks, profile="fcc_ti"):
    """Reference context in a common schema; pin membership is updated per sample.

    The named production profile is deliberately restricted to the current
    20-by-19 FCC body and Ti/Al A1 pin geometry. It accepts uniform coordinate
    rescaling/translation, but refuses an unrecognised lattice or crack mask.
    ``shared`` is an explicit geometry-only option for synthetic/custom data.
    """
    coords = np.asarray(coords, dtype=float)
    origin = coords.min(axis=0)
    span = np.ptp(coords, axis=0)
    xy = (coords - origin) / np.where(span > 0, span, 1)
    names = ["x0", "y0", "designable", "present"]
    features = {
        mode: np.column_stack([xy, designable, node_masks[mode]]).astype(np.float32)
        for mode in DUAL_MODES
    }
    spec = {"profile": profile, "origin": origin, "span": span,
            "shared_features": names[:3], "coordinate_units": "input-coordinate units (not inferred SI units)"}
    if profile == "shared":
        return features, names, spec
    if profile != "fcc_ti":
        raise ValueError("context_profile must be 'fcc_ti' or explicit geometry-only 'shared'.")
    cell = span[0] / 20
    if cell <= 0 or not np.isclose(span[1], 19 * cell) or len(coords) != 800:
        raise ValueError("fcc_ti context requires the complete 20-by-19 FCC body; check units/cropping.")
    grid = (coords - origin) / cell
    expected = {(float(x), float(y)) for y in range(20) for x in range(21)}
    expected |= {(x + .5, y + .5) for y in range(19) for x in range(20)}
    if {tuple(row) for row in np.round(grid, 5)} != expected:
        raise ValueError("Reference coordinates do not match the FCC corner/centre grid.")
    # A1 uses reference nodes for node deletion. Both periodic and 20% cutoffs
    # delete these same 12 centre nodes; the loaded FT mapping stays authoritative.
    removed = (grid[:, 0] > -1.6) & (grid[:, 0] < 11.8) & (abs(grid[:, 1] - 9.5) < .2)
    if not np.all(node_masks["UT"]) or not np.array_equal(~np.asarray(node_masks["FT"]), removed):
        raise ValueError("FT mapping disagrees with the A1 reference crack; inspect it before assigning BCs.")
    width = span[0] / 1.25
    fixed = origin + [span[0] - width, span[1] / 2 - .375 * width]
    moving = origin + [span[0] - width, span[1] / 2 + .375 * width]
    tip = origin + [.75 * width, span[1] / 2]
    names += ["bottom_interface", "top_interface", "coupled_fixity", "coupled_load",
              "tip_dx_ref", "tip_dy_ref", "tip_distance_ref"]
    for mode in DUAL_MODES:
        extra = np.zeros((len(coords), 7), dtype=np.float32)
        if mode == "UT":
            extra[:, 0] = np.isclose(grid[:, 1], 0, atol=1e-5, rtol=0)
            extra[:, 1] = np.isclose(grid[:, 1], 19, atol=1e-5, rtol=0)
        else:
            offset = (coords - tip) / width
            extra[:, 4:6] = offset
            extra[:, 6] = np.linalg.norm(offset, axis=1)
        features[mode] = np.column_stack([features[mode], extra])
    spec.update({"cell_size": cell, "W": width, "fixed_pin": fixed, "moving_pin": moving,
                 "pin_radius": .1875 * width / 2, "nominal_tip": tip,
                 "pin_feature_indices": [6, 7], "feature_names": names,
                 "pin_coordinate_basis": "sample-specific undeformed disordered coordinates",
                 "source": "A1_FractureToughness-Ductility.py Ti/Al pin and UT body-set rules",
                 "limitations": "Body-node membership only; no mesh-interior nodes. RP rotation not prescribed; coupling is not a zero-displacement constraint. Nominal tip a0, not deletion-rectangle endpoint."})
    features["FT"] = dual_sample_context(features["FT"], coords, node_masks["FT"], spec)
    return features, names, spec


def dual_sample_context(reference_features, disordered_coords, present, spec):
    """Replace only FT pin flags using undeformed *disordered* geometry, not U."""
    result = np.asarray(reference_features, dtype=np.float32).copy()
    if spec.get("profile") == "fcc_ti":
        for column, key in zip(spec["pin_feature_indices"], ("fixed_pin", "moving_pin")):
            distance = np.linalg.norm(np.asarray(disordered_coords) - spec[key], axis=-1)
            result[:, column] = (distance <= spec["pin_radius"]) & present
    return result


class _DualTensorDataset(Dataset):
    def __init__(self, split, node_masks, task_features, sample_ids, metadata):
        self.geometry = torch.as_tensor(split["geometry"], dtype=torch.float32)
        self.fields = {
            mode: torch.as_tensor(split["field"][mode], dtype=torch.float32)
            for mode in DUAL_MODES
        }
        self.field_masks = {
            mode: torch.as_tensor(split["field_mask"][mode], dtype=torch.bool)
            for mode in DUAL_MODES
        }
        self.curves = {
            mode: torch.as_tensor(split["curve"][mode], dtype=torch.float32)
            for mode in DUAL_MODES
        }
        self.node_masks = {
            mode: torch.as_tensor(node_masks[mode], dtype=torch.bool)
            for mode in DUAL_MODES
        }
        self.task_features = {
            mode: torch.as_tensor(task_features[mode], dtype=torch.float32)
            for mode in DUAL_MODES
        }
        self.sample_ids = [str(item) for item in sample_ids]
        self.context_spec = metadata.get("context_spec", {})
        self.pin_masks = torch.as_tensor(split["pin_masks"]) if "pin_masks" in split else None

    def __len__(self):
        return len(self.geometry)

    def __getitem__(self, index):
        context = dict(self.task_features)
        if self.pin_masks is not None:
            context["FT"] = context["FT"].clone()
            context["FT"][:, self.context_spec["pin_feature_indices"]] = self.pin_masks[index].float()
        return {
            "sample_id": self.sample_ids[index],
            "geometry": self.geometry[index],
            "task_features": context,
            "node_mask": {mode: self.node_masks[mode] for mode in DUAL_MODES},
            "field": {mode: self.fields[mode][index] for mode in DUAL_MODES},
            "field_mask": {mode: self.field_masks[mode][index] for mode in DUAL_MODES},
            "curve": {mode: self.curves[mode][index] for mode in DUAL_MODES},
        }


class DUAL_DATA:
    """Thin paired-data adapter around the existing ``DATA`` products.

    The class does not reimplement CSV/NPZ loading or splitting.  Use
    :meth:`from_data` with one loaded ``DATA(input_kind='field',
    output_kind='curve', mechMode='MULTI')`` object, or :meth:`from_files` as a
    convenience constructor for that exact legacy configuration.
    """

    def __init__(
        self,
        splits,
        node_masks,
        task_features,
        sample_ids=None,
        normalize=True,
        metadata=None,
        geometry_feature_names=None,
        task_feature_names=None,
    ):
        self.mechMode = "MULTI"
        self.input_kind = "geometry"
        self.output_kind = "dual"
        self.UTmechTest = True
        self.FTmechTest = True
        self.normalization = _normalization_flags(normalize)
        self.metadata = dict(metadata or {})
        self.geometry_feature_names = list(geometry_feature_names or ["dx", "dy"])
        self.task_feature_names = list(task_feature_names or [])

        self.node_masks = {
            mode: np.asarray(_mode_value(node_masks, mode, "node_masks"), dtype=bool).copy()
            for mode in DUAL_MODES
        }
        self.task_features = {
            mode: np.asarray(_mode_value(task_features, mode, "task_features"), dtype=np.float32).copy()
            for mode in DUAL_MODES
        }
        self.splits = self._copy_and_validate_splits(splits)
        self.sample_ids = self._resolve_sample_ids(sample_ids)
        self.normalizers = self._fit_normalizers()
        spec = self.metadata.get("context_spec", {})
        if spec.get("profile") == "fcc_ti":
            for split in self.splits.values():
                # Evaluate membership BEFORE standardisation: inverse-rounding
                # at the circle boundary must not change an input BC flag.
                coords = self.metadata["canonical_coords"] + split["geometry"]
                split["pin_masks"] = np.stack([
                    (np.linalg.norm(coords - spec[key], axis=-1) <= spec["pin_radius"])
                    & self.node_masks["FT"] for key in ("fixed_pin", "moving_pin")
                ], axis=-1)
        self._normalize_splits()

        self.datasets = {
            split: _DualTensorDataset(
                self.splits[split],
                self.node_masks,
                self.task_features,
                self.sample_ids[split],
                self.metadata,
            )
            for split in DUAL_SPLITS
        }
        for split, dataset in self.datasets.items():
            setattr(self, f"{split}_dataset", dataset)

        train = self.splits["train"]
        self.n_nodes = int(train["geometry"].shape[1])
        self.geometry_feature_size = int(train["geometry"].shape[2])
        self.context_feature_size = int(self.task_features["UT"].shape[1])
        self.field_feature_sizes = {
            mode: int(train["field"][mode].shape[2]) for mode in DUAL_MODES
        }
        self.curve_sizes = {
            mode: int(train["curve"][mode].shape[1]) for mode in DUAL_MODES
        }

    def _copy_and_validate_splits(self, splits):
        if not isinstance(splits, Mapping):
            raise TypeError("splits must be a train/val/test mapping.")
        prepared = {}
        node_count = None
        context_size = None

        for mode in DUAL_MODES:
            mask = self.node_masks[mode]
            features = self.task_features[mode]
            if mask.ndim != 1:
                raise ValueError(f"{mode} node_mask must have shape [nodes].")
            if features.ndim != 2 or features.shape[0] != len(mask):
                raise ValueError(f"{mode} task_features must have shape [nodes, features].")
            if not np.isfinite(features).all():
                raise ValueError(f"{mode} task_features must be finite.")
            node_count = len(mask) if node_count is None else node_count
            context_size = features.shape[1] if context_size is None else context_size
            if len(mask) != node_count:
                raise ValueError("UT and FT node masks must use the same canonical node count.")
            if features.shape[1] != context_size:
                raise ValueError("UT and FT task features must have the same feature count.")
            if not np.any(mask):
                raise ValueError(f"{mode} node_mask must retain at least one node.")

        for split_name in DUAL_SPLITS:
            if split_name not in splits:
                raise KeyError(f"splits is missing '{split_name}'.")
            split = splits[split_name]
            geometry = np.asarray(split["geometry"], dtype=np.float32)
            if geometry.ndim != 3 or geometry.shape[1] != node_count:
                raise ValueError(
                    f"{split_name} geometry must have shape [samples, {node_count}, features]."
                )
            if not np.isfinite(geometry).all():
                raise ValueError(f"{split_name} geometry must be finite.")
            n_samples = geometry.shape[0]
            fields, field_masks, curves = {}, {}, {}
            for mode in DUAL_MODES:
                field = np.asarray(_mode_value(split["field"], mode, "field"), dtype=np.float32)
                field_mask = np.asarray(
                    _mode_value(split.get("field_mask", {}), mode, "field_mask"),
                    dtype=bool,
                )
                curve = np.asarray(_mode_value(split["curve"], mode, "curve"), dtype=np.float32)
                if field.ndim != 3 or field.shape[:2] != (n_samples, node_count):
                    raise ValueError(
                        f"{split_name} {mode} field must have shape [samples, {node_count}, features]."
                    )
                if field_mask.shape != field.shape:
                    raise ValueError(f"{split_name} {mode} field_mask must match field shape.")
                if field_mask[:, ~self.node_masks[mode], :].any():
                    raise ValueError(f"{split_name} {mode} field_mask marks absent nodes as valid.")
                if not np.isfinite(field[field_mask]).all():
                    raise ValueError(f"{split_name} {mode} field contains invalid values marked as valid.")
                if n_samples and not field_mask.reshape(n_samples, -1).any(axis=1).all():
                    raise ValueError(f"{split_name} {mode} contains a sample with no valid field target.")
                if curve.ndim != 2 or curve.shape[0] != n_samples:
                    raise ValueError(f"{split_name} {mode} curve must have shape [samples, points].")
                if not np.isfinite(curve).all():
                    raise ValueError(f"{split_name} {mode} curve targets must be finite.")
                fields[mode], field_masks[mode], curves[mode] = field.copy(), field_mask.copy(), curve.copy()
            prepared[split_name] = {
                "geometry": geometry.copy(),
                "field": fields,
                "field_mask": field_masks,
                "curve": curves,
            }
        return prepared

    def _resolve_sample_ids(self, sample_ids):
        sample_ids = {} if sample_ids is None else sample_ids
        resolved = {}
        for split in DUAL_SPLITS:
            n_samples = len(self.splits[split]["geometry"])
            values = sample_ids.get(split, range(n_samples)) if isinstance(sample_ids, Mapping) else range(n_samples)
            values = list(values)
            if len(values) != n_samples:
                raise ValueError(f"{split} sample_ids length must match the split sample count.")
            resolved[split] = values
        return resolved

    def _fit_normalizers(self):
        train = self.splits["train"]
        geometry_stats = (
            _fit_standardizer(train["geometry"], axes=(0, 1))
            if self.normalization["geometry"]
            else _identity_standardizer(train["geometry"], axes=(0, 1))
        )
        field_stats, curve_stats = {}, {}
        for mode in DUAL_MODES:
            field_stats[mode] = (
                _fit_standardizer(
                    train["field"][mode],
                    axes=(0, 1),
                    mask=train["field_mask"][mode],
                )
                if self.normalization["field"]
                else _identity_standardizer(train["field"][mode], axes=(0, 1))
            )
            curve_stats[mode] = (
                _fit_standardizer(train["curve"][mode], axes=(0, 1))
                if self.normalization["curve"]
                else _identity_standardizer(train["curve"][mode], axes=(0, 1))
            )
        return {"geometry": geometry_stats, "field": field_stats, "curve": curve_stats}

    def _normalize_splits(self):
        for split in self.splits.values():
            split["geometry"] = _apply_standardizer(split["geometry"], self.normalizers["geometry"])
            for mode in DUAL_MODES:
                split["field"][mode] = _apply_standardizer(
                    split["field"][mode],
                    self.normalizers["field"][mode],
                    mask=split["field_mask"][mode],
                )
                split["curve"][mode] = _apply_standardizer(
                    split["curve"][mode],
                    self.normalizers["curve"][mode],
                )

    @classmethod
    def from_data(
        cls,
        data,
        normalize=True,
        node_tolerance=None,
        context_profile="fcc_ti",
        extra_task_features=None,
        extra_task_feature_names=None,
    ):
        if not all(bool(getattr(data, f"{mode}mechTest", False)) for mode in DUAL_MODES):
            raise ValueError("DUAL_DATA requires aligned UT and FT data.")
        if str(getattr(data, "mechMode", "")).upper() != "MULTI":
            raise ValueError("DUAL_DATA requires DATA(mechMode='MULTI').")
        if str(getattr(data, "input_kind", "")).lower() != "field" or str(
            getattr(data, "output_kind", "")
        ).lower() != "curve":
            raise ValueError(
                "DUAL_DATA.from_data requires DATA(input_kind='field', output_kind='curve') so both "
                "field histories and curves are loaded once."
            )
        if getattr(data, "scale", False) or getattr(data, "reduce_dim", False):
            raise ValueError(
                "Pass an unscaled, unreduced DATA object. DUAL_DATA owns the single shared field-to-curve "
                "normalization bridge, and the initial dual workflow uses full curves."
            )

        canonical_coords = _reference_coordinates(data, "UT")
        n_nodes = len(canonical_coords)
        if node_tolerance is None:
            lattice_length = float(getattr(getattr(data, "geom", None), "l", 1.0))
            node_tolerance = max(lattice_length * 1e-4, 1e-8)
        node_tolerance = float(node_tolerance)
        if node_tolerance <= 0:
            raise ValueError("node_tolerance must be positive.")

        task_maps, node_masks, field_maps = {}, {}, {}
        for mode in DUAL_MODES:
            mode_coords = _reference_coordinates(data, mode)
            task_maps[mode] = _match_nodes(
                canonical_coords,
                mode_coords,
                node_tolerance,
                f"{mode} geometry",
            )
            node_mask = np.zeros(n_nodes, dtype=bool)
            node_mask[task_maps[mode]] = True
            node_masks[mode] = node_mask

            values = getattr(data, f"{mode}_field_input_values")
            field_coords = getattr(data, f"{mode}_field_input_node_coords", None)
            if field_coords is None:
                if values.shape[2] != len(mode_coords):
                    raise ValueError(
                        f"{mode}: field coordinates are missing and field node count does not match geometry."
                    )
                field_coords = mode_coords
            if len(field_coords) != values.shape[2]:
                raise ValueError(f"{mode}: field coordinate count does not match field node count.")
            field_maps[mode] = _match_nodes(
                canonical_coords,
                field_coords,
                node_tolerance,
                f"{mode} field",
            )

        full_delta = getattr(data, "UT_dINr_df")
        design_columns = set(getattr(data, "UT_dIN_df").columns)
        paired_columns = np.asarray(full_delta.columns).reshape(-1, 2)
        designable = np.asarray(
            [any(column in design_columns for column in pair) for pair in paired_columns],
            dtype=np.float32,
        )
        if len(designable) != n_nodes:
            raise ValueError("UT designable-node mask cannot be aligned to the canonical body nodes.")

        task_features, base_feature_names, context_spec = dual_node_context(
            canonical_coords, designable, node_masks, profile=context_profile
        )
        extra_names = list(extra_task_feature_names or [])
        extra_feature_count = None
        for mode in DUAL_MODES:
            features = task_features[mode]

            if extra_task_features is not None:
                extra = np.asarray(
                    _mode_value(extra_task_features, mode, "extra_task_features"),
                    dtype=np.float32,
                )
                if extra.ndim == 1:
                    extra = extra[:, None]
                if extra.shape[0] == len(task_maps[mode]):
                    canonical_extra = np.zeros((n_nodes, extra.shape[1]), dtype=np.float32)
                    canonical_extra[task_maps[mode]] = extra
                    extra = canonical_extra
                if extra.ndim != 2 or extra.shape[0] != n_nodes:
                    raise ValueError(
                        f"{mode} extra_task_features must use canonical or native task node rows."
                    )
                if extra_feature_count is None:
                    extra_feature_count = extra.shape[1]
                    if not extra_names:
                        extra_names = [f"task_feature_{idx}" for idx in range(extra_feature_count)]
                elif extra.shape[1] != extra_feature_count:
                    raise ValueError("UT and FT extra task features must have the same feature count.")
                if extra_names and len(extra_names) != extra.shape[1]:
                    raise ValueError("extra_task_feature_names must match extra task feature columns.")
                features = np.concatenate([features, extra], axis=1)
            task_features[mode] = features

        split_payloads, split_ids = {}, {}
        for split in DUAL_SPLITS:
            ut_df = getattr(data, f"UT_{split}_in_df")
            ft_df = getattr(data, f"FT_{split}_in_df")
            ut_ids, ft_ids = pd.Index(ut_df.index), pd.Index(ft_df.index)
            if not ut_ids.equals(ft_ids):
                raise ValueError(f"{split}: UT and FT sample ids/order are not identical.")
            ids = ut_ids
            split_ids[split] = ids.tolist()

            geometry_flat = getattr(data, "UT_dINr_df").loc[ids].to_numpy(dtype=np.float32)
            if geometry_flat.shape[1] != n_nodes * 2:
                raise ValueError(f"{split}: canonical UT geometry width does not match {n_nodes} nodes.")
            geometry = geometry_flat.reshape(len(ids), n_nodes, 2)
            ft_delta = getattr(data, "FT_dINr_df").loc[ids].to_numpy(dtype=float).reshape(len(ids), -1, 2)
            if not np.allclose(geometry[:, task_maps["FT"]], ft_delta, atol=node_tolerance, rtol=0):
                raise ValueError(f"{split}: retained FT disorder differs from canonical UT; cannot infer FT pin roles.")

            fields, field_masks, curves = {}, {}, {}
            for mode in DUAL_MODES:
                field_index = pd.Index(getattr(data, f"{mode}_field_input_index"))
                positions = field_index.get_indexer(ids)
                if np.any(positions < 0):
                    missing = ids[positions < 0].tolist()
                    raise KeyError(f"{split}: {mode} fields are missing sample ids {missing[:10]}.")
                values = getattr(data, f"{mode}_field_input_values")[positions]
                valid = getattr(data, f"{mode}_field_input_valid_mask")[positions]
                flat, flat_mask = _data_field_flatten(values, valid)
                canonical_field = np.zeros((len(ids), n_nodes, flat.shape[2]), dtype=np.float32)
                canonical_valid = np.zeros_like(canonical_field, dtype=bool)
                canonical_field[:, field_maps[mode], :] = np.nan_to_num(
                    flat,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                canonical_valid[:, field_maps[mode], :] = flat_mask
                fields[mode], field_masks[mode] = canonical_field, canonical_valid

                curve_df = getattr(data, f"{mode}_{split}_out_df")
                curves[mode] = curve_df.loc[ids].to_numpy(dtype=np.float32)

            split_payloads[split] = {
                "geometry": geometry,
                "field": fields,
                "field_mask": field_masks,
                "curve": curves,
            }

        metadata = {
            "source": "DATA(input_kind='field', output_kind='curve', mechMode='MULTI')",
            "source_path": getattr(data, "PATH", None),
            "canonical_coords": canonical_coords,
            "context_spec": context_spec,
            "canonical_node_count": n_nodes,
            "node_tolerance": node_tolerance,
            "task_to_canonical": task_maps,
            "field_to_canonical": field_maps,
            "field_frame_values": {
                mode: getattr(data, f"{mode}_field_input_frame_values", None) for mode in DUAL_MODES
            },
            "field_components": {
                mode: getattr(data, f"{mode}_field_input_components", None) for mode in DUAL_MODES
            },
            "curve_x_values": {
                mode: getattr(data, f"{mode}_OUT_df")
                .iloc[0]
                .loc[getattr(data, f"{mode}_train_out_df").columns]
                .to_numpy(dtype=float)
                for mode in DUAL_MODES
            },
        }
        return cls(
            split_payloads,
            node_masks,
            task_features,
            sample_ids=split_ids,
            normalize=normalize,
            metadata=metadata,
            geometry_feature_names=["dx", "dy"],
            task_feature_names=base_feature_names + extra_names,
        )

    @classmethod
    def from_files(
        cls,
        path="HPC",
        normalize=True,
        node_tolerance=None,
        context_profile="fcc_ti",
        extra_task_features=None,
        extra_task_feature_names=None,
        **data_kwargs,
    ):
        required = {
            "load": True,
            "mechMode": "MULTI",
            "model": "TR",
            "input_kind": "field",
            "output_kind": "curve",
            "scale": False,
            "reduce_dim": False,
            "geom_feats": False,
        }
        conflicts = {
            key: data_kwargs[key]
            for key, expected in required.items()
            if key in data_kwargs and data_kwargs[key] != expected
        }
        if conflicts:
            raise ValueError(f"DUAL_DATA.from_files owns these DATA settings: {conflicts}.")
        config = {"d_data": "in", **data_kwargs, **required, "path": path}
        return cls.from_data(
            DATA(**config),
            normalize=normalize,
            node_tolerance=node_tolerance,
            context_profile=context_profile,
            extra_task_features=extra_task_features,
            extra_task_feature_names=extra_task_feature_names,
        )

    def make_dataloaders(self, batch_size=4, num_workers=0, pin_memory=False):
        return {
            split: DataLoader(
                self.datasets[split],
                batch_size=int(batch_size),
                shuffle=(split == "train"),
                num_workers=int(num_workers),
                pin_memory=bool(pin_memory),
            )
            for split in DUAL_SPLITS
        }

    def inverse_field(self, mode, values):
        mode = str(mode).upper().replace("CT", "FT")
        stats = self.normalizers["field"][mode]
        return np.asarray(values) * stats["scale"] + stats["mean"]

    def inverse_curve(self, mode, values):
        mode = str(mode).upper().replace("CT", "FT")
        stats = self.normalizers["curve"][mode]
        return np.asarray(values) * stats["scale"] + stats["mean"]

    def to_metadata(self):
        return {
            "class": self.__class__.__name__,
            "split_sizes": {split: len(self.datasets[split]) for split in DUAL_SPLITS},
            "sample_ids": self.sample_ids,
            "n_nodes": self.n_nodes,
            "geometry_feature_names": self.geometry_feature_names,
            "task_feature_names": self.task_feature_names,
            "reference_task_features": self.task_features,
            "field_feature_sizes": self.field_feature_sizes,
            "curve_sizes": self.curve_sizes,
            "node_masks": self.node_masks,
            "normalization": self.normalization,
            "normalizers": self.normalizers,
            "metadata": self.metadata,
        }

    def to_json(self, path, indent=2):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(self.to_metadata()), indent=indent), encoding="utf-8")
        return str(path)


def _sinusoidal_encoding(length, d_model):
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    encoding = torch.zeros(length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class DualTaskTransformer(nn.Module):
    """One shared Transformer encoder with UT and FT task conditioning/heads.

    ``tokenizer`` is a per-node linear projection: it changes feature width but
    does not mix information between nodes. Cross-node mixing begins only in
    ``encoder``, after each task's node mask has been applied.
    """

    def __init__(
        self,
        in_size,
        out_size,
        seq_len,
        context_size=0,
        d_model=128,
        n_heads=4,
        n_layers=3,
        ff_mult=4,
        dropout=0.1,
        activation="gelu",
        pool="node",
        pos_encoding="none",
        use_cls_token=None,
        bias=True,
    ):
        super().__init__()
        self.in_sizes = _mode_sizes(in_size, "in_size")
        self.out_sizes = _mode_sizes(out_size, "out_size")
        self.seq_len = int(seq_len)
        self.context_size = int(context_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_mult = int(ff_mult)
        self.dropout = float(dropout)
        self.activation = str(activation).lower()
        self.pool = str(pool).lower()
        self.pos_encoding = str(pos_encoding).lower()
        self.bias = bool(bias)
        self.use_cls_token = self.pool == "cls" if use_cls_token is None else bool(use_cls_token)

        if self.seq_len < 1:
            raise ValueError("seq_len must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.pool not in {"node", "mean", "cls"}:
            raise ValueError("pool must be 'node', 'mean', or 'cls'.")
        if self.pool == "cls" and not self.use_cls_token:
            raise ValueError("pool='cls' requires use_cls_token=True.")
        if self.pos_encoding not in {"none", "learned", "sinusoidal"}:
            raise ValueError("pos_encoding must be 'none', 'learned', or 'sinusoidal'.")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'.")

        shared_input = len(set(self.in_sizes.values())) == 1
        self.shared_tokenizer = shared_input
        if shared_input:
            self.tokenizer = nn.Linear(next(iter(self.in_sizes.values())), self.d_model, bias=self.bias)
        else:
            self.tokenizer = nn.ModuleDict(
                {
                    mode: nn.Linear(self.in_sizes[mode], self.d_model, bias=self.bias)
                    for mode in DUAL_MODES
                }
            )
        self.context_projection = (
            nn.Linear(self.context_size, self.d_model, bias=False) if self.context_size else None
        )
        self.task_embedding = nn.Embedding(len(DUAL_MODES), self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if self.use_cls_token else None

        total_len = self.seq_len + int(self.use_cls_token)
        self.pos_embed = None
        self.register_buffer("sinusoidal_pos", None, persistent=False)
        if self.pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, total_len, self.d_model))
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        elif self.pos_encoding == "sinusoidal":
            self.sinusoidal_pos = _sinusoidal_encoding(total_len, self.d_model)
        else:
            self.sinusoidal_pos = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ff_mult,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            bias=self.bias,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.heads = nn.ModuleDict(
            {
                mode: nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model, self.out_sizes[mode], bias=self.bias),
                )
                for mode in DUAL_MODES
            }
        )
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _tokenize(self, mode, values):
        tokenizer = self.tokenizer if self.shared_tokenizer else self.tokenizer[mode]
        return tokenizer(values)

    def _node_mask(self, masks, mode, batch_size, device):
        if masks is None:
            return torch.ones(batch_size, self.seq_len, dtype=torch.bool, device=device)
        mask = torch.as_tensor(_mode_value(masks, mode, "node_masks"), dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, self.seq_len):
            raise ValueError(
                f"{mode} node mask must have shape [{batch_size}, {self.seq_len}], got {tuple(mask.shape)}."
            )
        if not torch.all(mask.any(dim=1)):
            raise ValueError(f"{mode} node mask leaves at least one sample with no active nodes.")
        return mask

    def _task_context(self, task_features, mode, batch_size, device, dtype):
        if self.context_projection is None:
            return None
        if task_features is None:
            raise ValueError("task_features are required when context_size is nonzero.")
        features = torch.as_tensor(
            _mode_value(task_features, mode, "task_features"),
            dtype=dtype,
            device=device,
        )
        if features.ndim == 2:
            features = features.unsqueeze(0).expand(batch_size, -1, -1)
        expected = (batch_size, self.seq_len, self.context_size)
        if tuple(features.shape) != expected:
            raise ValueError(f"{mode} task features must have shape {expected}, got {tuple(features.shape)}.")
        return self.context_projection(features)

    def forward(self, inputs, task_features=None, node_masks=None):
        tokens, masks = [], []
        batch_size = None
        for task_index, mode in enumerate(DUAL_MODES):
            values = _mode_value(inputs, mode, "inputs") if isinstance(inputs, Mapping) else inputs
            if values.ndim != 3:
                raise ValueError(f"{mode} input must have shape [batch, nodes, features].")
            if values.shape[1:] != (self.seq_len, self.in_sizes[mode]):
                raise ValueError(
                    f"{mode} input must have trailing shape ({self.seq_len}, {self.in_sizes[mode]}), "
                    f"got {tuple(values.shape[1:])}."
                )
            if batch_size is None:
                batch_size = int(values.shape[0])
            elif values.shape[0] != batch_size:
                raise ValueError("UT and FT inputs must have the same batch size.")

            mask = self._node_mask(node_masks, mode, batch_size, values.device)
            masked_values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
            task_tokens = self._tokenize(mode, masked_values)
            context = self._task_context(
                task_features,
                mode,
                batch_size,
                values.device,
                task_tokens.dtype,
            )
            if context is not None:
                task_tokens = task_tokens + context
            task_id = torch.full((batch_size,), task_index, dtype=torch.long, device=values.device)
            task_tokens = task_tokens + self.task_embedding(task_id).unsqueeze(1)

            if self.use_cls_token:
                cls = self.cls_token.expand(batch_size, -1, -1) + self.task_embedding(task_id).unsqueeze(1)
                task_tokens = torch.cat([cls, task_tokens], dim=1)
                mask = torch.cat(
                    [torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device), mask],
                    dim=1,
                )
            if self.pos_embed is not None:
                task_tokens = task_tokens + self.pos_embed
            elif self.sinusoidal_pos is not None:
                task_tokens = task_tokens + self.sinusoidal_pos.to(dtype=task_tokens.dtype)
            tokens.append(task_tokens)
            masks.append(mask)

        stacked_tokens = torch.cat(tokens, dim=0)
        stacked_mask = torch.cat(masks, dim=0)
        encoded = self.encoder(stacked_tokens, src_key_padding_mask=~stacked_mask)
        encoded_by_task = dict(zip(DUAL_MODES, encoded.split(batch_size, dim=0)))

        outputs = {}
        for mode, task_encoded, mask in zip(DUAL_MODES, encoded_by_task.values(), masks):
            node_mask = mask[:, 1:] if self.use_cls_token else mask
            if self.pool == "node":
                representation = task_encoded[:, 1:, :] if self.use_cls_token else task_encoded
                output = self.heads[mode](representation)
                output = output.masked_fill(~node_mask.unsqueeze(-1), 0.0)
            elif self.pool == "cls":
                output = self.heads[mode](task_encoded[:, 0, :])
            else:
                node_encoded = task_encoded[:, 1:, :] if self.use_cls_token else task_encoded
                weights = node_mask.unsqueeze(-1).to(node_encoded.dtype)
                pooled = (node_encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                output = self.heads[mode](pooled)
            outputs[mode] = output
        return outputs

    def get_config(self):
        return {
            "in_size": self.in_sizes,
            "out_size": self.out_sizes,
            "seq_len": self.seq_len,
            "context_size": self.context_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "ff_mult": self.ff_mult,
            "dropout": self.dropout,
            "activation": self.activation,
            "pool": self.pool,
            "pos_encoding": self.pos_encoding,
            "use_cls_token": self.use_cls_token,
            "bias": self.bias,
        }


class DualStageTransformer(nn.Module):
    """Two jointly trained dual-task Transformers connected in series."""

    def __init__(self, field_model, curve_model):
        super().__init__()
        if field_model.pool != "node":
            raise ValueError("field_model must use pool='node'.")
        if curve_model.pool == "node":
            raise ValueError("curve_model must use global pooling for curve outputs.")
        if field_model.seq_len != curve_model.seq_len:
            raise ValueError("Field and curve stages must use the same canonical node count.")
        if field_model.out_sizes != curve_model.in_sizes:
            raise ValueError("Field output sizes must match curve-stage input sizes for each task.")
        self.field_model = field_model
        self.curve_model = curve_model

    @classmethod
    def from_data(cls, data, field_kwargs=None, curve_kwargs=None):
        field_kwargs = dict(field_kwargs or {})
        curve_kwargs = dict(curve_kwargs or {})
        field_model = DualTaskTransformer(
            in_size=data.geometry_feature_size,
            out_size=data.field_feature_sizes,
            seq_len=data.n_nodes,
            context_size=data.context_feature_size,
            pool="node",
            **field_kwargs,
        )
        curve_model = DualTaskTransformer(
            in_size=data.field_feature_sizes,
            out_size=data.curve_sizes,
            seq_len=data.n_nodes,
            context_size=data.context_feature_size,
            pool=curve_kwargs.pop("pool", "cls"),
            **curve_kwargs,
        )
        return cls(field_model, curve_model)

    def forward(self, geometry, task_features=None, node_masks=None):
        fields = self.field_model(
            geometry,
            task_features=task_features,
            node_masks=node_masks,
        )
        curves = self.curve_model(
            fields,
            task_features=task_features,
            node_masks=node_masks,
        )
        return {"field": fields, "curve": curves}

    def get_config(self):
        return {
            "field_model": self.field_model.get_config(),
            "curve_model": self.curve_model.get_config(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            DualTaskTransformer(**dict(config["field_model"])),
            DualTaskTransformer(**dict(config["curve_model"])),
        )


class DualLoss(nn.Module):
    """One scalar objective containing UT/FT field and curve losses."""

    def __init__(self, field_loss=None, curve_loss=None, weights=None, scales=None, curve_normalizers=None):
        super().__init__()
        field_loss = MaskedFieldMSELoss() if field_loss is None else field_loss
        curve_loss = nn.MSELoss() if curve_loss is None else curve_loss
        self.field_losses = self._losses_by_mode(field_loss, "field_loss")
        self.curve_losses = self._losses_by_mode(curve_loss, "curve_loss")
        self.weights = self._nested_values(weights, default=1.0, name="weights")
        self.scales = self._nested_values(scales, default=1.0, name="scales")
        # Curve-aware losses need physical peak/energy values. This fixed affine
        # inverse stays in Torch so curve gradients still reach the field stage.
        self.curve_normalizers = curve_normalizers
        for mode in DUAL_MODES:
            stats = curve_normalizers[mode] if curve_normalizers is not None else {"mean": 0.0, "scale": 1.0}
            self.register_buffer(f"curve_mean_{mode}", torch.as_tensor(stats["mean"], dtype=torch.float32))
            self.register_buffer(f"curve_scale_{mode}", torch.as_tensor(stats["scale"], dtype=torch.float32))
        for kind in ("field", "curve"):
            for mode in DUAL_MODES:
                if self.weights[kind][mode] < 0:
                    raise ValueError("Dual loss weights must be non-negative.")
                if self.scales[kind][mode] <= 0:
                    raise ValueError("Dual loss scales must be positive.")

    @staticmethod
    def _losses_by_mode(value, name):
        if isinstance(value, Mapping):
            losses = {mode: _mode_value(value, mode, name) for mode in DUAL_MODES}
        else:
            losses = {mode: copy.deepcopy(value) for mode in DUAL_MODES}
        if not all(isinstance(loss, nn.Module) for loss in losses.values()):
            raise TypeError(f"{name} entries must be torch.nn.Module instances.")
        return nn.ModuleDict(losses)

    @staticmethod
    def _nested_values(value, default, name):
        if value is None:
            return {kind: {mode: float(default) for mode in DUAL_MODES} for kind in ("field", "curve")}
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a nested field/curve mapping.")
        resolved = {}
        for kind in ("field", "curve"):
            kind_value = value.get(kind, default)
            if isinstance(kind_value, Mapping):
                resolved[kind] = {
                    mode: float(_mode_value(kind_value, mode, f"{name}['{kind}']"))
                    for mode in DUAL_MODES
                }
            else:
                resolved[kind] = {mode: float(kind_value) for mode in DUAL_MODES}
        return resolved

    def forward(self, predictions, targets, field_masks=None):
        raw = {"field": {}, "curve": {}}
        weighted = {"field": {}, "curve": {}}
        total = None
        for mode in DUAL_MODES:
            mask = None if field_masks is None else _mode_value(field_masks, mode, "field_masks")
            field_prediction = predictions["field"][mode]
            field_target = targets["field"][mode]
            field_loss = self.field_losses[mode]
            if isinstance(field_loss, MaskedFieldMSELoss):
                raw_field = field_loss(field_prediction, field_target, mask=mask)
            else:
                valid = torch.isfinite(field_target) & torch.isfinite(field_prediction)
                if mask is not None:
                    valid &= torch.broadcast_to(mask.bool(), field_target.shape)
                raw_field = (
                    field_loss(field_prediction[valid], field_target[valid])
                    if torch.any(valid)
                    else field_prediction.sum() * 0.0
                )
            raw_curve = self.curve_losses[mode](
                predictions["curve"][mode] * getattr(self, f"curve_scale_{mode}") + getattr(self, f"curve_mean_{mode}"),
                targets["curve"][mode] * getattr(self, f"curve_scale_{mode}") + getattr(self, f"curve_mean_{mode}"),
            )
            raw["field"][mode], raw["curve"][mode] = raw_field, raw_curve
            weighted["field"][mode] = (
                self.weights["field"][mode] * raw_field / self.scales["field"][mode]
            )
            weighted["curve"][mode] = (
                self.weights["curve"][mode] * raw_curve / self.scales["curve"][mode]
            )
            mode_total = weighted["field"][mode] + weighted["curve"][mode]
            total = mode_total if total is None else total + mode_total
        return total, {"raw": raw, "weighted": weighted}

    def get_config(self):
        return {
            "weights": self.weights,
            "scales": self.scales,
            "curve_normalizers": _json_safe(self.curve_normalizers),
            "field_losses": {
                mode: _model_loss_to_config(self.field_losses[mode]) for mode in DUAL_MODES
            },
            "curve_losses": {
                mode: _model_loss_to_config(self.curve_losses[mode]) for mode in DUAL_MODES
            },
        }


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


class DUAL_MODEL:
    """Joint trainer for one :class:`DualStageTransformer` and one optimizer."""

    def __init__(
        self,
        model,
        lossf,
        data=None,
        opt=("adamw", 0.0),
        batch=4,
        lr=2e-4,
        scheduler=None,
        device=None,
        dataloaders=None,
        num_workers=0,
        pin_memory=None,
    ):
        if not isinstance(model, DualStageTransformer):
            raise TypeError("DUAL_MODEL requires a DualStageTransformer.")
        if not isinstance(lossf, DualLoss):
            raise TypeError("DUAL_MODEL requires a DualLoss.")
        self.model = model
        self.lossf = lossf
        self.data = data
        self.opt_cfg = opt
        self.batch = int(batch)
        self.lr = float(lr)
        self.scheduler_cfg = scheduler
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.lossf.to(self.device)
        self.optimizer = _model_optimizer(self.model.parameters(), opt=self.opt_cfg, lr=self.lr)
        self.scheduler = _model_scheduler(self.optimizer, self.scheduler_cfg)
        if dataloaders is None:
            if data is None:
                raise ValueError("Pass DUAL_DATA or explicit train/val/test dataloaders.")
            pin_memory = self.device.type == "cuda" if pin_memory is None else bool(pin_memory)
            dataloaders = data.make_dataloaders(
                batch_size=self.batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        self.dataloaders = dict(dataloaders)
        if "train" not in self.dataloaders:
            raise KeyError("dataloaders must contain a train loader.")
        self.history = []
        self.best_epoch = 0
        self.best_loss = float("inf")

    @staticmethod
    def _flatten_components(details):
        return {
            f"{group}_{kind}_{mode}": float(value.detach().item())
            for group, grouped in details.items()
            for kind, modes in grouped.items()
            for mode, value in modes.items()
        }

    def _run_loader(self, loader, training):
        self.model.train(training)
        totals = {}
        n_samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = _move_to_device(batch, self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                predictions = self.model(
                    batch["geometry"],
                    task_features=batch["task_features"],
                    node_masks=batch["node_mask"],
                )
                targets = {"field": batch["field"], "curve": batch["curve"]}
                loss, details = self.lossf(
                    predictions,
                    targets,
                    field_masks=batch["field_mask"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite joint loss; refusing to continue training.")
                if training:
                    loss.backward()
                    self.optimizer.step()
                batch_size = int(batch["geometry"].shape[0])
                values = {"loss": float(loss.detach().item()), **self._flatten_components(details)}
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + value * batch_size
                n_samples += batch_size
        if n_samples == 0:
            raise ValueError("Cannot evaluate an empty dual dataloader.")
        return {key: value / n_samples for key, value in totals.items()}

    def train(
        self,
        n_epochs,
        verbose=1,
        early_stop_patience=None,
        early_stop_delta=0.0,
        checkpoint_path=None,
        metadata=None,
    ):
        n_epochs = int(n_epochs)
        if n_epochs < 1:
            raise ValueError("n_epochs must be positive.")
        patience_count = 0
        best_state = None
        val_loader = self.dataloaders.get("val")

        for epoch in range(1, n_epochs + 1):
            train_metrics = self._run_loader(self.dataloaders["train"], training=True)
            val_metrics = self._run_loader(val_loader, training=False) if val_loader is not None else train_metrics
            monitored = val_metrics["loss"]
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            self.history.append(row)

            if monitored < self.best_loss - float(early_stop_delta):
                self.best_loss = monitored
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_count = 0
                if checkpoint_path is not None:
                    self.save(checkpoint_path, metadata=metadata)
            else:
                patience_count += 1

            if self.scheduler is not None:
                self.scheduler.step(monitored)
            if checkpoint_path is not None:
                pd.DataFrame(self.history).to_csv(Path(self.model_file).with_name("loss_history.csv"), index=False)
            if verbose and (epoch == 1 or epoch == n_epochs or epoch % max(1, int(verbose)) == 0):
                print(
                    f"Dual epoch {epoch}/{n_epochs} | train={train_metrics['loss']:.6f} "
                    f"| val={val_metrics['loss']:.6f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
                )
            if early_stop_patience is not None and patience_count >= int(early_stop_patience):
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, split="test"):
        split = str(split).lower()
        if split not in self.dataloaders:
            raise KeyError(f"No '{split}' dataloader is configured.")
        prediction_parts = {kind: {mode: [] for mode in DUAL_MODES} for kind in ("field", "curve")}
        truth_parts = {kind: {mode: [] for mode in DUAL_MODES} for kind in ("field", "curve")}
        sample_ids = []
        mask_parts = {mode: [] for mode in DUAL_MODES}
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloaders[split]:
                sample_ids.extend(list(batch["sample_id"]))
                for mode in DUAL_MODES:
                    mask_parts[mode].append(batch["field_mask"][mode].cpu())
                batch = _move_to_device(batch, self.device)
                predictions = self.model(
                    batch["geometry"],
                    task_features=batch["task_features"],
                    node_masks=batch["node_mask"],
                )
                for kind in ("field", "curve"):
                    for mode in DUAL_MODES:
                        prediction_parts[kind][mode].append(predictions[kind][mode].detach().cpu())
                        truth_parts[kind][mode].append(batch[kind][mode].detach().cpu())

        if not sample_ids:
            raise ValueError(f"Cannot predict an empty '{split}' split.")
        predictions = {
            kind: {mode: torch.cat(parts).numpy() for mode, parts in modes.items()}
            for kind, modes in prediction_parts.items()
        }
        truth = {
            kind: {mode: torch.cat(parts).numpy() for mode, parts in modes.items()}
            for kind, modes in truth_parts.items()
        }
        result = {
            "prediction": predictions, "truth": truth, "sample_id": sample_ids,
            "field_mask": {mode: torch.cat(parts).numpy() for mode, parts in mask_parts.items()},
        }
        self.predictions = getattr(self, "predictions", {})
        self.predictions[split] = result
        return result

    def save(self, path, include_optimizer=False, metadata=None):
        path = Path(path)
        if path.suffix.lower() != ".mdl":
            path = path / "model.mdl"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = {
            "kind": "dual-stage-transformer",
            "version": 1,
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model_config": self.model.get_config(),
            "loss_config": self.lossf.get_config(),
            "training": {
                "opt": self.opt_cfg,
                "batch": self.batch,
                "lr": self.lr,
                "scheduler": self.scheduler_cfg,
                "best_epoch": self.best_epoch,
                "best_loss": self.best_loss,
            },
            "data": self.data.to_metadata() if self.data is not None else None,
            "run_layout": (metadata or {}).get("run_layout", {}),
            "metadata": metadata or {},
        }
        descriptor = _json_safe(descriptor)
        payload = {
            "version": 1,
            "model_state_dict": self.model.state_dict(),
            "descriptor": descriptor,
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(payload, path)
        path.with_suffix(".json").write_text(
            json.dumps(_json_safe(descriptor), indent=2),
            encoding="utf-8",
        )
        if self.history:
            pd.DataFrame(self.history).to_csv(path.with_name("loss_history.csv"), index=False)
        self.model_file = str(path)
        self.save_dir = str(path.parent)
        return str(path)

    def save_results(self, path=None, eval_split="val", run_config=None, metadata=None):
        """Save all four physical-unit predictions using existing diagnostics.

        Stored fields retain the canonical axis and masks. Field diagnostic
        tables exclude absent task nodes and use train-only mean baselines.
        """
        from resources.MLfunc import curve_default_zone_boundaries
        from resources.MLmetrics import curve_performance_diagnostics, field_performance_diagnostics

        if self.data is None:
            raise ValueError("Dual result diagnostics require DUAL_DATA.")
        results_dir = Path(path) if path is not None else Path(self.save_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result = self.predict(eval_split)
        arrays = {"sample_ids": np.asarray(result["sample_id"])}
        summaries = {"field": {}, "curve": {}}
        for mode in DUAL_MODES:
            present = self.data.node_masks[mode]
            arrays[f"{mode}_node_mask"] = present
            for kind in ("field", "curve"):
                inverse = getattr(self.data, f"inverse_{kind}")
                pred = inverse(mode, result["prediction"][kind][mode])
                truth = inverse(mode, result["truth"][kind][mode])
                train = self.data.splits["train"][kind][mode]
                if kind == "field":
                    mask = result["field_mask"][mode]
                    pred[:, ~present, :] = 0.0
                    truth = np.where(mask, truth, np.nan)
                    arrays[f"{mode}_{eval_split}_field_mask"] = mask
                    train_mask = self.data.splits["train"]["field_mask"][mode]
                    # Normalized invalid entries are already zero; avoid a second
                    # full training field allocation merely to compute a mean.
                    counts = train_mask.sum(axis=0, keepdims=True)
                    mean = train.sum(axis=0, keepdims=True, dtype=np.float64) / np.maximum(counts, 1)
                    baseline = np.where(counts > 0, inverse(mode, mean), np.nan)
                    components = self.data.metadata["field_components"][mode]
                    frames = self.data.metadata["field_frame_values"][mode]
                    shape = (train.shape[-1] // len(components), int(present.sum()), len(components))
                    coords = self.data.metadata.get("canonical_coords")
                    diag = field_performance_diagnostics(
                        pred[:, present], truth[:, present], field_shape=shape,
                        frame_values=frames, components=components,
                        node_labels=np.flatnonzero(present),
                        node_coords=None if coords is None else np.asarray(coords)[present],
                        train_truth=baseline[:, present],
                    )
                    arrays[f"{mode}_frame_values"] = frames
                    arrays[f"{mode}_components"] = np.asarray(components)
                else:
                    x = self.data.metadata["curve_x_values"][mode]
                    diag = curve_performance_diagnostics(
                        pred, truth, x_values=x, train_truth=inverse(mode, train),
                        zone_boundaries=curve_default_zone_boundaries(mode),
                    )
                    arrays[f"{mode}_curve_x_values"] = x
                arrays[f"{mode}_{eval_split}_{kind}_outputs"] = pred
                arrays[f"{mode}_{eval_split}_{kind}_truth"] = truth
                summaries[kind][mode] = diag["summary"]
                for key, table in diag.items():
                    if isinstance(table, pd.DataFrame):
                        table = table.copy()
                        if key == "sample_metrics":
                            table["sample_id"] = result["sample_id"]
                        table.to_csv(results_dir / f"{mode}_{eval_split}_{kind}_{key}.csv", index=False)
        arrays["canonical_coords"] = np.asarray(self.data.metadata["canonical_coords"])
        np.savez(results_dir / "predictions.npz", **arrays)
        metrics = {
            "kind": "dual-stage-transformer", "version": 1,
            "evaluation_split": eval_split, "prediction_space": "physical",
            "field_layout": "sample,node,frame*component", "node_index_base": 0,
            "best_epoch": self.best_epoch, "best_validation_loss": self.best_loss,
            "run_config": run_config or {}, "metadata": metadata or {},
            "split_sizes": {split: len(ds) for split, ds in self.data.datasets.items()},
            "diagnostics": summaries,
        }
        (results_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
        (results_dir / "diagnostics_summary.json").write_text(
            json.dumps(_json_safe(summaries), indent=2), encoding="utf-8"
        )
        pd.DataFrame(self.history).to_csv(results_dir / "loss_history.csv", index=False)
        self.results_dir = str(results_dir)
        return self.results_dir

    def load(self, path, load_optimizer=False, strict=True):
        state = torch.load(Path(path), map_location=self.device)
        payload = state if isinstance(state, Mapping) and "model_state_dict" in state else {"model_state_dict": state}
        self.model.load_state_dict(payload["model_state_dict"], strict=bool(strict))
        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        return self
