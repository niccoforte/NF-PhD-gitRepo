"""Opt-in, geometry-aware displacement supervision. No change to legacy defaults.

Fields use [batch, node, frame*component], with frame-major component order.
Spatial terms match signed differences, never penalise deformation itself.
Temporal terms match recorded-frame increments, not velocities.
"""
import numpy as np
import torch
from torch import nn


def field_loss_weights(variant, spatial=0.1, temporal=0.1, gain=0.):
    """Explicit ablation switches; weighted requires an intentional positive gain."""
    if variant not in {"baseline", "spatial", "temporal", "both", "weighted"}:
        raise ValueError("Unknown field-loss variant.")
    if min(spatial, temporal, gain) < 0 or not np.isfinite([spatial, temporal, gain]).all():
        raise ValueError("Field-loss settings must be finite and nonnegative.")
    if variant == "weighted" and gain <= 0:
        raise ValueError("Weighted ablation requires an explicit positive localization gain.")
    return dict(spatial_weight=spatial if variant in {"spatial", "both", "weighted"} else 0.,
                temporal_weight=temporal if variant in {"temporal", "both", "weighted"} else 0.,
                localization_gain=gain if variant == "weighted" else 0.)


def reference_field_edges(coords, mode="UT", present=None):
    """Reconstruct the current FCC body graph with the existing periodic algorithm.

    Normalise reference coordinates to the producer's cell size 10 because its
    parity rule is scale dependent. FT additionally applies the A1 crack cut.
    Other geometries should supply their validated edge list directly to the loss.
    """
    from resources.lattices import Geometry, connectivity, insidePoint
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite periodic [node,2] coordinates.")
    span = np.ptp(xy, axis=0)
    if not np.isclose(span[1] / span[0], 19 / 20):
        raise ValueError("Automatic edges support the established 20-by-19 FCC body only.")
    native = (xy - xy.min(axis=0)) * (200 / span[0])
    expected = {(10*x, 10*y) for y in range(20) for x in range(21)}
    expected |= {(10*x+5, 10*y+5) for y in range(19) for x in range(20)}
    actual = {tuple(row) for row in np.round(native, 5)}
    removed = {(10*x+5, 95) for x in range(12)}
    mode = str(mode).upper()
    if mode not in {"UT", "FT"} or actual not in (expected, expected-removed):
        raise ValueError("Coordinates must be the periodic FCC body or its native FT subset.")
    if mode == "UT" and actual != expected:
        raise ValueError("UT requires the full body.")
    native = np.round(native, 5)  # restore exact periodic parity after unit conversion
    nodes = np.column_stack([np.arange(1, len(xy)+1), native])
    edges = connectivity("FCC", nodes, Geometry("FCC", 10, 20))[:, 1:].astype(int)-1
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    if mode == "FT":
        retain = np.array([tuple(row) not in removed for row in np.round(native, 5)])
        cut = np.array([insidePoint((-16, 93), (117.6, 97), p)
                        for p in native[edges].mean(axis=1)])
        edges = edges[retain[edges].all(axis=1) & ~cut]
    if present is not None:
        edges = edges[np.asarray(present, dtype=bool)[edges].all(axis=1)]
    return edges


class StructuredFieldLoss(nn.Module):
    """Masked MSE + signed spatial and temporal differences in physical units.

    Localisation weights depend only on each target's jumps, never its location
    or predicted error. The unnormalised weight is in [1,1+localization_gain].
    Each specimen's weights are normalised over its valid values. gain=0 disables
    weighting exactly; spatial_weight=temporal_weight=0 recovers masked MSE.
    """
    structured_field = True

    def __init__(self, edges, n_nodes, n_components=2, mean=0., scale=1.,
                 spatial_scale=1., temporal_scale=1., spatial_weight=0.1,
                 temporal_weight=0.1, localization_gain=0., eps=1e-8):
        super().__init__()
        self.n_nodes, self.n_components = int(n_nodes), int(n_components)
        self.spatial_weight, self.temporal_weight = float(spatial_weight), float(temporal_weight)
        self.localization_gain, self.eps = float(localization_gain), float(eps)
        edge = np.asarray(edges, dtype=int).reshape(-1, 2)
        if len(edge) == 0 or edge.min() < 0 or edge.max() >= n_nodes or np.any(edge[:, 0] == edge[:, 1]):
            raise ValueError("Require nonempty valid, non-self edges.")
        if len(np.unique(np.sort(edge, axis=1), axis=0)) != len(edge):
            raise ValueError("Edges must be unique undirected pairs.")
        if (not np.isfinite([self.spatial_weight, self.temporal_weight, self.localization_gain, eps]).all()
                or min(self.spatial_weight, self.temporal_weight, self.localization_gain) < 0 or eps <= 0):
            raise ValueError("Loss weights must be nonnegative and eps positive.")
        self.register_buffer("edges", torch.as_tensor(edge, dtype=torch.long))
        for name, value in (("mean", mean), ("scale", scale),
                            ("spatial_scale", spatial_scale), ("temporal_scale", temporal_scale)):
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if not torch.isfinite(tensor).all() or (name != "mean" and torch.any(tensor <= 0)):
                raise ValueError(f"Invalid {name}.")
            self.register_buffer(name, tensor)
        self.last_components = {}

    def get_config(self):
        return {**{k: getattr(self, k) for k in ("n_nodes", "n_components", "spatial_weight",
                "temporal_weight", "localization_gain", "eps")},
                **{k: getattr(self, k).detach().cpu().tolist() for k in
                   ("edges", "mean", "scale", "spatial_scale", "temporal_scale")}}

    def _node_average(self, values, valid):
        shape = (values.shape[0], self.n_nodes, *values.shape[2:])
        total, count = values.new_zeros(shape), values.new_zeros(shape)
        for endpoint in (0, 1):
            total.index_add_(1, self.edges[:, endpoint], torch.where(valid, values, 0.))
            count.index_add_(1, self.edges[:, endpoint], valid.to(values.dtype))
        return total / count.clamp_min(1), count > 0

    def _prepare(self, pred, target, mask):
        if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1] != self.n_nodes:
            raise ValueError("Expected matching [batch,node,frame*component] fields.")
        if pred.shape[-1] % self.n_components:
            raise ValueError("Field width must be divisible by n_components.")
        valid = torch.isfinite(target)
        if mask is not None:
            valid &= torch.broadcast_to(torch.as_tensor(mask, device=target.device, dtype=torch.bool), target.shape)
        if torch.any(valid & ~torch.isfinite(pred)):
            raise FloatingPointError("Nonfinite prediction at a valid field target.")
        p, y = torch.where(valid, pred, 0.), torch.where(valid, target, 0.)
        shape = (*pred.shape[:2], -1, self.n_components)
        return p, y, (p*self.scale+self.mean).reshape(shape), (y*self.scale+self.mean).reshape(shape), valid.reshape(shape)

    @staticmethod
    def _mean(values, valid):
        return torch.where(valid, values, 0.).sum() / valid.sum().clamp_min(1)

    def component_losses(self, pred, target, mask=None):
        p, y, pu, yu, valid = self._prepare(pred, target, mask)
        i, j = self.edges.T
        edge_valid = valid[:, i] & valid[:, j]
        dp = (pu[:, j]-pu[:, i])/self.spatial_scale
        dy = (yu[:, j]-yu[:, i])/self.spatial_scale
        node_spatial, node_valid = self._node_average((dp-dy).square(), edge_valid)
        temporal_valid = valid[:, :, 1:] & valid[:, :, :-1]
        tp = torch.diff(pu, dim=2)/self.temporal_scale
        ty = torch.diff(yu, dim=2)/self.temporal_scale
        weights = torch.ones_like(yu)
        if self.localization_gain:
            with torch.no_grad():
                activity, _ = self._node_average(dy.square(), edge_valid)
                increments = torch.where(temporal_valid, ty.square(), 0.)
                temporal_activity = torch.zeros_like(yu)
                temporal_activity[:, :, 1:] += increments
                temporal_activity[:, :, :-1] += increments
                q = torch.sqrt((activity+temporal_activity).mean(dim=-1, keepdim=True).clamp_min(0))
                weights = (1+self.localization_gain*q/(1+q)).expand_as(yu)
                count = valid.sum(dim=(1,2,3), keepdim=True).clamp_min(1)
                avg = torch.where(valid, weights, 0.).sum(dim=(1,2,3), keepdim=True)/count
                weights = weights/avg.clamp_min(self.eps)
        components = {
            "displacement": self._mean((p-y).reshape_as(yu).square()*weights, valid),
            "spatial": self._mean(node_spatial, node_valid),
            "temporal": self._mean((tp-ty).square(), temporal_valid),
        }
        self.last_components = {k: v.detach() for k,v in components.items()}
        return components

    def forward(self, pred, target, mask=None):
        c = self.component_losses(pred, target, mask)
        return c["displacement"]+self.spatial_weight*c["spatial"]+self.temporal_weight*c["temporal"]


def fit_field_loss(train, coords, mode, n_components=2, mask=None, mean=0., scale=1., **kwargs):
    """Fit physical jump scales using training targets only, in bounded chunks."""
    edges = reference_field_edges(coords, mode)
    sums, counts = [np.zeros(n_components) for _ in range(2)], [np.zeros(n_components) for _ in range(2)]
    for start in range(0, len(train), 16):
        a = np.asarray(train[start:start+16], dtype=np.float64)
        v = np.isfinite(a)
        if mask is not None:
            v &= np.asarray(mask[start:start+16], dtype=bool)
        physical = (np.where(v,a,0)*np.asarray(scale)+np.asarray(mean)).reshape(len(a),len(coords),-1,n_components)
        v = v.reshape(physical.shape)
        i,j=edges.T
        differences = (physical[:,j]-physical[:,i], np.diff(physical,axis=2))
        validity = (v[:,j]&v[:,i], v[:,:,1:]&v[:,:,:-1])
        for k,(d,ok) in enumerate(zip(differences,validity)):
            sums[k] += np.where(ok,d*d,0).sum(axis=(0,1,2))
            counts[k] += ok.sum(axis=(0,1,2))
    if any(np.any(c == 0) for c in counts):
        raise ValueError("No valid training pairs for a spatial/temporal component.")
    scales = [np.sqrt(s/np.maximum(c,1)) for s,c in zip(sums,counts)]
    scales = [np.where(s > 1e-8,s,1.) for s in scales]
    return StructuredFieldLoss(edges, len(coords), n_components, mean, scale,
                               scales[0], scales[1], **kwargs)


def field_loss_from_data(data, mode, **kwargs):
    """One adapter for legacy DATA and DUAL_DATA; only affine scalers supported."""
    if hasattr(data, "normalizers") and hasattr(data, "splits"):
        return fit_field_loss(data.splits["train"]["field"][mode], data.metadata["canonical_coords"], mode,
            len(data.metadata["field_components"][mode]), mask=data.splits["train"]["field_mask"][mode],
            **data.normalizers["field"][mode], **kwargs)
    train = getattr(data, f"{mode}_train_out")
    coords = getattr(data,f"{mode}_IN_df").iloc[0].to_numpy().reshape(-1,2)
    scaler = getattr(data,f"{mode}_OUTscaler",None)
    mean, scale = 0., 1.
    if scaler is not None:
        # Evaluate the existing affine inverse rather than assume scaler names.
        if scaler.__class__.__name__ not in {"SymmetricScaler", "StandardScaler", "MinMaxScaler"}:
            raise ValueError("Structured loss requires an explicitly supported affine field scaler.")
        zero = np.zeros((1, train.shape[-1]))
        mean = scaler.inverse_transform(zero)
        scale = scaler.inverse_transform(np.ones_like(zero))-mean
    return fit_field_loss(train, coords, mode, len(getattr(data,f"{mode}_field_components")),
                          mask=getattr(data,f"{mode}_train_valid_mask",None), mean=mean, scale=scale, **kwargs)


def evaluate_independent_curve_bridge(field_data, curve_json, results_dir, mode, device="cpu", split="val"):
    """Frozen HPO curve model, ID-aligned validation intersection, two field sources.

    Reconstruct its own training-fitted input scaler from its DATA sidecar. Keep
    static node features unchanged and replace only displacement channels. Never
    evaluate on specimens used to train either stage. No new curve training.
    """
    import json
    from pathlib import Path
    import pandas as pd
    from resources.MLdata import DATA, _data_transform_array
    from resources.MLmodels import MODEL
    from resources.MLmetrics import curve_performance_diagnostics
    if split != "val": raise ValueError("Loss-ablation bridge uses validation only.")
    source=Path(curve_json);folder=Path(results_dir)
    data_config=json.loads(source.with_name(source.stem+"_data.json").read_text())["data_config"]
    curve_data=DATA(**{**data_config,"path":field_data.path})
    if curve_data.input_kind!='field' or curve_data.output_kind!='curve' or curve_data.reduce_dim:
        raise ValueError("Bridge requires full-curve field-input DATA without PCA.")
    for suffix in ("IN_df",):
        a=getattr(field_data,f'{mode}_{suffix}').iloc[0].to_numpy()
        b=getattr(curve_data,f'{mode}_{suffix}').iloc[0].to_numpy()
        if a.shape!=b.shape or not np.allclose(a,b): raise ValueError("Node coordinate orders differ between stages.")
    if list(getattr(field_data,f'{mode}_field_components')) != list(getattr(curve_data,f'{mode}_field_input_components')):
        raise ValueError("Field component order differs between stages.")
    if not np.allclose(getattr(field_data,f'{mode}_field_frame_values'),getattr(curve_data,f'{mode}_field_input_frame_values')):
        raise ValueError("Field sampling differs between stages.")
    field_ids=getattr(field_data,f'{mode}_val_in_df').index.astype(str)
    curve_ids=getattr(curve_data,f'{mode}_val_in_df').index.astype(str)
    common=field_ids.intersection(curve_ids,sort=False)
    if len(common)==0:raise ValueError("No shared held-out validation specimens for the two stages.")
    for data in (field_data,curve_data):
        if len(common.intersection(getattr(data,f'{mode}_train_in_df').index.astype(str))):
            raise ValueError("Curve bridge contains a training specimen.")
    fi,ci=field_ids.get_indexer(common),curve_ids.get_indexer(common)
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        physical=z[f'{mode}_{split}_outputs'][fi]
    true_tokens=getattr(curve_data,f'{mode}_val_in')[ci]
    scaler=getattr(curve_data,f'{mode}_INscaler')
    raw=scaler.inverse_transform(true_tokens.reshape(-1,true_tokens.shape[-1])).reshape(true_tokens.shape)
    width=physical.shape[-1]
    if width>=raw.shape[-1]:raise ValueError("Expected the verified HPO field-history plus static-feature schema.")
    raw[:,:,:width]=physical
    predicted_tokens=_data_transform_array(scaler,raw)
    if not np.isfinite(true_tokens).all() or not np.isfinite(predicted_tokens).all():
        raise ValueError("Nonfinite bridge inputs require an explicit imputation policy.")
    curve_model=MODEL.from_json(source,data=curve_data,device=device,scan_matches_on_init=False)
    network=getattr(curve_model,f'{mode}_model');network.eval()
    outputs={}
    for label,tokens in (('true_field',true_tokens),('pred_field',predicted_tokens)):
        with torch.no_grad():
            pieces=[network(torch.as_tensor(tokens[k:k+8],dtype=torch.float32,device=device)).cpu().numpy()
                    for k in range(0,len(tokens),8)]
        outputs[label]=getattr(curve_data,f'{mode}_OUTscaler').inverse_transform(np.concatenate(pieces))
    truth=getattr(curve_data,f'{mode}_OUTscaler').inverse_transform(getattr(curve_data,f'{mode}_val_out')[ci])
    x=getattr(curve_data,f'{mode}_OUT_df')
    for label,pred in outputs.items():
        diag=curve_performance_diagnostics(pred,truth,x_values=x)
        table=diag['sample_metrics'].copy();table['sample_id']=common
        if label=='true_field':table['imputed_field_values']=0
        stem=f'{mode}_{split}_' + ('true_field_curve' if label=='true_field' else 'curve')
        table.to_csv(folder/f'{stem}_sample_metrics.csv',index=False)
    np.savez(folder/'independent_curve_bridge.npz',sample_ids=np.asarray(common,dtype=str),truth=truth,**outputs)
    (folder/'curve_bridge_metadata.json').write_text(json.dumps({
        'curve_checkpoint':str(source),'validation_intersection':len(common),
        'field_validation_population':len(field_ids),'curve_validation_population':len(curve_ids),
        'static_features':'Unchanged, sample-aligned curve DATA tokens',
        'comparison':'Same frozen curve network and train-fitted scaler for true and predicted fields'},indent=2))
    return outputs
