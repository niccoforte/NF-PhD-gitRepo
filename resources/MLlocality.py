"""Opt-in initial-geometry locality for node Transformers; no job entry points."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InitialEdgeBias(nn.Module):
    """Learned per-head edge bias; nonedges retain ordinary global attention."""
    def __init__(self, heads, coords, edges, geometry_mean, geometry_scale, relative=False):
        super().__init__()
        self.heads, self.relative = heads, relative
        self.adjacency = nn.Parameter(torch.zeros(heads))
        self.geometry = nn.Linear(3, heads, bias=False) if relative else None
        if self.geometry is not None:
            nn.init.zeros_(self.geometry.weight)
        for key, value in dict(coords=coords, geometry_mean=geometry_mean, geometry_scale=geometry_scale).items():
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        for mode, pairs in edges.items():
            pairs = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
            if (len(pairs) == 0 or pairs.min() < 0 or pairs.max() >= len(coords)
                    or torch.any(pairs[:, 0] == pairs[:, 1])
                    or len(torch.unique(pairs.sort(1).values, dim=0)) != len(pairs)):
                raise ValueError("Attention edges must be valid unique undirected non-self pairs.")
            self.register_buffer("edges_"+mode, pairs)

    def forward(self, geometry, mode, present):
        pairs = getattr(self, "edges_"+mode)
        src = torch.cat([pairs[:, 0], pairs[:, 1]])
        dst = torch.cat([pairs[:, 1], pairs[:, 0]])
        batch, nodes = present.shape
        weights = self.adjacency.view(1, -1, 1).expand(batch, -1, len(src))
        if self.geometry is not None:
            # No tensor for all N*N geometric pairs: project only actual struts.
            xy = self.coords+geometry*self.geometry_scale+self.geometry_mean
            span = (self.coords.max(0).values-self.coords.min(0).values).max()
            delta = (xy[:, src]-xy[:, dst])/span.clamp_min(1e-8)
            weights = weights+self.geometry(torch.cat([delta, delta.norm(dim=-1, keepdim=True)], -1)).transpose(1, 2)
        valid = present[:, src] & present[:, dst]
        bias = geometry.new_zeros(batch, self.heads, nodes, nodes)
        bias[:, :, dst, src] = weights*valid.unsqueeze(1)
        bias = bias.masked_fill(~present[:, None, None, :], float("-inf"))
        return bias.reshape(batch*self.heads, nodes, nodes)


def encode_with_edge_bias(encoder, tokens, bias):
    """Standard encoder equations with an explicit per-batch/head attention bias.

    Uses the existing layer parameters, not a second attention network. Explicit
    equations avoid nested-tensor/fast-path branches bypassing the supplied bias.
    """
    for layer in encoder.layers:
        def attention(x):
            return layer.dropout1(layer.self_attn(x, x, x, attn_mask=bias, need_weights=False)[0])
        if layer.norm_first:
            tokens = tokens+attention(layer.norm1(tokens))
            tokens = tokens+layer._ff_block(layer.norm2(tokens))
        else:
            tokens = layer.norm1(tokens+attention(tokens))
            tokens = layer.norm2(tokens+layer._ff_block(tokens))
    return encoder.norm(tokens) if encoder.norm is not None else tokens


class MaskedGridRefinement(nn.Module):
    """Depthwise 3x3 + pointwise residual blocks on the reference-coordinate grid.

    Only actual task sites contribute; vacant sites never become material. Grid
    neighbours are a spatial prior, not asserted physical-strut adjacency.
    """
    def __init__(self, width, coords, blocks=1):
        super().__init__()
        coords = np.asarray(coords, float)
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all() or blocks < 1:
            raise ValueError("CNN requires finite2D coordinates and positive block count.")
        spacings = [np.diff(np.unique(coords[:, axis])) for axis in (0, 1)]
        if any(len(gaps) == 0 for gaps in spacings):
            raise ValueError("CNN requires a nondegenerate two-dimensional reference grid.")
        step = np.array([gaps.min() for gaps in spacings])
        grid = (coords-coords.min(0))/step
        ij = np.rint(grid).astype(int)
        if not np.allclose(grid, ij, atol=1e-5) or len(np.unique(ij, axis=0)) != len(coords):
            raise ValueError("Reference nodes must map one-to-one to grid sites.")
        self.height, self.width = int(ij[:, 1].max()+1), int(ij[:, 0].max()+1)
        if self.height*self.width > 8*len(coords):
            raise ValueError("Refuse an excessively sparse or unvalidated CNN grid.")
        self.register_buffer("sites", torch.as_tensor(ij[:, 1]*self.width+ij[:, 0]))
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.GELU(), nn.Conv2d(width, width, 1, bias=False)) for _ in range(blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(blocks)])

    def forward(self, tokens, present):
        batch, nodes, channels = tokens.shape
        occupancy = tokens.new_zeros(batch, 1, self.height*self.width)
        occupancy[:, :, self.sites] = present[:, None].to(tokens.dtype)
        occupancy = occupancy.reshape(batch, 1, self.height, self.width)
        count = F.conv2d(occupancy, tokens.new_ones(1, 1, 3, 3), padding=1).clamp_min(1)
        tokens = tokens.masked_fill(~present.unsqueeze(-1), 0.)
        for block, norm in zip(self.blocks, self.norms):
            grid = tokens.new_zeros(batch, channels, self.height*self.width)
            grid[:, :, self.sites] = tokens.transpose(1, 2)
            grid = grid.reshape(batch, channels, self.height, self.width)
            local = block[0](grid)/count
            local = block[2](block[1](local))*occupancy
            local = local.flatten(2)[:, :, self.sites].transpose(1, 2)
            tokens = (tokens+norm(local)).masked_fill(~present.unsqueeze(-1), 0.)
        return tokens


def configure_field_locality(network, data, variant):
    """Attach exactly one ablation after base initialization; preserve common weights."""
    from resources.MLdual import LocalFieldGraph
    from resources.MLfield import reference_field_edges
    if variant == "control":
        return
    coords = data.metadata["canonical_coords"]
    edges = {mode: reference_field_edges(coords, mode, data.node_masks[mode]) for mode in ("UT", "FT")}
    graph = dict(coords=coords, edges=edges, geometry_mean=data.normalizers["geometry"]["mean"],
                 geometry_scale=data.normalizers["geometry"]["scale"])
    if variant == "degree":
        for mode in ("UT", "FT"):
            degree = np.bincount(edges[mode].reshape(-1), minlength=data.n_nodes).astype(np.float32)/8
            degree[~data.node_masks[mode]] = 0
            data.task_features[mode] = np.column_stack([data.task_features[mode], degree])
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
        data.task_feature_names += ["initial_degree_over_eight"]
        if "context_spec" in data.metadata:
            data.metadata["context_spec"]["feature_names"] = list(data.task_feature_names)
        return
    for stage in network.field_model.tasks.values():
        if variant in ("graph", "gated_graph"):
            stage.local_graph = LocalFieldGraph(stage.d_model, **graph,
                **({"gate_init": -3.} if variant == "gated_graph" else {}))
        elif variant in ("adjacency", "geometry"):
            stage.set_locality(dict(kind=variant, **graph))
        elif variant in ("cnn1", "cnn2", "cnn3"):
            stage.set_locality(dict(kind="cnn", coords=coords, blocks=int(variant[-1])))
        else:
            raise ValueError("Unknown field locality variant: "+variant)
