"""Spatial GraphSAGE and frozen Zero-Shot spatiotemporal message passing."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F


def build_spatial_graph(
    spatial: np.ndarray, time_codes: np.ndarray, *, neighbors: int
) -> np.ndarray:
    codes = np.asarray(time_codes, dtype=np.int64)
    spatial = np.asarray(spatial, dtype=np.float32)
    if len(codes) != len(spatial):
        raise ValueError("spatial coordinates and time codes differ in length")
    k = int(neighbors)
    graph = np.empty((len(codes), k), dtype=np.int32)
    for code in np.unique(codes[codes >= 0]):
        ids = np.flatnonzero(codes == code)
        if len(ids) <= k:
            raise ValueError("every stage must contain more spots than spatial k")
        local = cKDTree(spatial[ids]).query(
            spatial[ids], k=k + 1, workers=-1
        )[1][:, 1:]
        graph[ids] = ids[np.asarray(local, dtype=np.int64)]
    return graph


class SpatialGraphSAGE(nn.Module):
    """Two-layer additive GraphSAGE with hidden-state neighbor messages."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        *,
        norm: bool = True,
        activation: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        self.output = int(n_features)
        self.norm = bool(norm)
        self.activation = bool(activation)
        self.dropout_rate = float(dropout)
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            raise ValueError("dropout must be in [0, 1]")
        self.self1 = nn.Linear(self.n_features, self.hidden)
        self.neigh1 = nn.Linear(self.n_features, self.hidden, bias=False)
        self.norm1 = nn.LayerNorm(self.hidden) if self.norm else nn.Identity()
        self.self2 = nn.Linear(self.hidden, self.output)
        self.neigh2 = nn.Linear(self.hidden, self.output, bias=False)
        self.norm2 = nn.LayerNorm(self.output) if self.norm else nn.Identity()
        self.dropout = nn.Dropout(self.dropout_rate)

    def encode_hidden(self, values, neighbor_mean):
        hidden = self.norm1(self.self1(values) + self.neigh1(neighbor_mean))
        if self.activation:
            hidden = F.gelu(hidden)
        return self.dropout(hidden)

    def decode(self, hidden, hidden_neighbor_mean):
        return self.norm2(self.self2(hidden) + self.neigh2(hidden_neighbor_mean))

    def forward(self, values, graph):
        """Full-graph forward pass for small graphs and API compatibility."""
        graph = torch.as_tensor(graph, dtype=torch.long, device=values.device)
        neighbor_values = values[graph].mean(dim=1)
        hidden = self.encode_hidden(values, neighbor_values)
        hidden_neighbors = hidden[graph].mean(dim=1)
        return self.decode(hidden, hidden_neighbors)

    def forward_batch(self, values, graph, ids, feature_mask=None):
        """Compute a batch using its exact two-hop receptive field.

        ``values`` remains on CPU in the training/inference loops. Only the
        batch's one- or two-hop support is transferred to the accelerator,
        keeping memory proportional to the sampled receptive field rather
        than all nodes.
        """
        ids = np.asarray(ids, dtype=np.int64)
        graph = np.asarray(graph, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("batch node IDs must be one-dimensional")
        if len(ids) == 0:
            return values.new_empty((0, self.output), device=next(self.parameters()).device)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("batch node IDs must be unique")
        if graph.ndim != 2 or graph.shape[0] != len(values):
            raise ValueError("graph must have shape (n_nodes, n_neighbors)")
        if ids.min() < 0 or ids.max() >= len(values):
            raise IndexError("batch node ID is outside the graph")

        first_hop = graph[ids]
        if first_hop.size and (first_hop.min() < 0 or first_hop.max() >= len(values)):
            raise IndexError("graph contains a neighbor outside the node range")
        frontier = np.unique(np.concatenate((ids, first_hop.reshape(-1))))
        target_in_frontier = _checked_local_indices(
            frontier, ids, name="batch targets in frontier"
        )
        target_neighbors_in_frontier = _checked_local_indices(
            frontier, first_hop, name="target neighbors in frontier"
        )
        frontier_neighbors = graph[frontier]
        if frontier_neighbors.size and (
            frontier_neighbors.min() < 0
            or frontier_neighbors.max() >= len(values)
        ):
            raise IndexError("graph contains a two-hop neighbor outside the node range")

        # ``frontier`` nodes need first-layer states. ``support`` additionally
        # contains every raw two-hop feature required to compute those states.
        support = np.unique(
            np.concatenate((frontier, frontier_neighbors.reshape(-1)))
        )
        frontier_in_support = _checked_local_indices(
            support, frontier, name="frontier in two-hop support"
        )
        frontier_neighbors_in_support = _checked_local_indices(
            support, frontier_neighbors, name="frontier neighbors in two-hop support"
        )

        support_values = _masked_support_values(
            values, support, ids, feature_mask
        )

        device = next(self.parameters()).device
        hidden = self.encode_hidden(
            support_values[frontier_in_support].to(device),
            support_values[frontier_neighbors_in_support].to(device).mean(dim=1),
        )
        target_hidden = hidden[target_in_frontier]
        target_neighbor_hidden = hidden[target_neighbors_in_frontier].mean(dim=1)
        return self.decode(target_hidden, target_neighbor_hidden)


def _checked_local_indices(sorted_nodes, node_ids, *, name):
    """Map global node IDs into a sorted local support without silent misses."""
    sorted_nodes = np.asarray(sorted_nodes, dtype=np.int64)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    if sorted_nodes.ndim != 1 or not len(sorted_nodes):
        raise ValueError(f"{name}: local support must be a non-empty vector")
    positions = np.searchsorted(sorted_nodes, node_ids)
    valid = positions < len(sorted_nodes)
    clipped = np.minimum(positions, len(sorted_nodes) - 1)
    if not np.all(valid & (sorted_nodes[clipped] == node_ids)):
        missing = np.unique(node_ids[~(valid & (sorted_nodes[clipped] == node_ids))])
        preview = ", ".join(map(str, missing[:5]))
        raise ValueError(f"{name}: node IDs are absent from local support: {preview}")
    return positions


def _masked_support_values(values, support, ids, feature_mask):
    """Read support features and consistently mask every target occurrence."""
    support_values = values[support]
    if feature_mask is None:
        return support_values
    feature_mask = torch.as_tensor(
        feature_mask, dtype=support_values.dtype, device=support_values.device
    )
    if feature_mask.shape != (len(ids), values.shape[1]):
        raise ValueError("feature mask must have shape (batch_size, n_features)")
    target_in_support = _checked_local_indices(
        support, ids, name="batch targets in receptive-field support"
    )
    support_mask = torch.ones_like(support_values)
    support_mask[target_in_support] = feature_mask
    return support_values * support_mask


def train_graphsage(
    latent: np.ndarray,
    graph: np.ndarray,
    *,
    norm: bool,
    activation: bool,
    dropout: float,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_probability: float,
    consistency_weight: float,
    device: str,
    seed: int,
) -> tuple[SpatialGraphSAGE, list[float], list[dict]]:
    values = torch.as_tensor(np.array(latent, dtype=np.float32, copy=True))
    graph = np.asarray(graph, dtype=np.int64)
    torch.manual_seed(int(seed))
    model = SpatialGraphSAGE(
        values.shape[1],
        hidden=hidden,
        norm=norm,
        activation=activation,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    history = []
    components = []
    model.train()
    for _epoch in range(int(epochs)):
        total = reconstruction_total = consistency_total = 0.0
        order = rng.permutation(len(values))
        for start in range(0, len(values), int(batch_size)):
            ids = order[start : start + int(batch_size)]
            x = values[ids]
            target = x.to(device)
            keep = (torch.rand_like(x) > float(mask_probability)).to(x.dtype)
            output = model.forward_batch(
                values, graph, ids, feature_mask=keep
            )
            target_neighbor_ids = graph[ids]
            neighbor_mean = values[target_neighbor_ids].mean(dim=1).to(device)
            reconstruction_loss = F.smooth_l1_loss(output, target)
            consistency_loss = (
                output - neighbor_mean.to(device)
            ).square().mean()
            loss = reconstruction_loss + float(consistency_weight) * consistency_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            reconstruction_total += float(reconstruction_loss.detach()) * len(ids)
            consistency_total += float(consistency_loss.detach()) * len(ids)
        history.append(total / len(values))
        components.append(
            {
                "reconstruction": reconstruction_total / len(values),
                "consistency": consistency_total / len(values),
                "weighted_consistency": float(consistency_weight)
                * consistency_total
                / len(values),
                "total": total / len(values),
            }
        )
    return model, history, components


@torch.no_grad()
def encode(
    model: SpatialGraphSAGE,
    latent: np.ndarray,
    graph: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    values = torch.as_tensor(np.array(latent, dtype=np.float32, copy=True))
    graph = np.asarray(graph, dtype=np.int64)
    model = model.to(device).eval()
    result = np.empty((len(values), model.output), dtype=np.float32)
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        ids = np.arange(start, stop)
        result[start:stop] = model.forward_batch(
            values, graph, ids
        ).cpu().numpy()
    return result


def save_checkpoint(
    path: Path | str,
    model: SpatialGraphSAGE,
    *,
    history: list[float],
    loss_components: list[dict] | None = None,
    config: dict,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sparse_st_fgw_graphsage_v3",
            "architecture": {
                "model": "two_layer_additive_graphsage",
                "n_features": model.n_features,
                "hidden": model.hidden,
                "output": model.output,
                "aggregator": "mean",
                "root_neighbor_combine": "add",
                "norm": model.norm,
                "activation": "gelu" if model.activation else None,
                "dropout": model.dropout_rate,
            },
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "history": [float(x) for x in history],
            "loss_components": list(loss_components or []),
            "config": dict(config),
        },
        target,
    )


def load_checkpoint(path: Path | str, *, device: str) -> SpatialGraphSAGE:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    architecture = payload["architecture"]

    model = SpatialGraphSAGE(
        architecture["n_features"],
        hidden=architecture["hidden"],
        norm=architecture["norm"],
        activation=architecture["activation"] == "gelu",
        dropout=architecture["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def add_bidirectional_temporal_edges(
    spatial_graph: np.ndarray,
    edge_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    stage_offsets: np.ndarray,
    *,
    topk: int,
) -> np.ndarray:
    """Add separate future/history top-k OT neighbors to every spot."""
    graph = np.asarray(spatial_graph, dtype=np.int64)
    n_nodes = len(graph)
    k = int(topk)
    future = [[] for _ in range(n_nodes)]
    history = [[] for _ in range(n_nodes)]
    for pair_index, (rows, cols, values) in enumerate(edge_batches):
        left = np.asarray(rows, dtype=np.int64) + int(stage_offsets[pair_index])
        right = np.asarray(cols, dtype=np.int64) + int(stage_offsets[pair_index + 1])
        mass = np.asarray(values, dtype=np.float32)
        for source, target, weight in zip(left.tolist(), right.tolist(), mass.tolist()):
            if np.isfinite(weight) and weight > 0:
                future[source].append((weight, target))
                history[target].append((weight, source))
    fallback = graph[:, 0]
    temporal = np.empty((n_nodes, 2 * k), dtype=np.int64)
    for node in range(n_nodes):
        for channel, candidates in enumerate((future[node], history[node])):
            best = {}
            for weight, target in candidates:
                best[target] = max(weight, best.get(target, -np.inf))
            selected = [target for target, _ in sorted(
                best.items(), key=lambda item: item[1], reverse=True
            )[:k]]
            selected.extend([int(fallback[node])] * (k - len(selected)))
            temporal[node, channel * k : (channel + 1) * k] = selected
    return np.concatenate((graph, temporal), axis=1)
