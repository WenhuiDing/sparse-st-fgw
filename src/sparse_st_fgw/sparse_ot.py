"""Linear-memory bidirectional kNN UOT and sparse Masked-FGW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def edge_sqeuclidean(source, target, rows, cols, batch_size=262_144):
    source, target = np.asarray(source), np.asarray(target)
    output = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), int(batch_size)):
        stop = min(start + int(batch_size), len(rows))
        delta = source[rows[start:stop]] - target[cols[start:stop]]
        output[start:stop] = np.einsum("ij,ij->i", delta, delta)
    return output


@dataclass(frozen=True)
class NeighborIndex:
    index: Any
    size: int

    def query(self, points, k):
        result = self.index.query(
            np.asarray(points, dtype=np.float32), k=min(int(k), self.size)
        )[0]
        return np.asarray(result, dtype=np.int64).reshape(len(points), -1)


def build_neighbor_index(points: np.ndarray, *, k: int, seed: int) -> NeighborIndex:
    from pynndescent import NNDescent

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or len(points) < 3:
        raise ValueError("pynndescent requires at least three 2D points")
    index = NNDescent(
        points,
        n_neighbors=min(max(2 * int(k), 16), len(points) - 1),
        metric="euclidean",
        random_state=int(seed),
        n_jobs=-1,
        low_memory=True,
    )
    index.prepare()
    return NeighborIndex(index=index, size=len(points))


def build_bidirectional_candidates(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k: int,
    cap: int,
    global_drift: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bounded bidirectional support with optional global query drift."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source_index = build_neighbor_index(source, k=k, seed=seed)
    target_index = build_neighbor_index(target, k=k, seed=seed + 1)
    if global_drift:
        delta = target.mean(axis=0) - source.mean(axis=0)
        source_query = source + delta
        target_query = target - delta
    else:
        source_query, target_query = source, target
    forward = target_index.query(source_query, k)
    reverse = source_index.query(target_query, k)
    forward_keys = (
        np.repeat(np.arange(len(source), dtype=np.int64), forward.shape[1])
        * len(target)
        + forward.ravel()
    )
    reverse_keys = (
        reverse.ravel() * len(target)
        + np.repeat(np.arange(len(target), dtype=np.int64), reverse.shape[1])
    )
    keys = np.unique(np.concatenate((forward_keys, reverse_keys)))
    rows, cols = keys // len(target), keys % len(target)
    rows, cols = _cap_rows(source, target, rows, cols, int(cap))
    return rows.astype(np.int32), cols.astype(np.int32)


def _cap_rows(source, target, rows, cols, cap):
    """Cap source degree while reserving at least one edge for every target."""
    if cap <= 0 or len(target) > len(source) * cap:
        raise ValueError("candidate row cap cannot cover every target")
    costs = edge_sqeuclidean(source, target, rows, cols)
    order = np.lexsort((costs, cols))
    col_values, starts, counts = np.unique(
        cols[order], return_index=True, return_counts=True
    )
    mandatory = np.zeros(len(rows), dtype=bool)
    degree = np.zeros(len(source), dtype=np.int64)
    for col, start, count in zip(col_values, starts, counts):
        positions = order[start : start + count]
        ranked = positions[np.lexsort((costs[positions], degree[rows[positions]]))]
        available = ranked[degree[rows[ranked]] < cap]
        if not len(available):
            raise ValueError(f"row cap is infeasible for target {int(col)}")
        chosen = int(available[0])
        mandatory[chosen] = True
        degree[rows[chosen]] += 1
    keep = mandatory.copy()
    row_order = np.lexsort((costs, rows))
    optional = row_order[~mandatory[row_order]]
    if len(optional):
        optional_rows = rows[optional]
        _, starts, counts = np.unique(
            optional_rows, return_index=True, return_counts=True
        )
        within = np.arange(len(optional)) - np.repeat(starts, counts)
        keep[optional[within < cap - degree[optional_rows]]] = True
    return rows[keep], cols[keep]


def normalize_cost(cost: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    return (cost / max(float(np.quantile(cost, quantile)), 1e-12)).astype(np.float32)


def uniform_marginals(stage_sizes):
    """Assign unit total mass to every stage and equal mass to every spot.

    The solver normalizes these vectors again before iteration. Keeping the
    canonical vectors unit-normalized makes that invariant explicit and avoids
    carrying an arbitrary absolute mass scale through the pipeline.
    """
    stages = [
        np.full(int(size), 1.0 / int(size), dtype=np.float64)
        for size in stage_sizes
    ]
    return [(stages[i], stages[i + 1]) for i in range(len(stages) - 1)]


def _segment_logsumexp(values, segment_ids, num_segments):
    import jax.numpy as jnp
    from jax import ops

    maxima = ops.segment_max(values, segment_ids, num_segments=num_segments)
    safe = jnp.where(jnp.isfinite(maxima), maxima, 0.0)
    summed = ops.segment_sum(
        jnp.exp(values - safe[segment_ids]), segment_ids, num_segments=num_segments
    )
    return safe + jnp.log(jnp.maximum(summed, 1e-30))


def _compiled_solver(n_source, n_target, iterations):
    import jax
    import jax.numpy as jnp
    from jax import lax

    @jax.jit
    def run(
        rows, cols, cost, source_mass, target_mass, epsilon, rho_a, rho_b, scale
    ):
        log_kernel = -(cost / jnp.maximum(scale, 1e-12)) / jnp.maximum(
            epsilon, 1e-8
        )
        contraction_a = rho_a / (rho_a + epsilon)
        contraction_b = rho_b / (rho_b + epsilon)
        f0 = jnp.zeros((n_source,), dtype=cost.dtype)
        g0 = jnp.zeros((n_target,), dtype=cost.dtype)

        def body(_index, state):
            f, g = state
            f = contraction_a * (
                jnp.log(jnp.maximum(source_mass, 1e-30))
                - _segment_logsumexp(log_kernel + g[cols], rows, n_source)
            )
            g = contraction_b * (
                jnp.log(jnp.maximum(target_mass, 1e-30))
                - _segment_logsumexp(log_kernel + f[rows], cols, n_target)
            )
            return f, g

        f, g = lax.fori_loop(0, int(iterations), body, (f0, g0))
        values = jnp.exp(
            jnp.clip(log_kernel + f[rows] + g[cols], -745.0, 80.0)
        )
        return values

    return run


def _resolve_jax_device(device):
    """Resolve an explicit logical JAX device without silently choosing one."""
    import jax

    name = str(device)
    if name == "cpu":
        candidates = jax.devices("cpu")
        if len(candidates) != 1:
            raise RuntimeError("JAX CPU device is unavailable")
        return candidates[0]
    if name in {"cuda", "gpu"}:
        name = "cuda:0"
    if name.startswith(("cuda:", "gpu:")):
        prefix, index_text = name.split(":", 1)
        if not index_text.isdigit():
            raise ValueError(f"invalid JAX device index: {device!r}")
        index = int(index_text)
        candidates = jax.devices("cuda")
        if index >= len(candidates):
            raise ValueError(
                f"requested {device!r}, but only {len(candidates)} CUDA devices "
                "are visible"
            )
        return candidates[index]
    raise ValueError(
        f"unsupported JAX device {device!r}; use 'cuda:<index>' or 'cpu'"
    )


def solve_sparse_uot(
    source,
    target,
    rows,
    cols,
    cost,
    source_mass,
    target_mass,
    *,
    epsilon,
    rho_a,
    rho_b,
    iterations,
    device,
) -> tuple[np.ndarray, dict]:
    import jax
    import jax.numpy as jnp

    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    cost = np.asarray(cost, dtype=np.float32)
    if np.unique(rows).size != len(source) or np.unique(cols).size != len(target):
        raise ValueError("candidate graph contains orphan nodes")
    selected_device = _resolve_jax_device(device)
    if selected_device.platform != "gpu":
        raise RuntimeError("Sparse-UOT requires a CUDA JAX device")
    aa = np.asarray(source_mass, dtype=np.float32)
    bb = np.asarray(target_mass, dtype=np.float32)
    aa /= aa.sum()
    bb /= bb.sum()
    scale = float(np.quantile(cost, 0.99))
    with jax.default_device(selected_device):
        values = _compiled_solver(len(source), len(target), int(iterations))(
            jnp.asarray(rows),
            jnp.asarray(cols),
            jnp.asarray(cost),
            jnp.asarray(aa),
            jnp.asarray(bb),
            jnp.asarray(float(epsilon), dtype=jnp.float32),
            jnp.asarray(float(rho_a), dtype=jnp.float32),
            jnp.asarray(float(rho_b), dtype=jnp.float32),
            jnp.asarray(scale, dtype=jnp.float32),
        )
        values = np.asarray(values, dtype=np.float64)
    row_mass = np.zeros(len(source), dtype=np.float64)
    col_mass = np.zeros(len(target), dtype=np.float64)
    np.add.at(row_mass, rows, values)
    np.add.at(col_mass, cols, values)
    return values, {
        "backend": "jax_xla_sparse_uot",
        "device": str(selected_device),
        "converged": bool(np.isfinite(values).all()),
        "iterations": int(iterations),
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "transport_mass": float(values.sum()),
        "row_marginal_l1": float(np.abs(row_mass - aa).sum()),
        "col_marginal_l1": float(np.abs(col_mass - bb).sum()),
        "cost_scale": scale,
        "candidate_edges": int(len(rows)),
    }


def squared_euclidean_gw_cost(
    source_xy, target_xy, rows, cols, values, source_mass, target_mass
):
    x = np.asarray(source_xy, dtype=np.float64)
    y = np.asarray(target_xy, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    p = np.asarray(values, dtype=np.float64)
    mu = np.asarray(source_mass, dtype=np.float64)
    nu = np.asarray(target_mass, dtype=np.float64)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    sx, ty = np.sum(x * x, axis=1), np.sum(y * y, axis=1)
    total = float(p.sum())
    x1, y1 = mu @ x, nu @ y
    xa, yb = mu @ (x * sx[:, None]), nu @ (y * ty[:, None])
    mxx, myy = (x * mu[:, None]).T @ x, (y * nu[:, None]).T @ y
    a1, b1 = float(mu @ sx), float(nu @ ty)
    a2, b2 = float(mu @ (sx * sx)), float(nu @ (ty * ty))
    ex, ey, es, et = x[rows], y[cols], sx[rows], ty[cols]
    source_term = (
        total * es**2
        + a2
        + 4.0 * np.einsum("ni,ij,nj->n", ex, mxx, ex)
        + 2.0 * es * a1
        - 4.0 * es * (ex @ x1)
        - 4.0 * (ex @ xa)
    )
    target_term = (
        total * et**2
        + b2
        + 4.0 * np.einsum("ni,ij,nj->n", ey, myy, ey)
        + 2.0 * et * b1
        - 4.0 * et * (ey @ y1)
        - 4.0 * (ey @ yb)
    )
    ab = float(np.sum(p * sx[rows] * ty[cols]))
    by_target = np.zeros(len(y), dtype=np.float64)
    by_source = np.zeros(len(x), dtype=np.float64)
    np.add.at(by_target, cols, p * sx[rows])
    np.add.at(by_source, rows, p * ty[cols])
    va_y, ux_b = by_target @ y, by_source @ x
    cxy = (ex * p[:, None]).T @ ey
    cross = (
        total * es * et
        + es * b1
        - 2.0 * es * (ey @ y1)
        + et * a1
        + ab
        - 2.0 * (ey @ va_y)
        - 2.0 * et * (ex @ x1)
        - 2.0 * (ex @ ux_b)
        + 4.0 * np.einsum("ni,ij,nj->n", ex, cxy, ey)
    )
    return np.maximum(source_term + target_term - 2.0 * cross, 0.0)


def solve_masked_fgw(
    source,
    target,
    source_xy,
    target_xy,
    rows,
    cols,
    source_mass,
    target_mass,
    *,
    epsilon,
    rho_a,
    rho_b,
    warmstart_iterations,
    outer_iterations,
    inner_iterations,
    weight,
    step,
    device,
) -> tuple[np.ndarray, dict]:
    expression_cost = normalize_cost(
        edge_sqeuclidean(source, target, rows, cols), 0.95
    )
    values, info = solve_sparse_uot(
        source,
        target,
        rows,
        cols,
        expression_cost,
        source_mass,
        target_mass,
        epsilon=epsilon,
        rho_a=rho_a,
        rho_b=rho_b,
        iterations=warmstart_iterations,
        device=device,
    )
    history = []
    for _ in range(int(outer_iterations)):
        normalized = values / max(float(values.sum()), 1e-12)
        gw_cost = normalize_cost(
            squared_euclidean_gw_cost(
                source_xy,
                target_xy,
                rows,
                cols,
                normalized,
                source_mass,
                target_mass,
            ),
            0.95,
        )
        total_cost = (
            (1.0 - float(weight)) * expression_cost + float(weight) * gw_cost
        ).astype(np.float32)
        updated, info = solve_sparse_uot(
            source,
            target,
            rows,
            cols,
            total_cost,
            source_mass,
            target_mass,
            epsilon=epsilon,
            rho_a=rho_a,
            rho_b=rho_b,
            iterations=inner_iterations,
            device=device,
        )
        values = ((1.0 - float(step)) * values + float(step) * updated).astype(
            np.float64
        )
        history.append(float(np.mean(gw_cost)))
    final_row_mass = np.zeros(len(source), dtype=np.float64)
    final_col_mass = np.zeros(len(target), dtype=np.float64)
    np.add.at(final_row_mass, np.asarray(rows, dtype=np.int64), values)
    np.add.at(final_col_mass, np.asarray(cols, dtype=np.int64), values)
    normalized_source_mass = np.asarray(source_mass, dtype=np.float64)
    normalized_target_mass = np.asarray(target_mass, dtype=np.float64)
    normalized_source_mass /= normalized_source_mass.sum()
    normalized_target_mass /= normalized_target_mass.sum()
    info.update(
        {
            # Report the damped outer-loop coupling that is returned and
            # persisted, not the last undamped inner UOT update.
            "converged": bool(np.isfinite(values).all()),
            "transport_mass": float(values.sum()),
            "row_marginal_l1": float(
                np.abs(final_row_mass - normalized_source_mass).sum()
            ),
            "col_marginal_l1": float(
                np.abs(final_col_mass - normalized_target_mass).sum()
            ),
            "masked_fgw": True,
            "masked_fgw_weight": float(weight),
            "masked_fgw_outer_iterations": int(outer_iterations),
            "masked_fgw_inner_iterations": int(inner_iterations),
            "masked_fgw_step": float(step),
            "masked_fgw_warmstart_iterations": int(warmstart_iterations),
            "masked_fgw_history": history,
        }
    )
    return values, info
