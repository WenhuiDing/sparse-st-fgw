import numpy as np

from sparse_st_fgw.sparse_ot import (
    _cap_rows,
    _resolve_jax_device,
    uniform_marginals,
    normalize_cost,
    squared_euclidean_gw_cost,
)


def test_jax_device_selector_accepts_logical_cuda_device():
    import jax

    if not jax.devices("cuda"):
        import pytest
        pytest.skip("CUDA JAX backend is unavailable")
    selected = _resolve_jax_device("cuda:0")
    assert selected == jax.devices("cuda")[0]


def test_uniform_marginals_have_unit_stage_mass():
    pairs = uniform_marginals([3, 5, 7])
    assert len(pairs) == 2
    assert all(np.isclose(source.sum(), 1.0) for source, _ in pairs)
    assert all(np.isclose(target.sum(), 1.0) for _, target in pairs)
    assert np.allclose(pairs[0][0], 1.0 / 3.0)
    assert np.allclose(pairs[0][1], 1.0 / 5.0)




def test_cap_preserves_every_target():
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)
    target = np.asarray([[0.0], [0.8], [1.2], [2.0]], dtype=np.float32)
    rows = np.repeat(np.arange(3), 4)
    cols = np.tile(np.arange(4), 3)
    kept_rows, kept_cols = _cap_rows(source, target, rows, cols, cap=2)
    assert np.unique(kept_cols).size == len(target)
    assert np.bincount(kept_rows, minlength=3).max() <= 2


def test_sparse_gw_is_finite_and_edge_sized():
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    rows = np.repeat(np.arange(3), 3)
    cols = np.tile(np.arange(3), 3)
    values = np.full(9, 1.0 / 9.0)
    mass = np.full(3, 1.0 / 3.0)
    cost = squared_euclidean_gw_cost(x, y, rows, cols, values, mass, mass)
    assert cost.shape == rows.shape
    assert np.isfinite(cost).all()
    assert (cost >= 0).all()
    assert np.isfinite(normalize_cost(cost)).all()
