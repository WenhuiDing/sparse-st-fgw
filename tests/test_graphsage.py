import numpy as np
import pytest


def test_spatial_graph_uses_loaded_h5ad_arrays():
    pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import build_spatial_graph

    spatial = np.asarray(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32
    )
    time_codes = np.asarray([0, 0, 0, 1, 1, 1])
    graph = build_spatial_graph(spatial, time_codes, neighbors=1)
    assert graph.shape == (6, 1)
    assert np.all(time_codes[graph[:3]] == 0)
    assert np.all(time_codes[graph[3:]] == 1)


@pytest.mark.parametrize(
    "norm,activation,dropout", [(True, True, 0.1), (False, False, 0.0)]
)
def test_minibatch_graphsage_matches_full_graph(norm, activation, dropout):
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import SpatialGraphSAGE

    torch.manual_seed(8)
    values = torch.randn(8, 5)
    graph = np.asarray(
        [[1, 2], [0, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 7], [5, 6]],
        dtype=np.int64,
    )
    model = SpatialGraphSAGE(
        n_features=5,
        hidden=7,
        norm=norm,
        activation=activation,
        dropout=dropout,
    ).eval()
    expected = model(values, graph)[[1, 4, 7]]
    actual = model.forward_batch(values, graph, np.asarray([1, 4, 7]))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_second_layer_aggregates_first_layer_neighbor_states():
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import SpatialGraphSAGE

    torch.manual_seed(7)
    graph = np.asarray([[1], [2], [3], [3]], dtype=np.int64)
    values = torch.randn(4, 3)
    model = SpatialGraphSAGE(n_features=3, hidden=6).eval()
    before = model.forward_batch(values, graph, np.asarray([0]))
    changed = values.clone()
    changed[2] += 5.0
    after = model.forward_batch(changed, graph, np.asarray([0]))
    assert not torch.allclose(before, after)


def test_batch_feature_mask_applies_to_all_target_occurrences():
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import SpatialGraphSAGE

    torch.manual_seed(11)
    values = torch.randn(6, 4)
    graph = np.asarray(
        [[1, 2], [0, 3], [0, 4], [1, 5], [2, 5], [3, 4]], dtype=np.int64
    )
    ids = np.asarray([0, 1, 4])
    feature_mask = torch.asarray(
        [[0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 1]], dtype=values.dtype
    )
    model = SpatialGraphSAGE(n_features=4, hidden=7).eval()

    globally_masked = values.clone()
    globally_masked[ids] *= feature_mask
    expected = model(globally_masked, graph)[ids]
    actual = model.forward_batch(values, graph, ids, feature_mask=feature_mask)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_checked_local_indices_rejects_missing_node_ids():
    pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import _checked_local_indices

    with pytest.raises(ValueError, match="absent from local support: 3"):
        _checked_local_indices(
            np.asarray([1, 2, 4]), np.asarray([2, 3]), name="test mapping"
        )


def test_forward_batch_rejects_out_of_range_graph_neighbors():
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import SpatialGraphSAGE

    values = torch.randn(3, 2)
    graph = np.asarray([[1], [3], [0]], dtype=np.int64)
    model = SpatialGraphSAGE(n_features=2, hidden=4).eval()
    with pytest.raises(IndexError, match="two-hop neighbor"):
        model.forward_batch(values, graph, np.asarray([0]))


def test_forward_batch_handles_empty_batch_and_rejects_duplicate_ids():
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import SpatialGraphSAGE

    values = torch.randn(3, 2)
    graph = np.asarray([[1], [2], [0]], dtype=np.int64)
    model = SpatialGraphSAGE(n_features=2, hidden=4).eval()

    empty = model.forward_batch(values, graph, np.asarray([], dtype=np.int64))
    assert empty.shape == (0, 2)
    with pytest.raises(ValueError, match="must be unique"):
        model.forward_batch(values, graph, np.asarray([0, 0]))


def test_checkpoint_round_trip_records_the_fixed_architecture(tmp_path):
    torch = pytest.importorskip("torch")
    from sparse_st_fgw.graphsage import (
        SpatialGraphSAGE,
        load_checkpoint,
        save_checkpoint,
    )

    path = tmp_path / "model.pt"
    original = SpatialGraphSAGE(3, hidden=5).eval()
    save_checkpoint(path, original, history=[], loss_components=[], config={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = load_checkpoint(path, device="cpu").eval()

    assert payload["format"] == "sparse_st_fgw_graphsage_v3"
    assert payload["architecture"] == {
        "model": "two_layer_additive_graphsage",
        "n_features": 3,
        "hidden": 5,
        "output": 3,
        "aggregator": "mean",
        "root_neighbor_combine": "add",
        "norm": True,
        "activation": "gelu",
        "dropout": 0.1,
    }
    values = torch.randn(4, 3)
    graph = np.asarray([[1], [2], [3], [0]], dtype=np.int64)
    assert torch.allclose(original(values, graph), restored(values, graph))
