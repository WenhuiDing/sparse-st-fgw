import inspect
from pathlib import Path

import numpy as np
import pytest

from sparse_st_fgw.data import normalize_spatial


PROJECT_ROOT = Path(__file__).parents[1]


def test_model_interface_is_preprocessed_data_only():
    pytest.importorskip("torch")
    from sparse_st_fgw.pipeline import run_model

    parameters = inspect.signature(run_model).parameters
    assert "data_dir" in parameters
    assert "data_path" not in parameters
    assert "transition_graph" not in parameters


def test_workflow_scripts_are_real_entry_points():
    expected_calls = {
        "preprocess_data.py": "preprocess_dataset(",
        "train_model.py": "train_spatial_model(",
        "run_model.py": "run_model(",
        "calculate_accuracy.py": "calculate_accuracy(",
    }
    for filename, call in expected_calls.items():
        source = (PROJECT_ROOT / "scripts" / filename).read_text()
        assert "sparse_st_fgw.cli" not in source
        assert "argparse.ArgumentParser" in source
        assert call in source


def test_runtime_package_does_not_contain_tests():
    package = PROJECT_ROOT / "src" / "sparse_st_fgw"
    assert not (package / "tests").exists()
    assert not (package / "test_core.py").exists()


def test_model_has_no_unregistered_cross_slice_coordinate_cost():
    package = PROJECT_ROOT / "src" / "sparse_st_fgw"
    pipeline = (package / "pipeline.py").read_text()
    sparse_ot = (package / "sparse_ot.py").read_text()
    assert "fused_edge_cost" not in pipeline
    assert "def fused_edge_cost" not in sparse_ot


def test_spatial_normalization_is_rigid_transform_and_scale_invariant():
    xy = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]], dtype=np.float32)
    angle = 0.73
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = 3.7 * (xy @ rotation.T) + np.asarray([12.0, -4.0])
    left = normalize_spatial(xy)
    right = normalize_spatial(transformed)
    left_distances = np.linalg.norm(left[:, None] - left[None, :], axis=2)
    right_distances = np.linalg.norm(right[:, None] - right[None, :], axis=2)
    assert np.allclose(left_distances, right_distances, atol=1e-6)
