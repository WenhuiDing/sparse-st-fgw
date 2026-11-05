import json

import anndata as ad
import numpy as np
import pandas as pd

from sparse_st_fgw.data import (
    load_dataset_config,
    load_evaluation_data,
    load_model_data,
    preprocess_dataset,
)


def test_h5ad_contract_reorders_stages_and_preserves_spot_ids(tmp_path):
    raw = tmp_path / "raw.h5ad"
    model = tmp_path / "data" / "processed.h5ad"
    evaluation = tmp_path / "data" / "evaluation.h5ad"
    ids = ["spot-0", "spot-1", "spot-2", "spot-3", "spot-4", "spot-5"]
    obs = pd.DataFrame(
        {
            "time": ["late", "early", "late", "early", "late", "early"],
            "label": ["b", "a", "b", "a", "b", "a"],
        },
        index=ids,
    )
    raw_data = ad.AnnData(
        X=np.asarray(
            [
                [8, 0, 1, 0], [0, 7, 0, 1], [9, 0, 2, 0],
                [0, 8, 0, 2], [7, 0, 1, 1], [0, 9, 1, 2],
            ],
            dtype=np.float32,
        ),
        obs=obs,
    )
    raw_data.obsm["xy"] = np.arange(12, dtype=np.float32).reshape(6, 2)
    raw_data.write_h5ad(raw)
    config_path = tmp_path / "dataset.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "tiny",
                "raw_h5ad": str(raw),
                "processed_h5ad": str(model),
                "evaluation_h5ad": str(evaluation),
                "count_layer": "X",
                "expression_is_raw_counts": True,
                "time_key": "time",
                "annotation_key": "label",
                "spatial_key": "xy",
                "spot_id_key": "spot_id",
                "stage_order": ["early", "late"],
                "hvg_count": 4,
                "hvg_cells_per_stage": 3,
                "pca_components": 2,
                "library_size": 1000,
                "preprocessing_batch_size": 3,
            }
        )
    )
    metadata = preprocess_dataset(load_dataset_config(config_path), batch_size=3)
    pca, spatial, stages, loaded, spot_ids = load_model_data(model.parent)
    labels, evaluation_ids, _ = load_evaluation_data(model.parent)
    assert metadata["stage_names"] == ["early", "late"]
    assert loaded["stage_slices"] == [
        {"name": "early", "start": 0, "stop": 3},
        {"name": "late", "start": 3, "stop": 6},
    ]
    assert pca.shape == (6, 2)
    assert spatial.shape == (6, 2)
    assert stages.tolist() == [0, 0, 0, 1, 1, 1]
    assert spot_ids.tolist() == ["spot-1", "spot-3", "spot-5", "spot-0", "spot-2", "spot-4"]
    assert evaluation_ids.tolist() == spot_ids.tolist()
    assert labels.tolist() == ["a", "a", "a", "b", "b", "b"]
