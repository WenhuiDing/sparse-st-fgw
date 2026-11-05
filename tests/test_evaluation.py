import json

import anndata as ad
import numpy as np

from sparse_st_fgw.evaluation import calculate_accuracy, evaluate


def test_transition_metric_uses_raw_coupling():
    rows = np.asarray([0, 0, 1, 1])
    cols = np.asarray([0, 1, 0, 1])
    values = np.asarray([3.0, 1.0, 1.0, 3.0])
    labels = np.asarray(["a", "b"])
    accuracy = evaluate(
        rows, cols, values, labels, labels, {("a", "a"), ("b", "b")}
    )
    assert accuracy == 0.75


def test_accuracy_reads_saved_transports(tmp_path):
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    (results_dir / "transports").mkdir(parents=True)
    ids = np.asarray(["s0", "s1", "t0", "t1"])
    eval_data = ad.AnnData(X=np.zeros((4, 1), dtype=np.float32))
    eval_data.obs_names = ids
    eval_data.obs["annotation"] = np.asarray(["a", "b", "a", "b"], dtype=object)
    eval_data.write_h5ad(data_dir / "evaluation.h5ad")
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "n_spots": 4,
                "stage_slices": [
                    {"name": "t0", "start": 0, "stop": 2},
                    {"name": "t1", "start": 2, "stop": 4},
                ],
                "processed_h5ad": str(data_dir / "processed.h5ad"),
                "evaluation_h5ad": str(data_dir / "evaluation.h5ad"),
                "config": {"annotation_key": "annotation"},
            }
        )
    )
    np.savez(
        results_dir / "transports" / "pair_00.npz",
        rows=np.asarray([0, 0, 1, 1]),
        cols=np.asarray([0, 1, 0, 1]),
        values=np.asarray([3.0, 1.0, 1.0, 3.0]),
        n_source=np.int64(2),
        n_target=np.int64(2),
        source_stage=np.asarray("t0"),
        target_stage=np.asarray("t1"),
    )
    (results_dir / "model_manifest.json").write_text(
        json.dumps(
            {
                "pairs": [{"transport": "transports/pair_00.npz"}],
                "stage_sizes": [2, 2],
                "stage_slices": [
                    {"name": "t0", "start": 0, "stop": 2},
                    {"name": "t1", "start": 2, "stop": 4},
                ],
            }
        )
    )
    transitions = tmp_path / "transitions.tsv"
    transitions.write_text("Cell_type\tKnown_descendants\na\t/\nb\t/\n")
    report = calculate_accuracy(data_dir, results_dir, transitions)
    assert report["metric"] == "raw_sparse_coupling_no_row_or_column_normalization"
    assert report["summary"]["mean_accuracy"] == 0.75
    assert (results_dir / "accuracy.json").exists()
