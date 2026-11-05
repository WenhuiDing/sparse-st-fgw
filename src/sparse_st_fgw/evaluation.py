"""Raw sparse transition aggregation without row or column normalization."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd

from .data import load_evaluation_data


def load_allowed_transitions(path: Path | str) -> set[tuple[str, str]]:
    frame = pd.read_csv(Path(path), sep="\t")
    required = {"Cell_type", "Known_descendants"}
    if not required.issubset(frame.columns):
        raise ValueError("transition graph lacks Cell_type/Known_descendants")
    edges = set()
    for row in frame.itertuples(index=False):
        source = str(row.Cell_type)
        edges.add((source, source))
        if pd.isna(row.Known_descendants):
            continue
        for target in str(row.Known_descendants).split(", "):
            if target and target != "/":
                edges.add((source, target))
    return edges


def transition_table(rows, cols, values, source_labels, target_labels):
    """Aggregate the persisted coupling without row or column normalization."""
    source_labels = np.asarray(source_labels).astype(str)
    target_labels = np.asarray(target_labels).astype(str)
    source_names = list(dict.fromkeys(source_labels.tolist()))
    target_names = list(dict.fromkeys(target_labels.tolist()))
    source_lookup = {name: index for index, name in enumerate(source_names)}
    target_lookup = {name: index for index, name in enumerate(target_names)}
    source_codes = np.fromiter(
        (source_lookup[x] for x in source_labels), dtype=np.int64
    )
    target_codes = np.fromiter(
        (target_lookup[x] for x in target_labels), dtype=np.int64
    )
    table = np.zeros((len(source_names), len(target_names)), dtype=np.float64)
    np.add.at(
        table,
        (
            source_codes[np.asarray(rows, dtype=np.int64)],
            target_codes[np.asarray(cols, dtype=np.int64)],
        ),
        np.asarray(values, dtype=np.float64),
    )
    return source_names, target_names, table


def score_table(table, source_names, target_names, allowed):
    mask = np.asarray(
        [
            [(source, target) in allowed for target in target_names]
            for source in source_names
        ],
        dtype=bool,
    )
    total = float(table.sum())
    return float(table[mask].sum() / total) if total > 0 else float("nan")


def evaluate(rows, cols, values, source_labels, target_labels, allowed):
    """Score the coupling directly, without row or column normalization."""
    source_names, target_names, table = transition_table(
        rows, cols, values, source_labels, target_labels
    )
    return score_table(table, source_names, target_names, allowed)


def calculate_accuracy(data_dir, results_dir, transition_graph, output_path=None):
    """Score persisted transports; this is the only label-aware workflow."""
    data_dir, results_dir = Path(data_dir), Path(results_dir)
    manifest = json.loads(
        (results_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    labels, evaluation_ids, _annotation_key = load_evaluation_data(data_dir)
    if len(labels) != int(sum(manifest["stage_sizes"])):
        raise ValueError("evaluation H5AD and model observations differ in length")
    if "spot_id_sha256" in manifest:
        import hashlib
        digest = hashlib.sha256("\n".join(evaluation_ids.tolist()).encode()).hexdigest()
        if digest != manifest["spot_id_sha256"]:
            raise ValueError("evaluation spot IDs do not match model spot IDs")
    allowed = load_allowed_transitions(transition_graph)
    pair_results = []
    slices = manifest["stage_slices"]
    if len(manifest["pairs"]) != len(slices) - 1:
        raise ValueError("transport manifest does not cover all adjacent stages")
    for index, pair in enumerate(manifest["pairs"]):
        source_slice, target_slice = slices[index], slices[index + 1]
        source_labels = labels[
            int(source_slice["start"]) : int(source_slice["stop"])
        ]
        target_labels = labels[
            int(target_slice["start"]) : int(target_slice["stop"])
        ]
        transport_path = results_dir / pair["transport"]
        with np.load(transport_path, allow_pickle=False) as transport:
            rows = transport["rows"]
            cols = transport["cols"]
            values = transport["values"]
            if int(transport["n_source"]) != len(source_labels):
                raise ValueError(f"source size mismatch in {transport_path}")
            if int(transport["n_target"]) != len(target_labels):
                raise ValueError(f"target size mismatch in {transport_path}")
            source_stage = str(transport["source_stage"])
            target_stage = str(transport["target_stage"])
        expected = (source_slice["name"], target_slice["name"])
        if (source_stage, target_stage) != expected:
            raise ValueError(f"stage mismatch in {transport_path}")
        accuracy = evaluate(
            rows, cols, values, source_labels, target_labels, allowed
        )
        pair_results.append(
            {
                "pair": [source_stage, target_stage],
                "accuracy": accuracy,
                "transport": pair["transport"],
            }
        )
    accuracies = np.asarray([item["accuracy"] for item in pair_results])
    source_sizes = np.asarray(
        [int(item["stop"]) - int(item["start"]) for item in slices[:-1]]
    )
    report = {
        "format": "sparse_st_fgw_accuracy_v1",
        "status": "completed",
        "metric": "raw_sparse_coupling_no_row_or_column_normalization",
        "transition_graph": str(Path(transition_graph).resolve()),
        "summary": {
            "mean_accuracy": float(accuracies.mean()),
            "source_weighted_accuracy": float(
                np.average(accuracies, weights=source_sizes)
            ),
        },
        "pairs": pair_results,
    }
    target = Path(output_path) if output_path else results_dir / "accuracy.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
