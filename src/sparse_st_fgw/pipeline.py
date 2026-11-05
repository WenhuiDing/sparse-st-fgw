"""Label-free training and inference over the persisted PCA30 dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import BestConfig
from .data import load_model_data, sha256, stage_views, standardize
from .graphsage import (
    add_bidirectional_temporal_edges,
    build_spatial_graph,
    encode,
    load_checkpoint,
    save_checkpoint,
    train_graphsage,
)
from .sparse_ot import (
    build_bidirectional_candidates,
    uniform_marginals,
    edge_sqeuclidean,
    normalize_cost,
    solve_masked_fgw,
    solve_sparse_uot,
)


def train_spatial_model(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    *,
    config: BestConfig = BestConfig(),
    device: str = "cuda",
) -> Path:
    """Train the spatial GraphSAGE using only files in ``data_dir``."""
    pca, spatial, time_codes, _metadata, _spot_ids = load_model_data(data_dir)
    print("[train] building the within-stage spatial graph", flush=True)
    graph = build_spatial_graph(
        spatial, time_codes, neighbors=config.spatial_neighbors
    )
    model, history, loss_components = train_graphsage(
        pca,
        graph,
        norm=config.gnn_norm,
        activation=config.gnn_activation,
        dropout=config.gnn_dropout,
        hidden=config.gnn_hidden,
        epochs=config.gnn_epochs,
        batch_size=config.gnn_batch_size,
        lr=config.gnn_lr,
        mask_probability=config.gnn_mask_probability,
        consistency_weight=config.gnn_consistency_weight,
        device=device,
        seed=config.seed,
    )
    save_checkpoint(
        checkpoint_path,
        model,
        history=history,
        loss_components=loss_components,
        config=config.to_dict(),
    )
    print(f"[train] completed {config.gnn_epochs} GraphSAGE epochs", flush=True)
    return Path(checkpoint_path)


def build_zero_shot_embedding(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    config: BestConfig = BestConfig(),
    device: str = "cuda",
):
    """Create the frozen Zero-Shot representation without evaluation labels."""
    pca, spatial, time_codes, metadata, _spot_ids = load_model_data(data_dir)
    base = np.asarray(pca, dtype=np.float32)
    spatial_graph = build_spatial_graph(
        spatial, time_codes, neighbors=config.spatial_neighbors
    )
    model = load_checkpoint(checkpoint_path, device=device)
    print("[zero-shot] encoding the spatial GraphSAGE representation", flush=True)
    spatial_latent = encode(
        model,
        base,
        spatial_graph,
        batch_size=2 * config.gnn_batch_size,
        device=device,
    )
    names, stages, spaces, slices = stage_views(spatial_latent, spatial, metadata)
    masses = uniform_marginals([len(stage) for stage in stages])
    edge_batches, diagnostics = [], []
    for index, (source, target) in enumerate(zip(stages[:-1], stages[1:])):
        print(
            f"[zero-shot] pair {index + 1}/{len(stages) - 1}: "
            f"{names[index]} -> {names[index + 1]}",
            flush=True,
        )
        rows, cols = build_bidirectional_candidates(
            source,
            target,
            k=config.temporal_knn,
            cap=config.temporal_cap,
            global_drift=True,
            seed=config.seed + index,
        )
        if config.temporal_solver == "expression":
            cost = normalize_cost(
                edge_sqeuclidean(source, target, rows, cols), 0.95
            )
            values, info = solve_sparse_uot(
                source,
                target,
                rows,
                cols,
                cost,
                masses[index][0],
                masses[index][1],
                epsilon=config.temporal_epsilon,
                rho_a=config.rho_a,
                rho_b=config.rho_b,
                iterations=config.temporal_iterations,
                device=config.ot_device,
            )
        elif config.temporal_solver == "masked_fgw":
            values, info = solve_masked_fgw(
                source,
                target,
                spaces[index],
                spaces[index + 1],
                rows,
                cols,
                masses[index][0],
                masses[index][1],
                epsilon=config.temporal_epsilon,
                rho_a=config.rho_a,
                rho_b=config.rho_b,
                warmstart_iterations=config.temporal_iterations,
                outer_iterations=config.temporal_fgw_outer_iterations,
                inner_iterations=config.temporal_fgw_inner_iterations,
                weight=config.temporal_fgw_weight,
                step=config.temporal_fgw_step,
                device=config.ot_device,
            )
        else:
            raise ValueError(
                "temporal_solver must be 'expression' or 'masked_fgw'"
            )
        edge_batches.append((rows, cols, values))
        diagnostics.append(
            {
                "pair": [names[index], names[index + 1]],
                "temporal_solver": config.temporal_solver,
                **info,
            }
        )
    offsets = np.asarray([start for start, _stop in slices], dtype=np.int64)
    temporal_graph = add_bidirectional_temporal_edges(
        spatial_graph, edge_batches, offsets, topk=config.temporal_topk
    )
    zero_shot = encode(
        model,
        base,
        temporal_graph,
        batch_size=2 * config.gnn_batch_size,
        device=device,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, zero_shot)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "completed",
                "label_free": True,
                "frozen_weights": True,
                "loss": None,
                "backward": False,
                "embedding": output.name,
                "sha256": sha256(output),
                "pairs": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, diagnostics


def run_final_transport(
    data_dir: Path | str,
    latent_path: Path | str,
    results_dir: Path | str,
    *,
    config: BestConfig = BestConfig(),
) -> dict:
    """Solve all stage-pair transports and persist them without scoring."""
    _pca, spatial, _time_codes, metadata, _spot_ids = load_model_data(data_dir)
    latent = standardize(np.load(latent_path, mmap_mode="r"))
    names, stages, spaces, _slices = stage_views(latent, spatial, metadata)
    masses = uniform_marginals([len(stage) for stage in stages])
    results_dir = Path(results_dir)
    transport_dir = results_dir / "transports"
    transport_dir.mkdir(parents=True, exist_ok=True)
    pair_results = []
    started = time.perf_counter()
    for index, (source, target) in enumerate(zip(stages[:-1], stages[1:])):
        print(
            f"[final] pair {index + 1}/{len(stages) - 1}: "
            f"{names[index]} -> {names[index + 1]}",
            flush=True,
        )
        rows, cols = build_bidirectional_candidates(
            source,
            target,
            k=config.final_knn,
            cap=config.final_cap,
            global_drift=False,
            seed=config.seed + index,
        )
        values, info = solve_masked_fgw(
            source,
            target,
            spaces[index],
            spaces[index + 1],
            rows,
            cols,
            masses[index][0],
            masses[index][1],
            epsilon=config.final_epsilon,
            rho_a=config.rho_a,
            rho_b=config.rho_b,
            warmstart_iterations=config.warmstart_iterations,
            outer_iterations=config.fgw_outer_iterations,
            inner_iterations=config.fgw_inner_iterations,
            weight=config.fgw_weight,
            step=config.fgw_step,
            device=config.ot_device,
        )
        filename = f"pair_{index:02d}.npz"
        path = transport_dir / filename
        np.savez(
            path,
            rows=np.asarray(rows, dtype=np.int32),
            cols=np.asarray(cols, dtype=np.int32),
            values=np.asarray(values, dtype=np.float64),
            n_source=np.int64(len(source)),
            n_target=np.int64(len(target)),
            source_stage=np.asarray(names[index]),
            target_stage=np.asarray(names[index + 1]),
        )
        pair_results.append(
            {
                "pair": [names[index], names[index + 1]],
                "transport": f"transports/{filename}",
                "transport_sha256": sha256(path),
                **info,
            }
        )
    return {
        "status": "completed",
        "label_free": True,
        "wall_seconds": time.perf_counter() - started,
        "pairs": pair_results,
    }


def run_model(
    data_dir: Path | str,
    checkpoint_path: Path | str,
    results_dir: Path | str,
    *,
    config: BestConfig = BestConfig(),
    device: str = "cuda",
) -> dict:
    """Run label-free Zero-Shot GraphSAGE and final Sparse Masked-FGW."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    latent_path, zero_shot_diagnostics = build_zero_shot_embedding(
        data_dir,
        checkpoint_path,
        results_dir / "zero_shot_top3.npy",
        config=config,
        device=device,
    )
    final = run_final_transport(data_dir, latent_path, results_dir, config=config)
    _pca, _spatial, _times, metadata, _spot_ids = load_model_data(data_dir)
    manifest = {
        "format": "sparse_st_fgw_model_results_v1",
        "status": "completed",
        "label_free": True,
        "data_dir": str(Path(data_dir).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "input_h5ad_sha256": metadata["sha256"]["processed_h5ad"],
        "spot_id_sha256": metadata["spot_id_sha256"],
        "spatial_normalization": "center_rms_radius",
        "configuration": config.to_dict(),
        "stages": metadata["stage_names"],
        "stage_slices": metadata["stage_slices"],
        "stage_sizes": [
            int(item["stop"]) - int(item["start"])
            for item in metadata["stage_slices"]
        ],
        "zero_shot_pairs": zero_shot_diagnostics,
        "final_wall_seconds": final["wall_seconds"],
        "pairs": final["pairs"],
    }
    (results_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
