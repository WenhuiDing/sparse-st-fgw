"""Dataset-configured H5AD preprocessing and model-data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import hashlib
import json

import numpy as np
import pandas as pd


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_config(path: Path | str) -> dict:
    """Load and validate a dataset-specific JSON configuration."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("raw_h5ad", "processed_h5ad", "evaluation_h5ad"):
        value = Path(config[key]) if key in config else None
        if value is not None and not value.is_absolute():
            config[key] = str((path.parent / value).resolve())
    required = {
        "dataset_id", "raw_h5ad", "processed_h5ad", "evaluation_h5ad",
        "count_layer", "time_key", "annotation_key", "spatial_key",
        "spot_id_key", "stage_order", "hvg_count", "pca_components",
        "library_size", "preprocessing_batch_size", "hvg_cells_per_stage",
        "expression_is_raw_counts",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"dataset config missing fields: {', '.join(missing)}")
    if not config["stage_order"]:
        raise ValueError("stage_order must contain at least one stage")
    if int(config["hvg_count"]) <= 0 or int(config["pca_components"]) <= 0:
        raise ValueError("hvg_count and pca_components must be positive")
    if float(config["library_size"]) <= 0:
        raise ValueError("library_size must be positive")
    return config


def _open_anndata(path: Path | str, *, backed: str | None = "r"):
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("anndata is required for H5AD workflows") from exc
    return ad.read_h5ad(Path(path), backed=backed)


def _read_h5ad_vector(handle, group, key: str) -> tuple[np.ndarray, list[str] | None]:
    """Read one obs/var vector without materializing the expression matrix."""
    item = handle[f"{group}/{key}"]
    if hasattr(item, "keys") and "codes" in item:
        values = np.asarray(item["codes"][:])
        categories = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in item["categories"][:]
        ]
        return values, categories
    values = np.asarray(item[:])
    categories = None
    if "categories" in item.attrs:
        categories = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in handle[item.attrs["categories"]][:]
        ]
    return values, categories


def _read_raw_h5ad_metadata(path: Path | str, config: dict):
    """Read metadata and coordinates through H5AD schema, not backed X."""
    import h5py

    with h5py.File(path, "r") as handle:
        x = handle["X"]
        shape = tuple(int(v) for v in x.attrs["shape"]) if isinstance(x, h5py.Group) else x.shape
        obs = handle["obs"]
        obs_index_key = str(obs.attrs.get("_index", "_index"))
        spot_ids, _ = _read_h5ad_vector(handle, "obs", obs_index_key)
        stages, stage_categories = _read_h5ad_vector(handle, "obs", config["time_key"])
        annotations, annotation_categories = _read_h5ad_vector(handle, "obs", config["annotation_key"])
        spatial = np.asarray(handle[f"obsm/{config['spatial_key']}"][:], dtype=np.float32)
        var = handle["var"]
        var_index_key = str(var.attrs.get("_index", "_index"))
        if var_index_key in var:
            var_names, _ = _read_h5ad_vector(handle, "var", var_index_key)
        else:
            var_names = np.asarray([f"gene_{i}" for i in range(shape[1])], dtype=object)
    def decode(values, categories):
        values = np.asarray(values)
        if categories is not None:
            return np.asarray(categories, dtype=object)[values.astype(np.int64)].astype(str)
        return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])
    return (
        int(shape[0]), int(shape[1]), np.asarray(spot_ids).astype(str),
        decode(stages, stage_categories), decode(annotations, annotation_categories),
        spatial, np.asarray(var_names).astype(str),
    )


def _raw_expression_block(matrix, start: int, stop: int, n_vars: int):
    """Read rows from a dense HDF5 dataset or CSR-encoded H5AD group."""
    import h5py
    from scipy import sparse

    if isinstance(matrix, h5py.Group):
        if matrix.attrs.get("encoding-type") != "csr_matrix":
            raise TypeError("X must use dense or CSR H5AD encoding")
        pointer = np.asarray(matrix["indptr"][start : stop + 1], dtype=np.int64)
        lo, hi = int(pointer[0]), int(pointer[-1])
        return sparse.csr_matrix(
            (
                np.asarray(matrix["data"][lo:hi], dtype=np.float32),
                np.asarray(matrix["indices"][lo:hi], dtype=np.int32),
                pointer - lo,
            ),
            shape=(stop - start, int(n_vars)),
        )
    return np.asarray(matrix[start:stop], dtype=np.float32)


def _normalized_block(
    matrix, start: int, stop: int, library_size: float, n_vars: int,
    expression_is_raw_counts: bool,
):
    """Normalize one expression block while preserving CSR sparsity."""
    from scipy import sparse

    block = _raw_expression_block(matrix, start, stop, n_vars)
    if not expression_is_raw_counts:
        return block
    if sparse.issparse(block):
        block = block.tocsr().astype(np.float32, copy=True)
        library = np.maximum(np.asarray(block.sum(axis=1)).ravel(), 1.0)
        block = block.multiply((float(library_size) / library)[:, None]).tocsr()
        block.data = np.log1p(block.data).astype(np.float32)
        return block
    block = np.asarray(block, dtype=np.float32)
    library = np.maximum(block.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(float(library_size) * block / library).astype(np.float32)


def iter_count_blocks(
    matrix, n_obs: int, n_vars: int, *, batch_size: int, library_size: float,
    expression_is_raw_counts: bool, gene_indices=None,
) -> Iterator[np.ndarray]:
    """Yield normalized dense blocks, optionally restricted to selected genes."""
    for start in range(0, int(n_obs), int(batch_size)):
        stop = min(start + int(batch_size), int(n_obs))
        block = _normalized_block(
            matrix, start, stop, library_size, n_vars, expression_is_raw_counts
        )
        if gene_indices is not None:
            block = block[:, gene_indices]
        if hasattr(block, "toarray"):
            block = block.toarray()
        yield np.asarray(block, dtype=np.float32)


def select_hvg(matrix, n_obs: int, n_vars: int, *, count: int, batch_size: int,
               library_size: float, expression_is_raw_counts: bool) -> np.ndarray:
    """Select HVGs by normalized log-expression variance without an HVG file."""
    sums = np.zeros(n_vars, dtype=np.float64)
    squares = np.zeros(n_vars, dtype=np.float64)
    observed = 0
    for start in range(0, n_obs, int(batch_size)):
        stop = min(start + int(batch_size), n_obs)
        block = _normalized_block(
            matrix, start, stop, library_size, n_vars, expression_is_raw_counts
        )
        sums += np.asarray(block.sum(axis=0)).ravel()
        if hasattr(block, "multiply"):
            squares += np.asarray(block.multiply(block).sum(axis=0)).ravel()
        else:
            squares += np.square(block).sum(axis=0, dtype=np.float64)
        observed += block.shape[0]
    if observed < 2:
        raise ValueError("at least two observations are required for HVG selection")
    variance = (squares - (sums * sums) / observed) / (observed - 1)
    count = min(int(count), n_vars)
    order = np.lexsort((np.arange(n_vars), -variance))
    return np.sort(order[:count]).astype(np.int64)


def _row_blocks(matrix, row_indices: np.ndarray, n_vars: int):
    """Read arbitrary sorted rows by decomposing them into contiguous runs."""
    from scipy import sparse

    ids = np.asarray(row_indices, dtype=np.int64)
    if not len(ids):
        return sparse.csr_matrix((0, n_vars), dtype=np.float32)
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    pieces = np.split(ids, breaks)
    blocks = [
        _raw_expression_block(matrix, int(piece[0]), int(piece[-1]) + 1, n_vars)
        for piece in pieces
    ]
    if any(sparse.issparse(block) for block in blocks):
        return sparse.vstack(blocks, format="csr")
    return np.vstack(blocks)


def select_stage_sampled_hvg(
    matrix,
    stages: np.ndarray,
    stage_order: list[str],
    n_vars: int,
    *,
    count: int,
    cells_per_stage: int,
    library_size: float,
    expression_is_raw_counts: bool,
) -> np.ndarray:
    """Reproduce the historical stage-balanced normalized-variance selector."""
    sampled = []
    stages = np.asarray(stages).astype(str)
    for stage in map(str, stage_order):
        ids = np.flatnonzero(stages == stage)[: int(cells_per_stage)]
        if not len(ids):
            raise ValueError(f"stage {stage!r} has no observations")
        block = _row_blocks(matrix, ids, n_vars)
        if hasattr(block, "toarray"):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float32)
        if expression_is_raw_counts:
            library = np.maximum(block.sum(axis=1, keepdims=True), 1.0)
            block = np.log1p(float(library_size) * block / library)
        sampled.append(block.astype(np.float32, copy=False))
    values = np.vstack(sampled)
    variance = values.var(axis=0)
    count = min(int(count), n_vars)
    genes = np.argpartition(variance, -count)[-count:]
    return genes[np.argsort(variance[genes])[::-1]].astype(np.int64)


def fit_global_pca(
    matrix, n_obs: int, n_vars: int, hvg_indices: np.ndarray, *, components: int,
    batch_size: int, library_size: float, expression_is_raw_counts: bool,
) -> tuple[np.ndarray, object]:
    from sklearn.decomposition import IncrementalPCA

    components = int(components)
    if len(hvg_indices) < components:
        raise ValueError("hvg_count must be at least pca_components")
    batch_size = max(3 * components, int(batch_size))
    model = IncrementalPCA(n_components=components, batch_size=batch_size)
    for block in iter_count_blocks(
        matrix, n_obs, n_vars, batch_size=batch_size, library_size=library_size,
        expression_is_raw_counts=expression_is_raw_counts, gene_indices=hvg_indices,
    ):
        if len(block) >= components:
            model.partial_fit(block)
    if not hasattr(model, "components_"):
        raise ValueError("no usable expression batches were found")
    latent = np.empty((n_obs, components), dtype=np.float32)
    cursor = 0
    for block in iter_count_blocks(
        matrix, n_obs, n_vars, batch_size=batch_size, library_size=library_size,
        expression_is_raw_counts=expression_is_raw_counts, gene_indices=hvg_indices,
    ):
        transformed = model.transform(block).astype(np.float32)
        latent[cursor : cursor + len(transformed)] = transformed
        cursor += len(transformed)
    return latent, model


def normalize_spatial(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("spatial coordinates must have shape (n_spots, 2)")
    centered = xy - xy.mean(axis=0, dtype=np.float64)
    rms_radius = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    return (centered / max(float(rms_radius), 1e-12)).astype(np.float32)


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64)
    std = np.maximum(values.std(axis=0, dtype=np.float64), 1e-6)
    return ((values - mean) / std).astype(np.float32)


def _stage_order_and_slices(stage_values: np.ndarray, stage_order: list[str]):
    stage_values = np.asarray(stage_values).astype(str)
    configured = [str(x) for x in stage_order]
    observed = set(stage_values.tolist())
    missing = [name for name in configured if name not in observed]
    extra = sorted(observed - set(configured))
    if missing or extra:
        raise ValueError(f"stage mismatch; missing={missing}, unexpected={extra}")
    parts, slices, cursor = [], [], 0
    for name in configured:
        ids = np.flatnonzero(stage_values == name)
        parts.append(ids)
        stop = cursor + len(ids)
        slices.append({"name": name, "start": cursor, "stop": stop})
        cursor = stop
    return np.concatenate(parts).astype(np.int64), slices


def _category_series(values, categories: list[str]) -> pd.Series:
    return pd.Series(pd.Categorical(values, categories=categories, ordered=True))


def preprocess_dataset(config: dict, *, batch_size: int = 4096) -> dict:
    """Create model and evaluation H5AD files from a dataset config."""
    raw = Path(config["raw_h5ad"])
    processed = Path(config["processed_h5ad"])
    evaluation = Path(config["evaluation_h5ad"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    n_obs, n_vars, spot_ids, stages, annotations, spatial, var_names = _read_raw_h5ad_metadata(raw, config)
    if spatial.ndim != 2 or spatial.shape[1] != 2:
        raise ValueError("configured spatial coordinates must be 2D")
    if len(np.unique(spot_ids)) != len(spot_ids):
        raise ValueError("spot IDs must be unique")
    import h5py
    with h5py.File(raw, "r") as raw_file:
        source = str(config["count_layer"])
        matrix_path = "X" if source.upper() == "X" else f"layers/{source}"
        if matrix_path not in raw_file:
            raise KeyError(f"configured expression source {matrix_path!r} is missing")
        matrix = raw_file[matrix_path]
        hvg = select_stage_sampled_hvg(
            matrix, stages, config["stage_order"], n_vars,
            count=int(config["hvg_count"]),
            cells_per_stage=int(config["hvg_cells_per_stage"]),
            library_size=float(config["library_size"]),
            expression_is_raw_counts=bool(config["expression_is_raw_counts"]),
        )
        pca, model = fit_global_pca(
            matrix, n_obs, n_vars, hvg, components=int(config["pca_components"]),
            batch_size=batch_size, library_size=float(config["library_size"]),
            expression_is_raw_counts=bool(config["expression_is_raw_counts"]),
        )
    hvg_names = var_names[hvg]

    row_order, slices = _stage_order_and_slices(stages, config["stage_order"])
    ordered_ids, ordered_stages = spot_ids[row_order], stages[row_order]
    ordered_annotations, ordered_pca = annotations[row_order], pca[row_order]
    ordered_spatial = spatial[row_order]
    import anndata as ad
    obs = pd.DataFrame(index=pd.Index(ordered_ids, name=config["spot_id_key"]))
    obs[config["time_key"]] = _category_series(ordered_stages, [str(x) for x in config["stage_order"]]).to_numpy()
    out = ad.AnnData(X=ordered_pca.astype(np.float32), obs=obs)
    out.var_names = [f"PC{i + 1}" for i in range(ordered_pca.shape[1])]
    out.obsm[config["spatial_key"]] = ordered_spatial.astype(np.float32)
    out.uns["sparse_st_fgw"] = {
        "dataset_id": config["dataset_id"], "hvg_count": int(len(hvg)),
        "hvg_indices": hvg.tolist(), "pca_components": int(config["pca_components"]),
        "hvg_names": hvg_names.tolist(),
        "pca_model": {
            "components": np.asarray(model.components_, dtype=np.float32),
            "mean": np.asarray(model.mean_, dtype=np.float32),
            "variance": np.asarray(model.var_, dtype=np.float32),
            "explained_variance": np.asarray(model.explained_variance_, dtype=np.float32),
        },
        "library_size": float(config["library_size"]), "stage_order": [str(x) for x in config["stage_order"]],
        "expression_source": matrix_path,
        "expression_is_raw_counts": bool(config["expression_is_raw_counts"]),
        "hvg_selection": {
            "method": "stage_sampled_normalized_variance",
            "cells_per_stage": int(config["hvg_cells_per_stage"]),
        },
        "spot_id_key": config["spot_id_key"], "source_raw_h5ad": str(raw.resolve()),
    }
    out.write_h5ad(processed)
    eval_obs = pd.DataFrame(index=pd.Index(ordered_ids, name=config["spot_id_key"]))
    eval_obs[config["annotation_key"]] = _category_series(ordered_annotations, sorted(set(ordered_annotations.tolist()))).to_numpy()
    eval_out = ad.AnnData(obs=eval_obs)
    eval_out.uns["sparse_st_fgw"] = {"dataset_id": config["dataset_id"], "annotation_key": config["annotation_key"], "spot_id_key": config["spot_id_key"]}
    eval_out.write_h5ad(evaluation)
    metadata = {
        "format": "sparse_st_fgw_h5ad_model_data_v2", "dataset_id": config["dataset_id"],
        "n_spots": int(len(ordered_pca)), "pca_components": int(ordered_pca.shape[1]),
        "stage_names": [item["name"] for item in slices], "stage_slices": slices,
        "processed_h5ad": str(processed.resolve()), "evaluation_h5ad": str(evaluation.resolve()),
        "spot_id_key": config["spot_id_key"],
        "spot_id_sha256": hashlib.sha256("\n".join(ordered_ids.tolist()).encode()).hexdigest(),
        "sha256": {"processed_h5ad": sha256(processed), "evaluation_h5ad": sha256(evaluation)},
        "config": config,
    }
    (processed.parent / "dataset_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_model_data(data_dir: Path | str):
    """Load the canonical processed H5AD model input."""
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    adata = _open_anndata(metadata["processed_h5ad"], backed="r")
    try:
        pca = np.asarray(adata.X, dtype=np.float32)
        spatial = np.asarray(adata.obsm[metadata["config"]["spatial_key"]], dtype=np.float32)
        stage = np.asarray(adata.obs[metadata["config"]["time_key"]].cat.codes, dtype=np.int16)
        spot_ids = np.asarray(adata.obs_names.astype(str))
    finally:
        adata.file.close()
    expected = int(metadata["n_spots"])
    if not (len(pca) == len(spatial) == len(stage) == len(spot_ids) == expected):
        raise ValueError("processed H5AD arrays do not match dataset_manifest.json")
    if spatial.shape[1] != 2:
        raise ValueError("processed spatial coordinates must be two-dimensional")
    return pca, spatial, stage, metadata, spot_ids


def load_evaluation_data(data_dir: Path | str):
    """Load labels only for the independent accuracy workflow."""
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    adata = _open_anndata(metadata["evaluation_h5ad"], backed="r")
    try:
        labels = np.asarray(adata.obs[config["annotation_key"]]).astype(str)
        spot_ids = np.asarray(adata.obs_names.astype(str))
    finally:
        adata.file.close()
    return labels, spot_ids, config["annotation_key"]


def stage_views(latent: np.ndarray, spatial: np.ndarray, metadata: dict):
    stages, spaces, slices = [], [], []
    for item in metadata["stage_slices"]:
        start, stop = int(item["start"]), int(item["stop"])
        stages.append(np.asarray(latent[start:stop], dtype=np.float32))
        spaces.append(normalize_spatial(spatial[start:stop]))
        slices.append((start, stop))
    return metadata["stage_names"], stages, spaces, slices
