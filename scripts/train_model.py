#!/usr/bin/env python
"""Train GraphSAGE from the persisted PCA30 inputs and save its checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_st_fgw.config import BestConfig
from sparse_st_fgw.data import load_model_data, sha256
from sparse_st_fgw.pipeline import train_spatial_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and persist the spatial GraphSAGE checkpoint."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights" / "spatial_graphsage.pt"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    pca, _spatial, _time_codes, _metadata, _spot_ids = load_model_data(args.data_dir)
    if pca.shape[1] != BestConfig().pca_components:
        raise ValueError("persisted PCA dimension does not match the model")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_spatial_model(
        args.data_dir,
        checkpoint,
        config=BestConfig(),
        device=args.device,
    )
    report = {
        "status": "completed",
        "mode": "trained",
        "checkpoint": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
