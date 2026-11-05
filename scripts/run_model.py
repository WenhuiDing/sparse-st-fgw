#!/usr/bin/env python
"""Run the label-free full-atlas model and persist sparse transports."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from sparse_st_fgw.config import BestConfig
from sparse_st_fgw.pipeline import run_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GraphSAGE, temporal message passing, and Sparse Masked-FGW."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "weights" / "spatial_graphsage.pt"),
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ot-device",
        default=None,
        help="logical JAX device, e.g. cuda:0 after CUDA_VISIBLE_DEVICES",
    )
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = BestConfig()
    if args.ot_device is not None:
        config = replace(config, ot_device=args.ot_device)
    manifest = run_model(
        args.data_dir,
        args.checkpoint,
        args.results_dir,
        config=config,
        device=args.device,
    )
    summary = {
        "status": manifest["status"],
        "label_free": manifest["label_free"],
        "transport_pairs": len(manifest["pairs"]),
        "manifest": str((Path(args.results_dir) / "model_manifest.json").resolve()),
    }
    print(json.dumps(summary, indent=2))
    return manifest


if __name__ == "__main__":
    main()
