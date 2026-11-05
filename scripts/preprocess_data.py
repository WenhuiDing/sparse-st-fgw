#!/usr/bin/env python
"""Create canonical model/evaluation H5AD files from a dataset config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_st_fgw.data import load_dataset_config, preprocess_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "datasets" / "mosta.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a configured H5AD dataset.")
    parser.add_argument("--dataset-config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--batch-size",
        type=int,
        help="optional override of preprocessing_batch_size in the dataset config",
    )
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    config = load_dataset_config(args.dataset_config)
    batch_size = args.batch_size or int(config["preprocessing_batch_size"])
    metadata = preprocess_dataset(config, batch_size=batch_size)
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    main()
