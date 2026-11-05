#!/usr/bin/env python
"""Calculate transition accuracy from persisted sparse transports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_st_fgw.evaluation import calculate_accuracy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score saved sparse transports without row or column normalization."
    )
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--transitions", required=True)
    parser.add_argument("--output")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    report = calculate_accuracy(
        args.data_dir,
        args.results_dir,
        args.transitions,
        output_path=args.output,
    )
    print(json.dumps(report["summary"], indent=2))
    return report


if __name__ == "__main__":
    main()
