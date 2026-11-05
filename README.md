# SparseST-FGW

Scalable sparse Fused Gromov-Wasserstein alignment for time-series spatial
transcriptomics.

SparseST-FGW matches spots across developmental stages using two signals:
their expression profiles and the relative organization of their spatial
neighborhoods. It is designed for large datasets where a dense all-to-all
matching matrix is too expensive to construct.

## What is implemented

- within-stage spatial kNN graphs and a two-layer GraphSAGE encoder;
- masked latent denoising and neighborhood consistency training;
- bidirectional sparse candidate generation with optional global drift;
- entropy-regularized unbalanced OT solved in the log domain on GPU;
- sparse Masked-FGW updates that combine expression and spatial structure;
- checkpointing, preprocessing, inference, tests, and resource benchmarking.

The solver stores only candidate edges and their transport values. It never
materializes a dense source-by-target coupling matrix.

## Benchmark snapshot

The following numbers come from an internal run on the 520,815-spot MOSTA
embryo atlas, covering seven adjacent-stage pairs. The comparison is against
the moscot rank-500 implementation used in the same benchmark. Accuracy is
still being evaluated, so this table reports resource usage only.

| Metric | moscot rank-500 | SparseST-FGW | Change |
|---|---:|---:|---:|
| Total time | 30,135.118 s | 1,085.332 s | 27.77x faster |
| Peak GPU memory | 10,644 MB | 1,658 MiB | 84.4% lower |
| Peak RSS | 33,049.789 MB | 8,338.660 MiB | 74.8% lower |

These figures are provided as an engineering benchmark, not as a claim that
the current implementation has the best biological matching accuracy.

## Quick start

The repository contains code and tests, but does not include the raw H5AD
dataset or trained weights. Place the licensed dataset at the path specified
by `configs/datasets/mosta.json`, then run:

```bash
python -m pip install -e .
python scripts/preprocess_data.py \
  --dataset-config configs/datasets/mosta.json
python scripts/train_model.py \
  --data-dir data \
  --checkpoint weights/spatial_graphsage.pt
python scripts/run_model.py \
  --data-dir data \
  --checkpoint weights/spatial_graphsage.pt \
  --results-dir results \
  --device cuda \
  --ot-device cuda:0
```

The full atlas workflow requires a CUDA-compatible PyTorch and JAX
installation. `environment.lock.txt` records the versions used for the
benchmark run. CPU execution is suitable for unit tests and small examples.

Run the tests with:

```bash
python -m pytest
```

## Repository layout

```text
src/sparse_st_fgw/       model, sparse OT, FGW, and data code
scripts/                 preprocessing, training, inference, and benchmarks
configs/                 dataset configuration templates
tests/                   regression tests
benchmarks/              sanitized resource comparison
```

## Scope and license

This repository is an engineering implementation intended to make the
algorithm and its resource profile inspectable and reproducible. Dataset
files and trained weights are intentionally excluded and remain subject to
their original licenses and access conditions.

Released under the MIT License.
