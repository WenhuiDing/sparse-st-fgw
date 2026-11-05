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

